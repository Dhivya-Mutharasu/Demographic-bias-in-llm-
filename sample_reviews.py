#!/usr/bin/env python3
"""
sample_reviews.py  --  Step 1 of the demographic-bias project.

Draws a stratified, reproducible sample of REAL reviews (Yelp, Amazon, or any
JSONL / CSV / Parquet file) and removes reviews that would interfere with a
name-swap experiment: personal names, gender / ethnicity / religion / location
cues, URLs, very short or very long text, non-English text and duplicates.

Outputs (in --out-dir):
  reviews.csv             the stimulus set (one row per base review)
  sampling_manifest.json  parameters, counts, rejection reasons, seed, script hash
                          (use it to write the "Dataset" part of your Methods)
  annotation_sheet.csv    BLIND sheet for two human annotators (stars hidden)

Example
  python sample_reviews.py \
      --source yelp=/path/yelp_academic_dataset_review.json \
      --source amazon=/path/amazon_electronics.jsonl \
      --out-dir data --seed 42
"""
import argparse
import csv
import hashlib
import json
import random
import re
import string
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

# --------------------------------------------------------------------------
# Defaults
# --------------------------------------------------------------------------
# Per source. Stars 1-2 = negative, 3 = neutral/mixed, 4-5 = positive.
# 3-star reviews are over-sampled on purpose: ambiguous text is where an
# identity cue is most likely to tip the label.
DEFAULT_PER_STAR = {1: 45, 2: 45, 3: 80, 4: 40, 5: 40}  # = 250 per source

TEXT_CANDIDATES = ["text", "review_text", "reviewText", "review", "content", "body"]
RATING_CANDIDATES = ["stars", "rating", "overall", "score", "star_rating"]
ID_CANDIDATES = ["review_id", "id", "reviewID"]

# --------------------------------------------------------------------------
# Cue lists. Deliberately over-inclusive: we have millions of reviews, so it is
# fine to throw away some good ones. Extend with --blocklist-file if needed.
# --------------------------------------------------------------------------
ETHNIC_NATIONAL_RELIGION = """
indian indians pakistani bangladeshi sri-lankan nepali nepalese bhutanese afghan
british english irish scottish welsh american americans canadian australian european
chinese japanese korean thai vietnamese filipino asian asians mexican cuban
puerto-rican hispanic latino latina latinx italian french german greek spanish
russian polish african african-american caucasian arab arabic middle-eastern
persian turkish israeli jewish muslim muslims islamic hindu hindus christian
christians catholic sikh buddhist halal kosher desi brahmin dalit caste
immigrant immigrants foreigner foreigners accent minority minorities racist
racism sexist sexism gay lesbian transgender
""".split()

LOCATIONS = """
india pakistan bangladesh china japan korea mexico canada england scotland
uk united-kingdom united-states usa america london paris delhi mumbai bangalore
bengaluru chennai hyderabad kolkata dubai singapore toronto
alabama alaska arizona arkansas california colorado connecticut delaware florida
georgia hawaii idaho illinois indiana iowa kansas kentucky louisiana maine
maryland massachusetts michigan minnesota mississippi missouri montana nebraska
nevada new-hampshire new-jersey new-mexico new-york north-carolina north-dakota
ohio oklahoma oregon pennsylvania rhode-island south-carolina south-dakota
tennessee texas utah vermont virginia washington west-virginia wisconsin wyoming
philadelphia philly tampa indianapolis nashville new-orleans nola tucson reno
santa-barbara saint-louis st.-louis boise edmonton sparks las-vegas phoenix
""".split()

GENDER_PATTERNS = [
    r"\b(?:my|our)\s+(?:husband|wife|boyfriend|girlfriend|hubby|hubs|wifey|fianc[eé]e?|"
    r"mom|mother|mum|dad|father|son|daughter|brother|sister|grandma|grandmother|"
    r"grandpa|grandfather)\b",
    r"\bas\s+an?\s+(?:man|woman|guy|girl|lady|gentleman|mother|father|mom|dad|"
    r"husband|wife|female|male)\b",
    r"\bi\s*(?:am|'m|’m)\s+an?\s+(?:man|woman|guy|girl|lady|mother|father|mom|dad|"
    r"female|male)\b",
    r"\b(?:ma'am|ma’am|sir|mrs|mr|ms)\b",
]

# Words that gender-guesser treats as first names but that are ordinary English.
# Allowed anywhere (function words):
ALLOWED_ANYWHERE = {
    "Am", "Are", "Be", "Can", "Do", "Done", "Even", "Go", "He", "Her", "Here", "His",
    "Me", "Mine", "My", "So", "The", "You", "Ok", "Okay", "Beer", "Tea", "Night",
}
# Allowed only as the first word of a sentence ("Love this place", "Just ordered"...).
# Mid-sentence they still count as possible names ("our server Hope").
ALLOWED_AT_SENTENCE_START = {
    "Love", "Just", "Price", "Fine", "Long", "Tiny", "Will", "May", "Hope",
    "Ah", "Con", "Deep", "Edit", "Five", "Free", "Hey", "Hi", "Ideal", "Line",
    "Oh", "One", "Soon", "Thin",
    "Way", "Lot", "Made", "Take", "Said", "Than", "Tell", "Job", "Side", "Run", "Per",
    "Young", "Ago", "Win", "Else", "Due", "Age", "Red", "Chance", "Won", "Save", "Key",
    "Door", "Floor", "Carry", "Sad", "Brand", "Bet", "Till", "Lucky", "Core", "Sky",
    "Rain", "Solo", "Hang", "Kinda", "Mile", "Diet", "Lane", "Mate", "Hat", "Fee", "Tie",
    "Tone", "Rush", "Odd", "Sake", "Pace", "Vital", "Ate", "Bang", "Dare", "Jet", "Mall",
    "Tale", "Skip", "Flip", "Loyal", "Harsh", "Lean", "Haven", "Miracle", "Fair", "Kick",
    "Beat", "Rock", "Ice",
}

STOPWORDS = set(
    "the a an and or but if of to in on at for with is are was were be been it its this "
    "that these those i my me we our you your they them their he she his her not no so "
    "as by from have has had do did does very".split()
)

URL_CONTACT_RE = re.compile(
    r"(https?://|www\.|\b[\w.+-]+@[\w-]+\.[\w.]+|\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b)", re.I
)


def _term_regex(terms):
    parts = [re.escape(t).replace(r"\-", r"[\s-]") for t in terms]
    return re.compile(r"\b(?:" + "|".join(sorted(parts, key=len, reverse=True)) + r")\b", re.I)


# --------------------------------------------------------------------------
# Readers
# --------------------------------------------------------------------------
def iter_records(path):
    """Yield dict records from .json/.jsonl (JSON lines), .csv or .parquet."""
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix in (".json", ".jsonl", ".ndjson"):
        with open(p, "r", encoding="utf-8") as f:
            first = f.read(1)
            f.seek(0)
            if first == "[":  # a single JSON array (small files only)
                for rec in json.load(f):
                    yield rec
            else:  # JSON lines (Yelp's format)
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        continue
    elif suffix == ".csv":
        csv.field_size_limit(sys.maxsize)
        with open(p, "r", encoding="utf-8", newline="") as f:
            yield from csv.DictReader(f)
    elif suffix == ".parquet":
        import pyarrow.parquet as pq

        pf = pq.ParquetFile(str(p))
        for batch in pf.iter_batches(batch_size=10000):
            cols = batch.to_pydict()
            keys = list(cols)
            for i in range(batch.num_rows):
                yield {k: cols[k][i] for k in keys}
    else:
        raise ValueError(f"Unsupported file type: {p.suffix} ({p})")


def detect_column(record, candidates, override, what):
    if override:
        if override not in record:
            raise SystemExit(f"--{what}-col '{override}' not found. Columns: {list(record)}")
        return override
    for c in candidates:
        if c in record:
            return c
    raise SystemExit(f"Could not auto-detect the {what} column. Columns: {list(record)}. "
                     f"Pass --{what}-col.")


# --------------------------------------------------------------------------
# Filters
# --------------------------------------------------------------------------
class NameDetector:
    """Flags capitalised tokens that look like first names (gender-guesser DB)."""

    def __init__(self, extra_allowed=()):
        self.extra_allowed = set(extra_allowed)
        try:
            import gender_guesser.detector as gg

            self._det = gg.Detector(case_sensitive=False)
            self.available = True
        except Exception:
            self._det = None
            self.available = False

    _ROLE_FALLBACK = re.compile(
        r"\b(?:server|waiter|waitress|manager|owner|host|hostess|bartender|cashier|"
        r"agent|rep|representative|seller|technician|driver|stylist|dentist|doctor|dr\.?)"
        r"\s+(?:named\s+)?[A-Z][a-z]{2,}"
    )

    def find(self, text):
        if not self.available:  # weak fallback; install gender-guesser instead
            m = self._ROLE_FALLBACK.search(text)
            return m.group(0) if m else None
        for sentence in re.split(r"(?<=[.!?])\s+", text):
            for i, raw in enumerate(sentence.split()):
                tok = re.sub(r"['’]s$", "", raw).strip(string.punctuation + "“”‘’")
                if len(tok) < 3 or not (tok[0].isupper() and tok[1:].islower()):
                    continue
                if tok in ALLOWED_ANYWHERE or tok in self.extra_allowed or \
                        (i == 0 and tok in ALLOWED_AT_SENTENCE_START):
                    continue
                if self._det.get_gender(tok) != "unknown":
                    return tok
        return None


class Filters:
    def __init__(self, min_words, max_words, extra_terms=(), allow_words=()):
        self.min_words = min_words
        self.max_words = max_words
        self.names = NameDetector(allow_words)
        self.flagged_name_tokens = Counter()
        self.cue_re = _term_regex(ETHNIC_NATIONAL_RELIGION + LOCATIONS + list(extra_terms))
        self.gender_re = re.compile("|".join(GENDER_PATTERNS), re.I)

    def check(self, text):
        """Return a rejection reason, or None if the review is usable."""
        n = len(text.split())
        if n < self.min_words or n > self.max_words:
            return "length"
        if URL_CONTACT_RE.search(text):
            return "url_or_contact"
        ascii_ratio = sum(ch.isascii() for ch in text) / max(len(text), 1)
        toks = re.findall(r"[a-z']+", text.lower())
        stop_ratio = sum(t in STOPWORDS for t in toks) / max(len(toks), 1)
        if ascii_ratio < 0.98 or stop_ratio < 0.20:
            return "non_english"
        flagged = self.names.find(text)
        if flagged:
            self.flagged_name_tokens[flagged] += 1
            return "person_name"
        if self.gender_re.search(text):
            return "gender_cue"
        if self.cue_re.search(text):
            return "ethnicity_religion_location_cue"
        return None


def stars_to_label(stars):
    return "negative" if stars <= 2 else ("neutral" if stars == 3 else "positive")


def parse_per_star(spec):
    out = {}
    for part in spec.split(","):
        k, v = part.split(":")
        out[int(k)] = int(v)
    return out


# --------------------------------------------------------------------------
# Sampling (streaming reservoir sampling per source x star stratum)
# --------------------------------------------------------------------------
def sample_source(name, path, args, filters, rng, seen_hashes):
    quotas = args.per_star
    reservoirs = {s: [] for s in quotas}
    seen_per_stratum = Counter()
    rejected = Counter()
    scanned = eligible = 0
    cols = None
    t0 = time.time()

    for rec in iter_records(path):
        if cols is None:
            cols = (
                detect_column(rec, TEXT_CANDIDATES, args.text_col, "text"),
                detect_column(rec, RATING_CANDIDATES, args.rating_col, "rating"),
                next((c for c in ID_CANDIDATES if c in rec), None) if not args.id_col else args.id_col,
            )
            print(f"[{name}] columns -> text='{cols[0]}' rating='{cols[1]}' id='{cols[2]}'")
        scanned += 1
        if args.max_scan and scanned > args.max_scan:
            break
        if scanned % 500000 == 0:
            print(f"[{name}] scanned {scanned:,} rows, eligible {eligible:,} "
                  f"({time.time() - t0:.0f}s)")

        try:
            stars = int(round(float(rec[cols[1]])))
        except (TypeError, ValueError, KeyError):
            rejected["bad_rating"] += 1
            continue
        if stars not in quotas:
            rejected["rating_not_in_plan"] += 1
            continue
        text = " ".join(str(rec.get(cols[0]) or "").split())
        reason = filters.check(text)
        if reason:
            rejected[reason] += 1
            continue
        h = hashlib.md5(re.sub(r"[^a-z0-9]", "", text.lower()).encode()).hexdigest()
        if h in seen_hashes:
            rejected["duplicate"] += 1
            continue
        seen_hashes.add(h)
        eligible += 1

        item = {
            "source": name,
            "source_review_id": (rec.get(cols[2]) if cols[2] else "") or f"row{scanned}",
            "stars": stars,
            "sentiment_label": stars_to_label(stars),
            "n_words": len(text.split()),
            "text": text,
        }
        seen_per_stratum[stars] += 1
        res = reservoirs[stars]
        if len(res) < quotas[stars]:
            res.append(item)
        else:
            j = rng.randrange(seen_per_stratum[stars])
            if j < quotas[stars]:
                res[j] = item

    selected = [it for s in sorted(reservoirs) for it in reservoirs[s]]
    shortfall = {s: quotas[s] - len(reservoirs[s]) for s in quotas if len(reservoirs[s]) < quotas[s]}
    report = {
        "path": str(path),
        "columns_used": {"text": cols[0], "rating": cols[1], "id": cols[2]} if cols else None,
        "rows_scanned": scanned - (1 if args.max_scan and scanned > args.max_scan else 0),
        "eligible_after_filters": eligible,
        "rejected": dict(rejected),
        "eligible_per_star": dict(seen_per_stratum),
        "selected_per_star": {s: len(reservoirs[s]) for s in sorted(reservoirs)},
        "shortfall": shortfall,
    }
    if shortfall:
        print(f"[{name}] WARNING: not enough eligible reviews for stars {shortfall}")
    return selected, report


def write_csv(path, rows, fieldnames):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", action="append", required=True, metavar="NAME=PATH",
                    help="repeatable, e.g. --source yelp=reviews.json --source amazon=elec.jsonl")
    ap.add_argument("--out-dir", default="data")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--per-star", type=parse_per_star, default=DEFAULT_PER_STAR,
                    help='per source, e.g. "1:45,2:45,3:80,4:40,5:40"')
    ap.add_argument("--min-words", type=int, default=20)
    ap.add_argument("--max-words", type=int, default=120)
    ap.add_argument("--annotate", type=int, default=150,
                    help="size of the blind two-annotator sheet (0 to skip)")
    ap.add_argument("--max-scan", type=int, default=0, help="rows to scan per source (testing)")
    ap.add_argument("--text-col")
    ap.add_argument("--rating-col")
    ap.add_argument("--id-col")
    ap.add_argument("--blocklist-file", help="extra cue terms to reject, one per line")
    ap.add_argument("--allow-words", default="",
                    help="comma-separated capitalised words that are NOT names (see the "
                         "'most flagged tokens' printout), e.g. --allow-words Way,Kind")
    ap.add_argument("--require-name-filter", action="store_true",
                    help="fail if gender-guesser is not installed")
    args = ap.parse_args()

    extra = []
    if args.blocklist_file:
        extra = [ln.strip() for ln in open(args.blocklist_file, encoding="utf-8") if ln.strip()]
    allow = [w.strip() for w in args.allow_words.split(",") if w.strip()]
    filters = Filters(args.min_words, args.max_words, extra, allow)
    if not filters.names.available:
        msg = "gender-guesser is not installed: name filtering is WEAK. Run: pip install gender-guesser"
        if args.require_name_filter:
            raise SystemExit("ERROR: " + msg)
        print("WARNING: " + msg)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    all_items, reports = [], {}
    seen_hashes = set()  # shared: no text may appear twice across ALL sources
    for spec in args.source:
        if "=" not in spec:
            raise SystemExit(f"--source must look like NAME=PATH, got: {spec}")
        name, path = spec.split("=", 1)
        items, rep = sample_source(name, path, args, filters, rng, seen_hashes)
        all_items += items
        reports[name] = rep

    # Shuffle so review ids do not encode source or star rating, then number them.
    rng.shuffle(all_items)
    for i, it in enumerate(all_items, 1):
        it["review_id"] = f"R{i:04d}"
    fields = ["review_id", "source", "source_review_id", "stars", "sentiment_label", "n_words", "text"]
    write_csv(out_dir / "reviews.csv", all_items, fields)

    if args.annotate:
        rng2 = random.Random(args.seed + 1)
        k = min(args.annotate, len(all_items))
        strata = defaultdict(list)
        for it in all_items:
            strata[(it["source"], it["stars"])].append(it)
        chosen = []
        for key in sorted(strata):  # proportional allocation, at least 1 per stratum
            share = max(1, round(k * len(strata[key]) / len(all_items)))
            chosen += rng2.sample(strata[key], min(share, len(strata[key])))
        rng2.shuffle(chosen)
        sheet = [{"review_id": c["review_id"], "text": c["text"],
                  "annotator1_label": "", "annotator2_label": "", "notes": ""} for c in chosen]
        write_csv(out_dir / "annotation_sheet.csv", sheet,
                  ["review_id", "text", "annotator1_label", "annotator2_label", "notes"])

    manifest = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seed": args.seed,
        "python": sys.version.split()[0],
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "name_filter": "gender-guesser" if filters.names.available else "weak-fallback",
        "filters": {"min_words": args.min_words, "max_words": args.max_words,
                    "per_star_target_per_source": args.per_star},
        "allow_words_added": allow,
        "most_flagged_name_tokens": filters.flagged_name_tokens.most_common(50),
        "sources": reports,
        "total_reviews": len(all_items),
        "total_by_label": dict(Counter(it["sentiment_label"] for it in all_items)),
        "annotation_sheet_size": len(sheet) if args.annotate else 0,
    }
    (out_dir / "sampling_manifest.json").write_text(json.dumps(manifest, indent=2))

    print("\n=== SUMMARY ===")
    print(f"Total reviews: {len(all_items)}  by label: {manifest['total_by_label']}")
    for name, rep in reports.items():
        print(f"[{name}] scanned={rep['rows_scanned']:,} eligible={rep['eligible_after_filters']:,} "
              f"selected={rep['selected_per_star']} rejected={rep['rejected']}")
    print("\nMost-flagged 'name' tokens (real names, or ordinary words to whitelist with --allow-words):")
    print("  " + ", ".join(f"{t}({c})" for t, c in filters.flagged_name_tokens.most_common(20)))
    print(f"\nWrote: {out_dir/'reviews.csv'}, {out_dir/'sampling_manifest.json'}"
          + (f", {out_dir/'annotation_sheet.csv'}" if args.annotate else ""))
    print("\nEyeball 10 random reviews before continuing:")
    for it in random.Random(args.seed + 2).sample(all_items, min(10, len(all_items))):
        print(f"  {it['review_id']} [{it['source']} {it['stars']}*] {it['text'][:140]}")


if __name__ == "__main__":
    main()
