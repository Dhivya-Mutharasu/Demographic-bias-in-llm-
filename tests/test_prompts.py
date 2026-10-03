"""Run: python tests/test_prompts.py  (needs /tmp/run2/reviews.csv from test_sampler.py's run,
or point --reviews at any reviews.csv)"""
import csv, json, subprocess, sys, os, hashlib, tempfile
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
REVIEWS = sys.argv[1] if len(sys.argv) > 1 else "/tmp/run2/reviews.csv"
ok = True
def check(name, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(("PASS" if cond else "FAIL"), "-", name, detail)

if not os.path.exists(REVIEWS):
    subprocess.run([sys.executable, f"{ROOT}/tests/make_mock_data.py"], cwd=f"{ROOT}/tests", check=True)
    tmp = tempfile.mkdtemp()
    subprocess.run([sys.executable, f"{ROOT}/sample_reviews.py",
                    "--source", f"yelp={HERE}/mock_yelp.json",
                    "--source", f"amazon={HERE}/mock_amazon.csv",
                    "--out-dir", tmp, "--seed", "42"], check=True, capture_output=True)
    REVIEWS = f"{tmp}/reviews.csv"

def run_build(out, seed, extra=()):
    cmd = [sys.executable, f"{ROOT}/build_prompts.py", "--reviews", REVIEWS, "--out-dir", out,
           "--seed", str(seed), *extra]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout, r.stderr)
    return r

d1, d2, d3 = (tempfile.mkdtemp() for _ in range(3))
run_build(d1, 42, ["--with-location-arm"])
run_build(d2, 42, ["--with-location-arm"])
run_build(d3, 43, ["--with-location-arm"])

h = lambda p: hashlib.md5(open(p, "rb").read()).hexdigest()
check("same seed -> identical prompts.csv", h(f"{d1}/prompts.csv") == h(f"{d2}/prompts.csv"))
check("different seed -> different prompts.csv", h(f"{d1}/prompts.csv") != h(f"{d3}/prompts.csv"))

reviews = list(csv.DictReader(open(REVIEWS, encoding="utf-8")))
n_rev = len(reviews)
prompts = list(csv.DictReader(open(f"{d1}/prompts.csv", encoding="utf-8")))
check(f"5 conditions x {n_rev} reviews = {5*n_rev} prompts", len(prompts) == 5 * n_rev,
      f"(got {len(prompts)})")
check("every prompt_id is unique", len({p["prompt_id"] for p in prompts}) == len(prompts))

by_cond = defaultdict(list)
for p in prompts:
    by_cond[p["condition"]].append(p)
check("all 5 conditions present, each = n_reviews", set(by_cond) == {"none", "western_male",
      "western_female", "south_asian_male", "south_asian_female"} and
      all(len(v) == n_rev for v in by_cond.values()))

# --- name balance ---
for cond in ["western_male", "western_female", "south_asian_male", "south_asian_female"]:
    counts = Counter(p["name"] for p in by_cond[cond])
    check(f"{cond}: 12 names, balanced (max-min <= 1)",
          len(counts) == 12 and max(counts.values()) - min(counts.values()) <= 1,
          f"({dict(counts)})")

# --- sub_region metadata ---
sys.path.insert(0, ROOT)
import build_prompts as bp
for cond in ["south_asian_male", "south_asian_female"]:
    sub_regions = {p["sub_region"] for p in by_cond[cond]}
    check(f"{cond}: spans multiple South Asian sub-regions (not just India)",
          sub_regions == {"India", "Pakistan", "Bangladesh", "Sri Lanka", "Nepal"}, f"({sub_regions})")
    check(f"{cond}: every prompt's sub_region matches names.csv",
          all(p["sub_region"] == bp.DEFAULT_NAMES[cond][
              [n for n, _ in bp.DEFAULT_NAMES[cond]].index(p["name"])][1] for p in by_cond[cond]))
check("'none' condition has sub_region == 'none'",
      all(p["sub_region"] == "none" for p in by_cond["none"]))

# --- text identity: the review text inside the prompt must be identical across conditions ---
text_by_review_cond = defaultdict(dict)
for p in prompts:
    # text is everything between the quotes after 'Review: "'
    body = p["prompt_text"].split('Review: "', 1)[1].rsplit('"', 1)[0]
    text_by_review_cond[p["review_id"]][p["condition"]] = body
bad = [rid for rid, d in text_by_review_cond.items() if len(set(d.values())) != 1]
check("review text identical across all 5 conditions", len(bad) == 0, f"bad={bad[:5]}")

# --- preamble differs appropriately ---
sample_rid = prompts[0]["review_id"]
check("'none' condition has no 'Reviewer:' line",
      "Reviewer:" not in [p for p in prompts if p["review_id"] == sample_rid and p["condition"] == "none"][0]["prompt_text"])
check("named conditions have a 'Reviewer:' line with the assigned name",
      all(("Reviewer: " + p["name"]) in p["prompt_text"] for p in prompts if p["condition"] != "none"))

# --- no cross-condition pairing: a review's name in one condition should not predict its
#     position/name-index in another condition (weak randomness check via Cramer's V-ish count) ---
idx = {}
for cond in ["western_male", "western_female"]:
    pool_order = sorted({p["name"] for p in by_cond[cond]})
    idx[cond] = {p["review_id"]: pool_order.index(p["name"]) for p in by_cond[cond]}
matches = sum(idx["western_male"][r["review_id"]] == idx["western_female"][r["review_id"]] for r in reviews)
expected = n_rev / 10
check("name-index NOT locked across conditions (independent RNG streams)",
      abs(matches - expected) < expected,  # loose bound: not suspiciously perfect/never matching
      f"(matches={matches}, expected~{expected:.0f} if independent)")

# --- ground truth carried through correctly ---
rev_label = {r["review_id"]: r["sentiment_label"] for r in reviews}
check("sentiment_label matches the source review for every prompt",
      all(p["sentiment_label"] == rev_label[p["review_id"]] for p in prompts))

# --- manifest ---
man = json.load(open(f"{d1}/prompt_manifest.json"))
check("manifest has seed, hash, per-condition usage counts, total_prompts",
      man["seed"] == 42 and len(man["script_sha256"]) == 64 and
      man["total_prompts"] == len(prompts) and
      set(man["name_usage_counts"]) == {"western_male", "western_female",
                                         "south_asian_male", "south_asian_female"})

# --- names.csv override path ---
custom = tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w", newline="")
w = csv.writer(custom)
w.writerow(["condition", "name"])
for c, names in [("western_male", ["Alpha One", "Beta Two", "Gamma Three"]),
                 ("western_female", ["Delta Four", "Epsilon Five", "Zeta Six"]),
                 ("south_asian_male", ["Eta Seven", "Theta Eight", "Iota Nine"]),
                 ("south_asian_female", ["Kappa Ten", "Lambda Eleven", "Mu Twelve"])]:
    for n in names:
        w.writerow([c, n])
custom.close()
d4 = tempfile.mkdtemp()
r = run_build(d4, 42, ["--names-file", custom.name])
check("custom --names-file is respected", r.returncode == 0)
p4 = list(csv.DictReader(open(f"{d4}/prompts.csv", encoding="utf-8")))
check("custom names appear, default names do not",
      any(p["name"] == "Alpha One" for p in p4) and not any(p["name"] == "James Mitchell" for p in p4))

# --- location arm ---
loc = list(csv.DictReader(open(f"{d1}/location_prompts.csv", encoding="utf-8")))
check("location arm: 2 regions x n_reviews prompts", len(loc) == 2 * n_rev, f"(got {len(loc)})")

all_names = [n for pool in bp.DEFAULT_NAMES.values() for n, _region in pool]
check("location arm never mentions a person name",
      not any(nm in p["prompt_text"] for p in loc for nm in all_names))
check("location arm has no 'Reviewer:' line, only 'Reviewer location:'",
      all("Reviewer location:" in p["prompt_text"] and "Reviewer: " not in p["prompt_text"] for p in loc))
loc_by_region = defaultdict(list)
for p in loc:
    loc_by_region[p["region"]].append(p)
for region in ["western", "south_asian"]:
    counts = Counter(p["location"] for p in loc_by_region[region])
    check(f"location '{region}': 4 places, balanced (max-min <= 1)",
          len(counts) == 4 and max(counts.values()) - min(counts.values()) <= 1, f"({dict(counts)})")

print("\nALL TESTS PASSED" if ok else "\nSOME TESTS FAILED")
sys.exit(0 if ok else 1)
