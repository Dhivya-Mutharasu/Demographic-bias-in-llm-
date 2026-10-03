"""Run:  python tests/test_sampler.py   (needs mock data from make_mock_data.py)"""
import csv, json, subprocess, sys, hashlib, tempfile, os
from collections import Counter, defaultdict
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sample_reviews as sr

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ok = True
def check(name, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(("PASS" if cond else "FAIL"), "-", name, detail)

# 1) filter accuracy against known row kinds
F = sr.Filters(20, 120)
tab = defaultdict(Counter)
for line in open(f"{HERE}/mock_yelp.json"):
    r = json.loads(line)
    tab[r["_kind"]][F.check(" ".join(r["text"].split())) or "PASS"] += 1
clean_total = sum(tab["clean"].values())
check("clean reviews wrongly rejected < 2%", tab["clean"]["person_name"] / clean_total < 0.02,
      f"({tab['clean']['person_name']}/{clean_total})")
for kind in ["name", "name2", "gender", "ethnic", "loc", "url", "short", "long", "nonen"]:
    passed = tab[kind]["PASS"]
    check(f"dirty kind '{kind}' never passes", passed == 0, f"(passed={passed})")

# 2) end-to-end run, twice with same seed, once with a different seed
def run(out, seed):
    cmd = [sys.executable, f"{ROOT}/sample_reviews.py", "--source", f"yelp={HERE}/mock_yelp.json",
           "--source", f"amazon={HERE}/mock_amazon.csv", "--out-dir", out, "--seed", str(seed)]
    subprocess.run(cmd, check=True, capture_output=True)
    return out
d1, d2, d3 = (tempfile.mkdtemp() for _ in range(3))
run(d1, 42); run(d2, 42); run(d3, 43)
h = lambda d: hashlib.md5(open(f"{d}/reviews.csv", "rb").read()).hexdigest()
check("same seed -> identical reviews.csv", h(d1) == h(d2))
check("different seed -> different sample", h(d1) != h(d3))

rows = list(csv.DictReader(open(f"{d1}/reviews.csv", encoding="utf-8")))
check("500 reviews selected", len(rows) == 500, f"(got {len(rows)})")
by = Counter((r["source"], int(r["stars"])) for r in rows)
check("exact per-source per-star quotas", all(by[(s, k)] == v for s in ("yelp", "amazon")
      for k, v in sr.DEFAULT_PER_STAR.items()))
check("no duplicate texts", len({r["text"] for r in rows}) == 500)
check("review ids unique & shuffled", len({r["review_id"] for r in rows}) == 500 and
      [r["source"] for r in rows[:20]].count("yelp") not in (0, 20))

# 3) every selected row must trace back to a 'clean' mock row
kind = {}
for line in open(f"{HERE}/mock_yelp.json"):
    r = json.loads(line); kind[("yelp", r["review_id"])] = r["_kind"]
for i, r in enumerate(csv.DictReader(open(f"{HERE}/mock_amazon.csv", encoding="utf-8")), 1):
    kind[("amazon", f"row{i}")] = r["_kind"]
sel_kinds = Counter(kind[(r["source"], r["source_review_id"])] for r in rows)
check("all selected reviews are clean rows", set(sel_kinds) == {"clean"}, str(dict(sel_kinds)))

# 4) annotation sheet is blind and sized right
sheet = list(csv.DictReader(open(f"{d1}/annotation_sheet.csv", encoding="utf-8")))
check("annotation sheet ~150 rows", 145 <= len(sheet) <= 160, f"(got {len(sheet)})")
check("annotation sheet hides stars/labels", set(sheet[0]) == {"review_id", "text", "annotator1_label", "annotator2_label", "notes"})
man = json.load(open(f"{d1}/sampling_manifest.json"))
check("manifest has seed, script hash, rejection counts",
      man["seed"] == 42 and len(man["script_sha256"]) == 64 and "rejected" in man["sources"]["yelp"])
print("\nALL TESTS PASSED" if ok else "\nSOME TESTS FAILED"); sys.exit(0 if ok else 1)
