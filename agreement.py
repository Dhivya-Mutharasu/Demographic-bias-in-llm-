#!/usr/bin/env python3
"""
agreement.py  --  score the two-annotator validation sheet.

Usage
  python agreement.py --sheet data/annotation_sheet.csv --reviews data/reviews.csv

Both annotators fill annotator1_label / annotator2_label with exactly one of:
positive, neutral, negative   (blind: they never see the star rating).

Reports: % agreement, Cohen's kappa (nominal) and quadratic-weighted kappa with
95% bootstrap CIs, each annotator's agreement with the star-derived label, and a
list of disagreements to resolve (e.g. by discussion or a third annotator).
"""
import argparse
import csv
import random
from collections import Counter

LABELS = ["negative", "neutral", "positive"]
ORDER = {l: i for i, l in enumerate(LABELS)}


def cohen_kappa(a, b, weighted=False):
    n = len(a)
    if n == 0:
        return float("nan")
    k = len(LABELS)

    def w(i, j):  # disagreement weight
        return ((i - j) ** 2) / ((k - 1) ** 2) if weighted else float(i != j)

    obs = Counter(zip(a, b))
    ca, cb = Counter(a), Counter(b)
    do = sum(w(ORDER[x], ORDER[y]) * c for (x, y), c in obs.items()) / n
    de = sum(w(ORDER[x], ORDER[y]) * ca[x] * cb[y] for x in LABELS for y in LABELS) / (n * n)
    return 1.0 if de == 0 else 1.0 - do / de


def bootstrap_ci(a, b, weighted, reps=2000, seed=0):
    rng = random.Random(seed)
    n = len(a)
    vals = []
    for _ in range(reps):
        idx = [rng.randrange(n) for _ in range(n)]
        vals.append(cohen_kappa([a[i] for i in idx], [b[i] for i in idx], weighted))
    vals = sorted(v for v in vals if v == v)
    return vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals)) - 1]


def read_csv(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sheet", required=True)
    ap.add_argument("--reviews", required=True)
    args = ap.parse_args()

    stars_label = {r["review_id"]: r["sentiment_label"] for r in read_csv(args.reviews)}
    rows = read_csv(args.sheet)
    bad, a, b, ids = [], [], [], []
    for r in rows:
        l1, l2 = r["annotator1_label"].strip().lower(), r["annotator2_label"].strip().lower()
        if l1 not in ORDER or l2 not in ORDER:
            bad.append(r["review_id"])
            continue
        a.append(l1)
        b.append(l2)
        ids.append(r["review_id"])
    if bad:
        print(f"Skipped {len(bad)} rows with a missing/invalid label, e.g. {bad[:5]}")
    n = len(a)
    print(f"\nRated reviews: {n}")
    if n < 30:
        print("Too few rated reviews for a stable estimate (aim for 100+).")
        return

    agree = sum(x == y for x, y in zip(a, b)) / n
    k, kl, kh = cohen_kappa(a, b), *bootstrap_ci(a, b, False)
    kw, kwl, kwh = cohen_kappa(a, b, True), *bootstrap_ci(a, b, True)
    print(f"Raw agreement:               {agree:.1%}")
    print(f"Cohen's kappa (nominal):     {k:.3f}  (95% CI {kl:.3f}-{kh:.3f})")
    print(f"Weighted kappa (quadratic):  {kw:.3f}  (95% CI {kwl:.3f}-{kwh:.3f})")

    print("\nConfusion table (rows = annotator 1, cols = annotator 2):")
    t = Counter(zip(a, b))
    print(f"{'':10}" + "".join(f"{l:>10}" for l in LABELS))
    for x in LABELS:
        print(f"{x:10}" + "".join(f"{t[(x, y)]:>10}" for y in LABELS))

    star = [stars_label[i] for i in ids]
    for name, ann in (("Annotator 1", a), ("Annotator 2", b)):
        print(f"\n{name} vs star-derived label: agreement "
              f"{sum(x == y for x, y in zip(ann, star)) / n:.1%}, "
              f"kappa {cohen_kappa(ann, star):.3f}")

    dis = [i for i, x, y in zip(ids, a, b) if x != y]
    print(f"\n{len(dis)} disagreements to resolve: {dis[:20]}{' ...' if len(dis) > 20 else ''}")


if __name__ == "__main__":
    main()
