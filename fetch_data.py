#!/usr/bin/env python3
"""
fetch_data.py  --  Step 0: pull real review data, no scraping, no multi-GB manual downloads.

Uses the `datasets` library to pull two small, citable, real-review sources and writes
them as JSONL with the exact field names sample_reviews.py already auto-detects
("text" and "stars"), so you can go straight to Step 1 with no column-mapping flags.

Sources:
  yelp    Yelp/yelp_review_full on Hugging Face -- a standard derivative of the Yelp Dataset
          Challenge 2015 data (Zhang, Zhao & LeCun, NeurIPS 2015), 650k train + 50k test
          reviews, star ratings 1-5. Much smaller / faster than the ~4.35 GB raw Yelp Open
          Dataset tarball, and doesn't require pulling business.json to join star ratings.
          Cite: Zhang, Zhao & LeCun (2015), "Character-level Convolutional Networks for
          Text Classification", NeurIPS 28.

  amazon  McAuley-Lab/Amazon-Reviews-2023 on Hugging Face, ONE category only (default
          "Gift_Cards", one of the smallest -- avoid "Books" or "Electronics", which are
          many GB each). Cite: Hou et al. (2024), arXiv:2403.03952.

NOT TESTED in this sandbox: Hugging Face is not reachable from here, so this script has not
been run end-to-end. Run it yourself first with --limit 2000 (fast) and sanity-check the
output before pulling the full thing.

Usage
  pip install datasets
  python fetch_data.py --out-dir raw_data
  python fetch_data.py --out-dir raw_data --amazon-category All_Beauty --limit 5000  # quick test

Then:
  python sample_reviews.py --source yelp=raw_data/yelp_reviews.jsonl \
                            --source amazon=raw_data/amazon_reviews.jsonl \
                            --out-dir data --seed 42
"""
import argparse
import json
import sys
from pathlib import Path


def write_jsonl(records, path):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def fetch_yelp(out_path, limit=None):
    from datasets import load_dataset

    print("Loading Yelp/yelp_review_full (train split) ...")
    ds = load_dataset("Yelp/yelp_review_full", split="train")
    if limit:
        ds = ds.select(range(min(limit, len(ds))))

    def gen():
        for row in ds:
            yield {"text": row["text"], "stars": row["label"] + 1}  # label is 0-4 -> stars 1-5

    write_jsonl(gen(), out_path)
    print(f"Wrote {len(ds):,} Yelp reviews -> {out_path}")


def fetch_amazon(out_path, category="Gift_Cards", limit=None):
    from datasets import load_dataset

    config = f"raw_review_{category}"
    print(f"Loading McAuley-Lab/Amazon-Reviews-2023 [{config}] ...")
    try:
        ds = load_dataset("McAuley-Lab/Amazon-Reviews-2023", config, split="full",
                          trust_remote_code=True)
    except Exception as e:
        raise SystemExit(
            f"Could not load category '{category}' ({e}).\n"
            f"Check the exact category name in all_categories.txt at "
            f"https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023 -- "
            f"category names are case-sensitive with underscores, e.g. 'Gift_Cards', "
            f"'All_Beauty', 'Musical_Instruments'. Small categories load much faster; "
            f"avoid 'Books', 'Electronics', 'Clothing_Shoes_and_Jewelry' (many GB each)."
        )
    if limit:
        ds = ds.select(range(min(limit, len(ds))))

    def gen():
        for row in ds:
            text = (row.get("text") or "").strip()
            if row.get("title"):
                text = f"{row['title'].strip()} {text}".strip()
            if text:
                yield {"text": text, "stars": int(round(row["rating"]))}

    write_jsonl(gen(), out_path)
    print(f"Wrote up to {len(ds):,} Amazon ({category}) reviews -> {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default="raw_data")
    ap.add_argument("--amazon-category", default="Gift_Cards",
                    help="see https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023 "
                         "'all_categories.txt' for the full list. Pick a SMALL one.")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap rows per source, for a fast test run before the full pull")
    ap.add_argument("--skip-yelp", action="store_true")
    ap.add_argument("--skip-amazon", action="store_true")
    args = ap.parse_args()

    try:
        import datasets  # noqa: F401
    except ImportError:
        raise SystemExit("Run: pip install datasets")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_yelp:
        fetch_yelp(out_dir / "yelp_reviews.jsonl", args.limit)
    if not args.skip_amazon:
        fetch_amazon(out_dir / "amazon_reviews.jsonl", args.amazon_category, args.limit)

    print("\nNext step:")
    print(f"  python sample_reviews.py --source yelp={out_dir}/yelp_reviews.jsonl "
          f"--source amazon={out_dir}/amazon_reviews.jsonl --out-dir data --seed 42 "
          f"--max-scan 20000   # dry run first")


if __name__ == "__main__":
    main()
