# Paper ID 805 — Revision Runbook

Deadline: **October 15, 2026** (camera-ready + payment + registration). Today is October 1 — 14 days.

Every command below assumes you're in the `review_project/` folder with all six scripts
(`fetch_data.py`, `sample_reviews.py`, `build_prompts.py`, `run_models.py`, `analyze.py`,
`agreement.py`). Copy-paste the commands as-is; only the bracketed bits need changing.

---

## Timeline at a glance

| Day | Date | What |
|---|---|---|
| 1 | Oct 1 | Fetch data, sample reviews, send annotation sheet, build prompts |
| 2 | Oct 2 | Start Gemini run (long pole), start first open-weight model on Kaggle |
| 3–6 | Oct 3–6 | Gemini + open models keep running (resume daily); annotators work in parallel |
| 7 | Oct 7 | Collect annotator responses, run `agreement.py` |
| 8 | Oct 8 | **Checkpoint: if Gemini isn't done, cut it or drop to `--runs 1`** |
| 9 | Oct 9 | Run `analyze.py`, read the real results |
| 10–13 | Oct 10–13 | Write the revised paper + reviewer response |
| 14 | Oct 14 | Similarity check, register, pay, submit — a day early, not on the deadline |

---

## Day 1 (Oct 1) — Data

### 1. Install dependencies
```bash
pip install datasets gender-guesser statsmodels scipy pandas matplotlib scikit-learn
```

### 2. Pull real review data (no scraping)
```bash
python fetch_data.py --out-dir raw_data --limit 2000      # quick test first
```
Open `raw_data/yelp_reviews.jsonl` and `raw_data/amazon_reviews.jsonl`, check a few lines look
like real reviews with sensible `stars` values. **This script hasn't been run end-to-end by me
(no Hugging Face access in my sandbox) — if it errors, paste the error back and I'll fix it.**

If the test looks right, pull the full data:
```bash
python fetch_data.py --out-dir raw_data
```

### 3. Sample and clean 500 reviews
```bash
python sample_reviews.py \
    --source yelp=raw_data/yelp_reviews.jsonl \
    --source amazon=raw_data/amazon_reviews.jsonl \
    --out-dir data --seed 42
```
**Hand-check (10 minutes, don't skip):**
- Read the 10 random reviews printed at the end.
- Open `data/reviews.csv`, read ~50 more rows.
- Check `data/sampling_manifest.json` → `"shortfall"` should be `{}` for both sources. If not,
  you don't have enough eligible reviews in that star bracket — pull a bigger `--limit` in
  `fetch_data.py`, or pick a different Amazon category.
- Check the "Most-flagged name tokens" printout. Real names are expected; if you see ordinary
  English words being rejected, re-run with `--allow-words Word1,Word2`.

### 4. Send out the annotation sheet TODAY
`data/annotation_sheet.csv` (~150 reviews, stars hidden) is ready. Send it to your two (ideally
three) annotators now — labeling takes 2-3 hours and you want it done well before Day 7, not
blocking your last week.
Instructions for annotators: fill `annotator1_label` / `annotator2_label` with exactly one of
`positive`, `neutral`, `negative`, independently, without discussing with each other.

### 5. Build the prompts
```bash
python build_prompts.py --reviews data/reviews.csv --out-dir data --seed 42 --with-location-arm
```
Read the example prompts printed at the end — confirm only the "Reviewer:" line changes across
the 5 conditions for the same review. Skim `data/names.csv`.

---

## Day 2 (Oct 2) — Start the model runs

**Start Gemini first — it's the long pole because of the daily free-tier quota.**

### 6. Get a free Gemini API key
https://aistudio.google.com/apikey → set it:
```bash
export GEMINI_API_KEY=your_key_here
```

### 7. Start the Gemini run
```bash
python run_models.py --prompts data/prompts.csv --backend gemini \
    --model gemini-2.0-flash-lite --out-dir data --runs 3 --rpm 25 --rpd 1400
```
This will almost certainly NOT finish today (2,500 prompts × 3 runs = 7,500 calls vs. a ~1,400
call daily cap guess). That's expected. **Re-run this exact same command every day** — it
resumes automatically, skipping everything already done.

### 8. Start an open-weight model on Kaggle
In a new Kaggle notebook (free GPU, T4/P100):
```bash
pip install torch transformers accelerate
```
Upload or `%cd` into your `review_project` files, then:
```bash
python run_models.py --prompts data/prompts.csv --backend hf \
    --model Qwen/Qwen2.5-7B-Instruct --out-dir data --limit 10
```
**Read the 10 outputs in `data/raw_responses_*.csv` before going further.** If an obviously
positive review doesn't come back "positive", fix the prompt template in `build_prompts.py`
before spending GPU hours on the full 2,500. Once it looks right:
```bash
python run_models.py --prompts data/prompts.csv --backend hf \
    --model Qwen/Qwen2.5-7B-Instruct --out-dir data
```
This backend is deterministic (1 run/prompt) so it should finish well within a single 9-12 hour
Kaggle session — 2,500 forward passes on a free GPU.

---

## Days 3–6 (Oct 3–6) — Keep things running

- **Every day**, re-run the exact same Gemini command from Step 7 to continue where it left off.
  Check the printed "Progress: X/7500" line.
- **Run 2 more open-weight models** (different families — e.g. Llama-3, Gemma, or Mistral; check
  current model IDs on Hugging Face, some need a licence click-through first) the same way as
  Step 8, one Kaggle session each.
- **Check in on your annotators.** Nudge if needed — you want this done by Day 6 at the latest.

---

## Day 7 (Oct 7) — Annotation agreement

Once both/all annotators have filled in `data/annotation_sheet.csv`:
```bash
python agreement.py --sheet data/annotation_sheet.csv --reviews data/reviews.csv
```
Note the kappa and the raw agreement percentage — this goes directly in your Methods section
and reviewer response. If kappa is low (below ~0.6), look at the printed list of disagreements
and consider a third annotator to adjudicate before moving on.

---

## Day 8 (Oct 8) — Hard checkpoint

**Check Gemini's progress.** If it's not finished:
- Option A: let it run 1-2 more days if you have slack.
- Option B: cut `--runs` down — re-run with `--runs 1` isn't retroactive, but you can treat
  whatever partial data you have as valid (document this honestly in Methods: "N of M planned
  repeats completed due to free-tier quota limits").
- Option C: drop Gemini from the paper entirely and report only the open-weight models, noting
  this as a limitation. This is a legitimate fallback, not a failure.

**Do not let this slip past Day 9** — the statistics and writing need the remaining time more
than a third model needs its last few hundred API calls.

---

## Day 9 (Oct 9) — Analysis

```bash
python analyze.py --data-dir data --out-dir results --sesoi 0.2 --n-boot 2000
```
Read the printed summary first. Then open, per model:
- `results/bootstrap_<model>.json` — your headline numbers: mean difference, 95% CI, Cohen's d,
  TOST verdict, Holm-adjusted p-value.
- `results/lmm_<model>.json` and `results/subregion_lmm_<model>.json` — full mixed-model output.
- `results/classification_<model>.csv` — per-condition accuracy/F1 with CIs.
- `results/flip_rate_<model>.json` — ambiguous vs. clear-cut flip rates.
- `results/figures/` — the bar chart and confusion matrices for your paper's figures.

**Whatever you find is the result.** If the effect is small or absent at this scale, that is a
valid, publishable finding — write it up honestly rather than searching for a way to recover the
original 0.34-point effect.

---

## Days 10–13 (Oct 10–13) — Writing

Your own words throughout (the AI-content rule). Things to pull directly from the generated files:

| Paper section | Source file |
|---|---|
| Sample size / power justification | Your own power calculation (see earlier discussion); cite `sampling_manifest.json` for actual counts |
| Dataset description | `sampling_manifest.json`, `prompt_manifest.json` |
| Annotator agreement | `agreement.py` output (kappa, CI, raw agreement) |
| Model versions / dates queried | `run_manifest_<model>.json` for each model |
| Statistical method description | `analyze.py`'s own docstring describes the method in plain terms — paraphrase, don't copy |
| Results numbers | `bootstrap_<model>.json`, `classification_<model>.csv` |
| Figures | `results/figures/*.png` |
| Limitations | Each script's README.md "Known limits" section — review all of them |

Also: rewrite the abstract in formal academic style (Reviewer 3), update references so most are
2024–2026 (use the reading list from earlier in this conversation), fix the Table II / figure
numbering issues, and write the point-by-point reviewer response referencing what changed and
where.

---

## Day 14 (Oct 14) — Submit, don't wait for Oct 15

1. Run your institution's / Turnitin-equivalent similarity check — must be under 20%.
2. Register under "Regular Author" (or join SCRS first for the member rate) and pay.
3. Submit the camera-ready package per
   https://www.scrs.in/conference/icsiscet2026/page/Camera_Ready_Paper_Submission
   (paper PDF + source files + reviewer response + payment proof).
4. Keep Paper ID 805 handy for any correspondence to icsiscet@scrs.in.

---

## If something breaks

- **`fetch_data.py` errors** — paste the error, I'll fix it (untested here due to no HF access).
- **A Kaggle/HF model needs a licence click-through** — accept it on huggingface.co while logged
  in, then re-run; gated models fail silently otherwise.
- **Gemini quota is smaller/larger than expected** — adjust `--rpd` downward if you're getting
  429 errors in bulk, or just let the daily resume handle it either way.
- **`analyze.py` reports `"error"` in an LMM file** — usually too few reviews in a group, or a
  singular design matrix (a factor perfectly confounded with another). Check which factor.
