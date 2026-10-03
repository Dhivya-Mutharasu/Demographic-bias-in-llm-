# Step 1: Build the review stimulus set

Draws ~500 real reviews (250 per source) from Yelp / Amazon, stratified by star rating,
after removing anything that would interfere with a name-swap experiment.

## Get the data (do not scrape)
- **Yelp Open Dataset** - `yelp_academic_dataset_review.json` (JSON lines). Official site or the Kaggle mirror.
- **Amazon Reviews 2023 (McAuley Lab)** - download ONE or two category files only (the full repo is ~750 GB).
  Kaggle mirrors exist too.
- Read the licence/terms on the original page and cite the original source in the paper.

## Run
Dry run first (fast, ~200k rows per source):

    pip install gender-guesser
    python sample_reviews.py \
        --source yelp=/path/yelp_academic_dataset_review.json \
        --source amazon=/path/amazon_category.jsonl \
        --out-dir data --seed 42 --max-scan 200000

If the summary looks right, drop `--max-scan` for the real run. On Kaggle, use the paths shown in
the notebook's *Input* panel (`/kaggle/input/...`) and `--out-dir /kaggle/working/data`.
Columns are auto-detected and printed; override with `--text-col` / `--rating-col` if needed.

## What the filters remove
length outside 20-120 words - URLs/emails/phone numbers - non-English - **person names** (first-name
database) - gender cues ("my husband", "as a woman", "sir") - ethnicity / nationality / religion / caste
words - place names (countries, US states, big cities) - exact duplicates (across both sources).

Why: if the review itself mentions "our server Maria" or "as an Indian", the name you attach in the
experiment is no longer the only identity signal.

## Checks you must do by hand (10 minutes)
1. Read the 10 random reviews printed at the end, then open `reviews.csv` and read ~50 more.
   Look for: a person's name, a place, a gendered self-reference, or a review about something odd.
2. Look at **"Most-flagged name tokens"** in the printout. Real names are expected. If you see ordinary
   words (e.g. `Way`, `Kind`), re-run with `--allow-words Way,Kind` so you don't lose good reviews.
3. Check `sampling_manifest.json` -> `shortfall` is empty for both sources.

## Known limits (state these in the paper)
- Name filter is a heuristic: it can miss lowercase names or nicknames and over-excludes some words.
- Removing cuisine/nationality words drops "Indian / Chinese / Italian ..." restaurant reviews, so the
  Yelp sample under-represents them.
- Star rating is a noisy sentiment label (especially 3 stars) - hence the human validation below.

## Human validation (answers Reviewer 2's "how were labels validated?")
`annotation_sheet.csv` (~150 reviews, stars hidden). Two people label independently, using only
`positive`, `neutral`, `negative`, without discussing. Then:

    python agreement.py --sheet data/annotation_sheet.csv --reviews data/reviews.csv

Reports raw agreement, Cohen's kappa and weighted kappa with 95% bootstrap CIs, agreement with the
star-derived label, and the disagreements to resolve. Report the kappa (and which kappa) in the paper.

## For the Methods section
`sampling_manifest.json` records the seed, filters, rows scanned, rejections per reason, selected counts
and the SHA-256 of the script. Cite these numbers directly. Same seed + same input files = identical sample.

## Tests
    python tests/make_mock_data.py && python tests/test_sampler.py
Runs on generated mock data (no real Yelp/Amazon data in the tests).

---

# Step 2: Attach identity conditions and build prompts

Takes `reviews.csv` and produces one prompt per (review, condition). **The review text is
byte-identical across conditions** - only a one-line "Reviewer:" preamble changes - so any
difference an LLM produces is attributable to that one line, not to different wording.

## Conditions
`none` (no reviewer line), `western_male`, `western_female`, `south_asian_male`, `south_asian_female`.
Each named condition draws from a pool of **12 names spanning multiple sub-regions** - the
south_asian pools cover India, Pakistan, Bangladesh, Sri Lanka and Nepal (not just North-Indian,
Hindu-majority names), and the western pools include a couple of Irish/Scottish-coded surnames
alongside the Anglo-American majority. Every prompt carries a `sub_region` field
(e.g. "India", "Pakistan", "Anglo-American", "Irish") so you can check in the analysis whether an
effect is uniform across the region it claims to represent, or concentrated in one sub-group -
report both. Edit `names.csv` (columns: `condition,name,sub_region`) and re-run with
`--names-file` to change the pools.

Names are assigned with an **independent, balanced round-robin per condition**: every name is
used the same number of times (± 1), and a review's name in one condition is unrelated to its
name in another condition - this directly answers Reviewer 2's "same 26 reviews confounded
across conditions" objection.

## Run
    python build_prompts.py --reviews data/reviews.csv --out-dir data --seed 42 --with-location-arm

`--with-location-arm` additionally builds `location_prompts.csv`: a **separate**, name-free
mini-experiment that varies only a place tag (e.g. "Mumbai, India" vs "Boston, USA"). Use it to
check whether an effect you see with names is really about the name, or about an
associated place - i.e. to start disentangling the confounded cues reviewers flagged.

## What you get
- `prompts.csv` - 5 x N rows: `prompt_id, review_id, condition, region, gender, name, prompt_text,
  sentiment_label, source, stars`. `sentiment_label` is the ground truth carried over from Step 1.
- `names.csv` - the name pools actually used (put this in the paper's appendix).
- `prompt_manifest.json` - seed, template text, script hash, per-condition name-usage counts.
  Cite these numbers in Methods; re-running with the same seed reproduces `prompts.csv` exactly.

## Checks you must do by hand
1. Read the example prompts printed at the end for one review across all 5 conditions - confirm
   only the "Reviewer:" line changes.
2. Skim `names.csv` - if any name looks off to you or your guides, edit it and re-run.
3. Decide now (before running models) how you'll aggregate the 3 repeated API runs per prompt -
   this is Step 3, and the analysis in Step 4 needs to know the repeated-measures structure.

## Known limits (state these in the paper)
- 12 names/pool, spread across 5 (South Asian) or 3 (Western) sub-regions, means only 2-4 names
  represent each sub-region - still a compromise between diversity and a tractable design, and far
  from the full name space of any of these countries.
- The south_asian pools use one common, non-exhaustive name per sub-region x gender cell; they do
  not capture the religious, linguistic or caste diversity within India, Pakistan, Bangladesh,
  Sri Lanka or Nepal. Report the sub-region breakdown alongside the pooled South Asian result.
- The location arm is a secondary, exploratory check, not a full crossed factorial with name x
  location - flag it as such if you report it.

## Tests
    python tests/test_prompts.py [path/to/reviews.csv]
Checks: exact prompt counts, name balance, sub-region spread, text-identity across conditions,
reproducibility under a fresh Python process (guards against seed bugs from hash-randomization),
the `--names-file` override path, and the location arm.

---

# Step 3: Run the models (free)

Sends every prompt to one model and appends results to `data/raw_responses_<model>.csv`.
Run this script once per model. It is **resumable**: Ctrl-C any time, re-run the exact same
command, and it picks up where it left off without duplicating or re-sending anything.

## Backends

**`hf` - open-weight models on a free Kaggle/Colab GPU.** Scores the three labels by
teacher-forced log-likelihood (the same idea as lm-eval-harness) instead of free-form
generation. This is fully deterministic given fixed weights, so it needs only **one run per
prompt** - no sampling noise to explain away, unlike the original paper's 156-vs-468-calls issue.

    pip install torch transformers accelerate
    python run_models.py --prompts data/prompts.csv --backend hf \
        --model Qwen/Qwen2.5-7B-Instruct --out-dir data

Run this three times with three different model families (e.g. Qwen, Llama-3, Gemma or Mistral -
check current model names/versions on Hugging Face; some need a licence click-through first).

**`gemini` - Google's free tier, for one closed model.** Free-tier calls are not guaranteed
deterministic even at temperature 0, and don't reliably expose log-probabilities, so this backend
asks for a one-word label and repeats each prompt `--runs` times (default 3) - the repeats are
kept as separate rows (`run_index`) for Step 4 to model explicitly, not silently averaged away.

    export GEMINI_API_KEY=...   # free key: https://aistudio.google.com/apikey
    python run_models.py --prompts data/prompts.csv --backend gemini \
        --model gemini-2.0-flash-lite --out-dir data --runs 3 --rpm 25 --rpd 1400

Free-tier daily caps are typically a few hundred to ~1500 requests and Google changes them
without notice - check your quota at https://aistudio.google.com before running. At 2,500
prompts x 3 runs = 7,500 calls, **this usually takes several days**. That's expected: `--rpd`
stops the script cleanly after that many new calls, and re-running the identical command
tomorrow continues from where it stopped. Lower `--runs` to 1 if you're short on time or quota,
and say so plainly in the paper's limitations.

**`mock` - no model, for testing.** Deterministic pseudo-labels from a hash of the prompt text.
Use it to test your file paths, quotas and Step 4 analysis code end-to-end before spending real
API quota or GPU hours.

## What you get
`data/raw_responses_<model>.csv`: one row per (prompt, run) with `predicted_label`,
per-label probabilities (`hf` backend only - Gemini gives text, not probabilities),
`raw_text`, `latency_ms`, `error`, and a UTC timestamp. `data/run_manifest_<model>.json` records
the backend, model, version string, and cumulative call counts - cite this in Methods for
Reviewer 2's "exact model, version, date" request.

## Checks you must do by hand
1. **Before running the full 2,500:** run with `--limit 10` and read the `raw_text`/`predicted_label`
   for a few obviously-positive and obviously-negative reviews. If they look wrong, fix the
   prompt template in `build_prompts.py` before spending quota on the rest.
2. For the `hf` backend specifically: there's no automated test of the actual model output in this
   repo (no GPU/model download here) - only the label-scoring math is tested. Sanity-check the
   first real run yourself.
3. Watch `error` counts in the printed summary. A high `unparseable_label` count from Gemini means
   the model isn't returning a clean one-word answer - tighten the prompt.

## Known limits (state these in the paper)
- The `hf` backend measures the log-likelihood the model assigns to each label word, which is
  standard practice but is not identical to asking the model to generate an answer and reading
  it - note this as the scoring method in Methods.
- Gemini's free-tier lack of log-probabilities means the two backends aren't measured on quite
  the same footing (discrete generated label vs. a continuous probability). Report them side by
  side, not pooled into one number.
- Daily quota limits mean the Gemini run spans several days; if Google changes its free-tier
  limits mid-run (they do this without notice), note the date range you queried in the paper.

## Tests
    python tests/test_run_models.py
All using the mock backend and fake HTTP responses (no GPU, no API key, no network needed):
resumability after a simulated crash, no duplicate rows, daily-cap stop/resume, repeated-run
handling for non-deterministic backends, retried vs. silently-dropped errors, the label-scoring
math in isolation, and Gemini's retry/backoff/parsing logic against fake responses.

---

# Step 4: Analysis

Joins `reviews.csv` + `prompts.csv` + every `raw_responses_<model>.csv` into one table and runs
the statistics reviewers actually asked for.

## Run
    python analyze.py --data-dir data --out-dir results --sesoi 0.2 --n-boot 2000

`--sesoi` is the smallest effect size (Cohen's d) you'd call "practically meaningful" - the default
0.2 is a conventional "small effect" threshold; discuss and justify your choice with your guides
rather than treating 0.2 as automatically correct for this context.

## What it does
1. **Mixed-effects model** (`score ~ region * gender`, random intercept per review) instead of a
   one-way ANOVA - the correct model when the same reviews are scored under every condition
   (Reviewer 2's main statistics complaint). A second model does the same thing for `sub_region`
   within the South Asian subset only, so "South Asian" isn't treated as one block.
2. **Cluster bootstrap CIs** for the two primary contrasts (South Asian vs. Western, Male vs.
   Female) - resampling whole *reviews*, not individual rows, since rows from the same review
   aren't independent observations.
3. **TOST equivalence test** on each contrast: if the 90% CI sits entirely inside ±SESOI, that
   contrast is reported as "equivalent" (a real "no meaningful difference" finding), not just "not
   significant" - these are different claims and reviewers increasingly expect the distinction.
4. **Holm-Bonferroni correction** across the small, pre-specified set of primary contrasts (not
   across every possible pairing you could imagine).
5. **Confusion matrices and accuracy/precision/recall/F1 per condition**, with bootstrap CIs, to
   check whether error rates differ by group on identical text.
6. **Flip-rate**: how often the *same* review gets a different label depending only on which name
   is attached, reported separately for 3-star (ambiguous) vs. clear-cut reviews.

## What you get
- `merged_long.csv` - the full joined dataset, one row per (model, review, condition); useful for
  your own spot-checks or a different analysis.
- `lmm_<model>.json`, `subregion_lmm_<model>.json` - fixed-effect coefficients, p-values, CIs.
- `bootstrap_<model>.json` - the two primary contrasts: mean difference, 95% CI, Cohen's d, TOST
  verdict, and the Holm-adjusted p-value. **This file has the numbers for your Results section.**
- `classification_<model>.csv` - per-condition accuracy/precision/recall/F1 with bootstrap CIs.
- `flip_rate_<model>.json` - overall and ambiguous-vs-clear-cut flip rates.
- `figures/mean_score_by_condition.png`, `figures/confusion_<model>.png`.

## Checks you must do by hand
1. Read the printed summary first - it states the two headline numbers per model in plain
   language (mean difference, CI, effect size, TOST verdict).
2. If `lmm_<model>.json` has an `"error"` field instead of `"fixed_effects"`, the model didn't
   converge or the design matrix was singular (often means too few reviews, or a factor that ended
   up perfectly confounded with another - the way sub_region and gender would be if your names
   file ever ties a specific sub-region only to one gender).
3. Look at `mean_within_condition_self_consistency` in the Gemini model's `flip_rate_*.json` (if
   you ran it with `--runs 3`) - a low number there means the model itself is noisy even holding
   the identity condition fixed, which should temper how much of the flip-rate you attribute to
   the name versus ordinary model inconsistency.

## Known limits (state these in the paper)
- The TOST SESOI (0.2 by default) is a judgment call, not a fact about the world - report it
  explicitly and consider a sensitivity check at 0.1 and 0.3.
- The LMM's own p-values (Wald tests) and the cluster-bootstrap CIs are two different estimators
  and can disagree slightly near a boundary - this is expected, not a bug; report both rather than
  picking whichever one looks more favorable.
- A single 95% CI will exclude zero by chance about 5% of the time even when there is truly no
  effect - a lone "significant" result for a sub-region or gender contrast you didn't
  pre-specify should be treated as exploratory, not confirmatory.

## Tests
    python tests/test_analyze.py
The important test here isn't just "does the code run" - it builds two synthetic datasets with a
**known, injected** effect (one with none, one with a large deliberate South-Asian-vs-Western gap)
and checks that the pipeline recovers the right answer in both directions: the null case comes
back small-and-equivalent, the injected case comes back large-and-significant with the correct
sign, and an effect injected on gender instead of region shows up only on the gender contrast, not
region. Also checks the cluster bootstrap is actually respecting the review-level clustering
(vs. a naive row-level bootstrap), Holm-Bonferroni arithmetic against a hand-computed example, and
the flip-rate logic against a hand-built 2-review toy case.
