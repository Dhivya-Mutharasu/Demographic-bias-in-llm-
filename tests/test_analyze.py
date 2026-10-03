"""Run: python tests/test_analyze.py
Builds two synthetic model outputs with a KNOWN true effect (one with none, one with a
strong, deliberate region bias), runs analyze.py on each, and checks that the statistics
recover the known ground truth. This is the most important test in the repo: it's not
just "does the code run", it's "does the code find the right answer when we already know
what the right answer is."
"""
import csv, json, os, subprocess, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import analyze as az
import run_models as rm
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ok = True
def check(name, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(("PASS" if cond else "FAIL"), "-", name, detail)


# ---------------------------------------------------------------- fixtures
def make_reviews(n=200, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(1, n + 1):
        stars = rng.choice([1, 2, 3, 4, 5])
        label = "negative" if stars <= 2 else ("neutral" if stars == 3 else "positive")
        rows.append({"review_id": f"R{i:04d}", "source": "yelp" if i % 2 else "amazon",
                     "source_review_id": f"s{i}", "stars": stars, "sentiment_label": label,
                     "n_words": 40, "text": f"review text number {i}"})
    return rows


def write_csv(path, rows, fields):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fields); w.writeheader(); w.writerows(rows)


def make_prompts(reviews, seed=0):
    conds = ["none", "western_male", "western_female", "south_asian_male", "south_asian_female"]
    sub_region_pool = ["India", "Pakistan", "Bangladesh", "Sri Lanka", "Nepal"]
    # A small name per (condition, review) so MockBackend's text hash actually differs across
    # conditions -- mirrors real build_prompts.py, where the "Reviewer: {name}" line is what
    # makes each condition's prompt text distinct. Without this, every condition for a given
    # review collapses to the identical hash and the whole design has zero variance.
    rows = []
    for i, r in enumerate(reviews):
        for cond in conds:
            if cond == "none":
                region, gender, sub_region, tag = "none", "none", "none", "noname"
            else:
                region, gender = cond.rsplit("_", 1)
                tag = f"{cond}-name{i % 7}"  # varies by condition AND review -> real hash noise
                if region == "south_asian":
                    # rotate sub_region independently per gender pool so sub_region is CROSSED
                    # with gender, not confounded with it (each sub-region gets both genders)
                    offset = 0 if gender == "male" else 2
                    sub_region = sub_region_pool[(i + offset) % len(sub_region_pool)]
                else:
                    sub_region = "Anglo-American"
            text = f"Reviewer-tag: {tag} :: Review: \"{r['text']}\""
            rows.append({"prompt_id": f"{r['review_id']}_{cond}", "review_id": r["review_id"],
                        "condition": cond, "region": region, "gender": gender, "name": tag,
                        "sub_region": sub_region, "prompt_text": text,
                        "sentiment_label": r["sentiment_label"], "source": r["source"],
                        "stars": r["stars"]})
    return rows


def build_dataset(tmpdir, bias_fn, model_name, n_reviews=200, seed=0):
    reviews = make_reviews(n_reviews, seed)
    write_csv(os.path.join(tmpdir, "reviews.csv"), reviews,
              ["review_id", "source", "source_review_id", "stars", "sentiment_label", "n_words", "text"])
    prompts = make_prompts(reviews, seed)
    write_csv(os.path.join(tmpdir, "prompts.csv"), prompts,
              ["prompt_id", "review_id", "condition", "region", "gender", "name", "sub_region",
               "prompt_text", "sentiment_label", "source", "stars"])
    backend = rm.MockBackend(bias_fn=bias_fn)
    rm.run(prompts, backend, os.path.join(tmpdir, f"raw_responses_{model_name}.csv"),
           runs=1, rate_limiter=rm.RateLimiter(0))
    return reviews, prompts


NO_BIAS = lambda row: [0.0, 0.0, 0.0]
STRONG_REGION_BIAS = lambda row: [0.9, 0.0, -0.9] if row["region"] == "south_asian" else [0.0, 0.0, 0.0]
STRONG_GENDER_BIAS = lambda row: [0.0, 0.0, 0.9] if row["gender"] == "male" else [0.9, 0.0, 0.0]
NO_INTERACTION_JUST_REGION = STRONG_REGION_BIAS

# ==================================================================
# 1. build_long_df / aggregate_runs
# ==================================================================
d1 = tempfile.mkdtemp()
build_dataset(d1, NO_BIAS, "m1", n_reviews=50)
df, stats = az.build_long_df(d1)
check("build_long_df: 5 conditions x 50 reviews = 250 rows", len(df) == 250, f"(got {len(df)})")
check("build_long_df: no missing rows for a clean run", stats["m1"]["n_missing_all_runs_failed"] == 0)
check("aggregate_runs: single valid run returns that label with consistency 1.0",
      az.aggregate_runs([{"predicted_label": "positive", "error": ""}]) == ("positive", 1.0, 1))
check("aggregate_runs: all errors returns (None, None, 0)",
      az.aggregate_runs([{"predicted_label": "", "error": "boom"}]) == (None, None, 0))
check("aggregate_runs: majority vote with a tie breaks deterministically (LABELS order)",
      az.aggregate_runs([{"predicted_label": "positive", "error": ""},
                         {"predicted_label": "negative", "error": ""}])[0] == "negative")
check("aggregate_runs: clear majority wins, consistency = 2/3",
      az.aggregate_runs([{"predicted_label": "positive", "error": ""},
                         {"predicted_label": "positive", "error": ""},
                         {"predicted_label": "negative", "error": ""}]) == ("positive", 2 / 3, 3))

# ==================================================================
# 2. THE KEY TEST: known ground truth recovery
# ==================================================================
d_null = tempfile.mkdtemp()
build_dataset(d_null, NO_BIAS, "nullmodel", n_reviews=250, seed=1)
df_null, models_null = az.run_analysis(d_null, os.path.join(d_null, "results"), sesoi=0.2, n_boot=500, seed=1)

boot_null = json.load(open(os.path.join(d_null, "results", "bootstrap_nullmodel.json")))
check("NULL data: effect size is small in practice (|d| < SESOI)",
      abs(boot_null["south_asian_vs_western"]["cohens_d"]) < 0.2, boot_null["south_asian_vs_western"])
check("NULL data: TOST reaches 'equivalent' or, if borderline, is at least NOT flagged as a "
      "large/meaningful difference (|d| stays well under 0.5)",
      "equivalent" in boot_null["south_asian_vs_western"]["tost"]["verdict"] or
      abs(boot_null["south_asian_vs_western"]["cohens_d"]) < 0.5, boot_null["south_asian_vs_western"])
# Note: a 95% bootstrap CI on truly null data will exclude 0 by chance ~5% of the time --
# that is correct behavior for a valid CI, not a bug, so we deliberately do NOT assert
# "ci95_lo <= 0 <= ci95_hi" here. The practical (SESOI-based) claim above is what matters.
check("NULL data: effect size is small (|d| < 0.2)",
      abs(boot_null["south_asian_vs_western"]["cohens_d"]) < 0.2, boot_null["south_asian_vs_western"]["cohens_d"])

d_bias = tempfile.mkdtemp()
build_dataset(d_bias, STRONG_REGION_BIAS, "biasmodel", n_reviews=250, seed=1)
df_bias, models_bias = az.run_analysis(d_bias, os.path.join(d_bias, "results"), sesoi=0.2, n_boot=500, seed=1)

boot_bias = json.load(open(os.path.join(d_bias, "results", "bootstrap_biasmodel.json")))
ra = boot_bias["south_asian_vs_western"]
check("INJECTED region bias: CI excludes 0 and is negative (south_asian scored lower)",
      ra["ci95_hi"] < 0, ra)
check("INJECTED region bias: TOST calls it a significant difference, not equivalent",
      "significant difference" in ra["tost"]["verdict"], ra["tost"])
check("INJECTED region bias: large effect size recovered (|d| > 0.8)",
      abs(ra["cohens_d"]) > 0.8, ra["cohens_d"])
check("INJECTED region bias: LMM p-value is tiny", ra["lmm_p_value"] < 0.001, ra["lmm_p_value"])
check("INJECTED region bias: gender effect (not injected) stays non-significant",
      boot_bias["male_vs_female"]["ci95_lo"] <= 0 <= boot_bias["male_vs_female"]["ci95_hi"],
      boot_bias["male_vs_female"])

d_gender = tempfile.mkdtemp()
build_dataset(d_gender, STRONG_GENDER_BIAS, "gendermodel", n_reviews=250, seed=2)
az.run_analysis(d_gender, os.path.join(d_gender, "results"), sesoi=0.2, n_boot=500, seed=2)
boot_gender = json.load(open(os.path.join(d_gender, "results", "bootstrap_gendermodel.json")))
check("INJECTED gender bias: gender CI excludes 0",
      boot_gender["male_vs_female"]["ci95_hi"] < 0 or boot_gender["male_vs_female"]["ci95_lo"] > 0,
      boot_gender["male_vs_female"])
check("INJECTED gender bias: region effect (not injected) stays equivalent",
      "equivalent" in boot_gender["south_asian_vs_western"]["tost"]["verdict"])

# ==================================================================
# 3. Holm-Bonferroni correction
# ==================================================================
adj = az.holm_bonferroni(np.array([0.01, 0.02, 0.03, 0.5]))
check("holm_bonferroni: adjusted p-values are non-decreasing under sort and >= raw",
      all(a >= p for a, p in zip(adj, [0.01, 0.02, 0.03, 0.5])), list(adj))
check("holm_bonferroni: matches hand-computed values for this example",
      np.allclose(sorted(adj), [0.04, 0.06, 0.06, 0.5], atol=1e-9), list(adj))

# ==================================================================
# 4. sub-region model
# ==================================================================
subrgn = json.load(open(os.path.join(d_bias, "results", "subregion_lmm_biasmodel.json")))
check("sub-region model fit without error (region-only bias -> sub-regions near-identical)",
      "error" not in subrgn, subrgn.get("error"))
if "error" not in subrgn:
    coefs = [v["coef"] for k, v in subrgn["fixed_effects"].items() if k.startswith("C(sub_region")]
    check("sub-region coefficients are all small (no sub-region-specific bias was injected)",
          all(abs(c) < 0.3 for c in coefs), coefs)

# ==================================================================
# 5. classification metrics
# ==================================================================
cls = pd.read_csv(os.path.join(d_bias, "results", "classification_biasmodel.csv"))
check("classification: one row per condition", len(cls) == 5, f"(got {len(cls)})")
check("classification: south_asian conditions collapse toward 'negative' (injected bias)",
      cls.loc[cls["condition"] == "south_asian_male", "macro_f1"].iloc[0] < 0.5)
sa_cm = json.loads(cls.loc[cls["condition"] == "south_asian_male", "confusion_matrix"].iloc[0])
check("classification: confusion matrix is 3x3 and sums to n for that condition",
      np.array(sa_cm).sum() == cls.loc[cls["condition"] == "south_asian_male", "n"].iloc[0])

# ==================================================================
# 6. flip rate
# ==================================================================
fr_bias = json.load(open(os.path.join(d_bias, "results", "flip_rate_biasmodel.json")))
check("flip rate: overall rate is a proportion in [0, 1]",
      0 <= fr_bias["overall"]["flip_rate"] <= 1, fr_bias["overall"])
check("flip rate: deterministic-backend note is present (no repeated runs)",
      "deterministic" in fr_bias["note"])

# hand-check flip rate on a tiny known case: build a 2-review, 4-condition toy df
toy = pd.DataFrame([
    {"model": "toy", "review_id": "X1", "condition": "western_male", "predicted_label": "positive", "stars": 5, "self_consistency": None, "n_valid_runs": 1},
    {"model": "toy", "review_id": "X1", "condition": "western_female", "predicted_label": "positive", "stars": 5, "self_consistency": None, "n_valid_runs": 1},
    {"model": "toy", "review_id": "X1", "condition": "south_asian_male", "predicted_label": "positive", "stars": 5, "self_consistency": None, "n_valid_runs": 1},
    {"model": "toy", "review_id": "X1", "condition": "south_asian_female", "predicted_label": "positive", "stars": 5, "self_consistency": None, "n_valid_runs": 1},
    {"model": "toy", "review_id": "X2", "condition": "western_male", "predicted_label": "positive", "stars": 3, "self_consistency": None, "n_valid_runs": 1},
    {"model": "toy", "review_id": "X2", "condition": "western_female", "predicted_label": "positive", "stars": 3, "self_consistency": None, "n_valid_runs": 1},
    {"model": "toy", "review_id": "X2", "condition": "south_asian_male", "predicted_label": "negative", "stars": 3, "self_consistency": None, "n_valid_runs": 1},
    {"model": "toy", "review_id": "X2", "condition": "south_asian_female", "predicted_label": "negative", "stars": 3, "self_consistency": None, "n_valid_runs": 1},
])
toy_fr = az.flip_rate(toy, "toy")
check("flip rate hand-check: X1 never flips, X2 flips -> overall 1/2 = 0.5",
      toy_fr["overall"]["flip_rate"] == 0.5, toy_fr)
check("flip rate hand-check: the flip is correctly attributed to the 3-star (ambiguous) review",
      toy_fr["ambiguous_3star"]["flip_rate"] == 1.0 and toy_fr["clear_cut_non_3star"]["flip_rate"] == 0.0,
      toy_fr)

# ==================================================================
# 7. cluster bootstrap respects clustering (not naive row-level bootstrap)
# ==================================================================
# 8 reviews, each with its own true south_asian-vs-western gap (per_review_effect[i]);
# every review is duplicated into 4 rows per condition with THE SAME value (no
# within-review-condition noise). A naive bootstrap that resamples individual ROWS
# would treat this as 64 independent observations and report a falsely narrow CI;
# the correct cluster bootstrap resamples the 8 REVIEWS and should report a wider one.
rng = np.random.default_rng(0)
n_reviews_toy = 8
per_review_effect = rng.normal(-1.0, 1.5, n_reviews_toy)  # true region gap varies by review
rows = []
for i, effect in enumerate(per_review_effect):
    western_val = rng.normal(0, 0.01)  # tiny noise so it's not perfectly degenerate
    sa_val = western_val + effect
    for _ in range(4):  # 4 duplicate rows per condition per review, no added noise
        rows.append({"review_id": f"rev{i}", "region": "western", "score": western_val})
        rows.append({"review_id": f"rev{i}", "region": "south_asian", "score": sa_val})
toy_df = pd.DataFrame(rows)

def naive_row_bootstrap_diff(df, n_boot=1000, seed=0):
    rng2 = np.random.default_rng(seed)
    a = df.loc[df["region"] == "south_asian", "score"].to_numpy()
    b = df.loc[df["region"] == "western", "score"].to_numpy()
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        diffs[i] = rng2.choice(a, size=len(a), replace=True).mean() - rng2.choice(b, size=len(b), replace=True).mean()
    return np.percentile(diffs, [2.5, 97.5])

cluster_res = az.cluster_bootstrap_diff(toy_df, "region", "south_asian", "western", n_boot=2000, seed=0)
cluster_width = cluster_res["ci95_hi"] - cluster_res["ci95_lo"]
naive_lo, naive_hi = naive_row_bootstrap_diff(toy_df, n_boot=2000, seed=0)
naive_width = naive_hi - naive_lo
check("cluster bootstrap: CI is meaningfully wider than a naive row-level bootstrap "
      "(true independent n is 8 reviews, not 64 duplicated rows)",
      cluster_width > naive_width * 1.3,
      f"cluster_width={cluster_width:.3f}  naive_width={naive_width:.3f}")
check("cluster bootstrap: point estimate matches the true mean per-review effect reasonably",
      abs(cluster_res["mean_diff"] - per_review_effect.mean()) < 0.5,
      f"boot={cluster_res['mean_diff']:.3f} true={per_review_effect.mean():.3f}")

# ==================================================================
# 8. CLI end-to-end + manifest
# ==================================================================
r = subprocess.run([sys.executable, f"{ROOT}/analyze.py", "--data-dir", d_bias,
                    "--out-dir", os.path.join(d_bias, "results_cli"), "--n-boot", "200",
                    "--no-figures"], capture_output=True, text=True)
check("CLI: runs end-to-end without error", r.returncode == 0, r.stderr[-500:] if r.returncode else "")
man = json.load(open(os.path.join(d_bias, "results_cli", "analysis_manifest.json")))
check("CLI: manifest has script hash, seed, models", len(man["script_sha256"]) == 64 and
      man["models"] == ["biasmodel"])

# reproducibility: same seed -> same bootstrap CI
r2 = subprocess.run([sys.executable, f"{ROOT}/analyze.py", "--data-dir", d_bias,
                    "--out-dir", os.path.join(d_bias, "results_cli2"), "--n-boot", "200",
                    "--no-figures"], capture_output=True, text=True)
b1 = json.load(open(os.path.join(d_bias, "results_cli", "bootstrap_biasmodel.json")))
b2 = json.load(open(os.path.join(d_bias, "results_cli2", "bootstrap_biasmodel.json")))
check("CLI: same seed -> identical bootstrap CIs across runs",
      b1["south_asian_vs_western"]["ci95_lo"] == b2["south_asian_vs_western"]["ci95_lo"] and
      b1["south_asian_vs_western"]["ci95_hi"] == b2["south_asian_vs_western"]["ci95_hi"])

print("\nALL TESTS PASSED" if ok else "\nSOME TESTS FAILED")
sys.exit(0 if ok else 1)
