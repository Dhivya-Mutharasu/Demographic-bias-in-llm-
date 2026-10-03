#!/usr/bin/env python3
"""
analyze.py  --  Step 4 of the demographic-bias project.

Joins reviews.csv + prompts.csv + every raw_responses_<model>.csv (from Steps
1-3) into one long table, then runs the analysis reviewers actually asked for:

  1. A linear mixed-effects model (score ~ region * gender, random intercept
     per review) instead of the one-way ANOVA the original paper used -- this
     is the correct model for a repeated-measures design where the same
     reviews are scored under every condition (Reviewer 2's main statistics
     complaint).
  2. Cluster bootstrap confidence intervals (resampling whole reviews, not
     individual rows, since rows from the same review are not independent)
     for every reported effect, plus a TOST equivalence test so a small,
     tight effect can be reported as "no meaningful difference" rather than
     just "not significant" (which is not the same claim).
  3. A separate model on the South-Asian subset only, testing sub_region
     (India / Pakistan / Bangladesh / Sri Lanka / Nepal), so "South Asian"
     isn't treated as a monolith.
  4. Per-condition confusion matrices and classification metrics against the
     star-derived ground truth, with bootstrap CIs (error-parity check).
  5. A flip-rate analysis: how often the SAME review gets a different label
     depending only on which name is attached, reported separately for
     3-star (ambiguous) vs. clear-cut reviews.

Multiple-comparison correction (Holm-Bonferroni) is applied across the small,
pre-specified set of primary contrasts -- not across every possible pairing.

Usage
  python analyze.py --data-dir data --out-dir results --sesoi 0.2 --n-boot 2000

Outputs (in --out-dir):
  merged_long.csv                 the joined, row-level dataset (for your own checks)
  lmm_<model>.json                 mixed-model fixed effects + Holm-adjusted p-values
  subregion_lmm_<model>.json       South-Asian-only sub-region model
  bootstrap_<model>.json           cluster-bootstrap CIs, Cohen's d, TOST verdicts
  classification_<model>.csv       per-condition accuracy / precision / recall / F1 + CIs
  flip_rate_<model>.json           cross-condition label-flip analysis
  figures/*.png                    bar chart (mean score by condition) + confusion matrices
  analysis_manifest.json           seed, settings, models found, script hash
"""
import argparse
import csv
import glob
import hashlib
import json
import os
import random
import re
import sys
import time
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score

LABELS = ["negative", "neutral", "positive"]
SCORE = {"negative": -1, "neutral": 0, "positive": 1}
NAMED_CONDITIONS = ["western_male", "western_female", "south_asian_male", "south_asian_female"]


def read_csv(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


# --------------------------------------------------------------------------
# 1. Load + aggregate repeated runs
# --------------------------------------------------------------------------
def aggregate_runs(rows):
    """rows: raw_responses rows for ONE prompt_id (>=1 runs).
    Returns (majority_label_or_None, self_consistency_or_None, n_valid_runs).
    Ties broken deterministically by LABELS order. All-error prompts -> (None, None, 0).
    """
    valid = [r["predicted_label"] for r in rows if r.get("predicted_label") and not r.get("error")]
    if not valid:
        return None, None, 0
    counts = Counter(valid)
    top = max(counts.values())
    winners = [lab for lab in LABELS if counts.get(lab) == top]
    majority = winners[0]
    consistency = top / len(valid)
    return majority, consistency, len(valid)


def build_long_df(data_dir):
    reviews = {r["review_id"]: r for r in read_csv(Path(data_dir) / "reviews.csv")}
    prompts = {r["prompt_id"]: r for r in read_csv(Path(data_dir) / "prompts.csv")}

    response_files = sorted(glob.glob(str(Path(data_dir) / "raw_responses_*.csv")))
    if not response_files:
        raise SystemExit(f"No raw_responses_*.csv files found in {data_dir}. Run run_models.py first.")

    records = []
    per_model_stats = {}
    for path in response_files:
        model_slug = re.sub(r"^raw_responses_|\.csv$", "", os.path.basename(path))
        rows = read_csv(path)
        by_prompt = defaultdict(list)
        for r in rows:
            by_prompt[r["prompt_id"]].append(r)

        n_missing = 0
        for prompt_id, prows in by_prompt.items():
            prompt = prompts.get(prompt_id)
            if prompt is None:
                continue  # stale rows from an old prompt set; ignore rather than crash
            review = reviews.get(prompt["review_id"])
            if review is None:
                continue
            majority, consistency, n_valid = aggregate_runs(prows)
            if majority is None:
                n_missing += 1
                continue
            records.append({
                "model": model_slug,
                "review_id": prompt["review_id"],
                "prompt_id": prompt_id,
                "condition": prompt["condition"],
                "region": prompt["region"],
                "gender": prompt["gender"],
                "sub_region": prompt.get("sub_region", "none"),
                "predicted_label": majority,
                "score": SCORE[majority],
                "self_consistency": consistency,
                "n_valid_runs": n_valid,
                "ground_truth_label": review["sentiment_label"],
                "ground_truth_score": SCORE[review["sentiment_label"]],
                "correct": int(majority == review["sentiment_label"]),
                "stars": int(review["stars"]),
                "source": review["source"],
            })
        per_model_stats[model_slug] = {"n_prompts_seen": len(by_prompt), "n_missing_all_runs_failed": n_missing}

    df = pd.DataFrame.from_records(records)
    if df.empty:
        raise SystemExit("No usable (non-error) responses found across any model file.")
    return df, per_model_stats


# --------------------------------------------------------------------------
# 2. Cluster bootstrap + TOST
# --------------------------------------------------------------------------
def cluster_bootstrap_diff(df, group_col, level_a, level_b, value_col="score",
                            cluster_col="review_id", n_boot=2000, seed=0):
    """Bootstrap the difference in mean(value_col) between two levels of group_col,
    resampling whole clusters (reviews) with replacement -- respects the fact that
    rows from the same review, across conditions, are not independent observations.
    Every row for a resampled review is included (not resampled again within-review),
    so the paired review-level structure is preserved.
    """
    sub = df[df[group_col].isin([level_a, level_b])]
    clusters = sub[cluster_col].unique()
    rng = np.random.default_rng(seed)
    n = len(clusters)

    def point_estimate(data):
        a = data.loc[data[group_col] == level_a, value_col].mean()
        b = data.loc[data[group_col] == level_b, value_col].mean()
        return a - b

    observed = point_estimate(sub)
    boot_diffs = np.empty(n_boot)
    grouped = {c: idx for c, idx in sub.groupby(cluster_col).groups.items()}
    for i in range(n_boot):
        sampled_clusters = rng.choice(clusters, size=n, replace=True)
        idx = np.concatenate([grouped[c].to_numpy() for c in sampled_clusters])
        boot_diffs[i] = point_estimate(sub.loc[idx])
    ci_lo, ci_hi = np.percentile(boot_diffs, [2.5, 97.5])

    pooled_sd = sub[value_col].std(ddof=1)
    cohen_d = observed / pooled_sd if pooled_sd > 0 else float("nan")
    return {
        "level_a": level_a, "level_b": level_b, "n_clusters": int(n),
        "mean_diff": float(observed), "ci95_lo": float(ci_lo), "ci95_hi": float(ci_hi),
        "cohens_d": float(cohen_d), "boot_diffs": boot_diffs,  # kept for TOST reuse; stripped before JSON dump
    }


def tost_decision(boot_result, sesoi_d, pooled_sd_for_sesoi):
    """Two one-sided-test style equivalence check using the bootstrap distribution:
    equivalent if the 90% CI of the difference lies entirely within +/- SESOI (in raw
    score units, converted from the Cohen's-d SESOI using this contrast's pooled SD).
    """
    sesoi_raw = sesoi_d * pooled_sd_for_sesoi
    lo90, hi90 = np.percentile(boot_result["boot_diffs"], [5, 95])
    equivalent = (lo90 > -sesoi_raw) and (hi90 < sesoi_raw)
    significant = (boot_result["ci95_lo"] > 0) or (boot_result["ci95_hi"] < 0)
    if equivalent:
        verdict = "equivalent (practically no difference)"
    elif significant:
        verdict = "significant difference, and NOT shown equivalent"
    else:
        verdict = "inconclusive (neither significant nor equivalent -- likely underpowered)"
    return {"sesoi_cohens_d": sesoi_d, "sesoi_raw_units": float(sesoi_raw),
            "ci90_lo": float(lo90), "ci90_hi": float(hi90), "verdict": verdict}


def holm_bonferroni(pvals):
    """Returns adjusted p-values (Holm's step-down method)."""
    idx = np.argsort(pvals)
    m = len(pvals)
    adjusted = np.empty(m)
    running_max = 0.0
    for rank, i in enumerate(idx):
        val = (m - rank) * pvals[i]
        running_max = max(running_max, val)
        adjusted[i] = min(running_max, 1.0)
    return adjusted


# --------------------------------------------------------------------------
# 3. Mixed-effects model
# --------------------------------------------------------------------------
def fit_mixed_model(df, group_var, model_name):
    """score ~ C(region)*C(gender) [or C(sub_region)*C(gender)], random intercept
    per review_id. Returns a JSON-safe dict of fixed effects, or an error note if
    the model can't be fit (e.g. too few groups)."""
    import statsmodels.formula.api as smf

    sub = df[df["model"] == model_name].copy()
    formula = f"score ~ C({group_var}, Treatment('western' if group_var=='region' else 'India')) * C(gender)"
    # statsmodels needs a concrete formula string; build it explicitly instead of the
    # conditional expression above (kept only as a comment for readability).
    ref = "western" if group_var == "region" else "India"
    formula = f"score ~ C({group_var}, Treatment('{ref}')) * C(gender, Treatment('male'))"
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            md = smf.mixedlm(formula, sub, groups=sub["review_id"])
            fit = md.fit(reml=True)
    except Exception as e:  # noqa: BLE001 - surface as a result, don't crash the whole run
        return {"model": model_name, "group_var": group_var, "error": str(e)}

    params = fit.params.to_dict()
    pvals = fit.pvalues.to_dict()
    conf = fit.conf_int()
    out_terms = {}
    for term in params:
        if term == "Group Var":
            continue
        out_terms[term] = {
            "coef": float(params[term]),
            "p_value": float(pvals[term]) if term in pvals else None,
            "ci95_lo": float(conf.loc[term, 0]) if term in conf.index else None,
            "ci95_hi": float(conf.loc[term, 1]) if term in conf.index else None,
        }
    return {
        "model": model_name, "group_var": group_var, "formula": formula,
        "n_obs": int(fit.nobs), "n_groups_reviews": int(sub["review_id"].nunique()),
        "random_intercept_var": float(fit.cov_re.iloc[0, 0]),
        "residual_var": float(fit.scale),
        "fixed_effects": out_terms,
    }


# --------------------------------------------------------------------------
# 4. Classification metrics
# --------------------------------------------------------------------------
def classification_by_condition(df, model_name, n_boot=1000, seed=0):
    sub = df[df["model"] == model_name]
    rows = []
    rng = np.random.default_rng(seed)
    for cond, g in sub.groupby("condition"):
        y_true, y_pred = g["ground_truth_label"].to_numpy(), g["predicted_label"].to_numpy()
        acc = accuracy_score(y_true, y_pred)
        p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=LABELS,
                                                        average="macro", zero_division=0)
        cm = confusion_matrix(y_true, y_pred, labels=LABELS)

        review_ids = g["review_id"].to_numpy()
        clusters = np.unique(review_ids)
        pos = {rid: np.where(review_ids == rid)[0] for rid in clusters}
        accs = np.empty(n_boot)
        for i in range(n_boot):
            sampled = rng.choice(clusters, size=len(clusters), replace=True)
            idx = np.concatenate([pos[c] for c in sampled])
            accs[i] = accuracy_score(y_true[idx], y_pred[idx])
        acc_lo, acc_hi = np.percentile(accs, [2.5, 97.5])

        rows.append({
            "model": model_name, "condition": cond, "n": len(g),
            "accuracy": acc, "accuracy_ci95_lo": acc_lo, "accuracy_ci95_hi": acc_hi,
            "macro_precision": p, "macro_recall": r, "macro_f1": f1,
            "confusion_matrix": json.dumps(cm.tolist()),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# 5. Flip-rate
# --------------------------------------------------------------------------
def flip_rate(df, model_name):
    sub = df[(df["model"] == model_name) & (df["condition"].isin(NAMED_CONDITIONS))]
    piv = sub.pivot_table(index="review_id", columns="condition", values="predicted_label", aggfunc="first")
    piv = piv.dropna(subset=[c for c in NAMED_CONDITIONS if c in piv.columns])
    if piv.empty:
        return {"model": model_name, "error": "no reviews with all named conditions present"}
    n_labels = piv[NAMED_CONDITIONS].nunique(axis=1)
    flips = (n_labels > 1)

    stars_map = sub.drop_duplicates("review_id").set_index("review_id")["stars"]
    ambiguous_mask = stars_map.reindex(piv.index) == 3

    def rate(mask):
        vals = flips[mask]
        return {"n_reviews": int(mask.sum()), "n_flipped": int(vals.sum()),
                "flip_rate": float(vals.mean()) if mask.sum() else None}

    out = {
        "model": model_name,
        "overall": rate(pd.Series(True, index=piv.index)),
        "ambiguous_3star": rate(ambiguous_mask.fillna(False)),
        "clear_cut_non_3star": rate(~ambiguous_mask.fillna(False)),
    }
    avg_consistency = sub.loc[sub["n_valid_runs"] > 1, "self_consistency"]
    if len(avg_consistency):
        out["mean_within_condition_self_consistency"] = float(avg_consistency.mean())
        out["note"] = ("This model had repeated runs; compare flip_rate above against "
                       "(1 - mean_within_condition_self_consistency) as a rough noise floor.")
    else:
        out["note"] = ("This model's backend is deterministic (1 run/prompt): any flip here "
                       "is attributable to the identity cue alone, not run-to-run noise.")
    return out


# --------------------------------------------------------------------------
# 6. Figures
# --------------------------------------------------------------------------
def make_figures(df, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = Path(out_dir) / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    models = sorted(df["model"].unique())

    # bar chart: mean score by condition, one panel per model
    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 4), squeeze=False)
    for ax, model in zip(axes[0], models):
        g = df[(df["model"] == model) & (df["condition"] != "none")]
        means = g.groupby("condition")["score"].mean().reindex(NAMED_CONDITIONS)
        sems = g.groupby("condition")["score"].sem().reindex(NAMED_CONDITIONS)
        ax.bar(range(len(means)), means.values, yerr=sems.values * 1.96, capsize=4)
        ax.set_xticks(range(len(means)))
        ax.set_xticklabels(means.index, rotation=30, ha="right")
        ax.set_title(model)
        ax.set_ylabel("mean score (-1 .. 1)")
        ax.axhline(0, color="grey", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(fig_dir / "mean_score_by_condition.png", dpi=150)
    plt.close(fig)

    # confusion matrices, one figure per model x condition
    for model in models:
        cls = classification_by_condition(df, model, n_boot=1)  # n_boot=1: figure doesn't need CIs
        n = len(cls)
        fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3), squeeze=False)
        for ax, (_, row) in zip(axes[0], cls.iterrows()):
            cm = np.array(json.loads(row["confusion_matrix"]))
            im = ax.imshow(cm, cmap="Blues")
            ax.set_xticks(range(3)); ax.set_xticklabels(LABELS, rotation=45, ha="right", fontsize=7)
            ax.set_yticks(range(3)); ax.set_yticklabels(LABELS, fontsize=7)
            for i in range(3):
                for j in range(3):
                    ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8)
            ax.set_title(row["condition"], fontsize=9)
        fig.suptitle(f"{model}: confusion matrices by condition")
        fig.tight_layout()
        fig.savefig(fig_dir / f"confusion_{model}.png", dpi=150)
        plt.close(fig)


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------
def strip_boot_arrays(d):
    return {k: v for k, v in d.items() if k != "boot_diffs"}


def run_analysis(data_dir, out_dir, sesoi, n_boot, seed, make_figs=True):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df, per_model_stats = build_long_df(data_dir)
    df.to_csv(out_dir / "merged_long.csv", index=False)

    models = sorted(df["model"].unique())
    summary = {"models": models, "per_model_load_stats": per_model_stats}

    for model in models:
        mdf = df[df["model"] == model]

        # --- LMM: region x gender on the 4 named conditions ---
        named = mdf[mdf["condition"].isin(NAMED_CONDITIONS)]
        lmm = fit_mixed_model(named, "region", model)
        (out_dir / f"lmm_{model}.json").write_text(json.dumps(lmm, indent=2))

        # --- sub-region LMM, South Asian subset only ---
        sa = named[named["region"] == "south_asian"]
        sub_lmm = fit_mixed_model(sa, "sub_region", model) if sa["sub_region"].nunique() > 1 else \
            {"model": model, "error": "not enough sub_region levels"}
        (out_dir / f"subregion_lmm_{model}.json").write_text(json.dumps(sub_lmm, indent=2))

        # --- cluster bootstrap + TOST on the two primary contrasts ---
        boot_region = cluster_bootstrap_diff(named, "region", "south_asian", "western",
                                              n_boot=n_boot, seed=seed)
        boot_gender = cluster_bootstrap_diff(named, "gender", "male", "female",
                                              n_boot=n_boot, seed=seed + 1)
        pooled_sd = named["score"].std(ddof=1)
        tost_region = tost_decision(boot_region, sesoi, pooled_sd)
        tost_gender = tost_decision(boot_gender, sesoi, pooled_sd)

        p_region = lmm.get("fixed_effects", {}).get(
            [t for t in lmm.get("fixed_effects", {}) if t.startswith("C(region")][0]
            if any(t.startswith("C(region") for t in lmm.get("fixed_effects", {})) else "", {}
        ).get("p_value") if "error" not in lmm else None
        p_gender = lmm.get("fixed_effects", {}).get(
            [t for t in lmm.get("fixed_effects", {}) if t.startswith("C(gender")][0]
            if any(t.startswith("C(gender") for t in lmm.get("fixed_effects", {})) else "", {}
        ).get("p_value") if "error" not in lmm else None
        pvals_for_holm = [p for p in [p_region, p_gender] if p is not None]
        adj = list(holm_bonferroni(np.array(pvals_for_holm))) if pvals_for_holm else []
        holm_map = dict(zip(["region_main_effect", "gender_main_effect"][:len(adj)], adj))

        bootstrap_out = {
            "model": model, "sesoi_cohens_d": sesoi, "n_boot": n_boot,
            "south_asian_vs_western": {**strip_boot_arrays(boot_region), "tost": tost_region,
                                        "lmm_p_value": p_region,
                                        "holm_adjusted_p": holm_map.get("region_main_effect")},
            "male_vs_female": {**strip_boot_arrays(boot_gender), "tost": tost_gender,
                              "lmm_p_value": p_gender,
                              "holm_adjusted_p": holm_map.get("gender_main_effect")},
        }
        (out_dir / f"bootstrap_{model}.json").write_text(json.dumps(bootstrap_out, indent=2))

        # --- classification metrics ---
        cls = classification_by_condition(mdf, model, n_boot=max(200, n_boot // 2), seed=seed)
        cls.to_csv(out_dir / f"classification_{model}.csv", index=False)

        # --- flip rate ---
        fr = flip_rate(df, model)
        (out_dir / f"flip_rate_{model}.json").write_text(json.dumps(fr, indent=2))

    if make_figs:
        make_figures(df, out_dir)

    manifest = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seed": seed, "sesoi_cohens_d": sesoi, "n_boot": n_boot,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "models": models, "n_rows_total": len(df),
        "per_model_load_stats": per_model_stats,
    }
    (out_dir / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2))
    return df, models


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--sesoi", type=float, default=0.2, help="smallest effect size of interest, Cohen's d")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()

    df, models = run_analysis(args.data_dir, args.out_dir, args.sesoi, args.n_boot, args.seed,
                              make_figs=not args.no_figures)

    print(f"Models analyzed: {models}")
    print(f"Total (model, review, condition) rows: {len(df)}\n")
    for model in models:
        boot = json.loads((Path(args.out_dir) / f"bootstrap_{model}.json").read_text())
        fr = json.loads((Path(args.out_dir) / f"flip_rate_{model}.json").read_text())
        print(f"=== {model} ===")
        ra = boot["south_asian_vs_western"]
        print(f"  South Asian - Western: diff={ra['mean_diff']:.3f}  "
              f"95% CI [{ra['ci95_lo']:.3f}, {ra['ci95_hi']:.3f}]  d={ra['cohens_d']:.3f}  "
              f"LMM p={ra['lmm_p_value']}  Holm-adj p={ra['holm_adjusted_p']}")
        print(f"    TOST: {ra['tost']['verdict']}")
        ga = boot["male_vs_female"]
        print(f"  Male - Female: diff={ga['mean_diff']:.3f}  "
              f"95% CI [{ga['ci95_lo']:.3f}, {ga['ci95_hi']:.3f}]  d={ga['cohens_d']:.3f}")
        print(f"    TOST: {ga['tost']['verdict']}")
        if "error" not in fr:
            print(f"  Flip rate overall: {fr['overall']['flip_rate']:.1%}  "
                  f"(ambiguous 3-star: {fr['ambiguous_3star']['flip_rate']}, "
                  f"clear-cut: {fr['clear_cut_non_3star']['flip_rate']})")
        print()
    print(f"Full results in {args.out_dir}/ (JSON + CSV per model, figures/ for plots)")


if __name__ == "__main__":
    main()
