#!/usr/bin/env python3
"""
run_models.py  --  Step 3 of the demographic-bias project.

Sends every prompt in prompts.csv (from build_prompts.py) to one model and
appends the result to data/raw_responses_<model_slug>.csv. Run it once per
model (Kaggle/Colab for open-weight models, your own machine for Gemini).

Two free backends:

  hf      An open-weight instruction model, run locally on a free Kaggle/Colab
          GPU. Scores the three labels by teacher-forced log-likelihood
          (like lm-eval-harness "loglikelihood" scoring) instead of free-form
          generation: greedy and fully deterministic given fixed weights, so
          it needs only ONE run per prompt (no sampling noise to average out).
          Needs: pip install torch transformers accelerate

  gemini  Google's Gemini API, free tier (generativelanguage.googleapis.com).
          Free-tier requests are NOT guaranteed deterministic even at
          temperature 0, and the free tier does not reliably expose
          log-probabilities, so this backend asks for a one-word label and
          repeats each prompt --runs times (default 3) so the repeated-run
          noise can be modeled explicitly in Step 4, instead of being thrown
          away or treated as extra independent data (Reviewer 2's complaint
          about the original 156-vs-468 mismatch).
          Needs: a free API key from https://aistudio.google.com/apikey in
          the GEMINI_API_KEY environment variable.
          Free-tier daily caps are usually a few hundred to ~1500 requests
          (Google changes these without notice - check your quota at
          https://aistudio.google.com before running). At 2,500 prompts this
          backend usually needs SEVERAL DAYS. That's fine: just re-run the
          same command tomorrow. It skips everything already in the output
          file and stops cleanly at the daily-cap guess.

  mock    No model at all. Deterministic pseudo-random labels from a hash of
          the prompt. Use this to test the harness, your file paths and your
          Step 4 analysis code before spending real quota.

Output columns (data/raw_responses_<model_slug>.csv):
  prompt_id, review_id, condition, run_index, model_name, model_version,
  backend, predicted_label, prob_negative, prob_neutral, prob_positive,
  raw_text, latency_ms, error, queried_at_utc

Resuming: safe to Ctrl-C and re-run the exact same command; already-completed
(prompt_id, run_index) pairs are skipped, never re-sent, never duplicated.

Examples
  # dry run / test the pipeline, no model or API key needed
  python run_models.py --prompts data/prompts.csv --backend mock --out-dir data

  # open-weight model on a Kaggle/Colab GPU
  python run_models.py --prompts data/prompts.csv --backend hf \
      --model Qwen/Qwen2.5-7B-Instruct --out-dir data

  # Gemini free tier (run this same command again each day until it finishes)
  export GEMINI_API_KEY=...
  python run_models.py --prompts data/prompts.csv --backend gemini \
      --model gemini-2.0-flash-lite --out-dir data --runs 3 --rpm 25 --rpd 1400
"""
import argparse
import csv
import hashlib
import json
import re
import sys
import time
from pathlib import Path

LABELS = ["negative", "neutral", "positive"]
OUT_FIELDS = ["prompt_id", "review_id", "condition", "run_index", "model_name", "model_version",
              "backend", "predicted_label", "prob_negative", "prob_neutral", "prob_positive",
              "raw_text", "latency_ms", "error", "queried_at_utc"]


def now_iso():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def slugify(name):
    return re.sub(r"[^a-zA-Z0-9]+", "-", name).strip("-").lower()


def read_csv(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


# --------------------------------------------------------------------------
# Rate limiter: simple token bucket, one token per (60/rpm) seconds.
# --------------------------------------------------------------------------
class RateLimiter:
    def __init__(self, rpm, sleep_fn=time.sleep, clock=time.monotonic):
        self.min_interval = 60.0 / rpm if rpm else 0.0
        self._last = None
        self._sleep = sleep_fn
        self._clock = clock

    def wait(self):
        if self.min_interval <= 0:
            return
        now = self._clock()
        if self._last is not None:
            elapsed = now - self._last
            remaining = self.min_interval - elapsed
            if remaining > 0:
                self._sleep(remaining)
        self._last = self._clock()


# --------------------------------------------------------------------------
# Backends. Each implements score(prompt_text) -> ScoreResult.
# --------------------------------------------------------------------------
class ScoreResult:
    __slots__ = ("predicted_label", "probs", "raw_text", "error")

    def __init__(self, predicted_label=None, probs=None, raw_text="", error=None):
        self.predicted_label = predicted_label
        self.probs = probs or {}  # {"negative": p, "neutral": p, "positive": p}
        self.raw_text = raw_text
        self.error = error


class MockBackend:
    """No model. Deterministic pseudo-label from a hash of the prompt, with a
    small, injectable 'bias' so tests can simulate a condition effect."""

    name = "mock"
    version = "v1"
    deterministic = True  # needs only 1 run

    def __init__(self, bias_fn=None, fail_rate=0.0, seed=0):
        self.bias_fn = bias_fn  # optional: prompt_row -> extra logit per label
        self.fail_rate = fail_rate
        self._calls = 0

    def score(self, prompt_text, meta=None):
        self._calls += 1
        h = int(hashlib.md5(prompt_text.encode()).hexdigest(), 16)
        if self.fail_rate and (h % 1000) / 1000.0 < self.fail_rate:
            return ScoreResult(error="mock_transient_error")
        base = [((h >> (i * 8)) % 100) / 100.0 for i in range(3)]
        if self.bias_fn and meta is not None:
            extra = self.bias_fn(meta)
            base = [b + e for b, e in zip(base, extra)]
        m = max(base)
        exps = [pow(2.718281828, b - m) for b in base]
        s = sum(exps)
        probs = {lab: e / s for lab, e in zip(LABELS, exps)}
        pred = max(probs, key=probs.get)
        return ScoreResult(predicted_label=pred, probs=probs, raw_text=pred)


def aggregate_label_logprobs(per_label_token_logprobs):
    """Pure-math core of the HF backend's scoring, kept dependency-free so it
    can be unit-tested without torch/transformers/a GPU.

    per_label_token_logprobs: {"negative": [lp1, lp2, ...], "neutral": [...], "positive": [...]}
    (one float per token of that label's continuation, from teacher forcing).

    Returns (predicted_label, probs_dict). Length-normalizes each label's
    total log-likelihood by its token count before comparing, so a
    single-token label isn't unfairly favored over a multi-token one.
    """
    import math

    normed = {}
    for lab, lps in per_label_token_logprobs.items():
        if not lps:
            raise ValueError(f"no token log-probs for label '{lab}'")
        normed[lab] = sum(lps) / len(lps)
    m = max(normed.values())
    exps = {lab: math.exp(v - m) for lab, v in normed.items()}
    s = sum(exps.values())
    probs = {lab: e / s for lab, e in exps.items()}
    pred = max(probs, key=probs.get)
    return pred, probs


class HFLocalBackend:
    """Open-weight instruction model, scored by teacher-forced log-likelihood
    of each label string. Deterministic given fixed weights -> 1 run needed.

    NOT exercised by the automated tests in this repo (no GPU / model
    download in this environment) -- aggregate_label_logprobs() above, which
    is the part that turns model output into a label, IS unit-tested.
    Sanity-check this class yourself on Kaggle/Colab with a handful of
    obviously-positive/negative reviews before running the full 2,500.
    """

    deterministic = True

    def __init__(self, model_id, device=None, dtype="auto", label_strings=None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.name = model_id
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=dtype, device_map=device or "auto"
        )
        self.model.eval()
        self.version = getattr(self.model.config, "_name_or_path", model_id)
        self.label_strings = label_strings or {"negative": " negative", "neutral": " neutral",
                                                "positive": " positive"}
        self._torch = torch

    def score(self, prompt_text, meta=None):
        torch = self._torch
        prompt_ids = self.tok(prompt_text, return_tensors="pt").input_ids.to(self.model.device)
        per_label = {}
        with torch.no_grad():
            for lab, s in self.label_strings.items():
                label_ids = self.tok(s, add_special_tokens=False, return_tensors="pt").input_ids
                label_ids = label_ids.to(self.model.device)
                full = torch.cat([prompt_ids, label_ids], dim=1)
                out = self.model(full)
                logits = out.logits[0]  # (seq_len, vocab)
                start = prompt_ids.shape[1] - 1  # predict label_ids[0] from last prompt token
                logprobs = torch.log_softmax(logits[start:start + label_ids.shape[1]], dim=-1)
                token_lps = [logprobs[i, label_ids[0, i]].item() for i in range(label_ids.shape[1])]
                per_label[lab] = token_lps
        pred, probs = aggregate_label_logprobs(per_label)
        return ScoreResult(predicted_label=pred, probs=probs, raw_text=pred)


def parse_label_from_text(text):
    t = text.strip().lower()
    t = re.sub(r"[^a-z]", "", t)
    for lab in LABELS:
        if lab in t:
            return lab
    return None


class GeminiBackend:
    """Free-tier Gemini via REST. `http_post` is injectable for testing so no
    real network call is needed to exercise the retry/parse/rate-limit logic."""

    deterministic = False

    def __init__(self, model_id, api_key, http_post=None, max_retries=5):
        self.name = model_id
        self.version = model_id
        self.api_key = api_key
        self.max_retries = max_retries
        self._post = http_post or self._real_post

    @staticmethod
    def _real_post(url, json_body, timeout=60):
        import requests
        return requests.post(url, params={}, json=json_body, timeout=timeout)

    def score(self, prompt_text, meta=None):
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.name}:generateContent"
        body = {
            "contents": [{"parts": [{"text": prompt_text}]}],
            "generationConfig": {"temperature": 0, "maxOutputTokens": 5},
        }
        delay = 1.0
        last_err = None
        for attempt in range(self.max_retries):
            try:
                resp = self._post(url + f"?key={self.api_key}", body)
            except Exception as e:  # noqa: BLE001 - network errors of any kind
                last_err = f"request_exception:{e}"
                time.sleep(delay)
                delay *= 2
                continue
            status = getattr(resp, "status_code", 200)
            if status == 429 or status >= 500:
                last_err = f"http_{status}"
                time.sleep(delay)
                delay *= 2
                continue
            if status != 200:
                return ScoreResult(error=f"http_{status}:{getattr(resp, 'text', '')[:200]}")
            try:
                data = resp.json()
                text = data["candidates"][0]["content"]["parts"][0]["text"]
            except Exception as e:  # noqa: BLE001 - malformed/blocked response
                return ScoreResult(error=f"parse_error:{e}", raw_text=str(getattr(resp, "text", ""))[:300])
            label = parse_label_from_text(text)
            if label is None:
                return ScoreResult(error="unparseable_label", raw_text=text)
            return ScoreResult(predicted_label=label, raw_text=text)
        return ScoreResult(error=f"retries_exhausted:{last_err}")


# --------------------------------------------------------------------------
# Harness: resumable, incremental, rate-limited.
# --------------------------------------------------------------------------
def load_done(out_path):
    done = set()
    if Path(out_path).exists():
        for r in read_csv(out_path):
            if not r.get("error"):
                done.add((r["prompt_id"], int(r["run_index"])))
    return done


def run(prompts, backend, out_path, runs, rate_limiter, max_calls=None, retry_errors=True):
    done = load_done(out_path)
    if not retry_errors and Path(out_path).exists():
        for r in read_csv(out_path):
            done.add((r["prompt_id"], int(r["run_index"])))

    is_new = not Path(out_path).exists()
    f = open(out_path, "a", encoding="utf-8", newline="")
    writer = csv.DictWriter(f, fieldnames=OUT_FIELDS)
    if is_new:
        writer.writeheader()

    n_calls = 0
    n_skipped = 0
    n_errors = 0
    stopped_early = False
    for row in prompts:
        n_run = 1 if getattr(backend, "deterministic", False) else runs
        for run_idx in range(1, n_run + 1):
            if (row["prompt_id"], run_idx) in done:
                n_skipped += 1
                continue
            if max_calls is not None and n_calls >= max_calls:
                stopped_early = True
                break
            rate_limiter.wait()
            t0 = time.monotonic()
            result = backend.score(row["prompt_text"], meta=row)
            latency_ms = int((time.monotonic() - t0) * 1000)
            n_calls += 1
            if result.error:
                n_errors += 1
            out_row = {
                "prompt_id": row["prompt_id"], "review_id": row["review_id"],
                "condition": row["condition"], "run_index": run_idx,
                "model_name": backend.name, "model_version": getattr(backend, "version", ""),
                "backend": backend.__class__.__name__, "predicted_label": result.predicted_label or "",
                "prob_negative": result.probs.get("negative", ""), "prob_neutral": result.probs.get("neutral", ""),
                "prob_positive": result.probs.get("positive", ""), "raw_text": result.raw_text,
                "latency_ms": latency_ms, "error": result.error or "", "queried_at_utc": now_iso(),
            }
            writer.writerow(out_row)
            f.flush()
        if stopped_early:
            break
    f.close()
    return {"calls_made": n_calls, "skipped_already_done": n_skipped, "errors": n_errors,
            "stopped_early_daily_cap": stopped_early}


def build_backend(args):
    if args.backend == "mock":
        return MockBackend(fail_rate=args.mock_fail_rate)
    if args.backend == "hf":
        return HFLocalBackend(args.model, device=args.device)
    if args.backend == "gemini":
        import os
        key = os.environ.get(args.api_key_env)
        if not key:
            raise SystemExit(f"Set {args.api_key_env} to your free Gemini API key "
                             f"(get one at https://aistudio.google.com/apikey)")
        return GeminiBackend(args.model, key)
    raise SystemExit(f"Unknown backend: {args.backend}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prompts", default="data/prompts.csv")
    ap.add_argument("--out-dir", default="data")
    ap.add_argument("--backend", choices=["mock", "hf", "gemini"], required=True)
    ap.add_argument("--model", default="mock", help="HF model id, or Gemini model name")
    ap.add_argument("--device", default=None, help="hf backend only, e.g. 'cuda:0'; default auto")
    ap.add_argument("--api-key-env", default="GEMINI_API_KEY")
    ap.add_argument("--runs", type=int, default=3, help="repeats per prompt for non-deterministic backends")
    ap.add_argument("--rpm", type=float, default=0, help="max requests per minute (0 = no limit)")
    ap.add_argument("--rpd", type=int, default=None,
                    help="stop after this many NEW calls this invocation (re-run tomorrow for more)")
    ap.add_argument("--mock-fail-rate", type=float, default=0.0, help="mock backend: fraction of calls that error")
    ap.add_argument("--limit", type=int, default=None, help="only process the first N prompt rows (testing)")
    args = ap.parse_args()

    prompts = read_csv(args.prompts)
    if args.limit:
        prompts = prompts[: args.limit]
    if not prompts:
        raise SystemExit(f"No prompts found in {args.prompts}")

    backend = build_backend(args)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    slug = slugify(getattr(backend, "name", args.model))
    out_path = out_dir / f"raw_responses_{slug}.csv"

    rate_limiter = RateLimiter(args.rpm)
    stats = run(prompts, backend, out_path, args.runs, rate_limiter, max_calls=args.rpd)

    manifest_path = out_dir / f"run_manifest_{slug}.json"
    manifest = {
        "updated_utc": now_iso(), "backend": args.backend, "model": args.model,
        "model_version": getattr(backend, "version", ""), "runs_requested": args.runs,
        "deterministic": getattr(backend, "deterministic", False),
        "rpm": args.rpm, "rpd_this_invocation": args.rpd, "prompts_file": str(args.prompts),
        "n_prompt_rows": len(prompts), "last_invocation_stats": stats,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))

    print(f"Backend: {args.backend}  Model: {args.model}")
    print(f"Calls made this run: {stats['calls_made']}   "
          f"Already done (skipped): {stats['skipped_already_done']}   Errors: {stats['errors']}")
    if stats["stopped_early_daily_cap"]:
        print(f"Stopped at --rpd={args.rpd} for this invocation. "
              f"Re-run the SAME command to continue (it will skip everything already done).")
    total_needed = len(prompts) * (1 if getattr(backend, "deterministic", False) else args.runs)
    total_done = len(load_done(out_path))
    print(f"Progress: {total_done}/{total_needed} (prompt, run) pairs completed -> {out_path}")
    if total_done < total_needed:
        print("Not finished yet -- run this exact command again to continue.")


if __name__ == "__main__":
    main()
