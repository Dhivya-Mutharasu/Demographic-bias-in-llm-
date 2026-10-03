"""Run: python tests/test_run_models.py"""
import csv, json, os, subprocess, sys, tempfile, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import run_models as rm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ok = True
def check(name, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(("PASS" if cond else "FAIL"), "-", name, detail)

PROMPTS = [
    {"prompt_id": f"P{i}", "review_id": f"R{i}", "condition": "none",
     "prompt_text": f"Review number {i}: it was fine I guess."} for i in range(1, 31)
]

# ---------------------------------------------------------------- harness / mock
d = tempfile.mkdtemp()
out = os.path.join(d, "out.csv")
backend = rm.MockBackend()
stats = rm.run(PROMPTS, backend, out, runs=1, rate_limiter=rm.RateLimiter(0))
check("mock: all 30 calls made first time", stats["calls_made"] == 30, stats)
rows = list(csv.DictReader(open(out, encoding="utf-8")))
check("output has exactly 30 rows", len(rows) == 30, f"(got {len(rows)})")
check("every OUT_FIELDS column present", set(rows[0]) == set(rm.OUT_FIELDS))
check("predicted_label is always one of the 3 labels", all(r["predicted_label"] in rm.LABELS for r in rows))
probs_ok = all(abs(sum(float(r[f"prob_{l}"]) for l in rm.LABELS) - 1.0) < 1e-6 for r in rows)
check("probabilities sum to 1 for every row", probs_ok)

# resume: re-run same backend/out-path, nothing new should be written
backend2 = rm.MockBackend()
stats2 = rm.run(PROMPTS, backend2, out, runs=1, rate_limiter=rm.RateLimiter(0))
check("resume: 0 new calls, 30 skipped", stats2["calls_made"] == 0 and stats2["skipped_already_done"] == 30, stats2)
rows2 = list(csv.DictReader(open(out, encoding="utf-8")))
check("resume: file still has exactly 30 rows (no duplicates)", len(rows2) == 30, f"(got {len(rows2)})")

# partial crash simulation: process 10, "crash" (stop), then finish the rest
d2 = tempfile.mkdtemp()
out2 = os.path.join(d2, "out.csv")
rm.run(PROMPTS[:10], rm.MockBackend(), out2, runs=1, rate_limiter=rm.RateLimiter(0))
stats3 = rm.run(PROMPTS, rm.MockBackend(), out2, runs=1, rate_limiter=rm.RateLimiter(0))
check("partial run then full run: 20 new, 10 skipped", stats3["calls_made"] == 20 and stats3["skipped_already_done"] == 10, stats3)
rows3 = list(csv.DictReader(open(out2, encoding="utf-8")))
check("no duplicate prompt_ids after resume", len({r["prompt_id"] for r in rows3}) == 30 and len(rows3) == 30)

# max_calls / daily-cap guard: stop partway, resume continues, no duplicates
d3 = tempfile.mkdtemp()
out3 = os.path.join(d3, "out.csv")
s1 = rm.run(PROMPTS, rm.MockBackend(), out3, runs=1, rate_limiter=rm.RateLimiter(0), max_calls=12)
check("daily cap: stops at exactly 12 calls", s1["calls_made"] == 12 and s1["stopped_early_daily_cap"], s1)
s2 = rm.run(PROMPTS, rm.MockBackend(), out3, runs=1, rate_limiter=rm.RateLimiter(0), max_calls=100)
check("daily cap: next invocation finishes the remaining 18", s2["calls_made"] == 18 and s2["skipped_already_done"] == 12, s2)
rows4 = list(csv.DictReader(open(out3, encoding="utf-8")))
check("daily cap: 30 total rows, no duplicates, no gaps", len(rows4) == 30 and len({r["prompt_id"] for r in rows4}) == 30)

# non-deterministic backend: multiple runs per prompt, each run_index unique
d4 = tempfile.mkdtemp()
out4 = os.path.join(d4, "out.csv")
class FakeNonDetBackend(rm.MockBackend):
    deterministic = False
rm.run(PROMPTS[:5], FakeNonDetBackend(), out4, runs=3, rate_limiter=rm.RateLimiter(0))
rows5 = list(csv.DictReader(open(out4, encoding="utf-8")))
check("non-deterministic backend: 5 prompts x 3 runs = 15 rows", len(rows5) == 15, f"(got {len(rows5)})")
run_idx_per_prompt = {}
for r in rows5:
    run_idx_per_prompt.setdefault(r["prompt_id"], set()).add(r["run_index"])
check("each prompt has run_index 1,2,3 exactly once", all(v == {"1", "2", "3"} for v in run_idx_per_prompt.values()))

# errors are retried on next invocation (not silently marked "done")
d5 = tempfile.mkdtemp()
out5 = os.path.join(d5, "out.csv")
flaky = rm.MockBackend(fail_rate=1.0)  # everything fails first pass
s = rm.run(PROMPTS[:5], flaky, out5, runs=1, rate_limiter=rm.RateLimiter(0))
check("flaky backend: errors recorded, not silently dropped", s["errors"] == 5 and s["calls_made"] == 5, s)
ok_backend = rm.MockBackend(fail_rate=0.0)
s2 = rm.run(PROMPTS[:5], ok_backend, out5, runs=1, rate_limiter=rm.RateLimiter(0))
check("errored rows are retried (not treated as done)", s2["calls_made"] == 5 and s2["errors"] == 0, s2)
rows6 = list(csv.DictReader(open(out5, encoding="utf-8")))
succeeded = [r for r in rows6 if r["prompt_id"] == "P1" and not r["error"]]
check("after retry, the successful row exists", len(succeeded) >= 1)

# ---------------------------------------------------------------- rate limiter
sleeps = []
rl = rm.RateLimiter(rpm=30, sleep_fn=lambda s: sleeps.append(s), clock=iter([0, 0, 1, 1, 3, 3]).__next__)
rl.wait(); rl.wait(); rl.wait()
check("rate limiter sleeps ~ (60/rpm - elapsed) when calls are too fast",
      len(sleeps) >= 1 and all(s >= 0 for s in sleeps), sleeps)

# ---------------------------------------------------------------- HF scoring math (no torch needed)
pred, probs = rm.aggregate_label_logprobs({
    "negative": [-5.0, -5.0], "neutral": [-3.0, -3.0], "positive": [-0.1, -0.1],
})
check("aggregate_label_logprobs: highest-logprob label wins", pred == "positive", probs)
check("aggregate_label_logprobs: probs sum to 1", abs(sum(probs.values()) - 1.0) < 1e-9)

# length normalization: a longer but per-token-equal-quality label shouldn't be penalized
pred2, probs2 = rm.aggregate_label_logprobs({
    "negative": [-1.0], "neutral": [-1.0, -1.0, -1.0], "positive": [-1.0, -1.0],
})
check("aggregate_label_logprobs: length-normalized (equal per-token logprob -> tie)",
      abs(probs2["negative"] - probs2["neutral"]) < 1e-9 and abs(probs2["neutral"] - probs2["positive"]) < 1e-9,
      probs2)

try:
    rm.aggregate_label_logprobs({"negative": [], "neutral": [-1.0], "positive": [-1.0]})
    check("aggregate_label_logprobs: raises on empty label", False)
except ValueError:
    check("aggregate_label_logprobs: raises on empty label", True)

# ---------------------------------------------------------------- Gemini backend (fake HTTP, no network)
class FakeResp:
    def __init__(self, status_code, body=None, text=""):
        self.status_code = status_code
        self._body = body
        self.text = text
    def json(self):
        return self._body

calls = {"n": 0}
def fake_post_ok(url, body, timeout=60):
    calls["n"] += 1
    return FakeResp(200, {"candidates": [{"content": {"parts": [{"text": "Positive"}]}}]})

gb = rm.GeminiBackend("gemini-test", api_key="fake", http_post=fake_post_ok)
r = gb.score("some prompt")
check("gemini: parses 'Positive' text into label", r.predicted_label == "positive" and not r.error)

def fake_post_retry_then_ok(url, body, timeout=60):
    calls["n"] += 1
    if calls["n"] < 3:
        return FakeResp(429)
    return FakeResp(200, {"candidates": [{"content": {"parts": [{"text": "negative."}]}}]})

calls["n"] = 0
gb2 = rm.GeminiBackend("gemini-test", api_key="fake", http_post=fake_post_retry_then_ok)
t0 = time.monotonic()
r2 = gb2.score("some prompt")
check("gemini: retries past 429 and eventually succeeds", r2.predicted_label == "negative" and calls["n"] == 3)

def fake_post_always_429(url, body, timeout=60):
    calls["n"] += 1
    return FakeResp(429)
calls["n"] = 0
gb3 = rm.GeminiBackend("gemini-test", api_key="fake", http_post=fake_post_always_429, max_retries=3)
r3 = gb3.score("some prompt")
check("gemini: gives up after max_retries and reports error (not a crash)",
      r3.error is not None and "retries_exhausted" in r3.error and calls["n"] == 3)

def fake_post_garbage(url, body, timeout=60):
    return FakeResp(200, {"unexpected": "shape"})
gb4 = rm.GeminiBackend("gemini-test", api_key="fake", http_post=fake_post_garbage)
r4 = gb4.score("some prompt")
check("gemini: malformed response body -> error, not a crash", r4.error is not None and "parse_error" in r4.error)

def fake_post_unparseable(url, body, timeout=60):
    return FakeResp(200, {"candidates": [{"content": {"parts": [{"text": "I cannot help with that."}]}}]})
gb5 = rm.GeminiBackend("gemini-test", api_key="fake", http_post=fake_post_unparseable)
r5 = gb5.score("some prompt")
check("gemini: unparseable label text -> flagged, raw_text kept for review",
      r5.error == "unparseable_label" and "cannot help" in r5.raw_text.lower())

# gemini requires an API key via env var -- CLI should fail fast, not silently proceed
import argparse
env_backup = os.environ.pop("GEMINI_API_KEY", None)
try:
    args = argparse.Namespace(backend="gemini", api_key_env="GEMINI_API_KEY", model="gemini-test")
    try:
        rm.build_backend(args)
        check("gemini: missing API key raises SystemExit", False)
    except SystemExit:
        check("gemini: missing API key raises SystemExit", True)
finally:
    if env_backup:
        os.environ["GEMINI_API_KEY"] = env_backup

# hf backend without torch/transformers installed should fail clearly, not cryptically
try:
    import torch  # noqa: F401
    print("SKIP - torch is installed here; not testing the missing-dependency message")
except ModuleNotFoundError:
    try:
        rm.HFLocalBackend("some/model")
        check("hf backend without torch: raises a clear error", False)
    except ModuleNotFoundError as e:
        check("hf backend without torch: raises a clear ModuleNotFoundError", "torch" in str(e).lower())

# ---------------------------------------------------------------- CLI end-to-end (subprocess)
d6 = tempfile.mkdtemp()
pfile = os.path.join(d6, "prompts.csv")
with open(pfile, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, ["prompt_id", "review_id", "condition", "prompt_text"])
    w.writeheader()
    w.writerows(PROMPTS[:8])
r = subprocess.run([sys.executable, f"{ROOT}/run_models.py", "--prompts", pfile,
                    "--backend", "mock", "--out-dir", d6], capture_output=True, text=True)
check("CLI: mock backend runs end-to-end without error", r.returncode == 0, r.stderr[-300:] if r.returncode else "")
check("CLI: manifest file written", os.path.exists(os.path.join(d6, "run_manifest_mock.json")))
man = json.load(open(os.path.join(d6, "run_manifest_mock.json")))
check("CLI: manifest records backend/model/counts", man["backend"] == "mock" and
      man["last_invocation_stats"]["calls_made"] == 8, man)

print("\nALL TESTS PASSED" if ok else "\nSOME TESTS FAILED")
sys.exit(0 if ok else 1)
