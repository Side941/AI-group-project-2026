#!/usr/bin/env python3
"""
Qwen3 CPU latency benchmark for the RAG mental-health project.

Measures per-post inference time across model sizes and thinking on/off,
using Ollama's built-in timing fields (more accurate than wall-clock alone).
No retriever or knowledge base required — runs on a few sample texts.

Usage:
    1. ollama serve            # in another terminal
    2. ollama pull qwen3:0.6b  # (and 1.7b / 4b if benchmarking those)
    3. python3 qwen3_benchmark.py

Only depends on the Python standard library.
"""

import json
import sys
import time
import csv
import random
import re
import statistics
import urllib.request
import urllib.error
from datetime import datetime

# ----------------------------- CONFIG -----------------------------
OLLAMA_HOST = "http://localhost:11434"

# Start with just 0.6b. Add the others once 0.6b works end-to-end.
MODELS = ["qwen3:0.6b", "qwen3:1.7b", "qwen3:4b"]

THINKING_MODES = [False, True]   # the on/off contrast is the whole point
REPEATS = 3                      # runs per (model, mode, post) -> mean +/- std
WARMUP = True                    # one discarded call per model to load weights into RAM
NUM_PREDICT_CAP = 2048           # safety valve so a runaway thinking chain can't hang the box
SEED = 42                        # fixed for reproducibility
TIMEOUT_S = 1800                 # 4B + thinking on CPU can be slow; be generous

# How to toggle thinking. "param" uses Ollama's think flag (newer Ollama).
# If your Ollama is older and ignores it, switch to "prompt" (Qwen3 /think //no_think soft switch).
THINK_CONTROL = "param"          # "param" | "prompt"

OUT_CSV = f"qwen3_benchmark_{datetime.now():%Y%m%d_%H%M}.csv"

# ---- Dataset: real posts give honest prompt lengths for timing ----
CSV_PATH = "Depression_Severity_Levels_Dataset.csv"  # dataset path (relative to where you run this)
TEXT_COLUMN = "text"     # column holding the post
LABEL_COLUMN = "label"   # severity label; carried into the output, not needed for timing
SAMPLE_SIZE = 12         # how many posts to time. Keep SMALL — this is a latency probe, not the real eval.
SAMPLE_SEED = 42         # reproducible random sample (random = unbiased estimate of mean per-post time)

# Wrap each post so prompt length approximates the real classification task.
# (The real RAG prompt will be longer still — it also carries retrieved CDDR chunks.)
PROMPT_TEMPLATE = (
    "You are a clinical text classifier. Read the post and reply with exactly one "
    "label from [minimum, mild, moderate, severe], then one short sentence of "
    "justification.\n\nPost: {post}\n\nAnswer:"
)
# ------------------------------------------------------------------


def _http_post(path, payload):
    req = urllib.request.Request(
        OLLAMA_HOST + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=TIMEOUT_S) as r:
        return json.loads(r.read().decode())


def check_ollama():
    """Return list of installed model tags, or exit with a helpful message."""
    try:
        req = urllib.request.Request(OLLAMA_HOST + "/api/tags")
        with urllib.request.urlopen(req, timeout=10) as r:
            data = json.loads(r.read().decode())
        return [m["name"] for m in data.get("models", [])]
    except Exception as e:
        print(f"ERROR: cannot reach Ollama at {OLLAMA_HOST}  ({e})")
        print("Fix: run `ollama serve` in another terminal, then re-run this script.")
        sys.exit(1)


def ensure_models(installed):
    missing = [m for m in MODELS
               if not any(name == m or name.startswith(m) for name in installed)]
    if missing:
        print("These models are not pulled yet. Pull them first:")
        for m in missing:
            print(f"  ollama pull {m}")
        sys.exit(1)


def _split_thinking(response_text, thinking_field):
    """Return (thinking, answer) regardless of how Ollama reports it.

    With the `think` API param, Ollama puts the reasoning in a separate
    `thinking` field. With the /no_think prompt switch (older path) it may
    arrive inline as <think>...</think>. Handle both.
    """
    if thinking_field:
        return thinking_field.strip(), response_text.strip()
    m = re.search(r"<think>(.*?)</think>", response_text, re.DOTALL)
    if m:
        return m.group(1).strip(), response_text[m.end():].strip()
    return "", response_text.strip()


def run_once(model, post, think):
    """One generation. Returns timing dict parsed from Ollama's response."""
    prompt = PROMPT_TEMPLATE.format(post=post)
    payload = {
        "model": model,
        "stream": False,
        "options": {"seed": SEED, "num_predict": NUM_PREDICT_CAP},
    }
    if THINK_CONTROL == "param":
        payload["prompt"] = prompt
        payload["think"] = think
    else:  # prompt-based soft switch fallback
        switch = " /think" if think else " /no_think"
        payload["prompt"] = prompt + switch

    t0 = time.perf_counter()
    resp = _http_post("/api/generate", payload)
    wall = time.perf_counter() - t0

    response_text = resp.get("response", "")
    thinking_field = resp.get("thinking", "") or ""
    thinking_text, answer_text = _split_thinking(response_text, thinking_field)
    gen_eval_s = resp.get("eval_duration", 0) / 1e9
    gen_tokens = resp.get("eval_count", 0)
    return {
        "wall_s": round(wall, 3),
        "total_s": round(resp.get("total_duration", 0) / 1e9, 3),
        "load_s": round(resp.get("load_duration", 0) / 1e9, 3),
        "prompt_tokens": resp.get("prompt_eval_count", 0),
        "prompt_eval_s": round(resp.get("prompt_eval_duration", 0) / 1e9, 3),
        "gen_tokens": gen_tokens,
        "gen_eval_s": round(gen_eval_s, 3),
        "tok_per_s": round(gen_tokens / gen_eval_s, 2) if gen_eval_s else 0.0,
        "hit_cap": gen_tokens >= NUM_PREDICT_CAP,
        "thinking_seen": bool(thinking_text),
        "thinking_trace": thinking_text,  # full reasoning trace (empty when thinking is off)
        "model_output": answer_text,      # the model's actual answer text
    }


def load_posts():
    """Load posts from the dataset, skip empty text, return a small random sample.

    Uses the stdlib csv module so quoted fields with commas/newlines parse
    correctly (this dataset has both). Random sampling gives an unbiased
    estimate of the average per-post time across the real length distribution.
    """
    csv.field_size_limit(10_000_000)  # some posts are very long; raise the cell cap
    try:
        with open(CSV_PATH, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if TEXT_COLUMN not in reader.fieldnames:
                print(f"ERROR: column '{TEXT_COLUMN}' not in {CSV_PATH} "
                      f"(found: {reader.fieldnames})")
                sys.exit(1)
            pool = [
                {"text": r[TEXT_COLUMN].strip(),
                 "label": (r.get(LABEL_COLUMN) or "").strip()}
                for r in reader
                if r.get(TEXT_COLUMN) and r[TEXT_COLUMN].strip()
            ]
    except FileNotFoundError:
        print(f"ERROR: dataset not found at '{CSV_PATH}'. "
              f"Put the CSV next to this script or edit CSV_PATH.")
        sys.exit(1)

    if not pool:
        print(f"ERROR: no usable rows found in {CSV_PATH}.")
        sys.exit(1)

    random.seed(SAMPLE_SEED)
    sample = random.sample(pool, min(SAMPLE_SIZE, len(pool)))
    print(f"Loaded {len(pool)} posts; timing on a sample of {len(sample)}.\n")
    return sample


def main():
    installed = check_ollama()
    ensure_models(installed)
    posts = load_posts()

    rows = []
    for model in MODELS:
        if WARMUP:
            print(f"[warmup] loading {model} into memory ...")
            try:
                run_once(model, "hello", False)
            except Exception as e:
                print(f"  warmup failed for {model}: {e}")
        for think in THINKING_MODES:
            for pi, item in enumerate(posts):
                post = item["text"]
                for rep in range(REPEATS):
                    try:
                        m = run_once(model, post, think)
                    except urllib.error.URLError as e:
                        print(f"  FAILED ({model}, think={think}, post{pi}): {e}")
                        continue
                    rows.append({"model": model, "think": think,
                                 "post_idx": pi, "label": item["label"],
                                 "rep": rep, **m})
                    cap = "  [HIT CAP]" if m["hit_cap"] else ""
                    print(f"  {model:12} think={str(think):5} post{pi} rep{rep}: "
                          f"{m['total_s']:7.2f}s  in={m['prompt_tokens']:4d}tok "
                          f"out={m['gen_tokens']:4d}tok ({m['tok_per_s']:5.1f} tok/s){cap}")

    if not rows:
        print("No successful runs — nothing to summarise.")
        return

    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # ---- summary: mean total seconds per (model, think) + thinking multiplier ----
    print("\n" + "=" * 72)
    print(f"{'model':12} {'think':6} {'mean_s':>8} {'std_s':>7} "
          f"{'out_tok':>8} {'tok/s':>7}")
    print("-" * 72)
    agg = {}
    for model in MODELS:
        for think in THINKING_MODES:
            sub = [r for r in rows if r["model"] == model and r["think"] == think]
            if not sub:
                continue
            totals = [r["total_s"] for r in sub]
            mean_s = statistics.mean(totals)
            std_s = statistics.pstdev(totals) if len(totals) > 1 else 0.0
            mean_out = statistics.mean([r["gen_tokens"] for r in sub])
            mean_tps = statistics.mean([r["tok_per_s"] for r in sub])
            agg[(model, think)] = mean_s
            print(f"{model:12} {str(think):6} {mean_s:8.2f} {std_s:7.2f} "
                  f"{mean_out:8.0f} {mean_tps:7.1f}")

    print("-" * 72)
    for model in MODELS:
        off = agg.get((model, False))
        on = agg.get((model, True))
        if off and on and off > 0:
            print(f"{model:12} thinking ON is {on / off:4.1f}x slower than OFF "
                  f"({off:.1f}s -> {on:.1f}s per post)")
    print("=" * 72)
    print(f"\nPer-run detail written to: {OUT_CSV}")
    print("NOTE: these timings are specific to THIS machine's CPU. Run on the "
          "target/deployment hardware for numbers the team can plan around.")


if __name__ == "__main__":
    main()