"""
run_smoke_suite.py
==================
Run the no-LLM smoke experiments in order (paths → retrieval → few-shot).

    python experiments/run_smoke_suite.py
    python experiments/run_smoke_suite.py --skip-dense
    python experiments/run_smoke_suite.py --with-rag   # also needs Ollama

Dense steps download/load BioLORD on first run and can take a while.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def run_step(script: str, extra: list[str]) -> int:
    cmd = [sys.executable, str(HERE / script), *extra]
    print("\n" + "=" * 60)
    print("RUN:", " ".join(cmd))
    print("=" * 60)
    completed = subprocess.run(cmd, cwd=str(HERE.parent))
    return completed.returncode


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run pipeline smoke experiments.")
    parser.add_argument(
        "--skip-dense",
        action="store_true",
        help="Pass --skip-dense to retrieval/few-shot smokes.",
    )
    parser.add_argument(
        "--with-rag",
        action="store_true",
        help="Also run the Ollama oneshot RAG smoke (BM25 + qwen3:0.6b).",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    dense_flags = ["--skip-dense"] if args.skip_dense else []

    steps: list[tuple[str, list[str]]] = [
        ("smoke_paths.py", []),
        ("smoke_retrieval.py", dense_flags),
        ("smoke_fewshot.py", dense_flags),
    ]
    if args.with_rag:
        steps.append(("smoke_rag_oneshot.py", ["--retriever", "bm25"]))

    failures = 0
    for script, flags in steps:
        code = run_step(script, flags)
        if code != 0:
            failures += 1
            print(f"\nSTEP FAILED ({code}): {script}")

    print("\n" + "=" * 60)
    if failures:
        print(f"DONE — {failures} step(s) failed.")
        return 1
    print("DONE — all requested smoke steps passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
