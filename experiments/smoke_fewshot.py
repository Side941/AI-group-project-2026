"""
smoke_fewshot.py
================
Smoke-test the few-shot example store (built by rebuild_all_artifacts.py).

This store is not yet wired into RAG prompts; this script only checks that
BM25 / dense / hybrid retrieval over the examples works.

    python experiments/smoke_fewshot.py
    python experiments/smoke_fewshot.py --k 3
    python experiments/smoke_fewshot.py --skip-dense
"""

from __future__ import annotations

import argparse

from _common import SAMPLE_QUERIES, bootstrap, print_hits

bootstrap()

from components.config import (  # noqa: E402
    FEWSHOT_CHROMA_PATH,
    FEWSHOT_MULTICLASS_EXAMPLES_PATH,
)
from fewshot_retrievers import (  # noqa: E402
    FewShotBM25Retriever,
    build_fewshot_retrievers,
    initialise_fewshot_dense_retrieval,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Smoke-test few-shot retrievers.")
    parser.add_argument(
        "--query",
        choices=sorted(SAMPLE_QUERIES),
        default="depression",
    )
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--skip-dense", action="store_true")
    parser.add_argument("--alpha", type=float, default=0.3)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    query = SAMPLE_QUERIES[args.query]
    examples_path = FEWSHOT_MULTICLASS_EXAMPLES_PATH
    head = "multiclass"

    print("=== Few-shot retrieval smoke ===")
    print(f"head={head}  query={args.query}  k={args.k}")
    print(f"examples: {examples_path}")
    print(f"chroma:   {FEWSHOT_CHROMA_PATH}\n")

    if not examples_path.exists():
        print("FAIL — few-shot examples JSON missing.")
        print("Build with: python src/builders/rebuild_all_artifacts.py")
        return 1

    bm25 = FewShotBM25Retriever(head=head)
    print("--- Few-shot BM25 ---")
    print_hits(bm25.search(query, k=args.k), score_key="bm25_score")

    if args.skip_dense:
        print("\nSkipped dense/hybrid (--skip-dense).")
        print("PASS — few-shot BM25 OK.")
        return 0

    if not FEWSHOT_CHROMA_PATH.exists():
        print(f"\nFAIL — few-shot Chroma missing at {FEWSHOT_CHROMA_PATH}")
        return 1

    initialise_fewshot_dense_retrieval()
    retrievers = build_fewshot_retrievers(head=head, alpha=args.alpha)

    print("\n--- Few-shot Dense ---")
    print_hits(retrievers["dense"].search(query, k=args.k), score_key="dense_score")

    print(f"\n--- Few-shot Hybrid (alpha={args.alpha}) ---")
    print_hits(retrievers["hybrid"].search(query, k=args.k), score_key="hybrid_score")

    print("\nPASS — few-shot retrieval OK.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
