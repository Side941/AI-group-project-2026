"""
smoke_retrieval.py
==================
Smoke-test ICD-11 BM25 / dense / hybrid retrieval (no LLM).

Matches notebook policy: mood-disorder prefixes + Essential Features /
Boundary with Normality, then compares expand vs flat for BM25.

    python experiments/smoke_retrieval.py
    python experiments/smoke_retrieval.py --query suicidal --k 4
    python experiments/smoke_retrieval.py --skip-dense   # BM25 only (fast)
"""

from __future__ import annotations

import argparse

from _common import SAMPLE_QUERIES, bootstrap, load_mood_chunks, print_hits

bootstrap()

from bm25_retriever import BM25Retriever  # noqa: E402
from components.config import CHROMA_PATH, RETRIEVAL_SECTIONS  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Smoke-test ICD-11 retrievers.")
    parser.add_argument(
        "--query",
        choices=sorted(SAMPLE_QUERIES),
        default="depression",
        help="Which canned social-media query to use.",
    )
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument(
        "--skip-dense",
        action="store_true",
        help="Skip dense/hybrid (avoids loading BioLORD + Chroma).",
    )
    parser.add_argument("--alpha", type=float, default=0.3, help="Hybrid BM25 weight.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    query = SAMPLE_QUERIES[args.query]
    sections = list(RETRIEVAL_SECTIONS)

    print("=== ICD-11 retrieval smoke ===")
    print(f"query key : {args.query}")
    print(f"query text: {query}")
    print(f"k={args.k}  sections={sections}\n")

    mood_chunks = load_mood_chunks()
    print(f"Mood-filtered chunk pool: {len(mood_chunks)}")

    bm25 = BM25Retriever(chunks=mood_chunks, sections=sections)
    print("\n--- BM25 (expand=True) ---")
    print_hits(bm25.search(query, k=args.k, expand=True), score_key="bm25_score")

    print("\n--- BM25 (expand=False / flat) ---")
    print_hits(bm25.search(query, k=args.k, expand=False), score_key="bm25_score")

    if args.skip_dense:
        print("\nSkipped dense/hybrid (--skip-dense).")
        print("PASS — BM25 retrieval OK.")
        return 0

    if not CHROMA_PATH.exists():
        print(f"\nFAIL — ChromaDB missing at {CHROMA_PATH}")
        return 1

    from dense_retriever import DenseRetriever, initialise_retrieval
    from hybrid_retriever import HybridRetriever

    initialise_retrieval(chroma_path=str(CHROMA_PATH))
    dense = DenseRetriever(sections=sections)
    hybrid = HybridRetriever(chunks=mood_chunks, sections=sections, alpha=args.alpha)

    print("\n--- Dense ---")
    print_hits(dense.search(query, k=args.k, expand=True), score_key="dense_score")

    print(f"\n--- Hybrid (alpha={args.alpha}) ---")
    print_hits(hybrid.search(query, k=args.k, expand=True), score_key="hybrid_score")

    print("\nPASS — BM25 / dense / hybrid retrieval OK.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
