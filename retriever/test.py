# test.py
# Tests all three retrievers against the real ICD-11 CDDR knowledge base.
# Run from the project root: python retriever/test.py

import sys
from pathlib import Path

# Add project root to path so components package can be found
sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

# Add retriever folder to path so bm25_retriever and retrieval can be found
sys.path.insert(0, str(Path(__file__).parent.resolve()))

from bm25_retriever import BM25Retriever
from hybrid_retriever import HybridRetriever
from retrieval import initialise_retrieval, get_icd11_context
import components.ingestion as ingestion

queries = [
    "I feel so depressed and hopeless",
    "I feel empty inside",
    "I don't see the point in living anymore",
]

# ── BM25 ──────────────────────────────────────────────────────────────────────
print("=" * 70)
print("BM25 RETRIEVER")
print("=" * 70)
bm25 = BM25Retriever()
for q in queries:
    print(f"\nQuery: '{q}'")
    results = bm25.search(q, k=3)
    for i, r in enumerate(results):
        # Use prompt_text for clean display — text contains embed_text with metadata headers
        print(f"  {i+1}. Score: {r['bm25_score']:.4f} | {r.get('prompt_text', r['text'])[:80]}...")

# ── Dense (Ramy's ChromaDB retriever) ─────────────────────────────────────────
print("\n" + "=" * 70)
print("DENSE RETRIEVER (BioLORD-2023 + ChromaDB)")
print("=" * 70)
initialise_retrieval()
for q in queries:
    print(f"\nQuery: '{q}'")
    context = get_icd11_context(q, n_results=3)
    print(context[:300])
    print("...")

# ── Hybrid ────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("HYBRID RETRIEVER (BM25 + BioLORD, alpha=0.3)")
print("=" * 70)
hybrid = HybridRetriever(alpha=0.3)
for q in queries:
    print(f"\nQuery: '{q}'")
    results = hybrid.search(q, k=3)
    for i, r in enumerate(results):
        print(f"  {i+1}. Hybrid: {r['hybrid_score']:.4f} | BM25: {r['bm25_norm']:.4f} | Dense: {r['dense_norm']:.4f}")
        # Use prompt_text for clean display
        print(f"     {r.get('prompt_text', r['text'])[:80]}...")

# ── Boundary with Normality filter ────────────────────────────────────────────
print("\n" + "=" * 70)
print("DENSE RETRIEVER — Boundary with Normality filter")
print("=" * 70)
for q in queries[:1]:
    print(f"\nQuery: '{q}'")
    context = get_icd11_context(q, n_results=3, section_filter="Boundary with Normality")
    print(context[:300])
    print("...")