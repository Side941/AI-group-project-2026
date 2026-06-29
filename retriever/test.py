# test.py
# Tests all retrievers against the ICD-11 knowledge base.
# Run from anywhere: python retriever/test.py

from paths import setup_import_paths

setup_import_paths()

from bm25_retriever import BM25Retriever
from hybrid_retriever import HybridRetriever
from retrieval_retriever import RetrievalRetriever
from retrieval import initialise_retrieval, get_icd11_context
from components.config import CHUNKS_PATH

queries = [
    "I feel so depressed and hopeless",
    "I feel empty inside",
    "I don't see the point in living anymore",
]

# ── BM25 ──────────────────────────────────────────────────────────────────────
print("=" * 70)
print("BM25 RETRIEVER")
print("=" * 70)
bm25 = BM25Retriever(json_path=CHUNKS_PATH)
for q in queries:
    print(f"\nQuery: '{q}'")
    results = bm25.search(q, k=3)
    for i, r in enumerate(results):
        print(f"  {i+1}. Score: {r['bm25_score']:.4f} | {r.get('prompt_text', r['text'])[:80]}...")

# ── Retrieval.py context generation (ChromaDB) ────────────────────────────────
print("\n" + "=" * 70)
print("RETRIEVAL.PY CONTEXT (BioLORD-2023 + ChromaDB)")
print("=" * 70)
initialise_retrieval()
for q in queries:
    print(f"\nQuery: '{q}'")
    context = get_icd11_context(q, n_results=3)
    print(context[:300])
    print("...")

# ── Hybrid ────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("HYBRID RETRIEVER (BM25 + retrieval.py dense, alpha=0.3)")
print("=" * 70)
hybrid = HybridRetriever(json_path=CHUNKS_PATH, alpha=0.3)
for q in queries:
    print(f"\nQuery: '{q}'")
    results = hybrid.search(q, k=3)
    for i, r in enumerate(results):
        print(f"  {i+1}. Hybrid: {r['hybrid_score']:.4f} | BM25: {r['bm25_norm']:.4f} | Dense: {r['dense_norm']:.4f}")
        print(f"     {r.get('prompt_text', r['text'])[:80]}...")

# ── Boundary with Normality filter ────────────────────────────────────────────
print("\n" + "=" * 70)
print("RETRIEVAL.PY — Boundary with Normality filter")
print("=" * 70)
for q in queries[:1]:
    print(f"\nQuery: '{q}'")
    context = get_icd11_context(q, n_results=3, section_filter="Boundary with Normality")
    print(context[:300])
    print("...")

# ── RetrievalRetriever with section filters ───────────────────────────────────
print("\n" + "=" * 70)
print("RETRIEVAL RETRIEVER (Boundary with Normality + Essential Features)")
print("=" * 70)
retrieval_ret = RetrievalRetriever(
    sections=["Boundary with Normality", "Essential Features"],
    json_path=CHUNKS_PATH,
)
for q in queries:
    print(f"\nQuery: '{q}'")
    results = retrieval_ret.search(q, k=3)
    for i, r in enumerate(results):
        print(f"  {i+1}. Dense: {r['dense_score']:.4f} | Section: {r['section']}")
        print(f"     {r.get('prompt_text', r['text'])[:80]}...")
