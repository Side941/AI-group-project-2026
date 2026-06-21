# test.py

from bm25_retriever import BM25Retriever
from dense_retriever import DenseRetriever
from hybrid_retriever import HybridRetriever

queries = [
    "I feel so depressed and hopeless",
    "I feel empty inside"
]

print("=" * 70)
print("BM25 RETRIEVER")
print("=" * 70)
bm25 = BM25Retriever()
for q in queries:
    print(f"\nQuery: '{q}'")
    results = bm25.search(q, k=7)
    for i, r in enumerate(results):
        print(f"  {i+1}. Score: {r['bm25_score']:.4f} | {r['text'][:80]}...")

print("\n" + "=" * 70)
print("DENSE RETRIEVER")
print("=" * 70)
dense = DenseRetriever()
for q in queries:
    print(f"\nQuery: '{q}'")
    results = dense.search(q, k=7)
    for i, r in enumerate(results):
        print(f"  {i+1}. Score: {r['dense_score']:.4f} | {r['text'][:80]}...")

print("\n" + "=" * 70)
print("HYBRID RETRIEVER (alpha=0.5)")
print("=" * 70)
hybrid = HybridRetriever(alpha=0.5)
for q in queries:
    print(f"\nQuery: '{q}'")
    results = hybrid.search(q, k=7)
    for i, r in enumerate(results):
        print(f"  {i+1}. Hybrid: {r['hybrid_score']:.4f} | BM25: {r['bm25_norm']:.4f} | Dense: {r['dense_norm']:.4f}")
        print(f"     {r['text'][:80]}...")