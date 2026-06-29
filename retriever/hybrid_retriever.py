from bm25_retriever import BM25Retriever
from retrieval_retriever import RetrievalRetriever


class HybridRetriever:
    def __init__(self, chunks=None, json_path=None, alpha=0.3, sections=None):
        """
        Hybrid between BM25 and dense retrieval (via RetrievalRetriever).

        Args:
            chunks/json_path: Passed through to BM25Retriever for sparse index.
            alpha: Weight for BM25 vs dense (0=dense only, 1=BM25 only).
            sections: Optional list of section names for the dense side.
        """
        self.bm25 = BM25Retriever(chunks=chunks, json_path=json_path)
        self.dense = RetrievalRetriever(sections=sections, json_path=json_path)
        self.alpha = alpha

    def search(self, query, k=5):
        if not query or not query.strip():
            return []

        # Pull broader candidates from both methods before fusion.
        bm25_results = self.bm25.search(query, k=k * 2)
        dense_results = self.dense.search(query, k=k * 2)

        bm25_max = max((r["bm25_score"] for r in bm25_results), default=1)
        dense_max = max((r["dense_score"] for r in dense_results), default=1)

        combined = {}
        for r in bm25_results:
            combined[r["id"]] = {
                "chunk": r,
                "bm25_norm": r["bm25_score"] / bm25_max if bm25_max > 0 else 0,
                "dense_norm": 0,
            }

        for r in dense_results:
            dense_norm = r["dense_score"] / dense_max if dense_max > 0 else 0
            if r["id"] in combined:
                combined[r["id"]]["dense_norm"] = dense_norm
            else:
                combined[r["id"]] = {
                    "chunk": r,
                    "bm25_norm": 0,
                    "dense_norm": dense_norm,
                }

        scored = []
        for _, data in combined.items():
            hybrid_score = (
                self.alpha * data["bm25_norm"] + (1 - self.alpha) * data["dense_norm"]
            )
            chunk = data["chunk"].copy()
            chunk["hybrid_score"] = hybrid_score
            chunk["bm25_norm"] = data["bm25_norm"]
            chunk["dense_norm"] = data["dense_norm"]
            scored.append(chunk)

        scored.sort(key=lambda x: x["hybrid_score"], reverse=True)
        return scored[:k]
