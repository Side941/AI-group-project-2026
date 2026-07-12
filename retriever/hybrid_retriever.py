from bm25_retriever import BM25Retriever
from retrieval_retriever import RetrievalRetriever
from utils import load_chunks
from components.config import CHUNKS_PATH
from section_expander import build_sections_by_disorder, expansion_fetch_k, finish_search

# Standard RRF smoothing constant (Cormack et al., 2009).
RRF_K = 60


class HybridRetriever:
    """
    Weighted Reciprocal Rank Fusion (RRF) of BM25 and dense retrieval.

    Fuses ranks rather than raw scores, so BM25 and dense results can be
    combined without score-scale calibration. Chunks ranked highly by both
    retrievers rise to the top.

    When *sections* is provided, every top-k disorder is expanded to include
    one chunk per requested section via the shared expand_sections() utility,
    so the LLM receives a complete clinical picture regardless of which section
    happened to win the fusion score.
    """

    def __init__(self, chunks=None, json_path=None, alpha=0.3, sections=None, rrf_k=RRF_K):
        """
        Args:
            chunks / json_path: Forwarded to BM25Retriever for sparse index.
            alpha: BM25 weight in [0, 1]. 0 = dense only, 1 = BM25 only.
            sections: Optional section names for post-fusion expansion.
            rrf_k: RRF smoothing constant (default 60).
        """
        self.sections = list(sections) if sections else []
        self.alpha = alpha
        self.rrf_k = rrf_k

        self.bm25 = BM25Retriever(chunks=chunks, json_path=json_path,
                                  sections=self.sections)
        self.dense = RetrievalRetriever(sections=self.sections,
                                        json_path=json_path)

        # Single section map for fusion expansion — avoids duplicating the
        # lookup that BM25 and dense would each build independently.
        self._sections_by_disorder = (
            build_sections_by_disorder(
                chunks if chunks is not None else load_chunks(json_path or CHUNKS_PATH),
                self.sections,
            )
            if self.sections
            else {}
        )

    def _fuse(
        self,
        bm25_results: list[dict],
        dense_results: list[dict],
    ) -> list[dict]:
        """Fuse BM25 and dense candidates with weighted Reciprocal Rank Fusion."""
        bm25_ranks = {r["id"]: i + 1 for i, r in enumerate(bm25_results)}
        dense_ranks = {r["id"]: i + 1 for i, r in enumerate(dense_results)}

        chunk_by_id = {r["id"]: r for r in bm25_results}
        chunk_by_id.update({r["id"]: r for r in dense_results})

        scored = []
        for chunk_id, chunk_src in chunk_by_id.items():
            bm25_rank = bm25_ranks.get(chunk_id)
            dense_rank = dense_ranks.get(chunk_id)

            hybrid_score = 0.0
            if bm25_rank is not None:
                hybrid_score += self.alpha / (self.rrf_k + bm25_rank)
            if dense_rank is not None:
                hybrid_score += (1 - self.alpha) / (self.rrf_k + dense_rank)

            chunk = chunk_src.copy()
            chunk["hybrid_score"] = hybrid_score
            chunk["score"] = hybrid_score
            chunk["bm25_rank"] = bm25_rank
            chunk["dense_rank"] = dense_rank
            scored.append(chunk)

        scored.sort(key=lambda x: x["hybrid_score"], reverse=True)
        return scored

    def search(self, query: str, k: int = 5) -> list[dict]:
        if not query or not query.strip():
            return []

        fetch_k = expansion_fetch_k(k, self.sections)
        bm25_results = self.bm25.search(query, k=fetch_k, expand=False)
        dense_results = self.dense.search(query, k=fetch_k, expand=False)

        scored = self._fuse(bm25_results, dense_results)
        return finish_search(
            scored,
            k,
            self.sections,
            self._sections_by_disorder,
            "hybrid_score",
        )
