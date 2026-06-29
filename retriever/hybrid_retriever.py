from bm25_retriever import BM25Retriever
from retrieval_retriever import RetrievalRetriever
from utils import load_chunks
from components.config import CHUNKS_PATH
from section_expander import build_sections_by_disorder, expansion_fetch_k, finish_search


class HybridRetriever:
    """
    Reciprocal-normalisation fusion of BM25 and dense retrieval.

    When *sections* is provided, every top-k disorder is expanded to include
    one chunk per requested section via the shared expand_sections() utility,
    so the LLM receives a complete clinical picture regardless of which section
    happened to win the fusion score.
    """

    def __init__(self, chunks=None, json_path=None, alpha=0.3, sections=None):
        """
        Args:
            chunks / json_path: Forwarded to BM25Retriever for sparse index.
            alpha: BM25 weight in [0, 1]. 0 = dense only, 1 = BM25 only.
            sections: Optional section names for post-fusion expansion.
        """
        self.sections = list(sections) if sections else []
        self.alpha = alpha

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
        """Normalise each retriever's scores then combine with alpha weighting."""
        bm25_max  = max((r["bm25_score"]  for r in bm25_results),  default=1) or 1
        dense_max = max((r["dense_score"] for r in dense_results), default=1) or 1

        combined: dict[str, dict] = {}
        for r in bm25_results:
            combined[r["id"]] = {
                "chunk":      r,
                "bm25_norm":  r["bm25_score"] / bm25_max,
                "dense_norm": 0.0,
            }
        for r in dense_results:
            dense_norm = r["dense_score"] / dense_max
            if r["id"] in combined:
                combined[r["id"]]["dense_norm"] = dense_norm
            else:
                combined[r["id"]] = {
                    "chunk":      r,
                    "bm25_norm":  0.0,
                    "dense_norm": dense_norm,
                }

        scored = []
        for data in combined.values():
            hybrid_score = (
                self.alpha       * data["bm25_norm"]
                + (1 - self.alpha) * data["dense_norm"]
            )
            chunk = data["chunk"].copy()
            chunk["hybrid_score"] = hybrid_score
            chunk["score"]        = hybrid_score
            chunk["bm25_norm"]    = data["bm25_norm"]
            chunk["dense_norm"]   = data["dense_norm"]
            scored.append(chunk)

        scored.sort(key=lambda x: x["hybrid_score"], reverse=True)
        return scored

    def search(self, query: str, k: int = 5) -> list[dict]:
        if not query or not query.strip():
            return []

        fetch_k = expansion_fetch_k(k, self.sections)
        bm25_results  = self.bm25.search(query,  k=fetch_k, expand=False)
        dense_results = self.dense.search(query, k=fetch_k, expand=False)

        scored = self._fuse(bm25_results, dense_results)
        return finish_search(
            scored,
            k,
            self.sections,
            self._sections_by_disorder,
            "hybrid_score",
        )
