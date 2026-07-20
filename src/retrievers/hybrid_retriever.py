"""
src/retrievers/hybrid_retriever.py
===================================
Hybrid retriever with RRF fusion and section expansion.
Adapted from GitHub version.
"""

from __future__ import annotations

import numpy as np
from typing import List, Tuple, Dict, Any, Sequence, Optional, TYPE_CHECKING

from .base import BaseRetriever
from .bm25_retriever import BM25Retriever
from .dense_retriever import DenseRetriever
from src.embedder import Embedder
from src.section_expander import (
    build_sections_by_disorder,
    expansion_fetch_k,
    finish_search,
)

if TYPE_CHECKING:
    from src.vector_store import VectorStore

RRF_K = 60  # RRF smoothing constant


class HybridRetriever(BaseRetriever):
    """
    Hybrid retriever with RRF fusion of BM25 and dense.
    """

    def __init__(
        self,
        corpus_texts: List[str],
        corpus_labels: List[str],
        vector_store: "VectorStore",
        embedder: Embedder | None = None,
        alpha: float = 0.5,
        metadata: Optional[List[Dict]] = None,
        sections: Optional[Sequence[str]] = None,
    ):
        super().__init__(corpus_texts, corpus_labels)
        self.alpha = alpha
        self.embedder = embedder or Embedder()
        self.vector_store = vector_store
        self.metadata = metadata or [{}] * len(corpus_texts)
        self.sections = list(sections) if sections else []
        self._is_indexed = False
        self._sections_by_disorder: dict | None = None

    def _build_chunks_with_metadata(self) -> List[Dict]:
        """Build chunk dicts with metadata for section expansion."""
        chunks = []
        for i, (text, label, meta) in enumerate(zip(self.corpus_texts, self.corpus_labels, self.metadata)):
            chunk = {
                'id': meta.get('id', f'chunk_{i}'),
                'text': text,
                'prompt_text': text,
                'section': meta.get('section', ''),
                'label': label,
                'disorder_code': meta.get('disorder_code', ''),
                'disorder_name': meta.get('disorder_name', ''),
                'source': meta.get('source', ''),
            }
            chunks.append(chunk)
        return chunks

    def _section_map(self) -> dict:
        if self._sections_by_disorder is None:
            chunks = self._build_chunks_with_metadata()
            self._sections_by_disorder = build_sections_by_disorder(chunks, self.sections)
        return self._sections_by_disorder

    def index(self) -> None:
        """Index BM25 and dense."""
        self._chunks = self._build_chunks_with_metadata()
        
        # Build BM25
        self._bm25 = BM25Retriever(
            self.corpus_texts, 
            self.corpus_labels, 
            self.metadata,
            self.sections
        )
        self._bm25.index()
        
        # Build dense
        self._dense = DenseRetriever(
            self.corpus_texts,
            self.corpus_labels,
            self.embedder,
            None,
            self.metadata,
            self.sections
        )
        self._dense.index()
        
        if self.vector_store.count() == 0:
            raise RuntimeError(
                "HybridRetriever's vector_store is empty. It must be built/loaded "
                "(via create_or_load + add) before constructing HybridRetriever."
            )
        self._is_indexed = True
        print(f"Hybrid retriever indexed: {len(self.corpus_texts)} examples (alpha={self.alpha})")

    def _fuse(self, bm25_results: List[Dict], dense_results: List[Dict]) -> List[Dict]:
        """Fuse BM25 and dense candidates with RRF."""
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
                hybrid_score += self.alpha / (RRF_K + bm25_rank)
            if dense_rank is not None:
                hybrid_score += (1 - self.alpha) / (RRF_K + dense_rank)

            chunk = chunk_src.copy()
            chunk["hybrid_score"] = hybrid_score
            scored.append(chunk)

        scored.sort(key=lambda x: x["hybrid_score"], reverse=True)
        return scored

    def retrieve(
        self,
        query_text: str,
        query_embedding: np.ndarray | None = None,
        k: int = 3,
    ) -> List[Tuple[str, str]]:
        """Retrieve Top-K using RRF fusion."""
        if not self._is_indexed:
            raise RuntimeError("HybridRetriever not indexed. Call index() first.")

        if query_embedding is None:
            query_embedding = self.embedder.encode_single(query_text)

        fetch_k = expansion_fetch_k(k, self.sections)
        
        # Get results from both retrievers
        bm25_results = self._bm25._score_chunks(query_text, fetch_k)
        dense_results = self._dense._score_chunks(query_text, query_embedding, fetch_k)
        
        # Fuse with RRF
        fused = self._fuse(bm25_results, dense_results)
        
        # Apply section expansion
        section_map = self._section_map() if (self.sections) else {}
        expanded = finish_search(
            fused,
            k,
            self.sections,
            section_map,
            "hybrid_score",
            expand=True,
        )

        return [(chunk.get("text", ""), chunk.get("section", "")) for chunk in expanded]