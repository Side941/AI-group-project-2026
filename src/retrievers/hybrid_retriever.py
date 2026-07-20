"""
src/retrievers/hybrid_retriever.py
===================================
Hybrid retriever (BM25 + Dense) for knowledge base with RRF fusion.
"""

from __future__ import annotations

import numpy as np
from typing import List, Tuple

from .base import BaseRetriever
from .bm25_retriever import BM25Retriever
from .dense_retriever import DenseRetriever
from src.embedder import Embedder

RRF_K = 60  # RRF smoothing constant


class HybridRetriever(BaseRetriever):
    """
    Hybrid retriever with RRF fusion of BM25 and dense for KB.
    """

    def __init__(
        self,
        corpus_texts: List[str],
        corpus_labels: List[str],
        embedder: Embedder | None = None,
        alpha: float = 0.5,
    ):
        super().__init__(corpus_texts, corpus_labels)
        self.alpha = alpha
        self.embedder = embedder or Embedder()
        self._bm25 = BM25Retriever(corpus_texts, corpus_labels)
        self._dense = DenseRetriever(corpus_texts, corpus_labels, self.embedder)
        self._is_indexed = False

    def index(self) -> None:
        """Index both BM25 and dense."""
        self._bm25.index()
        self._dense.index()
        self._is_indexed = True
        print(f"Hybrid retriever indexed: {len(self.corpus_texts)} KB chunks (alpha={self.alpha})")

    def retrieve(
        self,
        query_text: str,
        query_embedding: np.ndarray | None = None,
        k: int = 3,
    ) -> List[Tuple[str, str]]:
        """Retrieve top-k using RRF fusion."""
        if not self._is_indexed:
            raise RuntimeError("HybridRetriever not indexed. Call index() first.")

        if query_embedding is None:
            query_embedding = self.embedder.encode_single(query_text)

        # Get results from both retrievers (fetch more for RRF)
        bm25_results = self._bm25.retrieve(query_text, query_embedding, k=k*2)
        dense_results = self._dense.retrieve(query_text, query_embedding, k=k*2)

        # RRF Fusion
        bm25_ranks = {doc: i + 1 for i, (doc, _) in enumerate(bm25_results)}
        dense_ranks = {doc: i + 1 for i, (doc, _) in enumerate(dense_results)}

        all_docs = set(bm25_ranks.keys()) | set(dense_ranks.keys())

        scores = {}
        for doc in all_docs:
            score = 0.0
            if doc in bm25_ranks:
                score += self.alpha / (RRF_K + bm25_ranks[doc])
            if doc in dense_ranks:
                score += (1 - self.alpha) / (RRF_K + dense_ranks[doc])
            scores[doc] = score

        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]

        # Get labels
        doc_to_label = {doc: label for doc, label in zip(self.corpus_texts, self.corpus_labels)}
        return [(doc, doc_to_label.get(doc, '')) for doc, _ in sorted_docs]