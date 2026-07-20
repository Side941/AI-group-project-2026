"""
src/retrievers/bm25_retriever.py
=================================
BM25 retriever for knowledge base (mhGAP chunks).
"""

from __future__ import annotations

import numpy as np
from rank_bm25 import BM25Okapi
from typing import List, Tuple

from .base import BaseRetriever


def tokenize(text: str) -> List[str]:
    """Simple whitespace tokenization + lowercase."""
    return text.lower().split() if text else []


class BM25Retriever(BaseRetriever):
    """
    BM25 retriever for knowledge base.
    """

    def __init__(self, corpus_texts: List[str], corpus_labels: List[str]):
        super().__init__(corpus_texts, corpus_labels)
        self._tokenized_corpus = []
        self._bm25 = None
        self._is_indexed = False

    def index(self) -> None:
        """Build BM25 index on KB chunks."""
        self._tokenized_corpus = [tokenize(text) for text in self.corpus_texts]
        self._bm25 = BM25Okapi(self._tokenized_corpus)
        self._is_indexed = True
        print(f"BM25 retriever indexed: {len(self.corpus_texts)} KB chunks")

    def retrieve(
        self,
        query_text: str,
        query_embedding: np.ndarray | None = None,
        k: int = 3,
    ) -> List[Tuple[str, str]]:
        """Retrieve top-k KB chunks using BM25."""
        if not self._is_indexed:
            raise RuntimeError("BM25Retriever not indexed. Call index() first.")

        tokenized_query = tokenize(query_text)
        scores = self._bm25.get_scores(tokenized_query)
        top_k_indices = np.argsort(scores)[::-1][:k]

        return [
            (self.corpus_texts[i], self.corpus_labels[i])
            for i in top_k_indices
        ]