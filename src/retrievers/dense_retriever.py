"""
src/retrievers/dense_retriever.py
==================================
Dense retriever with section expansion.
Adapted from GitHub version.
"""

from __future__ import annotations

import numpy as np
from typing import List, Tuple, Dict, Any, Sequence, Optional
from sklearn.metrics.pairwise import cosine_similarity

from .base import BaseRetriever
from src.embedder import Embedder
from src.section_expander import (
    build_sections_by_disorder,
    expansion_fetch_k,
    finish_search,
)


class DenseRetriever(BaseRetriever):
    """
    Dense retriever with section expansion.
    """

    def __init__(
        self,
        corpus_texts: List[str],
        corpus_labels: List[str],
        embedder: Embedder | None = None,
        precomputed_embeddings: np.ndarray | None = None,
        metadata: Optional[List[Dict]] = None,
        sections: Optional[Sequence[str]] = None,
    ):
        super().__init__(corpus_texts, corpus_labels)
        self.embedder = embedder or Embedder()
        self._corpus_embeddings = precomputed_embeddings
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
        """Encode all corpus texts into embeddings."""
        if self._corpus_embeddings is None:
            self._corpus_embeddings = self.embedder.encode(self.corpus_texts)
        else:
            print(f"Dense retriever: using precomputed embeddings ({len(self._corpus_embeddings):,} vectors)")
        self._is_indexed = True
        self._chunks = self._build_chunks_with_metadata()
        print(f"Dense retriever indexed: {len(self.corpus_texts)} examples")

    def _score_chunks(self, query: str, query_embedding: np.ndarray, fetch_k: int) -> List[Dict]:
        """Score all chunks and return top fetch_k."""
        similarities = cosine_similarity([query_embedding], self._corpus_embeddings)[0]
        top_k_indices = np.argsort(similarities)[::-1][:fetch_k]

        results = []
        for idx in top_k_indices:
            chunk = self._chunks[idx].copy()
            chunk["dense_score"] = float(similarities[idx])
            results.append(chunk)
        return results

    def retrieve(
        self,
        query_text: str,
        query_embedding: np.ndarray | None = None,
        k: int = 3,
    ) -> List[Tuple[str, str]]:
        """Retrieve Top-K examples with section expansion."""
        if not self._is_indexed:
            raise RuntimeError("DenseRetriever not indexed. Call index() first.")

        if query_embedding is None:
            query_embedding = self.embedder.encode_single(query_text)

        fetch_k = expansion_fetch_k(k, self.sections)
        scored = self._score_chunks(query_text, query_embedding, fetch_k)
        section_map = self._section_map() if (self.sections) else {}
        
        expanded = finish_search(
            scored,
            k,
            self.sections,
            section_map,
            "dense_score",
            expand=True,
        )

        return [(chunk.get("text", ""), chunk.get("section", "")) for chunk in expanded]