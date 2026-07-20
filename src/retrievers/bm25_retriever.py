"""
src/retrievers/bm25_retriever.py
=================================
BM25 retriever with disorder name boosting and section expansion.
Adapted from GitHub version.
"""

from __future__ import annotations

import numpy as np
from rank_bm25 import BM25Okapi
from typing import List, Tuple, Dict, Any, Sequence, Optional

from .base import BaseRetriever
from src.section_expander import (
    build_sections_by_disorder,
    expansion_fetch_k,
    finish_search,
)


def tokenize(text: str) -> List[str]:
    """Simple whitespace tokenization + lowercase."""
    return text.lower().split() if text else []


class BM25Retriever(BaseRetriever):
    """
    BM25 retriever with disorder name boosting and section expansion.
    """

    def __init__(
        self,
        corpus_texts: List[str],
        corpus_labels: List[str],
        metadata: Optional[List[Dict]] = None,
        sections: Optional[Sequence[str]] = None,
    ):
        """
        Args:
            corpus_texts: List of text chunks
            corpus_labels: List of section labels  
            metadata: List of metadata dicts (disorder_code, disorder_name, section, etc.)
            sections: Optional section names for expansion
        """
        super().__init__(corpus_texts, corpus_labels)
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
        """Tokenize corpus and build BM25 index."""
        # Use prompt_text (clean clinical text) not text
        # This avoids metadata pollution
        self._chunks = self._build_chunks_with_metadata()
        self._tokenized_corpus = [
            tokenize(chunk.get("prompt_text") or chunk["text"])
            for chunk in self._chunks
        ]
        self._disorder_name_tokens = [
            set(tokenize(chunk.get("disorder_name", "")))
            for chunk in self._chunks
        ]
        self._bm25 = BM25Okapi(self._tokenized_corpus)
        self._is_indexed = True
        print(f"BM25 retriever indexed: {len(self.corpus_texts)} examples")

    def _score_chunks(self, query: str, fetch_k: int) -> List[Dict]:
        """Score all chunks and return top fetch_k with positive scores."""
        tokenized_query = tokenize(query)
        if not tokenized_query:
            return []

        base_scores = self._bm25.get_scores(tokenized_query)
        query_token_set = set(tokenized_query)
        adjusted_scores = np.array(base_scores, dtype=float)

        # Boost chunks whose disorder name matches query tokens
        for idx, name_tokens in enumerate(self._disorder_name_tokens):
            if not name_tokens:
                continue
            overlap = len(query_token_set & name_tokens)
            if overlap:
                adjusted_scores[idx] += 0.75 * overlap

        ranked_indices = np.argsort(adjusted_scores)[::-1]

        results = []
        seen_ids = set()
        for idx in ranked_indices:
            if adjusted_scores[idx] > 0:
                chunk = self._chunks[idx].copy()
                chunk_id = chunk.get("id")
                if chunk_id and chunk_id in seen_ids:
                    continue
                if chunk_id:
                    seen_ids.add(chunk_id)
                chunk["bm25_score"] = float(adjusted_scores[idx])
                results.append(chunk)
                if len(results) >= fetch_k:
                    break
        return results

    def retrieve(
        self,
        query_text: str,
        query_embedding: np.ndarray | None = None,
        k: int = 3,
    ) -> List[Tuple[str, str]]:
        """
        Retrieve Top-K examples with BM25 + disorder boosting.
        """
        if not self._is_indexed:
            raise RuntimeError("BM25Retriever not indexed. Call index() first.")

        scored = self._score_chunks(query_text, expansion_fetch_k(k, self.sections))
        section_map = self._section_map() if (self.sections) else {}
        
        expanded = finish_search(
            scored,
            k,
            self.sections,
            section_map,
            "bm25_score",
            expand=True,
        )

        return [(chunk.get("text", ""), chunk.get("section", "")) for chunk in expanded]