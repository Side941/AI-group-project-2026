# bm25_retriever.py

import numpy as np
from rank_bm25 import BM25Okapi
from components.config import CHUNKS_PATH
from utils import load_chunks, tokenize, filter_chunks_by_sections
from section_expander import (
    build_sections_by_disorder,
    expansion_fetch_k,
    finish_search,
)


class BM25Retriever:
    def __init__(self, chunks=None, json_path=None, sections=None):
        all_chunks = chunks if chunks is not None else load_chunks(json_path or CHUNKS_PATH)

        self.sections: list[str] = list(sections) if sections else []
        self.chunks = (
            filter_chunks_by_sections(all_chunks, self.sections)
            if self.sections
            else all_chunks
        )

        # Tokenize using prompt_text (clean clinical text) not text
        # (embed_text with metadata headers). embed_text contains repeated
        # generic words like "Source", "Domain", "Disorder" which pollute BM25
        # keyword matching and cause false positives.
        self.tokenized_chunks = [
            tokenize(chunk.get("prompt_text") or chunk["text"])
            for chunk in self.chunks
        ]

        self.bm25 = BM25Okapi(self.tokenized_chunks)

        # Built lazily on first expanded search — not needed when expand=False.
        self._sections_by_disorder: dict | None = None

        print(f"BM25 retriever ready: {len(self.chunks)} chunks indexed")

    def _section_map(self) -> dict:
        if self._sections_by_disorder is None:
            self._sections_by_disorder = build_sections_by_disorder(
                self.chunks, self.sections
            )
        return self._sections_by_disorder

    def _score_chunks(self, query: str, fetch_k: int) -> list[dict]:
        """Score all chunks against *query* and return the top *fetch_k* with
        a positive BM25 score, sorted descending."""
        tokenized_query = tokenize(query)
        if not tokenized_query:
            return []

        scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:fetch_k]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                chunk = self.chunks[idx].copy()
                chunk["bm25_score"] = float(scores[idx])
                results.append(chunk)
        return results

    def search(self, query: str, k: int = 5, *, expand: bool = True) -> list[dict]:
        if not query or not query.strip():
            return []

        scored = self._score_chunks(query, expansion_fetch_k(k, self.sections))
        section_map = self._section_map() if (self.sections and expand) else {}
        return finish_search(
            scored,
            k,
            self.sections,
            section_map,
            "bm25_score",
            expand=expand,
        )
