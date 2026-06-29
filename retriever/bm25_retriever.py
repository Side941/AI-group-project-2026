# bm25_retriever.py

from collections import defaultdict

import numpy as np
from rank_bm25 import BM25Okapi
from components.config import CHUNKS_PATH
from utils import load_chunks, tokenize
from section_expander import top_k_disorder_keys, expand_sections


class BM25Retriever:
    def __init__(self, chunks=None, json_path=None, sections=None):
        if chunks is None:
            all_chunks = load_chunks(json_path or CHUNKS_PATH)
        else:
            all_chunks = chunks

        self.sections: list[str] = list(sections) if sections else []
        section_allowlist = set(self.sections) if self.sections else None

        if section_allowlist is not None:
            self.chunks = [
                c for c in all_chunks
                if c.get("section", "") in section_allowlist
            ]
        else:
            self.chunks = all_chunks

        # Tokenize using prompt_text (clean clinical text) not text
        # (embed_text with metadata headers). embed_text contains repeated
        # generic words like "Source", "Domain", "Disorder" which pollute BM25
        # keyword matching and cause false positives.
        self.tokenized_chunks = [
            tokenize(chunk.get("prompt_text") or chunk["text"])
            for chunk in self.chunks
        ]

        # Build BM25 index.
        self.bm25 = BM25Okapi(self.tokenized_chunks)

        # Build disorder -> section -> chunk lookup for post-retrieval expansion.
        # Populated only when sections are requested; keyed identically to
        # RetrievalRetriever._sections_by_disorder so expand_sections() can be
        # called with either retriever's map interchangeably.
        self._sections_by_disorder: dict[tuple[str, str], dict[str, dict]] = defaultdict(dict)
        if self.sections:
            for chunk in all_chunks:
                if chunk.get("section", "") not in section_allowlist:
                    continue
                key = (chunk.get("disorder_code", ""), chunk.get("disorder_name", ""))
                self._sections_by_disorder[key][chunk.get("section", "")] = chunk

        print(f"BM25 retriever ready: {len(self.chunks)} chunks indexed")

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

        # Over-fetch so expansion has enough candidate seeds to find k
        # distinct disorders before injecting siblings.
        fetch_k = k * 4 if self.sections else k
        scored = self._score_chunks(query, fetch_k)

        if not self.sections or not expand:
            return scored[:k] if not self.sections else scored

        top_keys = top_k_disorder_keys(scored, k)
        return expand_sections(
            scored_chunks=scored,
            top_k_keys=top_keys,
            sections_by_disorder=self._sections_by_disorder,
            score_field="bm25_score",
            sections=set(self.sections),
        )
