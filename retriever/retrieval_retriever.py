from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Sequence

from retrieval import search_icd11
from utils import load_chunks
from components.config import CHUNKS_PATH
from section_expander import top_k_disorder_keys, expand_sections


class RetrievalRetriever:
    """
    Dense retriever backed by retrieval.py / ChromaDB.

    When *sections* is provided, each section is queried separately and results
    are merged by chunk id (highest dense_score wins). A post-retrieval
    expansion step then ensures every top-k disorder is represented by all
    requested sections before returning.
    """

    def __init__(
        self,
        sections: Sequence[str] | None = None,
        json_path: str | None = None,
        expand_same_disorder: bool = True,
    ):
        """
        Args:
            sections: Section names to filter on, e.g.
                ["Boundary with Normality", "Essential Features"].
                If None or empty, no section filter is applied and no
                expansion is performed.
            json_path: Path to the knowledge-base JSON used for expansion.
                Defaults to CHUNKS_PATH.
            expand_same_disorder: When True, build the disorder lookup so
                expand_sections() can inject sibling sections after retrieval.
        """
        self.sections: list[str] = list(sections) if sections else []
        self._section_allowlist = set(self.sections) if self.sections else None
        self.expand_same_disorder = expand_same_disorder

        # Disorder -> section -> chunk lookup, shared with HybridRetriever.
        self._sections_by_disorder: dict[tuple[str, str], dict[str, dict]] = defaultdict(dict)
        if expand_same_disorder:
            for chunk in load_chunks(json_path or CHUNKS_PATH):
                section = chunk.get("section", "")
                if self._section_allowlist is not None and section not in self._section_allowlist:
                    continue
                key = (chunk.get("disorder_code", ""), chunk.get("disorder_name", ""))
                self._sections_by_disorder[key][section] = chunk

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _iter_section_results(self, query: str, k: int) -> Iterable[dict]:
        """Yield raw ChromaDB rows, optionally filtered per section."""
        if not self.sections:
            yield from search_icd11(query_text=query, n_results=k)
            return
        for section in self.sections:
            yield from search_icd11(query_text=query, n_results=k,
                                    section_filter=section)

    def _deduplicate(self, rows: Iterable[dict]) -> list[dict]:
        """Keep the highest-scoring chunk per id, sorted descending."""
        best: dict[str, dict] = {}
        for row in rows:
            cid = row["id"]
            if cid not in best or row["dense_score"] > best[cid]["dense_score"]:
                best[cid] = row
        result = list(best.values())
        result.sort(key=lambda r: r["dense_score"], reverse=True)
        return result

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(self, query: str, k: int = 5, *, expand: bool = True) -> list[dict]:
        if not query or not query.strip():
            return []

        fetch_k = k * 4 if self.sections else k
        scored = self._deduplicate(self._iter_section_results(query, fetch_k))

        if not self.sections or not expand:
            return scored[:k] if not self.sections else scored

        top_keys = top_k_disorder_keys(scored, k)
        return expand_sections(
            scored_chunks=scored,
            top_k_keys=top_keys,
            sections_by_disorder=self._sections_by_disorder,
            score_field="dense_score",
            sections=set(self.sections),
        )
