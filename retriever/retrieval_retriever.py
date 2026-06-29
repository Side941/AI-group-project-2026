from __future__ import annotations

from typing import Iterable, Sequence

from retrieval import search_icd11
from utils import load_chunks
from components.config import CHUNKS_PATH
from section_expander import (
    build_sections_by_disorder,
    expansion_fetch_k,
    finish_search,
)


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
    ):
        """
        Args:
            sections: Section names to filter on, e.g.
                ["Boundary with Normality", "Essential Features"].
                If None or empty, no section filter is applied and no
                expansion is performed.
            json_path: Path to the knowledge-base JSON used for expansion.
                Defaults to CHUNKS_PATH.
        """
        self.sections: list[str] = list(sections) if sections else []
        self._json_path = json_path or CHUNKS_PATH
        self._sections_by_disorder: dict | None = None

    def _section_map(self) -> dict:
        if self._sections_by_disorder is None:
            self._sections_by_disorder = build_sections_by_disorder(
                load_chunks(self._json_path), self.sections
            )
        return self._sections_by_disorder

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

    def search(self, query: str, k: int = 5, *, expand: bool = True) -> list[dict]:
        if not query or not query.strip():
            return []

        scored = self._deduplicate(
            self._iter_section_results(query, expansion_fetch_k(k, self.sections))
        )
        section_map = self._section_map() if (self.sections and expand) else {}
        return finish_search(
            scored,
            k,
            self.sections,
            section_map,
            "dense_score",
            expand=expand,
        )
