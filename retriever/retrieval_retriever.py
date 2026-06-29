from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Sequence

from retrieval import search_icd11
from utils import load_chunks
from components.config import CHUNKS_PATH


class RetrievalRetriever:
    """
    Dense retriever backed by retrieval.py / ChromaDB.

    Supports optional section filtering; when multiple sections are provided,
    results from each section are merged and de-duplicated by chunk id, keeping
    the highest dense_score per id.
    """

    def __init__(
        self,
        sections: Sequence[str] | None = None,
        json_path: str | None = None,
        expand_same_disorder: bool = True,
    ):
        """
        Args:
            sections: Optional list of section names to filter on, e.g.
                ["Boundary with Normality", "Essential Features"].
                If None or empty, no section filter is applied.
            json_path: Optional knowledge-base JSON path used for same-disorder
                expansion. Defaults to utils.load_chunks() path.
            expand_same_disorder: If True, retrieve dense seeds first, then add
                requested sections from the same disorder code/name so the LLM
                sees a more complete clinical picture.
        """
        self.sections: list[str] = list(sections) if sections else []
        self.expand_same_disorder = expand_same_disorder
        self._sections_by_disorder: dict[tuple[str, str], dict[str, dict]] = defaultdict(dict)

        if expand_same_disorder:
            chunks = load_chunks(json_path or CHUNKS_PATH)
            for chunk in chunks:
                key = (chunk.get("disorder_code", ""), chunk.get("disorder_name", ""))
                self._sections_by_disorder[key][chunk.get("section", "")] = chunk

    def _iter_section_results(self, query: str, k: int) -> Iterable[dict]:
        if not self.sections:
            # No filtering — simple dense retrieval
            for row in search_icd11(query_text=query, n_results=k):
                yield row
            return

        # With filtering — query each section separately
        for section in self.sections:
            for row in search_icd11(
                query_text=query,
                n_results=k,
                section_filter=section,
            ):
                yield row

    def _expand_with_same_disorder_sections(self, seeds: list[dict]) -> list[dict]:
        if not self.expand_same_disorder or not seeds:
            return seeds

        if not self.sections:
            return seeds

        expanded: dict[str, dict] = {}

        # Keep seed rows first.
        for row in seeds:
            expanded[row["id"]] = row

        # Add requested sections from the same disorder as each seed row.
        for seed in seeds:
            key = (seed.get("disorder_code", ""), seed.get("disorder_name", ""))
            section_map = self._sections_by_disorder.get(key, {})
            for section in self.sections:
                related = section_map.get(section)
                if not related:
                    continue
                rid = related["id"]
                if rid in expanded:
                    continue

                # Related chunks are ranked just below the seed disorder hit.
                row = {
                    "id": rid,
                    "text": related.get("prompt_text", related.get("text", "")),
                    "prompt_text": related.get("prompt_text", related.get("text", "")),
                    "dense_score": float(seed.get("dense_score", 0.0)) * 0.999,
                    "domain": related.get("domain", seed.get("domain", "")),
                    "disorder_code": related.get("disorder_code", seed.get("disorder_code", "")),
                    "disorder_name": related.get("disorder_name", seed.get("disorder_name", "")),
                    "section": related.get("section", section),
                    "word_count": related.get("word_count"),
                }
                expanded[rid] = row

        rows = list(expanded.values())
        rows.sort(key=lambda r: r["dense_score"], reverse=True)
        return rows

    def search(self, query: str, k: int = 5) -> list[dict]:
        if not query or not query.strip():
            return []

        best_by_id: dict[str, dict] = {}
        for row in self._iter_section_results(query=query, k=k):
            cid = row["id"]
            prev = best_by_id.get(cid)
            if prev is None or row["dense_score"] > prev["dense_score"]:
                best_by_id[cid] = row

        results = list(best_by_id.values())
        results = self._expand_with_same_disorder_sections(results)
        results.sort(key=lambda r: r["dense_score"], reverse=True)
        return results[:k]

