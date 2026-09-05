"""
dense_retriever.py
==================
Dense retrieval over the ICD-11 ChromaDB collection.

Responsibilities
----------------
- Initialise (or reuse) the embedding model and ChromaDB collection.
- Low-level Chroma query helper (`search_icd11`).
- `DenseRetriever` class with the same search interface as BM25/Hybrid
  (optional per-section queries + post-retrieval section expansion).

Public API
----------
    initialise_retrieval(chroma_path, collection_name, embedding_model_name)
    search_icd11(query_text, n_results, section_filter) -> list[dict]
    DenseRetriever(sections, json_path).search(query, k, expand=True)
"""

from __future__ import annotations

from typing import Iterable, Sequence

import components.ingestion as ingestion
from components.config import CHROMA_PATH, CHUNKS_PATH, COLLECTION_NAME, EMBEDDING_MODEL
from section_expander import (
    build_sections_by_disorder,
    expansion_fetch_k,
    finish_search,
)
from utils import load_chunks


def initialise_retrieval(
    chroma_path: str = CHROMA_PATH,
    collection_name: str = COLLECTION_NAME,
    embedding_model_name: str = EMBEDDING_MODEL,
) -> None:
    import torch
    import chromadb
    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading embedding model on {device} …")
    ingestion._embedding_model = SentenceTransformer(embedding_model_name, device=device)

    chroma_client = chromadb.PersistentClient(path=str(chroma_path))
    ingestion._collection = chroma_client.get_collection(collection_name)
    print(f"Ready. Collection '{collection_name}' has {ingestion._collection.count()} vectors.")


def _require_retrieval() -> None:
    if ingestion._embedding_model is None or ingestion._collection is None:
        raise RuntimeError(
            "Retrieval not initialised. Call initialise_retrieval() or run_ingestion() first."
        )


def search_icd11(
    query_text: str,
    n_results: int = 5,
    section_filter: str | None = None,
) -> list[dict]:
    """Return dense retrieval results from ChromaDB as structured chunk dicts."""
    _require_retrieval()

    query_embedding = ingestion._embedding_model.encode(
        [query_text],
        normalize_embeddings=True,
    ).tolist()

    where = {"section": {"$eq": section_filter}} if section_filter else None

    results = ingestion._collection.query(
        query_embeddings=query_embedding,
        n_results=n_results,
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    rows: list[dict] = []
    for i in range(len(results["documents"][0])):
        meta = results["metadatas"][0][i]
        doc = results["documents"][0][i]
        dist = results["distances"][0][i]
        rows.append({
            # Must match the id utils.load_chunks assigns on the BM25 side so
            # hybrid RRF can fuse rows for the same chunk. Prefer the stored
            # chunk_uid; rebuild the part-aware form for legacy indexes.
            "id": meta.get("chunk_uid") or (
                f"{meta['disorder_code']}_"
                f"{meta['section'].lower().replace(' ', '_')}_"
                f"p{meta.get('chunk_part') or 1}"
            ),
            "text": doc,
            "prompt_text": doc,
            "dense_score": 1 - dist,
            **meta,
        })
    return rows


class DenseRetriever:
    """
    Dense retriever backed by ChromaDB (`search_icd11`).

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
        if not self.sections:
            yield from search_icd11(query_text=query, n_results=k)
            return
        for section in self.sections:
            yield from search_icd11(
                query_text=query,
                n_results=k,
                section_filter=section,
            )

    def _deduplicate(self, rows: Iterable[dict]) -> list[dict]:
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
