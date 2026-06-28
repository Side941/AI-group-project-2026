"""
retrieval.py
============
Step 3 of the ICD-11 knowledge base pipeline.

Responsibilities
----------------
- Initialise (or reuse) the embedding model and ChromaDB collection.
- Query the collection for the top-k chunks matching a natural-language input.
- Format results as a ready-to-inject LLM context string.

Public API
----------
    initialise_retrieval(chroma_path, collection_name, embedding_model_name)
    query_icd11(query_text, n_results, section_filter)
    get_icd11_context(query_text, n_results, section_filter, include_metadata) -> str
"""
from __future__ import annotations

import components.ingestion as ingestion
from components.config import CHROMA_PATH, COLLECTION_NAME, EMBEDDING_MODEL


def initialise_retrieval(
    chroma_path: str          = CHROMA_PATH,
    collection_name: str      = COLLECTION_NAME,
    embedding_model_name: str = EMBEDDING_MODEL,
) -> None:
    import torch
    import chromadb
    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading embedding model on {device} …")
    ingestion._embedding_model = SentenceTransformer(embedding_model_name, device=device)

    chroma_client = chromadb.PersistentClient(path=chroma_path)
    ingestion._collection = chroma_client.get_collection(collection_name)
    print(f"Ready. Collection '{collection_name}' has {ingestion._collection.count()} vectors.")


def _require_retrieval() -> None:
    if ingestion._embedding_model is None or ingestion._collection is None:
        raise RuntimeError(
            "Retrieval not initialised. Call initialise_retrieval() or run_ingestion() first."
        )


def query_icd11(
    query_text: str,
    n_results: int             = 5,
    section_filter: str | None = None,
) -> None:
    _require_retrieval()

    query_embedding = ingestion._embedding_model.encode(
        [query_text], normalize_embeddings=True,
    ).tolist()

    where = {"section": {"$eq": section_filter}} if section_filter else None

    results = ingestion._collection.query(
        query_embeddings=query_embedding,
        n_results=n_results,
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    print(f"Query : '{query_text}'")
    if section_filter:
        print(f"Filter: section = '{section_filter}'")
    print("-" * 65)

    for i in range(len(results["documents"][0])):
        meta = results["metadatas"][0][i]
        doc  = results["documents"][0][i]
        dist = results["distances"][0][i]
        sim  = 1 - dist

        print(f"[{i+1}] {meta['disorder_name']} ({meta['disorder_code']})")
        print(f"     Section    : {meta['section']}")
        print(f"     Domain     : {meta['domain']}")
        print(f"     Similarity : {sim:.4f}")
        print(f"     Text       : {doc}")
        print()


def get_icd11_context(
    query_text: str,
    n_results: int             = 5,
    section_filter: str | None = None,
    include_metadata: bool     = True,
) -> str:
    _require_retrieval()

    query_embedding = ingestion._embedding_model.encode(
        [query_text], normalize_embeddings=True,
    ).tolist()

    where = {"section": {"$eq": section_filter}} if section_filter else None

    results = ingestion._collection.query(
        query_embeddings=query_embedding,
        n_results=n_results,
        where=where,
        include=["documents", "metadatas"],
    )

    parts: list[str] = []
    for i in range(len(results["documents"][0])):
        meta = results["metadatas"][0][i]
        doc  = results["documents"][0][i]

        if include_metadata:
            header = (
                f"[{meta['disorder_name']} ({meta['disorder_code']}) "
                f"— {meta['section']}]"
            )
            parts.append(f"{header}\n{doc}")
        else:
            parts.append(doc)

    return "\n\n".join(parts)

