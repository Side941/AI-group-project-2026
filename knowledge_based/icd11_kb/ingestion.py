"""
ingestion.py
============
Step 2 of the ICD-11 knowledge base pipeline.

Responsibilities
----------------
- Load chunk JSON produced by chunker.py.
- Embed each chunk with BioLORD-2023 (via sentence-transformers).
- Upsert embeddings into a persistent ChromaDB collection.
- Expose the populated _embedding_model and _collection handles for
  use by retrieval.py without re-loading.

Public API
----------
    run_ingestion(chunks_path, chroma_path, collection_name,
                  embedding_model_name, batch_size, rebuild) -> None
"""

from __future__ import annotations

import json

from config import (
    BATCH_SIZE,
    CHUNKS_PATH,
    CHROMA_PATH,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
)

# Module-level handles — populated by run_ingestion() or initialise_retrieval()
# (imported and mutated by retrieval.py as needed)
_embedding_model = None
_collection      = None


def run_ingestion(
    chunks_path: str          = CHUNKS_PATH,
    chroma_path: str          = CHROMA_PATH,
    collection_name: str      = COLLECTION_NAME,
    embedding_model_name: str = EMBEDDING_MODEL,
    batch_size: int           = BATCH_SIZE,
    rebuild: bool             = False,
) -> None:
    """
    Embed each chunk with BioLORD-2023 and upsert into a ChromaDB collection.

    Args:
        chunks_path          : path to the JSON produced by run_chunking().
        chroma_path          : directory for the persistent ChromaDB store.
        collection_name      : name of the ChromaDB collection to create/use.
        embedding_model_name : HuggingFace model ID for sentence-transformers.
        batch_size           : number of chunks to embed per forward pass.
        rebuild              : if True, wipe and re-ingest an existing collection.
    """
    import torch
    import chromadb
    from sentence_transformers import SentenceTransformer
    from tqdm import tqdm

    global _embedding_model, _collection

    # ── Device ────────────────────────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        print(f"GPU : {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("No GPU detected — using CPU")

    # ── Load embedding model ──────────────────────────────────────────────────
    print(f"\nLoading: {embedding_model_name} …")
    _embedding_model = SentenceTransformer(embedding_model_name, device=device)
    print(f"Loaded on  : {_embedding_model.device}")
    print(f"Output dims: {_embedding_model.get_sentence_embedding_dimension()}")

    # ── ChromaDB client ───────────────────────────────────────────────────────
    chroma_client = chromadb.PersistentClient(path=chroma_path)
    print(f"\nChromaDB path: {chroma_path}")

    existing_names = [c.name for c in chroma_client.list_collections()]
    if collection_name in existing_names:
        count = chroma_client.get_collection(collection_name).count()
        print(f"Collection '{collection_name}' already exists with {count} vectors.")
        if rebuild:
            chroma_client.delete_collection(collection_name)
            print("Deleted existing collection (rebuild=True).")
        else:
            print("Skipping ingestion. Pass rebuild=True to re-ingest.")
            _collection = chroma_client.get_collection(collection_name)
            return

    _collection = chroma_client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )
    print(f"Collection ready: '{collection_name}'  ({_collection.count()} vectors)")

    # ── Load chunks ───────────────────────────────────────────────────────────
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    print(f"Loaded {len(chunks)} chunks from {chunks_path}")

    if _collection.count() == len(chunks):
        print("Collection already fully ingested. Skipping.")
        return

    print(f"\nIngesting {len(chunks)} chunks into '{collection_name}' …\n")

    for i in tqdm(range(0, len(chunks), batch_size), desc="Batches"):
        batch = chunks[i : i + batch_size]

        texts_to_embed = [c["embed_text"] for c in batch]

        ids = [
            f"{c['disorder_code']}_{c['section'].replace(' ', '_').replace('/', '_')}_{i + j}"
            for j, c in enumerate(batch)
        ]

        metadatas = [
            {
                "domain":        c["domain"],
                "disorder_code": c["disorder_code"],
                "disorder_name": c["disorder_name"],
                "section":       c["section"],
                "word_count":    c["word_count"],
            }
            for c in batch
        ]

        documents = [c["text"] for c in batch]

        embeddings = _embedding_model.encode(
            texts_to_embed,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        ).tolist()

        _collection.add(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents,
        )

    print(f"\nDone. Collection '{collection_name}' now has {_collection.count()} vectors.")
