"""
build_curated_fewshot_db.py
===========================
Build the few-shot Chroma collection from the hand-curated example store
(knowledge_base/fewshot/multiclass_examples_curated.json).

The full auto-sampled store (multiclass_examples.json) is still exported by
rebuild_all_artifacts.py and kept as the pool the curated examples were picked
from, but it is NOT embedded: dense few-shot retrieval runs only over the
curated collection built here. Any stale full-pool collection is dropped so
there is no ambiguity about which vectors are in use.

Run standalone:
    python src/builders/build_curated_fewshot_db.py

Or let rebuild_all_artifacts.py call it as part of the full rebuild.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_BUILDERS_DIR = Path(__file__).resolve().parent
_SRC = _BUILDERS_DIR.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from components.config import (  # noqa: E402
    EMBEDDING_MODEL,
    FEWSHOT_CHROMA_PATH,
    FEWSHOT_COLLECTION_MULTICLASS,
    FEWSHOT_COLLECTION_MULTICLASS_CURATED,
    FEWSHOT_DB_META_PATH,
    FEWSHOT_EMBED_BATCH_SIZE,
    FEWSHOT_EMBEDDING_MAX_SEQ_LENGTH,
    FEWSHOT_MULTICLASS_CURATED_EXAMPLES_PATH,
    RAG_EVAL_LABELS,
)


def load_curated_rows() -> list[dict]:
    path = FEWSHOT_MULTICLASS_CURATED_EXAMPLES_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Curated few-shot examples missing at {path}. "
            "Regenerate with: python scripts/make_curated_fewshot.py"
        )
    rows = json.loads(path.read_text(encoding="utf-8"))

    labels = {label: 0 for label in RAG_EVAL_LABELS}
    for row in rows:
        if row["label"] not in labels:
            raise ValueError(f"Unexpected label {row['label']!r} in {path}")
        labels[row["label"]] += 1
    print(f"Curated few-shot store: {len(rows)} examples {labels}")
    return rows


def build_curated_fewshot_collection(*, rebuild: bool = True) -> dict:
    """Embed the curated examples into their own Chroma collection."""
    import chromadb
    import torch
    from sentence_transformers import SentenceTransformer

    rows = load_curated_rows()

    FEWSHOT_CHROMA_PATH.parent.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(FEWSHOT_CHROMA_PATH))
    existing = [c.name for c in client.list_collections()]

    # The full-pool collection is superseded by the curated one; drop it (and
    # any legacy collection) so the store only contains vectors that are used.
    for stale in (FEWSHOT_COLLECTION_MULTICLASS, "fewshot_binary"):
        if stale in existing:
            client.delete_collection(stale)
            print(f"Removed superseded few-shot collection '{stale}'.")

    if FEWSHOT_COLLECTION_MULTICLASS_CURATED in existing:
        if not rebuild:
            collection = client.get_collection(FEWSHOT_COLLECTION_MULTICLASS_CURATED)
            if collection.count() == len(rows):
                print(
                    f"Collection '{FEWSHOT_COLLECTION_MULTICLASS_CURATED}' already has "
                    f"{collection.count()} vectors; skipping re-embed."
                )
                return _summary(rows)
        client.delete_collection(FEWSHOT_COLLECTION_MULTICLASS_CURATED)

    collection = client.create_collection(
        name=FEWSHOT_COLLECTION_MULTICLASS_CURATED,
        metadata={"hnsw:space": "cosine"},
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Embedding {len(rows)} curated examples with {EMBEDDING_MODEL} on {device} ...")
    embedder = SentenceTransformer(EMBEDDING_MODEL, device=device)
    embedder.max_seq_length = FEWSHOT_EMBEDDING_MAX_SEQ_LENGTH

    texts = [row["text"] for row in rows]
    embeddings = embedder.encode(
        texts,
        batch_size=FEWSHOT_EMBED_BATCH_SIZE,
        show_progress_bar=False,
        normalize_embeddings=True,
    ).tolist()

    collection.add(
        ids=[row["id"] for row in rows],
        embeddings=embeddings,
        documents=texts,
        metadatas=[
            {"label": row["label"], "source_split": row.get("source_split", "source")}
            for row in rows
        ],
    )
    print(
        f"Done. Collection '{FEWSHOT_COLLECTION_MULTICLASS_CURATED}' now has "
        f"{collection.count()} vectors."
    )
    return _summary(rows)


def _summary(rows: list[dict]) -> dict:
    per_label: dict[str, int] = {}
    for row in rows:
        per_label[row["label"]] = per_label.get(row["label"], 0) + 1
    return {
        "collection_name": FEWSHOT_COLLECTION_MULTICLASS_CURATED,
        "examples_json": str(FEWSHOT_MULTICLASS_CURATED_EXAMPLES_PATH),
        "labels": list(RAG_EVAL_LABELS),
        "per_label": per_label,
        "total_vectors": len(rows),
    }


def update_meta(curated_summary: dict) -> None:
    """Record the curated collection in the few-shot meta file."""
    meta: dict = {}
    if FEWSHOT_DB_META_PATH.exists():
        meta = json.loads(FEWSHOT_DB_META_PATH.read_text(encoding="utf-8"))
    meta.setdefault("artifact", "fewshot_db")
    meta["embedding_model"] = EMBEDDING_MODEL
    collections = meta.setdefault("collections", {})
    collections["multiclass_curated"] = curated_summary
    # The full pool is exported as JSON only; make that explicit in the meta.
    if "multiclass" in collections:
        collections["multiclass"]["collection_name"] = None
        collections["multiclass"]["total_vectors"] = 0
        collections["multiclass"]["note"] = (
            "Pool JSON export only; vectors live in the curated collection."
        )
    FEWSHOT_DB_META_PATH.parent.mkdir(parents=True, exist_ok=True)
    FEWSHOT_DB_META_PATH.write_text(
        json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Updated few-shot meta: {FEWSHOT_DB_META_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the curated few-shot Chroma collection."
    )
    parser.add_argument(
        "--reuse",
        action="store_true",
        help="Skip re-embedding when the curated collection already matches.",
    )
    args = parser.parse_args()

    summary = build_curated_fewshot_collection(rebuild=not args.reuse)
    update_meta(summary)


if __name__ == "__main__":
    main()
