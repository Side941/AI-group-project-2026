"""
rebuild_all_artifacts.py
========================
Rebuild both processed evaluation datasets (multiclass + binary) and build a
few-shot Chroma vector store from the raw datasets.

Blueprint alignment:
- Multiclass head uses the HF mental-health dataset.
- Binary head uses `datasets/raw/suicide_detection_raw.csv`.
- Few-shot vector DB stores (example_text, label) pairs that can later be
  retrieved to assemble dynamic few-shot prompts.

Important:
- We build the vector DB, but we do NOT wire retrieval from it into the
  classification pipeline yet.
- Few-shot examples are taken from the raw datasets, but they are passed
  through the same cleaning / filtering pipeline used by the processed
  evaluation datasets before they are sampled and embedded.

Run:
    python experiments/rebuild_all_artifacts.py
"""

from __future__ import annotations

import argparse
import json
import random
import runpy
import sys
from pathlib import Path

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SRC = _PROJECT_ROOT / "src"
_EXPERIMENTS = _PROJECT_ROOT / "experiments"
for _path in (_SRC, _SRC / "retriever"):
    _s = str(_path)
    if _s not in sys.path:
        sys.path.insert(0, _s)
_exp = str(_EXPERIMENTS)
if _exp not in sys.path:
    sys.path.insert(0, _exp)

from components.config import (  # noqa: E402
    BINARY_LABELS,
    EMBEDDING_MODEL,
    FEWSHOT_BUILD_SEED,
    FEWSHOT_CHROMA_PATH,
    FEWSHOT_COLLECTION_BINARY,
    FEWSHOT_COLLECTION_MULTICLASS,
    FEWSHOT_DB_META_PATH,
    FEWSHOT_EMBED_BATCH_SIZE,
    FEWSHOT_EMBEDDING_MAX_SEQ_LENGTH,
    FEWSHOT_BINARY_MAX_PER_CLASS,
    FEWSHOT_MULTICLASS_MAX_PER_LABEL,
    RAG_EVAL_LABELS,
)
import build_binary_dataset as binary_builder  # noqa: E402
import build_multiclass_dataset as multiclass_builder  # noqa: E402


def _build_multiclass_fewshot_pool() -> pd.DataFrame:
    raw, _ = multiclass_builder.load_raw_frames()
    cleaned, _ = multiclass_builder.filter_and_clean(raw)
    return cleaned.reset_index(drop=True)


def _build_binary_fewshot_pool() -> pd.DataFrame:
    raw, _ = binary_builder.load_raw_frame()
    cleaned, _ = binary_builder.filter_and_clean(raw)
    return cleaned.reset_index(drop=True)


def _sample_balanced(df: pd.DataFrame, *, labels: list[str], per_label: int, seed: int) -> pd.DataFrame:
    random.seed(seed)
    parts: list[pd.DataFrame] = []
    for label in labels:
        pool = df[df["label"] == label].copy()
        pool = pool.sort_values(["text"], kind="mergesort")
        if len(pool) < per_label:
            raise RuntimeError(
                f"Not enough rows for label '{label}' in few-shot pool: need {per_label}, have {len(pool)}."
            )
        parts.append(pool.sample(n=per_label, random_state=seed))
    out = pd.concat(parts, ignore_index=True)
    # Deterministic final order for stable ids.
    label_order = {label: i for i, label in enumerate(labels)}
    out["_label_order"] = out["label"].map(label_order)
    out = out.sort_values(["_label_order", "text"], kind="mergesort").drop(
        columns=["_label_order"]
    )
    return out.reset_index(drop=True)


def _embed_and_write_collection(
    *,
    chroma_client,
    collection_name: str,
    texts: list[str],
    labels: list[str],
    metadatas: list[dict],
    ids: list[str],
    rebuild: bool,
) -> None:
    import torch
    import chromadb
    from sentence_transformers import SentenceTransformer
    from tqdm import tqdm

    global _EMBEDDER
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nEmbedding for collection '{collection_name}' on device: {device}")

    embedder = SentenceTransformer(EMBEDDING_MODEL, device=device)
    embedder.max_seq_length = FEWSHOT_EMBEDDING_MAX_SEQ_LENGTH
    print("Embedding model dims:", embedder.get_sentence_embedding_dimension())

    if rebuild:
        existing = [c.name for c in chroma_client.list_collections()]
        if collection_name in existing:
            chroma_client.delete_collection(collection_name)

    collection = chroma_client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    existing_count = collection.count()
    if existing_count == len(texts):
        print(f"Collection '{collection_name}' already has {collection.count()} vectors; skipping.")
        return
    if existing_count != 0 and not rebuild:
        raise RuntimeError(
            f"Few-shot collection '{collection_name}' already exists with {existing_count} vectors, "
            f"but expected {len(texts)}. Re-run with --reuse-fewshot-db only when counts match, "
            f"or use default (rebuild)."
        )

    print(f"Writing {len(texts)} vectors into '{collection_name}' …")
    # Embed in batches.
    for i in tqdm(range(0, len(texts), FEWSHOT_EMBED_BATCH_SIZE), desc="Embedding batches"):
        batch_texts = texts[i : i + FEWSHOT_EMBED_BATCH_SIZE]
        batch_embeddings = embedder.encode(
            batch_texts,
            batch_size=FEWSHOT_EMBED_BATCH_SIZE,
            show_progress_bar=False,
            normalize_embeddings=True,
        ).tolist()

        batch_ids = ids[i : i + FEWSHOT_EMBED_BATCH_SIZE]
        batch_metas = metadatas[i : i + FEWSHOT_EMBED_BATCH_SIZE]
        collection.add(
            ids=batch_ids,
            embeddings=batch_embeddings,
            metadatas=batch_metas,
            documents=batch_texts,
        )

    print(f"Done. Collection '{collection_name}' now has {collection.count()} vectors.")


def build_fewshot_db(*, rebuild: bool, seed: int) -> dict:
    # Build pools from raw datasets using the same cleaning/filter pipeline
    # as the processed eval/dev builders for consistency.
    print("\nBuilding few-shot pools from raw datasets via the shared preprocessing pipeline …")
    multi_pool = _build_multiclass_fewshot_pool()
    bin_pool = _build_binary_fewshot_pool()

    # Sample balanced caps for deterministic size.
    multi_sample = _sample_balanced(
        multi_pool,
        labels=list(RAG_EVAL_LABELS),
        per_label=FEWSHOT_MULTICLASS_MAX_PER_LABEL,
        seed=seed,
    )
    bin_sample = _sample_balanced(
        bin_pool,
        labels=list(BINARY_LABELS),
        per_label=FEWSHOT_BINARY_MAX_PER_CLASS,
        seed=seed,
    )

    # Chroma persistence
    FEWSHOT_CHROMA_PATH.parent.mkdir(parents=True, exist_ok=True)
    import chromadb

    chroma_client = chromadb.PersistentClient(path=str(FEWSHOT_CHROMA_PATH))

    # Multiclass collection
    multi_texts = multi_sample["text"].tolist()
    multi_labels = multi_sample["label"].tolist()
    multi_ids = [f"mc_{i:06d}" for i in range(len(multi_texts))]
    multi_metas = [{"head": "multiclass", "label": lbl} for lbl in multi_labels]
    _embed_and_write_collection(
        chroma_client=chroma_client,
        collection_name=FEWSHOT_COLLECTION_MULTICLASS,
        texts=multi_texts,
        labels=multi_labels,
        metadatas=multi_metas,
        ids=multi_ids,
        rebuild=rebuild,
    )

    # Binary collection
    bin_texts = bin_sample["text"].tolist()
    bin_labels = bin_sample["label"].tolist()
    bin_ids = [f"bin_{i:06d}" for i in range(len(bin_texts))]
    bin_metas = [{"head": "binary", "label": lbl} for lbl in bin_labels]
    _embed_and_write_collection(
        chroma_client=chroma_client,
        collection_name=FEWSHOT_COLLECTION_BINARY,
        texts=bin_texts,
        labels=bin_labels,
        metadatas=bin_metas,
        ids=bin_ids,
        rebuild=rebuild,
    )

    meta = {
        "artifact": "fewshot_db",
        "seed": seed,
        "embedding_model": EMBEDDING_MODEL,
        "collections": {
            "multiclass": {
                "collection_name": FEWSHOT_COLLECTION_MULTICLASS,
                "per_label": FEWSHOT_MULTICLASS_MAX_PER_LABEL,
                "labels": list(RAG_EVAL_LABELS),
                "total_vectors": int(len(multi_texts)),
            },
            "binary": {
                "collection_name": FEWSHOT_COLLECTION_BINARY,
                "per_class": FEWSHOT_BINARY_MAX_PER_CLASS,
                "labels": list(BINARY_LABELS),
                "total_vectors": int(len(bin_texts)),
            },
        },
        "multiclass_raw_pool_size": int(len(multi_pool)),
        "binary_raw_pool_size": int(len(bin_pool)),
        "rebuild": rebuild,
    }
    FEWSHOT_DB_META_PATH.write_text(
        json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"\nWrote few-shot meta: {FEWSHOT_DB_META_PATH}")
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rebuild eval/dev splits and build few-shot vector DB (no retrieval wiring yet)."
    )
    parser.add_argument(
        "--skip-evals",
        action="store_true",
        help="Skip rebuilding eval/dev CSVs (still builds few-shot vector DB).",
    )
    parser.add_argument(
        "--reuse-fewshot-db",
        action="store_true",
        help="Reuse existing few-shot Chroma vectors when counts match (skips expensive re-embedding).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=FEWSHOT_BUILD_SEED,
        help="Seed used for deterministic few-shot sampling (also used by builder scripts).",
    )
    args = parser.parse_args()

    # CLI seed currently only affects few-shot sampling because the two eval/dev
    # builders hardcode their own seeds via config constants. If you want them
    # to follow a different seed, edit config and re-run.
    print("=== Rebuild eval/dev ===")
    if args.skip_evals:
        print("Skipping eval/dev rebuild.")
    else:
        runpy.run_path(str(_PROJECT_ROOT / "experiments" / "build_multiclass_dataset.py"), run_name="__main__")
        runpy.run_path(str(_PROJECT_ROOT / "experiments" / "build_binary_dataset.py"), run_name="__main__")

    # Always build few-shot DB (this is the main request).
    print("=== Build few-shot vector DB ===")
    build_fewshot_db(rebuild=not bool(args.reuse_fewshot_db), seed=int(args.seed))

    print("\nDone.")


if __name__ == "__main__":
    main()

