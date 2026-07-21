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

Run:
    python experiments/rebuild_all_artifacts.py
"""

from __future__ import annotations

import hashlib
import argparse
import json
import random
import re
import runpy
import sys
from pathlib import Path

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SRC = _PROJECT_ROOT / "src"
for _path in (_SRC, _SRC / "retriever"):
    _s = str(_path)
    if _s not in sys.path:
        sys.path.insert(0, _s)

from components.config import (  # noqa: E402
    BINARY_EVAL_PER_CLASS,
    BINARY_EVAL_SEED,
    BINARY_EVAL_SUBSET_PATH,
    BINARY_DEV_PER_CLASS,
    BINARY_DEV_SEED,
    BINARY_DEV_META_PATH,
    BINARY_DEV_SLICE_PATH,
    BINARY_LABELS,
    BINARY_MAX_CHARS,
    BINARY_MIN_CHARS,
    BINARY_SOURCE_PATH,
    DATASET_TEST_PATH,
    DATASET_TRAIN_PATH,
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
    HF_DATASET_REPO,
    HF_TEST_FILE,
    HF_TRAIN_FILE,
    RAG_EVAL_EXCLUDE,
    RAG_EVAL_LABELS,
    RAG_EVAL_MAX_CHARS,
    RAG_EVAL_MIN_CHARS,
    RAG_EVAL_PER_CLASS,
    RAG_EVAL_SEED,
    RAG_EVAL_SUBSET_PATH,
    RAG_DEV_META_PATH,
    RAG_DEV_PER_CLASS,
    RAG_DEV_SEED,
    RAG_DEV_SLICE_PATH,
    BATCH_SIZE,
    COLLECTION_NAME,
)

_WHITESPACE_RE = re.compile(r"\s+")
_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)

def _clean_text(text: str) -> str:
    text = str(text).strip()
    text = _URL_RE.sub(" ", text)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


def _ensure_multiclass_cached() -> pd.DataFrame:
    """Return concatenated train+test dataframe with columns: text, label."""
    local_ok = DATASET_TRAIN_PATH.exists() and DATASET_TEST_PATH.exists()
    if local_ok:
        train_df = pd.read_csv(DATASET_TRAIN_PATH)
        test_df = pd.read_csv(DATASET_TEST_PATH)
    else:
        from datasets import load_dataset

        ds = load_dataset(
            HF_DATASET_REPO,
            data_files={"train": HF_TRAIN_FILE, "test": HF_TEST_FILE},
        )
        train_df = ds["train"].to_pandas()
        test_df = ds["test"].to_pandas()

        # Cache for offline reuse.
        DATASET_TRAIN_PATH.parent.mkdir(parents=True, exist_ok=True)
        test_df.to_csv(DATASET_TEST_PATH, index=False)
        train_df.to_csv(DATASET_TRAIN_PATH, index=False)

    def normalize(df: pd.DataFrame, split: str) -> pd.DataFrame:
        if "status" not in df.columns or "text" not in df.columns:
            raise ValueError(
                f"Expected columns 'text' and 'status' in multiclass dataset ({split})."
            )
        return pd.DataFrame(
            {
                "text": df["text"].astype(str),
                "label": df["status"].astype(str).str.strip().str.lower(),
                "source_split": split,
            }
        )

    return pd.concat(
        [normalize(train_df, "train"), normalize(test_df, "test")],
        ignore_index=True,
    )


def _build_multiclass_fewshot_pool() -> pd.DataFrame:
    df = _ensure_multiclass_cached()

    exclude = {label.lower() for label in RAG_EVAL_EXCLUDE}
    keep = {label.lower() for label in RAG_EVAL_LABELS}
    df = df[~df["label"].isin(exclude)].copy()
    df = df[df["label"].isin(keep)].copy()

    df["text"] = df["text"].map(_clean_text)
    df = df[df["text"].str.len() > 0].copy()

    df = df[
        (df["text"].str.len() >= RAG_EVAL_MIN_CHARS)
        & (df["text"].str.len() <= RAG_EVAL_MAX_CHARS)
    ].copy()

    # Dedupe for consistent sampling.
    df["_text_key"] = df["text"].str.lower()
    df = df.drop_duplicates(subset=["_text_key"], keep="first").drop(
        columns=["_text_key"]
    )
    return df.reset_index(drop=True)


def _build_binary_fewshot_pool() -> pd.DataFrame:
    if not BINARY_SOURCE_PATH.exists():
        raise FileNotFoundError(
            f"Missing {BINARY_SOURCE_PATH}. Place the raw binary source CSV under datasets/raw/."
        )
    raw = pd.read_csv(BINARY_SOURCE_PATH)
    if "text" not in raw.columns or "class" not in raw.columns:
        raise ValueError(
            f"The binary raw dataset must contain columns 'text' and 'class'. Got {list(raw.columns)}"
        )

    df = pd.DataFrame(
        {"text": raw["text"].astype(str), "label": raw["class"].astype(str).str.strip().str.lower()}
    )
    keep = {label.lower() for label in BINARY_LABELS}
    df = df[df["label"].isin(keep)].copy()

    df["text"] = df["text"].map(_clean_text)
    df = df[df["text"].str.len() > 0].copy()

    df = df[
        (df["text"].str.len() >= BINARY_MIN_CHARS)
        & (df["text"].str.len() <= BINARY_MAX_CHARS)
    ].copy()

    df["_text_key"] = df["text"].str.lower()
    df = df.drop_duplicates(subset=["_text_key"], keep="first").drop(
        columns=["_text_key"]
    )
    return df.reset_index(drop=True)


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
    # Build pools from raw datasets
    print("\nBuilding few-shot pools from raw datasets …")
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

