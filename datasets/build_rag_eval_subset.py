"""
build_rag_eval_subset.py
========================
Build reproducible stratified RAG evaluation artifacts from the
ourafla/Mental-Health_Text-Classification_Dataset corpus:

1) rag_eval_subset.csv  — final reporting set (150/class = 450)
2) rag_dev_slice.csv    — prompt/k tuning slice drawn from (1) (10/class = 30)

Prefer local CSVs under datasets/; fall back to the Hugging Face Hub.
Re-running with the same inputs and seeds must produce identical CSVs.

Usage (from repo root):
    python datasets/build_rag_eval_subset.py
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

import pandas as pd

# Allow `python datasets/build_rag_eval_subset.py` from the repo root.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from components.config import (
    DATASET_TEST_PATH,
    DATASET_TRAIN_PATH,
    HF_DATASET_REPO,
    HF_TEST_FILE,
    HF_TRAIN_FILE,
    RAG_DEV_META_PATH,
    RAG_DEV_PER_CLASS,
    RAG_DEV_SEED,
    RAG_DEV_SLICE_PATH,
    RAG_EVAL_EXCLUDE,
    RAG_EVAL_LABELS,
    RAG_EVAL_MAX_CHARS,
    RAG_EVAL_META_PATH,
    RAG_EVAL_MIN_CHARS,
    RAG_EVAL_PER_CLASS,
    RAG_EVAL_SEED,
    RAG_EVAL_SUBSET_PATH,
)

_WHITESPACE_RE = re.compile(r"\s+")
_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)


def _file_info(path: Path) -> dict:
    if not path.exists():
        return {"path": str(path), "exists": False}
    return {
        "path": str(path),
        "exists": True,
        "size_bytes": path.stat().st_size,
    }


def _normalize_frame(df: pd.DataFrame, source_split: str) -> pd.DataFrame:
    if "status" not in df.columns or "text" not in df.columns:
        raise ValueError(
            f"Expected columns 'text' and 'status' in {source_split} split; "
            f"got {list(df.columns)}"
        )
    out = pd.DataFrame(
        {
            "text": df["text"].astype(str),
            "label": df["status"].astype(str).str.strip().str.lower(),
            "source_split": source_split,
        }
    )
    return out


def load_raw_frames() -> tuple[pd.DataFrame, dict]:
    """Load train+test from local CSVs, or Hugging Face Hub if missing."""
    local_ok = DATASET_TRAIN_PATH.exists() and DATASET_TEST_PATH.exists()
    stats: dict = {
        "load_mode": "local" if local_ok else "huggingface",
        "train_input": _file_info(DATASET_TRAIN_PATH),
        "test_input": _file_info(DATASET_TEST_PATH),
        "hf_repo": HF_DATASET_REPO,
    }

    if local_ok:
        print(f"Loading local CSVs:\n  {DATASET_TRAIN_PATH}\n  {DATASET_TEST_PATH}")
        train_df = pd.read_csv(DATASET_TRAIN_PATH)
        test_df = pd.read_csv(DATASET_TEST_PATH)
    else:
        print(
            "Local CSVs missing; downloading from Hugging Face Hub "
            f"({HF_DATASET_REPO})..."
        )
        from datasets import load_dataset

        ds = load_dataset(
            HF_DATASET_REPO,
            data_files={
                "train": HF_TRAIN_FILE,
                "test": HF_TEST_FILE,
            },
        )
        train_df = ds["train"].to_pandas()
        test_df = ds["test"].to_pandas()

        # Cache for offline reuse by teammates / later runs.
        DATASET_TRAIN_PATH.parent.mkdir(parents=True, exist_ok=True)
        train_df.to_csv(DATASET_TRAIN_PATH, index=False)
        test_df.to_csv(DATASET_TEST_PATH, index=False)
        print(f"Cached Hub downloads to:\n  {DATASET_TRAIN_PATH}\n  {DATASET_TEST_PATH}")
        stats["train_input"] = _file_info(DATASET_TRAIN_PATH)
        stats["test_input"] = _file_info(DATASET_TEST_PATH)

    frames = [
        _normalize_frame(train_df, "train"),
        _normalize_frame(test_df, "test"),
    ]
    merged = pd.concat(frames, ignore_index=True)
    stats["rows_raw"] = int(len(merged))
    stats["label_counts_raw"] = merged["label"].value_counts().to_dict()
    return merged, stats


def clean_text(text: str) -> str:
    text = text.strip()
    text = _URL_RE.sub(" ", text)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


def filter_and_clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    stats: dict = {}
    exclude = {label.lower() for label in RAG_EVAL_EXCLUDE}
    keep = {label.lower() for label in RAG_EVAL_LABELS}

    before_exclude = len(df)
    df = df[~df["label"].isin(exclude)].copy()
    stats["rows_after_exclude"] = int(len(df))
    stats["rows_dropped_exclude"] = int(before_exclude - len(df))

    df = df[df["label"].isin(keep)].copy()
    stats["rows_after_label_filter"] = int(len(df))

    df["text"] = df["text"].map(clean_text)
    before_empty = len(df)
    df = df[df["text"].str.len() > 0].copy()
    stats["rows_dropped_empty"] = int(before_empty - len(df))

    before_len = len(df)
    df = df[
        (df["text"].str.len() >= RAG_EVAL_MIN_CHARS)
        & (df["text"].str.len() <= RAG_EVAL_MAX_CHARS)
    ].copy()
    stats["rows_after_length_filter"] = int(len(df))
    stats["rows_dropped_length"] = int(before_len - len(df))

    before_dedupe = len(df)
    df["_text_key"] = df["text"].str.lower()
    df = df.drop_duplicates(subset=["_text_key"], keep="first").drop(
        columns=["_text_key"]
    )
    stats["rows_after_dedupe"] = int(len(df))
    stats["rows_dropped_dedupe"] = int(before_dedupe - len(df))
    stats["label_counts_clean"] = df["label"].value_counts().to_dict()
    return df.reset_index(drop=True), stats


def stratified_sample(
    df: pd.DataFrame,
    *,
    per_class: int,
    seed: int,
    id_prefix: str,
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for label in RAG_EVAL_LABELS:
        pool = df[df["label"] == label]
        n_available = len(pool)
        if n_available < per_class:
            raise RuntimeError(
                f"Not enough rows for label '{label}': "
                f"need {per_class}, have {n_available} after filtering."
            )
        # Stable order within class before sampling so pandas RNG is deterministic.
        sort_cols = [c for c in ("source_split", "text", "row_id") if c in pool.columns]
        pool = pool.sort_values(sort_cols, kind="mergesort")
        sampled = pool.sample(n=per_class, random_state=seed)
        parts.append(sampled)

    # Deterministic final order: by label order, then text.
    out = pd.concat(parts, ignore_index=True)
    label_order = {label: i for i, label in enumerate(RAG_EVAL_LABELS)}
    out["_label_order"] = out["label"].map(label_order)
    out = out.sort_values(["_label_order", "text"], kind="mergesort").drop(
        columns=["_label_order"]
    )
    out = out.reset_index(drop=True)

    # Preserve parent eval row_id when carving the dev slice from the final set.
    if "row_id" in out.columns and id_prefix == "dev":
        out = out.rename(columns={"row_id": "parent_row_id"})
        out.insert(0, "row_id", [f"dev_{i:04d}" for i in range(len(out))])
        cols = ["row_id", "parent_row_id", "text", "label", "source_split"]
        return out[cols]

    out = out.drop(columns=["row_id"], errors="ignore")
    out.insert(0, "row_id", [f"{id_prefix}_{i:04d}" for i in range(len(out))])
    return out[["row_id", "text", "label", "source_split"]]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_csv_and_meta(
    frame: pd.DataFrame,
    csv_path: Path,
    meta_path: Path,
    meta: dict,
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(csv_path, index=False, lineterminator="\n")

    meta["output_csv"] = str(csv_path)
    meta["output_rows"] = int(len(frame))
    meta["output_label_counts"] = frame["label"].value_counts().to_dict()
    meta["csv_sha256"] = sha256_file(csv_path)

    meta_path.write_text(
        json.dumps(meta, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {csv_path} ({len(frame)} rows)")
    print(f"Wrote {meta_path}")
    print(f"SHA256: {meta['csv_sha256']}")


def build() -> tuple[pd.DataFrame, pd.DataFrame]:
    raw, load_stats = load_raw_frames()
    cleaned, clean_stats = filter_and_clean(raw)

    subset = stratified_sample(
        cleaned,
        per_class=RAG_EVAL_PER_CLASS,
        seed=RAG_EVAL_SEED,
        id_prefix="rag",
    )
    eval_meta = {
        "artifact": "rag_eval_subset",
        "seed": RAG_EVAL_SEED,
        "per_class": RAG_EVAL_PER_CLASS,
        "labels": list(RAG_EVAL_LABELS),
        "exclude_labels": list(RAG_EVAL_EXCLUDE),
        "min_chars": RAG_EVAL_MIN_CHARS,
        "max_chars": RAG_EVAL_MAX_CHARS,
        **load_stats,
        **clean_stats,
    }
    write_csv_and_meta(subset, RAG_EVAL_SUBSET_PATH, RAG_EVAL_META_PATH, eval_meta)
    print("Final eval label counts:\n" + subset["label"].value_counts().to_string())

    # Dev slice is carved from the locked final set (not re-sampled from the full pool),
    # so tuning posts are a transparent subset of the reporting set.
    dev = stratified_sample(
        subset,
        per_class=RAG_DEV_PER_CLASS,
        seed=RAG_DEV_SEED,
        id_prefix="dev",
    )
    dev_meta = {
        "artifact": "rag_dev_slice",
        "seed": RAG_DEV_SEED,
        "per_class": RAG_DEV_PER_CLASS,
        "labels": list(RAG_EVAL_LABELS),
        "parent_csv": str(RAG_EVAL_SUBSET_PATH),
        "parent_sha256": eval_meta["csv_sha256"],
        "parent_row_ids": dev["parent_row_id"].tolist(),
        "purpose": "prompt/k tuning only; switch notebook eval_mode to final for reporting",
    }
    write_csv_and_meta(dev, RAG_DEV_SLICE_PATH, RAG_DEV_META_PATH, dev_meta)
    print("Dev slice label counts:\n" + dev["label"].value_counts().to_string())
    return subset, dev


def main() -> None:
    build()


if __name__ == "__main__":
    main()
