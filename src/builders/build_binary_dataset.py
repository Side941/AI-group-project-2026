"""
build_binary_dataset.py
=======================
Build reproducible stratified binary suicide evaluation artifacts from:
    datasets/raw/suicide_detection_raw.csv

Outputs:
1) datasets/processed/binary_suicide_eval.csv  — final reporting set (150/class = 300)
2) datasets/processed/binary_suicide_dev.csv   — prompt/k tuning slice drawn from (1) (10/class = 20)

Automation entrypoint:
    python src/builders/build_binary_dataset.py

Re-running with the same inputs and seeds must produce identical processed
CSVs and metadata.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

import pandas as pd

_BUILDERS_DIR = Path(__file__).resolve().parent
_SRC = _BUILDERS_DIR.parent
_PROJECT_ROOT = _SRC.parent
for _path in (_SRC, _SRC / "retriever", _BUILDERS_DIR):
    _s = str(_path)
    if _s not in sys.path:
        sys.path.insert(0, _s)

from components.config import (
    BINARY_DEV_META_PATH,
    BINARY_DEV_PER_CLASS,
    BINARY_DEV_SEED,
    BINARY_DEV_SLICE_PATH,
    BINARY_EVAL_META_PATH,
    BINARY_EVAL_PER_CLASS,
    BINARY_EVAL_SEED,
    BINARY_EVAL_SUBSET_PATH,
    BINARY_LABELS,
    BINARY_MAX_CHARS,
    BINARY_MIN_CHARS,
    BINARY_SOURCE_PATH,
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


def load_raw_frame() -> tuple[pd.DataFrame, dict]:
    if not BINARY_SOURCE_PATH.exists():
        raise FileNotFoundError(
            f"Binary source CSV not found: {BINARY_SOURCE_PATH}. "
            "Place the raw source file under datasets/raw/."
        )
    print(f"Loading local CSV:\n  {BINARY_SOURCE_PATH}")
    raw = pd.read_csv(BINARY_SOURCE_PATH)
    if "text" not in raw.columns or "class" not in raw.columns:
        raise ValueError(
            f"Expected columns 'text' and 'class'; got {list(raw.columns)}"
        )

    out = pd.DataFrame(
        {
            "text": raw["text"].astype(str),
            "label": raw["class"].astype(str).str.strip().str.lower(),
            "source_split": "source",
        }
    )
    stats = {
        "load_mode": "local",
        "source_input": _file_info(BINARY_SOURCE_PATH),
        "rows_raw": int(len(out)),
        "label_counts_raw": out["label"].value_counts().to_dict(),
    }
    return out, stats


def clean_text(text: str) -> str:
    text = text.strip()
    text = _URL_RE.sub(" ", text)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


def filter_and_clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    stats: dict = {}
    keep = {label.lower() for label in BINARY_LABELS}

    df = df[df["label"].isin(keep)].copy()
    stats["rows_after_label_filter"] = int(len(df))

    df["text"] = df["text"].map(clean_text)
    before_empty = len(df)
    df = df[df["text"].str.len() > 0].copy()
    stats["rows_dropped_empty"] = int(before_empty - len(df))

    before_len = len(df)
    df = df[
        (df["text"].str.len() >= BINARY_MIN_CHARS)
        & (df["text"].str.len() <= BINARY_MAX_CHARS)
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
    for label in BINARY_LABELS:
        pool = df[df["label"] == label]
        n_available = len(pool)
        if n_available < per_class:
            raise RuntimeError(
                f"Not enough rows for label '{label}': "
                f"need {per_class}, have {n_available} after filtering."
            )
        sort_cols = [c for c in ("source_split", "text", "row_id") if c in pool.columns]
        pool = pool.sort_values(sort_cols, kind="mergesort")
        sampled = pool.sample(n=per_class, random_state=seed)
        parts.append(sampled)

    out = pd.concat(parts, ignore_index=True)
    label_order = {label: i for i, label in enumerate(BINARY_LABELS)}
    out["_label_order"] = out["label"].map(label_order)
    out = out.sort_values(["_label_order", "text"], kind="mergesort").drop(
        columns=["_label_order"]
    )
    out = out.reset_index(drop=True)

    if "row_id" in out.columns and id_prefix == "bdev":
        out = out.rename(columns={"row_id": "parent_row_id"})
        out.insert(0, "row_id", [f"bdev_{i:04d}" for i in range(len(out))])
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
    meta_path.parent.mkdir(parents=True, exist_ok=True)
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
    raw, load_stats = load_raw_frame()
    cleaned, clean_stats = filter_and_clean(raw)

    subset = stratified_sample(
        cleaned,
        per_class=BINARY_EVAL_PER_CLASS,
        seed=BINARY_EVAL_SEED,
        id_prefix="bin",
    )
    eval_meta = {
        "artifact": "binary_suicide_eval",
        "task": "binary_suicide_classification",
        "seed": BINARY_EVAL_SEED,
        "per_class": BINARY_EVAL_PER_CLASS,
        "labels": list(BINARY_LABELS),
        "min_chars": BINARY_MIN_CHARS,
        "max_chars": BINARY_MAX_CHARS,
        **load_stats,
        **clean_stats,
    }
    write_csv_and_meta(
        subset, BINARY_EVAL_SUBSET_PATH, BINARY_EVAL_META_PATH, eval_meta
    )
    print("Final eval label counts:\n" + subset["label"].value_counts().to_string())

    dev = stratified_sample(
        subset,
        per_class=BINARY_DEV_PER_CLASS,
        seed=BINARY_DEV_SEED,
        id_prefix="bdev",
    )
    dev_meta = {
        "artifact": "binary_suicide_dev",
        "task": "binary_suicide_classification",
        "seed": BINARY_DEV_SEED,
        "per_class": BINARY_DEV_PER_CLASS,
        "labels": list(BINARY_LABELS),
        "parent_csv": str(BINARY_EVAL_SUBSET_PATH),
        "parent_sha256": eval_meta["csv_sha256"],
        "parent_row_ids": dev["parent_row_id"].tolist(),
        "purpose": "prompt/k tuning only; switch notebook eval_mode to final for reporting",
    }
    write_csv_and_meta(dev, BINARY_DEV_SLICE_PATH, BINARY_DEV_META_PATH, dev_meta)
    print("Dev slice label counts:\n" + dev["label"].value_counts().to_string())
    return subset, dev


def main() -> None:
    build()


if __name__ == "__main__":
    main()
