"""
rebuild_all_artifacts.py
========================
Rebuild the processed multiclass evaluation dataset and the few-shot store.

Blueprint alignment:
- Multiclass head uses the HF mental-health dataset.
- Few-shot examples support dynamic few-shot prompts at classification time.

Few-shot store layout:
- multiclass_examples.json: the full auto-sampled pool. It is still exported
  here exactly as before (same cleaning, eval holdout, seeding and per-label
  caps) but it is NOT embedded or used at retrieval time - it is kept only as
  the pool the curated examples were picked from.
- multiclass_examples_curated.json: hand-curated subset (see
  scripts/make_curated_fewshot.py). This is what BM25 loads and what
  build_curated_fewshot_db.py embeds into the Chroma collection used by
  dense retrieval.

Important:
- Eval posts (and therefore the nested dev slice) are held out of the few-shot
  pool so retrieved examples cannot leak into evaluation.

Run:
    python src/builders/rebuild_all_artifacts.py
"""

from __future__ import annotations

import argparse
import json
import random
import runpy
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

from components.config import (  # noqa: E402
    EMBEDDING_MODEL,
    FEWSHOT_BUILD_SEED,
    FEWSHOT_DB_META_PATH,
    FEWSHOT_MULTICLASS_EXAMPLES_PATH,
    FEWSHOT_MULTICLASS_MAX_PER_LABEL,
    RAG_EVAL_LABELS,
    RAG_EVAL_SUBSET_PATH,
)
import build_curated_fewshot_db as curated_builder  # noqa: E402
import build_multiclass_dataset as multiclass_builder  # noqa: E402


def _eval_holdout_keys() -> set[str]:
    """Lowercased texts from the committed eval split (dev is nested inside it)."""
    if not RAG_EVAL_SUBSET_PATH.exists():
        raise FileNotFoundError(
            f"Eval CSV required to hold out leakage from few-shot pool: {RAG_EVAL_SUBSET_PATH}. "
            "Run without --skip-evals first, or build with: python src/builders/build_multiclass_dataset.py"
        )
    eval_df = pd.read_csv(RAG_EVAL_SUBSET_PATH)
    return set(eval_df["text"].astype(str).str.strip().str.lower())


def _build_multiclass_fewshot_pool() -> tuple[pd.DataFrame, dict]:
    raw, _ = multiclass_builder.load_raw_frames()
    cleaned, _ = multiclass_builder.filter_and_clean(raw)
    holdout = _eval_holdout_keys()
    before = len(cleaned)
    pool = cleaned[~cleaned["text"].astype(str).str.strip().str.lower().isin(holdout)].copy()
    stats = {
        "cleaned_pool_size": int(before),
        "eval_holdout_size": int(len(holdout)),
        "fewshot_pool_size": int(len(pool)),
        "rows_dropped_eval_holdout": int(before - len(pool)),
    }
    print(
        f"Few-shot pool after eval holdout: {stats['fewshot_pool_size']} "
        f"(dropped {stats['rows_dropped_eval_holdout']} eval texts)"
    )
    return pool.reset_index(drop=True), stats


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


def _export_examples_json(
    frame: pd.DataFrame,
    *,
    path: Path,
    id_prefix: str,
) -> list[dict]:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for idx, row in frame.reset_index(drop=True).iterrows():
        rows.append(
            {
                "id": f"{id_prefix}_{idx:06d}",
                "text": row["text"],
                "label": row["label"],
                "source_split": row.get("source_split", "source"),
            }
        )
    path.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {path} ({len(rows)} examples)")
    return rows


def build_fewshot_db(*, rebuild: bool, seed: int) -> dict:
    # Build pool from raw multiclass data using the same cleaning/filter pipeline
    # as the processed eval/dev builder, then hold out eval texts to prevent leakage.
    print("\nBuilding few-shot pool from raw multiclass data via the shared preprocessing pipeline …")
    multi_pool, pool_stats = _build_multiclass_fewshot_pool()

    # Sample balanced caps for deterministic size.
    multi_sample = _sample_balanced(
        multi_pool,
        labels=list(RAG_EVAL_LABELS),
        per_label=FEWSHOT_MULTICLASS_MAX_PER_LABEL,
        seed=seed,
    )

    # Export the full pool JSON exactly as before. It is kept only as the pool
    # the curated examples were picked from; nothing loads it at runtime.
    multi_rows = _export_examples_json(
        multi_sample,
        path=FEWSHOT_MULTICLASS_EXAMPLES_PATH,
        id_prefix="mc",
    )

    # The vector DB is built from the hand-curated subset only.
    curated_summary = curated_builder.build_curated_fewshot_collection(rebuild=rebuild)

    meta = {
        "artifact": "fewshot_db",
        "seed": seed,
        "embedding_model": EMBEDDING_MODEL,
        "collections": {
            "multiclass": {
                "collection_name": None,
                "note": "Pool JSON export only; vectors live in the curated collection.",
                "examples_json": str(FEWSHOT_MULTICLASS_EXAMPLES_PATH),
                "per_label": FEWSHOT_MULTICLASS_MAX_PER_LABEL,
                "labels": list(RAG_EVAL_LABELS),
                "total_vectors": 0,
                "pool_rows": int(len(multi_rows)),
            },
            "multiclass_curated": curated_summary,
        },
        "multiclass_raw_pool_size": pool_stats["cleaned_pool_size"],
        "eval_holdout_size": pool_stats["eval_holdout_size"],
        "fewshot_pool_size_after_holdout": pool_stats["fewshot_pool_size"],
        "rows_dropped_eval_holdout": pool_stats["rows_dropped_eval_holdout"],
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

    # CLI seed currently only affects few-shot sampling because the eval/dev
    # builder hardcodes its own seeds via config constants. If you want them
    # to follow a different seed, edit config and re-run.
    print("=== Rebuild eval/dev ===")
    if args.skip_evals:
        print("Skipping eval/dev rebuild.")
    else:
        runpy.run_path(str(_BUILDERS_DIR / "build_multiclass_dataset.py"), run_name="__main__")

    # Always build few-shot DB (this is the main request).
    print("=== Build few-shot vector DB ===")
    build_fewshot_db(rebuild=not bool(args.reuse_fewshot_db), seed=int(args.seed))

    print("\nDone.")


if __name__ == "__main__":
    main()
