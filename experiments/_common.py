"""
Shared helpers for small pipeline smoke experiments.

Usage from any script in experiments/:
    from _common import bootstrap, SAMPLE_QUERIES, print_hits, load_mood_chunks
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def bootstrap() -> Path:
    """Put src/ and src/retriever/ on sys.path; return project root."""
    src = PROJECT_ROOT / "src"
    for path in (src, src / "retriever"):
        s = str(path)
        if s not in sys.path:
            sys.path.insert(0, s)
    return PROJECT_ROOT


def print_hits(hits: list[dict], *, score_key: str | None = None, limit: int = 5) -> None:
    """Pretty-print retrieval hits (ICD-11 chunks or few-shot examples)."""
    if not hits:
        print("  (no hits)")
        return

    for i, hit in enumerate(hits[:limit], start=1):
        score = None
        if score_key and score_key in hit:
            score = hit[score_key]
        else:
            for key in ("hybrid_score", "dense_score", "bm25_score", "score"):
                if key in hit:
                    score = hit[key]
                    break

        if "disorder_code" in hit or "disorder_name" in hit:
            label = (
                f"{hit.get('disorder_code', '?')} | "
                f"{hit.get('disorder_name', '?')} | "
                f"{hit.get('section', '?')}"
            )
            text = (hit.get("prompt_text") or hit.get("text") or "").replace("\n", " ")
        else:
            label = f"{hit.get('label', '?')} | {hit.get('head', 'fewshot')}"
            text = (hit.get("post") or hit.get("text") or "").replace("\n", " ")

        score_s = f"  score={score:.4f}" if isinstance(score, (int, float)) else ""
        print(f"  [{i}]{score_s}  {label}")
        print(f"      {text[:160]}{'…' if len(text) > 160 else ''}")


def load_mood_chunks():
    """ICD-11 chunks filtered to mood disorders + retrieval sections (notebook policy)."""
    from components.config import MOOD_DISORDER_PREFIXES, RETRIEVAL_SECTIONS
    from utils import (
        filter_chunks_by_disorder_codes,
        filter_chunks_by_sections,
        load_chunks,
    )

    chunks = load_chunks()
    chunks = filter_chunks_by_disorder_codes(chunks, MOOD_DISORDER_PREFIXES)
    chunks = filter_chunks_by_sections(chunks, RETRIEVAL_SECTIONS)
    return chunks


SAMPLE_QUERIES: dict[str, str] = {
    "depression": (
        "I've been sad most days, nothing feels enjoyable anymore, "
        "and I feel worthless even though I wouldn't hurt myself."
    ),
    "suicidal": (
        "I don't see the point in living anymore. "
        "I've been thinking about ending it and making a plan."
    ),
    "normal": (
        "Had a long day at work but hanging out with friends tonight. "
        "Feeling okay overall."
    ),
}
