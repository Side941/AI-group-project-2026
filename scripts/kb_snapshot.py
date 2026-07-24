"""
kb_snapshot.py
==============
Snapshot what is currently CHUNKED (knowledge-base inventory) and what is
currently RETRIEVED (per-retriever results for fixed sample queries), so the
state of the pipeline can be compared before and after a fix.

Usage (from the repo root):

    python scripts/kb_snapshot.py before      # run on the current state
    ... apply fix, rm -rf knowledge_base/chroma_db, re-run components.main ...
    python scripts/kb_snapshot.py after       # run on the fixed state

    diff kb_snapshot_before.txt kb_snapshot_after.txt   # or open side by side

Works with both the pre-fix and post-fix code: it only uses the public
retriever APIs and degrades gracefully if the dense index is unavailable.
No Jupyter and no Ollama required.
"""

from __future__ import annotations

import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "retriever"))

from src.components.config import (  # noqa: E402
    CHUNKS_PATH,
    CHROMA_PATH,
    MOOD_DISORDER_PREFIXES,
    RETRIEVAL_SECTIONS,
)
from utils import (  # noqa: E402
    load_chunks,
    filter_chunks_by_disorder_codes,
    filter_chunks_by_sections,
)

LABEL = sys.argv[1] if len(sys.argv) > 1 else "snapshot"
OUT_PATH = ROOT / f"kb_snapshot_{LABEL}.txt"
TOP_K = 5

# One query per class, fixed so before/after runs are directly comparable.
SAMPLE_QUERIES = {
    "suicidal":   ("I keep thinking about ending my life. I feel hopeless "
                   "and worthless and I don't see the point in going on."),
    "depression": ("I've been sad most days for weeks, can't sleep, lost "
                   "interest in everything I used to enjoy, no energy at all."),
    "normal":     ("Work was stressful this week but I'm looking forward to "
                   "the weekend with friends, feeling okay overall."),
}

_lines: list[str] = []


def emit(text: str = "") -> None:
    print(text)
    _lines.append(text)


def rule(title: str) -> None:
    emit()
    emit("=" * 78)
    emit(title)
    emit("=" * 78)


def chunk_label(c: dict) -> str:
    part = c.get("chunk_part")
    part_s = f" p{part}" if part else ""
    return (f"{c.get('disorder_code', '?'):10} "
            f"{(c.get('disorder_name') or '?')[:34]:34} "
            f"{c.get('section', '?'):24}{part_s}")


def preview(c: dict, width: int = 100) -> str:
    text = (c.get("prompt_text") or c.get("text") or "").strip()
    return text[:width].replace("\n", " ")


# ── Part 1: what is chunked ───────────────────────────────────────────────────
rule(f"PART 1 — KNOWLEDGE BASE INVENTORY  ({LABEL})")

all_chunks = load_chunks(CHUNKS_PATH)
emit(f"Total chunks in {Path(str(CHUNKS_PATH)).name}: {len(all_chunks)}")
emit(f"chunks carrying a chunk_uid field: "
     f"{sum(1 for c in all_chunks if c.get('chunk_uid'))}")

mood = filter_chunks_by_disorder_codes(all_chunks, MOOD_DISORDER_PREFIXES)
pool = filter_chunks_by_sections(mood, RETRIEVAL_SECTIONS)
emit(f"Mood-disorder chunks (prefixes {tuple(MOOD_DISORDER_PREFIXES)}): {len(mood)}")
emit(f"Retrieval pool (sections {RETRIEVAL_SECTIONS}): {len(pool)} chunks")

emit()
emit("Retrieval pool by disorder:")
by_disorder: dict[tuple, list] = defaultdict(list)
for c in pool:
    by_disorder[(c.get("disorder_code"), c.get("disorder_name"))].append(c)
for (code, name), rows in sorted(by_disorder.items()):
    secs = Counter(r.get("section") for r in rows)
    sec_s = ", ".join(f"{s} x{n}" if n > 1 else s for s, n in sorted(secs.items()))
    emit(f"  {code:10} {str(name)[:38]:38} {len(rows):>2} chunks  [{sec_s}]")

emit()
emit("Retriever-visible id collisions in the pool "
     "(chunks sharing one id — only ONE of each group is ever retrievable):")
by_id: dict[str, list] = defaultdict(list)
for c in pool:
    by_id[c.get("id", "?")].append(c)
collided = {i: rows for i, rows in by_id.items() if len(rows) > 1}
if not collided:
    emit("  none — every pool chunk has a unique retriever id")
else:
    hidden = sum(len(r) - 1 for r in collided.values())
    emit(f"  {len(collided)} colliding ids hide {hidden} chunks:")
    for cid, rows in sorted(collided.items()):
        emit(f"  id={cid}  ({len(rows)} chunks share it)")
        for r in rows:
            emit(f"      {chunk_label(r)} | {preview(r, 70)}")

emit()
emit("Where suicid* content lives in the pool:")
suic = [c for c in pool
        if "suicid" in (c.get("prompt_text") or c.get("text") or "").lower()]
if not suic:
    emit("  (none found in the retrieval pool)")
for c in suic:
    emit(f"  {chunk_label(c)}")
    emit(f"      {preview(c)}")

# ── Part 2: what is retrieved ─────────────────────────────────────────────────
rule(f"PART 2 — RETRIEVAL RESULTS, k={TOP_K}  ({LABEL})")

retrievers: dict[str, object] = {}

from bm25_retriever import BM25Retriever  # noqa: E402
retrievers["bm25"] = BM25Retriever(chunks=pool, sections=RETRIEVAL_SECTIONS)

try:
    from dense_retriever import initialise_retrieval, DenseRetriever  # noqa: E402
    initialise_retrieval(chroma_path=str(CHROMA_PATH))
    retrievers["dense"] = DenseRetriever(sections=RETRIEVAL_SECTIONS,
                                         json_path=str(CHUNKS_PATH))
    from hybrid_retriever import HybridRetriever  # noqa: E402
    retrievers["hybrid"] = HybridRetriever(chunks=pool, alpha=0.3,
                                           sections=RETRIEVAL_SECTIONS)
except Exception as e:
    emit(f"[dense/hybrid unavailable: {type(e).__name__}: {e} — "
         f"showing BM25 only]")

for query_class, query in SAMPLE_QUERIES.items():
    emit()
    emit(f'>>> query class "{query_class}": "{query[:70]}..."')
    for name, retriever in retrievers.items():
        try:
            results = retriever.search(query, k=TOP_K)
        except Exception as e:
            emit(f"  [{name}] ERROR {type(e).__name__}: {e}")
            continue
        emit(f"  [{name}] returned {len(results)} chunks:")
        for i, c in enumerate(results, 1):
            emit(f"    [{i}] {chunk_label(c)}")

    # Flat mode (expand off) — count only; shows whether flat honours k.
    for name in ("bm25", "dense"):
        if name not in retrievers:
            continue
        try:
            flat = retrievers[name].search(query, k=TOP_K, expand=False)
            emit(f"  [{name}, expand=False] returned {len(flat)} chunks "
                 f"(requested k={TOP_K})")
        except TypeError:
            emit(f"  [{name}, expand=False] not supported by this code version")
        except Exception as e:
            emit(f"  [{name}, expand=False] ERROR {type(e).__name__}: {e}")

# ── Part 3: what the LLM actually sees ────────────────────────────────────────
rule(f"PART 3 — PROMPT CONTEXT BLOCK (bm25, suicidal query)  ({LABEL})")
emit("This is the clinical-knowledge block exactly as build_prompt() formats it:")
emit()
try:
    results = retrievers["bm25"].search(SAMPLE_QUERIES["suicidal"], k=TOP_K)
    for i, c in enumerate(results):
        emit(f"[{i + 1}] ({c.get('disorder_name', '')} — {c.get('section', '')}): "
             f"{preview(c, 240)}")
except Exception as e:
    emit(f"ERROR {type(e).__name__}: {e}")

emit()
emit("=" * 78)
OUT_PATH.write_text("\n".join(_lines) + "\n", encoding="utf-8")
print(f"\nReport saved to {OUT_PATH}")
