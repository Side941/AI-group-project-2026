"""
retriever_comparison.py
=======================
Empirical retriever selection for the suicide/depression/normal task:
BM25 vs dense (BioLORD) vs hybrid (RRF fusion), compared on the dev split.

Metrics
-------
Per retriever, per class:
  - suicid* chunk hit rate:   posts whose top-k contains >=1 chunk mentioning
                              suicid* (the safety-critical retrieval target)
  - MRR(suicid*):             mean reciprocal rank of the FIRST suicid* chunk
                              (0 when none retrieved) — rewards ranking it high
  - disorder-code profile:    which disorders dominate the retrieved chunks
Per retriever:
  - init time / approx. incremental memory (peak-RSS deltas; coarse)
  - query latency mean / p95 (dense includes query encoding — that is the
    real deployment cost of a local dense retriever)
  - score decisiveness: median (top1 - topk)/top1 margin WITHIN the retriever.
    NOTE: raw scores are NOT comparable across retrievers (BM25 term-weight
    sums vs cosine similarity vs RRF rank sums) — only rank-based metrics are
    compared across systems.
Cross-retriever:
  - mean pairwise top-k overlap (Jaccard on chunk ids) per class — low overlap
    between BM25 and dense is the empirical case for hybrid fusion.

All retrieval is flat (expand=False) at fixed k so every system returns the
same number of chunks and differences are attributable to ranking alone.

Usage (from the repo root, after ChromaDB has been built by the pipeline):

    python analysis/retriever_comparison.py

Writes analysis/retriever_comparison_report.txt. No Ollama needed.
"""

from __future__ import annotations

import statistics
import sys
import time
from collections import Counter
from itertools import combinations
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "retriever"))

from src.components.config import (  # noqa: E402
    CHROMA_PATH,
    CHUNKS_PATH,
    MOOD_DISORDER_PREFIXES,
    MULTICLASS_DEV_PATH,
    RETRIEVAL_SECTIONS,
)
from utils import (  # noqa: E402
    load_chunks,
    filter_chunks_by_disorder_codes,
    filter_chunks_by_sections,
)
from bm25_retriever import BM25Retriever  # noqa: E402

TOP_K = 5
OUT_PATH = ROOT / "analysis" / "retriever_comparison_report.txt"
CLASSES = ["suicidal", "depression", "normal"]

_lines: list[str] = []


def emit(text: str = "") -> None:
    print(text)
    _lines.append(text)


def rule(title: str) -> None:
    emit()
    emit("=" * 78)
    emit(title)
    emit("=" * 78)


# Peak-RSS probe: POSIX getrusage on Linux/macOS, Win32 GetProcessMemoryInfo
# on Windows (the `resource` module does not exist there).
if sys.platform == "win32":
    import ctypes
    from ctypes import wintypes

    class _PROCESS_MEMORY_COUNTERS(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _psapi = ctypes.WinDLL("psapi", use_last_error=True)
    _psapi.GetProcessMemoryInfo.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(_PROCESS_MEMORY_COUNTERS),
        wintypes.DWORD,
    ]
    _psapi.GetProcessMemoryInfo.restype = wintypes.BOOL

    def peak_rss_mb() -> float:
        """Peak working set (Windows' peak RSS equivalent) of this process in MB."""
        counters = _PROCESS_MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(_PROCESS_MEMORY_COUNTERS)
        ok = _psapi.GetProcessMemoryInfo(
            _kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb
        )
        if not ok:
            raise ctypes.WinError(ctypes.get_last_error())
        return counters.PeakWorkingSetSize / (1024 ** 2)
else:
    import resource

    def peak_rss_mb() -> float:
        """Peak RSS of this process in MB (ru_maxrss: bytes on macOS, KB on Linux)."""
        raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return raw / (1024 ** 2) if sys.platform == "darwin" else raw / 1024


def is_suicid(chunk: dict) -> bool:
    return "suicid" in ((chunk.get("prompt_text") or chunk.get("text") or "").lower())


def flat_search(retriever, query: str, k: int):
    try:
        return retriever.search(query, k=k, expand=False)
    except TypeError:  # older signature without expand kwarg
        return retriever.search(query, k=k)


# ── Setup ─────────────────────────────────────────────────────────────────────
pool = filter_chunks_by_sections(
    filter_chunks_by_disorder_codes(load_chunks(CHUNKS_PATH),
                                    MOOD_DISORDER_PREFIXES),
    RETRIEVAL_SECTIONS,
)
df = pd.read_csv(MULTICLASS_DEV_PATH)

rule(f"RETRIEVER COMPARISON — dev split (n={len(df)}), flat top-{TOP_K}")
emit(f"Pool: {len(pool)} chunks | sections {RETRIEVAL_SECTIONS} | "
     f"suicid* chunks in pool: {sum(map(is_suicid, pool))}")
emit("(Caveat: retriever-visible id collisions in KB v1 cap what any system")
emit(" can return — see KNOWN_ISSUES.md. All systems face the same cap, so")
emit(" the comparison between them remains fair.)")

# ── Init each retriever, measuring time and incremental peak memory ───────────
retrievers: dict[str, object] = {}
init_stats: dict[str, tuple[float, float]] = {}

mem0 = peak_rss_mb()
t0 = time.perf_counter()
retrievers["bm25"] = BM25Retriever(chunks=pool, sections=RETRIEVAL_SECTIONS)
flat_search(retrievers["bm25"], "warmup query", TOP_K)
init_stats["bm25"] = (time.perf_counter() - t0, peak_rss_mb() - mem0)

try:
    from dense_retriever import initialise_retrieval, DenseRetriever  # noqa: E402
    mem0 = peak_rss_mb()
    t0 = time.perf_counter()
    initialise_retrieval(chroma_path=str(CHROMA_PATH))
    retrievers["dense"] = DenseRetriever(sections=RETRIEVAL_SECTIONS,
                                         json_path=str(CHUNKS_PATH))
    flat_search(retrievers["dense"], "warmup query", TOP_K)  # loads encoder
    init_stats["dense"] = (time.perf_counter() - t0, peak_rss_mb() - mem0)

    from hybrid_retriever import HybridRetriever  # noqa: E402
    mem0 = peak_rss_mb()
    t0 = time.perf_counter()
    retrievers["hybrid"] = HybridRetriever(chunks=pool, alpha=0.3,
                                           sections=RETRIEVAL_SECTIONS)
    flat_search(retrievers["hybrid"], "warmup query", TOP_K)
    init_stats["hybrid"] = (time.perf_counter() - t0, peak_rss_mb() - mem0)
except Exception as e:
    emit(f"\n[dense/hybrid unavailable: {type(e).__name__}: {e} — BM25 only]")

emit()
emit(f"  {'retriever':<10}{'init+warmup (s)':>17}{'approx. +memory (MB)':>22}")
for name, (secs, mb) in init_stats.items():
    emit(f"  {name:<10}{secs:>17.2f}{mb:>22.1f}")
emit("  (memory = peak-RSS delta around init+first query; coarse, order-")
emit("   dependent, but captures the encoder cost that dominates dense.)")

# ── Run all queries through all retrievers ────────────────────────────────────
results: dict[str, list[dict]] = {name: [] for name in retrievers}

for _, row in df.iterrows():
    text, cls = str(row["text"]), row["label"]
    for name, retriever in retrievers.items():
        t0 = time.perf_counter()
        hits = flat_search(retriever, text, TOP_K)
        latency = time.perf_counter() - t0
        first_suic = next((i + 1 for i, h in enumerate(hits) if is_suicid(h)), 0)
        score_field = {"bm25": "bm25_score", "dense": "similarity",
                       "hybrid": "hybrid_score"}[name]
        scores = [h.get(score_field) for h in hits
                  if isinstance(h.get(score_field), (int, float))]
        results[name].append({
            "class": cls,
            "ids": [h.get("id") for h in hits],
            "codes": [h.get("disorder_code") for h in hits],
            "suic_hit": first_suic > 0,
            "rr": 1.0 / first_suic if first_suic else 0.0,
            "latency": latency,
            "margin": ((scores[0] - scores[-1]) / scores[0]
                       if len(scores) >= 2 and scores[0] else None),
        })

# ── Per-class retrieval quality ───────────────────────────────────────────────
rule("PART 1 — SUICID* CHUNK RETRIEVAL (safety-critical target)")
emit(f"  {'retriever':<10}{'class':<12}{'hit rate':>10}{'MRR(suicid*)':>14}")
for name in retrievers:
    for cls in CLASSES:
        rows = [r for r in results[name] if r["class"] == cls]
        hits = sum(r["suic_hit"] for r in rows)
        mrr = statistics.mean(r["rr"] for r in rows) if rows else 0.0
        emit(f"  {name:<10}{cls:<12}{f'{hits}/{len(rows)}':>10}{mrr:>14.3f}")
    emit()

rule("PART 2 — WHAT EACH RETRIEVER RETURNS (disorder-code profile, all classes)")
for name in retrievers:
    prof = Counter()
    for r in results[name]:
        prof.update(c for c in r["codes"] if c)
    top = ", ".join(f"{c}({n})" for c, n in prof.most_common(6))
    emit(f"  {name:<10}: {top}")

# ── Latency and decisiveness ──────────────────────────────────────────────────
rule("PART 3 — QUERY LATENCY AND WITHIN-RETRIEVER SCORE SHAPE")
emit(f"  {'retriever':<10}{'mean (ms)':>12}{'p95 (ms)':>11}"
     f"{'median top1->top5 margin':>27}")
for name in retrievers:
    lats = sorted(r["latency"] * 1000 for r in results[name])
    margins = [r["margin"] for r in results[name] if r["margin"] is not None]
    p95 = lats[max(0, int(0.95 * len(lats)) - 1)] if lats else 0.0
    med_margin = statistics.median(margins) if margins else float("nan")
    emit(f"  {name:<10}{statistics.mean(lats):>12.1f}{p95:>11.1f}"
         f"{med_margin:>26.0%}")
emit("  Margin = (top1 - top5) / top1 within each retriever's own score scale.")
emit("  Raw scores are NOT comparable across retrievers (term-weight sums vs")
emit("  cosine similarity vs RRF rank sums); only rank-based metrics above are.")

# ── Cross-retriever agreement ─────────────────────────────────────────────────
if len(retrievers) > 1:
    rule(f"PART 4 — TOP-{TOP_K} OVERLAP BETWEEN RETRIEVERS (mean Jaccard)")
    emit(f"  {'pair':<16}" + "".join(f"{c:>12}" for c in CLASSES) + f"{'all':>12}")
    for a, b in combinations(retrievers, 2):
        cells = []
        for cls in CLASSES + [None]:
            pairs = [(ra, rb) for ra, rb in zip(results[a], results[b])
                     if cls is None or ra["class"] == cls]
            jac = [len(set(ra["ids"]) & set(rb["ids"]))
                   / max(len(set(ra["ids"]) | set(rb["ids"])), 1)
                   for ra, rb in pairs]
            cells.append(statistics.mean(jac) if jac else 0.0)
        emit(f"  {a + ' vs ' + b:<16}" + "".join(f"{v:>12.2f}" for v in cells))
    emit("  Low BM25-vs-dense overlap = the two see different evidence = the")
    emit("  empirical case for hybrid fusion; high overlap = hybrid adds little.")

# ── Summary ───────────────────────────────────────────────────────────────────
rule("PART 5 — SUMMARY: WHICH RETRIEVER FOR THIS TASK?")
emit("Decision inputs, in order of importance for this system:")
emit("  1. suicid* hit rate + MRR on suicidal-class posts (Part 1) — the")
emit("     safety-critical retrieval target this pipeline exists for.")
emit("  2. Latency and memory (Part 3 / init table) — the system is pitched")
emit("     as lightweight and locally deployable; dense pays encoder cost")
emit("     per query, BM25 is effectively free.")
emit("  3. BM25-dense overlap (Part 4) — justifies (or argues against) the")
emit("     added complexity of hybrid fusion.")
emit("Interpret with n=10 per class: report counts (x/10), not percentages,")
emit("and treat differences of 1-2 posts as noise. The same script pointed at")
emit("the eval split gives confirmatory numbers at higher n.")
emit("Whatever wins here selects the retriever for the final configuration;")
emit("the classification-level effect is then confirmed by the RQ-grid runs.")

emit()
OUT_PATH.parent.mkdir(exist_ok=True)
OUT_PATH.write_text("\n".join(_lines) + "\n", encoding="utf-8")
print(f"\nReport saved to {OUT_PATH}")
