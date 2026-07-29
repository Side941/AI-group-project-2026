"""
page_boundary_experiment.py
===========================
Is the chunking page range (CONTENT_START_PAGE=109, CONTENT_END_PAGE=694)
optimal — in particular, does clinically useful suicide/self-harm content
exist just past the end boundary (the MB symptom appendix, ~PDF 695+), and
would it be correctly attributed and retrievable if included?

For each candidate boundary, the PDF is chunked to a TEMPORARY json under
analysis/page_boundary_runs/ — the committed knowledge_base JSON and the
ChromaDB store are never touched (KB freeze respected).

Per variant, compared against a freshly-chunked baseline (same code path):
  1. chunk-count delta and which (code, section) chunks were added/removed
  2. suicid*/self-harm mentions among ADDED chunks, with text previews
  3. attribution audit: are added chunks under plausible codes, or absorbed
     into the tail code of the previous range (the episode-bug failure mode —
     the chunker's code regex does not recognise MB symptom codes)
  4. pool impact: do any added chunks pass the CURRENT retrieval filter
     (mood prefixes x retrieval sections), i.e. would retrieval change today?

Usage (from the repo root; needs the local PDF + pdftotext, no Ollama):

    python analysis/page_boundary_experiment.py
    python analysis/page_boundary_experiment.py --ends 700 710 724 --starts 95

Writes analysis/page_boundary_report.txt.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "retriever"))

from src.components.config import (  # noqa: E402
    CONTENT_END_PAGE,
    CONTENT_START_PAGE,
    MOOD_DISORDER_PREFIXES,
    PDF_PATH,
    RETRIEVAL_SECTIONS,
)
from src.components.chunker import run_chunking  # noqa: E402

SUIC_RE = re.compile(r"suicid|self[- ]harm|self[- ]injur", re.IGNORECASE)

RUN_DIR = ROOT / "analysis" / "page_boundary_runs"
OUT_PATH = ROOT / "analysis" / "page_boundary_report.txt"

_lines: list[str] = []


def emit(text: str = "") -> None:
    print(text)
    _lines.append(text)


def rule(title: str) -> None:
    emit()
    emit("=" * 78)
    emit(title)
    emit("=" * 78)


def key(c: dict) -> tuple:
    return (c.get("disorder_code"), c.get("section"), c.get("chunk_part"))


def in_current_pool(c: dict) -> bool:
    return (str(c.get("disorder_code", "")).startswith(tuple(MOOD_DISORDER_PREFIXES))
            and c.get("section") in RETRIEVAL_SECTIONS)


def chunk_variant(name: str, start: int, end: int) -> list[dict]:
    out = RUN_DIR / f"chunks_{name}.json"
    print(f"\n--- chunking variant {name}: pages {start}-{end} -> {out.name} ---")
    return run_chunking(
        pdf_path=str(PDF_PATH),
        chunks_path=str(out),
        start_page=start,
        end_page=end,
    )


def analyse_variant(name: str, start: int, end: int,
                    base: list[dict], var: list[dict]) -> None:
    rule(f"VARIANT {name}: pages {start}-{end}  "
         f"(baseline {CONTENT_START_PAGE}-{CONTENT_END_PAGE})")

    base_keys = {key(c): c for c in base}
    var_keys = {key(c): c for c in var}
    added = [var_keys[k] for k in var_keys.keys() - base_keys.keys()]
    removed = [base_keys[k] for k in base_keys.keys() - var_keys.keys()]

    emit(f"chunks: baseline {len(base)} -> variant {len(var)} "
         f"({len(added):+d} added, {len(removed)} removed)")

    if removed:
        prof = Counter(c.get("disorder_code") for c in removed)
        emit(f"removed, by code: {dict(prof.most_common(8))}")

    if not added:
        emit("no chunks added — the extra pages contribute nothing "
             "under the current chunker.")
        return

    # 1. who owns the new content?
    prof = Counter(c.get("disorder_code") for c in added)
    emit(f"added, by code: {dict(prof.most_common(8))}")
    top_code, top_n = prof.most_common(1)[0]
    if top_n / len(added) > 0.5:
        emit(f"ATTRIBUTION WARNING: {top_n}/{len(added)} added chunks fell "
             f"under a single code ({top_code}).")
        emit("  The chunker's code regex does not recognise MB symptom codes,")
        emit("  so appendix content is likely being absorbed by the last coded")
        emit("  entry — same failure mode as the 6A8Z episode issue")
        emit("  (KNOWN_ISSUES.md). Including these pages without a chunker")
        emit("  change would add mislabelled chunks, not usable ones.")

    # 2. suicide/self-harm content among added chunks
    suic_added = [c for c in added if SUIC_RE.search(c.get("text", ""))]
    emit(f"added chunks mentioning suicid*/self-harm: {len(suic_added)}")
    for c in suic_added[:8]:
        emit(f"  {c.get('disorder_code'):10} {str(c.get('disorder_name'))[:30]:30} "
             f"{c.get('section')}")
        emit(f"      {c.get('text', '')[:150]!r}")

    # 3. would retrieval change TODAY, under the current pool filter?
    pool_added = [c for c in added if in_current_pool(c)]
    emit(f"added chunks passing the CURRENT retrieval filter "
         f"(mood prefixes x {RETRIEVAL_SECTIONS}): {len(pool_added)}")
    if not pool_added:
        emit("  -> retrieval pool unchanged: with the current filter, these")
        emit("     pages cannot affect the live system either way. Any benefit")
        emit("     would additionally require widening the pool filter (and,")
        emit("     for MB content, teaching the chunker MB codes).")
    else:
        for c in pool_added[:6]:
            emit(f"  {c.get('disorder_code'):10} {c.get('section')} | "
                 f"{c.get('text', '')[:100]!r}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ends", type=int, nargs="*", default=[700, 710, 724],
                        help="candidate CONTENT_END_PAGE values")
    parser.add_argument("--starts", type=int, nargs="*", default=[95],
                        help="candidate CONTENT_START_PAGE values")
    args = parser.parse_args()

    if not Path(PDF_PATH).exists():
        sys.exit(f"PDF not found at {PDF_PATH} — this experiment needs the "
                 f"local CDDR PDF (gitignored).")
    RUN_DIR.mkdir(parents=True, exist_ok=True)

    rule("PAGE BOUNDARY EXPERIMENT")
    emit(f"Baseline range: {CONTENT_START_PAGE}-{CONTENT_END_PAGE} "
         f"(config comment: CDDR clinical content; MB symptom appendix ~695+)")
    emit(f"End candidates:   {args.ends}")
    emit(f"Start candidates: {args.starts}")
    emit("All variant chunk files are written under analysis/page_boundary_runs/")
    emit("— the committed KB and ChromaDB are not modified.")

    base = chunk_variant("baseline",
                         CONTENT_START_PAGE, CONTENT_END_PAGE)

    for end in args.ends:
        if end == CONTENT_END_PAGE:
            continue
        var = chunk_variant(f"end{end}", CONTENT_START_PAGE, end)
        analyse_variant(f"end={end}", CONTENT_START_PAGE, end, base, var)

    for start in args.starts:
        if start == CONTENT_START_PAGE:
            continue
        var = chunk_variant(f"start{start}", start, CONTENT_END_PAGE)
        analyse_variant(f"start={start}", start, CONTENT_END_PAGE, base, var)

    rule("CONCLUSION TEMPLATE")
    emit("The boundary question decomposes into three findings per variant:")
    emit("  (a) does extra content exist?          (chunk delta)")
    emit("  (b) is it suicide/self-harm relevant?  (suicid* previews)")
    emit("  (c) is it usable as-is?                (attribution + pool filter)")
    emit("A recommendation to move the boundary is justified only when (a),")
    emit("(b) AND (c) hold — otherwise the current 109-694 range stands, with")
    emit("(b)-without-(c) cases logged as future work requiring chunker/filter")
    emit("changes (KB v2 territory).")

    emit()
    OUT_PATH.write_text("\n".join(_lines) + "\n", encoding="utf-8")
    print(f"\nReport saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
