"""
chunker.py
==========
Step 1 of the ICD-11 knowledge base pipeline.

Responsibilities
----------------
- Extract a page range from the ICD-11 CDDR PDF via pdftotext.
- Parse the raw text into structured chunk dicts (one per disorder section).
- Post-process chunks: merge near-empty consecutive chunks, add embed_text.
- Save the final chunks to JSON.

Public API
----------
    run_chunking(pdf_path, chunks_path, start_page, end_page) -> list[dict]
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

try:
    from config import (        # when run directly: python src/components/chunker.py
        CHUNK_COUNT_WARNING_THRESHOLD,
        CONTENT_END_PAGE,
        CONTENT_START_PAGE,
        CHUNKS_PATH,
        PDF_PATH,
        DOMAIN_MAP,
        SECTION_NORMALISE_MAP,
    )
except ModuleNotFoundError:
    from components.config import (   # when imported as a package (src on sys.path)
        CHUNK_COUNT_WARNING_THRESHOLD,
        CONTENT_END_PAGE,
        CONTENT_START_PAGE,
        CHUNKS_PATH,
        PDF_PATH,
        DOMAIN_MAP,
        SECTION_NORMALISE_MAP,
    )

# Max-size guardrails so section-based chunks do not become too large.
MAX_CHUNK_WORDS = 220
CHUNK_WORD_OVERLAP = 35

# ── Compiled regex patterns ────────────────────────────────────────────────────
_SECTION_PATTERNS = [
    r"Essential \(required\) features",
    r"Essential features",
    r"Additional clinical features",
    r"Boundary with normality",
    r"Boundary with normality \(threshold\)",
    r"Course features",
    r"Developmental presentations",
    r"Culture-related features",
    r"Sex- and/or gender-related features",
    # Full heading, unwrapped short form, and layout-wrapped "(differential" line.
    r"Boundaries with other disorders and conditions \(differential diagnosis\)",
    r"Boundaries with other disorders and conditions \(differential",
    r"Boundaries with other disorders and conditions",
    r"General diagnostic requirements(?: for .+)?",
    r"Diagnostic requirements",
    r"Specifiers",
    r"Coded elsewhere",
    r"Note:",
]

SECTION_RE = re.compile(
    r"^(" + "|".join(_SECTION_PATTERNS) + r")\s*$",
    re.IGNORECASE | re.MULTILINE,
)

# Chapter 6 CDDR codes, plus cross-listed 8A (tics) and GA (e.g. PMDD) entries.
DISORDER_CODE_RE = re.compile(
    r"^((?:6[A-Z0-9]{2,5}|8A\d{2}|GA\d{2})(?:\.[A-Z0-9]{1,3})?)\s{2,}(.+)$",
    re.MULTILINE,
)

# ── Mood episode descriptions ─────────────────────────────────────────────────
# The CDDR mood chapter opens with descriptions of the four mood *episodes*
# (depressive, manic, mixed, hypomanic) before any coded disorder entry.
# These headings carry no ICD-11 code, so the plain code-tracking loop
# mis-attributed all of their content to the last code seen in the chapter's
# category listing (6A8Z, "Mood disorder, unspecified"). Episodes are the
# clinical building blocks the disorders are defined in terms of — the
# depressive-episode Essential Features (including the suicidal ideation item)
# are the single most relevant retrieval content for depression/suicide-risk
# classification — so they get their own pseudo-codes under the "EP." prefix.
# Case-sensitive and end-anchored: a layout-wrapped sentence fragment such as
# "hypomanic episode" (lower-case, or mid-sentence) must not match.
EPISODE_HEADING_RE = re.compile(r"^(Depressive|Manic|Mixed|Hypomanic) episode$")
EPISODE_SECTION_MARKER_RE = re.compile(r"^Mood episode descriptions$", re.IGNORECASE)
EPISODE_CODES = {
    "Depressive": "EP.DEP",
    "Manic":      "EP.MAN",
    "Mixed":      "EP.MIX",
    "Hypomanic":  "EP.HYP",
}
EPISODE_DOMAIN = "Mood disorders"

# Continuation of a layout-wrapped differential-diagnosis heading.
DIFF_HEADING_CONTINUATION_RE = re.compile(
    r"^diagnosis\)(?:\s|$)",
    re.IGNORECASE,
)

# Stop parsing when appendix / index material begins (not clinical CDDR entries).
APPENDIX_STOP_RE = re.compile(
    r"(?:"
    # Full phrase, or line-wrapped form ending at "include the"
    r"Mental or behavioural symptoms, signs or clinical findings include the(?:\s+following)?"
    r"|Mental or behavioural symptoms, signs or clinical findings\s+\d{2,3}\s*$"
    r"|Relationship problems and maltreatment as factors influencing health status"
    r"|^Acknowledgements?\b"
    r"|^Site directors?:"
    r"|^Other contributors?:"
    r"|^MB\d{2}(?:\.[A-Z0-9]+)?\s+Symptoms"
    r"|^QA\d{2}"
    r")",
    re.IGNORECASE,
)

CHAPTER_INDEX_RE = re.compile(r"\bList of categories\b", re.IGNORECASE)
ICD10_CROSSWALK_RE = re.compile(r"\bF\d{2}(?:\.\d+)?\b")
MB_SYMPTOM_CODE_RE = re.compile(r"\bMB\d{2}(?:\.[A-Z0-9]+)?\b")
INLINE_DISORDER_CODE_RE = re.compile(
    r"\b(?:6[A-Z0-9]{2,5}|8A\d{2}|GA\d{2})(?:\.[A-Z0-9]{1,3})?\b"
)
NON_CDDR_CODE_RE = re.compile(r"\b(?:QA|PJ|QE|MB|F)\d", re.IGNORECASE)
PAGE_HEADER_BLEED_RE = re.compile(
    r"\b\d{2,3}\s+Clinical Descriptions and Diagnostic Requirements\b",
    re.IGNORECASE,
)

_CHAPTER_BOILERPLATE_MARKERS = (
    "neurodevelopmental disorders include the following",
    "many mental and behavioural disorders that can arise during the developmental period",
    "mental, behavioural and neurodevelopmental disorders are syndromes",
    "mental or behavioural symptoms, signs or clinical findings",
    "schizophrenia and other primary psychotic disorders include",
    "mood disorders include the following",
    "anxiety and fear-related disorders include",
    "relationship problems and maltreatment as factors influencing health status",
    "examination or observation for suspected",
    "intimate partner physical abuse",
    "caregiver-child relationship",
    "caregiver\u2013child relationship",
    "child maltreatment",
    "crosswalk from icd-11",
    "boundary with spouse or partner",
    "intimate partner",
    "spouse or partner",
    "child sexual abuse",
    "child neglect",
    "relationship distress with",
    "significantly affect physical, mental and social well-being",
)


# ── Helper functions ───────────────────────────────────────────────────────────

def get_domain(code: str) -> str:
    """Return the clinical domain for an ICD-11 code prefix."""
    for prefix, domain in DOMAIN_MAP.items():
        if code.startswith(prefix):
            return domain
    return "Other / Unclassified"


def normalise_section(heading: str) -> str:
    """Standardise a raw section heading string."""
    key = heading.strip().lower()
    if key in SECTION_NORMALISE_MAP:
        return SECTION_NORMALISE_MAP[key]
    # Prefix fallbacks for headings that vary by disorder name or wrap in layout.
    if key.startswith("general diagnostic requirements"):
        return "Diagnostic Requirements"
    if key.startswith("boundaries with other disorders and conditions"):
        return "Differential Diagnosis"
    return heading.strip()


def is_appendix_boundary(line: str) -> bool:
    """Return True when a line marks the start of non-clinical appendix content."""
    return bool(APPENDIX_STOP_RE.search(line))


def is_index_bleed_chunk(text: str, section: str) -> bool:
    """Drop chapter-index pages that were mis-attributed to a disorder."""
    return section == "Overview" and CHAPTER_INDEX_RE.search(text) is not None


def is_misattributed_6E6Z_chunk(chunk: dict) -> bool:
    """
    Drop chunks wrongly accumulated under 6E6Z during chapter/index boundaries.

    6E6Z is the last disorder in its chapter; appendix and chapter-intro pages are
    often mis-attributed to it when no new disorder-code line is detected.
    """
    if chunk.get("disorder_code") != "6E6Z":
        return False

    text = chunk.get("text", "")
    text_lower = text.lower()
    if "secondary" not in text_lower:
        if (
            chunk.get("section") == "Overview"
            and "unspecified" in text_lower
            and len(text.split()) <= 80
            and CHAPTER_INDEX_RE.search(text) is None
            and not any(marker in text_lower for marker in _CHAPTER_BOILERPLATE_MARKERS)
        ):
            return False
        return True

    if any(marker in text_lower for marker in _CHAPTER_BOILERPLATE_MARKERS):
        return True

    inline_codes = set(INLINE_DISORDER_CODE_RE.findall(text))
    inline_codes.discard("6E6Z")
    if len(inline_codes) >= 3:
        return True

    if NON_CDDR_CODE_RE.search(text) or PAGE_HEADER_BLEED_RE.search(text):
        return True

    section = chunk.get("section", "")
    if section == "Diagnostic Requirements" and ICD10_CROSSWALK_RE.search(text):
        return True

    return False


def is_junk_chunk(chunk: dict) -> bool:
    """
    Drop appendix tables, ICD-10 crosswalk rows, and contributor lists that
    slipped through line-based parsing.
    """
    text = chunk.get("text", "")
    section = chunk.get("section", "")

    if is_index_bleed_chunk(text, section):
        return True
    if is_misattributed_6E6Z_chunk(chunk):
        return True
    if chunk.get("disorder_code") == "6E6Z" and NON_CDDR_CODE_RE.search(text):
        return True
    if APPENDIX_STOP_RE.search(text):
        return True
    if MB_SYMPTOM_CODE_RE.search(text):
        return True
    if ICD10_CROSSWALK_RE.findall(text) and len(ICD10_CROSSWALK_RE.findall(text)) >= 3:
        return True
    if re.search(
        r"\b(?:National coordinator|Site coordinators?|Site directors?|Other contributors?)\b",
        text,
        re.IGNORECASE,
    ):
        return True
    if re.search(r",\s*Chair\b", text):
        return True
    return False


def filter_junk_chunks(chunks: list[dict]) -> list[dict]:
    """Remove non-clinical chunks produced by PDF layout edge cases."""
    kept = [c for c in chunks if not is_junk_chunk(c)]
    dropped = len(chunks) - len(kept)
    if dropped:
        print(f"  Dropped {dropped} junk/appendix chunks")
    return kept


def print_validation_report(
    chunks: list[dict],
    warn_threshold: int = CHUNK_COUNT_WARNING_THRESHOLD,
) -> None:
    """Print chunk distribution stats and flag suspicious disorder counts."""
    from collections import Counter

    by_disorder = Counter(c["disorder_code"] for c in chunks)
    names = {c["disorder_code"]: c["disorder_name"] for c in chunks}

    print("\nTop disorders by chunk count:")
    for code, count in by_disorder.most_common(10):
        print(f"  {count:>4}  {code}  {names[code][:55]}")

    flagged = sorted(
        ((code, count) for code, count in by_disorder.items() if count > warn_threshold),
        key=lambda item: -item[1],
    )
    if flagged:
        print(f"\nWarning: disorders with >{warn_threshold} chunks:")
        for code, count in flagged:
            print(f"  {count:>4}  {code}  {names[code][:55]}")


def clean_line(line: str) -> str:
    """Strip page headers, footers, and standalone page numbers."""
    line = line.strip()
    line = re.sub(
        r"^(Clinical Descriptions and Diagnostic Requirements for ICD-11 Mental.*|"
        r"Schizophrenia and other primary psychotic disorders \|.*|"
        r"[A-Z][a-z]+ (disorders?|syndrome|behaviour) \| .+)$",
        "", line, flags=re.IGNORECASE,
    )
    line = re.sub(r"^\d{1,4}\s*$", "", line)
    return line.strip()


def extract_text_range(pdf_path: str, first: int, last: int) -> str:
    """
    Call pdftotext to extract a page range from the PDF.

    Searches PATH first; falls back to the active conda environment.
    Raises FileNotFoundError if pdftotext cannot be located.
    """
    import os

    pdftotext_cmd = shutil.which("pdftotext")

    if pdftotext_cmd is None:
        conda_env = Path(os.environ.get("CONDA_PREFIX", ""))
        candidates = [
            conda_env / "Library" / "bin" / "pdftotext.exe",
            conda_env / "bin" / "pdftotext",
        ]
        for c in candidates:
            if c.exists():
                pdftotext_cmd = str(c)
                break

    if pdftotext_cmd is None:
        raise FileNotFoundError(
            "pdftotext not found. Install via: conda install -c conda-forge poppler"
        )

    abs_path = str(Path(pdf_path).resolve())
    print(f"pdftotext     : {pdftotext_cmd}")
    print(f"PDF path      : {abs_path}")
    print(f"File exists   : {Path(abs_path).exists()}")

    result = subprocess.run(
        [pdftotext_cmd, "-f", str(first), "-l", str(last), "-layout", abs_path, "-"],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    if result.returncode != 0:
        print(f"pdftotext error: {result.stderr}")

    return result.stdout


def chunk_text(full_text: str) -> list[dict]:
    """
    Parse raw PDF text into a list of structured chunk dicts.

    Each chunk represents one named section of one ICD-11 disorder entry,
    e.g. {'disorder_code': '6A70', 'section': 'Essential Features', ...}.
    """
    lines = full_text.split("\n")
    chunks: list[dict] = []
    current_disorder_code: str | None = None
    current_disorder_name: str | None = None
    current_section: str | None       = None
    current_lines: list[str]          = []
    current_domain                    = "Unclassified"

    def flush() -> None:
        nonlocal current_lines
        if not current_disorder_code or not current_section:
            current_lines = []
            return
        text = " ".join(l for l in current_lines if l).strip()
        text = re.sub(r"\s{2,}", " ", text)
        if (
            len(text.split()) < 20
            or is_index_bleed_chunk(text, current_section)
            or (
                current_disorder_code == "6E6Z"
                and is_misattributed_6E6Z_chunk({
                    "disorder_code": current_disorder_code,
                    "section": current_section,
                    "text": text,
                })
            )
        ):
            current_lines = []
            return
        chunks.append({
            "source":        "ICD-11 CDDR",
            "domain":        current_domain,
            "disorder_code": current_disorder_code,
            "disorder_name": current_disorder_name,
            "section":       current_section,
            "text":          text,
            "word_count":    len(text.split()),
        })
        current_lines = []

    for line in lines:
        cleaned = clean_line(line)
        if not cleaned:
            continue

        if is_appendix_boundary(cleaned):
            flush()
            break

        if CHAPTER_INDEX_RE.search(cleaned):
            current_lines = []
            current_disorder_code = None
            current_disorder_name = None
            current_section = None
            continue

        code_match = DISORDER_CODE_RE.match(cleaned)
        if code_match:
            flush()
            current_disorder_code = code_match.group(1)
            current_disorder_name = code_match.group(2).strip()
            current_domain        = get_domain(current_disorder_code)
            current_section       = "Overview"
            current_lines         = [current_disorder_name]
            continue

        # Mood episode descriptions: uncoded headings inside the mood chapter.
        # Guarded on the current domain so an identical line elsewhere in the
        # 600-page document can never start an episode block.
        if current_domain == EPISODE_DOMAIN:
            if EPISODE_SECTION_MARKER_RE.match(cleaned):
                flush()
                continue
            episode_match = EPISODE_HEADING_RE.match(cleaned)
            if episode_match:
                flush()
                episode = episode_match.group(1)
                current_disorder_code = EPISODE_CODES[episode]
                current_disorder_name = f"{episode} episode"
                current_domain        = EPISODE_DOMAIN
                current_section       = "Overview"
                current_lines         = [current_disorder_name]
                continue

        # Skip second half of a layout-wrapped differential-diagnosis heading.
        if DIFF_HEADING_CONTINUATION_RE.match(cleaned):
            continue

        sec_match = SECTION_RE.match(cleaned)
        if sec_match:
            flush()
            current_section = normalise_section(sec_match.group(1))
            continue

        current_lines.append(cleaned)

    flush()
    return chunks


def build_embed_text(chunk: dict) -> str:
    """
    Prepend structured metadata to the clinical text before embedding.

    Baking disorder + section context into the embedding means the model
    encodes clinical structure, not just raw surface text.
    """
    return (
        f"Source: {chunk['source']}\n"
        f"Domain: {chunk['domain']}\n"
        f"Disorder: {chunk['disorder_name']} ({chunk['disorder_code']})\n"
        f"Section: {chunk['section']}\n\n"
        f"{chunk['text']}"
    )


def split_long_chunks(
    chunks: list[dict],
    max_words: int = MAX_CHUNK_WORDS,
    overlap_words: int = CHUNK_WORD_OVERLAP,
) -> list[dict]:
    """
    Split oversized chunks with a sliding-word window while preserving metadata.

    This keeps section-level semantics but prevents very long sections from
    becoming single huge embeddings.
    """
    if max_words <= 0:
        return chunks
    if overlap_words < 0:
        overlap_words = 0
    if overlap_words >= max_words:
        overlap_words = max_words - 1

    split_chunks: list[dict] = []
    for chunk in chunks:
        words = chunk["text"].split()
        if len(words) <= max_words:
            split_chunks.append(chunk)
            continue

        step = max_words - overlap_words
        part = 1
        for start in range(0, len(words), step):
            window = words[start : start + max_words]
            if not window:
                break
            sub = dict(chunk)
            sub["text"] = " ".join(window)
            sub["word_count"] = len(window)
            sub["chunk_part"] = part
            split_chunks.append(sub)
            part += 1
            if start + max_words >= len(words):
                break
    return split_chunks


def postprocess(
    chunks: list[dict],
    min_words: int = 40,
    max_words: int = MAX_CHUNK_WORDS,
    overlap_words: int = CHUNK_WORD_OVERLAP,
) -> list[dict]:
    """
    Merge consecutive near-empty chunks from the same disorder/section.

    Also splits oversized chunks and adds the 'embed_text' field to every chunk.
    """
    merged: list[dict] = []
    i = 0
    while i < len(chunks):
        c = chunks[i]
        while (
            c["word_count"] < min_words
            and i + 1 < len(chunks)
            and chunks[i + 1]["disorder_code"] == c["disorder_code"]
            and chunks[i + 1]["section"]       == c["section"]
        ):
            i += 1
            c = dict(c)
            c["text"]       = c["text"] + " " + chunks[i]["text"]
            c["word_count"] = len(c["text"].split())
        c["embed_text"] = build_embed_text(c)
        merged.append(c)
        i += 1

    final_chunks = split_long_chunks(
        merged,
        max_words=max_words,
        overlap_words=overlap_words,
    )
    for c in final_chunks:
        c["embed_text"] = build_embed_text(c)
    return final_chunks


# ── Pipeline entry point ───────────────────────────────────────────────────────

def assign_chunk_uids(chunks: list[dict]) -> None:
    """
    Assign a stable, globally unique id to every chunk, in place.

    Base form: <disorder_code>_<section>_p<part>. When the same
    (code, section, part) triple occurs more than once — e.g. a code that
    appears in both a chapter listing and its actual entry — an occurrence
    suffix (_2, _3, …) disambiguates. The uid is stored in the chunks JSON
    and in ChromaDB metadata so BM25, dense and hybrid all share one id per
    chunk instead of rebuilding (and colliding on) <code>_<section>.
    """
    seen: dict[str, int] = {}
    for c in chunks:
        base = (
            f"{c.get('disorder_code', 'unknown')}_"
            f"{c.get('section', 'unknown').lower().replace(' ', '_')}_"
            f"p{c.get('chunk_part') or 1}"
        )
        n = seen.get(base, 0) + 1
        seen[base] = n
        c["chunk_uid"] = base if n == 1 else f"{base}_{n}"


def run_chunking(
    pdf_path: str   = PDF_PATH,
    chunks_path: str = CHUNKS_PATH,
    start_page: int  = CONTENT_START_PAGE,
    end_page: int    = CONTENT_END_PAGE,
    max_words: int = MAX_CHUNK_WORDS,
    overlap_words: int = CHUNK_WORD_OVERLAP,
) -> list[dict]:
    """
    End-to-end chunking stage: extract PDF text → parse → post-process → save.

    Returns the final list of chunk dicts.
    """
    print(f"Extracting pages {start_page}–{end_page} from PDF …")
    raw_text = extract_text_range(pdf_path, start_page, end_page)
    print(f"  Extracted {len(raw_text):,} characters")

    print("\nParsing into chunks …")
    chunks = chunk_text(raw_text)
    print(f"  Raw chunks : {len(chunks)}")

    chunks = postprocess(
        chunks,
        max_words=max_words,
        overlap_words=overlap_words,
    )
    print(f"  After split: {len(chunks)} (max_words={max_words}, overlap={overlap_words})")

    chunks = filter_junk_chunks(chunks)
    print(f"  Final chunks: {len(chunks)}")

    assign_chunk_uids(chunks)

    print_validation_report(chunks)

    # ── Stats ──────────────────────────────────────────────────────────────────
    domains:  dict[str, int] = {}
    sections: dict[str, int] = {}
    for c in chunks:
        domains [c["domain"] ] = domains.get (c["domain"],  0) + 1
        sections[c["section"]] = sections.get(c["section"], 0) + 1

    print("\nChunks per domain:")
    for d, n in sorted(domains.items(),  key=lambda x: -x[1]):
        print(f"  {n:>4}  {d}")

    print("\nChunks per section:")
    for s, n in sorted(sections.items(), key=lambda x: -x[1]):
        print(f"  {n:>4}  {s}")

    # ── Save ───────────────────────────────────────────────────────────────────
    Path(chunks_path).parent.mkdir(parents=True, exist_ok=True)
    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f"\nSaved {len(chunks)} chunks → {chunks_path}")

    return chunks


if __name__ == "__main__":
    run_chunking()