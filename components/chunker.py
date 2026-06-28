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

from config import (
    CONTENT_END_PAGE,
    CONTENT_START_PAGE,
    CHUNKS_PATH,
    PDF_PATH,
    DOMAIN_MAP,
    SECTION_NORMALISE_MAP,
)

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
    r"Boundaries with other disorders and conditions",
    r"Boundaries with other disorders and conditions \(differential diagnosis\)",
    r"Diagnostic requirements",
    r"Specifiers",
    r"Coded elsewhere",
    r"Note:",
]

SECTION_RE = re.compile(
    r"^(" + "|".join(_SECTION_PATTERNS) + r")\s*$",
    re.IGNORECASE | re.MULTILINE,
)

DISORDER_CODE_RE = re.compile(
    r"^(6[A-Z0-9]{2,5}(?:\.[A-Z0-9]{1,3})?)\s{2,}(.+)$",
    re.MULTILINE,
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
    return SECTION_NORMALISE_MAP.get(heading.strip().lower(), heading.strip())


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
        if len(text.split()) < 20:
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

        code_match = DISORDER_CODE_RE.match(cleaned)
        if code_match:
            flush()
            current_disorder_code = code_match.group(1)
            current_disorder_name = code_match.group(2).strip()
            current_domain        = get_domain(current_disorder_code)
            current_section       = "Overview"
            current_lines         = [current_disorder_name]
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


def postprocess(chunks: list[dict], min_words: int = 40) -> list[dict]:
    """
    Merge consecutive near-empty chunks from the same disorder/section.

    Also adds the 'embed_text' field to every chunk.
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
    return merged


# ── Pipeline entry point ───────────────────────────────────────────────────────

def run_chunking(
    pdf_path: str   = PDF_PATH,
    chunks_path: str = CHUNKS_PATH,
    start_page: int  = CONTENT_START_PAGE,
    end_page: int    = CONTENT_END_PAGE,
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

    chunks = postprocess(chunks)
    print(f"  After merge: {len(chunks)}")

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