import json
import string
from pathlib import Path
from typing import Sequence

import nltk  # type: ignore
from nltk.corpus import stopwords  # type: ignore
from nltk.stem import PorterStemmer  # type: ignore

from components.config import (
    CHUNKS_PATH,
    FEWSHOT_BINARY_EXAMPLES_PATH,
    FEWSHOT_MULTICLASS_EXAMPLES_PATH,
    resolve_path,
)

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)

_STEMMER = PorterStemmer()
_STOPWORDS = set(stopwords.words("english"))


def load_chunks(json_path: str | Path | None = None) -> list[dict]:
    path = resolve_path(json_path, CHUNKS_PATH)
    if not path.exists():
        raise FileNotFoundError(f"Knowledge base file not found: {path}")

    with open(path, encoding="utf-8") as f:
        try:
            chunks = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in knowledge base file: {e}") from e

    for chunk in chunks:
        # Prefer the globally unique chunk_uid assigned at chunking time.
        # Previously ids were rebuilt here as <code>_<section> only, so every
        # part of a multi-part section (and every distinct block sharing a
        # code/section pair) collapsed to one id — de-duplication in the BM25
        # and dense paths then kept a single chunk and discarded the rest.
        # The part-aware fallback covers chunk JSONs built before chunk_uid.
        chunk["id"] = chunk.get("chunk_uid") or (
            f"{chunk.get('disorder_code', 'unknown')}_"
            f"{chunk.get('section', 'unknown').lower().replace(' ', '_')}_"
            f"p{chunk.get('chunk_part') or 1}"
        )
        # Keep original short text for prompt injection.
        chunk["prompt_text"] = chunk.get("text", "")
        # Use richer embed_text for retrieval indexing.
        chunk["text"] = chunk.get("embed_text") or chunk.get("prompt_text", "")

    return chunks


def load_fewshot_examples(
    json_path: str | Path | None = None,
    *,
    head: str | None = None,
) -> list[dict]:
    """Load few-shot example rows exported by rebuild_all_artifacts.py."""
    if json_path is None:
        if head == "binary":
            path = FEWSHOT_BINARY_EXAMPLES_PATH
        else:
            path = FEWSHOT_MULTICLASS_EXAMPLES_PATH
    else:
        default = FEWSHOT_BINARY_EXAMPLES_PATH if head == "binary" else FEWSHOT_MULTICLASS_EXAMPLES_PATH
        path = resolve_path(json_path, default)

    if not path.exists():
        raise FileNotFoundError(f"Few-shot examples file not found: {path}")

    with open(path, encoding="utf-8") as f:
        try:
            rows = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in few-shot examples file: {e}") from e

    for row in rows:
        row["text"] = row.get("text") or row.get("post", "")
        row["post"] = row.get("post") or row.get("text", "")
        row["id"] = row.get("id") or f"{row.get('head', head or 'fewshot')}_{row.get('label', 'unknown')}"
    return rows


def filter_chunks_by_sections(
    chunks: list[dict],
    sections: Sequence[str],
) -> list[dict]:
    """Return only chunks whose section is in *sections*."""
    allowlist = set(sections)
    return [c for c in chunks if c.get("section", "") in allowlist]


def filter_chunks_by_disorder_codes(
    chunks: list[dict],
    prefixes: Sequence[str],
) -> list[dict]:
    """Return only chunks whose disorder_code starts with one of *prefixes*."""
    return [
        c for c in chunks
        if any(str(c.get("disorder_code", "")).startswith(p) for p in prefixes)
    ]


def tokenize(text: str) -> list[str]:
    """
    BM25-friendly normalization:
    - lowercase
    - drop punctuation/very short tokens
    - drop stop words (e.g. "i", "so", "and")
    - stem to reduce sparse variants ("depressed" -> "depress")
    """
    tokens = nltk.word_tokenize(text.lower())
    normalized: list[str] = []
    for token in tokens:
        if token in string.punctuation:
            continue
        if len(token) <= 2:
            continue
        if token in _STOPWORDS:
            continue
        # Keep alpha-numeric tokens that carry semantics ("icd11", "6a72").
        stemmed = _STEMMER.stem(token)
        if stemmed:
            normalized.append(stemmed)
    return normalized
