import json
import string
from pathlib import Path
from typing import Sequence

import nltk  # type: ignore
from nltk.corpus import stopwords  # type: ignore
from nltk.stem import PorterStemmer  # type: ignore

from components.config import CHUNKS_PATH, resolve_path

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
        chunk["id"] = (
            f"{chunk.get('disorder_code', 'unknown')}_"
            f"{chunk.get('section', 'unknown').lower().replace(' ', '_')}"
        )
        # Keep original short text for prompt injection.
        chunk["prompt_text"] = chunk.get("text", "")
        # Use richer embed_text for retrieval indexing.
        chunk["text"] = chunk.get("embed_text") or chunk.get("prompt_text", "")

    return chunks


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
