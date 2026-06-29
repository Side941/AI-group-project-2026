import json
import string
from pathlib import Path
from typing import Sequence

import nltk  # type: ignore

from components.config import CHUNKS_PATH, resolve_path

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)


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


def tokenize(text: str) -> list[str]:
    tokens = nltk.word_tokenize(text.lower())
    return [token for token in tokens if token not in string.punctuation]
