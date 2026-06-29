"""
section_expander.py
===================
Pure utility functions for post-retrieval section expansion.

Stateless helpers: no knowledge of BM25, dense embeddings, or ChromaDB.
Retrievers score/filter chunks, then delegate expansion here.

Public API
----------
    build_sections_by_disorder(chunks, sections) -> dict
    expansion_fetch_k(k, sections) -> int
    top_k_disorder_keys(scored_chunks, k) -> list[tuple[str, str]]
    expand_sections(...) -> list[dict]
    finish_search(...) -> list[dict]
"""
from __future__ import annotations

from collections import defaultdict
from typing import Sequence


def build_sections_by_disorder(
    chunks: list[dict],
    sections: Sequence[str] | None = None,
) -> dict[tuple[str, str], dict[str, dict]]:
    """Map (disorder_code, disorder_name) -> {section -> chunk} for expansion."""
    allowlist = set(sections) if sections else None
    result: dict[tuple[str, str], dict[str, dict]] = defaultdict(dict)
    for chunk in chunks:
        section = chunk.get("section", "")
        if allowlist is not None and section not in allowlist:
            continue
        key = (chunk.get("disorder_code", ""), chunk.get("disorder_name", ""))
        result[key][section] = chunk
    return result


def expansion_fetch_k(k: int, sections: Sequence[str] | None) -> int:
    """Over-fetch multiplier so top-k distinct disorders can be found before expansion."""
    return k * 4 if sections else k


def top_k_disorder_keys(
    scored_chunks: list[dict],
    k: int,
) -> list[tuple[str, str]]:
    """
    Walk *scored_chunks* (pre-sorted descending by score) and return the first
    *k* distinct (disorder_code, disorder_name) pairs encountered.

    These are the disorders whose full section sets should survive expansion —
    any disorder that didn't rank in this set is discarded before expanding so
    it can't inflate the result.
    """
    seen: set[tuple[str, str]] = set()
    keys: list[tuple[str, str]] = []
    for chunk in scored_chunks:
        key = (chunk.get("disorder_code", ""), chunk.get("disorder_name", ""))
        if key not in seen:
            seen.add(key)
            keys.append(key)
        if len(keys) == k:
            break
    return keys


def expand_sections(
    scored_chunks: list[dict],
    top_k_keys: list[tuple[str, str]],
    sections_by_disorder: dict[tuple[str, str], dict[str, dict]],
    score_field: str,
    sibling_score_fn=None,
    sections: set[str] | None = None,
) -> list[dict]:
    """
    Interleave section siblings into *scored_chunks* for every disorder in
    *top_k_keys*, without truncating mid-disorder.

    Algorithm
    ---------
    Walk *scored_chunks* in score order. On the first encounter of a top-k
    disorder, emit that chunk then immediately inject every sibling section
    that isn't already present. Later chunks from the same disorder are
    appended only if they weren't already emitted as a sibling.

    Chunks belonging to disorders outside *top_k_keys* are skipped entirely.

    Args:
        scored_chunks:
            Pre-sorted list of chunks (descending by *score_field*).
        top_k_keys:
            Ordered list of (disorder_code, disorder_name) pairs from
            ``top_k_disorder_keys()``.
        sections_by_disorder:
            Mapping of (disorder_code, disorder_name) -> {section -> chunk}.
            Built once at retriever init time and passed in here.
        score_field:
            Name of the score key on each chunk dict (e.g. "bm25_score",
            "dense_score", "hybrid_score").
        sibling_score_fn:
            Optional callable(seed_chunk) -> float that computes the score
            to assign injected siblings. Defaults to
            ``lambda seed: seed[score_field] * 0.999``.
        sections:
            Optional allowlist of section names. *scored_chunks* and
            *sections_by_disorder* should already be restricted to this set;
            any stray rows are skipped defensively.

    Returns:
        Expanded list of chunks. No hard length cap — every requested section
        for every top-k disorder is included.
    """
    if sibling_score_fn is None:
        sibling_score_fn = lambda seed: seed[score_field] * 0.999  # noqa: E731

    top_k_set = set(top_k_keys)
    seen_ids: set[str] = set()
    emitted_disorders: set[tuple[str, str]] = set()
    expanded: list[dict] = []

    for chunk in scored_chunks:
        key = (chunk.get("disorder_code", ""), chunk.get("disorder_name", ""))

        if key not in top_k_set:
            continue

        section_name = chunk.get("section", "")
        if sections is not None and section_name not in sections:
            continue

        chunk_id = chunk.get("id", "")
        if chunk_id in seen_ids:
            continue

        if key not in emitted_disorders:
            # First encounter: emit seed, then inject all sibling sections.
            emitted_disorders.add(key)
            seen_ids.add(chunk_id)
            expanded.append(chunk)

            sibling_score = sibling_score_fn(chunk)
            section_map = sections_by_disorder.get(key, {})
            for section, sibling in section_map.items():
                if section == section_name:
                    continue
                sibling_id = sibling.get("id", "")
                if sibling_id in seen_ids:
                    continue
                sibling_out = sibling.copy()
                sibling_out[score_field] = sibling_score
                seen_ids.add(sibling_id)
                expanded.append(sibling_out)
        else:
            # Later chunk from an already-emitted disorder; include only if
            # it wasn't already injected as a sibling.
            seen_ids.add(chunk_id)
            expanded.append(chunk)

    return expanded


def finish_search(
    scored: list[dict],
    k: int,
    sections: Sequence[str] | None,
    sections_by_disorder: dict[tuple[str, str], dict[str, dict]],
    score_field: str,
    *,
    expand: bool = True,
) -> list[dict]:
    """
    Shared tail for retriever search(): optionally expand top-k disorders to
    their full section set, otherwise return scored rows as-is.
    """
    if not sections or not expand:
        return scored[:k] if not sections else scored

    top_keys = top_k_disorder_keys(scored, k)
    return expand_sections(
        scored_chunks=scored,
        top_k_keys=top_keys,
        sections_by_disorder=sections_by_disorder,
        score_field=score_field,
        sections=set(sections),
    )
