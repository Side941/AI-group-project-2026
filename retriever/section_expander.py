"""
section_expander.py
===================
Pure utility functions for post-retrieval section expansion.

Both functions are stateless: they operate only on the data passed in and have
no knowledge of BM25, dense embeddings, or score fields. Any retriever that
needs to guarantee every top-k disorder is represented by all requested
sections can call them in two lines.

Public API
----------
    top_k_disorder_keys(scored_chunks, k) -> list[tuple[str, str]]
    expand_sections(scored_chunks, top_k_keys, sections_by_disorder,
                    score_field, sibling_score_fn) -> list[dict]
"""
from __future__ import annotations


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
            Optional allowlist of section names. When provided, only sibling
            sections whose name is in this set are injected. When None, all
            available sibling sections are injected (legacy behaviour).

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
                if section == chunk.get("section"):
                    continue
                if sections is not None and section not in sections:
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
