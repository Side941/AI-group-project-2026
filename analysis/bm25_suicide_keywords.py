"""
bm25_suicide_keywords.py
========================
Empirical justification for using BM25 as the (primary) retriever for
suicide-risk classification: quantifies the *lexical bridge* between the
language of suicidal/depressive Reddit posts and the vocabulary of the
ICD-11 retrieval pool, and shows which query terms actually drive retrieval.

The claim being tested: suicidal-class posts contain explicit, high-signal
keywords ("suicide", "kill", "die", "end it", ...) that appear near-verbatim
in the ICD-11 diagnostic criteria ("suicidal ideation", "attempted suicide"),
so exact lexical matching (BM25) is well suited to routing these posts to the
relevant clinical chunks — no embedding model required for the core signal.

Usage (from the repo root):

    python analysis/bm25_suicide_keywords.py

Reads  datasets/processed/multiclass_dev.csv  and the ICD-11 chunk pool.
Writes analysis/bm25_justification_report.txt. No Ollama needed.
"""

from __future__ import annotations

import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "retriever"))

from src.components.config import (  # noqa: E402
    CHUNKS_PATH,
    MOOD_DISORDER_PREFIXES,
    MULTICLASS_DEV_PATH,
    RETRIEVAL_SECTIONS,
)
from utils import (  # noqa: E402
    load_chunks,
    filter_chunks_by_disorder_codes,
    filter_chunks_by_sections,
    tokenize,
)
from bm25_retriever import BM25Retriever  # noqa: E402

TOP_K = 5
OUT_PATH = ROOT / "analysis" / "bm25_justification_report.txt"

_lines: list[str] = []


def emit(text: str = "") -> None:
    print(text)
    _lines.append(text)


def rule(title: str) -> None:
    emit()
    emit("=" * 78)
    emit(title)
    emit("=" * 78)


# ── Setup: pool, index, dev posts ─────────────────────────────────────────────
pool = filter_chunks_by_sections(
    filter_chunks_by_disorder_codes(load_chunks(CHUNKS_PATH),
                                    MOOD_DISORDER_PREFIXES),
    RETRIEVAL_SECTIONS,
)
retriever = BM25Retriever(chunks=pool, sections=RETRIEVAL_SECTIONS)
bm25 = retriever.bm25  # underlying rank_bm25.BM25Okapi

# Vocabulary and document frequency of the pool, under the SAME tokenizer
# the live pipeline uses (so numbers here transfer to the real system).
chunk_tokens = [tokenize(c.get("embed_text") or c.get("text", "")) for c in pool]
doc_freq: Counter = Counter()
for toks in chunk_tokens:
    doc_freq.update(set(toks))
vocab = set(doc_freq)

df = pd.read_csv(MULTICLASS_DEV_PATH)
classes = ["suicidal", "depression", "normal"]

rule("PART 1 — LEXICAL BRIDGE: post vocabulary vs ICD-11 pool vocabulary")
emit(f"Retrieval pool: {len(pool)} chunks | pool vocabulary: {len(vocab)} terms "
     f"(tokenizer: pipeline's own)")
emit(f"Dev posts: {len(df)} ({', '.join(f'{c} n={sum(df.label == c)}' for c in classes)})")
emit()
emit("Per class: how much of the posts' (stopword-stripped) vocabulary exists")
emit("in the pool at all — the precondition for BM25 to see any signal.")
emit()

class_terms: dict[str, Counter] = {}
for cls in classes:
    terms: Counter = Counter()
    for text in df.loc[df.label == cls, "text"]:
        terms.update(set(tokenize(str(text))))
    class_terms[cls] = terms
    matched = {t: n for t, n in terms.items() if t in vocab}
    emit(f"  {cls:10}: {len(terms):4} distinct terms, "
         f"{len(matched):4} in pool vocab "
         f"({100 * len(matched) / max(len(terms), 1):.0f}%)")

rule("PART 2 — SUICIDE-SIGNAL TERMS: frequency in posts, presence and IDF in pool")
emit("Terms are discovered from the data, not hand-picked: the 25 terms most")
emit("characteristic of suicidal-class posts (frequency in suicidal posts")
emit("relative to normal-class posts), shown with their pool statistics.")
emit()

suic_terms = class_terms["suicidal"]
norm_terms = class_terms["normal"]
scored = sorted(
    ((t, n, n / (norm_terms.get(t, 0) + 1)) for t, n in suic_terms.items()),
    key=lambda x: (-x[2], -x[1]),
)[:25]

emit(f"  {'term':<14}{'suicidal posts':>15}{'normal posts':>14}"
     f"{'pool chunks':>12}{'BM25 IDF':>10}")
for term, n_suic, _ in scored:
    n_pool = doc_freq.get(term, 0)
    idf = bm25.idf.get(term, 0.0)
    emit(f"  {term:<14}{n_suic:>15}{norm_terms.get(term, 0):>14}"
         f"{n_pool:>12}{idf:>10.2f}")
emit()
emit("Reading: a term with high suicidal-post frequency, low normal-post")
emit("frequency, presence in few pool chunks, and high IDF is exactly the")
emit("profile BM25 rewards — rare, discriminative, exact-match signal.")

rule(f"PART 3 — WHAT DRIVES RETRIEVAL: per-post term contributions (top-{TOP_K})")
emit("For every dev post: retrieve top-k, then decompose each hit's BM25 score")
emit("into per-term contributions. Reported per class: the terms that most")
emit("often carry the retrieval, and whether suicid* chunks are reached.")
emit()


def term_contributions(query_tokens: list[str], doc_idx: int) -> dict[str, float]:
    """Per-term BM25 score contribution of this query against one pool doc."""
    contrib: dict[str, float] = {}
    doc = chunk_tokens[doc_idx]
    doc_len = len(doc)
    freqs = Counter(doc)
    k1, b, avgdl = bm25.k1, bm25.b, bm25.avgdl
    for term in set(query_tokens):
        f = freqs.get(term, 0)
        if not f:
            continue
        idf = bm25.idf.get(term, 0.0)
        contrib[term] = idf * (f * (k1 + 1)) / (f + k1 * (1 - b + b * doc_len / avgdl))
    return contrib


pool_index = {id(c): i for i, c in enumerate(pool)}
driver_terms: dict[str, Counter] = {c: Counter() for c in classes}
suic_chunk_hit: dict[str, int] = {c: 0 for c in classes}

for _, row in df.iterrows():
    cls = row["label"]
    q_tokens = tokenize(str(row["text"]))
    hits = retriever.search(str(row["text"]), k=TOP_K, expand=False)
    hit_suic = False
    for h in hits:
        idx = next((i for i, c in enumerate(pool)
                    if c.get("id") == h.get("id")
                    and c.get("text") == h.get("text")), None)
        if idx is None:
            continue
        contrib = term_contributions(q_tokens, idx)
        for term, _score in sorted(contrib.items(), key=lambda x: -x[1])[:3]:
            driver_terms[cls][term] += 1
        if "suicid" in (pool[idx].get("text") or "").lower():
            hit_suic = True
    if hit_suic:
        suic_chunk_hit[cls] += 1

for cls in classes:
    n_cls = int(sum(df.label == cls))
    emit(f"  {cls:10}: top retrieval-driving terms: "
         f"{', '.join(f'{t}({n})' for t, n in driver_terms[cls].most_common(10))}")
    emit(f"  {'':10}  posts whose top-{TOP_K} includes a suicid*-containing "
         f"chunk: {suic_chunk_hit[cls]}/{n_cls}")
emit()

rule("PART 4 — SUMMARY FOR THE REPORT")
suic_in_pool = [c for c in pool if "suicid" in (c.get("text") or "").lower()]
emit(f"- The pool contains {len(suic_in_pool)} chunks mentioning suicid*; "
     f"BM25 can reach them only via exact lexical overlap.")
emit(f"- {suic_chunk_hit['suicidal']}/{int(sum(df.label == 'suicidal'))} "
     f"suicidal-class dev posts retrieve at least one such chunk in the "
     f"top-{TOP_K} — the lexical bridge the BM25 choice relies on.")
emit("- The class-characteristic terms table (Part 2) shows the signal terms")
emit("  are rare in the pool (high IDF), i.e. when they match, they dominate")
emit("  ranking — the regime where sparse lexical retrieval is strongest and")
emit("  a dense encoder adds least for the routing decision.")
emit("- Caveat to state honestly: this justifies BM25 for posts with explicit")
emit("  lexical markers. Posts expressing ideation only implicitly (no shared")
emit("  vocabulary) are the residual case for dense/hybrid retrieval — cf.")
emit("  the per-class hit rates above and RQ3 error analysis.")

emit()
OUT_PATH.parent.mkdir(exist_ok=True)
OUT_PATH.write_text("\n".join(_lines) + "\n", encoding="utf-8")
print(f"\nReport saved to {OUT_PATH}")
