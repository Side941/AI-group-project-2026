"""
exp_fewshot_rag.py
==================
Experiment: wire few-shot *retrieval* into the classification prompt.

Unlike the notebooks (which use static few-shot templates), this script:
  1. Retrieves labeled posts from knowledge_base/fewshot/
  2. Optionally also retrieves ICD-11 clinical chunks
  3. Builds a dynamic few-shot prompt and calls Ollama

    # Show retrieved examples + assembled prompt (no LLM)
    python experiments/exp_fewshot_rag.py --dry-run

    # End-to-end with few-shot BM25 + ICD-11 BM25
    python experiments/exp_fewshot_rag.py --query depression

    # Few-shot hybrid only (no ICD-11 context)
    python experiments/exp_fewshot_rag.py --kb-retriever none --fewshot-retriever hybrid

    # Compare prompt variants side-by-side
    python experiments/exp_fewshot_rag.py --compare --query suicidal
"""

from __future__ import annotations

import argparse
import json
import re
import urllib.error
import urllib.request

from _common import SAMPLE_QUERIES, bootstrap, load_mood_chunks, print_hits

bootstrap()

from bm25_retriever import BM25Retriever  # noqa: E402
from components.config import (  # noqa: E402
    CHROMA_PATH,
    FEWSHOT_CHROMA_PATH,
    FEWSHOT_MULTICLASS_EXAMPLES_PATH,
    RETRIEVAL_SECTIONS,
)
from fewshot_retrievers import (  # noqa: E402
    FewShotBM25Retriever,
    build_fewshot_retrievers,
    initialise_fewshot_dense_retrieval,
)

OLLAMA_URL = "http://localhost:11434/api/generate"

SYSTEM_PROMPT = (
    "You are a mental-health risk classifier for social-media posts. "
    "You must reply with ONLY these two lines, nothing else:\n"
    "Label: <suicidal|depression|normal>\n"
    "Reason: <one sentence>\n"
    "Do NOT give advice. Do NOT explain. Just the two lines above."
)

LABELS = ["suicidal", "depression", "normal"]

PROMPT_TEMPLATE = """\
{system}
{kb_block}{examples_block}
Post: {post}

Answer:"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="RAG experiment with retrieved few-shot examples.",
    )
    parser.add_argument("--query", choices=sorted(SAMPLE_QUERIES), default="depression")
    parser.add_argument(
        "--fewshot-retriever",
        choices=("bm25", "dense", "hybrid"),
        default="bm25",
        help="Retriever over the few-shot example store.",
    )
    parser.add_argument(
        "--kb-retriever",
        choices=("none", "bm25", "dense", "hybrid"),
        default="bm25",
        help="Optional ICD-11 clinical retriever (none = examples only).",
    )
    parser.add_argument("--n-examples", type=int, default=3, help="Few-shot examples to inject.")
    parser.add_argument("--k-kb", type=int, default=3, help="ICD-11 chunks to inject.")
    parser.add_argument("--alpha", type=float, default=0.3, help="Hybrid BM25 weight.")
    parser.add_argument("--model", default="qwen3:0.6b")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print retrieved context + prompt; do not call Ollama.",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Run zero-shot / retrieved-fewshot / fewshot+KB variants.",
    )
    return parser


def build_kb_retriever(name: str, *, alpha: float):
    if name == "none":
        return None

    sections = list(RETRIEVAL_SECTIONS)
    mood_chunks = load_mood_chunks()
    if name == "bm25":
        return BM25Retriever(chunks=mood_chunks, sections=sections)

    if not CHROMA_PATH.exists():
        raise FileNotFoundError(f"ICD-11 ChromaDB missing at {CHROMA_PATH}")

    from dense_retriever import DenseRetriever, initialise_retrieval
    from hybrid_retriever import HybridRetriever

    initialise_retrieval(chroma_path=str(CHROMA_PATH))
    if name == "dense":
        return DenseRetriever(sections=sections)
    return HybridRetriever(chunks=mood_chunks, sections=sections, alpha=alpha)


def build_fewshot_retriever(name: str, *, alpha: float):
    examples_path = FEWSHOT_MULTICLASS_EXAMPLES_PATH
    if not examples_path.exists():
        raise FileNotFoundError(
            f"Few-shot examples missing at {examples_path}. "
            "Build with: python src/builders/rebuild_all_artifacts.py"
        )

    if name == "bm25":
        return FewShotBM25Retriever(head="multiclass")

    if not FEWSHOT_CHROMA_PATH.exists():
        raise FileNotFoundError(f"Few-shot Chroma missing at {FEWSHOT_CHROMA_PATH}")

    initialise_fewshot_dense_retrieval()
    return build_fewshot_retrievers(head="multiclass", alpha=alpha)[name]


def format_kb_block(hits: list[dict]) -> str:
    if not hits:
        return ""
    lines = []
    for i, hit in enumerate(hits, start=1):
        text = hit.get("prompt_text") or hit.get("text") or ""
        lines.append(
            f"[{i}] ({hit.get('disorder_name', '')} — {hit.get('section', '')}): {text}"
        )
    return (
        "\nRelevant clinical knowledge:\n"
        + "\n".join(lines)
        + "\n---\n"
    )


def format_examples_block(examples: list[dict]) -> str:
    if not examples:
        return ""
    lines = []
    for ex in examples:
        post = (ex.get("text") or "").strip()
        label = ex.get("label", "")
        lines.append(f'Post: "{post}"\nLabel: {label}')
    return "\nExamples:\n" + "\n\n".join(lines) + "\n---\n"


def build_prompt(
    *,
    system: str,
    post: str,
    kb_hits: list[dict],
    examples: list[dict],
) -> str:
    return PROMPT_TEMPLATE.format(
        system=system,
        kb_block=format_kb_block(kb_hits),
        examples_block=format_examples_block(examples),
        post=post,
    )


def parse_label(text: str, valid_labels: list[str]) -> str:
    cleaned = (text or "").strip()
    match = re.search(r"Label:\s*([A-Za-z0-9_\-]+)", cleaned, flags=re.IGNORECASE)
    if match:
        candidate = match.group(1).strip().lower().replace("_", "-")
        for label in valid_labels:
            if candidate == label.lower():
                return label
    lower = cleaned.lower()
    for label in valid_labels:
        if label.lower() in lower:
            return label
    return "UNKNOWN"


def call_ollama(prompt: str, *, model: str, timeout: int) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.0},
    }
    req = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Ollama request failed ({exc}). Is Ollama running and is '{model}' pulled?"
        ) from exc
    return body.get("response", "")


def run_variant(
    *,
    name: str,
    post: str,
    system: str,
    labels: list[str],
    kb_hits: list[dict],
    examples: list[dict],
    model: str,
    timeout: int,
    dry_run: bool,
) -> str:
    print(f"\n===== Variant: {name} =====")
    if examples:
        print(f"Few-shot examples ({len(examples)}):")
        print_hits(examples, limit=len(examples))
    else:
        print("Few-shot examples: (none)")

    if kb_hits:
        print(f"\nICD-11 chunks ({len(kb_hits)}):")
        print_hits(kb_hits, limit=len(kb_hits))
    else:
        print("\nICD-11 chunks: (none)")

    prompt = build_prompt(
        system=system,
        post=post,
        kb_hits=kb_hits,
        examples=examples,
    )

    if dry_run:
        print("\n--- Assembled prompt ---")
        print(prompt)
        return "DRY_RUN"

    print("\n--- Calling Ollama ---")
    raw = call_ollama(prompt, model=model, timeout=timeout)
    label = parse_label(raw, labels)
    print("--- Model reply ---")
    print(raw.strip() or "(empty)")
    print(f"Parsed label: {label}")
    return label


def main() -> int:
    args = build_parser().parse_args()
    post = SAMPLE_QUERIES[args.query]

    print("=== Few-shot retrieval RAG experiment ===")
    print(f"query={args.query}")
    print(f"fewshot={args.fewshot_retriever}  n_examples={args.n_examples}")
    print(f"kb={args.kb_retriever}  k_kb={args.k_kb}")
    print(f"model={args.model}  dry_run={args.dry_run}  compare={args.compare}")
    print(f"\nPost: {post}")

    fs_retriever = build_fewshot_retriever(args.fewshot_retriever, alpha=args.alpha)
    examples = fs_retriever.search(post, k=args.n_examples)

    kb_retriever = build_kb_retriever(args.kb_retriever, alpha=args.alpha)
    kb_hits = (
        kb_retriever.search(post, k=args.k_kb, expand=True)
        if kb_retriever is not None
        else []
    )

    if args.compare:
        variants = [
            ("zero_shot_no_kb", [], []),
            ("retrieved_fewshot_only", examples, []),
            ("retrieved_fewshot_plus_kb", examples, kb_hits),
        ]
        results: dict[str, str] = {}
        for name, ex, kb in variants:
            results[name] = run_variant(
                name=name,
                post=post,
                system=SYSTEM_PROMPT,
                labels=LABELS,
                kb_hits=kb,
                examples=ex,
                model=args.model,
                timeout=args.timeout,
                dry_run=args.dry_run,
            )

        print("\n===== Comparison summary =====")
        for name, label in results.items():
            print(f"  {name:28} -> {label}")
        if args.dry_run:
            print("PASS — few-shot retrieval compare (dry-run) OK.")
            return 0
        if any(v == "UNKNOWN" for v in results.values()):
            print("FAIL — at least one variant returned UNKNOWN.")
            return 1
        print("PASS — few-shot retrieval compare OK.")
        return 0

    label = run_variant(
        name="retrieved_fewshot_plus_kb"
        if kb_hits
        else "retrieved_fewshot_only",
        post=post,
        system=SYSTEM_PROMPT,
        labels=LABELS,
        kb_hits=kb_hits,
        examples=examples,
        model=args.model,
        timeout=args.timeout,
        dry_run=args.dry_run,
    )

    if args.dry_run:
        print("\nPASS — few-shot retrieval experiment (dry-run) OK.")
        return 0
    if label == "UNKNOWN":
        print("\nFAIL — could not parse a valid label.")
        return 1
    print("\nPASS — few-shot retrieval RAG experiment OK.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
