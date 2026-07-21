"""
smoke_rag_oneshot.py
====================
One-post end-to-end RAG smoke: retrieve ICD-11 chunks → prompt → Ollama → label.

Requires a running Ollama server with the chosen model pulled, e.g.:
    ollama pull qwen3:0.6b

    python experiments/smoke_rag_oneshot.py
    python experiments/smoke_rag_oneshot.py --head binary --query suicidal
    python experiments/smoke_rag_oneshot.py --retriever bm25 --model qwen3:0.6b
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
from components.config import CHROMA_PATH, RETRIEVAL_SECTIONS  # noqa: E402

OLLAMA_URL = "http://localhost:11434/api/generate"

MULTICLASS_SYSTEM = (
    "You are a mental-health risk classifier for social-media posts. "
    "You must reply with ONLY these two lines, nothing else:\n"
    "Label: <suicidal|depression|normal>\n"
    "Reason: <one sentence>\n"
    "Do NOT give advice. Do NOT explain. Just the two lines above."
)

BINARY_SYSTEM = (
    "You are a mental-health risk classifier for social-media posts. "
    "You must reply with ONLY these two lines, nothing else:\n"
    "Label: <suicide|non-suicide>\n"
    "Reason: <one sentence>\n"
    "Do NOT give advice. Do NOT explain. Just the two lines above."
)

PROMPT_TEMPLATE = """\
{system}

Relevant clinical knowledge:
{chunks}
---
Post: {post}

Answer:"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="One-shot RAG classification smoke.")
    parser.add_argument("--head", choices=("multiclass", "binary"), default="multiclass")
    parser.add_argument("--query", choices=sorted(SAMPLE_QUERIES), default="depression")
    parser.add_argument("--retriever", choices=("bm25", "dense", "hybrid"), default="bm25")
    parser.add_argument("--model", default="qwen3:0.6b")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--timeout", type=int, default=120)
    return parser


def build_retriever(name: str, *, alpha: float):
    sections = list(RETRIEVAL_SECTIONS)
    mood_chunks = load_mood_chunks()

    if name == "bm25":
        return BM25Retriever(chunks=mood_chunks, sections=sections)

    if not CHROMA_PATH.exists():
        raise FileNotFoundError(f"ChromaDB missing at {CHROMA_PATH}")

    from dense_retriever import DenseRetriever, initialise_retrieval
    from hybrid_retriever import HybridRetriever

    initialise_retrieval(chroma_path=str(CHROMA_PATH))
    if name == "dense":
        return DenseRetriever(sections=sections)
    return HybridRetriever(chunks=mood_chunks, sections=sections, alpha=alpha)


def format_chunks(hits: list[dict]) -> str:
    if not hits:
        return "(none)"
    lines = []
    for i, hit in enumerate(hits, start=1):
        text = hit.get("prompt_text") or hit.get("text") or ""
        lines.append(
            f"[{i}] ({hit.get('disorder_name', '')} — {hit.get('section', '')}): {text}"
        )
    return "\n".join(lines)


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


def main() -> int:
    args = build_parser().parse_args()
    post = SAMPLE_QUERIES[args.query]
    labels = (
        ["suicidal", "depression", "normal"]
        if args.head == "multiclass"
        else ["suicide", "non-suicide"]
    )
    system = MULTICLASS_SYSTEM if args.head == "multiclass" else BINARY_SYSTEM

    print("=== One-shot RAG smoke ===")
    print(f"head={args.head}  retriever={args.retriever}  model={args.model}")
    print(f"post: {post}\n")

    retriever = build_retriever(args.retriever, alpha=args.alpha)
    hits = retriever.search(post, k=args.k, expand=True)
    print(f"--- Retrieved ({args.retriever}, k={args.k}) ---")
    print_hits(hits)

    prompt = PROMPT_TEMPLATE.format(
        system=system,
        chunks=format_chunks(hits),
        post=post,
    )
    print("\n--- Calling Ollama ---")
    raw = call_ollama(prompt, model=args.model, timeout=args.timeout)
    label = parse_label(raw, labels)

    print("\n--- Model reply ---")
    print(raw.strip() or "(empty)")
    print(f"\nParsed label: {label}")

    if label == "UNKNOWN":
        print("FAIL — could not parse a valid label.")
        return 1

    print("PASS — end-to-end RAG oneshot OK.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
