"""
evaluate.py
===========
Runs a single experiment configuration for suicide detection.
"""

from __future__ import annotations

from typing import List, Tuple
import json

import pandas as pd
from tqdm import tqdm

from src.config import (
    ExperimentConfig,
    RetrieverType,
    get_labels,
    TEXT_COL,
    LABEL_COL,
    CHROMA_PATH,
    KB_CHROMA_PATH,
    KB_COLLECTION_NAME,
    MHGAP_SUICIDE_PATH,
)
from src.embedder import Embedder
from src.retrievers.base import BaseRetriever
from src.retrievers.bm25_retriever import BM25Retriever
from src.retrievers.dense_retriever import DenseRetriever
from src.retrievers.hybrid_retriever import HybridRetriever
from src.prompt_builder import PromptBuilder
from src.llm_inference import LLMInference
from src.vector_store import VectorStore

MAX_RETRIES = 2


def get_retriever(
    retriever_type: RetrieverType,
    corpus_texts: List[str],
    corpus_labels: List[str],
    embedder: Embedder,
    k: int = 3,
) -> BaseRetriever:
    """Factory: create the right retriever based on config."""
    if retriever_type == "bm25":
        return BM25Retriever(corpus_texts, corpus_labels)
    elif retriever_type == "dense":
        return DenseRetriever(corpus_texts, corpus_labels, embedder)
    elif retriever_type == "hybrid":
        raise ValueError("HybridRetriever must be constructed manually.")
    else:
        raise ValueError(f"Unknown retriever type: {retriever_type}")


def parse_prediction(raw_output: str, valid_labels: List[str]) -> str | None:
    """Extract the predicted label from LLM output."""
    raw_lower = raw_output.strip().lower()

    prefixes_to_strip = ["label:", "label", "classification:", "class:", "answer:", "output:", "response:"]
    for prefix in prefixes_to_strip:
        if raw_lower.startswith(prefix):
            raw_lower = raw_lower[len(prefix):].strip()

    for label in valid_labels:
        if raw_lower == label.lower():
            return label

    for label in sorted(valid_labels, key=len, reverse=True):
        if label.lower() in raw_lower:
            return label

    return None


def compute_metrics(y_true: List[str], y_pred: List[str | None]) -> dict:
    """Compute accuracy metrics."""
    from sklearn.metrics import accuracy_score

    valid_mask = [p is not None for p in y_pred]
    y_true_valid = [t for t, m in zip(y_true, valid_mask) if m]
    y_pred_valid = [p for p, m in zip(y_pred, valid_mask) if m]
    null_count = sum(1 for p in y_pred if p is None)

    accuracy_lenient = accuracy_score(y_true_valid, y_pred_valid) if y_pred_valid else 0.0

    y_pred_strict = [p if p is not None else "__NULL__" for p in y_pred]
    accuracy_strict = accuracy_score(y_true, y_pred_strict)

    return {
        "accuracy": accuracy_strict,
        "accuracy_lenient": accuracy_lenient,
        "correct": int(accuracy_strict * len(y_true)),
        "total": len(y_true),
        "null_predictions": null_count,
    }


def _load_knowledge_base(embedder: Embedder) -> VectorStore:
    """Load the mhGAP knowledge base."""
    vector_store = VectorStore(KB_CHROMA_PATH, KB_COLLECTION_NAME)
    vector_store.create_or_load(384)
    
    if vector_store.count() == 0:
        print(f"Indexing knowledge base from {MHGAP_SUICIDE_PATH} ...")
        with open(MHGAP_SUICIDE_PATH, "r", encoding="utf-8") as f:
            chunks = json.load(f)
        
        if isinstance(chunks, dict):
            chunks = [chunks]
        
        texts = []
        ids = []
        labels = []
        metadatas = []
        
        for i, chunk in enumerate(chunks):
            if 'text' in chunk:
                texts.append(chunk['text'])
                section = chunk.get('section', '')
                labels.append(section)
                metadatas.append({
                    'source': chunk.get('source', 'mhgap'),
                    'section': section,
                    'index': i,
                })
            else:
                texts.append(str(chunk))
                labels.append('')
                metadatas.append({'source': 'unknown', 'section': '', 'index': i})
            
            ids.append(f"kb_suicide_{i:04d}")
        
        embeddings = embedder.encode(texts)
        vector_store.add_with_metadata(ids, texts, embeddings, metadatas, labels)
        print(f"Indexed {vector_store.count()} knowledge chunks")
    else:
        print(f"Using cached knowledge base: {vector_store.count()} chunks")
    
    return vector_store


def run_experiment(
    config: ExperimentConfig,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    embedder: Embedder | None = None,
    debug: bool = False,
) -> pd.DataFrame:
    """Run a single experiment configuration."""

    # ── Setup ────────────────────────────────────────────────────────────
    corpus_texts = train_df[TEXT_COL].tolist()
    corpus_labels = train_df[LABEL_COL].tolist()
    valid_labels = get_labels()

    # ── Retrieval setup ──────────────────────────────────────────────────
    use_chroma = False
    use_kb = False
    vector_store = None
    retriever = None

    if config.prompt_type == "few-shot":
        use_kb = getattr(config.retriever, 'use_knowledge_base', False)
        use_chroma = config.retriever.type in ("dense", "hybrid")

        if use_kb:
            if embedder is None:
                embedder = Embedder()
            vector_store = _load_knowledge_base(embedder)
            use_chroma = True
        
        elif use_chroma:
            if embedder is None:
                embedder = Embedder()

            vector_store = VectorStore(CHROMA_PATH, "train_suicide")
            vector_store.create_or_load(384)

            if vector_store.count() == 0:
                print("Indexing corpus into ChromaDB...")
                embeddings = embedder.encode(corpus_texts)
                ids = [f"ex_{i:06d}" for i in range(len(corpus_texts))]
                vector_store.add(ids, corpus_texts, embeddings, corpus_labels)
                print(f"Indexed {vector_store.count():,} examples")
            else:
                print(f"Using cached ChromaDB: {vector_store.count():,} examples")

            if config.retriever.type == "hybrid":
                dense_weight = getattr(config.retriever, "dense_weight", 0.5)
                retriever = HybridRetriever(
                    corpus_texts, corpus_labels,
                    vector_store=vector_store,
                    embedder=embedder,
                    alpha=1 - dense_weight,
                )
                retriever.index()
                print(f"HYBRID retriever ready (alpha={1 - dense_weight:.1f})")
                use_chroma = False
        else:
            if embedder is None:
                embedder = Embedder()
            retriever = get_retriever(
                config.retriever.type, corpus_texts, corpus_labels, embedder, config.retriever.k
            )
            retriever.index()
            print(f"BM25 retriever indexed: {len(corpus_texts):,} examples")

    # ── Prompt builder ───────────────────────────────────────────────────
    prompt_builder = PromptBuilder()

    # ── LLM ──────────────────────────────────────────────────────────────
    llm = LLMInference(model_size=config.model_size, thinking_mode=config.thinking_mode)
    llm.load()

    # ── Run predictions ──────────────────────────────────────────────────
    results = []
    null_retries_saved = 0
    retriever_type = getattr(config.retriever, 'type', 'none')
    loop_desc = f"{config.model_size} | {config.prompt_type} | {retriever_type}"

    for i, (_, row) in enumerate(tqdm(test_df.iterrows(), total=len(test_df), desc=loop_desc)):
        text = row[TEXT_COL]
        true_label = row[LABEL_COL]

        examples = None
        if config.prompt_type == "few-shot":
            if use_chroma:
                query_embedding = embedder.encode_single(text)
                examples = vector_store.query(query_embedding, k=config.retriever.k)
            else:
                query_embedding = embedder.encode_single(text)
                examples = retriever.retrieve(text, query_embedding, k=config.retriever.k)
            
            prompt = prompt_builder.build(text, examples, is_knowledge_base=use_kb)
        else:
            prompt = prompt_builder.build_zero_shot(text)

        raw_output = ""
        attempt = 0
        for attempt in range(MAX_RETRIES):
            raw_output = llm.classify(prompt)
            if raw_output.strip():
                break

        if attempt > 0 and raw_output.strip():
            null_retries_saved += 1

        pred_label = parse_prediction(raw_output, valid_labels)

        if debug and i < 3:
            print(f"\n{'='*60}")
            print(f"DEBUG Example {i+1}")
            print(f"True: {true_label} | Pred: {pred_label} | Raw: '{raw_output[:80]}'")
            if config.prompt_type == "few-shot":
                print(f"Retrieved: {[(t[:60], l) for t, l in examples]}")
            print(f"{'='*60}")

        retrieved_str = "|||".join(ex_text for ex_text, _ in examples) if examples else ""

        results.append({
            "text": text,
            "true_label": true_label,
            "pred_label": pred_label,
            "correct": pred_label == true_label if pred_label else False,
            "raw_output": raw_output,
            "retrieved_examples": retrieved_str,
        })

    if null_retries_saved > 0:
        print(f"  ℹ️  Retry saved {null_retries_saved} empty response(s)")

    llm.unload()
    return pd.DataFrame(results)