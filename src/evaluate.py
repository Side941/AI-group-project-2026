"""
evaluate.py
===========
Runs experiments using KB retrieval with optional Reddit examples.
Supports: KB only, Reddit examples only, or Combined (KB + Reddit).
"""

from __future__ import annotations

from typing import List, Tuple
import json
import pandas as pd
from tqdm import tqdm

from src.config import (
    get_labels,
    TEXT_COL,
    LABEL_COL,
    MHGAP_SUICIDE_PATH,
    CHROMA_PATH,
)
from src.embedder import Embedder
from src.retrievers.dense_retriever import DenseRetriever
from src.retrievers.bm25_retriever import BM25Retriever
from src.retrievers.hybrid_retriever import HybridRetriever
from src.prompt_builder import PromptBuilder
from src.llm_inference import LLMInference
from src.vector_store import VectorStore

MAX_RETRIES = 2


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


def load_kb_chunks() -> Tuple[List[str], List[str]]:
    """Load mhGAP chunks for KB retrieval."""
    with open(MHGAP_SUICIDE_PATH, 'r') as f:
        data = json.load(f)
    
    if isinstance(data, dict):
        data = [data]
    
    texts = []
    labels = []
    for chunk in data:
        if 'text' in chunk:
            texts.append(chunk['text'])
            labels.append(chunk.get('section', ''))
        else:
            texts.append(str(chunk))
            labels.append('')
    
    return texts, labels


def load_training_examples(train_df: pd.DataFrame, embedder: Embedder) -> Tuple[DenseRetriever, VectorStore]:
    """Load or build training example retriever."""
    vector_store = VectorStore(CHROMA_PATH, "train_suicide")
    vector_store.create_or_load(384)
    
    example_texts = train_df[TEXT_COL].tolist()
    example_labels = train_df[LABEL_COL].tolist()
    
    if vector_store.count() == 0:
        print("Indexing training examples into ChromaDB...")
        embeddings = embedder.encode(example_texts)
        ids = [f"ex_{i:06d}" for i in range(len(example_texts))]
        vector_store.add(ids, example_texts, embeddings, example_labels)
        print(f"Indexed {vector_store.count():,} training examples")
    else:
        print(f"Using cached training examples: {vector_store.count():,} examples")
    
    example_retriever = DenseRetriever(example_texts, example_labels, embedder)
    example_retriever.index()
    
    return example_retriever, vector_store


def run_experiment(
    config,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    embedder: Embedder | None = None,
    debug: bool = False,
) -> pd.DataFrame:
    """Run a single experiment configuration."""
    
    # ── Setup ────────────────────────────────────────────────────────────
    valid_labels = get_labels()
    
    if embedder is None:
        embedder = Embedder()

    # ── Load KB chunks ──────────────────────────────────────────────────
    kb_texts, kb_labels = load_kb_chunks()
    print(f"Loaded {len(kb_texts)} KB chunks")

    # ── KB Retriever setup ──────────────────────────────────────────────
    kb_retriever = None
    
    if config.retriever_type == "dense":
        kb_retriever = DenseRetriever(kb_texts, kb_labels, embedder)
    elif config.retriever_type == "bm25":
        kb_retriever = BM25Retriever(kb_texts, kb_labels)
    elif config.retriever_type == "hybrid":
        kb_retriever = HybridRetriever(kb_texts, kb_labels, embedder, alpha=0.5)
    else:
        raise ValueError(f"Unknown retriever: {config.retriever_type}")
    
    kb_retriever.index()

    # ── Load training examples (if combined mode) ──────────────────────
    example_retriever = None
    if config.use_examples and train_df is not None:
        example_retriever, _ = load_training_examples(train_df, embedder)
        print(f"Loaded training examples for combined mode")

    # ── Prompt builder ───────────────────────────────────────────────────
    prompt_builder = PromptBuilder()

    # ── LLM ──────────────────────────────────────────────────────────────
    llm = LLMInference(model_size=config.model_size, thinking_mode=config.thinking_mode)
    llm.load()

    # ── Run predictions ──────────────────────────────────────────────────
    results = []
    null_retries_saved = 0
    mode = "Combined" if config.use_examples else "KB only"
    loop_desc = f"{config.model_size} | {config.prompt_type} | {config.retriever_type} | k={config.k} | {mode}"

    for i, (_, row) in enumerate(tqdm(test_df.iterrows(), total=len(test_df), desc=loop_desc)):
        text = row[TEXT_COL]
        true_label = row[LABEL_COL]

        if config.prompt_type == "zero-shot":
            prompt = prompt_builder.build_zero_shot(text)
            kb_chunks = None
            reddit_examples = None
            
        else:
            # Retrieve KB chunks
            query_embedding = embedder.encode_single(text)
            kb_chunks = kb_retriever.retrieve(text, query_embedding, k=config.k)
            
            # Retrieve Reddit examples (if combined mode)
            reddit_examples = None
            if config.use_examples and example_retriever is not None:
                reddit_examples = example_retriever.retrieve(text, query_embedding, k=config.k)
            
            # Build prompt based on what's available
            if kb_chunks and reddit_examples:
                # Combined: KB + Reddit examples
                prompt = prompt_builder.build_combined(text, kb_chunks, reddit_examples)
            elif kb_chunks:
                # KB only
                prompt = prompt_builder.build_knowledge_based(text, kb_chunks)
            else:
                # Fallback
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

        # ── Detailed Debug Output ──────────────────────────────────────
        if debug:
            print(f"\n{'='*80}")
            print(f"🔍 DEBUG Example {i+1}")
            print(f"{'='*80}")
            print(f"📝 Text: {text[:200]}...")
            print(f"🏷️  True Label: {true_label}")
            print(f"🤖 Predicted: {pred_label}")
            print(f"📄 Raw Output: '{raw_output}'")
            
            # Show model thinking trace if available
            if hasattr(llm, 'last_thinking_trace') and llm.last_thinking_trace:
                print(f"\n🧠 Model Thinking/Reasoning:")
                trace = llm.last_thinking_trace
                if len(trace) > 500:
                    print(f"{trace[:500]}...")
                else:
                    print(trace)
            
            if config.prompt_type == "few-shot":
                print(f"\n📚 Retrieved KB Chunks ({len(kb_chunks) if kb_chunks else 0}):")
                if kb_chunks:
                    for j, (chunk_text, section) in enumerate(kb_chunks, 1):
                        print(f"  [{j}] Section: {section}")
                        print(f"      Text: {chunk_text[:150]}...")
                
                if reddit_examples:
                    print(f"\n📊 Retrieved Reddit Examples ({len(reddit_examples)}):")
                    for j, (ex_text, ex_label) in enumerate(reddit_examples, 1):
                        print(f"  [{j}] Label: {ex_label}")
                        print(f"      Text: {ex_text[:150]}...")
            
            print(f"\n📋 Full Prompt:")
            print(f"{prompt}")
            print(f"{'='*80}\n")

        retrieved_str = "|||".join(ex_text for ex_text, _ in (kb_chunks or [])) if kb_chunks else ""

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