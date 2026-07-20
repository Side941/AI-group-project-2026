"""
config.py
=========
Shared configuration for the KB-only RAG pipeline.
"""

from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass
from typing import Literal

# ── Project Root ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent

def project_path(*parts: str) -> Path:
    """Build an absolute path under the project root."""
    return PROJECT_ROOT.joinpath(*parts)


# ── Paths ───────────────────────────────────────────────────────────────────────
SUICIDE_TRAIN_PATH = project_path("datasets", "suicide_train.csv")
SUICIDE_TEST_PATH = project_path("datasets", "suicide_test.csv")

# Knowledge Base
MHGAP_SUICIDE_PATH = project_path("knowledge_based", "mhgap.json")
KB_CHROMA_PATH = project_path("knowledge_based", "kb_chromadb")

CHROMA_PATH = project_path("datasets", "chroma_db")  # For training examples

RESULTS_DIR = project_path("results")


# ── Knowledge Base ─────────────────────────────────────────────────────────────
KB_COLLECTION_NAME = "knowledge_base"


# ── Dataset Column Names ────────────────────────────────────────────────────────
TEXT_COL = "text"
LABEL_COL = "label"


# ── Tasks ───────────────────────────────────────────────────────────────────────
SUICIDE_CLASSES = ["non-suicide", "suicide"]


# ── Embedding ───────────────────────────────────────────────────────────────────
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
BATCH_SIZE = 64


# ── LLM ─────────────────────────────────────────────────────────────────────────
LLM_MODEL_SIZES = ["0.6b", "1.7b"]
THINKING_MODES = [True, False]


# ── Prompt Templates ────────────────────────────────────────────────────────────
ZERO_SHOT_TEMPLATE = """Classify the following text as one of: {labels}.

Text: {text}
Label:"""

FEW_SHOT_TEMPLATE = """Classify the following text as one of: {labels}.

{examples}

Text: {text}
Label:"""

KB_TEMPLATE = """Using the clinical criteria below, classify the following text as one of: {labels}.

Clinical criteria for suicide risk assessment:
{context}

Based on these criteria, does the text indicate suicide risk?
Text: {text}
Label:"""

COMBINED_TEMPLATE = """Using the clinical criteria below AND the example posts, classify the following text as one of: {labels}.

Clinical criteria for suicide risk assessment:
{context}

Example posts:
{examples}

Based on the clinical criteria and the patterns in the examples, classify this text:
Text: {text}
Label:"""

EXAMPLE_TEMPLATE = """Text: {text}
Label: {label}"""


# ── Task Labels Helper ──────────────────────────────────────────────────────────
def get_labels() -> list[str]:
    """Return the class labels."""
    return SUICIDE_CLASSES