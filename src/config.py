"""
config.py
=========
Shared configuration for the Example-Based RAG pipeline.
Handles all paths, model settings, experiment grid, and retriever configs.
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
SUICIDE_RAW_DATA_PATH   = project_path("datasets", "Suicide_Detection.csv")
SUICIDE_TRAIN_PATH      = project_path("datasets", "suicide_train.csv")
SUICIDE_TEST_PATH       = project_path("datasets", "suicide_test.csv")

CHROMA_PATH     = project_path("datasets", "chroma_db")

# Knowledge Base ChromaDB
KB_CHROMA_PATH = project_path("knowledge_based", "kb_chromadb")

# Knowledge Base files
MHGAP_SUICIDE_PATH    = project_path("knowledge_based", "mhgap.json")

RESULTS_DIR     = project_path("results")
SUMMARY_PATH    = RESULTS_DIR / "summary.csv"
ERROR_ANALYSIS_PATH = RESULTS_DIR / "error_analysis.csv"


# ── Knowledge Base Collections ─────────────────────────────────────────────────
# Must be 3-512 characters for ChromaDB
KB_COLLECTION_NAME = "knowledge_base"


# ── Dataset Column Names ────────────────────────────────────────────────────────
TEXT_COL = "text"
LABEL_COL = "label"


# ── Tasks ───────────────────────────────────────────────────────────────────────
SUICIDE_CLASSES = ["non-suicide", "suicide"]


# ── Embedding ───────────────────────────────────────────────────────────────────
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM   = 384
BATCH_SIZE      = 64


# ── Retrievers ──────────────────────────────────────────────────────────────────
RetrieverType = Literal["bm25", "dense", "hybrid"]

@dataclass
class RetrieverConfig:
    """Base config for any retriever."""
    k: int = 3
    type: RetrieverType = "dense"
    use_knowledge_base: bool = False

@dataclass
class DenseRetrieverConfig(RetrieverConfig):
    type: RetrieverType = "dense"

@dataclass
class BM25RetrieverConfig(RetrieverConfig):
    type: RetrieverType = "bm25"
    k1: float = 1.5
    b: float = 0.75

@dataclass
class HybridRetrieverConfig(RetrieverConfig):
    type: RetrieverType = "hybrid"
    dense_weight: float = 0.5


# ── LLM ─────────────────────────────────────────────────────────────────────────
LLM_MODEL_SIZES = ["0.6b", "1.7b"]
LLM_BASE_NAME   = "Qwen/Qwen3-{size}"
THINKING_MODES  = [True, False]


# ── Experiment Grid ─────────────────────────────────────────────────────────────
@dataclass
class ExperimentConfig:
    """Single experiment configuration."""
    model_size: str
    prompt_type: Literal["zero-shot", "few-shot"]
    thinking_mode: bool
    retriever: RetrieverConfig


# ── Prompt Templates ────────────────────────────────────────────────────────────
ZERO_SHOT_TEMPLATE = """Classify the following text as one of: {labels}.

Text: {text}
Label:"""

FEW_SHOT_TEMPLATE = """Classify the following text as one of: {labels}.

{examples}

Text: {text}
Label:"""

EXAMPLE_TEMPLATE = """Text: {text}
Label: {label}"""


# ── Task Labels Helper ──────────────────────────────────────────────────────────
def get_labels() -> list[str]:
    """Return the class labels."""
    return SUICIDE_CLASSES