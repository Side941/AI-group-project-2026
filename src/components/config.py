"""
config.py
=========
Shared configuration and project paths for the ICD-11 knowledge base pipeline.

All filesystem paths are resolved relative to PROJECT_ROOT (the repo root),
so imports work whether you run from the project root, src/, or notebooks/.
"""

from __future__ import annotations

from pathlib import Path

# Repo root: src/components/config.py -> parents[0]=components, [1]=src, [2]=repo.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def project_path(*parts: str) -> Path:
    """Build an absolute path under the project root."""
    return PROJECT_ROOT.joinpath(*parts)


def resolve_path(path: str | Path | None = None, default: Path | None = None) -> Path:
    """
    Resolve a path against PROJECT_ROOT when it is not already absolute.

    If path is None, default must be provided.
    """
    chosen = Path(path) if path is not None else default
    if chosen is None:
        raise ValueError("No path provided")
    return chosen if chosen.is_absolute() else PROJECT_ROOT / chosen


# ── Model / collection ─────────────────────────────────────────────────────────
COLLECTION_NAME = "icd11_clinical"
EMBEDDING_MODEL = "FremyCompany/BioLORD-2023"
BATCH_SIZE = 64

# ── Paths ──────────────────────────────────────────────────────────────────────

ICD11_STORE_DIR = project_path("knowledge_base", "icd_11")
PDF_PATH = ICD11_STORE_DIR / "icd_11.pdf"
CHUNKS_PATH = ICD11_STORE_DIR / "icd11_chunks.json"
CHROMA_PATH = ICD11_STORE_DIR / "chroma_db"

RAW_DATASETS_DIR = project_path("datasets", "raw")
PROCESSED_DATASETS_DIR = project_path("datasets", "processed")
RESULTS_DIR = project_path("results")

# Hugging Face mental-health corpus (local cache + Hub fallback).
HF_DATASET_REPO = "ourafla/Mental-Health_Text-Classification_Dataset"
HF_TRAIN_FILE = "mental_health_train.csv"
HF_TEST_FILE = "mental_health_test.csv"
# File names inside the Hub repo (renamed upstream; the local cache keeps the
# original names above). The "heath"/"unbanlanced" typos are upstream's.
HF_HUB_TRAIN_FILE = "mental_heath_unbanlanced.csv"
HF_HUB_TEST_FILE = "mental_health_combined_test.csv"
DATASET_TRAIN_PATH = RAW_DATASETS_DIR / HF_TRAIN_FILE
DATASET_TEST_PATH = RAW_DATASETS_DIR / HF_TEST_FILE

# Stratified multiclass evaluation set (committed; regenerate via src/builders/build_multiclass_dataset.py).
MULTICLASS_EVAL_PATH = PROCESSED_DATASETS_DIR / "multiclass_eval.csv"
MULTICLASS_EVAL_META_PATH = PROCESSED_DATASETS_DIR / "multiclass_eval.meta.json"
MULTICLASS_LABELS: tuple[str, ...] = ("suicidal", "depression", "normal")
MULTICLASS_EXCLUDE: tuple[str, ...] = ("anxiety",)
MULTICLASS_EVAL_PER_CLASS = 150
MULTICLASS_EVAL_SEED = 42
MULTICLASS_MIN_CHARS = 40
MULTICLASS_MAX_CHARS = 2000

# Stratified development slice for prompt/k tuning.
MULTICLASS_DEV_PATH = PROCESSED_DATASETS_DIR / "multiclass_dev.csv"
MULTICLASS_DEV_META_PATH = PROCESSED_DATASETS_DIR / "multiclass_dev.meta.json"
MULTICLASS_DEV_PER_CLASS = 10
MULTICLASS_DEV_SEED = 43

# Backward-compatible aliases used by current scripts/notebooks.
RAG_EVAL_SUBSET_PATH = MULTICLASS_EVAL_PATH
RAG_EVAL_META_PATH = MULTICLASS_EVAL_META_PATH
RAG_EVAL_LABELS = MULTICLASS_LABELS
RAG_EVAL_EXCLUDE = MULTICLASS_EXCLUDE
RAG_EVAL_PER_CLASS = MULTICLASS_EVAL_PER_CLASS
RAG_EVAL_SEED = MULTICLASS_EVAL_SEED
RAG_EVAL_MIN_CHARS = MULTICLASS_MIN_CHARS
RAG_EVAL_MAX_CHARS = MULTICLASS_MAX_CHARS
RAG_DEV_SLICE_PATH = MULTICLASS_DEV_PATH
RAG_DEV_META_PATH = MULTICLASS_DEV_META_PATH
RAG_DEV_PER_CLASS = MULTICLASS_DEV_PER_CLASS
RAG_DEV_SEED = MULTICLASS_DEV_SEED
DATASET_PATH = MULTICLASS_EVAL_PATH

# Experiment run outputs (RAG predictions, summaries, error analysis).
MULTICLASS_RESULTS_FINAL_PATH = RESULTS_DIR / "multiclass_rag_results.csv"
MULTICLASS_RESULTS_SUMMARY_PATH = RESULTS_DIR / "multiclass_summary.csv"
MULTICLASS_ERROR_ANALYSIS_PATH = RESULTS_DIR / "multiclass_error_analysis.csv"

# Backward-compatible aliases.
RAG_RESULTS_FINAL_PATH = MULTICLASS_RESULTS_FINAL_PATH
RAG_RESULTS_SUMMARY_PATH = MULTICLASS_RESULTS_SUMMARY_PATH
RAG_ERROR_ANALYSIS_PATH = MULTICLASS_ERROR_ANALYSIS_PATH

# ── Few-shot vector store (blueprint: planned) ────────────────────────────────
# Built from the multiclass raw dataset (same cleaning/filter pipeline as eval/dev)
# and queried later to assemble dynamic few-shot prompts.
FEWSHOT_STORE_DIR = project_path("knowledge_base", "fewshot")
FEWSHOT_CHROMA_PATH = FEWSHOT_STORE_DIR / "chroma_db"
FEWSHOT_MULTICLASS_EXAMPLES_PATH = FEWSHOT_STORE_DIR / "multiclass_examples.json"
FEWSHOT_DB_META_PATH = PROCESSED_DATASETS_DIR / "fewshot_db.meta.json"

FEWSHOT_COLLECTION_MULTICLASS = "fewshot_multiclass"

# Deterministic per-label sampling caps (keeps the store size reasonable).
FEWSHOT_MULTICLASS_MAX_PER_LABEL = 200
FEWSHOT_BUILD_SEED = 1337

# Embedding model + performance knobs (reuses the same ICD-11 embedding model).
FEWSHOT_EMBEDDING_MAX_SEQ_LENGTH = 512
FEWSHOT_EMBED_BATCH_SIZE = BATCH_SIZE

# ── PDF page range (pdftotext page numbers; printed page ≈ PDF − 18) ──────────
# Clinical CDDR starts at Neurodevelopmental intro / 6A00 (not List of categories).
# End after secondary syndromes (6E6Z); MB symptom appendix begins ~PDF 695.
CONTENT_START_PAGE = 109
CONTENT_END_PAGE   = 694

# Flag disorders with more than this many chunks in the post-chunking report.
CHUNK_COUNT_WARNING_THRESHOLD = 30

# ── ICD-11 code prefix → clinical domain (aligned to CDDR TOC groupings) ──────
# Longer / more specific prefixes must appear before shorter overlapping ones
# when both could match; current keys are disjoint at the 3-char prefix level.
DOMAIN_MAP: dict[str, str] = {
    "6A0": "Neurodevelopmental disorders",
    "6A2": "Schizophrenia and other primary psychotic disorders",
    "6A4": "Catatonia",
    "6A6": "Mood disorders",
    "6A7": "Mood disorders",
    "6A8": "Mood disorders",
    "6B0": "Anxiety and fear-related disorders",
    "6B2": "Obsessive-compulsive and related disorders",
    "6B4": "Disorders specifically associated with stress",
    "6B6": "Dissociative disorders",
    "6B8": "Feeding and eating disorders",
    "6C0": "Elimination disorders",
    "6C2": "Disorders of bodily distress or experience",
    "6C4": "Disorders due to substance use or addictive behaviours",
    "6C5": "Disorders due to substance use or addictive behaviours",
    "6C7": "Impulse control disorders",
    "6C9": "Disruptive behaviour and dissocial disorders",
    "6D1": "Personality disorders",
    "6D3": "Paraphilic disorders",
    "6D5": "Factitious disorders",
    "6D7": "Neurocognitive disorders",
    "6D8": "Neurocognitive disorders",
    "6E0": "Mental or behavioural disorders associated with pregnancy, childbirth or puerperium",
    "6E2": "Psychological and behavioural factors affecting health conditions",
    "6E4": "Secondary mental or behavioural syndromes",
    "6E6": "Secondary mental or behavioural syndromes",
    # Secondary-parented / cross-listed clinical entries in the CDDR
    "8A0": "Neurodevelopmental disorders",  # primary tics / Tourette (Chapter 8)
    "GA3": "Mood disorders",                # Premenstrual dysphoric disorder
}

# ── Retrieval section allowlist ───────────────────────────────────────────────
# Canonical section names passed to retrievers and section_expander. Must match
# the normalised values produced by chunker.normalise_section().
RETRIEVAL_SECTIONS: list[str] = [
    "Essential Features",
    "Boundary with Normality",
]

# ICD-11 disorder-code prefixes for mood / depressive disorders (6A6–6A8).
# Used by the RAG risk-detection notebook to avoid retrieving unrelated domains.
MOOD_DISORDER_PREFIXES: tuple[str, ...] = ("6A6", "6A7", "6A8")

# ── Section heading normalisation map ─────────────────────────────────────────
SECTION_NORMALISE_MAP: dict[str, str] = {
    "essential (required) features":                                           "Essential Features",
    "essential features":                                                      "Essential Features",
    "additional clinical features":                                            "Additional Clinical Features",
    "boundary with normality":                                                 "Boundary with Normality",
    "boundary with normality (threshold)":                                     "Boundary with Normality",
    "course features":                                                         "Course Features",
    "developmental presentations":                                           "Developmental Presentations",
    "culture-related features":                                                "Culture-Related Features",
    "sex- and/or gender-related features":                                     "Sex- and/or Gender-Related Features",
    "boundaries with other disorders and conditions":                          "Differential Diagnosis",
    "boundaries with other disorders and conditions (differential diagnosis)": "Differential Diagnosis",
    "boundaries with other disorders and conditions (differential":            "Differential Diagnosis",
    "diagnostic requirements":                                                 "Diagnostic Requirements",
    "specifiers":                                                              "Specifiers",
}
