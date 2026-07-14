"""
config.py
=========
Shared configuration and project paths for the ICD-11 knowledge base pipeline.

All filesystem paths are resolved relative to PROJECT_ROOT (the repo root),
so imports work whether you run from the project root, retriever/, or notebooks/.
"""

from __future__ import annotations

from pathlib import Path

# Repo root: parent of the components/ package directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent


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


# ── Paths ──────────────────────────────────────────────────────────────────────

PDF_PATH     = project_path("knowledge_base", "icd_11.pdf")
CHUNKS_PATH  = project_path("knowledge_base", "icd11_chunks.json")
CHROMA_PATH  = project_path("knowledge_base", "chroma_db")
# Hugging Face mental-health corpus (local cache + Hub fallback).
HF_DATASET_REPO = "ourafla/Mental-Health_Text-Classification_Dataset"
HF_TRAIN_FILE = "mental_heath_unbanlanced.csv"
HF_TEST_FILE = "mental_health_combined_test.csv"
DATASET_TRAIN_PATH = project_path("datasets", HF_TRAIN_FILE)
DATASET_TEST_PATH = project_path("datasets", HF_TEST_FILE)

# Stratified RAG evaluation subset (committed; regenerate via datasets/build_rag_eval_subset.py).
RAG_EVAL_SUBSET_PATH = project_path("datasets", "rag_eval_subset.csv")
RAG_EVAL_META_PATH = project_path("datasets", "rag_eval_subset.meta.json")
RAG_EVAL_LABELS: tuple[str, ...] = ("suicidal", "depression", "normal")
RAG_EVAL_EXCLUDE: tuple[str, ...] = ("anxiety",)
RAG_EVAL_PER_CLASS = 150
RAG_EVAL_SEED = 42
RAG_EVAL_MIN_CHARS = 40
RAG_EVAL_MAX_CHARS = 2000

# Stratified development slice for prompt/k tuning (subset of the final eval set).
# Use eval_mode="dev" in the notebook while iterating; switch to "final" for reporting.
RAG_DEV_SLICE_PATH = project_path("datasets", "rag_dev_slice.csv")
RAG_DEV_META_PATH = project_path("datasets", "rag_dev_slice.meta.json")
RAG_DEV_PER_CLASS = 10
RAG_DEV_SEED = 43

# Default notebook dataset path points at the final eval set.
DATASET_PATH = RAG_EVAL_SUBSET_PATH

# ── Model / collection ─────────────────────────────────────────────────────────
COLLECTION_NAME = "icd11_clinical"
EMBEDDING_MODEL = "FremyCompany/BioLORD-2023"
BATCH_SIZE      = 64

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
