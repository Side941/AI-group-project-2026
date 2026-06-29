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

PDF_PATH     = project_path("data", "icd_11.pdf")
CHUNKS_PATH  = project_path("knowledge_based", "icd11_chunks.json")
CHROMA_PATH  = project_path("knowledge_based", "chroma_db")
DATASET_PATH = project_path(
    "datasets",
    "Depression_Severity_Levels_Dataset.csv",
)

# ── Model / collection ─────────────────────────────────────────────────────────
COLLECTION_NAME = "icd11_clinical"
EMBEDDING_MODEL = "FremyCompany/BioLORD-2023"
BATCH_SIZE      = 64

# ── PDF page range — clinical content starts around page 70 ───────────────────
CONTENT_START_PAGE = 70
CONTENT_END_PAGE   = 852

# ── ICD-11 code prefix → clinical domain ──────────────────────────────────────
DOMAIN_MAP: dict[str, str] = {
    "6A0": "Neurodevelopmental disorders",
    "6A2": "Schizophrenia and other primary psychotic disorders",
    "6A4": "Catatonia",
    "6A6": "Mood disorders",
    "6A7": "Mood disorders",
    "6A8": "Mood disorders",
    "6B0": "Anxiety and fear-related disorders",
    "6B1": "Obsessive-compulsive and related disorders",
    "6B2": "Disorders specifically associated with stress",
    "6B4": "Dissociative disorders",
    "6B6": "Feeding and eating disorders",
    "6B8": "Elimination disorders",
    "6C0": "Disorders of bodily distress or experience",
    "6C2": "Disorders of bodily distress or experience",
    "6C4": "Disorders due to substance use or addictive behaviours",
    "6C5": "Disorders due to substance use or addictive behaviours",
    "6C6": "Disorders due to substance use or addictive behaviours",
    "6C7": "Disorders due to substance use or addictive behaviours",
    "6C9": "Impulse control disorders",
    "6D1": "Disruptive behaviour and dissocial disorders",
    "6D3": "Personality disorders",
    "6D4": "Paraphilic disorders",
    "6D5": "Factitious disorders",
    "6D6": "Neurocognitive disorders",
    "6D7": "Neurocognitive disorders",
    "6D8": "Neurocognitive disorders",
    "6E0": "Mental or behavioural disorders associated with pregnancy, childbirth or puerperium",
    "6E2": "Psychological and behavioural factors affecting health conditions",
    "6E4": "Secondary mental or behavioural syndromes",
    "6E6": "Secondary mental or behavioural syndromes",
}

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
    "diagnostic requirements":                                                 "Diagnostic Requirements",
    "specifiers":                                                              "Specifiers",
}
