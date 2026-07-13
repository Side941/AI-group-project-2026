# AI Group Project 2026 — RAG Pipeline for Mental Health Risk Detection

## Project Structure

```
AI-group-project-2026/
├── components/                 # Knowledge-base build pipeline
│   ├── config.py               # Shared config + repo-root path resolution
│   ├── chunker.py              # Step 1: ICD-11 PDF → JSON chunks
│   ├── ingestion.py            # Step 2: JSON chunks → ChromaDB vectors
│   └── main.py                 # Run chunking + ingestion in one command
├── retriever/                  # Retrieval layer
│   ├── utils.py                # Load chunks, tokenize for BM25
│   ├── bm25_retriever.py       # Sparse keyword retrieval
│   ├── dense_retriever.py      # Dense Chroma retrieval + DenseRetriever
│   ├── hybrid_retriever.py     # BM25 + dense fusion (weighted RRF)
│   └── section_expander.py     # Post-retrieval section completion
├── notebooks/
│   ├── 01_kb_pipeline_demo.ipynb   # Documented KB pipeline (chunk + ingest + inspect)
│   ├── 02_dataset_prep_demo.ipynb  # Documented eval/dev dataset builder + inspect
│   └── multi_class_rag.ipynb       # Full RAG classification experiment
├── data/
│   └── icd_11.pdf              # Source ICD-11 PDF (not committed)
├── knowledge_based/
│   ├── icd11_chunks.json       # Extracted clinical chunks
│   └── chroma_db/              # Vector store (gitignored)
├── datasets/
│   ├── build_rag_eval_subset.py   # Reproducible eval + dev-slice builder
│   ├── rag_eval_subset.csv        # Final reporting set (450 rows)
│   ├── rag_eval_subset.meta.json  # Provenance + SHA256 of the final set
│   ├── rag_dev_slice.csv          # Prompt/k tuning slice (30 rows)
│   ├── rag_dev_slice.meta.json    # Provenance + parent row_ids
│   ├── mental_heath_unbanlanced.csv          # Optional local HF cache (gitignored)
│   └── mental_health_combined_test.csv       # Optional local HF cache (gitignored)
├── requirements.txt
└── README.md
```

All paths are resolved from the **repo root** via `components/config.py`, so code works whether you run from the project root, `retriever/`, or `notebooks/`.

## Setup

**1. Clone the repo**
```bash
git clone https://github.com/Side941/AI-group-project-2026.git
cd AI-group-project-2026
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Place required data files**
- `data/icd_11.pdf`
- `knowledge_based/icd11_chunks.json` (from chunker, or provided)
- `knowledge_based/chroma_db/` (from ingestion, or provided)
- `datasets/rag_eval_subset.csv` (committed final set; regenerate below if needed)
- `datasets/rag_dev_slice.csv` (committed 30-post tuning slice)

## RAG evaluation sets

Two committed stratified artifacts (anxiety excluded):

| File | Size | Purpose |
|------|------|---------|
| `rag_dev_slice.csv` | 30 (10/class) | Prompt / top-k / alpha tuning (`eval_mode="dev"`) |
| `rag_eval_subset.csv` | 450 (150/class) | Final reported results (`eval_mode="final"`) |

The dev slice is carved from the final set (see `parent_row_id`), so tuning posts are a transparent subset of the reporting set. Labels: `suicidal` | `depression` | `normal`.

In the notebook, switch modes via `CFG["eval_mode"]` (`"dev"` or `"final"`). Default is `"dev"`. Do not keep editing prompts after switching to `"final"`.

Regenerate both (deterministic; final seed `42`, dev seed `43`):

```bash
python datasets/build_rag_eval_subset.py
```

Load order for the builder:
1. Local CSVs `datasets/mental_heath_unbanlanced.csv` + `datasets/mental_health_combined_test.csv` if both exist
2. Otherwise download from Hugging Face (`ourafla/Mental-Health_Text-Classification_Dataset`) and cache those CSVs locally

Raw HF dumps stay gitignored; the eval/dev CSVs and their `.meta.json` files are tracked so teammates can run without re-downloading.

## Running

**Knowledge-base pipeline** (from project root):
```bash
python -m components.main

# Optional: ingestion only (reuse existing chunks)
python -m components.main --skip-chunking

# Optional: force re-ingestion into Chroma
python -m components.main --rebuild

# Existing step-by-step commands
python -m components.chunker
python -m components.ingestion
```

**Dataset rebuild** (eval + dev slices):
```bash
python datasets/build_rag_eval_subset.py
```

**Notebook experiment** — open `notebooks/multi_class_rag.ipynb` and run cell 1 first. It auto-detects the project root.

**Demo notebooks** (documented pipelines with explanations + inspection; CLIs remain for automation):
- `notebooks/01_kb_pipeline_demo.ipynb` — KB orchestration (chunk → ingest) as in `components/main.py`
- `notebooks/02_dataset_prep_demo.ipynb` — full eval/dev dataset builder stages as in `datasets/build_rag_eval_subset.py`

Rebuild flags default to off / write-on where noted; read the markdown cells before flipping them.

## Path configuration

Edit paths in one place: `components/config.py`

| Constant | Points to |
|----------|-----------|
| `PROJECT_ROOT` | Repo root (auto-detected) |
| `PDF_PATH` | `data/icd_11.pdf` |
| `CHUNKS_PATH` | `knowledge_based/icd11_chunks.json` |
| `CHROMA_PATH` | `knowledge_based/chroma_db` |
| `RAG_EVAL_SUBSET_PATH` / `DATASET_PATH` | `datasets/rag_eval_subset.csv` (final) |
| `RAG_DEV_SLICE_PATH` | `datasets/rag_dev_slice.csv` (tuning) |
| `DATASET_TRAIN_PATH` / `DATASET_TEST_PATH` | Local HF mental-health CSV caches |

## Notes
- Default Top-K is 5, adjustable via the `k` parameter in `search()`.
- Runs on CPU by default; uses GPU when available.
- Retriever types: `bm25`, `hybrid`, `dense`.
- Hybrid fusion uses weighted Reciprocal Rank Fusion (RRF).
