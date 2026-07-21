# AI Group Project 2026 — Depression / Suicide-Risk RAG Pipeline

Prompted RAG classifiers over ICD-11 clinical criteria (Qwen via Ollama). The repo is organized so a first-time runner can distinguish:

- `datasets/raw/` for source CSVs
- `datasets/processed/` for derived eval/dev CSVs
- `results/metadata/` for provenance files
- `results/experiments/` for notebook outputs
- `vector_stores/` for non-ICD Chroma stores such as the few-shot DB

## Quick orientation

| Workflow | Script | Notebook | Main output |
|----------|--------|----------|-------------|
| KB build | `experiments/run_kb_pipeline.py` | `notebooks/01_kb_pipeline_demo.ipynb` | `knowledge_base/chroma_db/` |
| Multiclass dataset prep | `experiments/build_multiclass_dataset.py` | `notebooks/02_multiclass_dataset_prep.ipynb` | `datasets/processed/multiclass_*.csv` |
| Multiclass RAG eval | notebook-driven | `notebooks/03_multiclass_rag.ipynb` | `results/experiments/multiclass_*.csv` |
| Binary dataset prep | `experiments/build_binary_dataset.py` | `notebooks/04_binary_rag.ipynb` | `datasets/processed/binary_suicide_*.csv` |
| Rebuild everything + few-shot DB | `experiments/rebuild_all_artifacts.py` | none | processed CSVs + `vector_stores/fewshot_chroma_db/` |

## Project structure

```text
AI-group-project-2026/
├── datasets/
│   ├── raw/                     # Source CSVs / local caches
│   └── processed/               # Derived eval/dev datasets used by notebooks
├── experiments/                 # Runnable scripts only
│   ├── run_kb_pipeline.py
│   ├── build_multiclass_dataset.py
│   ├── build_binary_dataset.py
│   └── rebuild_all_artifacts.py
├── knowledge_base/              # ICD-11 assets and main Chroma store
│   ├── icd11_chunks.json
│   └── chroma_db/
├── notebooks/                   # Human-facing walkthroughs / experiments
│   ├── 01_kb_pipeline_demo.ipynb
│   ├── 02_multiclass_dataset_prep.ipynb
│   ├── 03_multiclass_rag.ipynb
│   └── 04_binary_rag.ipynb
├── results/
│   ├── metadata/                # Dataset + few-shot provenance JSON
│   └── experiments/             # Notebook output CSVs
├── src/
│   ├── components/              # Config, chunker, ingestion
│   └── retriever/               # BM25, dense, hybrid, section expander
├── vector_stores/
│   └── fewshot_chroma_db/       # Planned few-shot retrieval store
├── rag_system_blueprint.html
├── requirements.txt
└── README.md
```

All paths resolve from the repo root through `src/components/config.py`.

## Setup

```bash
git clone https://github.com/Side941/AI-group-project-2026.git
cd AI-group-project-2026
pip install -r requirements.txt
```

Also needed locally:

| Path | Role |
|------|------|
| `knowledge_base/icd_11.pdf` | Source PDF for chunking |
| `knowledge_base/icd11_chunks.json` | Chunked ICD-11 criteria |
| `knowledge_base/chroma_db/` | Dense ICD-11 vector store |
| `datasets/raw/suicide_detection_raw.csv` | Binary source dataset |
| Ollama + `qwen3:0.6b` / `qwen3:1.7b` | LLM inference for notebooks |

The multiclass raw caches under `datasets/raw/` are created automatically if missing.

## Datasets

### Multiclass

Raw/cache inputs:
- `datasets/raw/mental_health_train.csv`
- `datasets/raw/mental_health_test.csv`

Processed outputs:
- `datasets/processed/multiclass_eval.csv` — 450 posts (150/class)
- `datasets/processed/multiclass_dev.csv` — 30 posts (10/class)
- `results/metadata/multiclass_eval.meta.json`
- `results/metadata/multiclass_dev.meta.json`

Labels: `suicidal`, `depression`, `normal`  
Excluded: `anxiety`

Build or rebuild:

```bash
python experiments/build_multiclass_dataset.py
```

### Binary suicide

Raw input:
- `datasets/raw/suicide_detection_raw.csv`

Processed outputs:
- `datasets/processed/binary_suicide_eval.csv` — 300 posts (150/class)
- `datasets/processed/binary_suicide_dev.csv` — 20 posts (10/class)
- `results/metadata/binary_suicide_eval.meta.json`
- `results/metadata/binary_suicide_dev.meta.json`

Labels: `suicide`, `non-suicide`

Build or rebuild:

```bash
python experiments/build_binary_dataset.py
```

## Running

### Knowledge-base pipeline

```bash
python experiments/run_kb_pipeline.py
python experiments/run_kb_pipeline.py --skip-chunking
python experiments/run_kb_pipeline.py --rebuild
```

### Rebuild all datasets + few-shot vector DB

```bash
python experiments/rebuild_all_artifacts.py
```

Useful flags:

```bash
python experiments/rebuild_all_artifacts.py --skip-evals
python experiments/rebuild_all_artifacts.py --reuse-fewshot-db
```

### Notebooks

Run the first path-setup cell before anything else:

- `notebooks/01_kb_pipeline_demo.ipynb`
- `notebooks/02_multiclass_dataset_prep.ipynb`
- `notebooks/03_multiclass_rag.ipynb`
- `notebooks/04_binary_rag.ipynb`

Current experiment outputs are written under `results/experiments/`.

## Notes

- Retrievers: `bm25`, `dense`, `hybrid` (weighted RRF).
- Both classifiers currently retrieve only from the ICD-11 KB.
- The few-shot vector DB is built and stored separately, but it is not wired into prompt retrieval yet.
- CPU is the default fallback; GPU is used when available for embeddings.
