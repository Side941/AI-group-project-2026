# AI Group Project 2026 — Depression / Suicide-Risk RAG Pipeline

Prompted RAG classifiers over ICD-11 clinical criteria (Qwen via Ollama). The repo is organized so a first-time runner can distinguish:

- `datasets/raw/` for source CSVs
- `datasets/processed/` for derived eval/dev CSVs and their provenance meta JSON
- `results/` for notebook experiment outputs
- `knowledge_base/` for ICD-11 assets (`icd_11/`) plus the few-shot store
- `src/builders/` for runnable build scripts

## Quick orientation

| Workflow | Script | Notebook | Main output |
|----------|--------|----------|-------------|
| KB build | `src/builders/run_kb_pipeline.py` | `notebooks/01_kb_pipeline_demo.ipynb` | `knowledge_base/icd_11/chroma_db/` |
| Multiclass dataset prep | `src/builders/build_multiclass_dataset.py` | `notebooks/02_multiclass_dataset_prep.ipynb` | `datasets/processed/multiclass_*.csv` |
| Multiclass RAG eval | notebook-driven | `notebooks/03_multiclass_rag.ipynb` | `results/multiclass_*.csv` |
| Rebuild everything + few-shot DB | `src/builders/rebuild_all_artifacts.py` | none | processed CSVs + `knowledge_base/fewshot/` |

## Project structure

```text
AI-group-project-2026/
├── datasets/
│   ├── raw/                     # Source CSVs / local caches
│   └── processed/               # Derived eval/dev CSVs + meta JSON
├── knowledge_base/              # Vector stores + source docs
│   ├── icd_11/                  # ICD-11 PDF, chunks, dense Chroma store
│   │   ├── icd_11.pdf
│   │   ├── icd11_chunks.json
│   │   └── chroma_db/
│   └── fewshot/                 # Few-shot examples JSON + Chroma DB
├── notebooks/                   # Human-facing walkthroughs / experiments
│   ├── 01_kb_pipeline_demo.ipynb
│   ├── 02_multiclass_dataset_prep.ipynb
│   └── 03_multiclass_rag.ipynb
├── experiments/                 # Small pipeline smoke tests (.py)
├── results/                     # Notebook experiment output CSVs
├── src/
│   ├── builders/                # Dataset + KB + few-shot build scripts
│   ├── components/              # Config, chunker, ingestion
│   └── retriever/               # BM25, dense, hybrid, few-shot retrievers
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
| `knowledge_base/icd_11/icd_11.pdf` | Source PDF for chunking |
| `knowledge_base/icd_11/icd11_chunks.json` | Chunked ICD-11 criteria |
| `knowledge_base/icd_11/chroma_db/` | Dense ICD-11 vector store |
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
- `datasets/processed/multiclass_eval.meta.json`
- `datasets/processed/multiclass_dev.meta.json`

Labels: `suicidal`, `depression`, `normal`  
Excluded: `anxiety`

Build or rebuild:

```bash
python src/builders/build_multiclass_dataset.py
```

## Running

### Knowledge-base pipeline

```bash
python src/builders/run_kb_pipeline.py
python src/builders/run_kb_pipeline.py --skip-chunking
python src/builders/run_kb_pipeline.py --rebuild
```

### Rebuild all datasets + few-shot vector DB

```bash
python src/builders/rebuild_all_artifacts.py
```

Useful flags:

```bash
python src/builders/rebuild_all_artifacts.py --skip-evals
python src/builders/rebuild_all_artifacts.py --reuse-fewshot-db
```

### Smoke experiments

Small scripts under `experiments/` to verify the current pipeline without opening notebooks:

```bash
python experiments/smoke_paths.py
python experiments/smoke_retrieval.py --skip-dense
python experiments/smoke_retrieval.py
python experiments/smoke_fewshot.py
python experiments/smoke_rag_oneshot.py          # needs Ollama
python experiments/run_smoke_suite.py --skip-dense
```

### Few-shot retrieval RAG experiment

Dynamic few-shot examples from `knowledge_base/fewshot/` (optional ICD-11 context):

```bash
# Inspect retrieved examples + assembled prompt (no Ollama)
python experiments/exp_fewshot_rag.py --dry-run

# Full run with few-shot BM25 + ICD-11 BM25
python experiments/exp_fewshot_rag.py --query depression

# Few-shot hybrid only (no clinical KB)
python experiments/exp_fewshot_rag.py --kb-retriever none --fewshot-retriever hybrid

# Compare: zero-shot vs few-shot vs few-shot+KB
python experiments/exp_fewshot_rag.py --compare --query suicidal
```

### Notebooks

Run the first path-setup cell before anything else:

- `notebooks/01_kb_pipeline_demo.ipynb`
- `notebooks/02_multiclass_dataset_prep.ipynb`
- `notebooks/03_multiclass_rag.ipynb`

Current experiment outputs are written under `results/`.

## Notes

- Retrievers: `bm25`, `dense`, `hybrid` (weighted RRF).
- The multiclass classifier currently retrieves only from the ICD-11 KB in the notebook.
- The few-shot vector DB can be exercised via `experiments/exp_fewshot_rag.py` (dynamic retrieved examples in the prompt); notebooks still use static few-shot templates.
- CPU is the default fallback; GPU is used when available for embeddings.
