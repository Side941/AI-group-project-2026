<<<<<<< HEAD
# AI Group Project 2026 — RAG Pipeline for Mental Health Risk Detection

## Project Structure

AI-group-project-2026/
├── datasets/                           # Dataset and ChromaDB storage
│   ├── Suicide_Detection.csv           # Raw suicide dataset
│   ├── suicide_train.csv               # Training set (177,672 samples)
│   ├── suicide_test.csv                # Test set (44,419 samples)
│   ├── prepare_data.py                 # Dataset preparation script
│   └── chroma_db/                      # Training examples ChromaDB (auto-built)
│       └── train_suicide/              # 177,672 training embeddings
├── knowledge_based/                    # Knowledge base for clinical criteria
│   ├── mhgap.json                      # WHO mhGAP clinical criteria
│   └── kb_chromadb/                    # Knowledge base ChromaDB (auto-built)
│       └── knowledge_base/             # KB embeddings (3 chunks)
├── experiments/                        # Experiment runners
│   └── run_suicide.py                  # Main experiment runner
├── src/                                # Core source code
│   ├── config.py                       # Shared configuration
│   ├── embedder.py                     # SentenceTransformer wrapper
│   ├── llm_inference.py                # Ollama Qwen3 wrapper with thinking trace
│   ├── prompt_builder.py               # Prompt templates
│   ├── evaluate.py                     # Experiment runner with debug output
│   ├── vector_store.py                 # ChromaDB wrapper
│   └── retrievers/                     # KB retrieval implementations
│       ├── base.py                     # Abstract base class
│       ├── bm25_retriever.py           # Sparse keyword retrieval on KB
│       ├── dense_retriever.py          # Dense semantic retrieval on KB
│       └── hybrid_retriever.py         # BM25 + dense fusion on KB
├── results/                            # Experiment results (auto-generated)
├── data/                               # Source data (not committed)
│   ├── ICD-11-CDDR.pdf
│   ├── WHO-mhGAP-intervention-guide.pdf
│   └── depression.txt
├── notebooks/
│   └── analysis.ipynb                  # Analysis notebook
├── requirements.txt
└── README.md

All paths are resolved from the repo root via src/config.py.

## Setup

1. Clone the repo
   git clone https://github.com/Side941/AI-group-project-2026.git
   cd AI-group-project-2026

2. Install dependencies
   pip install -r requirements.txt

3. Install Ollama
   curl -fsSL https://ollama.com/install.sh | sh
   ollama pull qwen3:0.6b
   ollama pull qwen3:1.7b
   ollama serve

4. Prepare datasets
   python3 datasets/prepare_data.py

Note: ChromaDB collections are auto-built on first run. No manual indexing needed.

## Running Experiments

Quick test: 1 sample, 1 experiment (zero-shot)
   python3 experiments/run_suicide.py --samples 1 --limit 1 --debug

KB retrieval: 1 sample, 2 experiments (zero-shot + KB)
   python3 experiments/run_suicide.py --samples 1 --limit 2 --debug

Combined mode (KB + Reddit examples): 1 sample, 10 experiments
   python3 experiments/run_suicide.py --samples 1 --limit 10 --debug

Full run with 10 samples
   python3 experiments/run_suicide.py --samples 10

Run all experiments with 100 samples
   python3 experiments/run_suicide.py --samples 100

Arguments:

   --samples   Number of test samples (default: 10)
   --limit     Number of experiments (default: All)
   --debug     Show debug output (default: False)

## Path Configuration (src/config.py)

   PROJECT_ROOT         Repo root
   SUICIDE_TRAIN_PATH   datasets/suicide_train.csv
   SUICIDE_TEST_PATH    datasets/suicide_test.csv
   MHGAP_SUICIDE_PATH   knowledge_based/mhgap.json
   CHROMA_PATH          datasets/chroma_db
   KB_CHROMA_PATH       knowledge_based/kb_chromadb
   LLM_MODEL_SIZES      ["0.6b", "1.7b"]
   KB_COLLECTION_NAME   "knowledge_base"

## Results

Saved to results/:

   summary_suicide_*.csv    All experiment summaries
   suicide_*_results.csv    Detailed predictions per experiment

Example Debug Output:

   🧠 Model Thinking/Reasoning:
   The clinical criteria mention suicide risk based on thoughts or plans...
   First, I need to check if the text contains any signs of imminent suicide risk...
   Therefore, the text doesn't indicate a suicide risk as per the criteria provided.

   📚 Retrieved KB Chunks (3):
     [1] Section: Assessment - Imminent Risk
         Text: Imminent risk of self-harm/suicide is indicated by...

## Notes

   - ChromaDB auto-builds on first run
   - Results are cached for faster re-runs
   - Default Top-K is 3, adjustable per experiment
   - Runs on CPU/MPS/GPU
   - Debug mode shows model thinking trace (🧠) for reasoning transparency
   - KB retrieval uses mhGAP clinical criteria from WHO
=======
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
| Binary dataset prep | `src/builders/build_binary_dataset.py` | `notebooks/04_binary_rag.ipynb` | `datasets/processed/binary_suicide_*.csv` |
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
│   ├── 03_multiclass_rag.ipynb
│   └── 04_binary_rag.ipynb
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
- `datasets/processed/multiclass_eval.meta.json`
- `datasets/processed/multiclass_dev.meta.json`

Labels: `suicidal`, `depression`, `normal`  
Excluded: `anxiety`

Build or rebuild:

```bash
python src/builders/build_multiclass_dataset.py
```

### Binary suicide

Raw input:
- `datasets/raw/suicide_detection_raw.csv`

Processed outputs:
- `datasets/processed/binary_suicide_eval.csv` — 300 posts (150/class)
- `datasets/processed/binary_suicide_dev.csv` — 20 posts (10/class)
- `datasets/processed/binary_suicide_eval.meta.json`
- `datasets/processed/binary_suicide_dev.meta.json`

Labels: `suicide`, `non-suicide`

Build or rebuild:

```bash
python src/builders/build_binary_dataset.py
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

### Notebooks

Run the first path-setup cell before anything else:

- `notebooks/01_kb_pipeline_demo.ipynb`
- `notebooks/02_multiclass_dataset_prep.ipynb`
- `notebooks/03_multiclass_rag.ipynb`
- `notebooks/04_binary_rag.ipynb`

Current experiment outputs are written under `results/`.

## Notes

- Retrievers: `bm25`, `dense`, `hybrid` (weighted RRF).
- Both classifiers currently retrieve only from the ICD-11 KB.
- The few-shot vector DB is built and stored separately, but it is not wired into prompt retrieval yet.
- CPU is the default fallback; GPU is used when available for embeddings.
>>>>>>> rami-experiment
