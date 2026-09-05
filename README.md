# AI Group Project 2026 — Depression / Suicide-Risk Detection with RAG

Can a small, locally-run language model classify social-media posts as **suicidal**,
**depression** or **normal** *and* justify itself against real clinical criteria?

This repo answers that with a retrieval-augmented generation (RAG) pipeline: ICD-11
clinical descriptions (the WHO CDDR) are chunked into a vector store, the most relevant
diagnostic criteria are retrieved for each post, and a Qwen3 model running locally under
Ollama produces a label plus a justification that cites the retrieved evidence. A
nine-phase ablation isolates the contribution of each component, and the final system is
compared against a supervised fine-tuned MentalRoBERTa baseline on the same held-out set.

**Labels:** `suicidal`, `depression`, `normal` (the source corpus's `anxiety` class is
excluded).

## Contents

- [How the pipeline works](#how-the-pipeline-works)
- [Headline results](#headline-results)
- [Repository layout](#repository-layout)
- [Prerequisites](#prerequisites)
- [Reproducing the results](#reproducing-the-results)
- [The two notebooks](#the-two-notebooks)
- [Output files](#output-files)
- [Datasets](#datasets)
- [Knowledge base and few-shot store](#knowledge-base-and-few-shot-store)
- [Supporting scripts](#supporting-scripts)
- [Known limitations](#known-limitations)
- [Generative AI use](#generative-ai-use)

## How the pipeline works

For each post:

1. **Retrieve** candidate ICD-11 chunks with a hybrid retriever — BM25 fused with dense
   BioLORD embeddings via weighted RRF (`alpha = 0.3`).
2. **Scope** the results to mood- and risk-related ICD-11 code prefixes
   (`6A6`, `6A7`, `6A8`, `EP.`, `6B42`, `6B43`, `6D11`), so unrelated chapters cannot
   crowd out the relevant criteria.
3. **Expand** each retrieved chunk to its sibling sections (Essential Features and
   Boundary with Normality) so a criterion is never shown half-truncated.
4. **Prompt** `qwen3:1.7b` zero-shot with the post and the retrieved evidence, requiring a
   `Label:` and a `Justification:` that cites the ICD-11 material.
5. **Parse and score** the response: accuracy and macro-F1 for classification, plus manual
   grounding criteria C1 (is the citation valid), C2 (is the retrieved category relevant)
   and C3 (is the retrieved content relevant).

That configuration is not assumed — it is the outcome of the ablation in
`notebooks/analysis.ipynb`, where every component was varied one at a time.

## Headline results

Final configuration, 450-post held-out evaluation set, three identical runs (Phase 9):

| System | Accuracy | Macro-F1 | Suicidal recall |
|--------|----------|----------|-----------------|
| RAG pipeline (Qwen3 1.7B, hybrid + scoped, k=5) | 0.678 | 0.644 | 0.787 |
| MentalRoBERTa, supervised fine-tune | 0.871 | 0.871 | 0.860 |

The three RAG runs agreed on 100% of predictions, so the pipeline is deterministic at
`seed=42`. The supervised baseline is significantly more accurate overall (McNemar's exact
test on 450 paired outcomes, p ≈ 0; paired bootstrap on the accuracy difference −0.193,
95% CI [−0.240, −0.149]), but the gap on **suicidal recall alone** is not significant
(p = 0.108, n = 150) — and MentalRoBERTa needs ~9,000 labelled training posts and produces
no auditable justification, whereas the RAG system trains nothing and cites its evidence.

## Repository layout

```text
AI-group-project-2026/
├── datasets/
│   ├── raw/                  # Source corpus cache (gitignored, auto-downloaded)
│   └── processed/            # Committed eval/dev splits + provenance meta JSON
├── knowledge_base/
│   ├── icd_11/               # ICD-11 chunks (committed), PDF + Chroma store (gitignored)
│   └── fewshot/              # Few-shot example store (gitignored, rebuildable)
├── notebooks/
│   ├── analysis.ipynb                # Main experiment: ablation Phases 1–9
│   └── 05_external_comparison.ipynb  # RAG vs MentalRoBERTa comparison
├── src/
│   ├── components/           # Config, PDF chunker, Chroma ingestion
│   ├── retriever/            # BM25, dense, hybrid, few-shot, section expander
│   ├── builders/             # Build scripts for the KB, datasets and few-shot store
│   └── evaluation/           # Paired significance testing + winner-selection rule
├── external_model/           # MentalRoBERTa supervised baseline
├── analysis/                 # Standalone pre-experiment scripts + their .txt reports
├── experiments/              # Few-shot RAG prompt-inspection script
├── benchmarking/             # Early model/latency benchmarks (historical)
├── scripts/                  # KB snapshot tool + few-shot curation helper
├── results/                  # All experiment outputs
├── KNOWN_ISSUES.md           # Knowledge-base defect report (KB v1)
└── requirements.txt
```

All filesystem paths resolve from the repo root via `src/components/config.py`, so scripts
and notebooks work regardless of the directory you launch them from.

Only the two notebooks listed above are part of the reported pipeline. Any other notebook
files present in `notebooks/` are earlier exploratory work and are not used to produce the
results in this README.

## Prerequisites

### Python packages

```bash
pip install -r requirements.txt
```

This covers everything: the core retrieval pipeline, the notebook analysis stack
(matplotlib, seaborn, scikit-learn, statsmodels) and the MentalRoBERTa baseline
(transformers, accelerate). NLTK's `punkt` and `stopwords` corpora download automatically
on first use.

### Ollama

The RAG notebook talks to Ollama over plain HTTP at `http://localhost:11434`; there is no
Python client dependency. Install Ollama, then:

```bash
ollama pull qwen3:1.7b
ollama pull qwen3:0.6b   # only needed for the Phase 5 model-size ablation
```

### poppler / `pdftotext`

Only needed if you rebuild the knowledge base from the PDF. `src/components/chunker.py`
shells out to `pdftotext`:

```bash
conda install -c conda-forge poppler
```

### Files you must supply yourself

| Path | How to obtain | Needed for |
|------|---------------|------------|
| `knowledge_base/icd_11/icd_11.pdf` | The WHO ICD-11 CDDR PDF. Not redistributed here for copyright reasons — place your own copy at this path. | Rebuilding the KB from scratch |
| Hugging Face account with access to `mental/mental-roberta-base` | The model is gated; run `huggingface-cli login` after requesting access. | MentalRoBERTa baseline only |

The raw corpus under `datasets/raw/` downloads automatically from the Hugging Face dataset
`ourafla/Mental-Health_Text-Classification_Dataset` if it is missing.

## Reproducing the results

```bash
git clone https://github.com/Side941/AI-group-project-2026.git
cd AI-group-project-2026
pip install -r requirements.txt
```

Then work through the steps below. Steps 1–3 build artifacts that are gitignored, so a
fresh clone needs all of them before the notebooks will run.

### Step 1 — Build the ICD-11 knowledge base

Chunks the PDF and embeds the chunks into Chroma. Requires `icd_11.pdf` and `pdftotext`.

```bash
python src/builders/run_kb_pipeline.py
```

Produces `knowledge_base/icd_11/icd11_chunks.json` (1,581 chunks, already committed) and
the Chroma collection `icd11_clinical` under `knowledge_base/icd_11/chroma_db/`.

Because the chunks JSON is committed, you can skip re-chunking and only build the vector
store — this avoids needing the PDF at all:

```bash
python src/builders/run_kb_pipeline.py --skip-chunking
python src/builders/run_kb_pipeline.py --rebuild        # force re-ingest from scratch
```

Embedding uses `FremyCompany/BioLORD-2023` on GPU when available, CPU otherwise.

### Step 2 — Build the evaluation splits

Both split CSVs are committed, so this is only needed if you want to regenerate them:

```bash
python src/builders/build_multiclass_dataset.py
```

### Step 3 — Build the few-shot store

`knowledge_base/fewshot/` is gitignored in its entirety, so this step is required on a
fresh clone even though only Phase 3 uses few-shot prompting:

```bash
python src/builders/rebuild_all_artifacts.py     # sampled pool + curated Chroma vectors
python scripts/make_curated_fewshot.py           # curated JSON from the pool
python src/builders/build_curated_fewshot_db.py  # embed the curated JSON
```

`rebuild_all_artifacts.py` regenerates the eval/dev CSVs too; pass `--skip-evals` to leave
them alone, and `--reuse-fewshot-db` to skip re-embedding when the vector counts already
match. Eval posts are held out of the few-shot pool, so retrieved examples cannot leak into
evaluation. Provenance is written to `datasets/processed/fewshot_db.meta.json`.

BM25 few-shot retrieval reads the curated JSON directly and needs no build step; only the
dense path needs the Chroma collection.

### Step 4 — Run the ablation

Start Ollama, then run `notebooks/analysis.ipynb` (see below for how it is structured).

### Step 5 — Run the external baseline and comparison

```bash
huggingface-cli login
python external_model/mentalroberta_baseline.py
```

Then run `notebooks/05_external_comparison.ipynb`.

## The two notebooks

### `notebooks/analysis.ipynb` — the main experiment

A nine-phase ablation. Phases 1–8 run on the 30-post dev split so that every response can
be manually graded for grounding; Phase 9 runs the winning configuration on the full
450-post evaluation set.

Each phase varies exactly one component and keeps everything else at the configuration
inherited from earlier phases. The starting baseline is `qwen3:1.7b`, `k=5`, zero-shot,
thinking off, expand strategy, full ICD-11 knowledge base.

| Phase | Varies | Outcome | Output |
|-------|--------|---------|--------|
| 1 | Retriever: BM25 / dense / hybrid | Hybrid — best grounding | `results/phase1_grounding_review.csv` |
| 2 | Scoped mood-code filter vs full KB | Scoped filter kept | `results/phase2_hybrid_scoped.csv` |
| 3 | Dynamic few-shot vs zero-shot | Zero-shot kept | `results/phase3_fewshot_scoped.csv` |
| 4 | Qwen3 thinking mode on/off | Thinking rejected | `results/phase4_thinking.csv` |
| 5 | `qwen3:0.6b` vs `qwen3:1.7b` | 1.7B kept | `results/phase5_06b.csv` |
| 6 | Section expansion vs flat top-k | Expansion kept | `results/phase6_flat.csv` |
| 7 | `k` = 3 and 8 vs 5 | k=5 kept | `results/phase7_k_check.csv` |
| 8 | RAG vs a fair no-retrieval prompt | RAG kept | `results/phase8_no_rag_fair.csv` |
| 9 | Nothing — 3 repeat runs on 450 posts | Final numbers | `results/phase9_run{1,2,3}.csv` |

**Running it.** Execute the setup cells (environment, imports, `CFG`, prompt templates,
dataset loading, retriever setup, Ollama helpers, phase runner) once, then run the phases
in order. Phase 2's in-memory `results` variable is the paired baseline for the McNemar
tests in Phases 3, 4, 6, 7 and 8, so skipping Phase 2 breaks those comparisons. Phase 9 is
self-contained apart from the setup cells, and switches the dataset to the 450-post eval
set itself.

**Configuration.** Everything tunable lives in the `CFG` dict in the configuration cell:
Ollama host and timeout, model list, `seed=42`, `num_predict`, `num_ctx`, `hybrid_alpha`,
and `eval_mode` (`"dev"` for 30 posts, `"final"` for 450). Phase 4 overrides `num_predict`
to 2048 and `num_ctx` to 16384 to give thinking mode room.

**Runtime.** Roughly 5–20 s per post for 1.7B with retrieval. A dev phase (30 posts) takes
a few minutes; Phase 9 is 1,350 generations and takes on the order of three hours.

**Grounding scores** are entered by hand as `manual_scores_phaseN` dicts in the notebook
and merged into the output CSVs. The `results/p*.txt` files are the raw C1/C2/C3 tally
sheets from that manual review — the notebook neither reads nor writes them.

The confusion-matrix cell writes `confusion_matrix_rag.png` to the working directory.

### `notebooks/05_external_comparison.ipynb` — RAG vs supervised baseline

A read-only analysis notebook: it runs no models and writes no files. It loads
`results/phase9_run1.csv` and `results/mentalroberta_eval_predictions.csv`, checks that
both cover the same 450 posts with the same ground truth, and compares them.

Sections: headline metrics → side-by-side confusion matrices (raw and row-normalised) →
per-class precision/recall/F1 → significance testing → agreement analysis → cost and
qualitative trade-offs → notes for the write-up.

Because both systems scored the same posts, the outcomes are paired, so significance uses
McNemar's exact test and a paired bootstrap on the accuracy difference (10,000 replicates,
seed 42) from `src/evaluation/selection.py` rather than overlapping per-system confidence
intervals. The agreement section reports where each system is uniquely correct and the
resulting oracle ceiling.

Both input CSVs must exist first. The MentalRoBERTa side comes from
`external_model/mentalroberta_baseline.py`, which fine-tunes `mental/mental-roberta-base`
with a 3-class head, excludes evaluation posts from training by normalised text hash, and
writes the predictions CSV:

```bash
python external_model/mentalroberta_baseline.py            # 9000 train / 1500 val, 3 epochs
python external_model/mentalroberta_baseline.py --smoke    # 300/150, 1 epoch, pipeline check
python external_model/mentalroberta_baseline.py --device cpu --epochs 2
```

Fine-tuning RoBERTa-base on CPU is slow; `--device auto` picks CUDA, then MPS, then CPU.
If local training is impractical, train on a Colab T4 and copy
`results/mentalroberta_eval_predictions.csv` back into `results/`. The script runs a
gradient-flow preflight check (some MPS/transformers combinations silently fail to train)
and warns if validation accuracy lands near chance.

Note that `mental/mental-roberta-base` is licensed CC BY-NC 4.0 — non-commercial use only.

## Output files

| File | Contents |
|------|----------|
| `results/phase1_grounding_review.csv` … `results/phase8_no_rag_fair.csv` | Per-post predictions, retrieved chunks, full model answers and grounding scores for each ablation phase (30 posts per configuration) |
| `results/phase9_run{1,2,3}.csv` | Final 450-post runs. Columns: `response_id`, `config`, `post_index`, `true_label`, `predicted`, `full_answer`, `retrieved_chunks`, `correct`, `c1`, `c2`, `c3`, `grounding_score` |
| `results/mentalroberta_eval_predictions.csv` | Baseline predictions with per-class probabilities |
| `results/p*.txt` | Manual C1/C2/C3 grounding tally sheets, one per phase |
| `analysis/*_report.txt` | Pre-experiment reports, regenerated by the matching script |

## Datasets

Source corpus: `ourafla/Mental-Health_Text-Classification_Dataset` on Hugging Face, cached
to `datasets/raw/` (gitignored, downloaded on demand).

Committed, deterministic splits under `datasets/processed/`:

| File | Size | Seed |
|------|------|------|
| `multiclass_eval.csv` | 450 posts, 150 per class | 42 |
| `multiclass_dev.csv` | 30 posts, 10 per class | 43 |

Each has a `.meta.json` recording how it was produced. The dev split is nested inside the
eval split, and posts are filtered to 40–2000 characters. Regenerate with
`python src/builders/build_multiclass_dataset.py`.

## Knowledge base and few-shot store

**ICD-11.** `knowledge_base/icd_11/icd11_chunks.json` holds 1,581 chunks covering PDF pages
92–694 of the CDDR, each carrying a `disorder_code`, `disorder_name`, `section`, `domain`
and a globally unique `chunk_uid`. The BM25 and hybrid paths index only the two most
diagnostic sections, Essential Features and Boundary with Normality (361 chunks); the dense
path queries the full Chroma collection. The mood-episode descriptions
that open the mood chapter carry no ICD code in the source document, so the chunker assigns
them pseudo-codes `EP.DEP`, `EP.MAN`, `EP.MIX` and `EP.HYP`.

**Few-shot examples.** `knowledge_base/fewshot/` holds two files with different roles:

- `multiclass_examples.json` — the full auto-sampled pool (200 per class, deterministic
  seed, eval posts held out). Kept only as provenance for where the curated examples came
  from; nothing loads it at runtime.
- `multiclass_examples_curated.json` — 20 hand-verified posts per class, because the full
  pool contains real label noise. This is what every few-shot retriever uses: BM25 reads
  the JSON directly, dense retrieval queries the `fewshot_multiclass_curated` Chroma
  collection built from it.

To change which examples are curated, edit `CURATED_IDS` in
`scripts/make_curated_fewshot.py`, regenerate the JSON, then re-embed it with
`python src/builders/build_curated_fewshot_db.py`.

## Supporting scripts

Pre-experiments that justify design choices. Each needs no Ollama and writes a `.txt`
report next to itself:

```bash
python analysis/retriever_comparison.py      # BM25 vs dense vs hybrid on the dev split
python analysis/bm25_suicide_keywords.py     # lexical overlap between posts and ICD-11 text
python analysis/page_boundary_experiment.py  # PDF page-range study (needs the local PDF)
```

Inspection and debugging:

```bash
python scripts/kb_snapshot.py                    # inventory the chunk pool and dump the
                                                 # retrieval + prompt context for fixed queries
python experiments/exp_fewshot_rag.py --dry-run  # show retrieved examples and the assembled
                                                 # prompt without calling Ollama
python experiments/exp_fewshot_rag.py --compare --query suicidal
```

`benchmarking/` holds earlier model and latency benchmarks kept for the record; they are
not part of the reported results.

## Known limitations

- **Corpus.** Labels come from a single social-media corpus with self-reported/derived
  labels; they are not clinical diagnoses. The comparison notebook's write-up section flags
  this explicitly.
- **Scope.** This is a research artifact for a coursework project. It is not a clinical
  tool and must not be used to make decisions about real people.

## Generative AI use

Generative AI tools were used during development of this project. They are declared here
in full:

| Tool | Used for |
|------|----------|
| ChatGPT (OpenAI) | Explaining concepts, drafting and debugging code, refining documentation |
| Claude (Anthropic) | Explaining concepts, drafting and debugging code, refining documentation |
| Cursor (and the models available in it, e.g. Claude, GPT, Composer) | In-editor code completion, refactoring, and agentic edits across the repo |
| DeepSeek | Explaining concepts, drafting and debugging code |

How we used them:

- AI assistance was used for boilerplate, refactoring, debugging, and wording — not for
  generating results, data, or findings.
- Every AI-suggested change was read, edited where needed, and run by a team member
  before being committed. The team is responsible for all code in this repository.
- No dataset content, evaluation numbers, or reported results were fabricated or produced
  by these tools; all numbers in `results/` come from actually running the pipeline.

Note that this is separate from the Qwen models (`qwen3:0.6b` / `qwen3:1.7b`) run via
Ollama inside the pipeline itself — those are the object of study, not authoring
assistance.
