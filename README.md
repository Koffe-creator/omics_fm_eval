# Omics Foundation Model Evaluation

Benchmark suite for single-cell transcriptomics foundation models (scGPT, Geneformer,
scFoundation, UCE, ...) against classical baselines, plus an LLM-as-evaluator layer
that uses general-purpose LLMs to judge cell-type predictions and marker-gene
biological plausibility.

## Structure

- `data/` — dataset download/preprocessing scripts and cached AnnData files
- `models/` — wrapper interfaces for each foundation model + classical baselines
- `eval/` — benchmark tasks (cell type classification, batch integration, zero-shot transfer)
- `llm_judge/` — LLM-as-evaluator pipeline (marker gene -> LLM cell type prediction, agreement scoring)
- `results/` — leaderboard tables, plots
- `notebooks/` — exploratory analysis

## Phase 1: Foundation model benchmark

Models: scGPT, Geneformer, scFoundation, UCE (subset TBD)
Datasets: PBMC, Tabula Sapiens subset, one OOD disease dataset
Tasks: linear-probe cell type classification, batch integration (kBET/ASW), zero-shot transfer
Baselines: PCA+kNN, scVI

## Phase 2: LLM-as-evaluator

For each model's predicted clusters, extract top marker genes, query an LLM for
cell-type prediction, and compare against ground truth and the foundation model's
own prediction.

## Setup

```bash
pip install -r requirements.txt
```

## Geneformer setup

(scGPT requires `flash-attn`/CUDA and is not runnable on macOS; using
Geneformer instead, which runs on CPU via HuggingFace transformers.)

1. Install: `pip install geneformer transformers loompy`
2. Download a checkpoint, e.g. `gf-12L-30M-i2048`, from
   [huggingface.co/ctheodoris/Geneformer](https://huggingface.co/ctheodoris/Geneformer)
   and place it at `models/checkpoints/geneformer-12L-30M/`
3. Ensure `adata.var_names` are Ensembl gene IDs (e.g. `ENSG00000139618`) —
   Geneformer's tokenizer maps gene IDs to tokens via its internal dictionary
4. Run: `python -m eval.run_benchmark --dataset pbmc --models pca scvi geneformer`
