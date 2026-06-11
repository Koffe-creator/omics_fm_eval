"""Wrappers for single-cell foundation models.

Each wrapper is a thin adapter around the model's published embedding API.
Install the corresponding package before use (not in requirements.txt due
to heavy/conflicting dependencies):

  - scGPT:       pip install scgpt
  - Geneformer:  pip install geneformer
  - scFoundation: see https://github.com/biomap-research/scFoundation
  - UCE:         see https://github.com/snap-stanford/UCE
"""

import numpy as np
from anndata import AnnData

from .base import EmbeddingModel


class ScGPTModel(EmbeddingModel):
    """Wrapper around scGPT's pretrained whole-human checkpoint.

    Download the checkpoint (e.g. "whole-human") from the scGPT repo and point
    `model_dir` at the folder containing `args.json`, `vocab.json`, and `best_model.pt`.

    Requires `adata.var_names` to be gene symbols matching scGPT's vocab.
    """

    name = "scgpt"

    def __init__(self, model_dir: str, gene_col: str = "index", batch_size: int = 64):
        self.model_dir = model_dir
        self.gene_col = gene_col
        self.batch_size = batch_size

    def embed(self, adata: AnnData) -> np.ndarray:
        from scgpt.tasks import embed_data

        embedded = embed_data(
            adata,
            model_dir=self.model_dir,
            gene_col=self.gene_col,
            batch_size=self.batch_size,
        )
        return embedded.obsm["X_scGPT"]


class GeneformerModel(EmbeddingModel):
    """Wrapper around Geneformer (HuggingFace transformers, runs on CPU).

    `model_dir` should point to a downloaded Geneformer checkpoint, e.g.
    `Geneformer/gf-12L-30M-i2048` from https://huggingface.co/ctheodoris/Geneformer

    `adata.var_names` must be Ensembl gene IDs (the tokenizer maps gene IDs
    to tokens via its internal dictionary).
    """

    name = "geneformer"

    def __init__(self, model_dir: str, emb_layer: int = -1, model_version: str = "V2"):
        self.model_dir = model_dir
        self.emb_layer = emb_layer
        self.model_version = model_version

    def embed(self, adata: AnnData) -> np.ndarray:
        import os
        import tempfile

        import geneformer
        from geneformer import EmbExtractor, TranscriptomeTokenizer

        # V1 checkpoints (e.g. 12L-30M) use the gc30M gene dictionaries,
        # not the package default gc104M (V2) dictionaries.
        if self.model_version == "V1":
            dict_dir = os.path.join(os.path.dirname(geneformer.__file__), "gene_dictionaries_30m")
            token_dictionary_file = os.path.join(dict_dir, "token_dictionary_gc30M.pkl")
            gene_median_file = os.path.join(dict_dir, "gene_median_dictionary_gc30M.pkl")
            gene_mapping_file = os.path.join(dict_dir, "ensembl_mapping_dict_gc30M.pkl")
        else:
            pkg_dir = os.path.dirname(geneformer.__file__)
            token_dictionary_file = os.path.join(pkg_dir, "token_dictionary_gc104M.pkl")
            gene_median_file = os.path.join(pkg_dir, "gene_median_dictionary_gc104M.pkl")
            gene_mapping_file = os.path.join(pkg_dir, "ensembl_mapping_dict_gc104M.pkl")

        with tempfile.TemporaryDirectory() as tmpdir:
            h5ad_dir = os.path.join(tmpdir, "h5ad")
            tok_dir = os.path.join(tmpdir, "tokenized")
            os.makedirs(h5ad_dir, exist_ok=True)
            os.makedirs(tok_dir, exist_ok=True)

            ad = adata.copy()
            ad.obs["n_counts"] = np.asarray(ad.X.sum(axis=1)).flatten()
            ad.write_h5ad(os.path.join(h5ad_dir, "data.h5ad"))

            tokenizer = TranscriptomeTokenizer(
                nproc=4,
                model_version=self.model_version,
                token_dictionary_file=token_dictionary_file,
                gene_median_file=gene_median_file,
                gene_mapping_file=gene_mapping_file,
                use_h5ad_index=True,
            )
            tokenizer.tokenize_data(h5ad_dir, tok_dir, "tokenized", file_format="h5ad")

            extractor = EmbExtractor(
                model_type="Pretrained",
                emb_layer=self.emb_layer,
                emb_mode="cell",
                cell_emb_style="mean_pool",
                forward_batch_size=8,
                max_ncells=adata.n_obs,
                nproc=4,
                model_version=self.model_version,
                token_dictionary_file=token_dictionary_file,
            )
            embs = extractor.extract_embs(
                self.model_dir,
                os.path.join(tok_dir, "tokenized.dataset"),
                tmpdir,
                "embs",
            )
        return embs.values


class ScFoundationModel(EmbeddingModel):
    name = "scfoundation"

    def __init__(self, model_dir: str):
        self.model_dir = model_dir

    def embed(self, adata: AnnData) -> np.ndarray:
        raise NotImplementedError("Run scFoundation get_embedding.py pipeline")


class UCEModel(EmbeddingModel):
    name = "uce"

    def __init__(self, model_dir: str):
        self.model_dir = model_dir

    def embed(self, adata: AnnData) -> np.ndarray:
        raise NotImplementedError("Run UCE eval.py to produce cell embeddings")
