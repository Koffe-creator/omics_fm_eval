"""Classical baseline embeddings."""

import numpy as np
import scanpy as sc
from anndata import AnnData

from .base import EmbeddingModel


class PCABaseline(EmbeddingModel):
    name = "pca"

    def __init__(self, n_components: int = 50):
        self.n_components = n_components

    def embed(self, adata: AnnData) -> np.ndarray:
        ad = adata.copy()
        sc.pp.pca(ad, n_comps=self.n_components)
        return ad.obsm["X_pca"]


class ScVIBaseline(EmbeddingModel):
    name = "scvi"

    def __init__(self, n_latent: int = 30, max_epochs: int = 100):
        self.n_latent = n_latent
        self.max_epochs = max_epochs

    def embed(self, adata: AnnData) -> np.ndarray:
        import scvi

        ad = adata.copy()
        scvi.model.SCVI.setup_anndata(ad)
        model = scvi.model.SCVI(ad, n_latent=self.n_latent)
        model.train(max_epochs=self.max_epochs)
        return model.get_latent_representation()
