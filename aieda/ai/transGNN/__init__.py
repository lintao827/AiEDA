"""Transformer + GNN models for wirelength prediction.

This package focuses on a patch-grid graph (GNN) for global context, and a
Transformer aggregator over the set/sequence of patches that each net touches.

Typical data source: iEDA vectors under:1
  <workspace>/output/iEDA/vectors/{nets,patchs}
"""

from .dataset import PatchGraphNetDataset, build_patch_graph_from_vectors
from .model import TransGNNWirelengthModel
from .train import train_transgnn

__all__ = [
    "PatchGraphNetDataset",
    "build_patch_graph_from_vectors",
    "TransGNNWirelengthModel",
    "train_transgnn",
]
