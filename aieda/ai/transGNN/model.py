from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphSAGEConv(nn.Module):
    """Minimal GraphSAGE-like layer without external deps.

    h_i' = W [h_i || mean_{j in N(i)} h_j]

    edge_index is [2, E] with (src, dst) meaning src -> dst message.
    """

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        self.lin = nn.Linear(in_dim * 2, out_dim)
        self.dropout = float(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        src, dst = edge_index
        num_nodes = x.size(0)

        # Aggregate neighbor messages by mean
        agg = torch.zeros((num_nodes, x.size(1)), device=x.device, dtype=x.dtype)
        deg = torch.zeros((num_nodes,), device=x.device, dtype=x.dtype)

        agg.index_add_(0, dst, x[src])
        deg.index_add_(0, dst, torch.ones_like(dst, dtype=x.dtype))
        deg = deg.clamp_min(1.0).unsqueeze(-1)
        neigh = agg / deg

        out = self.lin(torch.cat([x, neigh], dim=-1))
        out = F.relu(out)
        if self.dropout > 0:
            out = F.dropout(out, p=self.dropout, training=self.training)
        return out


class PatchGNN(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        layers = []
        dims = [in_dim] + [hidden_dim] * num_layers
        for i in range(num_layers):
            layers.append(GraphSAGEConv(dims[i], dims[i + 1], dropout=dropout))
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = x
        for layer in self.layers:
            h = layer(h, edge_index)
        return h


class PatchSetTransformer(nn.Module):
    """Transformer encoder over a padded set/sequence of patch embeddings."""

    def __init__(
        self,
        dim: int,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        encoder_layer = None
        try:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=nhead,
                dim_feedforward=dim * 4,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.batch_first = True
        except TypeError:
            # Older torch without batch_first/norm_first
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=nhead,
                dim_feedforward=dim * 4,
                dropout=dropout,
                activation="gelu",
            )
            self.batch_first = False

        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # A learned [CLS] token to summarize the set
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.normal_(self.cls, std=0.02)

    def forward(self, seq: torch.Tensor, key_padding_mask: torch.Tensor) -> torch.Tensor:
        """Args:
        seq: [B, L, D]
        key_padding_mask: [B, L] True = pad
        Returns:
          pooled: [B, D]
        """
        b, _l, d = seq.shape
        cls = self.cls.expand(b, 1, d)
        seq2 = torch.cat([cls, seq], dim=1)  # [B, 1+L, D]

        # Extend mask for CLS (not masked)
        mask2 = torch.cat(
            [torch.zeros((b, 1), device=key_padding_mask.device, dtype=torch.bool), key_padding_mask],
            dim=1,
        )

        if self.batch_first:
            h = self.encoder(seq2, src_key_padding_mask=mask2)
        else:
            # transformer expects [L, B, D]
            h = self.encoder(seq2.transpose(0, 1), src_key_padding_mask=mask2).transpose(0, 1)

        return h[:, 0, :]  # CLS


@dataclass
class TransGNNConfig:
    patch_hidden_dim: int = 128
    patch_gnn_layers: int = 3
    patch_gnn_dropout: float = 0.1
    tf_layers: int = 2
    tf_heads: int = 4
    tf_dropout: float = 0.1
    mlp_hidden: int = 128


class TransGNNWirelengthModel(nn.Module):
    def __init__(
        self,
        patch_in_dim: int,
        net_scalar_dim: int,
        cfg: Optional[TransGNNConfig] = None,
    ):
        super().__init__()
        self.cfg = cfg or TransGNNConfig()

        self.patch_gnn = PatchGNN(
            in_dim=patch_in_dim,
            hidden_dim=self.cfg.patch_hidden_dim,
            num_layers=self.cfg.patch_gnn_layers,
            dropout=self.cfg.patch_gnn_dropout,
        )

        self.patch_tf = PatchSetTransformer(
            dim=self.cfg.patch_hidden_dim,
            nhead=self.cfg.tf_heads,
            num_layers=self.cfg.tf_layers,
            dropout=self.cfg.tf_dropout,
        )

        self.patch_norm = nn.LayerNorm(self.cfg.patch_hidden_dim)
        self.net_norm = nn.LayerNorm(net_scalar_dim)

        self.mlp = nn.Sequential(
            nn.Linear(self.cfg.patch_hidden_dim + net_scalar_dim, self.cfg.mlp_hidden),
            nn.GELU(),
            nn.Dropout(self.cfg.tf_dropout),
            nn.Linear(self.cfg.mlp_hidden, 1),
        )

    def forward(
        self,
        x_patches: torch.Tensor,
        edge_index: torch.Tensor,
        patch_idx: torch.Tensor,
        attn_mask: torch.Tensor,
        net_scalar: torch.Tensor,
    ) -> torch.Tensor:
        """Forward.

        Args:
          x_patches: [P, F]
          edge_index: [2, E]
          patch_idx: [B, L] patch indices padded with -1
          attn_mask: [B, L] True = masked
          net_scalar: [B, S]

        Returns:
          y_hat: [B]
        """

        h_patches = self.patch_gnn(x_patches, edge_index)  # [P, D]

        # Gather per-net patch embeddings into [B, L, D]
        b, l = patch_idx.shape
        d = h_patches.size(1)

        safe_idx = patch_idx.clamp_min(0)
        seq = h_patches[safe_idx]  # [B, L, D]

        # For padded positions (where patch_idx == -1), zero out
        seq = seq.masked_fill(attn_mask.unsqueeze(-1), 0.0)

        pooled = self.patch_tf(seq, key_padding_mask=attn_mask)  # [B, D]

        pooled = self.patch_norm(pooled)
        net_scalar = self.net_norm(net_scalar)
        z = torch.cat([pooled, net_scalar], dim=-1)
        y_hat = self.mlp(z).squeeze(-1)
        return y_hat
