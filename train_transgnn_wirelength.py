#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Train TransGNN (patch-grid GNN + Transformer aggregator) for wirelength prediction.

Data source: iEDA vectors.

Example:
  python train_transgnn_wirelength.py \
    --workspace ./example/sky130_gcd \
    --epochs 30 --batch-size 64 --device auto

If you have your own design workspace, point --workspace to it.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from aieda.ai.transGNN.dataset import build_patch_graph_from_vectors
from aieda.ai.transGNN.train import train_transgnn, TrainConfig


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--workspace",
        required=True,
        help="Workspace directory containing output/iEDA/vectors/{nets,patchs}",
    )
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="auto", help="auto/cpu/cuda")
    ap.add_argument(
        "--target",
        default="ratio",
        choices=["ratio", "drwl"],
        help="Predict wirelength ratio (wire_len/rsmt) or absolute wire_len (drwl)",
    )
    ap.add_argument("--save", default="", help="Optional path to save torch model .pt")
    args = ap.parse_args()

    ws = Path(args.workspace)
    nets_dir = ws / "output" / "iEDA" / "vectors" / "nets"
    patchs_dir = ws / "output" / "iEDA" / "vectors" / "patchs"

    if not nets_dir.exists():
        raise SystemExit(f"nets dir not found: {nets_dir}")
    if not patchs_dir.exists():
        raise SystemExit(f"patchs dir not found: {patchs_dir}")

    patch_graph, net_samples = build_patch_graph_from_vectors(
        str(nets_dir), str(patchs_dir), use_target=args.target
    )

    print(
        f"loaded patches={patch_graph.x_patches.shape[0]} patch_feat_dim={patch_graph.x_patches.shape[1]} "
        f"edges={patch_graph.edge_index.shape[1]} nets={len(net_samples)}"
    )

    cfg = TrainConfig(
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        device=str(args.device),
    )

    model, metrics = train_transgnn(patch_graph, net_samples, train_cfg=cfg)

    print("final metrics:")
    for k, v in metrics.items():
        if k == "MAPE":
            print(f"  {k}: {v:.2f}%")
        else:
            print(f"  {k}: {v:.4f}")

    if args.save:
        save_path = Path(args.save)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state": model.state_dict(),
                "metrics": metrics,
                "input_norm": getattr(model, "input_norm", None),
            },
            save_path,
        )
        print(f"saved model checkpoint: {save_path}")


if __name__ == "__main__":
    main()
