from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

from .dataset import (
    PatchGraph,
    NetSample,
    PatchGraphNetDataset,
    collate_net_samples,
)
from .model import TransGNNWirelengthModel, TransGNNConfig


@dataclass
class TrainConfig:
    epochs: int = 50
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-5
    device: str = "auto"  # auto/cpu/cuda
    seed: int = 42
    test_size: float = 0.2
    normalize_patch_features: bool = True
    normalize_net_scalar: bool = True
    clip_after_norm: float = 5.0
    print_diagnostics: bool = True


def _resolve_device(device: str) -> torch.device:
    device = device.lower().strip()
    if device in {"auto", ""}:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    return torch.device(device)


def _set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
    mae = float(np.mean(np.abs(y_pred - y_true)))
    # avoid division by zero in mape
    denom = np.clip(np.abs(y_true), 1e-8, None)
    mape = float(np.mean(np.abs((y_pred - y_true) / denom)) * 100.0)
    r2 = float(r2_score(y_true, y_pred))
    return {"RMSE": rmse, "MAE": mae, "MAPE": mape, "R2": r2}


def _standardize(
    x: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    *,
    clip: float,
) -> torch.Tensor:
    y = (x - mean) / std
    if clip > 0:
        y = torch.clamp(y, -clip, clip)
    return y


def train_transgnn(
    patch_graph: PatchGraph,
    net_samples: List[NetSample],
    *,
    model_cfg: TransGNNConfig | None = None,
    train_cfg: TrainConfig | None = None,
) -> Tuple[TransGNNWirelengthModel, Dict[str, float]]:
    train_cfg = train_cfg or TrainConfig()
    _set_seed(train_cfg.seed)

    device = _resolve_device(train_cfg.device)

    if train_cfg.print_diagnostics:
        lengths = np.array([int(s.patch_indices.numel()) for s in net_samples], dtype=np.int64)
        targets = np.array([float(s.target.item()) for s in net_samples], dtype=np.float32)
        print(
            "data diagnostics: "
            f"nets={len(net_samples)} "
            f"patches_per_net[p50]={np.percentile(lengths, 50):.0f} "
            f"p99={np.percentile(lengths, 99):.0f} max={lengths.max():.0f} "
            f"target[min]={targets.min():.4g} p50={np.percentile(targets, 50):.4g} "
            f"p99={np.percentile(targets, 99):.4g} max={targets.max():.4g}"
        )

    # split by net
    idx = np.arange(len(net_samples))
    train_idx, test_idx = train_test_split(
        idx, test_size=train_cfg.test_size, random_state=train_cfg.seed, shuffle=True
    )

    train_ds = PatchGraphNetDataset([net_samples[i] for i in train_idx])
    test_ds = PatchGraphNetDataset([net_samples[i] for i in test_idx])

    train_dl = DataLoader(
        train_ds,
        batch_size=train_cfg.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_net_samples,
        drop_last=False,
    )
    test_dl = DataLoader(
        test_ds,
        batch_size=train_cfg.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_net_samples,
        drop_last=False,
    )

    patch_in_dim = int(patch_graph.x_patches.shape[1])
    net_scalar_dim = int(net_samples[0].net_scalar.numel())

    model = TransGNNWirelengthModel(
        patch_in_dim=patch_in_dim,
        net_scalar_dim=net_scalar_dim,
        cfg=model_cfg,
    ).to(device)

    x_patches = patch_graph.x_patches.to(device)
    edge_index = patch_graph.edge_index.to(device)

    # ---- Input normalization (important for stability; patch/net features can be huge in DBU scale) ----
    patch_mean = torch.zeros((1, patch_in_dim), device=device)
    patch_std = torch.ones((1, patch_in_dim), device=device)
    if train_cfg.normalize_patch_features:
        pm = patch_graph.x_patches.float().mean(dim=0, keepdim=True)
        ps = patch_graph.x_patches.float().std(dim=0, keepdim=True, unbiased=False)
        ps = torch.clamp(ps, min=1e-6)
        patch_mean = pm.to(device)
        patch_std = ps.to(device)
        x_patches = _standardize(x_patches, patch_mean, patch_std, clip=train_cfg.clip_after_norm)

    net_mean = torch.zeros((1, net_scalar_dim), device=device)
    net_std = torch.ones((1, net_scalar_dim), device=device)
    if train_cfg.normalize_net_scalar:
        train_net_scalar = torch.stack([s.net_scalar for s in train_ds.samples], dim=0).float()
        nm = train_net_scalar.mean(dim=0, keepdim=True)
        ns = train_net_scalar.std(dim=0, keepdim=True, unbiased=False)
        ns = torch.clamp(ns, min=1e-6)
        net_mean = nm.to(device)
        net_std = ns.to(device)

    opt = torch.optim.AdamW(
        model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay
    )

    best_rmse = math.inf
    best_state = None

    for epoch in range(1, train_cfg.epochs + 1):
        model.train()
        losses = []
        for batch in train_dl:
            patch_idx = batch["patch_idx"].to(device)
            attn_mask = batch["attn_mask"].to(device)
            net_scalar = batch["net_scalar"].to(device)
            targets = batch["targets"].to(device)

            if train_cfg.normalize_net_scalar:
                net_scalar = _standardize(
                    net_scalar, net_mean, net_std, clip=train_cfg.clip_after_norm
                )

            pred = model(
                x_patches=x_patches,
                edge_index=edge_index,
                patch_idx=patch_idx,
                attn_mask=attn_mask,
                net_scalar=net_scalar,
            )
            loss = torch.mean((pred - targets) ** 2)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

            losses.append(float(loss.detach().cpu().item()))

        # eval
        model.eval()
        y_true = []
        y_pred = []
        with torch.no_grad():
            for batch in test_dl:
                patch_idx = batch["patch_idx"].to(device)
                attn_mask = batch["attn_mask"].to(device)
                net_scalar = batch["net_scalar"].to(device)
                targets = batch["targets"].to(device)

                if train_cfg.normalize_net_scalar:
                    net_scalar = _standardize(
                        net_scalar, net_mean, net_std, clip=train_cfg.clip_after_norm
                    )

                pred = model(
                    x_patches=x_patches,
                    edge_index=edge_index,
                    patch_idx=patch_idx,
                    attn_mask=attn_mask,
                    net_scalar=net_scalar,
                )
                y_true.append(targets.detach().cpu().numpy())
                y_pred.append(pred.detach().cpu().numpy())

        y_true_np = np.concatenate(y_true, axis=0)
        y_pred_np = np.concatenate(y_pred, axis=0)
        m = _metrics(y_true_np, y_pred_np)

        avg_loss = float(np.mean(losses)) if losses else float("nan")

        if m["RMSE"] < best_rmse:
            best_rmse = m["RMSE"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if epoch == 1 or epoch % 5 == 0:
            print(
                f"epoch={epoch:03d} train_mse={avg_loss:.6f} "
                f"test_rmse={m['RMSE']:.4f} test_mae={m['MAE']:.4f} test_mape={m['MAPE']:.2f}% test_r2={m['R2']:.4f}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)

    # final metrics
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for batch in test_dl:
            patch_idx = batch["patch_idx"].to(device)
            attn_mask = batch["attn_mask"].to(device)
            net_scalar = batch["net_scalar"].to(device)
            targets = batch["targets"].to(device)

            if train_cfg.normalize_net_scalar:
                net_scalar = _standardize(
                    net_scalar, net_mean, net_std, clip=train_cfg.clip_after_norm
                )
            pred = model(
                x_patches=x_patches,
                edge_index=edge_index,
                patch_idx=patch_idx,
                attn_mask=attn_mask,
                net_scalar=net_scalar,
            )
            y_true.append(targets.detach().cpu().numpy())
            y_pred.append(pred.detach().cpu().numpy())

    y_true_np = np.concatenate(y_true, axis=0)
    y_pred_np = np.concatenate(y_pred, axis=0)
    final_metrics = _metrics(y_true_np, y_pred_np)

    # Attach normalization stats for checkpointing/inference convenience.
    model.input_norm = {
        "patch_mean": patch_mean.detach().cpu(),
        "patch_std": patch_std.detach().cpu(),
        "net_mean": net_mean.detach().cpu(),
        "net_std": net_std.detach().cpu(),
        "clip_after_norm": float(train_cfg.clip_after_norm),
        "normalize_patch_features": bool(train_cfg.normalize_patch_features),
        "normalize_net_scalar": bool(train_cfg.normalize_net_scalar),
    }

    return model, final_metrics
