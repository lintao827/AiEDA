from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from aieda.data.io.vectors_io import VectorsParserJson


def _safe_float(x: Optional[float], default: float = 0.0) -> float:
    if x is None:
        return default
    try:
        if isinstance(x, (int, float, np.number)):
            return float(x)
        return float(str(x))
    except Exception:
        return default


def _collect_json_files(root_dir: str, max_files: Optional[int] = None) -> List[str]:
    import os

    files: List[str] = []
    for root, _dirs, names in os.walk(root_dir):
        for name in names:
            if name.endswith(".json"):
                files.append(os.path.join(root, name))
                if max_files is not None and len(files) >= max_files:
                    return files
    return files


@dataclass
class PatchGraph:
    x_patches: torch.Tensor  # [P, F]
    edge_index: torch.Tensor  # [2, E]
    patch_rc: torch.Tensor  # [P, 2] row/col


@dataclass
class NetSample:
    net_id: int
    patch_indices: torch.Tensor  # [K]
    net_scalar: torch.Tensor  # [S]
    target: torch.Tensor  # [1]


def build_patch_graph_from_vectors(
    nets_dir: str,
    patchs_dir: str,
    *,
    use_target: str = "ratio",
) -> Tuple[PatchGraph, List[NetSample]]:
    """Build a patch grid graph and per-net samples from iEDA vectors.

    Args:
        nets_dir: path to vectors/nets directory1
        patchs_dir: path to vectors/patchs directory
        use_target: 'ratio' -> wire_len/rsmt, 'drwl' -> wire_len

    Returns:
        (patch_graph, net_samples)

    Notes:
      - Patch features come from VectorPatch + aggregated VectorPatchLayer.
      - Net-to-patch membership comes from patch_layer.nets list.
      - Targets come from vectors/nets (net.feature.wire_len and place_feature.rsmt).
    """

    # --- Load patches and build patch features + net->patch membership ---
    patch_files = _collect_json_files(patchs_dir)
    patches = []
    for fp in patch_files:
        parser = VectorsParserJson(fp)
        patches.extend([p for p in parser.get_patchs() if p is not None])

    if not patches:
        raise ValueError(f"No patches found under: {patchs_dir}")

    # Sort patches by (row, col) for deterministic indexing
    patches.sort(key=lambda p: (p.patch_id_row or 0, p.patch_id_col or 0))

    patch_rc = []
    patch_features = []

    net_to_patch_ids: Dict[int, set[int]] = {}

    for idx, p in enumerate(patches):
        row = int(p.patch_id_row) if p.patch_id_row is not None else -1
        col = int(p.patch_id_col) if p.patch_id_col is not None else -1
        patch_rc.append([row, col])

        # Aggregate per-layer features
        total_wire_len = 0.0
        total_wire_width = 0.0
        total_wire_density = 0.0
        total_congestion = 0.0
        total_net_num = 0.0
        layer_count = 0.0

        for pl in getattr(p, "patch_layer", []) or []:
            layer_count += 1.0
            total_net_num += _safe_float(getattr(pl, "net_num", None), 0.0)
            total_wire_len += _safe_float(getattr(pl, "wire_len", None), 0.0)
            total_wire_width += _safe_float(getattr(pl, "wire_width", None), 0.0)
            total_wire_density += _safe_float(getattr(pl, "wire_density", None), 0.0)
            total_congestion += _safe_float(getattr(pl, "congestion", None), 0.0)

            for n in getattr(pl, "nets", []) or []:
                if n is None or n.id is None:
                    continue
                net_to_patch_ids.setdefault(int(n.id), set()).add(idx)

        if layer_count > 0:
            avg_wire_width = total_wire_width / layer_count
            avg_wire_density = total_wire_density / layer_count
            avg_congestion = total_congestion / layer_count
        else:
            avg_wire_width = 0.0
            avg_wire_density = 0.0
            avg_congestion = 0.0

        # Patch-level features (keep simple & robust)
        feat = [
            _safe_float(getattr(p, "cell_density", None)),
            _safe_float(getattr(p, "pin_density", None)),
            _safe_float(getattr(p, "net_density", None)),
            _safe_float(getattr(p, "RUDY_congestion", None)),
            _safe_float(getattr(p, "EGR_congestion", None)),
            _safe_float(getattr(p, "macro_margin", None)),
            _safe_float(getattr(p, "timing_map", None)),
            _safe_float(getattr(p, "power_map", None)),
            _safe_float(getattr(p, "ir_drop_map", None)),
            total_net_num,
            total_wire_len,
            avg_wire_width,
            avg_wire_density,
            avg_congestion,
        ]
        patch_features.append(feat)

    x_patches = torch.tensor(patch_features, dtype=torch.float32)
    patch_rc_t = torch.tensor(patch_rc, dtype=torch.long)

    # --- Build patch grid edges using 4-neighborhood on (row, col) ---
    rc_to_idx: Dict[Tuple[int, int], int] = {
        (int(rc[0]), int(rc[1])): i for i, rc in enumerate(patch_rc)
    }

    edges_src = []
    edges_dst = []

    for i, (r, c) in enumerate(rc_to_idx.keys()):
        for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            j = rc_to_idx.get((r + dr, c + dc))
            if j is None:
                continue
            edges_src.append(i)
            edges_dst.append(j)

    if not edges_src:
        raise ValueError(
            "No patch-grid edges were created. Check patch_id_row/patch_id_col in vectors/patchs."
        )

    edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)

    patch_graph = PatchGraph(x_patches=x_patches, edge_index=edge_index, patch_rc=patch_rc_t)

    # --- Load nets to get targets and scalar net features ---
    net_files = _collect_json_files(nets_dir)
    nets = []
    for fp in net_files:
        parser = VectorsParserJson(fp)
        nets.extend([n for n in parser.get_nets() if n is not None])

    if not nets:
        raise ValueError(f"No nets found under: {nets_dir}")

    net_samples: List[NetSample] = []

    for n in nets:
        if n.id is None or n.feature is None:
            continue
        net_id = int(n.id)
        patch_ids = net_to_patch_ids.get(net_id)
        if not patch_ids:
            continue

        wire_len = _safe_float(getattr(n.feature, "wire_len", None), 0.0)
        place_feature = getattr(n.feature, "place_feature", None)
        rsmt = _safe_float(getattr(place_feature, "rsmt", None) if place_feature else None, 0.0)

        if use_target == "ratio":
            if rsmt <= 0:
                continue
            target = wire_len / rsmt
        elif use_target == "drwl":
            target = wire_len
        else:
            raise ValueError("use_target must be 'ratio' or 'drwl'")

        # Net scalar features (lightweight)
        llx = _safe_float(getattr(n.feature, "llx", None))
        lly = _safe_float(getattr(n.feature, "lly", None))
        urx = _safe_float(getattr(n.feature, "urx", None))
        ury = _safe_float(getattr(n.feature, "ury", None))
        bbox_w = max(0.0, urx - llx)
        bbox_h = max(0.0, ury - lly)

        scalar = [
            _safe_float(getattr(place_feature, "pin_num", None) if place_feature else None),
            _safe_float(getattr(place_feature, "aspect_ratio", None) if place_feature else None),
            _safe_float(getattr(place_feature, "width", None) if place_feature else None),
            _safe_float(getattr(place_feature, "height", None) if place_feature else None),
            _safe_float(getattr(place_feature, "area", None) if place_feature else None),
            _safe_float(getattr(place_feature, "l_ness", None) if place_feature else None),
            rsmt,
            _safe_float(getattr(place_feature, "hpwl", None) if place_feature else None),
            bbox_w,
            bbox_h,
            _safe_float(getattr(n.feature, "via_num", None)),
            _safe_float(getattr(n.feature, "drc_num", None)),
            _safe_float(getattr(n.feature, "R", None)),
            _safe_float(getattr(n.feature, "C", None)),
        ]

        net_samples.append(
            NetSample(
                net_id=net_id,
                patch_indices=torch.tensor(sorted(patch_ids), dtype=torch.long),
                net_scalar=torch.tensor(scalar, dtype=torch.float32),
                target=torch.tensor([target], dtype=torch.float32),
            )
        )

    if not net_samples:
        raise ValueError(
            "Built 0 net samples. Common causes: net ids don't match between nets and patchs, or rsmt is missing/0."
        )

    return patch_graph, net_samples


class PatchGraphNetDataset(Dataset):
    def __init__(self, samples: List[NetSample]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> NetSample:
        return self.samples[idx]


def collate_net_samples(batch: List[NetSample]) -> Dict[str, torch.Tensor]:
    """Collate variable-length patch indices into padded tensors."""

    net_ids = torch.tensor([b.net_id for b in batch], dtype=torch.long)
    targets = torch.cat([b.target for b in batch], dim=0)  # [B]
    net_scalar = torch.stack([b.net_scalar for b in batch], dim=0)  # [B, S]

    lengths = torch.tensor([b.patch_indices.numel() for b in batch], dtype=torch.long)
    max_len = int(lengths.max().item())

    patch_idx = torch.full((len(batch), max_len), fill_value=-1, dtype=torch.long)
    attn_mask = torch.ones((len(batch), max_len), dtype=torch.bool)  # True = masked

    for i, b in enumerate(batch):
        k = b.patch_indices.numel()
        patch_idx[i, :k] = b.patch_indices
        attn_mask[i, :k] = False

    return {
        "net_ids": net_ids,
        "targets": targets,
        "net_scalar": net_scalar,
        "patch_idx": patch_idx,
        "attn_mask": attn_mask,
        "lengths": lengths,
    }
