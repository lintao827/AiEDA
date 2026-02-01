#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""One-shot end-to-end net wirelength prediction closure demo.

This script automates the full pipeline:
  1) (Optional) Run iEDA physical flow for selected designs
  2) Generate net vectors under output/iEDA/vectors/nets
  3) Aggregate vectors into a CSV dataset (TabNetDataProcess)
  4) Train TabNet baseline wirelength-ratio predictor
  5) Export the trained model to ONNX (+ normalization JSON)
  6) (Optional) Run AI placement in iEDA using the exported model

It supports both:1
- GCD (vendored sky130_gcd example)
- AES (benchmarks/aes, synthesized via Yosys to a sky130 gate-level netlist)

Typical usage:
  uv run python test/test_ai_task_wirelength_auto.py --include gcd aes --ai-placement gcd

Notes:
- AES synthesis requires `yosys` in PATH.
- iEDA flow can be very slow; the script will reuse existing vectors if present.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable, Tuple

import sys

import torch

os.environ["iEDA"] = "ON"

# Allow importing helper scripts under `test/` (it is not a Python package).
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "test"))

from aieda.flows import DbFlow, RunIEDA, DataGeneration
from aieda.workspace import Workspace
from aieda.ai import TabNetDataConfig, TabNetDataProcess, TabNetModelConfig, TabNetTrainer


def _repo_root() -> Path:
    return _ROOT


def _net_vectors_dir(workspace_dir: str) -> Path:
    return Path(workspace_dir) / "output" / "iEDA" / "vectors" / "nets"


def _has_net_vectors(workspace_dir: str) -> bool:
    d = _net_vectors_dir(workspace_dir)
    if not d.exists() or not d.is_dir():
        return False
    # Any file/dir inside counts.
    try:
        next(d.iterdir())
        return True
    except StopIteration:
        return False


def _ensure_aes_synthesized(root: Path) -> Tuple[Path, Path]:
    netlist = root / "benchmarks" / "aes" / "syn_netlist" / "aes_sky130.v"
    sdc = root / "benchmarks" / "aes" / "syn_netlist" / "aes.sdc"

    if netlist.exists() and sdc.exists():
        return netlist, sdc

    # Reuse the existing synthesis helper.
    from synth_sky130_aes import main as synth_main

    rc = int(synth_main())
    if rc != 0:
        raise RuntimeError(f"AES synthesis failed (exit code {rc}).")

    if not netlist.exists() or not sdc.exists():
        raise RuntimeError(
            "AES synthesis reported success but expected outputs are missing: "
            f"netlist={netlist} sdc={sdc}"
        )

    return netlist, sdc


def _prepare_gcd_workspace(root: Path, patch_row_step: int, patch_col_step: int, force: bool) -> Workspace:
    # Reuse the proven gcd setup + flow runner.
    from test_sky130_gcd import (
        create_workspace_sky130_gcd,
        set_parameters,
        run_eda_flow,
        generate_vectors,
    )

    workspace_dir = str(root / "example" / "sky130_test")
    ws = create_workspace_sky130_gcd(workspace_dir)
    set_parameters(ws)

    if force or not _has_net_vectors(workspace_dir):
        run_eda_flow(ws)
        generate_vectors(ws, patch_row_step, patch_col_step)

    return ws


def _prepare_aes_workspace(root: Path, patch_row_step: int, patch_col_step: int, force: bool) -> Workspace:
    from test_sky130_aes import create_workspace_sky130_aes, run_eda_flow as run_aes_flow

    netlist, sdc = _ensure_aes_synthesized(root)

    workspace_dir = str(root / "example" / "sky130_aes")
    design = "aes"

    ws = create_workspace_sky130_aes(workspace_dir, design, str(netlist), str(sdc))

    if force or not _has_net_vectors(workspace_dir):
        run_aes_flow(ws)

        data_gen = DataGeneration(ws)
        data_gen.generate_vectors(
            input_def=ws.configs.get_output_def(
                DbFlow(eda_tool="iEDA", step=DbFlow.FlowStep.route), compressed=False
            ),
            vectors_dir=ws.paths_table.ieda_output["vectors"],
            patch_row_step=patch_row_step,
            patch_col_step=patch_col_step,
        )

    return ws


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="AiEDA net wirelength prediction: full automatic closure demo"
    )

    p.add_argument(
        "--include",
        nargs="+",
        default=["gcd", "aes"],
        choices=["gcd", "aes"],
        help="Which designs to include in the training dataset.",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Force rerun EDA/vector generation even if vectors already exist.",
    )
    p.add_argument(
        "--patch-row-step",
        type=int,
        default=18,
        help="Patch row step used by vectorization.",
    )
    p.add_argument(
        "--patch-col-step",
        type=int,
        default=18,
        help="Patch col step used by vectorization.",
    )

    p.add_argument(
        "--dataset-csv",
        default="net_dataset.csv",
        help="Output CSV dataset path (relative to repo root by default).",
    )

    p.add_argument(
        "--only-csv",
        action="store_true",
        help="Only generate the CSV dataset (skip training/export/AI placement).",
    )
    p.add_argument(
        "--saved-models-dir",
        default="saved_models",
        help="Directory to save TabNet .zip and ONNX models (relative to repo root).",
    )
    p.add_argument(
        "--analysis-fig-dir",
        default="analysis_fig",
        help="Directory to save analysis figures (relative to repo root).",
    )
    p.add_argument(
        "--normalization-json",
        default="normalization_params/wl_baseline_normalization_params.json",
        help="Normalization params JSON path (relative to repo root).",
    )

    p.add_argument(
        "--ai-placement",
        choices=["none", "gcd", "aes"],
        default="none",
        help="Optionally run iEDA AI placement using exported ONNX.",
    )

    return p.parse_args()


def main() -> int:
    args = _parse_args()
    root = _repo_root()

    include = set(args.include)

    # 1) Prepare workspaces + vectors
    workspaces: list[Workspace] = []
    if "gcd" in include:
        workspaces.append(
            _prepare_gcd_workspace(
                root,
                patch_row_step=args.patch_row_step,
                patch_col_step=args.patch_col_step,
                force=args.force,
            )
        )

    if "aes" in include:
        workspaces.append(
            _prepare_aes_workspace(
                root,
                patch_row_step=args.patch_row_step,
                patch_col_step=args.patch_col_step,
                force=args.force,
            )
        )

    if not workspaces:
        raise RuntimeError("No workspaces selected.")

    # 2) Build dataset CSV from vectors
    dataset_csv = str((root / args.dataset_csv).resolve())
    plot_dir = str((root / args.analysis_fig_dir).resolve())
    normalization_json = str((root / args.normalization_json).resolve())

    data_config = TabNetDataConfig(
        raw_input_dirs=workspaces,
        pattern="/output/iEDA/vectors/nets",
        model_input_file=dataset_csv,
        plot_dir=plot_dir,
        normalization_params_file=normalization_json,
        extracted_feature_columns=[
            "id",
            "wire_len",
            "width",
            "height",
            "fanout",
            "aspect_ratio",
            "l_ness",
            "rsmt",
            "via_num",
        ],
        wl_baseline_feature_columns=[
            "width",
            "height",
            "pin_num",
            "aspect_ratio",
            "l_ness",
            "rsmt",
            "area",
            "route_ratio_x",
            "route_ratio_y",
        ],
        test_size=0.2,
        random_state=42,
    )

    data_processor = TabNetDataProcess(data_config)
    data_processor.run_pipeline()

    if args.only_csv:
        print(f"CSV dataset generated: {dataset_csv}")
        return 0

    # 3) Train baseline model and export ONNX
    wirelength_baseline_model_config = {
        "n_d": 64,
        "n_a": 128,
        "n_steps": 4,
        "gamma": 1.8,
        "n_independent": 2,
        "n_shared": 2,
        "lambda_sparse": 1e-5,
        "learning_rate": 0.01,
        "batch_size": 2048,
        "max_epochs": 100,
        "patience": 20,
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        "num_workers": 4,
        "pin_memory": True,
    }

    model_config = TabNetModelConfig(
        do_train=True,
        do_eval=True,
        output_dir=str(root),
        baseline_model_config=wirelength_baseline_model_config,
    )

    trainer = TabNetTrainer(data_config=data_config, model_config=model_config)
    trainer.train()

    saved_models_dir = root / args.saved_models_dir
    saved_models_dir.mkdir(parents=True, exist_ok=True)

    trainer.save_models(str(saved_models_dir))

    model_zip = saved_models_dir / "baseline_model.zip"
    model_onnx = saved_models_dir / "baseline_model.onnx"

    onnx_path, normalization_path = trainer.export_model_to_onnx(
        model_type="wirelength",
        model_path=str(model_zip),
        onnx_path=str(model_onnx),
        num_features=9,
    )

    print(f"ONNX model exported to: {onnx_path}")
    print(f"Normalization parameters path: {normalization_path}")

    # 4) Optional: AI placement
    if args.ai_placement != "none":
        target = args.ai_placement
        ws = None
        for w in workspaces:
            if getattr(w, "design", None) == target or getattr(w, "project", None) == target:
                ws = w
                break

        if ws is None:
            # Fallback: match by workspace directory name.
            for w in workspaces:
                if target in str(getattr(w, "directory", "")):
                    ws = w
                    break

        if ws is None:
            raise RuntimeError(f"Requested AI placement on '{target}', but workspace not found.")

        run_ieda = RunIEDA(ws)
        run_ieda.run_ai_placement(
            input_def=ws.configs.get_output_def(
                DbFlow(eda_tool="iEDA", step=DbFlow.FlowStep.fixFanout)
            ),
            onnx_path=onnx_path,
            normalization_path=normalization_path,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
