#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""Run iEDA flow + vectorization for a synthesized AES netlist (sky130).

Prereq:
  1) Synthesize RTL to a sky130 gate-level netlist:
     uv run python test/synth_sky130_aes.py

Then run:
  uv run python test/test_sky130_aes.py

Outputs (under workspace_dir):
- output/iEDA/result/*.def.gz
- output/iEDA/vectors/nets (JSON vectors)

This is the same "Design-to-Vector" idea as the paper, but for your local AES.
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ["iEDA"] = "ON"

from aieda.workspace import workspace_create
from aieda.flows import DbFlow, RunIEDA, DataGeneration
from aieda.data.database import EDAParameters


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def create_workspace_sky130_aes(workspace_dir: str, design: str, verilog: str, sdc: str):
    flow_db_list = [
        DbFlow(eda_tool="iEDA", step=DbFlow.FlowStep.floorplan, state=DbFlow.FlowState.Unstart),
        DbFlow(eda_tool="iEDA", step=DbFlow.FlowStep.pdn, state=DbFlow.FlowState.Unstart),
        DbFlow(eda_tool="iEDA", step=DbFlow.FlowStep.fixFanout, state=DbFlow.FlowState.Unstart),
        DbFlow(eda_tool="iEDA", step=DbFlow.FlowStep.place, state=DbFlow.FlowState.Unstart),
        DbFlow(eda_tool="iEDA", step=DbFlow.FlowStep.cts, state=DbFlow.FlowState.Unstart),
        DbFlow(eda_tool="iEDA", step=DbFlow.FlowStep.optDrv, state=DbFlow.FlowState.Unstart),
        DbFlow(eda_tool="iEDA", step=DbFlow.FlowStep.optHold, state=DbFlow.FlowState.Unstart),
        DbFlow(eda_tool="iEDA", step=DbFlow.FlowStep.legalization, state=DbFlow.FlowState.Unstart),
        DbFlow(eda_tool="iEDA", step=DbFlow.FlowStep.route, state=DbFlow.FlowState.Unstart),
        DbFlow(eda_tool="iEDA", step=DbFlow.FlowStep.drc, state=DbFlow.FlowState.Unstart),
        DbFlow(eda_tool="iEDA", step=DbFlow.FlowStep.vectorization, state=DbFlow.FlowState.Unstart),
        DbFlow(eda_tool="iEDA", step=DbFlow.FlowStep.filler, state=DbFlow.FlowState.Unstart),
    ]

    workspace = workspace_create(directory=workspace_dir, design=design, flow_list=flow_db_list)

    root = str(_repo_root())
    foundry_dir = f"{root}/aieda/third_party/iEDA/scripts/foundry/sky130"

    workspace.set_verilog_input(verilog)
    workspace.set_sdc(sdc)

    # Use HS flavor (matches test_sky130_gcd.py)
    workspace.set_tech_lef(f"{foundry_dir}/lef/sky130_fd_sc_hs.tlef")

    lefs = [
        f"{foundry_dir}/lef/sky130_fd_sc_hs_merged.lef",
        f"{foundry_dir}/lef/sky130_ef_io__com_bus_slice_10um.lef",
        f"{foundry_dir}/lef/sky130_ef_io__com_bus_slice_1um.lef",
        f"{foundry_dir}/lef/sky130_ef_io__com_bus_slice_20um.lef",
        f"{foundry_dir}/lef/sky130_ef_io__com_bus_slice_5um.lef",
        f"{foundry_dir}/lef/sky130_ef_io__connect_vcchib_vccd_and_vswitch_vddio_slice_20um.lef",
        f"{foundry_dir}/lef/sky130_ef_io__corner_pad.lef",
        f"{foundry_dir}/lef/sky130_ef_io__disconnect_vccd_slice_5um.lef",
        f"{foundry_dir}/lef/sky130_ef_io__disconnect_vdda_slice_5um.lef",
        f"{foundry_dir}/lef/sky130_ef_io__gpiov2_pad_wrapped.lef",
        f"{foundry_dir}/lef/sky130_ef_io__vccd_hvc_pad.lef",
        f"{foundry_dir}/lef/sky130_ef_io__vccd_lvc_pad.lef",
        f"{foundry_dir}/lef/sky130_ef_io__vdda_hvc_pad.lef",
        f"{foundry_dir}/lef/sky130_ef_io__vdda_lvc_pad.lef",
        f"{foundry_dir}/lef/sky130_ef_io__vddio_hvc_pad.lef",
        f"{foundry_dir}/lef/sky130_ef_io__vddio_lvc_pad.lef",
        f"{foundry_dir}/lef/sky130_ef_io__vssa_hvc_pad.lef",
        f"{foundry_dir}/lef/sky130_ef_io__vssa_lvc_pad.lef",
        f"{foundry_dir}/lef/sky130_ef_io__vssd_hvc_pad.lef",
        f"{foundry_dir}/lef/sky130_ef_io__vssd_lvc_pad.lef",
        f"{foundry_dir}/lef/sky130_ef_io__vssio_hvc_pad.lef",
        f"{foundry_dir}/lef/sky130_ef_io__vssio_lvc_pad.lef",
        f"{foundry_dir}/lef/sky130_fd_io__top_xres4v2.lef",
        f"{foundry_dir}/lef/sky130io_fill.lef",
        f"{foundry_dir}/lef/sky130_sram_1rw1r_128x256_8.lef",
        f"{foundry_dir}/lef/sky130_sram_1rw1r_44x64_8.lef",
        f"{foundry_dir}/lef/sky130_sram_1rw1r_64x256_8.lef",
        f"{foundry_dir}/lef/sky130_sram_1rw1r_80x64_8.lef",
    ]
    workspace.set_lefs(lefs)

    libs = [
        f"{foundry_dir}/lib/sky130_fd_sc_hs__tt_025C_1v80.lib",
        f"{foundry_dir}/lib/sky130_dummy_io.lib",
        f"{foundry_dir}/lib/sky130_sram_1rw1r_128x256_8_TT_1p8V_25C.lib",
        f"{foundry_dir}/lib/sky130_sram_1rw1r_44x64_8_TT_1p8V_25C.lib",
        f"{foundry_dir}/lib/sky130_sram_1rw1r_64x256_8_TT_1p8V_25C.lib",
        f"{foundry_dir}/lib/sky130_sram_1rw1r_80x64_8_TT_1p8V_25C.lib",
    ]
    workspace.set_libs(libs)

    workspace.set_process_node("sky130")
    workspace.set_project(design)
    workspace.set_design(design)
    workspace.set_version("V1")
    workspace.set_task("run_eda")
    workspace.set_first_routing_layer("li1")

    # iEDA config (HS)
    workspace.set_ieda_fixfanout_buffer("sky130_fd_sc_hs__buf_8")
    workspace.set_ieda_cts_buffers(["sky130_fd_sc_hs__buf_1"])
    workspace.set_ieda_cts_root_buffer("sky130_fd_sc_hs__buf_1")
    workspace.set_ieda_placement_buffers(["sky130_fd_sc_hs__buf_1"])
    workspace.set_ieda_filler_cells_for_first_iteration(
        [
            "sky130_fd_sc_hs__fill_8",
            "sky130_fd_sc_hs__fill_4",
            "sky130_fd_sc_hs__fill_2",
            "sky130_fd_sc_hs__fill_1",
        ]
    )
    workspace.set_ieda_filler_cells_for_second_iteration(
        [
            "sky130_fd_sc_hs__fill_8",
            "sky130_fd_sc_hs__fill_4",
            "sky130_fd_sc_hs__fill_2",
            "sky130_fd_sc_hs__fill_1",
        ]
    )
    workspace.set_ieda_optdrv_buffers(["sky130_fd_sc_hs__buf_8"])
    workspace.set_ieda_opthold_buffers(["sky130_fd_sc_hs__buf_8"])
    workspace.set_ieda_optsetup_buffers(["sky130_fd_sc_hs__buf_8"])
    workspace.set_ieda_router_layer(bottom_layer="met1", top_layer="met4")

    # Optional parameters tweak
    params = EDAParameters()
    params.placement_target_density = 0.4
    workspace.update_parameters(parameters=params)

    return workspace


def run_eda_flow(workspace) -> None:
    """Run a minimal iEDA physical flow to generate a routed DEF.

    Note: `IEDAFloorplan._run_flow()` is currently a stub in this repo, so we
    follow the same pattern as `test/test_sky130_gcd.py`: do floorplan manually
    (read verilog -> init floorplan -> save DEF), then run the remaining steps
    using `RunIEDA` helpers.
    """

    def _out_def(step: DbFlow.FlowStep) -> str:
        return workspace.configs.get_output_def(DbFlow(eda_tool="iEDA", step=step), compressed=False)

    def _out_verilog(step: DbFlow.FlowStep) -> str:
        return workspace.configs.get_output_verilog(DbFlow(eda_tool="iEDA", step=step), compressed=False)

    def _require(path: str, label: str) -> None:
        if not Path(path).exists():
            raise RuntimeError(f"Missing {label}: {path}")

    def run_floorplan(workspace) -> None:
        from multiprocessing import Process
        from pathlib import Path

        def _floorplan_proc() -> None:
            from aieda.eda import IEDAFloorplan

            flow = DbFlow(
                eda_tool="iEDA",
                step=DbFlow.FlowStep.floorplan,
                input_def=workspace.configs.paths.def_input_path,
                input_verilog=workspace.configs.paths.verilog_input_path,
            )
            flow.set_state_running()
            workspace.configs.save_flow_state(flow)

            # Use uncompressed outputs; some environments don’t produce .gz files.
            flow.output_def = _out_def(DbFlow.FlowStep.floorplan)
            flow.output_verilog = _out_verilog(DbFlow.FlowStep.floorplan)

            ieda_fp = IEDAFloorplan(workspace, flow)
            ieda_fp.read_verilog()

            # Use the same sky130 track/PDN defaults as the gcd example.
            ieda_fp.init_floorplan_by_core_utilization(
                core_site="unit",
                io_site="unit",
                corner_site="unit",
                core_util=0.4,
                x_margin=0,
                y_margin=0,
                xy_ratio=1,
            )

            ieda_fp.gern_track(layer="li1", x_start=240, x_step=480, y_start=185, y_step=370)
            ieda_fp.gern_track(layer="met1", x_start=185, x_step=370, y_start=185, y_step=370)
            ieda_fp.gern_track(layer="met2", x_start=240, x_step=480, y_start=240, y_step=480)
            ieda_fp.gern_track(layer="met3", x_start=370, x_step=740, y_start=370, y_step=740)
            ieda_fp.gern_track(layer="met4", x_start=480, x_step=960, y_start=480, y_step=960)
            ieda_fp.gern_track(layer="met5", x_start=185, x_step=3330, y_start=185, y_step=3330)

            ieda_fp.add_pdn_io(net_name="VDD", direction="INOUT", is_power=True)
            ieda_fp.add_pdn_io(net_name="VSS", direction="INOUT", is_power=False)

            ieda_fp.global_net_connect(net_name="VDD", instance_pin_name="VPWR", is_power=True)
            ieda_fp.global_net_connect(net_name="VDD", instance_pin_name="VPB", is_power=True)
            ieda_fp.global_net_connect(net_name="VDD", instance_pin_name="vdd", is_power=True)
            ieda_fp.global_net_connect(net_name="VSS", instance_pin_name="VGND", is_power=False)
            ieda_fp.global_net_connect(net_name="VSS", instance_pin_name="VNB", is_power=False)
            ieda_fp.global_net_connect(net_name="VSS", instance_pin_name="gnd", is_power=False)

            ieda_fp.auto_place_pins(layer="met5", width=2000, height=2000)

            ieda_fp.tapcell(
                tapcell="sky130_fd_sc_hs__tap_1",
                distance=14,
                endcap="sky130_fd_sc_hs__fill_1",
            )

            # Mark clock net if present.
            ieda_fp.set_net(net_name="clk", net_type="CLOCK")

            ieda_fp.def_save()
            ieda_fp.verilog_save()

            flow.set_state_finished()
            workspace.configs.save_flow_state(flow)

        p = Process(target=_floorplan_proc, args=())
        p.start()
        p.join()

        output_def = Path(_out_def(DbFlow.FlowStep.floorplan))
        if p.exitcode != 0 or not output_def.exists():
            raise RuntimeError(
                "Floorplan failed: iEDA did not produce the expected DEF. "
                "This usually means the input verilog cannot be parsed by iEDA (Rust parser panic) "
                "or the process aborted.\n"
                f"floorplan process exitcode: {p.exitcode}\n"
                f"expected DEF: {output_def}"
            )

    run_floorplan(workspace)

    _require(_out_def(DbFlow.FlowStep.floorplan), "floorplan DEF")
    _require(_out_verilog(DbFlow.FlowStep.floorplan), "floorplan verilog")

    run_ieda = RunIEDA(workspace)

    # Note: `RunIEDA.run_pdn()` does not accept custom output paths. We need
    # uncompressed outputs here (".def" instead of ".def.gz"), so call
    # `run_flow()` with an explicit `DbFlow`.
    run_ieda.run_flow(
        DbFlow(
            eda_tool="iEDA",
            step=DbFlow.FlowStep.pdn,
            input_def=_out_def(DbFlow.FlowStep.floorplan),
            output_def=_out_def(DbFlow.FlowStep.pdn),
            output_verilog=_out_verilog(DbFlow.FlowStep.pdn),
        )
    )
    _require(_out_def(DbFlow.FlowStep.pdn), "pdn DEF")
    _require(_out_verilog(DbFlow.FlowStep.pdn), "pdn verilog")

    run_ieda.run_fix_fanout(
        input_def=_out_def(DbFlow.FlowStep.pdn),
        output_def=_out_def(DbFlow.FlowStep.fixFanout),
        output_verilog=_out_verilog(DbFlow.FlowStep.fixFanout),
    )
    _require(_out_def(DbFlow.FlowStep.fixFanout), "fixFanout DEF")
    _require(_out_verilog(DbFlow.FlowStep.fixFanout), "fixFanout verilog")

    run_ieda.run_placement(
        input_def=_out_def(DbFlow.FlowStep.fixFanout),
        output_def=_out_def(DbFlow.FlowStep.place),
        output_verilog=_out_verilog(DbFlow.FlowStep.place),
    )
    _require(_out_def(DbFlow.FlowStep.place), "place DEF")
    _require(_out_verilog(DbFlow.FlowStep.place), "place verilog")

    run_ieda.run_CTS(
        input_def=_out_def(DbFlow.FlowStep.place),
        output_def=_out_def(DbFlow.FlowStep.cts),
        output_verilog=_out_verilog(DbFlow.FlowStep.cts),
    )
    _require(_out_def(DbFlow.FlowStep.cts), "CTS DEF")
    _require(_out_verilog(DbFlow.FlowStep.cts), "CTS verilog")

    run_ieda.run_optimizing_drv(
        input_def=_out_def(DbFlow.FlowStep.cts),
        output_def=_out_def(DbFlow.FlowStep.optDrv),
        output_verilog=_out_verilog(DbFlow.FlowStep.optDrv),
    )
    _require(_out_def(DbFlow.FlowStep.optDrv), "optDrv DEF")
    _require(_out_verilog(DbFlow.FlowStep.optDrv), "optDrv verilog")

    # NOTE: iEDA's TO Hold step can SIGFPE on some designs (observed on AES).
    # For vector generation we can safely skip hold optimization.

    run_ieda.run_legalization(
        input_def=_out_def(DbFlow.FlowStep.optDrv),
        output_def=_out_def(DbFlow.FlowStep.legalization),
        output_verilog=_out_verilog(DbFlow.FlowStep.legalization),
    )
    _require(_out_def(DbFlow.FlowStep.legalization), "legalization DEF")
    _require(_out_verilog(DbFlow.FlowStep.legalization), "legalization verilog")

    run_ieda.run_routing(
        input_def=_out_def(DbFlow.FlowStep.legalization),
        output_def=_out_def(DbFlow.FlowStep.route),
        output_verilog=_out_verilog(DbFlow.FlowStep.route),
    )
    _require(_out_def(DbFlow.FlowStep.route), "route DEF")
    _require(_out_verilog(DbFlow.FlowStep.route), "route verilog")



def main() -> int:
    root = _repo_root()

    netlist = root / "benchmarks" / "aes" / "syn_netlist" / "aes_sky130.v"
    sdc = root / "benchmarks" / "aes" / "syn_netlist" / "aes.sdc"

    if not netlist.exists() or not sdc.exists():
        print("Missing synthesized inputs. Run:")
        print("  uv run python test/synth_sky130_aes.py")
        print(f"Expected netlist: {netlist}")
        print(f"Expected sdc: {sdc}")
        return 2

    workspace_dir = str(root / "example" / "sky130_aes")
    design = "aes"

    ws = create_workspace_sky130_aes(workspace_dir, design, str(netlist), str(sdc))

    run_eda_flow(ws)

    data_gen = DataGeneration(ws)
    data_gen.generate_vectors(
        input_def=ws.configs.get_output_def(DbFlow(eda_tool="iEDA", step=DbFlow.FlowStep.route), compressed=False),
        vectors_dir=ws.paths_table.ieda_output["vectors"],
        patch_row_step=18,
        patch_col_step=18,
    )

    print("Done. Net vectors should be under:")
    print(f"  {workspace_dir}/output/iEDA/vectors/nets")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
