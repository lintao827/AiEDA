#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""Synthesize secworks/aes RTL to a sky130 gate-level netlist using yosys.

Why this exists:
- iEDA physical design expects a mapped (standard-cell) gate-level netlist.
- The secworks/aes repo you cloned under benchmarks/ is RTL.

Outputs:
- benchmarks/aes/syn_netlist/aes_sky130.v
- benchmarks/aes/syn_netlist/aes.sdc

Usage:
  uv run python test/synth_sky130_aes.py

Notes:
- Requires yosys available in PATH.
- Uses the sky130 HS liberty already vendored in this repo.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


def _expand_concat_item(item: str) -> list[str] | None:
    item = item.strip()
    if not item:
        return []

    # Bit-select: foo[3]
    m = __import__("re").match(r"^(?P<base>[^\[]+)\[(?P<idx>\d+)\]$", item)
    if m:
        base = m.group("base").strip()
        idx = int(m.group("idx"))
        return [f"{base}[{idx}]"]

    # Part-select: foo[7:4] or foo[4:7]
    m = __import__("re").match(r"^(?P<base>[^\[]+)\[(?P<msb>\d+)\s*:\s*(?P<lsb>\d+)\]$", item)
    if m:
        base = m.group("base").strip()
        msb = int(m.group("msb"))
        lsb = int(m.group("lsb"))
        step = -1 if msb >= lsb else 1
        return [f"{base}[{i}]" for i in range(msb, lsb + step, step)]

    # Constants in concatenations (e.g. 1'b0) are rare here; skip.
    return None


def _sanitize_netlist_for_ieda(netlist_path: Path) -> int:
    """Rewrite unsupported 'assign {..} = {..};' into per-bit assigns.

    iEDA's Rust Verilog parser (in this repo) crashes on concatenation on the LHS.
    This sanitizer keeps semantics but avoids that syntax.
    """

    import re

    text = netlist_path.read_text()
    lines = text.splitlines(keepends=True)

    pattern = re.compile(
        r"^(?P<indent>\s*)assign\s*\{(?P<lhs>[^}]*)\}\s*=\s*\{(?P<rhs>[^}]*)\}\s*;\s*$"
    )
    changed = 0
    out_lines: list[str] = []

    for line in lines:
        m = pattern.match(line.rstrip("\n"))
        if not m:
            out_lines.append(line)
            continue

        indent = m.group("indent")
        lhs_items = [p.strip() for p in m.group("lhs").split(",") if p.strip()]
        rhs_items = [p.strip() for p in m.group("rhs").split(",") if p.strip()]

        lhs_bits: list[str] = []
        rhs_bits: list[str] = []

        ok = True
        for it in lhs_items:
            expanded = _expand_concat_item(it)
            if expanded is None:
                ok = False
                break
            lhs_bits.extend(expanded)
        if ok:
            for it in rhs_items:
                expanded = _expand_concat_item(it)
                if expanded is None:
                    ok = False
                    break
                rhs_bits.extend(expanded)

        if not ok or len(lhs_bits) != len(rhs_bits) or len(lhs_bits) == 0:
            # Leave as-is; better to keep original netlist than corrupt it.
            out_lines.append(line)
            continue

        for lb, rb in zip(lhs_bits, rhs_bits, strict=True):
            out_lines.append(f"{indent}assign {lb} = {rb};\n")
        changed += 1

    if changed:
        netlist_path.write_text("".join(out_lines))
    return changed


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def main() -> int:
    root = _repo_root()

    yosys = shutil.which("yosys")
    if not yosys:
        print("ERROR: yosys not found in PATH.\n")
        print("Install options:")
        print("  - Ubuntu/Debian: sudo apt-get update && sudo apt-get install -y yosys")
        print("  - Conda: conda install -c conda-forge yosys")
        print("  - Or run inside a container that includes yosys")
        return 2

    rtl_dir = root / "benchmarks" / "aes" / "src" / "rtl"
    if not rtl_dir.exists():
        print(f"ERROR: RTL directory not found: {rtl_dir}")
        return 2

    rtl_files = sorted(rtl_dir.glob("*.v"))
    if not rtl_files:
        print(f"ERROR: No .v files found under: {rtl_dir}")
        return 2

    # Prefer the wrapper top module `aes` (has clk/reset_n + simple bus interface)
    top = "aes"

    liberty = root / "aieda" / "third_party" / "iEDA" / "scripts" / "foundry" / "sky130" / "lib" / "sky130_fd_sc_hs__tt_025C_1v80.lib"
    if not liberty.exists():
        print(f"ERROR: liberty not found: {liberty}")
        return 2

    out_dir = root / "benchmarks" / "aes" / "syn_netlist"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_netlist = out_dir / "aes_sky130.v"
    out_sdc = out_dir / "aes.sdc"

    read_cmd = " ".join(str(p) for p in rtl_files)

    yosys_script = f"""
read_verilog -sv {read_cmd}

hierarchy -check -top {top}

# Normalize RTL into a synthesizable netlist
proc; opt
fsm; opt
memory; opt
techmap; opt

# Map sequential + combinational logic to the provided standard-cell liberty

dfflibmap -liberty {liberty}
abc -liberty {liberty}
clean

# iEDA’s Rust Verilog parser is strict about escaped identifiers (e.g. \\foo.bar).
# Hide (privatize) public internal names so Yosys will auto-rename them to
# simple identifiers when writing the netlist.
rename -hide
clean

# iEDA's Rust verilog parser is stricter than typical simulators.
# With Yosys 0.9 (common on Ubuntu apt), `write_verilog` does not support
# `-simple-lhs`. Instead, we emit expression-form Verilog (default, i.e. do NOT
# use `-noexpr`) to avoid concatenation on the LHS of assigns.
write_verilog -noattr {out_netlist}
stat -liberty {liberty}
""".strip()

    script_path = out_dir / "synth_aes_sky130.ys"
    script_path.write_text(yosys_script)

    print(f"RTL inputs: {len(rtl_files)} files under {rtl_dir}")
    print(f"Top module: {top}")
    print(f"Liberty: {liberty}")
    print(f"Running: yosys -s {script_path}")

    proc = subprocess.run([yosys, "-s", str(script_path)], text=True)
    if proc.returncode != 0:
        print("ERROR: yosys failed")
        return proc.returncode

    rewrites = _sanitize_netlist_for_ieda(out_netlist)
    if rewrites:
        print(f"Sanitized netlist for iEDA: rewrote {rewrites} concatenation assign(s)")

    # Minimal SDC for iEDA. Adjust period if you want.
    sdc_text = """# Auto-generated SDC for secworks/aes (top: aes)
create_clock -name clk -period 10 [get_ports clk]
set_false_path -from [get_ports reset_n]
"""
    out_sdc.write_text(sdc_text)

    print(f"Wrote netlist: {out_netlist}")
    print(f"Wrote SDC: {out_sdc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
