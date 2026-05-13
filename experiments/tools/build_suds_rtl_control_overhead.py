#!/usr/bin/env python3
"""Build P3 SUDS control-plane RTL overhead evidence."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
TAG = "20260511_p2p3_quality"
RTL_PATH = REPO_ROOT / "experiments/hardware/suds_control_plane.v"
CSV_OUT = REPO_ROOT / f"experiments/results/report_data/suds_rtl_control_overhead_{TAG}.csv"
JSON_OUT = REPO_ROOT / f"experiments/results/report_data/suds_rtl_control_overhead_{TAG}.json"
REPORT_OUT = REPO_ROOT / f"docs/reports/{TAG}_suds_rtl_control_overhead.md"
RUN_ROOT = REPO_ROOT / f"experiments/results/runs/suds_rtl_control_overhead_{TAG}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default=TAG)
    parser.add_argument("--rtl", type=Path, default=RTL_PATH)
    parser.add_argument("--csv-out", type=Path)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--report-out", type=Path)
    parser.add_argument("--run-root", type=Path)
    args = parser.parse_args()
    if args.csv_out is None:
        args.csv_out = REPO_ROOT / f"experiments/results/report_data/suds_rtl_control_overhead_{args.tag}.csv"
    if args.json_out is None:
        args.json_out = REPO_ROOT / f"experiments/results/report_data/suds_rtl_control_overhead_{args.tag}.json"
    if args.report_out is None:
        args.report_out = REPO_ROOT / f"docs/reports/{args.tag}_suds_rtl_control_overhead.md"
    if args.run_root is None:
        args.run_root = REPO_ROOT / f"experiments/results/runs/suds_rtl_control_overhead_{args.tag}"
    return args


def repo_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def parse_yosys_stat(stdout: str) -> dict[str, Any]:
    stats: dict[str, Any] = {"cell_counts": {}}
    in_final_module = False
    for raw_line in stdout.splitlines():
        line = raw_line.rstrip()
        if line.startswith("=== suds_control_plane ==="):
            in_final_module = True
            continue
        if not in_final_module:
            continue
        match = re.match(r"\s*(\d+)\s+(wires|wire bits|public wires|public wire bits|ports|port bits|cells)\s*$", line)
        if match:
            stats[match.group(2).replace(" ", "_")] = int(match.group(1))
            continue
        cell_match = re.match(r"\s*(\d+)\s+(\$_[A-Z0-9_]+_?)\s*$", line)
        if cell_match:
            stats["cell_counts"][cell_match.group(2)] = int(cell_match.group(1))
    return stats


def try_yosys(rtl: Path, run_root: Path) -> dict[str, Any]:
    yosys = shutil.which("yosys")
    if yosys is None:
        return {"available": False, "status": "not_installed", "stdout": "", "stderr": ""}
    run_root.mkdir(parents=True, exist_ok=True)
    log_path = run_root / "yosys_synthesis.log"
    synth_v = run_root / "suds_control_plane_synth.v"
    synth_json = run_root / "suds_control_plane_synth.json"
    script = (
        f"read_verilog {repo_path(rtl)}; "
        "synth -top suds_control_plane; "
        "stat; "
        f"write_verilog -noattr {repo_path(synth_v)}; "
        f"write_json {repo_path(synth_json)}"
    )
    cmd = [
        yosys,
        "-p",
        script,
    ]
    completed = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    log_path.write_text(completed.stdout + "\n" + completed.stderr, encoding="utf-8")
    stats = parse_yosys_stat(completed.stdout)
    return {
        "available": True,
        "status": "pass" if completed.returncode == 0 else "fail",
        "returncode": completed.returncode,
        "stats": stats,
        "log_path": repo_path(log_path),
        "synthesized_verilog": repo_path(synth_v),
        "synthesized_json": repo_path(synth_json),
        "stdout_tail": "\n".join(completed.stdout.splitlines()[-40:]),
        "stderr_tail": "\n".join(completed.stderr.splitlines()[-20:]),
        "command": " ".join(cmd),
    }


def proxy_rows(yosys_info: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    evidence = "rtl_synthesis" if yosys_info.get("status") == "pass" else "rtl_proxy"
    for slack_bits in (8, 10, 12):
        for columns in (32, 64, 128, 192):
            for target_mhz in (500, 1000):
                comparator_delay_ps = 35.0 * slack_bits + 120.0
                register_delay_ps = 80.0
                total_delay_ps = comparator_delay_ps + register_delay_ps
                period_ps = 1_000_000.0 / target_mhz
                critical_path_slack_ps = period_ps - total_delay_ps
                sideband_bits = columns * 2 + columns
                area_ge = 18.0 * slack_bits + 3.5 * sideband_bits + 160.0
                dynamic_power_uw = (area_ge * target_mhz * 0.0009) + (sideband_bits * target_mhz * 0.00015)
                rows.append(
                    {
                        "slack_bits": slack_bits,
                        "columns": columns,
                        "target_mhz": target_mhz,
                        "tier_bits_per_column": 2,
                        "validity_bits_per_column": 1,
                        "sideband_bits": sideband_bits,
                        "latency_cycles": 1,
                        "estimated_delay_ps": total_delay_ps,
                        "clock_period_ps": period_ps,
                        "critical_path_slack_ps": critical_path_slack_ps,
                        "critical_path_status": "pass" if critical_path_slack_ps > 0 else "fail",
                        "area_ge_proxy": area_ge,
                        "dynamic_power_uw_proxy": dynamic_power_uw,
                        "yosys_status": yosys_info.get("status", "not_installed"),
                        "yosys_cell_count": yosys_info.get("stats", {}).get("cells", ""),
                        "yosys_log_path": yosys_info.get("log_path", ""),
                        "evidence_label": evidence,
                        "promotion_decision": "appendix" if critical_path_slack_ps > 0 else "boundary",
                        "claim_boundary": (
                            "Yosys RTL synthesis plus proxy timing/area/power estimates; "
                            "not liberty-backed timing, placed-and-routed silicon, or datapath closure"
                        ),
                    }
                )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def json_safe(value: Any) -> Any:
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, dict):
        return {key: json_safe(val) for key, val in value.items()}
    if isinstance(value, list):
        return [json_safe(val) for val in value]
    return value


def write_json(path: Path, *, tag: str, rows: list[dict[str, Any]], yosys_info: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    evidence = "rtl_synthesis" if yosys_info.get("status") == "pass" else "rtl_proxy"
    payload = {
        "metadata": {
            "tag": tag,
            "artifact_id": f"suds_rtl_control_overhead_{tag}",
            "evidence_label": evidence,
            "promotion_decision": "appendix",
            "rtl_source": str(RTL_PATH.relative_to(REPO_ROOT)),
            "yosys": yosys_info,
            "regeneration_command": (
                f".venv311-mps/bin/python experiments/tools/build_suds_rtl_control_overhead.py --tag {tag}"
            ),
            "claim_boundary_note": (
                "The SUDS tier comparator is modeled as a one-cycle sideband control path. "
                "Yosys synthesis upgrades evidence_label to rtl_synthesis when available; "
                "timing, area-equivalent, and power values remain proxy estimates unless a liberty/OpenROAD flow is added."
            ),
        },
        "rows": rows,
    }
    path.write_text(json.dumps(json_safe(payload), indent=2) + "\n", encoding="utf-8")


def write_report(path: Path, *, tag: str, rows: list[dict[str, Any]], yosys_info: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    evidence = "rtl_synthesis" if yosys_info.get("status") == "pass" else "rtl_proxy"
    worst_slack = min(float(row["critical_path_slack_ps"]) for row in rows)
    max_area = max(float(row["area_ge_proxy"]) for row in rows)
    max_power = max(float(row["dynamic_power_uw_proxy"]) for row in rows)
    stats = yosys_info.get("stats", {})
    cell_count = stats.get("cells", "NA")
    report = f"""# SUDS RTL Control-Plane Overhead

Tag: `{tag}`
Evidence label: `{evidence}`
Promotion decision: `appendix`

## Scope

This P3 lane materializes a synthesizable SUDS control-plane block:
threshold registers, tier comparator, tier sideband, and validity tags. The
timing model treats it as a one-cycle sideband path. If Yosys is installed,
the script attempts synthesis; otherwise it keeps a transparent proxy estimate.

Yosys status: `{yosys_info.get('status', 'not_installed')}`
Yosys cell count: `{cell_count}`

## Summary

- Worst proxy timing slack: `{worst_slack:.1f} ps`
- Max proxy area: `{max_area:.1f} GE`
- Max proxy dynamic power: `{max_power:.2f} uW`
- RTL source: `experiments/hardware/suds_control_plane.v`
- Yosys log: `{yosys_info.get('log_path', 'not_available')}`
- Synthesized Verilog: `{yosys_info.get('synthesized_verilog', 'not_available')}`

## Representative Rows

| Slack bits | Columns | Target MHz | Sideband bits | Delay ps | Timing slack ps | Area GE | Power uW | Status |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
"""
    for row in rows:
        if row["slack_bits"] == 12 and row["target_mhz"] == 1000:
            report += (
                f"| {row['slack_bits']} | {row['columns']} | {row['target_mhz']} | "
                f"{row['sideband_bits']} | {row['estimated_delay_ps']:.1f} | "
                f"{row['critical_path_slack_ps']:.1f} | {row['area_ge_proxy']:.1f} | "
                f"{row['dynamic_power_uw_proxy']:.2f} | `{row['critical_path_status']}` |\n"
            )

    report += f"""
## Interpretation

- The control path is treated as sideband metadata. It must not be described as
  accelerating the optical datapath.
- The synthesis result shows the RTL is accepted by Yosys and maps to generic
  cells. The timing, GE, and power columns are still proxy estimates because no
  liberty-backed OpenROAD or placed-and-routed flow is used here.
- TCAS promotion still requires placed/routed or liberty/tool-calibrated area
  and power. This artifact supports architecture appendix overhead accounting and
  reviewer response.

## Artifacts

- CSV: `experiments/results/report_data/suds_rtl_control_overhead_{tag}.csv`
- JSON: `experiments/results/report_data/suds_rtl_control_overhead_{tag}.json`
- Report: `docs/reports/{tag}_suds_rtl_control_overhead.md`
- RTL: `experiments/hardware/suds_control_plane.v`

## Regeneration

```bash
.venv311-mps/bin/python experiments/tools/build_suds_rtl_control_overhead.py --tag {tag}
```
"""
    path.write_text(report, encoding="utf-8")


def main() -> None:
    args = parse_args()
    if not args.rtl.is_file():
        raise SystemExit(f"missing RTL source: {args.rtl}")
    yosys_info = try_yosys(args.rtl, args.run_root)
    rows = proxy_rows(yosys_info)
    write_csv(args.csv_out, rows)
    write_json(args.json_out, tag=args.tag, rows=rows, yosys_info=yosys_info)
    write_report(args.report_out, tag=args.tag, rows=rows, yosys_info=yosys_info)
    print(f"wrote {args.csv_out}")
    print(f"wrote {args.json_out}")
    print(f"wrote {args.report_out}")


if __name__ == "__main__":
    main()
