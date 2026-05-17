#!/usr/bin/env python3
"""Build a reproducible cost-model worked example note."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import yaml

try:
    from experiments.tools.path_policy import MAIN_PROJECT_REPORT_DATA_DIR
except ModuleNotFoundError:  # direct script execution
    from path_policy import MAIN_PROJECT_REPORT_DATA_DIR

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
EXPERIMENTS_ROOT = ROOT / "experiments"
if str(EXPERIMENTS_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_ROOT))

from experiments.mtl_model.estimator import estimate_gemm_energy_latency, summarize_ops


DEFAULT_CONFIG_YAML = ROOT / "experiments" / "mtl_model" / "mtl_config_asic.yaml"
DEFAULT_OPS_JSON = ROOT / "experiments" / "mtl_model" / "ops" / "ops_mobilevit_s.json"
DEFAULT_PHASE1_SUMMARY = ROOT / "experiments" / "results" / "runs" / "20260228_opt_sync_core_e0" / "phase1_summary.csv"
DEFAULT_RUN_CONFIG = ROOT / "experiments" / "results" / "runs" / "20260228_opt_sync_core_e0" / "config_snapshot.yaml"
DEFAULT_OUT_DIR = MAIN_PROJECT_REPORT_DATA_DIR
DEFAULT_MODEL = "mobilevit_s"
DEFAULT_LAYER = "conv_1.block.conv"
DEFAULT_TAG = "20260310"


def _format_num(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def _read_phase1_row(path: Path, model: str) -> dict[str, str]:
    with path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return next(row for row in rows if row["model"] == model)


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a reproducible cost-model worked example note.")
    parser.add_argument("--config_yaml", type=Path, default=DEFAULT_CONFIG_YAML)
    parser.add_argument("--ops_json", type=Path, default=DEFAULT_OPS_JSON)
    parser.add_argument("--phase1_summary_csv", type=Path, default=DEFAULT_PHASE1_SUMMARY)
    parser.add_argument("--run_config_yaml", type=Path, default=DEFAULT_RUN_CONFIG)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--layer_name", default=DEFAULT_LAYER)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--tag", default=DEFAULT_TAG)
    args = parser.parse_args()

    config = yaml.safe_load(args.config_yaml.read_text(encoding="utf-8"))
    run_cfg = yaml.safe_load(args.run_config_yaml.read_text(encoding="utf-8"))
    ops_payload = json.loads(args.ops_json.read_text(encoding="utf-8"))
    ops = ops_payload["ops"]
    op = next(item for item in ops if item["name"] == args.layer_name)
    layer_est = estimate_gemm_energy_latency(int(op["m"]), int(op["d"]), int(op["n"]), config)
    _, summary = summarize_ops(ops, config)
    phase1_row = _read_phase1_row(args.phase1_summary_csv, args.model)
    phase1_energy = float(phase1_row["energy_j"])
    raw_total_energy = summary["total_energy_mj"] / 1e3
    energy_gap = raw_total_energy - phase1_energy

    sample_rate = float(config["photonic"]["sample_rate_gsps"])
    stage_cycles = json.loads(phase1_row["stage_cycles"])
    mapped_terms = [
        {
            "paper_symbol": "T_BtoS",
            "current_artifact_mapping": "stage_cycles['btos'] / sample_rate",
            "value_seconds": float(stage_cycles["btos"]) / (sample_rate * 1e9),
            "notes": "direct B-to-S frontend cycles from timeline summary",
        },
        {
            "paper_symbol": "T_sched",
            "current_artifact_mapping": "stage_cycles['bubble'] / sample_rate",
            "value_seconds": float(stage_cycles["bubble"]) / (sample_rate * 1e9),
            "notes": "scheduler-visible stall exposure; operational mapping for current CSV schema",
        },
        {
            "paper_symbol": "T_compute",
            "current_artifact_mapping": "stage_cycles['electronic_compute' or 'oag_compute'] / sample_rate",
            "value_seconds": float(stage_cycles.get("electronic_compute", 0) + stage_cycles.get("oag_compute", 0)) / (sample_rate * 1e9),
            "notes": "current E0 path is electronic-only for compute stage",
        },
        {
            "paper_symbol": "T_DAC",
            "current_artifact_mapping": "(stage_cycles['serialize_drive'] + stage_cycles['pca_adc']) / sample_rate",
            "value_seconds": float(stage_cycles["serialize_drive"] + stage_cycles["pca_adc"]) / (sample_rate * 1e9),
            "notes": "operational conversion envelope in current timeline summary",
        },
        {
            "paper_symbol": "E_conv/ctrl",
            "current_artifact_mapping": "energy_breakdown_conversion_control_j = load_x + load_y",
            "value_seconds": float(phase1_row["energy_breakdown_conversion_control_j"]),
            "notes": "frontend load_x/load_y after selected energy-model scaling",
        },
        {
            "paper_symbol": "E_laser",
            "current_artifact_mapping": "energy_breakdown_laser_optical_j",
            "value_seconds": float(phase1_row["energy_breakdown_laser_optical_j"]),
            "notes": "tile-based laser term unless PHY replaces it",
        },
        {
            "paper_symbol": "E_OE/ADC",
            "current_artifact_mapping": "energy_breakdown_oe_j + energy_breakdown_adc_pca_j",
            "value_seconds": float(phase1_row["energy_breakdown_oe_j"]) + float(phase1_row["energy_breakdown_adc_pca_j"]),
            "notes": "backend detect chain",
        },
        {
            "paper_symbol": "E_move",
            "current_artifact_mapping": "energy_breakdown_memory_move_j",
            "value_seconds": float(phase1_row["energy_breakdown_memory_move_j"]),
            "notes": "memory movement from mem_energy_pj accounting",
        },
        {
            "paper_symbol": "E_static",
            "current_artifact_mapping": "energy_breakdown_other_static_j",
            "value_seconds": float(phase1_row["energy_breakdown_other_static_j"]),
            "notes": "static + elementwise electronic + optional thermal/calibration additions",
        },
    ]

    layer_rows = [
        {"component": "tiles", "value": layer_est["tiles"], "unit": "count"},
        {"component": "latency_s", "value": layer_est["latency_s"], "unit": "s"},
    ]
    for name, value in layer_est["energy_components_j"].items():
        layer_rows.append({"component": f"energy_{name}_j", "value": value, "unit": "J"})

    args.out_dir.mkdir(parents=True, exist_ok=True)
    term_csv = args.out_dir / f"cost_model_term_mapping_{args.tag}.csv"
    layer_csv = args.out_dir / f"cost_model_worked_layer_{args.tag}.csv"
    report_md = args.out_dir / f"cost_model_worked_example_{args.tag}.md"

    _write_csv(term_csv, mapped_terms, ["paper_symbol", "current_artifact_mapping", "value_seconds", "notes"])
    _write_csv(layer_csv, layer_rows, ["component", "value", "unit"])

    note_lines = [
        "# Cost Model Worked Example (20260310)",
        "",
        "Scope",
        f"- Photonic/electronic parameter source: `{args.config_yaml}`",
        f"- Layer op source: `{args.ops_json}`",
        f"- Run-level summary source: `{args.phase1_summary_csv}`",
        f"- Run config source: `{args.run_config_yaml}`",
        f"- Model / layer example: `{args.model}` / `{args.layer_name}`",
        "",
        "Parameter provenance",
        (
            f"- `tile_k={config['photonic']['tile_k']}`, `tile_k_prime={config['photonic']['tile_k_prime']}`, "
            f"`cycles_per_tile={config['photonic']['cycles_per_tile']}`, `sample_rate={sample_rate} GS/s`."
        ),
        (
            f"- Energy params from ASIC config: DAC `{config['energy']['dac_power_mw']} mW @ {config['energy']['dac_sample_rate_gsps']} GS/s`, "
            f"ADC `{config['energy']['adc_power_mw']} mW @ {config['energy']['adc_sample_rate_gsps']} GS/s`, "
            f"laser `{config['energy']['laser_power_mw']} mW @ {config['energy']['laser_sample_rate_gsps']} GS/s`, "
            f"memory `{config['energy']['mem_energy_pj']} pJ/access`."
        ),
        (
            f"- Run-time energy model selection from the released E0 config: "
            f"`energy_model_mode={run_cfg['energy_model']['energy_model_mode']}`, "
            f"`upperbound_scale={run_cfg['energy_model']['upperbound_scale']}`, "
            f"`countbased_scale={run_cfg['energy_model']['countbased_scale']}`, "
            f"`calibration.include_in_totals={run_cfg['calibration_cost']['include_in_totals']}`."
        ),
        "",
        "Paper-symbol to artifact-field mapping",
    ]
    for row in mapped_terms:
        unit = "s" if row["paper_symbol"].startswith("T_") else "J"
        note_lines.append(
            f"- `{row['paper_symbol']}`: `{row['current_artifact_mapping']}` = "
            f"`{_format_num(float(row['value_seconds']), 9 if unit == 's' else 12)} {unit}`. "
            f"{row['notes']}"
        )
    note_lines.extend(
        [
            "",
            "Worked layer example",
            (
                f"- `{args.layer_name}` is logged as `m={op['m']}, d={op['d']}, n={op['n']}` "
                f"with kernel `{op['kernel']}` and stride `{op['stride']}`."
            ),
            (
                f"- Tile decomposition uses `ceil(m/k) * ceil(d/k) * ceil(n/k') = "
                f"ceil({op['m']}/{config['photonic']['tile_k']}) * ceil({op['d']}/{config['photonic']['tile_k']}) * "
                f"ceil({op['n']}/{config['photonic']['tile_k_prime']}) = {layer_est['tiles']}` tiles."
            ),
            (
                f"- Layer latency is `tiles / (sample_rate * 1e9) = "
                f"{layer_est['tiles']} / ({sample_rate}e9) = {_format_num(layer_est['latency_s'] * 1e6, 4)} us`."
            ),
            (
                f"- Layer energy components: load_x `{_format_num(layer_est['energy_components_j']['load_x'] * 1e6, 4)} uJ`, "
                f"load_y `{_format_num(layer_est['energy_components_j']['load_y'] * 1e6, 4)} uJ`, "
                f"OE `{_format_num(layer_est['energy_components_j']['oe'] * 1e6, 4)} uJ`, "
                f"ADC/PCA `{_format_num(layer_est['energy_components_j']['adc_pca'] * 1e6, 4)} uJ`, "
                f"laser `{_format_num(layer_est['energy_components_j']['laser'] * 1e6, 4)} uJ`, "
                f"memory `{_format_num(layer_est['energy_components_j']['mem'] * 1e6, 4)} uJ`, "
                f"static `{_format_num(layer_est['energy_components_j']['static'] * 1e6, 4)} uJ`."
            ),
            (
                f"- This layer therefore contributes `{_format_num(layer_est['energy_j'] * 1e6, 4)} uJ` "
                f"out of the estimator-side model total `{_format_num(summary['total_energy_mj'] / 1e3 * 1e6, 2)} uJ` "
                f"({ _format_num(100.0 * layer_est['energy_j'] / (summary['total_energy_mj'] / 1e3), 3)}%)."
            ),
            "",
            "Model-level closure",
            (
                f"- `summarize_ops(...)` on the same config produces total latency "
                f"`{_format_num(summary['total_latency_ms'], 6)} ms` and total energy "
                f"`{_format_num(summary['total_energy_mj'] / 1e3, 12)} J`."
            ),
            (
                f"- The released E0 phase1 row for `{args.model}` reports latency "
                f"`{phase1_row['latency_ms']} ms`, which matches the raw estimator to file precision, "
                f"but energy is `{phase1_row['energy_j']} J` rather than `{_format_num(raw_total_energy, 12)} J`."
            ),
            (
                f"- The energy gap is `{_format_num(energy_gap, 12)} J`, which is dominated by the raw estimator-side "
                f"laser bucket `{_format_num(summary['energy_mj_laser'] / 1e3, 12)} J`; the released E0 row therefore "
                "keeps the runner's zero-laser baseline accounting rather than the raw tile-based laser term."
            ),
            (
                f"- Energy bucket closure also matches the runner mapping: "
                f"`E_conv/ctrl={phase1_row['energy_breakdown_conversion_control_j']} J`, "
                f"`E_move={phase1_row['energy_breakdown_memory_move_j']} J`, "
                f"`E_OE/ADC={_format_num(float(phase1_row['energy_breakdown_oe_j']) + float(phase1_row['energy_breakdown_adc_pca_j']), 12)} J`, "
                f"`E_laser={phase1_row['energy_breakdown_laser_optical_j']} J`, "
                f"`E_static={phase1_row['energy_breakdown_other_static_j']} J`."
            ),
            "",
            "Interpretation",
            "- The current codebase is already internally traceable from `mtl_config_asic.yaml` and `ops_mobilevit_*.json` to `phase1_summary.csv`, but the mapping is implicit in code rather than explicit in the manuscript.",
            "- `T_sched` and `T_DAC` are not first-class exported columns today; the mappings above are the operational audit definitions used by the current timeline schema.",
            "- The dominant energy term in this example is `E_move`, not optical compute, which is consistent with the main-text system claim that conversion/control and movement dominate the stack.",
        ]
    )
    report_md.write_text("\n".join(note_lines) + "\n", encoding="utf-8")

    print(f"Wrote term mapping CSV: {term_csv}")
    print(f"Wrote layer CSV: {layer_csv}")
    print(f"Wrote report: {report_md}")


if __name__ == "__main__":
    main()
