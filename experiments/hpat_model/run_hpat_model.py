"""Run HPAT parameterized energy/latency estimation for one config."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import yaml

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from exp_common.io_utils import backup_existing_file  # noqa: E402
from hpat_model import summarize_ops


def load_ops(path: str | Path) -> tuple[str, list[dict], dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        # Legacy format: a raw list of ops without metadata.
        ops = data
        model = Path(path).stem
        meta = {}
    elif isinstance(data, dict):
        # New format: {"model": ..., "ops": [...], ...}
        ops = data.get("ops", [])
        model = data.get("model") or Path(path).stem
        meta = {k: v for k, v in data.items() if k != "ops"}
    else:
        raise ValueError(f"Unsupported ops format: {path}")
    return model, ops, meta


def write_ops_csv(path: Path, op_results: list[dict]) -> None:
    fields = [
        "name",
        "type",
        "m",
        "d",
        "n",
        "elements",
        "tiles",
        "latency_ms",
        "energy_mj",
        "energy_mj_load_x",
        "energy_mj_load_y",
        "energy_mj_detect",
        "energy_mj_oe",
        "energy_mj_adc_pca",
        "energy_mj_laser",
        "energy_mj_mem",
        "energy_mj_static",
        "energy_mj_elementwise",
        "power_w",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in op_results:
            writer.writerow({k: row.get(k) for k in fields})


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run parameterized HPAT energy/latency estimation."
    )
    parser.add_argument(
        "--ops",
        action="append",
        default=[],
        help="Path to ops JSON (repeatable).",
    )
    parser.add_argument(
        "--ops_dir",
        default=None,
        help="Directory containing ops_*.json files.",
    )
    parser.add_argument(
        "--config",
        default="hpat_config.yaml",
        help="HPAT config YAML path.",
    )
    parser.add_argument(
        "--out",
        default="../results/hpat_estimates.csv",
        help="Output CSV for model-level summaries.",
    )
    parser.add_argument(
        "--out_ops_dir",
        default="../results/hpat_ops",
        help="Directory for per-op CSV outputs.",
    )
    args = parser.parse_args()

    ops_paths = list(args.ops)
    if args.ops_dir:
        ops_dir = Path(args.ops_dir)
        if not ops_dir.is_absolute():
            ops_dir = Path(__file__).resolve().parent / ops_dir
        ops_paths.extend(str(p) for p in ops_dir.glob("*.json"))

    if not ops_paths:
        raise SystemExit("No ops files provided. Use --ops or --ops_dir.")

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = Path(__file__).resolve().parent / config_path
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = Path(__file__).resolve().parent / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    out_ops_dir = Path(args.out_ops_dir)
    if not out_ops_dir.is_absolute():
        out_ops_dir = Path(__file__).resolve().parent / out_ops_dir
    out_ops_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for ops_path in ops_paths:
        model, ops, meta = load_ops(ops_path)
        op_results, summary = summarize_ops(ops, config)
        summary_rows.append(
            {
                "model": model,
                "ops_path": ops_path,
                "total_latency_ms": summary["total_latency_ms"],
                "total_energy_mj": summary["total_energy_mj"],
                "total_power_w": summary["total_power_w"],
                "energy_mj_load_x": summary.get("energy_mj_load_x"),
                "energy_mj_load_y": summary.get("energy_mj_load_y"),
                "energy_mj_detect": summary.get("energy_mj_detect"),
                "energy_mj_oe": summary.get("energy_mj_oe"),
                "energy_mj_adc_pca": summary.get("energy_mj_adc_pca"),
                "energy_mj_laser": summary.get("energy_mj_laser"),
                "energy_mj_mem": summary.get("energy_mj_mem"),
                "energy_mj_static": summary.get("energy_mj_static"),
                "energy_mj_elementwise": summary.get("energy_mj_elementwise"),
                "energy_j_total": (
                    summary.get("total_energy_mj", 0.0) / 1e3
                    if summary.get("total_energy_mj") is not None
                    else None
                ),
                "energy_j_conversion_control": (
                    (
                        (summary.get("energy_mj_load_x") or 0.0)
                        + (summary.get("energy_mj_load_y") or 0.0)
                    )
                    / 1e3
                ),
                "energy_j_memory_move": (
                    (summary.get("energy_mj_mem") or 0.0) / 1e3
                ),
                "energy_j_oe": (summary.get("energy_mj_oe") or 0.0) / 1e3,
                "energy_j_adc_pca": (
                    (summary.get("energy_mj_adc_pca") or 0.0) / 1e3
                ),
                "energy_j_laser_optical": (
                    (summary.get("energy_mj_laser") or 0.0) / 1e3
                ),
                "energy_j_other_static": (
                    (
                        (summary.get("energy_mj_static") or 0.0)
                        + (summary.get("energy_mj_elementwise") or 0.0)
                    )
                    / 1e3
                ),
            }
        )

        ops_csv = out_ops_dir / f"{model}_ops.csv"
        backup = backup_existing_file(ops_csv)
        if backup:
            print(f"Existing ops CSV moved to {backup}")
        write_ops_csv(ops_csv, op_results)

    fields = [
        "model",
        "ops_path",
        "total_latency_ms",
        "total_energy_mj",
        "total_power_w",
        "energy_mj_load_x",
        "energy_mj_load_y",
        "energy_mj_detect",
        "energy_mj_oe",
        "energy_mj_adc_pca",
        "energy_mj_laser",
        "energy_mj_mem",
        "energy_mj_static",
        "energy_mj_elementwise",
        "energy_j_total",
        "energy_j_conversion_control",
        "energy_j_memory_move",
        "energy_j_oe",
        "energy_j_adc_pca",
        "energy_j_laser_optical",
        "energy_j_other_static",
    ]
    backup = backup_existing_file(out_path)
    if backup:
        print(f"Existing summary moved to {backup}")
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow({k: row.get(k) for k in fields})

    print(f"Saved summary to {out_path}")
    print(f"Saved per-op CSVs to {out_ops_dir}")


if __name__ == "__main__":
    main()
