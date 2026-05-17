"""Run HPAT estimation across multiple configs (range or single-config)."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

ROOT_DIR = SCRIPT_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from exp_common.io_utils import backup_existing_file  # noqa: E402
from hpat_model import summarize_ops  # noqa: E402


def load_ops(path: str | Path) -> tuple[str, list[dict]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        ops = data
        model = Path(path).stem
    elif isinstance(data, dict):
        ops = data.get("ops", [])
        model = data.get("model") or Path(path).stem
    else:
        raise ValueError(f"Unsupported ops format: {path}")
    return model, ops


def collect_ops_paths(ops_paths: list[str], ops_dir: str | None) -> list[str]:
    paths = list(ops_paths)
    if ops_dir:
        ops_dir = Path(ops_dir)
        if not ops_dir.is_absolute():
            cwd_path = Path.cwd() / ops_dir
            if cwd_path.exists():
                ops_dir = cwd_path
            else:
                ops_dir = SCRIPT_DIR / ops_dir
        paths.extend(str(p) for p in ops_dir.glob("*.json"))
    if not paths:
        raise SystemExit("No ops files provided. Use --ops or --ops_dir.")
    return paths


def load_config(path: str | Path) -> tuple[dict, Path]:
    path = Path(path)
    if not path.is_absolute():
        cwd_path = Path.cwd() / path
        if cwd_path.exists():
            path = cwd_path
        else:
            path = SCRIPT_DIR / path
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    return config, path


def write_summary_csv(path: Path, rows: list[dict]) -> None:
    fields = [
        "model",
        "config",
        "ops_path",
        "total_latency_ms",
        "total_energy_mj",
        "total_power_w",
        "energy_mj_load_x",
        "energy_mj_load_y",
        "energy_mj_detect",
        "energy_mj_laser",
        "energy_mj_mem",
        "energy_mj_static",
        "energy_mj_elementwise",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fields})


def write_range_csv(path: Path, rows: list[dict]) -> None:
    fields = [
        "model",
        "latency_ms_min",
        "latency_ms_max",
        "energy_mj_min",
        "energy_mj_max",
        "power_w_min",
        "power_w_max",
        "configs",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fields})


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run HPAT end-to-end sensitivity analysis across multiple configs."
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
        action="append",
        default=[],
        help="HPAT config YAML path (repeatable).",
    )
    parser.add_argument(
        "--out",
        default="../results/hpat_estimates_range.csv",
        help="Output CSV for min/max range summary.",
    )
    parser.add_argument(
        "--save_per_config",
        action="store_true",
        help="Also save per-config summary CSVs.",
    )
    args = parser.parse_args()

    if not args.config:
        raise SystemExit("At least one --config is required.")

    ops_paths = collect_ops_paths(args.ops, args.ops_dir)

    config_entries = []
    for cfg in args.config:
        cfg_data, cfg_path = load_config(cfg)
        config_entries.append((cfg_data, cfg_path))

    # Aggregate per-config summaries first, then compute min/max ranges per model.
    per_config_rows = []
    by_model = {}

    for cfg_data, cfg_path in config_entries:
        cfg_name = cfg_path.stem
        for ops_path in ops_paths:
            model, ops = load_ops(ops_path)
            _, summary = summarize_ops(ops, cfg_data)
            entry = {
                "model": model,
                "config": cfg_name,
                "ops_path": ops_path,
                "total_latency_ms": summary["total_latency_ms"],
                "total_energy_mj": summary["total_energy_mj"],
                "total_power_w": summary["total_power_w"],
                "energy_mj_load_x": summary.get("energy_mj_load_x"),
                "energy_mj_load_y": summary.get("energy_mj_load_y"),
                "energy_mj_detect": summary.get("energy_mj_detect"),
                "energy_mj_laser": summary.get("energy_mj_laser"),
                "energy_mj_mem": summary.get("energy_mj_mem"),
                "energy_mj_static": summary.get("energy_mj_static"),
                "energy_mj_elementwise": summary.get("energy_mj_elementwise"),
            }
            per_config_rows.append(entry)
            by_model.setdefault(model, []).append(entry)

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = SCRIPT_DIR / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    backup = backup_existing_file(out_path)
    if backup:
        print(f"Existing range summary moved to {backup}")

    range_rows = []
    for model, entries in sorted(by_model.items()):
        latencies = [e["total_latency_ms"] for e in entries if e["total_latency_ms"] is not None]
        energies = [e["total_energy_mj"] for e in entries if e["total_energy_mj"] is not None]
        powers = [e["total_power_w"] for e in entries if e["total_power_w"] is not None]
        range_rows.append(
            {
                "model": model,
                "latency_ms_min": min(latencies) if latencies else None,
                "latency_ms_max": max(latencies) if latencies else None,
                "energy_mj_min": min(energies) if energies else None,
                "energy_mj_max": max(energies) if energies else None,
                "power_w_min": min(powers) if powers else None,
                "power_w_max": max(powers) if powers else None,
                "configs": ",".join(sorted({e["config"] for e in entries})),
            }
        )

    write_range_csv(out_path, range_rows)
    print(f"Saved range summary to {out_path}")

    if args.save_per_config:
        per_config_path = out_path.with_name(out_path.stem + "_per_config.csv")
        backup = backup_existing_file(per_config_path)
        if backup:
            print(f"Existing per-config summary moved to {backup}")
        write_summary_csv(per_config_path, per_config_rows)
        print(f"Saved per-config summary to {per_config_path}")


if __name__ == "__main__":
    main()
