#!/usr/bin/env python3
"""Generate the focused HOPS v3 reintegration replay configs for 2026-04-20.

This overlays the winning HOPS v3 flow settings onto the retained fuller reentry
template without mutating historical run directories.
"""

from __future__ import annotations

import argparse
import csv
from copy import deepcopy
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
BASE_TEMPLATE_PATH = REPO_ROOT / "configs" / "fuller_det_sparse_reentry_slice_template_20260331.yaml"
DEFAULT_WINNER_CONFIG_PATH = (
    REPO_ROOT
    / "experiments"
    / "results"
    / "generated_configs"
    / "20260420_hops_v3_sweep_batch2"
    / "hopsv3_b5p4e5_4x4_spill.yaml"
)
DEFAULT_OUT_DIR = (
    REPO_ROOT
    / "experiments"
    / "results"
    / "generated_configs"
    / "20260420_hops_v3_reintegration_batch"
)
DEFAULT_RUN_PREFIX = "20260420_hopsv3_reintegration"


CONFIG_SPECS = [
    {
        "variant_id": "FLOW_MESO",
        "experiment_id": "E1",
        "file_name": "hopsv3_reintegration_flow_meso.yaml",
        "notes_suffix": "FLOW_MESO lane",
        "run_suffix": "flow_meso",
        "switches": {"meso": True, "flow": True, "det": False, "sparse": False, "phy": False},
    },
    {
        "variant_id": "FLOW_PHY",
        "experiment_id": "E5",
        "file_name": "hopsv3_reintegration_flow_phy.yaml",
        "notes_suffix": "FLOW_PHY lane",
        "run_suffix": "flow_phy",
        "switches": {"meso": False, "flow": True, "det": False, "sparse": False, "phy": True},
    },
    {
        "variant_id": "FULLER",
        "experiment_id": "FULLER_REENTRY_V1",
        "file_name": "hopsv3_reintegration_fuller.yaml",
        "notes_suffix": "FULLER reentry lane",
        "run_suffix": "fuller",
        "switches": {"meso": True, "flow": True, "det": True, "sparse": True, "phy": True},
    },
]


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _set_section_enabled(cfg: dict, section_name: str, enabled: bool) -> None:
    section = cfg.get(section_name) or {}
    section["enabled"] = enabled
    cfg[section_name] = section


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate HOPS v3 reintegration replay configs.")
    parser.add_argument("--winner-config", type=Path, default=DEFAULT_WINNER_CONFIG_PATH)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--run-prefix", default=DEFAULT_RUN_PREFIX)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    base_cfg = _load_yaml(BASE_TEMPLATE_PATH)
    winner_cfg = _load_yaml(args.winner_config)
    winner_flow = deepcopy(winner_cfg["flow"])
    out_dir = args.out_dir
    manifest_path = out_dir / "manifest.csv"

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, str | int | bool]] = []

    for spec in CONFIG_SPECS:
        cfg = deepcopy(base_cfg)
        run_id = f"{args.run_prefix}_{spec['run_suffix']}"
        cfg["run"]["run_id"] = run_id
        cfg["run"]["experiment_id"] = spec["experiment_id"]
        cfg["run"]["notes"] = (
            f"HOPS v3 reintegration replay from {args.winner_config.name} into {spec['notes_suffix']}"
        )
        cfg["switches"] = deepcopy(spec["switches"])
        cfg["flow"] = deepcopy(winner_flow)
        cfg["flow"]["enabled"] = True
        cfg["accuracy"]["context_run_id"] = run_id
        cfg["accuracy"]["require_context_match"] = False

        _set_section_enabled(cfg, "flow", spec["switches"]["flow"])
        _set_section_enabled(cfg, "meso", spec["switches"]["meso"])
        _set_section_enabled(cfg, "sparse", spec["switches"]["sparse"])
        _set_section_enabled(cfg, "phy", spec["switches"]["phy"])

        sc_det = cfg.get("sc_det") or {}
        early_stop = sc_det.get("early_stop") or {}
        early_stop["enabled"] = spec["switches"]["det"]
        sc_det["early_stop"] = early_stop
        cfg["sc_det"] = sc_det

        out_path = out_dir / spec["file_name"]
        with out_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(cfg, handle, sort_keys=False, default_flow_style=False, width=120)

        manifest_rows.append(
            {
                "variant_id": spec["variant_id"],
                "experiment_id": spec["experiment_id"],
                "run_id": run_id,
                "file_path": str(out_path),
                "meso": spec["switches"]["meso"],
                "flow": spec["switches"]["flow"],
                "det": spec["switches"]["det"],
                "sparse": spec["switches"]["sparse"],
                "phy": spec["switches"]["phy"],
                "scheduler_mode": winner_flow["scheduler_mode"],
                "buffer_depth": winner_flow["buffer_depth"],
                "prefetch_credits": winner_flow["prefetch_credits"],
                "execute_credits": winner_flow["execute_credits"],
                "control_issue_width": winner_flow["control_issue_width"],
                "tile_rows": winner_flow["tile_rows"],
                "tile_cols": winner_flow["tile_cols"],
                "exception_lane_policy": winner_flow["exception_lane_policy"],
            }
        )

    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "variant_id",
                "experiment_id",
                "run_id",
                "file_path",
                "meso",
                "flow",
                "det",
                "sparse",
                "phy",
                "scheduler_mode",
                "buffer_depth",
                "prefetch_credits",
                "execute_credits",
                "control_issue_width",
                "tile_rows",
                "tile_cols",
                "exception_lane_policy",
            ],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)


if __name__ == "__main__":
    main()
