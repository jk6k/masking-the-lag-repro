#!/usr/bin/env python3
"""Generate the focused HOPS v3 batch-2 sweep configs and manifest.

This script is intentionally scoped to the 20260420 batch-2 generation only.
"""

from __future__ import annotations

import csv
from copy import deepcopy
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
TEMPLATE_PATH = REPO_ROOT / "configs" / "phase1_true_sc_e0_e6_canonical_template_20260418.yaml"
OUT_DIR = REPO_ROOT / "experiments" / "results" / "generated_configs" / "20260420_hops_v3_sweep_batch2"
MANIFEST_PATH = OUT_DIR / "manifest.csv"


CONFIG_SPECS = [
    {
        "config_id": "hopsv3_b4p3e4_4x4_spill",
        "purpose": "4x4 spill-lane baseline near the batch-1 winner",
        "run_id": "20260420_hopsv3_b4p3e4_4x4_spill",
        "file_name": "hopsv3_b4p3e4_4x4_spill.yaml",
        "buffer_depth": 4,
        "prefetch_credits": 3,
        "execute_credits": 4,
        "exception_lane_policy": "spill",
        "tile_rows": 4,
        "tile_cols": 4,
        "control_issue_width": 5,
    },
    {
        "config_id": "hopsv3_b5p4e5_4x4_spill",
        "purpose": "4x4 spill-lane stronger credit/buffer point",
        "run_id": "20260420_hopsv3_b5p4e5_4x4_spill",
        "file_name": "hopsv3_b5p4e5_4x4_spill.yaml",
        "buffer_depth": 5,
        "prefetch_credits": 4,
        "execute_credits": 5,
        "exception_lane_policy": "spill",
        "tile_rows": 4,
        "tile_cols": 4,
        "control_issue_width": 5,
    },
    {
        "config_id": "hopsv3_b5p4e5_4x4_defer",
        "purpose": "4x4 defer-lane comparator at the stronger point",
        "run_id": "20260420_hopsv3_b5p4e5_4x4_defer",
        "file_name": "hopsv3_b5p4e5_4x4_defer.yaml",
        "buffer_depth": 5,
        "prefetch_credits": 4,
        "execute_credits": 5,
        "exception_lane_policy": "defer",
        "tile_rows": 4,
        "tile_cols": 4,
        "control_issue_width": 5,
    },
    {
        "config_id": "hopsv3_b4p3e4_8x8_spill",
        "purpose": "8x8 spill-lane baseline near the batch-1 winner",
        "run_id": "20260420_hopsv3_b4p3e4_8x8_spill",
        "file_name": "hopsv3_b4p3e4_8x8_spill.yaml",
        "buffer_depth": 4,
        "prefetch_credits": 3,
        "execute_credits": 4,
        "exception_lane_policy": "spill",
        "tile_rows": 8,
        "tile_cols": 8,
        "control_issue_width": 5,
    },
    {
        "config_id": "hopsv3_b5p4e5_8x8_spill",
        "purpose": "8x8 spill-lane stronger credit/buffer point",
        "run_id": "20260420_hopsv3_b5p4e5_8x8_spill",
        "file_name": "hopsv3_b5p4e5_8x8_spill.yaml",
        "buffer_depth": 5,
        "prefetch_credits": 4,
        "execute_credits": 5,
        "exception_lane_policy": "spill",
        "tile_rows": 8,
        "tile_cols": 8,
        "control_issue_width": 5,
    },
    {
        "config_id": "hopsv3_b5p4e5_8x8_defer",
        "purpose": "8x8 defer-lane comparator at the stronger point",
        "run_id": "20260420_hopsv3_b5p4e5_8x8_defer",
        "file_name": "hopsv3_b5p4e5_8x8_defer.yaml",
        "buffer_depth": 5,
        "prefetch_credits": 4,
        "execute_credits": 5,
        "exception_lane_policy": "defer",
        "tile_rows": 8,
        "tile_cols": 8,
        "control_issue_width": 5,
    },
]


def set_path(obj: dict, path: list[str], value) -> None:
    cursor = obj
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = value


def main() -> None:
    with TEMPLATE_PATH.open("r", encoding="utf-8") as f:
        template = yaml.safe_load(f)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest_rows = []

    for spec in CONFIG_SPECS:
        cfg = deepcopy(template)

        set_path(cfg, ["run", "run_id"], spec["run_id"])
        set_path(cfg, ["run", "experiment_id"], "E2")
        set_path(cfg, ["run", "notes"], "HOPS v3 batch-2; focused interaction sweep centered on spill winners")
        set_path(cfg, ["switches", "meso"], False)
        set_path(cfg, ["switches", "flow"], True)
        set_path(cfg, ["switches", "det"], False)
        set_path(cfg, ["switches", "sparse"], False)
        set_path(cfg, ["switches", "phy"], False)

        set_path(cfg, ["accuracy", "require_context_match"], False)
        set_path(cfg, ["accuracy", "measurement_contract", "note"], "batch2_proxy_hops_v3_sweep; do_not_treat_as_measured_promotion_surface")

        set_path(cfg, ["flow", "enabled"], True)
        set_path(cfg, ["flow", "evidence_type"], "heuristic_proxy")
        set_path(cfg, ["flow", "calibration_source"], "")
        set_path(cfg, ["flow", "latency_scale"], 0.85)
        set_path(cfg, ["flow", "scheduler_mode"], "elastic_residency_v3")
        set_path(cfg, ["flow", "reuse_policy"], "operand_factored")
        set_path(cfg, ["flow", "prefetch_window"], 2)
        set_path(cfg, ["flow", "control_group_size"], 4)
        set_path(cfg, ["flow", "tile_rows"], spec["tile_rows"])
        set_path(cfg, ["flow", "tile_cols"], spec["tile_cols"])
        set_path(cfg, ["flow", "prefetch_credits"], spec["prefetch_credits"])
        set_path(cfg, ["flow", "execute_credits"], spec["execute_credits"])
        set_path(cfg, ["flow", "control_issue_width"], spec["control_issue_width"])
        set_path(cfg, ["flow", "admission_policy"], "reuse_first")
        set_path(cfg, ["flow", "eviction_policy"], "pinned_operand")
        set_path(cfg, ["flow", "service_policy"], "critical_path_first")
        set_path(cfg, ["flow", "reuse_residency_budget"], 5)
        set_path(cfg, ["flow", "broadcast_stability_window"], 2)
        set_path(cfg, ["flow", "prefetch_distance"], 2)
        set_path(cfg, ["flow", "exception_lane_policy"], spec["exception_lane_policy"])
        set_path(cfg, ["flow", "buffer_depth"], spec["buffer_depth"])

        out_path = OUT_DIR / spec["file_name"]
        with out_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False, width=120)

        manifest_rows.append(
            {
                "config_id": spec["config_id"],
                "purpose": spec["purpose"],
                "run_id": spec["run_id"],
                "file_path": str(out_path),
                "buffer_depth": spec["buffer_depth"],
                "prefetch_credits": spec["prefetch_credits"],
                "execute_credits": spec["execute_credits"],
                "exception_lane_policy": spec["exception_lane_policy"],
                "tile_rows": spec["tile_rows"],
                "tile_cols": spec["tile_cols"],
                "control_issue_width": spec["control_issue_width"],
            }
        )

    with MANIFEST_PATH.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "config_id",
                "purpose",
                "run_id",
                "file_path",
                "buffer_depth",
                "prefetch_credits",
                "execute_credits",
                "exception_lane_policy",
                "tile_rows",
                "tile_cols",
                "control_issue_width",
            ],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)


if __name__ == "__main__":
    main()
