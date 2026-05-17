#!/usr/bin/env python3
"""Stage FULLER phase1 config snapshots and dry-run accuracy launch manifests."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    from . import prepare_true_sc_e0_e6_accuracy_launch as legacy_launch
except ImportError:
    import prepare_true_sc_e0_e6_accuracy_launch as legacy_launch  # type: ignore


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BUNDLE = ROOT / "configs" / "fuller_phase1_runtime_smoke_current_bundle_20260421.yaml"
DEFAULT_LANE_ORDER = ("FULLER", "DET", "PHY", "SPARSE", "HOPS", "MESO")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare dry-run accuracy launch manifests for FULLER phase1 lanes."
    )
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--python_bin", default=sys.executable)
    parser.add_argument(
        "--lane_order",
        default=",".join(DEFAULT_LANE_ORDER),
        help="Comma-separated launch order. Defaults to FULLER,DET,PHY,SPARSE,HOPS,MESO.",
    )
    parser.add_argument(
        "--evidence_tier",
        choices=[legacy_launch.RUNTIME_SMOKE_TIER, legacy_launch.ANALYSIS_GRADE_TIER],
        default=legacy_launch.RUNTIME_SMOKE_TIER,
        help="Launch evidence tier. Defaults to runtime_smoke.",
    )
    parser.add_argument(
        "--write-scope",
        choices=(legacy_launch.WRITE_SCOPE_FULL, legacy_launch.WRITE_SCOPE_REPORT_DATA_ONLY),
        default=legacy_launch.WRITE_SCOPE_FULL,
        help="Write full outputs or only owned report-data artifacts.",
    )
    parser.add_argument("--inspect-existing-launch-prep", action="store_true")
    args = parser.parse_args()

    bundle_path = args.bundle if args.bundle.is_absolute() else ROOT / args.bundle
    lane_order = tuple(item.strip().upper() for item in str(args.lane_order).split(",") if item.strip())
    if args.inspect_existing_launch_prep:
        payload = legacy_launch.inspect_existing_launch_prep(
            bundle_path=bundle_path,
            lane_order=lane_order,
            evidence_tier=str(args.evidence_tier),
            write_scope=str(args.write_scope),
        )
    else:
        payload = legacy_launch.build_launch_artifacts(
            bundle_path=bundle_path,
            python_bin=str(args.python_bin),
            lane_order=lane_order,
            evidence_tier=str(args.evidence_tier),
            write_scope=str(args.write_scope),
        )
    print(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
