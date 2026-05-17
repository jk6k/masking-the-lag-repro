#!/usr/bin/env python3
"""Build the retained calibration profile that upgrades fuller to realistic proxy."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.exp_common.realism_proxy_calibration import (  # noqa: E402
    build_realistic_proxy_profile,
    write_realistic_proxy_profile,
)


DEFAULT_OUT_DIR = (
    ROOT / "experiments" / "results" / "calibration_audit" / "20260329_fuller_realistic_proxy_v1"
)
DEFAULT_FLOW_BASELINE = ROOT / "experiments" / "results" / "runs" / "20260228_opt_sync_core_e0" / "master_metrics.csv"
DEFAULT_FLOW_RUN = ROOT / "experiments" / "results" / "runs" / "20260228_opt_sync_core_e2" / "master_metrics.csv"
DEFAULT_MESO_SWEEP = (
    ROOT / "experiments" / "results" / "quick_reports" / "20260328_mps_full_eval_freeze" / "quickscan_e1_fanout_sweep.csv"
)
DEFAULT_PHY_SWEEP = (
    ROOT / "experiments" / "results" / "quick_reports" / "20260328_mps_full_eval_freeze" / "quickscan_e5_phy_n_sweep.csv"
)
DEFAULT_ASTRA_SUMMARY = (
    ROOT / "experiments" / "results" / "report_data" / "fuller_astra_substrate_model_summary_20260319_fullerexp_v1.csv"
)
DEFAULT_FULLER_SUMMARY = (
    ROOT / "experiments" / "results" / "report_data" / "fuller_slice_model_summary_20260319_fullerexp_v1.csv"
)
DEFAULT_PHASE6_SCOPE = (
    ROOT / "experiments" / "results" / "calibration_audit" / "20260316_phase6_scope_v1" / "phase6_scope_report.md"
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build fuller realistic-proxy calibration profile.")
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--model", default="mobilevit_s")
    parser.add_argument("--buffer_depth", type=int, default=2)
    parser.add_argument("--flow_baseline_csv", type=Path, default=DEFAULT_FLOW_BASELINE)
    parser.add_argument("--flow_run_csv", type=Path, default=DEFAULT_FLOW_RUN)
    parser.add_argument("--meso_sweep_csv", type=Path, default=DEFAULT_MESO_SWEEP)
    parser.add_argument("--phy_sweep_csv", type=Path, default=DEFAULT_PHY_SWEEP)
    parser.add_argument("--astra_summary_csv", type=Path, default=DEFAULT_ASTRA_SUMMARY)
    parser.add_argument("--fuller_summary_csv", type=Path, default=DEFAULT_FULLER_SUMMARY)
    parser.add_argument("--phase6_scope_md", type=Path, default=DEFAULT_PHASE6_SCOPE)
    args = parser.parse_args()

    profile = build_realistic_proxy_profile(
        model=args.model,
        buffer_depth=args.buffer_depth,
        baseline_flow_run_csv=args.flow_baseline_csv,
        flow_run_csv=args.flow_run_csv,
        meso_fanout_sweep_csv=args.meso_sweep_csv,
        phy_n_sweep_csv=args.phy_sweep_csv,
        astra_summary_csv=args.astra_summary_csv,
        fuller_summary_csv=args.fuller_summary_csv,
        phase6_scope_report_md=args.phase6_scope_md,
    )
    out_yaml = args.out_dir / "fuller_realistic_proxy_calibration_profile.yaml"
    out_report_md = args.out_dir / "fuller_realistic_proxy_calibration_report.md"
    write_realistic_proxy_profile(
        profile=profile,
        out_yaml=out_yaml,
        out_report_md=out_report_md,
    )
    print(f"[fuller-realistic-proxy] wrote profile={out_yaml}")
    print(f"[fuller-realistic-proxy] wrote report={out_report_md}")


if __name__ == "__main__":
    main()
