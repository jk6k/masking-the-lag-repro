#!/usr/bin/env python3
"""Validate the governed SPARSE operating point reentry package."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from repo_python_bootstrap import maybe_reexec_for_module
except ImportError:
    def maybe_reexec_for_module(_module: str, *, anchor: Path | None = None) -> None:
        return None

maybe_reexec_for_module("yaml", anchor=Path(__file__))

import yaml

try:
    from .path_policy import assert_main_project_path, resolve_repo_path
except ImportError:
    from path_policy import assert_main_project_path, resolve_repo_path  # type: ignore


DEFAULT_CONTRACT = ROOT / "configs" / "sparse_reentry_operating_point_contract_20260331.yaml"
DEFAULT_SUMMARY = ROOT / "experiments" / "results" / "report_data" / "sparse_reentry_operating_point_20260331.csv"


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise SystemExit(f"Expected YAML mapping in {path}")
    return payload


def _load_single_csv_row(path: Path) -> dict[str, str]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    if len(rows) != 1:
        raise SystemExit(f"Expected exactly one row in {path}, found {len(rows)}")
    return rows[0]


def _record(rows: list[dict[str, str]], check_id: str, status: str, detail: str) -> None:
    rows.append({"check_id": check_id, "status": status, "detail": detail})


def evaluate_operating_point(contract: dict[str, Any], summary_row: dict[str, str], *, summary_path: Path) -> tuple[list[dict[str, str]], dict[str, Any]]:
    checks: list[dict[str, str]] = []

    op = contract.get("operating_point") or {}
    run_cfg = contract.get("run") or {}
    switches = contract.get("switches") or {}
    sparse_cfg = contract.get("sparse") or {}
    accuracy = contract.get("accuracy") or {}

    _record(
        checks,
        "operating_point_label",
        "pass" if str(op.get("operating_point_label") or "") == "sparse_active_fraction_075_mobilevit_s_v1" else "fail",
        str(op.get("operating_point_label") or ""),
    )
    _record(
        checks,
        "control_mode",
        "pass" if str(op.get("control_mode") or "") == "tau_threshold_primary" else "fail",
        str(op.get("control_mode") or ""),
    )
    _record(
        checks,
        "run_surface",
        "pass"
        if str(run_cfg.get("experiment_id") or "") == "SPARSE_REENTRY_V1"
        and str(run_cfg.get("device") or "") == "mps"
        else "fail",
        repr({"experiment_id": run_cfg.get("experiment_id"), "device": run_cfg.get("device")}),
    )
    _record(
        checks,
        "switches.sparse_only",
        "pass" if switches == {"meso": False, "flow": False, "det": False, "sparse": True, "phy": False} else "fail",
        repr(switches),
    )
    _record(
        checks,
        "sparse.definition",
        "pass"
        if sparse_cfg.get("enabled") is True
        and abs(float(sparse_cfg.get("tau_global") or 0.0) - 0.25) <= 1e-9
        and bool(sparse_cfg.get("use_tau_for_gating")) is True
        and abs(float(sparse_cfg.get("active_fraction") or 0.0) - 0.75) <= 1e-9
        else "fail",
        repr(sparse_cfg),
    )
    accuracy_path = assert_main_project_path(resolve_repo_path(str(accuracy.get("source_csv") or "")), arg_name="accuracy.source_csv")
    _record(checks, "accuracy.source_csv", "pass" if accuracy_path.exists() else "fail", str(accuracy_path))

    required_fields = {
        "operating_point_label",
        "run_id",
        "experiment_id",
        "model",
        "latency_ms",
        "energy_j",
        "duty_cycle_avg",
        "sparse_active_fraction",
        "acc_top1",
        "acc_drop_pp",
        "accuracy_evidence",
        "projection_only",
        "promotion_safe",
        "performance_run_path",
        "per_layer_phy_csv",
    }
    missing = sorted(field for field in required_fields if field not in summary_row)
    _record(checks, "summary.required_fields", "pass" if not missing else "fail", repr(missing))
    _record(
        checks,
        "summary.alignment",
        "pass"
        if summary_row.get("operating_point_label") == str(op.get("operating_point_label") or "")
        and summary_row.get("experiment_id") == "SPARSE_REENTRY_V1"
        else "fail",
        repr({"operating_point_label": summary_row.get("operating_point_label"), "experiment_id": summary_row.get("experiment_id")}),
    )
    duty = float(summary_row.get("duty_cycle_avg") or 0.0)
    active = float(summary_row.get("sparse_active_fraction") or 0.0)
    acc_drop = float(summary_row.get("acc_drop_pp") or 999.0)
    _record(checks, "summary.duty_cycle_avg", "pass" if 0.70 <= duty <= 0.80 else "fail", str(duty))
    _record(checks, "summary.sparse_active_fraction", "pass" if abs(active - 0.75) <= 1e-9 else "fail", str(active))
    _record(checks, "summary.acc_drop_pp", "pass" if acc_drop <= 1.0 else "fail", str(acc_drop))
    _record(
        checks,
        "summary.accuracy_evidence",
        "pass" if "measured" in str(summary_row.get("accuracy_evidence") or "").lower() else "fail",
        str(summary_row.get("accuracy_evidence") or ""),
    )
    _record(
        checks,
        "summary.promotion_surface",
        "pass"
        if str(summary_row.get("projection_only") or "").lower() == "false"
        and str(summary_row.get("promotion_safe") or "").lower() == "true"
        else "fail",
        repr({"projection_only": summary_row.get("projection_only"), "promotion_safe": summary_row.get("promotion_safe")}),
    )
    perf_path = assert_main_project_path(resolve_repo_path(summary_row.get("performance_run_path") or ""), arg_name="summary.performance_run_path")
    phy_path = assert_main_project_path(resolve_repo_path(summary_row.get("per_layer_phy_csv") or ""), arg_name="summary.per_layer_phy_csv")
    _record(checks, "summary.performance_run_path", "pass" if perf_path.exists() else "fail", str(perf_path))
    _record(checks, "summary.per_layer_phy_csv", "pass" if phy_path.exists() else "fail", str(phy_path))

    overall_ok = all(row["status"] == "pass" for row in checks)
    return checks, {"overall_ok": overall_ok, "summary_path": str(summary_path), "operating_point_label": summary_row.get("operating_point_label", "")}


def _write_report(out_path: Path, checks: list[dict[str, str]], summary: dict[str, Any]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# SPARSE Reentry Operating Point Check",
        "",
        f"- overall_ok: `{summary['overall_ok']}`",
        f"- operating_point_label: `{summary['operating_point_label']}`",
        f"- summary_path: `{summary['summary_path']}`",
        "",
        "## Checks",
        "",
    ]
    for row in checks:
        lines.append(f"- `{row['status']}` `{row['check_id']}`: {row['detail']}")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the SPARSE operating point reentry package.")
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--summary_csv", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--out_dir", type=Path, default=ROOT / "experiments" / "results" / "report_data")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    contract = _load_yaml(args.contract.resolve())
    summary_row = _load_single_csv_row(args.summary_csv.resolve())
    checks, summary = evaluate_operating_point(contract, summary_row, summary_path=args.summary_csv.resolve())
    out_path = args.out_dir.resolve() / "check_sparse_reentry_operating_point_20260331.md"
    _write_report(out_path, checks, summary)
    if not summary["overall_ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
