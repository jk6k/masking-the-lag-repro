#!/usr/bin/env python3
"""Validate the integrated FULLER_REENTRY_V1 slice package."""

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


DEFAULT_TEMPLATE = ROOT / "configs" / "fuller_det_sparse_reentry_slice_template_20260331.yaml"
DEFAULT_MODEL_SUMMARY = ROOT / "experiments" / "results" / "report_data" / "fuller_reentry_model_summary_20260331.csv"
DEFAULT_ASTRA_SUMMARY = ROOT / "experiments" / "results" / "report_data" / "fuller_reentry_astra_substrate_summary_20260331.csv"


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


def evaluate_reentry_slice(
    template: dict[str, Any],
    model_row: dict[str, str],
    astra_row: dict[str, str],
    *,
    model_summary_path: Path,
    astra_summary_path: Path,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    checks: list[dict[str, str]] = []

    run_cfg = template.get("run") or {}
    switches = template.get("switches") or {}
    realism = template.get("realism") or {}
    sparse_cfg = template.get("sparse") or {}
    sc_det = template.get("sc_det") or {}
    early_stop = sc_det.get("early_stop") or {}
    reentry_lane = template.get("reentry_lane") or {}

    _record(
        checks,
        "run_surface",
        "pass"
        if str(run_cfg.get("experiment_id") or "") == "FULLER_REENTRY_V1"
        and str(run_cfg.get("device") or "") == "mps"
        else "fail",
        repr({"experiment_id": run_cfg.get("experiment_id"), "device": run_cfg.get("device")}),
    )
    _record(
        checks,
        "switches.full_stack",
        "pass" if switches == {"meso": True, "flow": True, "det": True, "sparse": True, "phy": True} else "fail",
        repr(switches),
    )
    _record(
        checks,
        "det_and_sparse_settings",
        "pass"
        if early_stop.get("enabled") is True
        and int(float(early_stop.get("k_global") or 0)) == 64
        and abs(float(sparse_cfg.get("active_fraction") or 0.0) - 0.75) <= 1e-9
        else "fail",
        repr({"early_stop": early_stop, "sparse": sparse_cfg}),
    )
    _record(
        checks,
        "realism_boundary",
        "pass"
        if str(realism.get("target_class") or "") == "realistic_accelerator_proxy"
        and str(realism.get("device_comparison_scope") or "") == "contextual_only"
        and realism.get("benchmark_equivalence") is False
        else "fail",
        repr(realism),
    )
    for key in ("det_candidate_yaml", "sparse_operating_point_yaml", "device_claim_note_md", "promotion_note_md"):
        path = assert_main_project_path(resolve_repo_path(str(reentry_lane.get(key) or "")), arg_name=f"reentry_lane.{key}")
        _record(checks, f"reentry_lane.{key}", "pass" if path.exists() else "fail", str(path))

    required_fields = {
        "run_id",
        "experiment_id",
        "core_latency_ms",
        "latency_ms",
        "system_latency_lower_ms",
        "system_latency_upper_ms",
        "core_energy_j",
        "energy_j",
        "system_energy_lower_j",
        "system_energy_upper_j",
        "integrated_hidden_system_cost_j",
        "integrated_hidden_system_latency_ms",
        "realism_class",
        "device_comparison_scope",
        "benchmark_equivalence",
        "comparison_boundary",
        "accuracy_source_csv",
        "acc_top1",
        "acc_drop_pp",
        "pass_delta",
        "det_net_gain_j",
        "pass_det_net_gain",
        "duty_cycle_avg",
    }
    missing = sorted(field for field in required_fields if field not in model_row)
    _record(checks, "model_summary.required_fields", "pass" if not missing else "fail", repr(missing))
    _record(
        checks,
        "model_summary.identity",
        "pass"
        if model_row.get("experiment_id") == "FULLER_REENTRY_V1"
        and model_row.get("realism_class") == "realistic_accelerator_proxy"
        and model_row.get("device_comparison_scope") == "contextual_only"
        and str(model_row.get("benchmark_equivalence") or "").lower() == "false"
        else "fail",
        repr({
            "experiment_id": model_row.get("experiment_id"),
            "realism_class": model_row.get("realism_class"),
            "device_comparison_scope": model_row.get("device_comparison_scope"),
            "benchmark_equivalence": model_row.get("benchmark_equivalence"),
        }),
    )
    _record(
        checks,
        "model_summary.pass_flags",
        "pass"
        if str(model_row.get("pass_delta") or "").lower() == "true"
        and str(model_row.get("pass_det_net_gain") or "").lower() == "true"
        else "fail",
        repr({"pass_delta": model_row.get("pass_delta"), "pass_det_net_gain": model_row.get("pass_det_net_gain")}),
    )
    _record(
        checks,
        "model_summary.det_and_sparse_metrics",
        "pass"
        if float(model_row.get("det_net_gain_j") or 0.0) > 0.0
        and 0.70 <= float(model_row.get("duty_cycle_avg") or 0.0) <= 0.80
        else "fail",
        repr({"det_net_gain_j": model_row.get("det_net_gain_j"), "duty_cycle_avg": model_row.get("duty_cycle_avg")}),
    )
    model_run_path = assert_main_project_path(resolve_repo_path(model_row.get("source_run_path") or model_row.get("summary_source_path") or ""), arg_name="model_summary.source_run_path")
    _record(checks, "model_summary.source_run_path", "pass" if model_run_path.exists() else "fail", str(model_run_path))
    astra_required = {"run_id", "experiment_id", "latency_ms", "energy_j"}
    astra_missing = sorted(field for field in astra_required if field not in astra_row)
    _record(checks, "astra_summary.required_fields", "pass" if not astra_missing else "fail", repr(astra_missing))
    _record(
        checks,
        "astra_summary.identity",
        "pass" if "ASTRA" in str(astra_row.get("experiment_id") or "").upper() else "fail",
        str(astra_row.get("experiment_id") or ""),
    )
    _record(checks, "model_summary_path", "pass" if model_summary_path.exists() else "fail", str(model_summary_path))
    _record(checks, "astra_summary_path", "pass" if astra_summary_path.exists() else "fail", str(astra_summary_path))

    overall_ok = all(row["status"] == "pass" for row in checks)
    return checks, {"overall_ok": overall_ok, "model_summary_path": str(model_summary_path), "astra_summary_path": str(astra_summary_path)}


def _write_report(out_path: Path, checks: list[dict[str, str]], summary: dict[str, Any]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Fuller Reentry Slice Check",
        "",
        f"- overall_ok: `{summary['overall_ok']}`",
        f"- model_summary_path: `{summary['model_summary_path']}`",
        f"- astra_summary_path: `{summary['astra_summary_path']}`",
        "",
        "## Checks",
        "",
    ]
    for row in checks:
        lines.append(f"- `{row['status']}` `{row['check_id']}`: {row['detail']}")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the FULLER_REENTRY_V1 slice package.")
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--model_summary_csv", type=Path, default=DEFAULT_MODEL_SUMMARY)
    parser.add_argument("--astra_summary_csv", type=Path, default=DEFAULT_ASTRA_SUMMARY)
    parser.add_argument("--out_dir", type=Path, default=ROOT / "experiments" / "results" / "report_data")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    template = _load_yaml(args.template.resolve())
    model_row = _load_single_csv_row(args.model_summary_csv.resolve())
    astra_row = _load_single_csv_row(args.astra_summary_csv.resolve())
    checks, summary = evaluate_reentry_slice(
        template,
        model_row,
        astra_row,
        model_summary_path=args.model_summary_csv.resolve(),
        astra_summary_path=args.astra_summary_csv.resolve(),
    )
    out_path = args.out_dir.resolve() / "check_fuller_reentry_slice_20260331.md"
    _write_report(out_path, checks, summary)
    if not summary["overall_ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
