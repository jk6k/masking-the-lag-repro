#!/usr/bin/env python3
"""Validate the bounded DET reentry candidate."""

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


DEFAULT_TEMPLATE = ROOT / "configs" / "det_reentry_candidate_template_20260331.yaml"
DEFAULT_SUMMARY = ROOT / "experiments" / "results" / "report_data" / "det_reentry_summary_20260331.csv"


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


def _to_bool_str(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _to_float_or_none(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def evaluate_candidate(template: dict[str, Any], summary_row: dict[str, str], *, summary_path: Path) -> tuple[list[dict[str, str]], dict[str, Any]]:
    checks: list[dict[str, str]] = []

    candidate = template.get("reentry_candidate") or {}
    run_cfg = template.get("run") or {}
    switches = template.get("switches") or {}
    sc_det = template.get("sc_det") or {}
    early_stop = sc_det.get("early_stop") or {}
    accuracy = template.get("accuracy") or {}

    _record(
        checks,
        "candidate_label",
        "pass" if str(candidate.get("candidate_label") or "") == "det_hybrid_prefix_quality_gate_k64_mobilevit_s" else "fail",
        str(candidate.get("candidate_label") or ""),
    )
    _record(
        checks,
        "mechanism_version",
        "pass" if str(candidate.get("mechanism_version") or "") == "DET" else "fail",
        str(candidate.get("mechanism_version") or ""),
    )
    _record(
        checks,
        "candidate_family",
        "pass" if str(candidate.get("family") or "") == "hybrid_prefix_quality_gate" else "fail",
        str(candidate.get("family") or ""),
    )
    forbidden_reuse = set(str(item) for item in (candidate.get("forbidden_reuse") or []))
    _record(
        checks,
        "forbidden_reuse",
        "pass"
        if forbidden_reuse >= {"fixed_k_global_det_family", "adaptive_budget_det_family", "prefix_proxy_only_gate"}
        else "fail",
        repr(sorted(forbidden_reuse)),
    )
    _record(
        checks,
        "run_surface",
        "pass"
        if str(run_cfg.get("experiment_id") or "") == "DET_REENTRY"
        and str(run_cfg.get("device") or "") == "mps"
        else "fail",
        repr({"experiment_id": run_cfg.get("experiment_id"), "device": run_cfg.get("device")}),
    )
    _record(
        checks,
        "switches.det_only",
        "pass" if switches == {"meso": False, "flow": False, "det": True, "sparse": False, "phy": False} else "fail",
        repr(switches),
    )
    _record(
        checks,
        "det_k64",
        "pass" if early_stop.get("enabled") is True and int(float(early_stop.get("k_global") or 0)) == 64 else "fail",
        repr(early_stop),
    )
    quality_gate = sc_det.get("quality_gate") or {}
    _record(
        checks,
        "det_quality_gate",
        "pass"
        if quality_gate.get("enabled") is True
        and str(quality_gate.get("policy_label") or "") == "hybrid_prefix_quality_gate"
        and str(quality_gate.get("fallback_policy") or "") == "disable_det"
        and quality_gate.get("max_prefix_error_mean") is not None
        and quality_gate.get("max_prefix_error_p95") is not None
        else "fail",
        repr(quality_gate),
    )
    accuracy_path = assert_main_project_path(resolve_repo_path(str(accuracy.get("source_csv") or "")), arg_name="accuracy.source_csv")
    _record(checks, "accuracy.source_csv", "pass" if accuracy_path.exists() else "fail", str(accuracy_path))

    required_fields = {
        "candidate_label",
        "run_id",
        "experiment_id",
        "model",
        "latency_ms",
        "energy_j",
        "det_net_gain_j",
        "acc_top1",
        "acc_drop_pp",
        "pass_delta",
        "pass_det_net_gain",
        "accuracy_evidence",
        "performance_run_path",
        "phase1_summary_csv",
        "accuracy_source_csv",
        "det_runtime_enabled",
        "det_quality_gate_policy",
        "det_quality_gate_status",
        "det_quality_gate_fallback_policy",
        "det_quality_gate_require_measured_accuracy",
        "det_quality_gate_max_prefix_error_mean",
        "det_quality_gate_max_prefix_error_p95",
        "det_prefix_error_mean",
        "det_prefix_error_p95",
    }
    missing = sorted(field for field in required_fields if field not in summary_row)
    _record(checks, "summary.required_fields", "pass" if not missing else "fail", repr(missing))
    _record(
        checks,
        "summary.candidate_alignment",
        "pass"
        if summary_row.get("candidate_label") == str(candidate.get("candidate_label") or "")
        and summary_row.get("experiment_id") == "DET_REENTRY"
        else "fail",
        repr({"candidate_label": summary_row.get("candidate_label"), "experiment_id": summary_row.get("experiment_id")}),
    )
    det_net_gain = float(summary_row.get("det_net_gain_j") or 0.0)
    acc_drop = float(summary_row.get("acc_drop_pp") or 999.0)
    _record(checks, "summary.det_net_gain_j", "pass" if det_net_gain > 0.0 else "fail", str(det_net_gain))
    _record(checks, "summary.acc_drop_pp", "pass" if acc_drop <= 1.0 else "fail", str(acc_drop))
    _record(
        checks,
        "summary.pass_flags",
        "pass"
        if str(summary_row.get("pass_delta") or "").lower() == "true"
        and str(summary_row.get("pass_det_net_gain") or "").lower() == "true"
        else "fail",
        repr({"pass_delta": summary_row.get("pass_delta"), "pass_det_net_gain": summary_row.get("pass_det_net_gain")}),
    )
    _record(
        checks,
        "summary.accuracy_evidence",
        "pass" if "measured" in str(summary_row.get("accuracy_evidence") or "").lower() else "fail",
        str(summary_row.get("accuracy_evidence") or ""),
    )
    _record(
        checks,
        "summary.det_runtime_enabled",
        "pass" if _to_bool_str(summary_row.get("det_runtime_enabled")) else "fail",
        str(summary_row.get("det_runtime_enabled") or ""),
    )
    _record(
        checks,
        "summary.det_quality_gate_status",
        "pass" if str(summary_row.get("det_quality_gate_status") or "") == "pass" else "fail",
        str(summary_row.get("det_quality_gate_status") or ""),
    )
    _record(
        checks,
        "summary.det_quality_gate_contract",
        "pass"
        if str(summary_row.get("det_quality_gate_policy") or "") == str(quality_gate.get("policy_label") or "")
        and str(summary_row.get("det_quality_gate_fallback_policy") or "") == str(quality_gate.get("fallback_policy") or "")
        and _to_bool_str(summary_row.get("det_quality_gate_require_measured_accuracy"))
        == bool(quality_gate.get("require_measured_accuracy"))
        else "fail",
        repr(
            {
                "policy": summary_row.get("det_quality_gate_policy"),
                "fallback_policy": summary_row.get("det_quality_gate_fallback_policy"),
                "require_measured_accuracy": summary_row.get("det_quality_gate_require_measured_accuracy"),
            }
        ),
    )
    gate_mean = _to_float_or_none(summary_row.get("det_quality_gate_max_prefix_error_mean"))
    gate_p95 = _to_float_or_none(summary_row.get("det_quality_gate_max_prefix_error_p95"))
    prefix_mean = _to_float_or_none(summary_row.get("det_prefix_error_mean"))
    prefix_p95 = _to_float_or_none(summary_row.get("det_prefix_error_p95"))
    required_mean = _to_float_or_none(str(quality_gate.get("max_prefix_error_mean") or ""))
    required_p95 = _to_float_or_none(str(quality_gate.get("max_prefix_error_p95") or ""))
    _record(
        checks,
        "summary.det_quality_gate_thresholds",
        "pass"
        if gate_mean is not None
        and gate_p95 is not None
        and prefix_mean is not None
        and prefix_p95 is not None
        and required_mean is not None
        and required_p95 is not None
        and abs(gate_mean - required_mean) <= 1e-12
        and abs(gate_p95 - required_p95) <= 1e-12
        and prefix_mean <= required_mean + 1e-12
        and prefix_p95 <= required_p95 + 1e-12
        else "fail",
        repr(
            {
                "det_quality_gate_max_prefix_error_mean": summary_row.get("det_quality_gate_max_prefix_error_mean"),
                "det_quality_gate_max_prefix_error_p95": summary_row.get("det_quality_gate_max_prefix_error_p95"),
                "det_prefix_error_mean": summary_row.get("det_prefix_error_mean"),
                "det_prefix_error_p95": summary_row.get("det_prefix_error_p95"),
            }
        ),
    )
    perf_path = assert_main_project_path(resolve_repo_path(summary_row.get("performance_run_path") or ""), arg_name="summary.performance_run_path")
    _record(checks, "summary.performance_run_path", "pass" if perf_path.exists() else "fail", str(perf_path))
    phase1_summary_path = assert_main_project_path(
        resolve_repo_path(summary_row.get("phase1_summary_csv") or ""),
        arg_name="summary.phase1_summary_csv",
    )
    _record(
        checks,
        "summary.phase1_summary_csv",
        "pass" if phase1_summary_path.exists() else "fail",
        str(phase1_summary_path),
    )
    summary_accuracy_path = assert_main_project_path(resolve_repo_path(summary_row.get("accuracy_source_csv") or ""), arg_name="summary.accuracy_source_csv")
    _record(checks, "summary.accuracy_source_csv", "pass" if summary_accuracy_path.exists() else "fail", str(summary_accuracy_path))

    overall_ok = all(row["status"] == "pass" for row in checks)
    return checks, {"overall_ok": overall_ok, "summary_path": str(summary_path), "candidate_label": summary_row.get("candidate_label", "")}


def _write_report(out_path: Path, checks: list[dict[str, str]], summary: dict[str, Any]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# DET Reentry Candidate Check",
        "",
        f"- overall_ok: `{summary['overall_ok']}`",
        f"- candidate_label: `{summary['candidate_label']}`",
        f"- summary_path: `{summary['summary_path']}`",
        "",
        "## Checks",
        "",
    ]
    for row in checks:
        lines.append(f"- `{row['status']}` `{row['check_id']}`: {row['detail']}")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the bounded DET reentry candidate.")
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--summary_csv", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--out_dir", type=Path, default=ROOT / "experiments" / "results" / "report_data")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    template = _load_yaml(args.template.resolve())
    summary_row = _load_single_csv_row(args.summary_csv.resolve())
    checks, summary = evaluate_candidate(template, summary_row, summary_path=args.summary_csv.resolve())
    out_path = args.out_dir.resolve() / "check_det_reentry_candidate_20260331.md"
    _write_report(out_path, checks, summary)
    if not summary["overall_ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
