#!/usr/bin/env python3
"""Validate the governed DET/SPARSE reentry program package."""

from __future__ import annotations

import argparse
import json
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


DEFAULT_PROGRAM = ROOT / "configs" / "fuller_det_sparse_reentry_program_20260331.yaml"


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise SystemExit(f"Expected YAML mapping in {path}")
    return payload


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle) or {}
    if not isinstance(payload, dict):
        raise SystemExit(f"Expected JSON mapping in {path}")
    return payload


def _require_mapping(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise SystemExit(f"Missing required mapping: {key}")
    return value


def _require_list(payload: dict[str, Any], key: str) -> list[str]:
    value = payload.get(key)
    if not isinstance(value, list) or not value:
        raise SystemExit(f"Missing required list: {key}")
    return [str(item) for item in value]


def _record(rows: list[dict[str, str]], check_id: str, status: str, detail: str) -> None:
    rows.append({"check_id": check_id, "status": status, "detail": detail})


def evaluate_program(program: dict[str, Any]) -> tuple[list[dict[str, str]], dict[str, Any]]:
    checks: list[dict[str, str]] = []

    meta = _require_mapping(program, "meta")
    legacy = _require_mapping(program, "legacy_mainline")
    lane = _require_mapping(program, "reentry_lane")
    promotion = _require_mapping(program, "promotion")
    governance = _require_mapping(program, "governance")

    _record(
        checks,
        "meta.mode",
        "pass" if str(meta.get("mode") or "") == "superpower" else "fail",
        str(meta.get("mode") or ""),
    )
    _record(
        checks,
        "reentry_lane.active_experiment_id",
        "pass" if str(lane.get("active_experiment_id") or "") == "FULLER_REENTRY_V1" else "fail",
        str(lane.get("active_experiment_id") or ""),
    )

    for key in (
        "contract_yaml",
        "template_yaml",
        "contract_checker_py",
        "smoke_checker_py",
    ):
        path = assert_main_project_path(resolve_repo_path(str(legacy.get(key) or "")), arg_name=f"legacy_mainline.{key}")
        _record(checks, f"legacy_mainline.{key}", "pass" if path.exists() else "fail", str(path))

    for key in (
        "slice_template_yaml",
        "det_candidate_yaml",
        "sparse_operating_point_yaml",
        "plan_md",
        "det_definition_md",
        "sparse_definition_md",
        "integrated_definition_md",
        "device_claim_md",
        "promotion_note_md",
    ):
        path = assert_main_project_path(resolve_repo_path(str(lane.get(key) or "")), arg_name=f"reentry_lane.{key}")
        _record(checks, f"reentry_lane.{key}", "pass" if path.exists() else "fail", str(path))

    for key, rel in _require_mapping(lane, "report_data").items():
        path = assert_main_project_path(resolve_repo_path(str(rel or "")), arg_name=f"reentry_lane.report_data.{key}")
        _record(checks, f"reentry_lane.report_data.{key}", "pass" if path.exists() else "fail", str(path))

    checker_suite = _require_list(lane, "checker_suite")
    test_suite = _require_list(lane, "test_suite")
    _record(
        checks,
        "reentry_lane.checker_suite",
        "pass" if len(checker_suite) >= 4 else "fail",
        repr(checker_suite),
    )
    _record(
        checks,
        "reentry_lane.test_suite",
        "pass" if len(test_suite) >= 4 else "fail",
        repr(test_suite),
    )
    for rel in checker_suite + test_suite:
        path = assert_main_project_path(resolve_repo_path(rel), arg_name="reentry_lane.suite_entry")
        _record(checks, f"suite_entry:{Path(rel).name}", "pass" if path.exists() else "fail", str(path))

    run_tag = str(promotion.get("new_run_tag") or "")
    _record(
        checks,
        "promotion.new_run_tag",
        "pass" if bool(run_tag.strip()) else "fail",
        run_tag,
    )
    for key in ("quick_reports_dir", "paper_figures_dir", "paper_sync_pointer_json"):
        path = assert_main_project_path(resolve_repo_path(str(promotion.get(key) or "")), arg_name=f"promotion.{key}")
        _record(checks, f"promotion.{key}", "pass" if path.exists() else "fail", str(path))
    active_note = assert_main_project_path(
        resolve_repo_path(str(promotion.get("active_promotion_note_md") or "")),
        arg_name="promotion.active_promotion_note_md",
    )
    _record(
        checks,
        "promotion.active_promotion_note_md",
        "pass" if active_note.exists() else "fail",
        str(active_note),
    )
    predecessor_run_tag = str(promotion.get("predecessor_run_tag") or "")
    predecessor_status = str(promotion.get("predecessor_status") or "")
    predecessor_note_raw = str(promotion.get("predecessor_promotion_note_md") or "")
    predecessor_note = (
        assert_main_project_path(
            resolve_repo_path(predecessor_note_raw),
            arg_name="promotion.predecessor_promotion_note_md",
        )
        if predecessor_note_raw
        else None
    )
    _record(
        checks,
        "promotion.predecessor_status",
        "pass"
        if predecessor_status == "historical_governance_only"
        and bool(predecessor_run_tag)
        and predecessor_note is not None
        and predecessor_note.exists()
        else "fail",
        repr(
            {
                "predecessor_run_tag": predecessor_run_tag,
                "predecessor_status": predecessor_status,
                "predecessor_promotion_note_md": str(predecessor_note) if predecessor_note else predecessor_note_raw,
            }
        ),
    )
    pointer_path = assert_main_project_path(
        resolve_repo_path(str(promotion.get("paper_sync_pointer_json") or "")),
        arg_name="promotion.paper_sync_pointer_json",
    )
    pointer = _load_json(pointer_path)
    quick_dir = str(promotion.get("quick_reports_dir") or "")
    figure_dir = str(promotion.get("paper_figures_dir") or "")
    _record(
        checks,
        "promotion.paper_sync_pointer_alignment",
        "pass"
        if str(pointer.get("freeze_tag") or "") == run_tag
        and str(pointer.get("quick_reports_dir") or "") == quick_dir
        and str(pointer.get("paper_figures_dir") or "") == figure_dir
        else "fail",
        repr(
            {
                "pointer.freeze_tag": pointer.get("freeze_tag"),
                "pointer.quick_reports_dir": pointer.get("quick_reports_dir"),
                "pointer.paper_figures_dir": pointer.get("paper_figures_dir"),
            }
        ),
    )

    replacement_set = _require_list(promotion, "replacement_set")
    expected_replacements = {
        "configs/fuller_implementation_experiment_design_contract_20260319.yaml",
        "configs/fuller_first_implementation_slice_template_20260319.yaml",
        "docs/reports/20260321_det_sparse_reentry_audit_retirement_note.md",
        "docs/reports/20260321_fuller_architecture_traceability_restoration_note.md",
        "experiments/tools/check_fuller_implementation_experiment_design.py",
        "experiments/tools/check_fuller_first_slice_smoke.py",
        "experiments/tests/test_check_fuller_implementation_experiment_design.py",
        "experiments/tests/test_check_fuller_first_slice_smoke.py",
    }
    _record(
        checks,
        "promotion.replacement_set",
        "pass" if set(replacement_set) == expected_replacements else "fail",
        repr(sorted(replacement_set)),
    )
    _record(
        checks,
        "governance.boundary_controls",
        "pass"
        if governance.get("weaken_legacy_mainline_before_gate") is False
        and governance.get("historical_evidence_context_only") is True
        and governance.get("projection_only_sparse_rows_forbidden_for_promotion") is True
        else "fail",
        repr(governance),
    )

    overall_ok = all(row["status"] == "pass" for row in checks)
    summary = {
        "overall_ok": overall_ok,
        "active_experiment_id": str(lane.get("active_experiment_id") or ""),
        "new_run_tag": run_tag,
        "replacement_count": len(replacement_set),
    }
    return checks, summary


def _write_report(out_path: Path, checks: list[dict[str, str]], summary: dict[str, Any]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# DET/SPARSE Reentry Program Check",
        "",
        f"- overall_ok: `{summary['overall_ok']}`",
        f"- active_experiment_id: `{summary['active_experiment_id']}`",
        f"- new_run_tag: `{summary['new_run_tag']}`",
        "",
        "## Checks",
        "",
    ]
    for row in checks:
        lines.append(f"- `{row['status']}` `{row['check_id']}`: {row['detail']}")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the DET/SPARSE reentry program package.")
    parser.add_argument("--program", type=Path, default=DEFAULT_PROGRAM)
    parser.add_argument("--out_dir", type=Path, default=ROOT / "experiments" / "results" / "report_data")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    program = _load_yaml(args.program.resolve())
    checks, summary = evaluate_program(program)
    tag = str((_require_mapping(program, "meta")).get("tag") or "det_sparse_reentry_program")
    out_path = args.out_dir.resolve() / f"check_{tag}.md"
    _write_report(out_path, checks, summary)
    if not summary["overall_ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
