#!/usr/bin/env python3
"""Validate a future bitstream model-level measured rerun package."""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[2]
EXPERIMENTS_ROOT = ROOT_DIR / "experiments"
if str(EXPERIMENTS_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_ROOT))

from accuracy.bitstream_semantics import (  # noqa: E402
    BITSTREAM_MODEL_LEVEL_MEASURED_TRUTH_CLASS,
)
from accuracy.bitstream_truth_authorization import (  # noqa: E402
    assess_truth_class_authorization,
)


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"Expected JSON object in {path}")
    return payload


def _command_flag_value(command: list[str], flag: str) -> str:
    if flag not in command:
        return ""
    index = command.index(flag)
    if index + 1 >= len(command):
        return ""
    return str(command[index + 1]).strip()


def _has_flag(command: list[str], flag: str) -> bool:
    return flag in command


def _collect_declared_artifact_status(
    declared_paths: dict[str, str],
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    status: dict[str, dict[str, Any]] = {}
    issues: list[str] = []
    for key, value in declared_paths.items():
        exists = bool(value) and Path(value).exists()
        status[key] = {"path": value, "exists": exists}
        if value and not exists:
            issues.append(f"{key}_missing_on_disk")
    return status, issues


def assess_rerun_package(
    *,
    manifest_payload: dict[str, Any],
    manifest_path: Path,
    current_eligibility_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    structural_issues: list[str] = []
    jobs_payload = manifest_payload.get("jobs") or []
    if not isinstance(jobs_payload, list) or not jobs_payload:
        structural_issues.append("jobs_missing")
        jobs_payload = []

    annotation_truth_class = str(
        manifest_payload.get("annotation_measurement_truth_class") or ""
    ).strip()
    if annotation_truth_class != BITSTREAM_MODEL_LEVEL_MEASURED_TRUTH_CLASS:
        structural_issues.append("annotation_truth_class_not_model_level_measured")

    top_level_required = {
        "results_csv": str(manifest_payload.get("results_csv") or "").strip(),
        "annotated_results_csv": str(
            manifest_payload.get("annotated_results_csv") or ""
        ).strip(),
        "bitstream_truth_class_authorization_root": str(
            manifest_payload.get("bitstream_truth_class_authorization_root") or ""
        ).strip(),
        "prepared_phase1_config_root": str(
            manifest_payload.get("prepared_phase1_config_root") or ""
        ).strip(),
        "prepared_eligibility_report_root": str(
            manifest_payload.get("prepared_eligibility_report_root") or ""
        ).strip(),
    }
    for key, value in top_level_required.items():
        if not value:
            structural_issues.append(f"{key}_missing")
    declared_artifact_status, declared_artifact_issues = _collect_declared_artifact_status(
        top_level_required
    )

    job_assessments: list[dict[str, Any]] = []
    all_authorized = True
    all_structurally_valid = not structural_issues
    for raw_job in jobs_payload:
        job = dict(raw_job or {})
        eval_run_id = str(job.get("eval_run_id") or "").strip()
        command = list(job.get("commands", [[]])[0] or [])
        if not isinstance(command, list):
            command = []
        authorization_note = str(
            job.get("bitstream_truth_class_authorization_note") or ""
        ).strip()
        prepared_phase1_config = str(job.get("prepared_phase1_config") or "").strip()
        eligibility_json = str(
            job.get("prepared_eligibility_report_json") or ""
        ).strip()
        eligibility_md = str(job.get("prepared_eligibility_report_md") or "").strip()

        job_issues: list[str] = []
        if not eval_run_id:
            job_issues.append("eval_run_id_missing")
        if _command_flag_value(command, "--device") != "mps":
            job_issues.append("device_not_mps")
        if not _has_flag(command, "--enable_bitstream_pilot"):
            job_issues.append("bitstream_pilot_flag_missing")
        if (
            _command_flag_value(command, "--bitstream_measurement_truth_class")
            != BITSTREAM_MODEL_LEVEL_MEASURED_TRUTH_CLASS
        ):
            job_issues.append("runtime_truth_class_not_model_level_measured")
        if (
            _command_flag_value(command, "--bitstream_truth_class_authorization_note")
            != authorization_note
        ):
            job_issues.append("authorization_note_flag_mismatch")
        if not _command_flag_value(command, "--bitstream_surface_scope"):
            job_issues.append("bitstream_surface_scope_missing")
        if not _command_flag_value(command, "--bitstream_contract_note"):
            job_issues.append("bitstream_contract_note_missing")
        if not prepared_phase1_config:
            job_issues.append("prepared_phase1_config_missing")
        if not eligibility_json:
            job_issues.append("prepared_eligibility_report_json_missing")
        if not eligibility_md:
            job_issues.append("prepared_eligibility_report_md_missing")

        resolved_authorization_note = (
            Path(authorization_note)
            if authorization_note and Path(authorization_note).exists()
            else None
        )
        job_declared_artifact_status, job_materialization_issues = _collect_declared_artifact_status(
            {
                "authorization_note": authorization_note,
                "prepared_phase1_config": prepared_phase1_config,
                "prepared_eligibility_report_json": eligibility_json,
                "prepared_eligibility_report_md": eligibility_md,
            }
        )
        authorization_assessment = assess_truth_class_authorization(
            resolved_authorization_note,
            expected_run_id=eval_run_id or None,
        )
        if job_issues:
            all_structurally_valid = False
        if not bool(authorization_assessment["authorized"]):
            all_authorized = False

        job_assessments.append(
            {
                "eval_run_id": eval_run_id,
                "authorization_note": authorization_note,
                "authorization_status": str(authorization_assessment["status"]),
                "authorization_note_exists": bool(resolved_authorization_note is not None),
                "authorized_run_id": str(
                    authorization_assessment.get("authorized_run_id") or ""
                ),
                "prepared_phase1_config": prepared_phase1_config,
                "prepared_eligibility_report_json": eligibility_json,
                "prepared_eligibility_report_md": eligibility_md,
                "declared_artifact_status": job_declared_artifact_status,
                "declared_artifact_materialization_issues": job_materialization_issues,
                "command": shlex.join(command),
                "job_issues": job_issues,
            }
        )

    if not all_structurally_valid:
        package_status = "invalid"
        runnable_now = False
    elif all_authorized:
        package_status = "ready_to_launch"
        runnable_now = True
    else:
        package_status = "draft_not_authorized"
        runnable_now = False

    current_observed_blockers = list(
        (current_eligibility_payload or {}).get("blockers") or []
    )
    aggregated_materialization_issues = list(declared_artifact_issues)
    for job in job_assessments:
        eval_run_id = str(job.get("eval_run_id") or "").strip() or "unknown_job"
        for issue in job.get("declared_artifact_materialization_issues") or []:
            aggregated_materialization_issues.append(f"{eval_run_id}:{issue}")
    aggregated_materialization_issues = list(dict.fromkeys(aggregated_materialization_issues))
    return {
        "manifest_path": str(manifest_path),
        "job_count": len(job_assessments),
        "annotation_measurement_truth_class": annotation_truth_class,
        "package_status": package_status,
        "runnable_now": runnable_now,
        "launch_wrapper_required": True,
        "launch_wrapper_recommendation": "caffeinate -dimsu with external 900s budget, SIGINT, 20s grace, then SIGKILL",
        "structural_issues": structural_issues,
        "current_observed_blockers": current_observed_blockers,
        "declared_artifact_status": declared_artifact_status,
        "declared_artifact_materialization_complete": not aggregated_materialization_issues,
        "declared_artifact_materialization_issues": aggregated_materialization_issues,
        "jobs": job_assessments,
        "next_action": (
            "await_explicit_authorization_then_remove_dry_run"
            if package_status == "draft_not_authorized"
            else (
                "repair_package_structure"
                if package_status == "invalid"
                else "launch_measured_rerun_under_governed_wrapper"
            )
        ),
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Bitstream Measured Rerun Package Check",
        "",
        f"- manifest_path: `{payload['manifest_path']}`",
        f"- package_status: `{payload['package_status']}`",
        f"- runnable_now: `{payload['runnable_now']}`",
        f"- launch_wrapper_required: `{payload['launch_wrapper_required']}`",
        f"- annotation_measurement_truth_class: `{payload['annotation_measurement_truth_class']}`",
        f"- next_action: `{payload['next_action']}`",
        f"- declared_artifact_materialization_complete: `{payload['declared_artifact_materialization_complete']}`",
    ]
    structural_issues = list(payload.get("structural_issues") or [])
    if structural_issues:
        lines.extend(["", "## Structural Issues", ""])
        lines.extend(f"- `{issue}`" for issue in structural_issues)
    blockers = list(payload.get("current_observed_blockers") or [])
    if blockers:
        lines.extend(["", "## Current Observed Blockers", ""])
        lines.extend(f"- `{item}`" for item in blockers)
    declared_artifact_status = dict(payload.get("declared_artifact_status") or {})
    declared_artifact_issues = list(
        payload.get("declared_artifact_materialization_issues") or []
    )
    if declared_artifact_status or declared_artifact_issues:
        lines.extend(["", "## Declared Artifact Materialization", ""])
        for label, status in declared_artifact_status.items():
            lines.append(
                f"- {label}: exists=`{status.get('exists')}` path=`{status.get('path')}`"
            )
        if declared_artifact_issues:
            lines.append(
                f"- materialization_issues: `{', '.join(str(item) for item in declared_artifact_issues)}`"
            )
    lines.extend(["", "## Jobs", ""])
    for job in payload.get("jobs") or []:
        lines.extend(
            [
                f"- eval_run_id: `{job['eval_run_id']}`",
                f"- authorization_status: `{job['authorization_status']}`",
                f"- authorization_note_exists: `{job['authorization_note_exists']}`",
                f"- authorization_note: `{job['authorization_note']}`",
                f"- prepared_phase1_config: `{job['prepared_phase1_config']}`",
                f"- prepared_eligibility_report_json: `{job['prepared_eligibility_report_json']}`",
                f"- prepared_eligibility_report_md: `{job['prepared_eligibility_report_md']}`",
            ]
        )
        job_artifact_status = dict(job.get("declared_artifact_status") or {})
        for label, status in job_artifact_status.items():
            lines.append(
                f"- {label}_exists: `{status.get('exists')}`"
            )
        if job.get("declared_artifact_materialization_issues"):
            lines.append(
                "- declared_artifact_materialization_issues: "
                f"`{', '.join(str(item) for item in job['declared_artifact_materialization_issues'])}`"
            )
        if job.get("job_issues"):
            lines.append(f"- job_issues: `{', '.join(job['job_issues'])}`")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate a future bitstream model-level measured rerun package manifest.",
    )
    parser.add_argument("--manifest_json", required=True)
    parser.add_argument("--current_eligibility_json", default=None)
    parser.add_argument("--output_json", default=None)
    parser.add_argument("--output_md", default=None)
    return parser


def main() -> int:
    parser = build_argparser()
    args = parser.parse_args()
    manifest_path = Path(args.manifest_json)
    manifest_payload = _load_json(manifest_path)
    current_eligibility_payload = (
        _load_json(Path(args.current_eligibility_json))
        if args.current_eligibility_json
        else None
    )
    payload = assess_rerun_package(
        manifest_payload=manifest_payload,
        manifest_path=manifest_path,
        current_eligibility_payload=current_eligibility_payload,
    )
    if args.output_json:
        _write_json(Path(args.output_json), payload)
    if args.output_md:
        _write_markdown(Path(args.output_md), payload)
    print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
