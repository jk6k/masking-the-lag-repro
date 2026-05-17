#!/usr/bin/env python3
"""Summarize observed status for a prepared true-SC accuracy launch surface."""

from __future__ import annotations

import argparse
import csv
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

DEFAULT_STALL_TIMEOUT_SECONDS = 180.0


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"Expected JSON object in {path}")
    return payload


def _load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _to_bool(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _to_int(value: Any) -> int | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        return None


def _format_int_list(values: list[int]) -> str:
    return "[" + ", ".join(str(value) for value in values) + "]"


def _infer_seed(*values: Any) -> int | None:
    for value in values:
        text = str(value or "").strip()
        if not text:
            continue
        for token in re.split(r"[^A-Za-z0-9]+", text):
            lowered = token.lower()
            if lowered.startswith("seed") and lowered[4:].isdigit():
                return int(lowered[4:])
            if lowered.startswith("s") and lowered[1:].isdigit():
                return int(lowered[1:])
    return None


def _parse_timestamp(raw: str | None, *, default_tz: Any) -> datetime | None:
    text = str(raw or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=default_tz)
    return parsed.astimezone(default_tz)


def _read_raw_accuracy(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "exists": False,
            "row_count": 0,
            "baseline_row_count": 0,
            "target_row_count": 0,
            "target_seeds_present": [],
            "target_measurement_windows": [],
            "target_truth_classes": [],
            "last_target_row": {},
        }

    rows = _load_csv_rows(path)
    baseline_rows = [row for row in rows if _to_bool(row.get("baseline"))]
    target_rows = [row for row in rows if not _to_bool(row.get("baseline"))]

    target_seeds = sorted(
        {
            seed
            for seed in (_to_int(row.get("seed")) for row in target_rows)
            if seed is not None
        }
    )
    target_measurement_windows = sorted(
        {
            str(row.get("measurement_window") or "").strip()
            for row in target_rows
            if str(row.get("measurement_window") or "").strip()
        }
    )
    target_truth_classes = sorted(
        {
            str(row.get("bitstream_measurement_truth_class") or "").strip()
            for row in target_rows
            if str(row.get("bitstream_measurement_truth_class") or "").strip()
        }
    )
    last_target_row = target_rows[-1] if target_rows else {}
    return {
        "exists": True,
        "row_count": len(rows),
        "baseline_row_count": len(baseline_rows),
        "target_row_count": len(target_rows),
        "target_seeds_present": target_seeds,
        "target_measurement_windows": target_measurement_windows,
        "target_truth_classes": target_truth_classes,
        "last_target_row": {
            "run_id": str(last_target_row.get("run_id") or "").strip(),
            "measurement_window": str(last_target_row.get("measurement_window") or "").strip(),
            "seed": _to_int(last_target_row.get("seed")),
            "top1": str(last_target_row.get("top1") or "").strip(),
            "top5": str(last_target_row.get("top5") or "").strip(),
        },
    }


def _read_annotated_accuracy(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False, "row_count": 0}
    rows = _load_csv_rows(path)
    return {"exists": True, "row_count": len(rows)}


def _scan_progress_events(
    path: Path,
    *,
    now: datetime,
    stall_timeout_seconds: float,
    default_tz: Any,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
        "event_count": 0,
        "quantized_event_count": 0,
        "command_started": False,
        "command_complete": False,
        "command_error": False,
        "baseline_started": False,
        "baseline_complete": False,
        "quantized_started": False,
        "quantized_complete": False,
        "latest_event_timestamp": "",
        "latest_event": "",
        "latest_pass_kind": "",
        "latest_processed_samples": None,
        "latest_total_samples": None,
        "latest_runtime_stage": "",
        "latest_runtime_detail": "",
        "latest_event_age_seconds": None,
        "max_quantized_processed_samples": None,
        "max_quantized_processed_timestamp": "",
        "max_quantized_total_samples": None,
        "max_quantized_completion_ratio": None,
        "max_quantized_completion_percent": None,
        "first_nonzero_processed_samples": None,
        "first_nonzero_processed_timestamp": "",
        "last_nonzero_processed_samples": None,
        "last_nonzero_processed_timestamp": "",
        "nonzero_progress_duration_seconds": None,
        "processed_samples_regression_count": 0,
        "first_processed_samples_regression_timestamp": "",
        "first_processed_samples_regression_from": None,
        "first_processed_samples_regression_to": None,
        "latest_processed_samples_below_peak": False,
        "latest_processed_samples_reset_to_zero_after_progress": False,
        "latest_processed_samples_partial_recovery_after_regression": False,
        "latest_processed_samples_gap_from_peak": None,
        "peak_quantized_progress_retained_ratio": None,
        "peak_quantized_progress_retained_percent": None,
        "latest_quantized_completion_ratio": None,
        "latest_quantized_completion_percent": None,
        "zero_after_progress_duration_seconds": None,
        "stalled": False,
    }
    if not path.exists():
        return summary

    latest_timestamp: datetime | None = None
    previous_quantized_processed_samples: int | None = None
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                continue

            summary["event_count"] += 1
            event = str(payload.get("event") or "").strip()
            pass_kind = str(payload.get("pass_kind") or "").strip()
            event_timestamp = _parse_timestamp(
                str(payload.get("timestamp") or "").strip(),
                default_tz=default_tz,
            )

            if event == "command_start":
                summary["command_started"] = True
            elif event == "command_complete":
                summary["command_complete"] = True
            elif event in {"command_error", "command_failed"}:
                summary["command_error"] = True

            if pass_kind == "baseline_eval_pass":
                summary["baseline_started"] = True
                if event == "pass_complete":
                    summary["baseline_complete"] = True
            elif pass_kind == "quantized_eval_pass":
                summary["quantized_event_count"] += 1
                summary["quantized_started"] = True
                if event == "pass_complete":
                    summary["quantized_complete"] = True
                processed_samples = _to_int(payload.get("processed_samples"))
                if processed_samples is not None:
                    if (
                        summary["first_nonzero_processed_samples"] is None
                        and processed_samples > 0
                    ):
                        summary["first_nonzero_processed_samples"] = processed_samples
                        summary["first_nonzero_processed_timestamp"] = (
                            event_timestamp.isoformat() if event_timestamp is not None else ""
                        )
                    if processed_samples > 0:
                        summary["last_nonzero_processed_samples"] = processed_samples
                        summary["last_nonzero_processed_timestamp"] = (
                            event_timestamp.isoformat() if event_timestamp is not None else ""
                        )
                    if (
                        previous_quantized_processed_samples is not None
                        and processed_samples < previous_quantized_processed_samples
                    ):
                        summary["processed_samples_regression_count"] += 1
                        if not summary["first_processed_samples_regression_timestamp"]:
                            summary["first_processed_samples_regression_timestamp"] = (
                                event_timestamp.isoformat()
                                if event_timestamp is not None
                                else ""
                            )
                            summary["first_processed_samples_regression_from"] = (
                                previous_quantized_processed_samples
                            )
                            summary["first_processed_samples_regression_to"] = (
                                processed_samples
                            )
                    previous_quantized_processed_samples = processed_samples
                if processed_samples is not None and (
                    summary["max_quantized_processed_samples"] is None
                    or processed_samples > summary["max_quantized_processed_samples"]
                ):
                    summary["max_quantized_processed_samples"] = processed_samples
                    summary["max_quantized_processed_timestamp"] = (
                        event_timestamp.isoformat() if event_timestamp is not None else ""
                    )
                    summary["max_quantized_total_samples"] = _to_int(
                        payload.get("total_samples")
                    )

            if event_timestamp is not None and (
                latest_timestamp is None or event_timestamp >= latest_timestamp
            ):
                latest_timestamp = event_timestamp
                summary["latest_event_timestamp"] = event_timestamp.isoformat()
                summary["latest_event"] = event
                summary["latest_pass_kind"] = pass_kind
                summary["latest_processed_samples"] = _to_int(
                    payload.get("processed_samples")
                )
                summary["latest_total_samples"] = _to_int(payload.get("total_samples"))
                summary["latest_runtime_stage"] = str(
                    payload.get("runtime_stage") or ""
                ).strip()
                summary["latest_runtime_detail"] = str(
                    payload.get("runtime_detail") or ""
                ).strip()

    if latest_timestamp is not None:
        age_seconds = max((now - latest_timestamp).total_seconds(), 0.0)
        summary["latest_event_age_seconds"] = round(age_seconds, 3)
        summary["stalled"] = bool(
            summary["quantized_started"]
            and not summary["command_complete"]
            and age_seconds > float(stall_timeout_seconds)
        )
    latest_processed_samples = summary["latest_processed_samples"]
    max_quantized_processed_samples = summary["max_quantized_processed_samples"]
    last_nonzero_processed_samples = summary["last_nonzero_processed_samples"]
    summary["latest_processed_samples_below_peak"] = bool(
        latest_processed_samples is not None
        and max_quantized_processed_samples is not None
        and latest_processed_samples < max_quantized_processed_samples
    )
    summary["latest_processed_samples_reset_to_zero_after_progress"] = bool(
        latest_processed_samples == 0
        and last_nonzero_processed_samples is not None
        and last_nonzero_processed_samples > 0
    )
    summary["latest_processed_samples_partial_recovery_after_regression"] = bool(
        summary["latest_processed_samples_below_peak"]
        and latest_processed_samples not in (None, 0)
    )
    if (
        latest_processed_samples is not None
        and max_quantized_processed_samples is not None
        and latest_processed_samples <= max_quantized_processed_samples
    ):
        summary["latest_processed_samples_gap_from_peak"] = (
            max_quantized_processed_samples - latest_processed_samples
        )
    if (
        latest_processed_samples is not None
        and max_quantized_processed_samples not in (None, 0)
        and max_quantized_processed_samples > 0
    ):
        retained_ratio = latest_processed_samples / max_quantized_processed_samples
        summary["peak_quantized_progress_retained_ratio"] = round(retained_ratio, 8)
        summary["peak_quantized_progress_retained_percent"] = round(
            retained_ratio * 100.0,
            6,
        )

    latest_total_samples = summary["latest_total_samples"]
    if (
        latest_processed_samples is not None
        and latest_total_samples not in (None, 0)
        and latest_total_samples > 0
    ):
        latest_completion_ratio = latest_processed_samples / latest_total_samples
        summary["latest_quantized_completion_ratio"] = round(latest_completion_ratio, 8)
        summary["latest_quantized_completion_percent"] = round(
            latest_completion_ratio * 100.0, 6
        )

    max_quantized_total_samples = summary["max_quantized_total_samples"]
    if (
        max_quantized_processed_samples is not None
        and max_quantized_total_samples not in (None, 0)
        and max_quantized_total_samples > 0
    ):
        max_completion_ratio = (
            max_quantized_processed_samples / max_quantized_total_samples
        )
        summary["max_quantized_completion_ratio"] = round(max_completion_ratio, 8)
        summary["max_quantized_completion_percent"] = round(
            max_completion_ratio * 100.0, 6
        )

    first_nonzero_timestamp = _parse_timestamp(
        summary["first_nonzero_processed_timestamp"],
        default_tz=default_tz,
    )
    last_nonzero_timestamp = _parse_timestamp(
        summary["last_nonzero_processed_timestamp"],
        default_tz=default_tz,
    )
    if (
        first_nonzero_timestamp is not None
        and last_nonzero_timestamp is not None
        and last_nonzero_timestamp >= first_nonzero_timestamp
    ):
        summary["nonzero_progress_duration_seconds"] = round(
            (last_nonzero_timestamp - first_nonzero_timestamp).total_seconds(),
            3,
        )

    first_regression_timestamp = _parse_timestamp(
        summary["first_processed_samples_regression_timestamp"],
        default_tz=default_tz,
    )
    if (
        summary["latest_processed_samples_reset_to_zero_after_progress"]
        and latest_timestamp is not None
        and first_regression_timestamp is not None
        and latest_timestamp >= first_regression_timestamp
    ):
        summary["zero_after_progress_duration_seconds"] = round(
            (latest_timestamp - first_regression_timestamp).total_seconds(),
            3,
        )
    return summary


def _summarize_seed_job(
    job: dict[str, Any],
    *,
    now: datetime,
    stall_timeout_seconds: float,
    default_tz: Any,
) -> dict[str, Any]:
    progress_jsonls = [
        str(progress_jsonl or "").strip()
        for progress_jsonl in (job.get("progress_jsonls") or [])
        if str(progress_jsonl or "").strip()
    ]
    event_summaries = [
        _scan_progress_events(
            Path(event_path),
            now=now,
            stall_timeout_seconds=stall_timeout_seconds,
            default_tz=default_tz,
        )
        for event_path in progress_jsonls
    ]
    latest_event_summary = max(
        event_summaries,
        key=lambda item: item.get("latest_event_timestamp") or "",
        default={},
    )
    any_event_file_exists = any(bool(item.get("exists")) for item in event_summaries)
    any_baseline_started = any(bool(item.get("baseline_started")) for item in event_summaries)
    any_baseline_complete = any(bool(item.get("baseline_complete")) for item in event_summaries)
    any_quantized_started = any(bool(item.get("quantized_started")) for item in event_summaries)
    any_quantized_complete = any(bool(item.get("quantized_complete")) for item in event_summaries)
    any_command_complete = any(bool(item.get("command_complete")) for item in event_summaries)
    any_command_error = any(bool(item.get("command_error")) for item in event_summaries)
    any_stalled = any(bool(item.get("stalled")) for item in event_summaries)

    if any_command_error:
        seed_status = "command_error"
    elif any_command_complete:
        seed_status = "command_complete"
    elif any_quantized_started:
        seed_status = (
            "stalled_quantized_eval_pass"
            if any_stalled
            else "quantized_eval_incomplete"
        )
    elif any_baseline_started:
        seed_status = (
            "stalled_baseline_eval_pass"
            if any_stalled
            else "baseline_complete_only"
            if any_baseline_complete
            else "baseline_eval_incomplete"
        )
    elif any_event_file_exists:
        seed_status = "event_file_without_progress_markers"
    else:
        seed_status = "planned_missing_event_file"

    return {
        "step_id": str(job.get("step_id") or "").strip(),
        "eval_run_id": str(job.get("eval_run_id") or "").strip(),
        "planned_seed": _infer_seed(
            job.get("eval_run_id"),
            job.get("step_id"),
            *progress_jsonls,
        ),
        "progress_jsonls": progress_jsonls,
        "event_file_exists": any_event_file_exists,
        "seed_status": seed_status,
        "latest_event_summary": latest_event_summary,
    }


def _scan_progress_manifest(
    path: Path,
    *,
    now: datetime,
    default_tz: Any,
) -> dict[str, Any]:
    if not path.exists():
        return {
            "exists": False,
            "planned_job_count": 0,
            "analysis_grade_ready": None,
            "analysis_grade_blockers": [],
            "runtime_health_gate": {},
            "event_files": [],
            "event_summaries": [],
            "seed_summaries": [],
        }

    payload = _load_json(path)
    jobs = payload.get("jobs") or []
    if not isinstance(jobs, list):
        jobs = []

    runtime_health_gate = payload.get("runtime_health_gate") or {}
    if not isinstance(runtime_health_gate, dict):
        runtime_health_gate = {}
    runtime_health_gate = dict(runtime_health_gate)
    stall_timeout_seconds = float(
        runtime_health_gate.get("stall_timeout_seconds") or DEFAULT_STALL_TIMEOUT_SECONDS
    )
    runtime_health_gate.setdefault("stall_timeout_seconds", stall_timeout_seconds)
    event_files: list[str] = []
    seed_summaries: list[dict[str, Any]] = []
    for job in jobs:
        if not isinstance(job, dict):
            continue
        for progress_jsonl in job.get("progress_jsonls") or []:
            text = str(progress_jsonl or "").strip()
            if text and text not in event_files:
                event_files.append(text)
        seed_summaries.append(
            _summarize_seed_job(
                job,
                now=now,
                stall_timeout_seconds=stall_timeout_seconds,
                default_tz=default_tz,
            )
        )

    event_summaries = [
        seed_summary["latest_event_summary"]
        for seed_summary in seed_summaries
        if seed_summary.get("latest_event_summary")
    ]
    latest_event_summary = max(
        event_summaries,
        key=lambda item: item.get("latest_event_timestamp") or "",
        default=None,
    )
    return {
        "exists": True,
        "planned_job_count": len(jobs),
        "analysis_grade_ready": payload.get("analysis_grade_ready"),
        "analysis_grade_blockers": list(payload.get("analysis_grade_blockers") or []),
        "runtime_health_gate": runtime_health_gate,
        "event_files": event_files,
        "event_summaries": event_summaries,
        "seed_summaries": seed_summaries,
        "latest_event_summary": latest_event_summary,
    }


def _build_contract_index(path: Path | None) -> dict[str, dict[str, dict[str, str]]]:
    if path is None or not path.exists():
        return {}
    rows = _load_csv_rows(path)
    index: dict[str, dict[str, dict[str, str]]] = {}
    for row in rows:
        experiment_id = str(row.get("experiment_id") or "").strip().upper()
        row_role = str(row.get("row_role") or "").strip().lower()
        if not experiment_id or not row_role:
            continue
        index.setdefault(experiment_id, {})[row_role] = row
    return index


def _contract_view(
    contract_rows: dict[str, dict[str, str]],
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row_role, row in contract_rows.items():
        result[row_role] = {
            "row_status": str(row.get("row_status") or "").strip(),
            "selected_row_id": str(row.get("selected_row_id") or "").strip(),
            "selected_source_run_id": str(row.get("selected_source_run_id") or "").strip(),
            "selected_measurement_window": str(
                row.get("selected_measurement_window") or ""
            ).strip(),
            "selected_truth_class": str(row.get("selected_truth_class") or "").strip(),
            "context_mismatches": str(row.get("context_mismatches") or "").strip(),
            "config_snapshot_contract_state": str(
                row.get("config_snapshot_contract_state") or ""
            ).strip(),
        }
    return result


def assess_launch_status(
    manifest_payload: dict[str, Any],
    *,
    contract_index: dict[str, dict[str, dict[str, str]]],
    experiments_filter: set[str] | None,
    now: datetime,
) -> dict[str, Any]:
    manifest_rows = manifest_payload.get("manifest_rows") or []
    if not isinstance(manifest_rows, list):
        raise SystemExit("Launch manifest is missing a valid manifest_rows list")

    assessed_rows: list[dict[str, Any]] = []
    overall_statuses: list[str] = []
    overall_blockers: list[str] = []
    default_tz = now.tzinfo
    for raw_row in manifest_rows:
        if not isinstance(raw_row, dict):
            continue
        experiment_id = str(raw_row.get("experiment_id") or "").strip().upper()
        if not experiment_id:
            continue
        if experiments_filter is not None and experiment_id not in experiments_filter:
            continue

        raw_accuracy = _read_raw_accuracy(Path(str(raw_row.get("results_csv") or "")))
        annotated_accuracy = _read_annotated_accuracy(
            Path(str(raw_row.get("annotated_results_csv") or ""))
        )
        progress_manifest = _scan_progress_manifest(
            Path(str(raw_row.get("progress_manifest_json") or "")),
            now=now,
            default_tz=default_tz,
        )

        event_summaries = list(progress_manifest.get("event_summaries") or [])
        seed_summaries = list(progress_manifest.get("seed_summaries") or [])
        latest_event_summary = dict(progress_manifest.get("latest_event_summary") or {})
        any_quantized_started = any(
            bool(item.get("quantized_started")) for item in event_summaries
        )
        any_quantized_complete = any(
            bool(item.get("quantized_complete")) for item in event_summaries
        )
        any_command_complete = any(
            bool(item.get("command_complete")) for item in event_summaries
        )
        any_stalled = any(bool(item.get("stalled")) for item in event_summaries)
        required_seeds = sorted(
            {
                seed
                for seed in (
                    _to_int(seed)
                    for seed in (raw_row.get("analysis_grade_required_seeds") or [])
                )
                if seed is not None
            }
        )
        target_seeds_present = list(raw_accuracy["target_seeds_present"])
        missing_required_seeds = [
            seed for seed in required_seeds if seed not in target_seeds_present
        ]
        progress_event_seeds_present = sorted(
            {
                seed
                for seed in (
                    _to_int(item.get("planned_seed"))
                    for item in seed_summaries
                    if item.get("event_file_exists")
                )
                if seed is not None
            }
        )
        missing_progress_event_seeds = [
            seed for seed in required_seeds if seed not in progress_event_seeds_present
        ]
        quantized_progress_seeds = sorted(
            {
                seed
                for seed in (
                    _to_int(item.get("planned_seed"))
                    for item in seed_summaries
                    if (
                        _to_int(
                            dict(item.get("latest_event_summary") or {}).get(
                                "max_quantized_processed_samples"
                            )
                        )
                        or 0
                    )
                    > 0
                )
                if seed is not None
            }
        )
        progressed_seeds_missing_target_rows = [
            seed for seed in quantized_progress_seeds if seed not in target_seeds_present
        ]
        stalled_started_seeds = sorted(
            {
                seed
                for seed in (
                    _to_int(item.get("planned_seed"))
                    for item in seed_summaries
                    if str(item.get("seed_status") or "").startswith("stalled_")
                )
                if seed is not None
            }
        )
        processed_samples_regressed_seeds = sorted(
            {
                seed
                for seed in (
                    _to_int(item.get("planned_seed"))
                    for item in seed_summaries
                    if _to_int(
                        dict(item.get("latest_event_summary") or {}).get(
                            "processed_samples_regression_count"
                        )
                    )
                )
                if seed is not None
            }
        )
        reset_to_zero_after_progress_seeds = sorted(
            {
                seed
                for seed in (
                    _to_int(item.get("planned_seed"))
                    for item in seed_summaries
                    if bool(
                        dict(item.get("latest_event_summary") or {}).get(
                            "latest_processed_samples_reset_to_zero_after_progress"
                        )
                    )
                )
                if seed is not None
            }
        )
        partial_recovery_after_regression_seeds = sorted(
            {
                seed
                for seed in (
                    _to_int(item.get("planned_seed"))
                    for item in seed_summaries
                    if bool(
                        dict(item.get("latest_event_summary") or {}).get(
                            "latest_processed_samples_partial_recovery_after_regression"
                        )
                    )
                )
                if seed is not None
            }
        )

        observed_blockers: list[str] = []
        if raw_accuracy["target_row_count"] == 0:
            observed_blockers.append("missing_analysis_grade_accuracy_row")
        if missing_required_seeds:
            observed_blockers.append(
                "missing_target_seeds=" + _format_int_list(missing_required_seeds)
            )
        if missing_progress_event_seeds:
            observed_blockers.append(
                "missing_progress_event_seeds="
                + _format_int_list(missing_progress_event_seeds)
            )
        if progressed_seeds_missing_target_rows:
            observed_blockers.append(
                "progressed_seeds_missing_target_rows="
                + _format_int_list(progressed_seeds_missing_target_rows)
            )
        if bool(raw_row.get("analysis_grade_require_full_eval")) and not any_quantized_complete:
            observed_blockers.append("full_eval_not_completed")
        if not annotated_accuracy["exists"]:
            observed_blockers.append("annotated_accuracy_missing")
        if not progress_manifest["exists"]:
            observed_blockers.append("progress_manifest_missing")
        elif not event_summaries:
            observed_blockers.append("progress_events_missing")
        elif any_quantized_started and not any_command_complete:
            observed_blockers.append(
                "progress_stalled_past_runtime_threshold"
                if any_stalled
                else "quantized_eval_incomplete"
            )
        if stalled_started_seeds:
            observed_blockers.append(
                "stalled_started_seeds=" + _format_int_list(stalled_started_seeds)
            )
        if processed_samples_regressed_seeds:
            observed_blockers.append(
                "processed_samples_regressed_seeds="
                + _format_int_list(processed_samples_regressed_seeds)
            )
        if reset_to_zero_after_progress_seeds:
            observed_blockers.append(
                "processed_samples_reset_to_zero_after_progress_seeds="
                + _format_int_list(reset_to_zero_after_progress_seeds)
            )
        if partial_recovery_after_regression_seeds:
            observed_blockers.append(
                "partial_recovery_after_regression_seeds="
                + _format_int_list(partial_recovery_after_regression_seeds)
            )

        contract_rows = contract_index.get(experiment_id) or {}
        contract_view = _contract_view(contract_rows)
        target_contract = contract_rows.get("target") or {}
        target_row_status = str(target_contract.get("row_status") or "").strip()
        if target_row_status and target_row_status != "ready":
            observed_blockers.append(f"contract_target_row_status={target_row_status}")
        target_context_mismatches = str(
            target_contract.get("context_mismatches") or ""
        ).strip()
        if target_context_mismatches:
            observed_blockers.append(
                f"contract_target_context_mismatches={target_context_mismatches}"
            )

        manifest_blockers = [
            str(item)
            for item in (
                raw_row.get("analysis_grade_blockers")
                or manifest_payload.get("analysis_grade_blockers")
                or []
            )
        ]

        if (
            raw_accuracy["target_row_count"] > 0
            and not missing_required_seeds
            and annotated_accuracy["exists"]
            and any_command_complete
            and not observed_blockers
        ):
            observed_status = "complete"
        elif raw_accuracy["target_row_count"] > 0:
            observed_status = (
                "partial_analysis_grade_rows_stalled"
                if any_stalled
                else "partial_analysis_grade_rows"
            )
        elif any_quantized_started:
            observed_status = (
                "stalled_quantized_eval_pass"
                if any_stalled
                else "quantized_eval_incomplete"
            )
        elif raw_accuracy["baseline_row_count"] > 0:
            observed_status = "baseline_only"
        elif progress_manifest["exists"]:
            observed_status = "manifest_only"
        else:
            observed_status = "missing_launch_artifacts"

        assessed = {
            "experiment_id": experiment_id,
            "run_id": str(raw_row.get("run_id") or "").strip(),
            "observed_status": observed_status,
            "manifest_analysis_grade_ready": bool(raw_row.get("analysis_grade_ready")),
            "manifest_runtime_launch_ready": bool(raw_row.get("runtime_launch_ready")),
            "manifest_analysis_grade_blockers": manifest_blockers,
            "raw_accuracy": raw_accuracy,
            "annotated_accuracy": annotated_accuracy,
            "progress_manifest": progress_manifest,
            "latest_event": latest_event_summary,
            "seed_summaries": seed_summaries,
            "required_seeds": required_seeds,
            "missing_required_seeds": missing_required_seeds,
            "progress_event_seeds_present": progress_event_seeds_present,
            "missing_progress_event_seeds": missing_progress_event_seeds,
            "quantized_progress_seeds": quantized_progress_seeds,
            "progressed_seeds_missing_target_rows": progressed_seeds_missing_target_rows,
            "stalled_started_seeds": stalled_started_seeds,
            "processed_samples_regressed_seeds": processed_samples_regressed_seeds,
            "reset_to_zero_after_progress_seeds": reset_to_zero_after_progress_seeds,
            "partial_recovery_after_regression_seeds": partial_recovery_after_regression_seeds,
            "contract": contract_view,
            "observed_blockers": list(dict.fromkeys(observed_blockers)),
        }
        assessed_rows.append(assessed)
        overall_statuses.append(observed_status)
        for blocker in assessed["observed_blockers"]:
            if blocker not in overall_blockers:
                overall_blockers.append(f"{experiment_id}:{blocker}")

    if not assessed_rows:
        raise SystemExit("No launch rows matched the requested experiment filter")

    if all(status == "complete" for status in overall_statuses):
        overall_status = "complete"
    elif any(status.endswith("_stalled") or status.startswith("stalled_") for status in overall_statuses):
        overall_status = "stalled"
    else:
        overall_status = "incomplete"

    return {
        "checked_at": now.isoformat(),
        "manifest_json": str(manifest_payload.get("manifest_json") or ""),
        "accuracy_evidence_tier": str(
            manifest_payload.get("accuracy_evidence_tier") or ""
        ).strip(),
        "overall_status": overall_status,
        "overall_blockers": overall_blockers,
        "experiments": assessed_rows,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_markdown(
    path: Path,
    payload: dict[str, Any],
    *,
    accuracy_contract_csv: Path | None,
) -> None:
    lines = [
        "# True SC Accuracy Launch Status",
        "",
        f"- checked_at: `{payload['checked_at']}`",
        f"- manifest_json: `{payload['manifest_json']}`",
        f"- accuracy_evidence_tier: `{payload['accuracy_evidence_tier']}`",
        f"- overall_status: `{payload['overall_status']}`",
        f"- accuracy_contract_csv: `{accuracy_contract_csv}`" if accuracy_contract_csv else "- accuracy_contract_csv: ``",
    ]
    overall_blockers = list(payload.get("overall_blockers") or [])
    if overall_blockers:
        lines.extend(["", "## Overall Blockers", ""])
        lines.extend(f"- `{item}`" for item in overall_blockers)

    for experiment in payload.get("experiments") or []:
        raw_accuracy = dict(experiment.get("raw_accuracy") or {})
        annotated_accuracy = dict(experiment.get("annotated_accuracy") or {})
        latest_event = dict(experiment.get("latest_event") or {})
        contract = dict(experiment.get("contract") or {})
        blockers = list(experiment.get("observed_blockers") or [])
        seed_summaries = list(experiment.get("seed_summaries") or [])
        lines.extend(
            [
                "",
                f"## {experiment['experiment_id']}",
                "",
                f"- observed_status: `{experiment['observed_status']}`",
                f"- run_id: `{experiment['run_id']}`",
                f"- manifest_analysis_grade_ready: `{experiment['manifest_analysis_grade_ready']}`",
                f"- manifest_runtime_launch_ready: `{experiment['manifest_runtime_launch_ready']}`",
                f"- raw_accuracy_exists: `{raw_accuracy.get('exists')}`",
                f"- raw_accuracy_row_count: `{raw_accuracy.get('row_count')}`",
                f"- baseline_row_count: `{raw_accuracy.get('baseline_row_count')}`",
                f"- target_row_count: `{raw_accuracy.get('target_row_count')}`",
                f"- target_seeds_present: `{raw_accuracy.get('target_seeds_present')}`",
                f"- target_measurement_windows: `{raw_accuracy.get('target_measurement_windows')}`",
                f"- target_truth_classes: `{raw_accuracy.get('target_truth_classes')}`",
                f"- annotated_accuracy_exists: `{annotated_accuracy.get('exists')}`",
                f"- annotated_accuracy_row_count: `{annotated_accuracy.get('row_count')}`",
                f"- progress_event_seeds_present: `{experiment.get('progress_event_seeds_present')}`",
                f"- missing_progress_event_seeds: `{experiment.get('missing_progress_event_seeds')}`",
                f"- quantized_progress_seeds: `{experiment.get('quantized_progress_seeds')}`",
                f"- progressed_seeds_missing_target_rows: `{experiment.get('progressed_seeds_missing_target_rows')}`",
                f"- stalled_started_seeds: `{experiment.get('stalled_started_seeds')}`",
                f"- processed_samples_regressed_seeds: `{experiment.get('processed_samples_regressed_seeds')}`",
                f"- reset_to_zero_after_progress_seeds: `{experiment.get('reset_to_zero_after_progress_seeds')}`",
                f"- partial_recovery_after_regression_seeds: `{experiment.get('partial_recovery_after_regression_seeds')}`",
                f"- latest_event_timestamp: `{latest_event.get('latest_event_timestamp')}`",
                f"- latest_event: `{latest_event.get('latest_event')}`",
                f"- latest_pass_kind: `{latest_event.get('latest_pass_kind')}`",
                f"- latest_processed_samples: `{latest_event.get('latest_processed_samples')}`",
                f"- latest_total_samples: `{latest_event.get('latest_total_samples')}`",
                f"- latest_runtime_stage: `{latest_event.get('latest_runtime_stage')}`",
                f"- latest_runtime_detail: `{latest_event.get('latest_runtime_detail')}`",
                f"- latest_event_age_seconds: `{latest_event.get('latest_event_age_seconds')}`",
                f"- quantized_event_count: `{latest_event.get('quantized_event_count')}`",
                f"- max_quantized_processed_samples: `{latest_event.get('max_quantized_processed_samples')}`",
                f"- max_quantized_total_samples: `{latest_event.get('max_quantized_total_samples')}`",
                f"- max_quantized_completion_ratio: `{latest_event.get('max_quantized_completion_ratio')}`",
                f"- max_quantized_completion_percent: `{latest_event.get('max_quantized_completion_percent')}`",
                f"- max_quantized_processed_timestamp: `{latest_event.get('max_quantized_processed_timestamp')}`",
                f"- first_nonzero_processed_samples: `{latest_event.get('first_nonzero_processed_samples')}`",
                f"- first_nonzero_processed_timestamp: `{latest_event.get('first_nonzero_processed_timestamp')}`",
                f"- last_nonzero_processed_samples: `{latest_event.get('last_nonzero_processed_samples')}`",
                f"- last_nonzero_processed_timestamp: `{latest_event.get('last_nonzero_processed_timestamp')}`",
                f"- nonzero_progress_duration_seconds: `{latest_event.get('nonzero_progress_duration_seconds')}`",
                f"- processed_samples_regression_count: `{latest_event.get('processed_samples_regression_count')}`",
                f"- first_processed_samples_regression_timestamp: `{latest_event.get('first_processed_samples_regression_timestamp')}`",
                f"- first_processed_samples_regression_from: `{latest_event.get('first_processed_samples_regression_from')}`",
                f"- first_processed_samples_regression_to: `{latest_event.get('first_processed_samples_regression_to')}`",
                f"- latest_processed_samples_below_peak: `{latest_event.get('latest_processed_samples_below_peak')}`",
                f"- latest_processed_samples_gap_from_peak: `{latest_event.get('latest_processed_samples_gap_from_peak')}`",
                f"- peak_quantized_progress_retained_ratio: `{latest_event.get('peak_quantized_progress_retained_ratio')}`",
                f"- peak_quantized_progress_retained_percent: `{latest_event.get('peak_quantized_progress_retained_percent')}`",
                f"- latest_quantized_completion_ratio: `{latest_event.get('latest_quantized_completion_ratio')}`",
                f"- latest_quantized_completion_percent: `{latest_event.get('latest_quantized_completion_percent')}`",
                f"- latest_processed_samples_reset_to_zero_after_progress: `{latest_event.get('latest_processed_samples_reset_to_zero_after_progress')}`",
                f"- zero_after_progress_duration_seconds: `{latest_event.get('zero_after_progress_duration_seconds')}`",
            ]
        )
        if blockers:
            lines.extend(["", "### Observed Blockers", ""])
            lines.extend(f"- `{item}`" for item in blockers)
        if seed_summaries:
            lines.extend(["", "### Seed Progress", ""])
            for seed_summary in seed_summaries:
                latest_seed_event = dict(seed_summary.get("latest_event_summary") or {})
                lines.append(
                    f"- seed `{seed_summary.get('planned_seed')}`: "
                    f"seed_status=`{seed_summary.get('seed_status')}` "
                    f"event_file_exists=`{seed_summary.get('event_file_exists')}` "
                    f"quantized_event_count=`{latest_seed_event.get('quantized_event_count')}` "
                    f"latest_event=`{latest_seed_event.get('latest_event')}` "
                    f"latest_event_timestamp=`{latest_seed_event.get('latest_event_timestamp')}` "
                    f"latest_processed_samples=`{latest_seed_event.get('latest_processed_samples')}` "
                    f"latest_total_samples=`{latest_seed_event.get('latest_total_samples')}` "
                    f"max_quantized_processed_samples=`{latest_seed_event.get('max_quantized_processed_samples')}` "
                    f"max_quantized_total_samples=`{latest_seed_event.get('max_quantized_total_samples')}` "
                    f"max_quantized_completion_percent=`{latest_seed_event.get('max_quantized_completion_percent')}` "
                    f"max_quantized_processed_timestamp=`{latest_seed_event.get('max_quantized_processed_timestamp')}` "
                    f"last_nonzero_processed_samples=`{latest_seed_event.get('last_nonzero_processed_samples')}` "
                    f"last_nonzero_processed_timestamp=`{latest_seed_event.get('last_nonzero_processed_timestamp')}` "
                    f"nonzero_progress_duration_seconds=`{latest_seed_event.get('nonzero_progress_duration_seconds')}` "
                    f"processed_samples_regression_count=`{latest_seed_event.get('processed_samples_regression_count')}` "
                    f"first_processed_samples_regression_timestamp=`{latest_seed_event.get('first_processed_samples_regression_timestamp')}` "
                    f"latest_processed_samples_gap_from_peak=`{latest_seed_event.get('latest_processed_samples_gap_from_peak')}` "
                    f"peak_quantized_progress_retained_percent=`{latest_seed_event.get('peak_quantized_progress_retained_percent')}` "
                    f"latest_quantized_completion_percent=`{latest_seed_event.get('latest_quantized_completion_percent')}` "
                    f"zero_after_progress_duration_seconds=`{latest_seed_event.get('zero_after_progress_duration_seconds')}` "
                    f"latest_processed_samples_reset_to_zero_after_progress=`{latest_seed_event.get('latest_processed_samples_reset_to_zero_after_progress')}` "
                    f"latest_processed_samples_partial_recovery_after_regression=`{latest_seed_event.get('latest_processed_samples_partial_recovery_after_regression')}`"
                )
        manifest_blockers = list(experiment.get("manifest_analysis_grade_blockers") or [])
        if manifest_blockers:
            lines.extend(["", "### Manifest Blockers", ""])
            lines.extend(f"- `{item}`" for item in manifest_blockers)
        if contract:
            lines.extend(["", "### Contract Rows", ""])
            for row_role, row in contract.items():
                lines.append(
                    f"- {row_role}: row_status=`{row.get('row_status')}` "
                    f"selected_row_id=`{row.get('selected_row_id')}` "
                    f"selected_truth_class=`{row.get('selected_truth_class')}` "
                    f"context_mismatches=`{row.get('context_mismatches')}`"
                )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Check observed true-SC launch status from a prepared accuracy launch manifest.",
    )
    parser.add_argument("--manifest_json", required=True)
    parser.add_argument("--accuracy_contract_csv", default=None)
    parser.add_argument(
        "--experiments",
        default=None,
        help="Comma-separated experiment ids to inspect, for example E1,E6",
    )
    parser.add_argument(
        "--checked_at",
        default=None,
        help="Override the check timestamp using ISO-8601 local time.",
    )
    parser.add_argument("--output_json", default=None)
    parser.add_argument("--output_md", default=None)
    return parser


def main() -> int:
    parser = build_argparser()
    args = parser.parse_args()

    now = (
        datetime.now().astimezone()
        if not args.checked_at
        else _parse_timestamp(args.checked_at, default_tz=datetime.now().astimezone().tzinfo)
    )
    if now is None:
        raise SystemExit(f"Could not parse --checked_at={args.checked_at!r}")

    manifest_path = Path(args.manifest_json)
    manifest_payload = _load_json(manifest_path)
    contract_path = Path(args.accuracy_contract_csv) if args.accuracy_contract_csv else None
    contract_index = _build_contract_index(contract_path)
    experiments_filter = (
        {
            token.strip().upper()
            for token in str(args.experiments or "").split(",")
            if token.strip()
        }
        if args.experiments
        else None
    )
    payload = assess_launch_status(
        manifest_payload,
        contract_index=contract_index,
        experiments_filter=experiments_filter,
        now=now,
    )
    if args.output_json:
        _write_json(Path(args.output_json), payload)
    if args.output_md:
        _write_markdown(
            Path(args.output_md),
            payload,
            accuracy_contract_csv=contract_path,
        )
    print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
