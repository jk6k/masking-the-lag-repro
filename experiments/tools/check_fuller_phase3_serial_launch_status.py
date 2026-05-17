#!/usr/bin/env python3
"""Report current status for the FULLER phase3 serial launch surface."""

from __future__ import annotations

import argparse
import csv
import json
import shlex
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    from .fuller_phase3_launch_common import (
        DEFAULT_CONTRACT,
        LANE_ORDER,
        ROOT,
        _command_flag_value,
        _ensure_command_list,
        _lane_status_output_paths,
        _load_json,
        _load_yaml,
        _resolve_path,
        _write_json,
        _write_text,
    )
    from .check_bitstream_true_sc_accuracy_launch import assess_launch_status
except ImportError:
    from fuller_phase3_launch_common import (  # type: ignore
        DEFAULT_CONTRACT,
        LANE_ORDER,
        ROOT,
        _command_flag_value,
        _ensure_command_list,
        _lane_status_output_paths,
        _load_json,
        _load_yaml,
        _resolve_path,
        _write_json,
        _write_text,
    )
    from check_bitstream_true_sc_accuracy_launch import assess_launch_status  # type: ignore


def _count_csv_rows(path: Path | None) -> int:
    if path is None or not path.exists() or path.is_dir():
        return 0
    with path.open("r", newline="", encoding="utf-8") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def _seed_status_started(seed_status: str) -> bool:
    return seed_status in {
        "baseline_eval_incomplete",
        "baseline_complete_only",
        "quantized_eval_incomplete",
        "command_complete",
        "command_error",
    } or seed_status.startswith("stalled_")


def _latest_event_age_seconds(progress_root: Path, now: datetime) -> float | None:
    event_paths = sorted(progress_root.glob("*.jsonl"))
    if not event_paths:
        return None
    latest: datetime | None = None
    for path in event_paths:
        with path.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                timestamp = str(payload.get("timestamp") or "").strip()
                if not timestamp:
                    continue
                try:
                    parsed = datetime.fromisoformat(timestamp)
                except ValueError:
                    continue
                if parsed.tzinfo is None:
                    parsed = parsed.astimezone()
                else:
                    parsed = parsed.astimezone()
                if latest is None or parsed >= latest:
                    latest = parsed
    if latest is None:
        return None
    return max((now.astimezone() - latest).total_seconds(), 0.0)


def _astra_manifest_command(manifest: dict[str, Any]) -> list[str]:
    launch_command = manifest.get("launch_command")
    if launch_command:
        return _ensure_command_list(launch_command, "ASTRA.progress_manifest.launch_command")
    jobs = manifest.get("jobs") or []
    if isinstance(jobs, list):
        for job in jobs:
            if not isinstance(job, dict):
                continue
            commands = job.get("commands") or []
            if isinstance(commands, list) and commands:
                first_command = commands[0]
                if isinstance(first_command, list):
                    return [str(item) for item in first_command]
                if isinstance(first_command, str) and first_command.strip():
                    return shlex.split(first_command)
            raw_command = str(job.get("command") or "").strip()
            if raw_command:
                return shlex.split(raw_command)
    return []


def _check_astra_status(row: dict[str, Any], *, now: datetime) -> dict[str, Any]:
    progress_manifest_path = Path(str(row.get("progress_manifest_json") or ""))
    if not progress_manifest_path.exists():
        return {
            "variant_id": "ASTRA",
            "internal_experiment_id": "E0",
            "overall_status": "missing_launch_artifacts",
            "overall_blockers": ["progress_manifest_missing"],
            "command_error_detected": False,
            "data_valid_for_resume": False,
            "repair_halt_eligible": False,
            "repair_stop_recommendation": "",
            "raw_accuracy_exists": False,
            "annotated_accuracy_exists": False,
        }
    manifest = _load_json(progress_manifest_path)
    command = _astra_manifest_command(manifest)
    raw_accuracy_raw = str(
        manifest.get("results_csv") or _command_flag_value(command, "--results_csv") or ""
    ).strip()
    annotated_accuracy_raw = str(
        manifest.get("annotated_results_csv") or _command_flag_value(command, "--annotated_results_csv") or ""
    ).strip()
    progress_root_raw = str(
        manifest.get("progress_root") or _command_flag_value(command, "--progress_root") or progress_manifest_path.parent
    ).strip()
    jobs = manifest.get("jobs") or []
    runtime_health_gate = {}
    if isinstance(jobs, list) and jobs and isinstance(jobs[0], dict):
        runtime_health_gate = dict(jobs[0].get("runtime_health_gate") or {})
    stall_timeout = float(
        runtime_health_gate.get("stall_timeout_seconds")
        or _command_flag_value(command, "--stall_timeout_seconds")
        or "180.0"
    )
    execution_started = bool(manifest.get("execution_started")) or bool(jobs) or bool(command)
    raw_accuracy = Path(raw_accuracy_raw) if raw_accuracy_raw else None
    annotated_accuracy = Path(annotated_accuracy_raw) if annotated_accuracy_raw else None
    progress_root = Path(progress_root_raw) if progress_root_raw else progress_manifest_path.parent
    raw_row_count = _count_csv_rows(raw_accuracy)
    annotated_row_count = _count_csv_rows(annotated_accuracy)
    durable_output_present = raw_row_count > 0 or annotated_row_count > 0
    blockers: list[str] = []
    if not execution_started:
        blockers.append("context_match_incomplete")
        overall_status = "pending_context_repair"
    elif raw_row_count > 0 and annotated_row_count > 0:
        overall_status = "complete"
    else:
        age_seconds = _latest_event_age_seconds(progress_root, now)
        if age_seconds is not None and age_seconds > stall_timeout:
            overall_status = "stalled"
            blockers.append("progress_stalled_past_runtime_threshold")
        else:
            overall_status = "incomplete"
    data_valid_for_resume = bool(progress_manifest_path.exists()) and execution_started and (
        durable_output_present or progress_root.exists()
    )
    repair_halt_eligible = data_valid_for_resume and overall_status != "complete"
    return {
        "variant_id": "ASTRA",
        "internal_experiment_id": "E0",
        "overall_status": overall_status,
        "overall_blockers": blockers,
        "command_error_detected": False,
        "data_valid_for_resume": data_valid_for_resume,
        "repair_halt_eligible": repair_halt_eligible,
        "repair_stop_recommendation": (
            "safe_pause_and_resume_after_repair" if repair_halt_eligible else ""
        ),
        "raw_accuracy_exists": bool(raw_accuracy_raw) and raw_accuracy is not None and raw_accuracy.exists(),
        "raw_accuracy_row_count": raw_row_count,
        "annotated_accuracy_exists": bool(annotated_accuracy_raw) and annotated_accuracy is not None and annotated_accuracy.exists(),
        "annotated_accuracy_row_count": annotated_row_count,
        "progress_manifest_json": str(progress_manifest_path),
    }


def _runtime_smoke_lane_status(
    variant_id: str,
    experiment_id: str,
    experiment: dict[str, Any],
) -> dict[str, Any]:
    raw_accuracy = dict(experiment.get("raw_accuracy") or {})
    annotated_accuracy = dict(experiment.get("annotated_accuracy") or {})
    progress_manifest = dict(experiment.get("progress_manifest") or {})
    seed_summaries = list(experiment.get("seed_summaries") or [])
    any_command_complete = any(
        str(item.get("seed_status") or "").strip() == "command_complete"
        for item in seed_summaries
    )
    any_command_error = any(
        str(item.get("seed_status") or "").strip() == "command_error"
        for item in seed_summaries
    )
    any_stalled = any(
        str(item.get("seed_status") or "").strip().startswith("stalled_")
        for item in seed_summaries
    )
    any_progress_started = any(
        _seed_status_started(str(item.get("seed_status") or "").strip())
        for item in seed_summaries
    )
    durable_output_present = int(raw_accuracy.get("row_count") or 0) > 0
    anomaly_blockers = [
        str(blocker)
        for blocker in (experiment.get("observed_blockers") or [])
        if str(blocker).startswith(
            (
                "progress_stalled_past_runtime_threshold",
                "stalled_started_seeds=",
                "processed_samples_regressed_seeds=",
                "processed_samples_reset_to_zero_after_progress_seeds=",
                "partial_recovery_after_regression_seeds=",
            )
        )
    ]
    blockers: list[str] = []
    if int(raw_accuracy.get("target_row_count") or 0) <= 0:
        blockers.append("missing_current_accuracy_row")
    if not bool(annotated_accuracy.get("exists")):
        blockers.append("annotated_accuracy_missing")
    if not bool(progress_manifest.get("exists")):
        blockers.append("progress_manifest_missing")
    elif not seed_summaries:
        blockers.append("progress_events_missing")
    elif not any_command_complete and not any_stalled:
        blockers.append("runtime_smoke_incomplete")
    for blocker in anomaly_blockers:
        if blocker not in blockers:
            blockers.append(blocker)

    if int(raw_accuracy.get("target_row_count") or 0) > 0 and bool(annotated_accuracy.get("exists")) and any_command_complete:
        overall_status = "complete"
        blockers = []
    elif any_stalled:
        overall_status = "stalled"
        if "progress_stalled_past_runtime_threshold" not in blockers:
            blockers.append("progress_stalled_past_runtime_threshold")
    else:
        overall_status = "incomplete"
    data_valid_for_resume = bool(progress_manifest.get("exists")) and not any_command_error and (
        durable_output_present or any_progress_started
    )
    repair_halt_eligible = data_valid_for_resume and overall_status != "complete"
    return {
        "variant_id": variant_id,
        "internal_experiment_id": experiment_id,
        "overall_status": overall_status,
        "overall_blockers": blockers,
        "command_error_detected": any_command_error,
        "data_valid_for_resume": data_valid_for_resume,
        "repair_halt_eligible": repair_halt_eligible,
        "repair_stop_recommendation": (
            "safe_pause_and_resume_after_repair" if repair_halt_eligible else ""
        ),
        "runtime_payload": experiment,
    }


def _active_queue_stall_grace(queue_status: dict[str, Any], lane: str, now: datetime) -> float | None:
    if str(queue_status.get("queue_state") or "") != "active_serial_queue":
        return None
    if str(queue_status.get("current_lane") or "").strip().upper() != lane:
        return None
    started_at_raw = str(queue_status.get("started_at") or "").strip()
    if not started_at_raw:
        return None
    try:
        started_at = datetime.fromisoformat(started_at_raw)
    except ValueError:
        return None
    if started_at.tzinfo is None:
        started_at = started_at.astimezone()
    else:
        started_at = started_at.astimezone()
    return max((now.astimezone() - started_at).total_seconds(), 0.0)


def _active_queue_started_at(queue_status: dict[str, Any], lane: str) -> datetime | None:
    if str(queue_status.get("queue_state") or "") != "active_serial_queue":
        return None
    if str(queue_status.get("current_lane") or "").strip().upper() != lane:
        return None
    started_at_raw = str(queue_status.get("started_at") or "").strip()
    if not started_at_raw:
        return None
    try:
        started_at = datetime.fromisoformat(started_at_raw)
    except ValueError:
        return None
    if started_at.tzinfo is None:
        return started_at.astimezone()
    return started_at.astimezone()


def _runtime_payload_latest_event_timestamp(runtime_payload: dict[str, Any]) -> datetime | None:
    candidates = [
        str((runtime_payload.get("latest_event") or {}).get("latest_event_timestamp") or "").strip(),
        str(
            ((runtime_payload.get("progress_manifest") or {}).get("latest_event_summary") or {}).get(
                "latest_event_timestamp"
            )
            or ""
        ).strip(),
    ]
    for candidate in candidates:
        if not candidate:
            continue
        try:
            parsed = datetime.fromisoformat(candidate)
        except ValueError:
            continue
        if parsed.tzinfo is None:
            return parsed.astimezone()
        return parsed.astimezone()
    return None


def check_fuller_phase3_serial_launch_status(
    contract_path: Path = DEFAULT_CONTRACT,
    *,
    variant_id: str | None = None,
    root_dir: Path = ROOT,
    checked_at: datetime | None = None,
) -> dict[str, Any]:
    resolved_contract_path = contract_path if contract_path.is_absolute() else root_dir / contract_path
    contract = _load_yaml(resolved_contract_path)
    sources = contract.get("sources") or {}
    outputs = contract.get("outputs") or {}
    now = checked_at or datetime.now().astimezone()

    execution_packet_payload = _load_json(_resolve_path(root_dir, sources["phase3_execution_packet_json"]))
    execution_rows = execution_packet_payload.get("rows")
    if not isinstance(execution_rows, list):
        raise SystemExit("Phase3 execution packet JSON must expose rows")
    execution_lookup = {
        str(row.get("variant_id") or "").strip().upper(): row
        for row in execution_rows
        if isinstance(row, dict)
    }
    requested_variants = [variant_id.upper()] if variant_id else LANE_ORDER
    launch_manifest_payload = _load_json(_resolve_path(root_dir, sources["phase1_runtime_smoke_launch_manifest_json"]))
    queue_status_path = _resolve_path(root_dir, outputs["serial_launch_status_json"])
    queue_status = _load_json(queue_status_path) if queue_status_path.exists() else {}

    lane_rows: list[dict[str, Any]] = []
    overall_blockers: list[str] = []
    for lane in requested_variants:
        if lane not in execution_lookup:
            raise SystemExit(f"Phase3 execution packet missing lane {lane}")
        if lane == "ASTRA":
            lane_payload = _check_astra_status(execution_lookup[lane], now=now)
        else:
            runtime_payload = assess_launch_status(
                launch_manifest_payload,
                contract_index={},
                experiments_filter={str(execution_lookup[lane].get("internal_experiment_id") or "").strip().upper()},
                now=now,
            )
            experiments = runtime_payload.get("experiments") or []
            if len(experiments) != 1:
                raise SystemExit(f"Expected exactly one runtime launch status row for {lane}")
            experiment = experiments[0]
            lane_payload = _runtime_smoke_lane_status(
                lane,
                str(execution_lookup[lane].get("internal_experiment_id") or "").strip().upper(),
                experiment,
            )
            queue_started_at = _active_queue_started_at(queue_status, lane)
            latest_runtime_event = _runtime_payload_latest_event_timestamp(experiment)
            if queue_started_at is not None and (
                latest_runtime_event is None or latest_runtime_event < queue_started_at
            ):
                blockers = [
                    str(item)
                    for item in (lane_payload.get("overall_blockers") or [])
                    if str(item)
                    not in {
                        "progress_stalled_past_runtime_threshold",
                        "processed_samples_regressed_seeds=[0]",
                    }
                    and not str(item).startswith(
                        (
                            "processed_samples_regressed_seeds=",
                            "processed_samples_reset_to_zero_after_progress_seeds=",
                            "partial_recovery_after_regression_seeds=",
                            "stalled_started_seeds=",
                        )
                    )
                ]
                blockers.append("active_queue_restart_grace")
                lane_payload["overall_status"] = "incomplete"
                lane_payload["overall_blockers"] = blockers
                lane_payload["command_error_detected"] = False
                lane_payload["data_valid_for_resume"] = True
                lane_payload["repair_halt_eligible"] = True
        grace_age_seconds = _active_queue_stall_grace(queue_status, lane, now)
        if lane_payload.get("overall_status") == "stalled" and grace_age_seconds is not None:
            runtime_payload = lane_payload.get("runtime_payload") or {}
            runtime_health_gate = dict(runtime_payload.get("runtime_health_gate") or {})
            stall_timeout_seconds = float(runtime_health_gate.get("stall_timeout_seconds") or 180.0)
            if grace_age_seconds < stall_timeout_seconds:
                blockers = [
                    str(item)
                    for item in (lane_payload.get("overall_blockers") or [])
                    if str(item) != "progress_stalled_past_runtime_threshold"
                ]
                blockers.append("active_queue_restart_grace")
                lane_payload["overall_status"] = "incomplete"
                lane_payload["overall_blockers"] = blockers
        lane_rows.append(lane_payload)
        for blocker in lane_payload.get("overall_blockers") or []:
            if blocker not in overall_blockers:
                overall_blockers.append(str(blocker))

    if all(str(item.get("overall_status") or "") == "complete" for item in lane_rows):
        overall_status = "complete"
    elif any(str(item.get("overall_status") or "") == "stalled" for item in lane_rows):
        overall_status = "stalled"
    elif any(bool(item.get("command_error_detected")) for item in lane_rows):
        overall_status = "command_error"
    else:
        overall_status = "incomplete"

    return {
        "checked_at": now.isoformat(),
        "contract_path": str(resolved_contract_path.resolve()),
        "variant_ids": requested_variants,
        "overall_status": overall_status,
        "overall_blockers": overall_blockers,
        "lanes": lane_rows,
        "queue_status": queue_status,
    }


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# FULLER Phase3 Serial Launch Status",
        "",
        f"- checked_at: `{payload['checked_at']}`",
        f"- overall_status: `{payload['overall_status']}`",
        f"- variant_ids: `{payload['variant_ids']}`",
    ]
    queue_status = payload.get("queue_status") or {}
    if queue_status:
        lines.extend(
            [
                f"- queue_state: `{queue_status.get('queue_state')}`",
                f"- current_lane: `{queue_status.get('current_lane')}`",
                f"- last_completed_lane: `{queue_status.get('last_completed_lane')}`",
                f"- halt_reason: `{queue_status.get('halt_reason')}`",
            ]
        )
    if payload.get("overall_blockers"):
        lines.extend(["", "## Overall Blockers", ""])
        lines.extend(f"- `{item}`" for item in payload["overall_blockers"])
    lines.extend(["", "## Lanes", ""])
    for lane in payload.get("lanes") or []:
        lines.append(
            f"- `{lane['variant_id']}`: overall_status=`{lane['overall_status']}` blockers=`{lane.get('overall_blockers')}` repair_halt_eligible=`{lane.get('repair_halt_eligible')}` data_valid_for_resume=`{lane.get('data_valid_for_resume')}`"
        )
    _write_text(path, "\n".join(lines) + "\n")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Check per-lane or aggregated status for the FULLER phase3 serial launch queue.",
    )
    parser.add_argument("--contract", default=str(DEFAULT_CONTRACT))
    parser.add_argument("--variant_id", default=None)
    parser.add_argument("--output_json", default=None)
    parser.add_argument("--output_md", default=None)
    return parser


def main() -> int:
    args = build_argparser().parse_args()
    payload = check_fuller_phase3_serial_launch_status(
        Path(args.contract),
        variant_id=args.variant_id,
    )
    output_json = Path(args.output_json) if args.output_json else None
    output_md = Path(args.output_md) if args.output_md else None
    if args.variant_id and output_json is None and output_md is None:
        lane_json, lane_md = _lane_status_output_paths(ROOT, args.variant_id)
        output_json = lane_json
        output_md = lane_md
    if output_json:
        _write_json(output_json, payload)
    if output_md:
        _write_markdown(output_md, payload)
    print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
