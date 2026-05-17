#!/usr/bin/env python3
"""Run the FULLER analysis-grade replay queue."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    from .check_fuller_analysis_grade_replay_surface import check_materialization_plan
    from .fuller_experiment_program_common import (
        ANALYSIS_GRADE_LANES,
        DEFAULT_CONTRACT,
        ROOT,
        _analysis_grade_redirected_support_lanes,
        _load_yaml,
        _resolve_path,
        _write_json,
        _write_text,
    )
except ImportError:
    from check_fuller_analysis_grade_replay_surface import check_materialization_plan  # type: ignore
    from fuller_experiment_program_common import (  # type: ignore
        ANALYSIS_GRADE_LANES,
        DEFAULT_CONTRACT,
        ROOT,
        _analysis_grade_redirected_support_lanes,
        _load_yaml,
        _resolve_path,
        _write_json,
        _write_text,
    )

LANE_ORDER = ANALYSIS_GRADE_LANES
QUEUE_STATE_NOT_STARTED = "analysis_grade_not_started"
QUEUE_STATE_ACTIVE = "analysis_grade_active_serial_queue"
QUEUE_STATE_COMPLETED = "analysis_grade_completed"
QUEUE_STATE_HALTED_PREFIX = "analysis_grade_halted_on_lane_"
QUEUE_MODEL = "astra_canonical_paired_then_mainline_quantized_only_reuse"


def _load_plan_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _parse_json_list(raw: str, field_name: str) -> list[str]:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid JSON in {field_name}: {raw!r}") from exc
    if not isinstance(payload, list):
        raise SystemExit(f"Expected JSON list in {field_name}")
    return [str(item) for item in payload]


def _write_status_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# FULLER Analysis-Grade Replay Status",
        "",
        f"- queue_state: `{payload.get('queue_state')}`",
        f"- queue_model: `{payload.get('queue_model')}`",
        f"- current_lane: `{payload.get('current_lane')}`",
        f"- last_completed_lane: `{payload.get('last_completed_lane')}`",
        f"- started_at: `{payload.get('started_at')}`",
        f"- updated_at: `{payload.get('updated_at')}`",
        f"- halt_reason: `{payload.get('halt_reason')}`",
        f"- exit_code: `{payload.get('exit_code')}`",
        f"- completed_lanes: `{payload.get('completed_lanes')}`",
        f"- pending_lanes: `{payload.get('pending_lanes')}`",
        f"- downstream_lane_order: `{payload.get('downstream_lane_order')}`",
        f"- support_family_redirects: `{payload.get('support_family_redirects')}`",
        f"- next_queue_command: `{payload.get('next_queue_command')}`",
    ]
    if payload.get("active_command"):
        lines.append(f"- active_command: `{payload.get('active_command')}`")
    _write_text(path, "\n".join(lines) + "\n")


def _status_payload(
    *,
    queue_state: str,
    current_lane: str | None,
    last_completed_lane: str | None,
    started_at: str | None,
    halt_reason: str | None,
    exit_code: int | None,
    active_command: list[str] | None,
    completed_lanes: list[str],
    pending_lanes: list[str],
    support_family_redirects: dict[str, str],
    next_queue_command: list[str],
) -> dict[str, Any]:
    return {
        "queue_state": queue_state,
        "queue_model": QUEUE_MODEL,
        "current_lane": current_lane or "",
        "last_completed_lane": last_completed_lane or "",
        "started_at": started_at or "",
        "updated_at": datetime.now().astimezone().isoformat(),
        "halt_reason": halt_reason or "",
        "exit_code": exit_code,
        "active_command": active_command or [],
        "completed_lanes": completed_lanes,
        "pending_lanes": pending_lanes,
        "downstream_lane_order": [lane for lane in LANE_ORDER if lane != "ASTRA"],
        "support_family_redirects": support_family_redirects,
        "next_queue_command": next_queue_command,
    }


def _write_status_surfaces(
    *,
    status_json: Path,
    status_md: Path,
    active_status_md: Path,
    payload: dict[str, Any],
) -> None:
    _write_json(status_json, payload)
    _write_status_markdown(status_md, payload)
    _write_status_markdown(active_status_md, payload)


def launch_analysis_grade_replay_queue(
    contract_path: Path = DEFAULT_CONTRACT,
    *,
    root_dir: Path = ROOT,
    mode: str = "execute",
    start_from: str | None = None,
) -> dict[str, Any]:
    resolved_contract = contract_path if contract_path.is_absolute() else root_dir / contract_path
    contract = _load_yaml(resolved_contract)
    outputs = contract.get("outputs") or {}
    plan_csv = _resolve_path(root_dir, outputs["analysis_grade_replay_materialization_plan_csv"])
    gate_csv = _resolve_path(root_dir, outputs["analysis_grade_replay_gate_matrix_csv"])
    status_json = _resolve_path(root_dir, outputs["analysis_grade_replay_status_json"])
    status_md = _resolve_path(root_dir, outputs["analysis_grade_replay_status_md"])
    active_status_md = _resolve_path(root_dir, outputs["analysis_grade_replay_active_status_md"])
    support_family_redirects = _analysis_grade_redirected_support_lanes(
        type("ContractOnlyContext", (), {"contract": contract})()  # type: ignore[arg-type]
    )

    check_materialization_plan(gate_csv=gate_csv, plan_csv=plan_csv)
    plan_rows = _load_plan_rows(plan_csv)
    row_by_lane = {str(row.get("lane_id") or "").strip().upper(): row for row in plan_rows}
    if list(row_by_lane) != LANE_ORDER:
        raise SystemExit(f"Analysis-grade lane order drifted: {list(row_by_lane)}")

    start_index = 0
    if start_from:
        start_index = LANE_ORDER.index(str(start_from).strip().upper())
    completed_lanes = LANE_ORDER[:start_index]
    pending_lanes = LANE_ORDER[start_index:]
    started_at = datetime.now().astimezone().isoformat()
    next_queue_command = (
        [str(ROOT / ".venv311-mps" / "bin" / "python"), str(Path(__file__).resolve()), "--start_from", pending_lanes[0]]
        if pending_lanes
        else []
    )
    _write_status_surfaces(
        status_json=status_json,
        status_md=status_md,
        active_status_md=active_status_md,
        payload=_status_payload(
            queue_state=QUEUE_STATE_NOT_STARTED if mode == "dry_run" else QUEUE_STATE_ACTIVE,
            current_lane=pending_lanes[0] if pending_lanes else "",
            last_completed_lane=completed_lanes[-1] if completed_lanes else "",
            started_at=started_at,
            halt_reason="",
            exit_code=None,
            active_command=[],
            completed_lanes=completed_lanes,
            pending_lanes=pending_lanes,
            support_family_redirects=support_family_redirects,
            next_queue_command=next_queue_command,
        ),
    )

    for lane_id in LANE_ORDER[start_index:]:
        row = row_by_lane[lane_id]
        command = _parse_json_list(str(row.get("command_json") or "[]"), f"{lane_id}.command_json")
        checker_command = _parse_json_list(
            str(row.get("checker_command_json") or "[]"),
            f"{lane_id}.checker_command_json",
        )
        wrapper = _parse_json_list(str(row.get("launch_wrapper_json") or "[]"), f"{lane_id}.launch_wrapper_json")
        active_command = wrapper + command
        pending_lanes = [lane for lane in pending_lanes if lane != lane_id]
        _write_status_surfaces(
            status_json=status_json,
            status_md=status_md,
            active_status_md=active_status_md,
            payload=_status_payload(
                queue_state=QUEUE_STATE_ACTIVE,
                current_lane=lane_id,
                last_completed_lane=completed_lanes[-1] if completed_lanes else "",
                started_at=started_at,
                halt_reason="",
                exit_code=None,
                active_command=active_command,
                completed_lanes=completed_lanes,
                pending_lanes=[lane_id] + pending_lanes,
                support_family_redirects=support_family_redirects,
                next_queue_command=(
                    [str(ROOT / ".venv311-mps" / "bin" / "python"), str(Path(__file__).resolve()), "--start_from", pending_lanes[0]]
                    if pending_lanes
                    else []
                ),
            ),
        )
        if mode == "dry_run":
            completed_lanes.append(lane_id)
            continue

        log_path = status_json.parent / f"{lane_id.lower()}_analysis_grade_launch.log"
        with log_path.open("a", encoding="utf-8") as handle:
            process = subprocess.run(
                active_command,
                cwd=root_dir,
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
        if process.returncode != 0:
            payload = _status_payload(
                queue_state=f"{QUEUE_STATE_HALTED_PREFIX}{lane_id}",
                current_lane=lane_id,
                last_completed_lane=completed_lanes[-1] if completed_lanes else "",
                started_at=started_at,
                halt_reason="command_error",
                exit_code=process.returncode,
                active_command=active_command,
                completed_lanes=completed_lanes,
                pending_lanes=[lane_id] + pending_lanes,
                support_family_redirects=support_family_redirects,
                next_queue_command=(
                    [str(ROOT / ".venv311-mps" / "bin" / "python"), str(Path(__file__).resolve()), "--start_from", lane_id]
                ),
            )
            _write_status_surfaces(
                status_json=status_json,
                status_md=status_md,
                active_status_md=active_status_md,
                payload=payload,
            )
            return payload

        checker = subprocess.run(
            checker_command,
            cwd=root_dir,
            capture_output=True,
            text=True,
        )
        if checker.returncode != 0:
            payload = _status_payload(
                queue_state=f"{QUEUE_STATE_HALTED_PREFIX}{lane_id}",
                current_lane=lane_id,
                last_completed_lane=completed_lanes[-1] if completed_lanes else "",
                started_at=started_at,
                halt_reason="checker_fail",
                exit_code=checker.returncode,
                active_command=checker_command,
                completed_lanes=completed_lanes,
                pending_lanes=[lane_id] + pending_lanes,
                support_family_redirects=support_family_redirects,
                next_queue_command=(
                    [str(ROOT / ".venv311-mps" / "bin" / "python"), str(Path(__file__).resolve()), "--start_from", lane_id]
                ),
            )
            _write_status_surfaces(
                status_json=status_json,
                status_md=status_md,
                active_status_md=active_status_md,
                payload=payload,
            )
            return payload

        completed_lanes.append(lane_id)

    payload = _status_payload(
        queue_state=QUEUE_STATE_COMPLETED,
        current_lane="",
        last_completed_lane=completed_lanes[-1] if completed_lanes else "",
        started_at=started_at,
        halt_reason="",
        exit_code=0,
        active_command=[],
        completed_lanes=completed_lanes,
        pending_lanes=[],
        support_family_redirects=support_family_redirects,
        next_queue_command=[],
    )
    _write_status_surfaces(
        status_json=status_json,
        status_md=status_md,
        active_status_md=active_status_md,
        payload=payload,
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch the FULLER analysis-grade replay queue.")
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--mode", choices=["dry_run", "execute"], default="execute")
    parser.add_argument("--start_from", default=None)
    args = parser.parse_args()
    payload = launch_analysis_grade_replay_queue(
        contract_path=args.contract,
        mode=args.mode,
        start_from=args.start_from,
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
