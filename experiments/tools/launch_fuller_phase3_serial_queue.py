#!/usr/bin/env python3
"""Run the FULLER phase3 serialized single-MPS execution queue."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    from .check_fuller_phase3_execution_efficiency_audit import check_phase3_execution_efficiency_audit
    from .check_fuller_phase3_serial_launch_queue import check_phase3_serial_launch_queue
    from .check_fuller_phase3_serial_launch_status import check_fuller_phase3_serial_launch_status
    from .fuller_phase3_launch_common import (
        DEFAULT_CONTRACT,
        LANE_ORDER,
        QUEUE_STATE_ACTIVE,
        QUEUE_STATE_COMPLETED,
        QUEUE_STATE_HALTED_PREFIX,
        QUEUE_STATE_NOT_STARTED,
        QUEUE_STATE_REPAIR_PAUSED_PREFIX,
        ROOT,
        _load_json,
        _load_yaml,
        _parse_json_list,
        _resolve_path,
        _write_json,
        _write_text,
    )
except ImportError:
    from check_fuller_phase3_execution_efficiency_audit import check_phase3_execution_efficiency_audit  # type: ignore
    from check_fuller_phase3_serial_launch_queue import check_phase3_serial_launch_queue  # type: ignore
    from check_fuller_phase3_serial_launch_status import check_fuller_phase3_serial_launch_status  # type: ignore
    from fuller_phase3_launch_common import (  # type: ignore
        DEFAULT_CONTRACT,
        LANE_ORDER,
        QUEUE_STATE_ACTIVE,
        QUEUE_STATE_COMPLETED,
        QUEUE_STATE_HALTED_PREFIX,
        QUEUE_STATE_NOT_STARTED,
        QUEUE_STATE_REPAIR_PAUSED_PREFIX,
        ROOT,
        _load_json,
        _load_yaml,
        _parse_json_list,
        _resolve_path,
        _write_json,
        _write_text,
    )


def _write_status_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# FULLER Phase3 Execution Status",
        "",
        f"- queue_state: `{payload.get('queue_state')}`",
        f"- current_lane: `{payload.get('current_lane')}`",
        f"- last_completed_lane: `{payload.get('last_completed_lane')}`",
        f"- started_at: `{payload.get('started_at')}`",
        f"- updated_at: `{payload.get('updated_at')}`",
        f"- halt_reason: `{payload.get('halt_reason')}`",
        f"- exit_code: `{payload.get('exit_code')}`",
        f"- completed_lanes: `{payload.get('completed_lanes')}`",
        f"- pending_lanes: `{payload.get('pending_lanes')}`",
    ]
    if payload.get("active_command"):
        lines.append(f"- active_command: `{payload.get('active_command')}`")
    if payload.get("lane_overall_status"):
        lines.append(f"- lane_overall_status: `{payload.get('lane_overall_status')}`")
        lines.append(f"- lane_overall_blockers: `{payload.get('lane_overall_blockers')}`")
        lines.append(f"- lane_data_valid_for_resume: `{payload.get('lane_data_valid_for_resume')}`")
        lines.append(f"- lane_repair_halt_eligible: `{payload.get('lane_repair_halt_eligible')}`")
        lines.append(f"- lane_command_error_detected: `{payload.get('lane_command_error_detected')}`")
        lines.append(f"- lane_checked_at: `{payload.get('lane_checked_at')}`")
    _write_text(path, "\n".join(lines) + "\n")


def _lane_snapshot(lane_status: dict[str, Any] | None) -> dict[str, Any]:
    if not lane_status:
        return {}
    lanes = lane_status.get("lanes")
    if isinstance(lanes, list) and lanes and isinstance(lanes[0], dict):
        return dict(lanes[0])
    return dict(lane_status)


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
    lane_status: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "current_lane": current_lane or "",
        "last_completed_lane": last_completed_lane or "",
        "queue_state": queue_state,
        "started_at": started_at or "",
        "updated_at": datetime.now().astimezone().isoformat(),
        "halt_reason": halt_reason or "",
        "exit_code": exit_code,
        "active_command": active_command or [],
        "completed_lanes": completed_lanes,
        "pending_lanes": pending_lanes,
    }
    lane_snapshot = _lane_snapshot(lane_status)
    if lane_snapshot:
        payload.update(
            {
                "lane_overall_status": str(lane_snapshot.get("overall_status") or ""),
                "lane_overall_blockers": [str(item) for item in (lane_snapshot.get("overall_blockers") or [])],
                "lane_data_valid_for_resume": bool(lane_snapshot.get("data_valid_for_resume")),
                "lane_repair_halt_eligible": bool(lane_snapshot.get("repair_halt_eligible")),
                "lane_command_error_detected": bool(lane_snapshot.get("command_error_detected")),
                "lane_checked_at": str(lane_status.get("checked_at") or ""),
            }
        )
    return payload


def _lane_status(variant_id: str, contract_path: Path, root_dir: Path) -> dict[str, Any]:
    return check_fuller_phase3_serial_launch_status(contract_path, variant_id=variant_id, root_dir=root_dir)


def _lane_should_pause_for_repair(
    lane_status: dict[str, Any],
    *,
    repairable_anomaly_policy: str,
) -> bool:
    if repairable_anomaly_policy != "pause_and_resume":
        return False
    if bool(lane_status.get("command_error_detected")):
        return False
    return bool(lane_status.get("repair_halt_eligible")) and bool(lane_status.get("data_valid_for_resume"))


def _lane_has_repairable_runtime_anomaly(lane_status: dict[str, Any]) -> bool:
    blockers = [str(item) for item in (lane_status.get("overall_blockers") or [])]
    anomaly_tokens = (
        "pathological",
        "progress_stalled_past_runtime_threshold",
        "stalled_started_seeds=",
    )
    return any(any(token in blocker for token in anomaly_tokens) for blocker in blockers)


def _normalized_halt_reason(halt_reason: str, *, repair_pause: bool) -> str:
    if not repair_pause:
        return halt_reason
    repair_map = {
        "stall_timeout": "repairable_stall_timeout",
        "pathological_slow": "repairable_pathological_slow",
        "runtime_anomaly": "repairable_runtime_anomaly",
        "checker_fail": "repairable_checker_pause",
        "command_error": "repairable_command_exit",
    }
    return repair_map.get(halt_reason, halt_reason)


def _queue_state_for_stop(variant_id: str, *, repair_pause: bool) -> str:
    prefix = QUEUE_STATE_REPAIR_PAUSED_PREFIX if repair_pause else QUEUE_STATE_HALTED_PREFIX
    return f"{prefix}{variant_id}"


def _stop_process(process: subprocess.Popen[str]) -> int:
    process.terminate()
    try:
        return process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        if hasattr(process, "kill"):
            process.kill()
        return process.wait(timeout=30)


def _write_status_surfaces(
    *,
    status_json: Path,
    status_md: Path,
    phase3_execution_status_md: Path,
    payload: dict[str, Any],
) -> dict[str, Any]:
    _write_json(status_json, payload)
    _write_status_markdown(status_md, payload)
    _write_status_markdown(phase3_execution_status_md, payload)
    return payload


def _terminal_status(
    *,
    status_json: Path,
    status_md: Path,
    phase3_execution_status_md: Path,
    queue_state: str,
    current_lane: str,
    last_completed_lane: str | None,
    started_at: str,
    halt_reason: str,
    exit_code: int | None,
    active_command: list[str],
    completed_lanes: list[str],
    pending_lanes: list[str],
    lane_status: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return _write_status_surfaces(
        status_json=status_json,
        status_md=status_md,
        phase3_execution_status_md=phase3_execution_status_md,
        payload=_status_payload(
            queue_state=queue_state,
            current_lane=current_lane,
            last_completed_lane=last_completed_lane,
            started_at=started_at,
            halt_reason=halt_reason,
            exit_code=exit_code,
            active_command=active_command,
            completed_lanes=completed_lanes,
            pending_lanes=pending_lanes,
            lane_status=lane_status,
        ),
    )


def launch_phase3_serial_queue(
    contract_path: Path = DEFAULT_CONTRACT,
    *,
    root_dir: Path = ROOT,
    mode: str = "execute",
    start_from: str | None = None,
    poll_interval_seconds: float = 15.0,
    repairable_anomaly_policy: str = "pause_and_resume",
    skip_preflight_checks: bool = False,
) -> dict[str, Any]:
    resolved_contract_path = contract_path if contract_path.is_absolute() else root_dir / contract_path
    contract = _load_yaml(resolved_contract_path)
    outputs = contract.get("outputs") or {}
    queue_json = _resolve_path(root_dir, outputs["serial_launch_queue_json"])
    status_json = _resolve_path(root_dir, outputs["serial_launch_status_json"])
    status_md = _resolve_path(root_dir, outputs["serial_launch_status_md"])
    phase3_execution_status_md = _resolve_path(root_dir, outputs["phase3_execution_status_md"])
    if not skip_preflight_checks:
        check_phase3_execution_efficiency_audit(resolved_contract_path, root_dir=root_dir)
        check_phase3_serial_launch_queue(resolved_contract_path, root_dir=root_dir)
    queue_payload = _load_json(queue_json)
    queue_rows = queue_payload.get("rows")
    if not isinstance(queue_rows, list):
        raise SystemExit("Serial launch queue JSON must expose rows")
    variant_order = [str(row.get("variant_id") or "").strip().upper() for row in queue_rows]
    if variant_order != LANE_ORDER:
        raise SystemExit(f"Serial launch queue order drifted from lane order: {variant_order}")
    start_index = 0
    if start_from:
        start_index = variant_order.index(start_from.upper())
    completed_lanes = variant_order[:start_index]
    pending_lanes = variant_order[start_index:]
    started_at = datetime.now().astimezone().isoformat()
    status = _status_payload(
        queue_state=QUEUE_STATE_NOT_STARTED if mode == "dry_run" else QUEUE_STATE_ACTIVE,
        current_lane=pending_lanes[0] if pending_lanes else "",
        last_completed_lane=completed_lanes[-1] if completed_lanes else "",
        started_at=started_at,
        halt_reason="",
        exit_code=None,
        active_command=[],
        completed_lanes=completed_lanes,
        pending_lanes=pending_lanes,
    )
    _write_status_surfaces(
        status_json=status_json,
        status_md=status_md,
        phase3_execution_status_md=phase3_execution_status_md,
        payload=status,
    )

    for row in queue_rows[start_index:]:
        variant_id = str(row.get("variant_id") or "").strip().upper()
        command = _parse_json_list(row.get("command_json"), f"{variant_id}.command_json")
        pending_lanes = [lane for lane in pending_lanes if lane != variant_id]
        status = _status_payload(
            queue_state=QUEUE_STATE_ACTIVE,
            current_lane=variant_id,
            last_completed_lane=completed_lanes[-1] if completed_lanes else "",
            started_at=started_at,
            halt_reason="",
            exit_code=None,
            active_command=command,
            completed_lanes=completed_lanes,
            pending_lanes=[variant_id] + pending_lanes,
        )
        _write_status_surfaces(
            status_json=status_json,
            status_md=status_md,
            phase3_execution_status_md=phase3_execution_status_md,
            payload=status,
        )
        if mode == "dry_run":
            completed_lanes.append(variant_id)
            continue

        log_path = status_json.parent / f"{variant_id.lower()}_launch.log"
        with log_path.open("a", encoding="utf-8") as log_handle:
            process = subprocess.Popen(
                command,
                cwd=root_dir,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
            halt_reason = ""
            repair_pause = False
            lane_status_for_stop: dict[str, Any] = {}
            try:
                while process.poll() is None:
                    time.sleep(max(poll_interval_seconds, 0.1))
                    lane_status = _lane_status(variant_id, resolved_contract_path, root_dir)
                    status = _status_payload(
                        queue_state=QUEUE_STATE_ACTIVE,
                        current_lane=variant_id,
                        last_completed_lane=completed_lanes[-1] if completed_lanes else "",
                        started_at=started_at,
                        halt_reason="",
                        exit_code=None,
                        active_command=command,
                        completed_lanes=completed_lanes,
                        pending_lanes=[variant_id] + pending_lanes,
                        lane_status=lane_status,
                    )
                    _write_status_surfaces(
                        status_json=status_json,
                        status_md=status_md,
                        phase3_execution_status_md=phase3_execution_status_md,
                        payload=status,
                    )
                    if str(lane_status.get("overall_status") or "") == "stalled":
                        lane_status_for_stop = lane_status
                        halt_reason = "stall_timeout"
                        repair_pause = _lane_should_pause_for_repair(
                            lane_status,
                            repairable_anomaly_policy=repairable_anomaly_policy,
                        )
                        _stop_process(process)
                        break
                    if _lane_has_repairable_runtime_anomaly(lane_status):
                        lane_status_for_stop = lane_status
                        blockers = json.dumps(lane_status.get("overall_blockers") or [], ensure_ascii=False)
                        halt_reason = (
                            "pathological_slow" if "pathological" in blockers else "runtime_anomaly"
                        )
                        repair_pause = _lane_should_pause_for_repair(
                            lane_status,
                            repairable_anomaly_policy=repairable_anomaly_policy,
                        )
                        _stop_process(process)
                        break
            except KeyboardInterrupt:
                halt_reason = "manual_repair_pause"
                lane_status_for_stop = _lane_status(variant_id, resolved_contract_path, root_dir)
                repair_pause = _lane_should_pause_for_repair(
                    lane_status_for_stop,
                    repairable_anomaly_policy=repairable_anomaly_policy,
                )
                _stop_process(process)
            return_code = process.wait()
        if halt_reason:
            return _terminal_status(
                status_json=status_json,
                status_md=status_md,
                phase3_execution_status_md=phase3_execution_status_md,
                queue_state=_queue_state_for_stop(variant_id, repair_pause=repair_pause),
                current_lane=variant_id,
                last_completed_lane=completed_lanes[-1] if completed_lanes else "",
                started_at=started_at,
                halt_reason=_normalized_halt_reason(halt_reason, repair_pause=repair_pause),
                exit_code=return_code,
                active_command=command,
                completed_lanes=completed_lanes,
                pending_lanes=[variant_id] + pending_lanes,
                lane_status=lane_status_for_stop,
            )
        if return_code != 0:
            lane_status = _lane_status(variant_id, resolved_contract_path, root_dir)
            repair_pause = _lane_should_pause_for_repair(
                lane_status,
                repairable_anomaly_policy=repairable_anomaly_policy,
            )
            return _terminal_status(
                status_json=status_json,
                status_md=status_md,
                phase3_execution_status_md=phase3_execution_status_md,
                queue_state=_queue_state_for_stop(variant_id, repair_pause=repair_pause),
                current_lane=variant_id,
                last_completed_lane=completed_lanes[-1] if completed_lanes else "",
                started_at=started_at,
                halt_reason=_normalized_halt_reason("command_error", repair_pause=repair_pause),
                exit_code=return_code,
                active_command=command,
                completed_lanes=completed_lanes,
                pending_lanes=[variant_id] + pending_lanes,
                lane_status=lane_status,
            )

        lane_status = _lane_status(variant_id, resolved_contract_path, root_dir)
        if str(lane_status.get("overall_status") or "") != "complete":
            repair_pause = _lane_should_pause_for_repair(
                lane_status,
                repairable_anomaly_policy=repairable_anomaly_policy,
            )
            halt_reason = "checker_fail"
            if str(lane_status.get("overall_status") or "") == "stalled":
                halt_reason = "stall_timeout"
            return _terminal_status(
                status_json=status_json,
                status_md=status_md,
                phase3_execution_status_md=phase3_execution_status_md,
                queue_state=_queue_state_for_stop(variant_id, repair_pause=repair_pause),
                current_lane=variant_id,
                last_completed_lane=completed_lanes[-1] if completed_lanes else "",
                started_at=started_at,
                halt_reason=_normalized_halt_reason(halt_reason, repair_pause=repair_pause),
                exit_code=0,
                active_command=command,
                completed_lanes=completed_lanes,
                pending_lanes=[variant_id] + pending_lanes,
                lane_status=lane_status,
            )

        completed_lanes.append(variant_id)
        status = _status_payload(
            queue_state=QUEUE_STATE_ACTIVE,
            current_lane="",
            last_completed_lane=variant_id,
            started_at=started_at,
            halt_reason="",
            exit_code=0,
            active_command=[],
            completed_lanes=completed_lanes,
            pending_lanes=pending_lanes,
        )
        _write_status_surfaces(
            status_json=status_json,
            status_md=status_md,
            phase3_execution_status_md=phase3_execution_status_md,
            payload=status,
        )

    final_status = _status_payload(
        queue_state=QUEUE_STATE_COMPLETED if mode == "execute" else QUEUE_STATE_ACTIVE,
        current_lane="",
        last_completed_lane=completed_lanes[-1] if completed_lanes else "",
        started_at=started_at,
        halt_reason="",
        exit_code=0,
        active_command=[],
        completed_lanes=completed_lanes,
        pending_lanes=[],
    )
    return _write_status_surfaces(
        status_json=status_json,
        status_md=status_md,
        phase3_execution_status_md=phase3_execution_status_md,
        payload=final_status,
    )


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Launch the FULLER phase3 serialized single-MPS queue.",
    )
    parser.add_argument("--contract", default=str(DEFAULT_CONTRACT))
    parser.add_argument("--mode", choices=["dry_run", "execute"], default="execute")
    parser.add_argument("--start_from", default=None)
    parser.add_argument("--poll_interval_seconds", type=float, default=15.0)
    parser.add_argument(
        "--repairable_anomaly_policy",
        choices=["pause_and_resume", "hard_fail"],
        default="pause_and_resume",
    )
    return parser


def main() -> int:
    args = build_argparser().parse_args()
    payload = launch_phase3_serial_queue(
        Path(args.contract),
        mode=args.mode,
        start_from=args.start_from,
        poll_interval_seconds=args.poll_interval_seconds,
        repairable_anomaly_policy=args.repairable_anomaly_policy,
    )
    print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
