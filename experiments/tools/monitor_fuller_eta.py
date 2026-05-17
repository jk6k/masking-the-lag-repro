#!/usr/bin/env python3
"""Read-only ETA monitor for fuller collection noise evaluation progress."""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any


def _float(value: Any, default: float | None = None) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    events: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                events.append(payload)
    return events


def _flatten_instrumented_commands(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    commands: list[dict[str, Any]] = []
    for job_index, job in enumerate(list(manifest.get("jobs") or [])):
        progress_jsonls = list(job.get("progress_jsonls") or [])
        if not progress_jsonls:
            continue
        pass_counts = list(job.get("planned_pass_count_per_command") or [])
        for command_index, progress_path in enumerate(progress_jsonls):
            planned_pass_count = 0
            if command_index < len(pass_counts):
                planned_pass_count = int(pass_counts[command_index] or 0)
            commands.append(
                {
                    "job_index": job_index,
                    "command_index": command_index,
                    "family_group": str(job.get("family_group") or ""),
                    "step_id": str(job.get("step_id") or ""),
                    "run_id": str(job.get("run_id") or ""),
                    "model": str(job.get("model") or ""),
                    "profile": str(job.get("profile") or ""),
                    "sweep_resolution": str(job.get("sweep_resolution") or ""),
                    "progress_jsonl": Path(str(progress_path)),
                    "planned_pass_count": planned_pass_count,
                }
            )
    return commands


def _command_state(command: dict[str, Any]) -> dict[str, Any]:
    events = _read_jsonl(command["progress_jsonl"])
    if not events:
        return {
            **command,
            "state": "pending",
            "last_event": None,
            "timestamp": None,
            "command_fraction": 0.0,
            "completed_pass_equivalents": 0.0,
            "command_elapsed_seconds": None,
            "eta_command_seconds": None,
            "projected_total_command_seconds": None,
            "pass_index": None,
            "pass_fraction": None,
        }

    last = events[-1]
    completed = str(last.get("event") or "") == "command_complete"
    command_fraction = 1.0 if completed else _float(last.get("command_fraction"), 0.0) or 0.0
    command_elapsed_seconds = _float(last.get("command_elapsed_seconds"), None)
    eta_command_seconds = 0.0 if completed else _float(last.get("eta_command_seconds"), None)
    projected_total_command_seconds = None
    if completed and command_elapsed_seconds is not None:
        projected_total_command_seconds = command_elapsed_seconds
    elif (
        command_elapsed_seconds is not None
        and command_fraction is not None
        and command_fraction > 0.0
    ):
        projected_total_command_seconds = command_elapsed_seconds / command_fraction

    return {
        **command,
        "state": "completed" if completed else "running",
        "last_event": str(last.get("event") or ""),
        "timestamp": last.get("timestamp"),
        "model": str(last.get("model") or command["model"]),
        "profile": str(last.get("profile") or command["profile"]),
        "sweep_resolution": str(last.get("sweep_resolution") or command["sweep_resolution"]),
        "command_fraction": command_fraction,
        "completed_pass_equivalents": float(command["planned_pass_count"]) * float(command_fraction),
        "command_elapsed_seconds": command_elapsed_seconds,
        "eta_command_seconds": eta_command_seconds,
        "projected_total_command_seconds": projected_total_command_seconds,
        "pass_index": int(last["pass_index"]) if last.get("pass_index") not in ("", None) else None,
        "pass_fraction": _float(last.get("pass_fraction"), None),
    }


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / float(len(values))


def summarize_manifest_eta(manifest_json: Path) -> dict[str, Any]:
    manifest = _read_json(manifest_json)
    commands = _flatten_instrumented_commands(manifest)
    if not commands:
        raise SystemExit(f"No instrumented progress metadata found in {manifest_json}")

    states = [_command_state(command) for command in commands]
    per_pass_seconds_by_model: dict[str, list[float]] = defaultdict(list)
    global_per_pass_seconds: list[float] = []
    for state in states:
        projected_total = state.get("projected_total_command_seconds")
        planned_pass_count = int(state.get("planned_pass_count") or 0)
        if projected_total is None or planned_pass_count <= 0:
            continue
        seconds_per_pass = float(projected_total) / float(planned_pass_count)
        global_per_pass_seconds.append(seconds_per_pass)
        if state["model"]:
            per_pass_seconds_by_model[state["model"]].append(seconds_per_pass)

    global_seconds_per_pass = _mean(global_per_pass_seconds)
    total_planned_passes = sum(int(state.get("planned_pass_count") or 0) for state in states)
    completed_pass_equivalents = sum(
        float(state.get("completed_pass_equivalents") or 0.0) for state in states
    )

    remaining_seconds = 0.0
    eta_known = True
    current_command: dict[str, Any] | None = None
    for state in states:
        if state["state"] == "completed":
            continue
        if current_command is None and state["state"] == "running":
            current_command = state
        if state["state"] == "running":
            eta_command_seconds = _float(state.get("eta_command_seconds"), None)
            if eta_command_seconds is not None:
                remaining_seconds += max(0.0, eta_command_seconds)
                continue
            projected_total = _float(state.get("projected_total_command_seconds"), None)
            command_elapsed_seconds = _float(state.get("command_elapsed_seconds"), None)
            if projected_total is not None and command_elapsed_seconds is not None:
                remaining_seconds += max(0.0, projected_total - command_elapsed_seconds)
                continue

        model_seconds_per_pass = _mean(per_pass_seconds_by_model.get(state["model"], []))
        seconds_per_pass = model_seconds_per_pass or global_seconds_per_pass
        planned_pass_count = int(state.get("planned_pass_count") or 0)
        if seconds_per_pass is None or planned_pass_count <= 0:
            eta_known = False
            continue
        estimated_total = float(seconds_per_pass) * float(planned_pass_count)
        if state["state"] == "running":
            remaining_fraction = max(0.0, 1.0 - float(state.get("command_fraction") or 0.0))
            remaining_seconds += estimated_total * remaining_fraction
        else:
            remaining_seconds += estimated_total

    if current_command is None:
        current_command = next((state for state in states if state["state"] == "pending"), None)

    all_completed = all(state["state"] == "completed" for state in states)
    finish_time = None
    if eta_known:
        finish_time = (datetime.now() + timedelta(seconds=remaining_seconds)).isoformat(
            timespec="seconds"
        )

    return {
        "run_tag": str(manifest.get("run_tag") or ""),
        "manifest_json": str(manifest_json),
        "instrumented_command_count": len(states),
        "completed_command_count": sum(1 for state in states if state["state"] == "completed"),
        "running_command_count": sum(1 for state in states if state["state"] == "running"),
        "pending_command_count": sum(1 for state in states if state["state"] == "pending"),
        "total_planned_passes": total_planned_passes,
        "completed_pass_equivalents": completed_pass_equivalents,
        "global_fraction": (
            float(completed_pass_equivalents) / float(total_planned_passes)
            if total_planned_passes > 0
            else 0.0
        ),
        "remaining_seconds": remaining_seconds if eta_known else None,
        "eta_known": eta_known,
        "finish_time": finish_time,
        "all_completed": all_completed,
        "current_command": current_command,
    }


def _format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "unknown"
    seconds = int(round(max(0.0, seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def render_summary(summary: dict[str, Any]) -> str:
    current = summary.get("current_command")
    lines = [
        f"run_tag={summary['run_tag']}",
        (
            "instrumented_commands="
            f"{summary['instrumented_command_count']} completed={summary['completed_command_count']} "
            f"running={summary['running_command_count']} pending={summary['pending_command_count']}"
        ),
        (
            "progress="
            f"{summary['completed_pass_equivalents']:.2f}/{summary['total_planned_passes']} "
            f"({summary['global_fraction'] * 100.0:.1f}%)"
        ),
        f"eta={_format_duration(summary['remaining_seconds'])}",
        f"finish_time={summary['finish_time'] or 'unknown'}",
    ]
    if current is not None:
        lines.append(
            "current="
            f"{current['step_id']} state={current['state']} model={current['model'] or 'unknown'} "
            f"profile={current['profile'] or 'unknown'} pass={current['pass_index'] or 0}/{current['planned_pass_count']}"
        )
    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Monitor ETA for fuller collection noise commands.")
    parser.add_argument("--manifest_json", type=Path, required=True)
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    parser.add_argument("--watch", action="store_true", help="Poll until completion.")
    parser.add_argument("--interval_seconds", type=float, default=30.0)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    while True:
        summary = summarize_manifest_eta(args.manifest_json)
        if args.json:
            print(json.dumps(summary, indent=2, ensure_ascii=False))
        else:
            print(render_summary(summary))
        if not args.watch or summary["all_completed"]:
            return
        time.sleep(max(1.0, float(args.interval_seconds)))
        print("")


if __name__ == "__main__":
    main()
