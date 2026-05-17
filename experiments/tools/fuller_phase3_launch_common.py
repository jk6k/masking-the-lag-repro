#!/usr/bin/env python3
"""Shared helpers for FULLER phase3 execution-efficiency and launch tooling."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from repo_python_bootstrap import maybe_reexec_for_module
except ImportError:
    def maybe_reexec_for_module(_module: str, *, anchor: Path | None = None) -> None:
        return None

maybe_reexec_for_module("yaml", anchor=Path(__file__))

import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONTRACT = ROOT / "configs" / "fuller_phase3_execution_efficiency_contract_20260422.yaml"
LANE_ORDER = ["ASTRA", "MESO", "HOPS", "DET", "SPARSE", "PHY", "FULLER"]
QUEUE_STATE_NOT_STARTED = "not_started"
QUEUE_STATE_ACTIVE = "active_serial_queue"
QUEUE_STATE_COMPLETED = "completed"
QUEUE_STATE_HALTED_PREFIX = "halted_on_lane_"
QUEUE_STATE_REPAIR_PAUSED_PREFIX = "paused_for_repair_on_lane_"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _resolve_path(root_dir: Path, raw_path: str | Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return root_dir / path


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise SystemExit(f"Expected YAML mapping in {path}")
    return payload


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise SystemExit(f"Expected JSON object in {path}")
    return payload


def _load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _serialize_csv_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _serialize_csv_value(row.get(key)) for key in fieldnames})


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _parse_json_list(raw: Any, field_name: str) -> list[str]:
    if isinstance(raw, list):
        return [str(item) for item in raw]
    try:
        payload = json.loads(str(raw or "[]"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid JSON list in {field_name}: {raw!r}") from exc
    if not isinstance(payload, list):
        raise SystemExit(f"Expected JSON list in {field_name}")
    return [str(item) for item in payload]


def _ensure_command_list(raw: Any, field_name: str) -> list[str]:
    if isinstance(raw, list):
        return [str(item) for item in raw]
    if isinstance(raw, str) and raw.strip():
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"Invalid JSON command in {field_name}: {raw!r}") from exc
        if not isinstance(payload, list):
            raise SystemExit(f"Expected JSON list for {field_name}")
        return [str(item) for item in payload]
    raise SystemExit(f"Missing command list for {field_name}")


def _command_has_flag(command: list[str], flag: str) -> bool:
    return flag in command


def _command_flag_value(command: list[str], flag: str) -> str | None:
    indexes = [index for index, token in enumerate(command) if token == flag]
    for index in reversed(indexes):
        if index + 1 < len(command):
            candidate = command[index + 1]
            if not candidate.startswith("--"):
                return candidate
    return None


def _drop_flag(command: list[str], flag: str) -> list[str]:
    return [token for token in command if token != flag]


def _drop_flag_with_value(command: list[str], flag: str) -> list[str]:
    updated: list[str] = []
    index = 0
    while index < len(command):
        token = command[index]
        if token == flag:
            index += 1
            if index < len(command) and not command[index].startswith("--"):
                index += 1
            continue
        updated.append(token)
        index += 1
    return updated


def _ensure_flag(command: list[str], flag: str) -> list[str]:
    updated = _drop_flag(command, flag)
    updated.append(flag)
    return updated


def _ensure_flag_value(command: list[str], flag: str, value: str) -> list[str]:
    updated = _drop_flag_with_value(command, flag)
    updated.extend([flag, str(value)])
    return updated


def _wrapped_with(command: list[str], wrapper: list[str]) -> bool:
    return bool(wrapper) and command[: len(wrapper)] == wrapper


def _strip_wrapper(command: list[str], wrapper: list[str]) -> list[str]:
    if _wrapped_with(command, wrapper):
        return command[len(wrapper) :]
    return list(command)


def _ensure_wrapper(command: list[str], wrapper: list[str]) -> list[str]:
    base = _strip_wrapper(command, wrapper)
    return list(wrapper) + base if wrapper else base


def _runtime_health_gate_from_command(command: list[str]) -> dict[str, Any]:
    keys = {
        "--progress_heartbeat_interval_seconds": "progress_heartbeat_interval_seconds",
        "--stall_timeout_seconds": "stall_timeout_seconds",
        "--prelaunch_runtime_smoke_samples": "prelaunch_runtime_smoke_samples",
        "--prelaunch_min_samples_per_hour": "prelaunch_min_samples_per_hour",
        "--prelaunch_max_seconds_per_sample": "prelaunch_max_seconds_per_sample",
        "--pathological_min_samples_per_hour": "pathological_min_samples_per_hour",
        "--pathological_max_seconds_per_sample": "pathological_max_seconds_per_sample",
        "--pathological_max_eta_current_rate_seconds": "pathological_max_eta_current_rate_seconds",
        "--pathological_min_processed_samples": "pathological_min_processed_samples",
        "--pathological_min_elapsed_seconds": "pathological_min_elapsed_seconds",
    }
    gate: dict[str, Any] = {}
    for flag, key in keys.items():
        value = _command_flag_value(command, flag)
        if value is not None:
            gate[key] = value
    return gate


def _build_mutated_command(
    command: list[str],
    *,
    wrapper: list[str],
    required_device: str,
    tuned_workers: int,
    tuned_eval_batch_size: int,
) -> tuple[list[str], list[str]]:
    updated = list(command)
    mutations: list[str] = []
    if _command_has_flag(updated, "--dry_run"):
        updated = _drop_flag(updated, "--dry_run")
        mutations.append("remove:--dry_run")
    current_device = _command_flag_value(updated, "--device")
    if current_device != required_device:
        updated = _ensure_flag_value(updated, "--device", required_device)
        mutations.append(f"set:--device={required_device}")
    current_workers = _command_flag_value(updated, "--workers")
    if current_workers != str(tuned_workers):
        updated = _ensure_flag_value(updated, "--workers", str(tuned_workers))
        mutations.append(f"set:--workers={tuned_workers}")
    current_eval_batch_size = _command_flag_value(updated, "--eval_batch_size")
    if current_eval_batch_size != str(tuned_eval_batch_size):
        updated = _ensure_flag_value(updated, "--eval_batch_size", str(tuned_eval_batch_size))
        mutations.append(f"set:--eval_batch_size={tuned_eval_batch_size}")
    if not _command_has_flag(updated, "--resume"):
        updated = _ensure_flag(updated, "--resume")
        mutations.append("add:--resume")
    if not _wrapped_with(updated, wrapper):
        updated = _ensure_wrapper(updated, wrapper)
        mutations.append(f"wrap:{' '.join(wrapper)}")
    return updated, mutations


def _lane_status_output_root(root_dir: Path) -> Path:
    return root_dir / "experiments" / "results" / "report_data" / "fuller_phase3_serial_launch_status_checks_20260422"


def _lane_status_output_paths(root_dir: Path, variant_id: str) -> tuple[Path, Path]:
    root = _lane_status_output_root(root_dir)
    lane_stub = str(variant_id).strip().lower()
    return (
        root / f"{lane_stub}.json",
        root / f"{lane_stub}.md",
    )
