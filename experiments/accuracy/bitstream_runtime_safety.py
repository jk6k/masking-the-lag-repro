"""Shared runtime safety helpers for bounded bitstream evaluation."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any

DEFAULT_MLX_EVAL_BATCH_SIZE = 64
LIMITED_LINEAR_ATTENTION_PILOT_TARGET_COUNT = 2
ESTIMATED_BYTES_PER_ACTIVE_TARGET_SAMPLE_AT_STREAM64 = 6 * 1024 * 1024
BITSTREAM_RUNTIME_WORKING_SET_LIMIT_BYTES = 2 * 1024 * 1024 * 1024
FULL_SURFACE_RUNTIME_VALIDATION_RELATIVE_ROOT = Path(
    "experiments/results/report_data/true_sc_all_target_postrepair_revalidation_20260420_run11/e0"
)
FULL_SURFACE_RUNTIME_VALIDATION_PROGRESS_EVENT_BASENAME = (
    "20260415_sc_default_dark_launch_candidate_mobilevit_s_s0.jsonl"
)
FULL_SURFACE_RUNTIME_VALIDATION_SURFACE_SCOPE = (
    "all_target_177_postrepair_revalidation_20260420"
)

_SURFACE_COUNT_PATTERN = re.compile(r"(?:all_target_|staged)(\d+)")


def _normalize_target_keys(keys: list[str]) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for raw_key in keys:
        key = str(raw_key).strip()
        if not key or key in seen:
            continue
        seen.add(key)
        normalized.append(key)
    return normalized


def parse_bitstream_target_module_keys(
    raw: str | None,
) -> tuple[list[str], bool]:
    token = str(raw or "").strip()
    if token in {"", "all", "*"}:
        return [], True
    if not token.startswith("@"):
        return _normalize_target_keys(token.replace("\n", ",").split(",")), False

    path = Path(token[1:]).expanduser()
    if not path.is_file():
        raise SystemExit(f"Bitstream target module key file not found: {path}")
    text = path.read_text(encoding="utf-8")
    parsed_keys: list[str] | None = None
    if path.suffix.lower() == ".json":
        payload = json.loads(text)
        if isinstance(payload, dict):
            raw_keys = payload.get("target_module_keys") or payload.get("keys") or []
            if isinstance(raw_keys, list):
                parsed_keys = [str(item) for item in raw_keys]
        elif isinstance(payload, list):
            parsed_keys = [str(item) for item in payload]
    if parsed_keys is None:
        parsed_keys = text.replace("\n", ",").split(",")
    return _normalize_target_keys(parsed_keys), False


def _runtime_targetable_families() -> set[str]:
    from accuracy.mlx_mobilevit import resolve_bitstream_runtime_family_policy

    policy = resolve_bitstream_runtime_family_policy()
    return {
        str(family)
        for family, entry in policy.items()
        if str((entry or {}).get("coverage_mechanism") or "").strip()
        == "targetable_module_family"
    }


def load_targetable_module_count(
    model_key: str,
    *,
    root_dir: Path | None = None,
) -> int:
    root = Path(root_dir) if root_dir is not None else Path(__file__).resolve().parents[2]
    ops_path = root / "experiments" / "mtl_model" / "ops" / f"ops_{model_key}.json"
    try:
        payload = json.loads(ops_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return 0
    ops = payload.get("ops")
    if not isinstance(ops, list):
        return 0
    targetable_families = _runtime_targetable_families()
    return sum(
        1
        for row in ops
        if isinstance(row, dict)
        and str(row.get("type") or "").strip() in targetable_families
    )


def surface_scope_implies_all_targets(surface_scope: str | None) -> bool:
    scope = str(surface_scope or "").strip().lower()
    return bool(scope) and "all" in scope


def _surface_scope_target_count(surface_scope: str | None) -> int | None:
    scope = str(surface_scope or "").strip().lower()
    if not scope:
        return None
    if scope == "limited_linear_attention_pilot":
        return LIMITED_LINEAR_ATTENTION_PILOT_TARGET_COUNT
    match = _SURFACE_COUNT_PATTERN.search(scope)
    if match:
        return int(match.group(1))
    return None


def estimate_active_target_module_count(
    *,
    model_key: str,
    surface_scope: str | None,
    target_module_keys_raw: str | None,
    root_dir: Path | None = None,
) -> dict[str, Any]:
    target_module_keys, all_target_requested = parse_bitstream_target_module_keys(
        target_module_keys_raw
    )
    targetable_module_count = load_targetable_module_count(
        model_key,
        root_dir=root_dir,
    )
    surface_count = _surface_scope_target_count(surface_scope)
    if target_module_keys:
        active_target_module_count = len(target_module_keys)
        count_source = "explicit_target_module_keys"
    elif surface_count is not None:
        active_target_module_count = int(surface_count)
        count_source = "surface_scope_encoded_count"
    else:
        active_target_module_count = int(targetable_module_count)
        count_source = "all_target_default"
    active_target_module_count = min(
        int(active_target_module_count),
        int(targetable_module_count) if int(targetable_module_count) > 0 else int(active_target_module_count),
    )
    all_target_surface = (
        bool(all_target_requested)
        or surface_scope_implies_all_targets(surface_scope)
        or (
            int(targetable_module_count) > 0
            and int(active_target_module_count) >= int(targetable_module_count)
        )
    )
    return {
        "model_key": str(model_key),
        "surface_scope": str(surface_scope or "").strip(),
        "target_module_keys": list(target_module_keys),
        "all_target_requested": bool(all_target_requested),
        "all_target_surface": bool(all_target_surface),
        "target_count_source": count_source,
        "targetable_module_count": int(targetable_module_count),
        "active_target_module_count": int(active_target_module_count),
    }


def safe_quantized_batch_cap(
    *,
    active_target_module_count: int,
    surface_scope: str | None,
) -> int:
    active_count = max(0, int(active_target_module_count))
    if surface_scope_implies_all_targets(surface_scope) or active_count >= 36:
        return 1
    if active_count >= 16:
        return 8
    if active_count >= 8:
        return 16
    if active_count >= 4:
        return 32
    return DEFAULT_MLX_EVAL_BATCH_SIZE


def estimate_runtime_working_set_bytes(
    *,
    active_target_module_count: int,
    quantized_batch_size: int,
    stream_length: int,
) -> int:
    active_count = max(0, int(active_target_module_count))
    batch_size = max(1, int(quantized_batch_size))
    resolved_stream_length = max(1, int(stream_length))
    bytes_per_active_target_sample = (
        ESTIMATED_BYTES_PER_ACTIVE_TARGET_SAMPLE_AT_STREAM64 * resolved_stream_length
    ) // 64
    return active_count * batch_size * bytes_per_active_target_sample


def default_full_surface_runtime_validation_root(root_dir: Path | None = None) -> Path:
    root = Path(root_dir) if root_dir is not None else Path(__file__).resolve().parents[2]
    return root / FULL_SURFACE_RUNTIME_VALIDATION_RELATIVE_ROOT


def governed_full_surface_runtime_validation_ready(validation_root: Path) -> bool:
    raw_csv = validation_root / "raw_accuracy.csv"
    progress_jsonl = (
        validation_root
        / "progress"
        / "events"
        / FULL_SURFACE_RUNTIME_VALIDATION_PROGRESS_EVENT_BASENAME
    )
    if not raw_csv.exists() or not progress_jsonl.exists():
        return False
    try:
        with raw_csv.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except OSError:
        return False
    quantized_row = next(
        (
            row
            for row in rows
            if str(row.get("baseline") or "").strip().lower() == "false"
            and str(row.get("execution_semantics") or "").strip() == "bitstream"
            and str(row.get("bitstream_surface_scope") or "").strip()
            == FULL_SURFACE_RUNTIME_VALIDATION_SURFACE_SCOPE
        ),
        None,
    )
    if quantized_row is None:
        return False
    try:
        events = [
            json.loads(line)
            for line in progress_jsonl.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, json.JSONDecodeError):
        return False
    terminal_events = {
        str(event.get("event") or "").strip()
        for event in events
        if isinstance(event, dict)
    }
    return "pass_complete" in terminal_events and "command_complete" in terminal_events


def build_bitstream_runtime_guardrail(
    *,
    model_key: str,
    surface_scope: str | None,
    target_module_keys_raw: str | None,
    explicit_eval_batch_size: int | None,
    stream_length: int,
    root_dir: Path | None = None,
) -> dict[str, Any]:
    target_counts = estimate_active_target_module_count(
        model_key=model_key,
        surface_scope=surface_scope,
        target_module_keys_raw=target_module_keys_raw,
        root_dir=root_dir,
    )
    safe_batch_cap = safe_quantized_batch_cap(
        active_target_module_count=int(target_counts["active_target_module_count"]),
        surface_scope=surface_scope,
    )
    explicit_batch = None if explicit_eval_batch_size is None else int(explicit_eval_batch_size)
    estimated_quantized_batch_size = min(
        int(explicit_batch or DEFAULT_MLX_EVAL_BATCH_SIZE),
        int(safe_batch_cap),
    )
    estimated_working_set_bytes = estimate_runtime_working_set_bytes(
        active_target_module_count=int(target_counts["active_target_module_count"]),
        quantized_batch_size=estimated_quantized_batch_size,
        stream_length=int(stream_length),
    )
    return {
        **target_counts,
        "explicit_eval_batch_size": explicit_batch,
        "safe_quantized_batch_cap": int(safe_batch_cap),
        "estimated_quantized_batch_size": int(estimated_quantized_batch_size),
        "stream_length": int(stream_length),
        "estimated_working_set_bytes": int(estimated_working_set_bytes),
        "working_set_limit_bytes": int(BITSTREAM_RUNTIME_WORKING_SET_LIMIT_BYTES),
    }
