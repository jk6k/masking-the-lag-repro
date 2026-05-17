#!/usr/bin/env python3
"""Helpers for strict accuracy bundle orchestration."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from exp_common.runtime import (
    apply_torch_device,
    ensure_validated_accuracy_backend,
    resolve_device_preference,
)


def _parse_model_batch_overrides(raw: str | None) -> dict[str, int]:
    if not raw:
        return {}
    overrides: dict[str, int] = {}
    for chunk in raw.split(","):
        item = chunk.strip()
        if not item:
            continue
        if "=" not in item:
            raise SystemExit(
                f"Invalid model batch override '{item}'. Expected model=batch."
            )
        model, value = item.split("=", 1)
        model = model.strip()
        value = value.strip()
        if not model or not value:
            raise SystemExit(
                f"Invalid model batch override '{item}'. Expected model=batch."
            )
        try:
            overrides[model] = int(value)
        except ValueError as exc:
            raise SystemExit(
                f"Invalid batch size '{value}' for model '{model}'. Expected integer."
            ) from exc
    return overrides


def _completed_seeds_by_model(
    rows: Iterable[dict[str, object]],
    required_run_prefixes: Iterable[str],
) -> dict[str, set[int]]:
    required = tuple(required_run_prefixes)
    per_model_seed_runs: dict[str, dict[int, set[str]]] = defaultdict(lambda: defaultdict(set))
    for row in rows:
        model = str(row.get("model", "")).strip()
        seed_raw = row.get("seed")
        run_id = str(row.get("run_id", "")).strip()
        baseline = str(row.get("baseline", "")).strip().lower()
        if not model or seed_raw in (None, "") or not run_id:
            continue
        try:
            seed = int(seed_raw)
        except (TypeError, ValueError):
            continue
        matched_prefix = next((prefix for prefix in required if run_id.startswith(prefix)), None)
        if matched_prefix is None:
            continue
        if baseline not in {"true", "false"}:
            continue
        per_model_seed_runs[model][seed].add(f"{matched_prefix}:{baseline}")

    completed: dict[str, set[int]] = {}
    expected = {f"{prefix}:{baseline}" for prefix in required for baseline in ("true", "false")}
    for model, seed_runs in per_model_seed_runs.items():
        seeds = {seed for seed, observed in seed_runs.items() if expected.issubset(observed)}
        if seeds:
            completed[model] = seeds
    return completed


def _resolve_device(opts, requested_device: str, *, allow_unvalidated_mps: bool = False):
    resolved_device, _note = resolve_device_preference(requested_device)
    ensure_validated_accuracy_backend(
        resolved_device,
        allow_unvalidated_mps=allow_unvalidated_mps,
    )
    apply_torch_device(opts, resolved_device)
    return opts


__all__ = [
    "_completed_seeds_by_model",
    "_parse_model_batch_overrides",
    "_resolve_device",
    "apply_torch_device",
    "ensure_validated_accuracy_backend",
    "resolve_device_preference",
]
