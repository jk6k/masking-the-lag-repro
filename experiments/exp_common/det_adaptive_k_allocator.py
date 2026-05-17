"""Minimal DET adaptive-k helpers used by the fuller lane."""

from __future__ import annotations

from typing import Any

from exp_common.det_prefix import compute_prefix_error_stats


def allocation_to_prefix_errors(
    k_by_layer: dict[str, Any],
    *,
    bsl_max: int,
) -> dict[str, float]:
    """Map per-layer K allocations to prefix-error estimates.

    The fuller lane only needs a stable per-layer metadata mapping, so this
    helper reuses the existing prefix-error simulator at the selected K values.
    """

    if not isinstance(k_by_layer, dict):
        return {}

    bounded_by_layer: dict[str, int] = {}
    for key, value in k_by_layer.items():
        try:
            k_value = int(round(float(value)))
        except (TypeError, ValueError):
            continue
        bounded_by_layer[str(key)] = max(1, min(k_value, int(bsl_max)))
    if not bounded_by_layer:
        return {}

    unique_ks = sorted(set(bounded_by_layer.values()))
    stats = compute_prefix_error_stats(
        bsl_max=max(1, int(bsl_max)),
        k_grid=unique_ks,
        enforce_monotonic=False,
    )
    prefix_error_by_k = {
        int(row["k"]): float(row["prefix_error_mean"])
        for row in stats
    }
    return {
        layer: prefix_error_by_k[k_value]
        for layer, k_value in bounded_by_layer.items()
        if k_value in prefix_error_by_k
    }


__all__ = ["allocation_to_prefix_errors"]
