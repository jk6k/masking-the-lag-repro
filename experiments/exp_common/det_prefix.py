"""Shared helpers for DET prefix-error simulation.

The simulation uses a 1D low-discrepancy stream (van der Corput base-2)
to estimate prefix truncation error statistics for stochastic bitstreams.
"""

from __future__ import annotations

import math
from typing import Any


def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    data = sorted(values)
    idx = max(0, min(len(data) - 1, int(math.ceil(0.95 * len(data)) - 1)))
    return data[idx]


def vdc(index: int, base: int = 2) -> float:
    """Van der Corput radical-inverse sequence in [0, 1)."""
    n = max(0, int(index))
    denom = 1.0
    value = 0.0
    while n > 0:
        n, remainder = divmod(n, base)
        denom *= base
        value += remainder / denom
    return value


def compute_prefix_error_stats(
    *,
    bsl_max: int,
    k_grid: list[int],
    num_prob_points: int = 129,
    p_min: float = 1e-3,
    p_max: float = 1.0 - 1e-3,
    det_mode: str | None = None,
    phase_shift: int = 0,
    scramble_seed: int = 0,
    enforce_monotonic: bool = False,
) -> list[dict[str, Any]]:
    """Compute mean/P95 absolute prefix error by explicit LD simulation."""
    bsl_max = max(1, int(bsl_max))
    k_grid = sorted({max(1, min(int(k), bsl_max)) for k in k_grid})
    if not k_grid:
        k_grid = [bsl_max]

    num_prob_points = max(3, int(num_prob_points))
    p_min = max(0.0, min(float(p_min), 1.0))
    p_max = max(0.0, min(float(p_max), 1.0))
    if p_max <= p_min:
        p_min, p_max = 1e-3, 1.0 - 1e-3

    ld_points = [vdc(i, base=2) for i in range(1, bsl_max + 1)]
    if num_prob_points == 1:
        p_values = [0.5]
    else:
        step = (p_max - p_min) / (num_prob_points - 1)
        p_values = [p_min + i * step for i in range(num_prob_points)]

    err_by_k: dict[int, list[float]] = {k: [] for k in k_grid}
    for p in p_values:
        bits = [1.0 if u < p else 0.0 for u in ld_points]
        prefix_sum = 0.0
        cumulative: list[float] = []
        for bit in bits:
            prefix_sum += bit
            cumulative.append(prefix_sum)

        full_est = cumulative[-1] / bsl_max
        for k in k_grid:
            prefix_est = cumulative[k - 1] / k
            err_by_k[k].append(abs(prefix_est - full_est))

    raw_mean_by_k: dict[int, float] = {}
    raw_p95_by_k: dict[int, float] = {}
    for k in k_grid:
        raw_mean_by_k[k] = sum(err_by_k[k]) / len(err_by_k[k]) if err_by_k[k] else 0.0
        raw_p95_by_k[k] = _p95(err_by_k[k])

    smoothed_mean_by_k: dict[int, float] = {}
    smoothed_p95_by_k: dict[int, float] = {}
    previous_smoothed_mean: float | None = None
    previous_smoothed_p95: float | None = None
    for k in k_grid:
        raw_mean = raw_mean_by_k[k]
        raw_p95 = raw_p95_by_k[k]
        if enforce_monotonic and previous_smoothed_mean is not None:
            smoothed_mean = min(previous_smoothed_mean, raw_mean)
            smoothed_p95 = min(previous_smoothed_p95, raw_p95) if previous_smoothed_p95 is not None else raw_p95
        else:
            smoothed_mean = raw_mean
            smoothed_p95 = raw_p95
        smoothed_mean_by_k[k] = smoothed_mean
        smoothed_p95_by_k[k] = smoothed_p95
        previous_smoothed_mean = smoothed_mean
        previous_smoothed_p95 = smoothed_p95

    baseline = max(1e-12, raw_mean_by_k[k_grid[0]])
    results: list[dict[str, Any]] = []
    for k in k_grid:
        mean_err_raw = raw_mean_by_k[k]
        p95_err_raw = raw_p95_by_k[k]
        mean_err_smoothed = smoothed_mean_by_k[k]
        p95_err_smoothed = smoothed_p95_by_k[k]
        mean_err = mean_err_raw
        p95_err = p95_err_raw
        energy_saved_pct = max(0.0, (1.0 - (k / bsl_max)) * 100.0)
        results.append(
            {
                "k": k,
                "bsl_max": bsl_max,
                "prefix_error_mean": round(mean_err, 8),
                "prefix_error_p95": round(p95_err, 8),
                "prefix_error_mean_raw": round(mean_err_raw, 8),
                "prefix_error_p95_raw": round(p95_err_raw, 8),
                "prefix_error_mean_smoothed": round(mean_err_smoothed, 8),
                "prefix_error_p95_smoothed": round(p95_err_smoothed, 8),
                "relative_error_mean": round(mean_err_raw / baseline, 6),
                "relative_error_mean_raw": round(mean_err_raw / baseline, 6),
                "relative_error_mean_smoothed": round(mean_err_smoothed / baseline, 6),
                "energy_saved_pct": round(energy_saved_pct, 2),
                "num_prob_points": num_prob_points,
            }
        )
    return results
