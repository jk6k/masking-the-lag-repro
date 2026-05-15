#!/usr/bin/env python3
"""Evaluate MobileViT accuracy under SUDS ternary weight perturbation — MLX backend.

Applies SUDS decisions to model weights as numpy array modifications in the
exported NPZ file, then runs inference with MLX.  This avoids the MPS tensor
in-place modification bug that affects the PyTorch/cvnets path.

Usage:
  .venv311-mps/bin/python3 experiments/tools/run_suds_eval_mlx.py \
    --imagenet_val <private_imagenet_val> \
    --model mobilevit_s \
    --max_eval_samples 5000 --seed 42
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "experiments"))

from exp_common.suds_decision import apply_suds_decisions, classify_column

SLACK_MANIFEST_DEFAULT = os.path.join(
    _PROJECT_ROOT, "experiments", "results", "runs", "slack_manifest.json"
)


def _strip_model_prefix(slack_name: str, *, model_key: str | None = None) -> str | None:
    """Return the NPZ layer path for a slack key, failing closed on mismatches."""
    if ":" not in slack_name:
        return slack_name
    prefix, layer_path = slack_name.split(":", 1)
    if model_key is not None and prefix != model_key:
        return None
    return layer_path


def _is_depthwise_npz(shape: tuple[int, ...]) -> bool:
    """NPZ conv weights are channels-last (C_out, kH, kW, C_in).
    Depthwise conv has C_in == 1."""
    return len(shape) == 4 and shape[3] == 1


def _resolve_column_dim_npz(shape: tuple[int, ...]) -> int:
    """Return the column dimension index in NPZ (channels-last) format.

    - Depthwise conv (C_out, kH, kW, 1): columns = output ch (dim 0)
    - Pointwise/regular conv (C_out, kH, kW, C_in): columns = input ch (dim 3)
    - Linear (C_out, C_in): columns = input features (dim 1)
    """
    if len(shape) == 4:
        if _is_depthwise_npz(shape):
            return 0
        return 3
    if len(shape) == 2:
        return 1
    raise ValueError(f"Cannot determine column dim for shape={shape}")


def build_slack_to_npz_mapping(
    npz_data: dict[str, np.ndarray],
    raw_slack_data: dict[str, dict[str, Any]],
    *,
    model_key: str | None = None,
) -> dict[str, dict[str, Any]]:
    """Map slack layer names to NPZ array keys.

    Returns dict: slack_name → {
        'npz_key': str,          # key in npz_data
        'column_dim': int,       # 0=depthwise output, 3=conv input, 1=linear input
        'target_size': int,      # number of columns in the weight array
        'num_cols_slack': int,   # original num_cols from slack
        'slack_values': list,    # total_slack_norm from slack
    }
    """
    mapping: dict[str, dict[str, Any]] = {}

    for slack_name, layer_data in raw_slack_data.items():
        if not isinstance(layer_data, dict) or "total_slack_norm" not in layer_data:
            continue

        # Strip model prefix to reach the NPZ key stem.  When model_key is
        # supplied, a transferred manifest fails closed by mapping no layers.
        layer_path = _strip_model_prefix(slack_name, model_key=model_key)
        if layer_path is None:
            continue

        # Skip activation-only entries
        if layer_path.endswith(".attn_scores") or layer_path.endswith(".attn_output"):
            continue

        npz_key = f"{layer_path}.weight"
        if npz_key not in npz_data:
            continue

        shape = npz_data[npz_key].shape
        if len(shape) < 2:
            continue  # skip 1D bias/norm vectors

        try:
            column_dim = _resolve_column_dim_npz(shape)
        except ValueError:
            continue

        target_size = int(shape[column_dim])
        num_cols_slack = layer_data.get("num_cols", len(layer_data["total_slack_norm"]))

        mapping[slack_name] = {
            "npz_key": npz_key,
            "column_dim": column_dim,
            "target_size": target_size,
            "num_cols_slack": num_cols_slack,
            "slack_values": layer_data["total_slack_norm"],
        }

    return mapping


def _resample_decisions(
    slack_values: list[float],
    decisions: list[str],
    target_size: int,
    *,
    tau_low: float | None = None,
    tau_high: float | None = None,
) -> np.ndarray:
    """Resample per-column decisions to match target weight dimension."""
    src_size = len(slack_values)
    if src_size == target_size:
        return np.array(decisions)

    src_arr = np.array(slack_values)
    src_dec = np.array(decisions)

    src_positions = np.linspace(0, 1, src_size)
    tgt_positions = np.linspace(0, 1, target_size)
    resampled_slack = np.interp(tgt_positions, src_positions, src_arr)

    # Prefer the actual SUDS thresholds.  Reconstructing thresholds from the
    # discretized source decisions is unsafe for conservative points with no
    # PRUNE tier: it can invent a narrow tau window and turn resampled columns
    # into false PRUNE decisions.
    if tau_low is None or tau_high is None:
        keep_mask = src_dec == "KEEP"
        prune_mask = src_dec == "PRUNE"
        if keep_mask.any():
            tau_low = float(src_arr[keep_mask].max())
        else:
            tau_low = float(src_arr.min())
        non_prune = ~prune_mask
        if non_prune.any():
            tau_high = float(src_arr[non_prune].max())
        else:
            tau_high = float(src_arr.max())

    eps = 1e-9
    if tau_low > tau_high:
        raise ValueError(f"tau_low ({tau_low}) must be <= tau_high ({tau_high})")

    new_decisions = np.array([
        classify_column(float(s), tau_low - eps, tau_high + eps)
        for s in resampled_slack
    ])
    return new_decisions


def _suds_thresholds(suds_data: dict[str, dict[str, Any]]) -> tuple[float | None, float | None]:
    thresholds = suds_data.get("_thresholds", {})
    try:
        return float(thresholds["tau_low"]), float(thresholds["tau_high"])
    except (KeyError, TypeError, ValueError):
        return None, None


def _column_l1(arr: np.ndarray, column_dim: int) -> np.ndarray:
    axes = tuple(axis for axis in range(arr.ndim) if axis != column_dim)
    return np.sum(np.abs(arr), axis=axes)


def _column_overflow_proxy(arr: np.ndarray, column_dim: int) -> np.ndarray:
    """HyAtten-like signal proxy: high peak/RMS columns are most precision-critical."""
    axes = tuple(axis for axis in range(arr.ndim) if axis != column_dim)
    abs_arr = np.abs(arr)
    peak = np.max(abs_arr, axis=axes)
    rms = np.sqrt(np.mean(np.square(arr), axis=axes))
    return peak * rms


def _add_column_noise(
    arr: np.ndarray,
    column_dim: int,
    col_indices: np.ndarray,
    noise_std: float,
    rng: np.random.RandomState,
) -> None:
    if len(col_indices) == 0:
        return
    if column_dim == 0:
        col_vals = arr[col_indices]
        noise = rng.randn(*col_vals.shape).astype(arr.dtype) * noise_std * np.abs(col_vals)
        arr[col_indices] += noise
    elif column_dim == 3:
        col_vals = arr[:, :, :, col_indices]
        noise = rng.randn(*col_vals.shape).astype(arr.dtype) * noise_std * np.abs(col_vals)
        arr[:, :, :, col_indices] += noise
    elif column_dim == 1:
        col_vals = arr[:, col_indices]
        noise = rng.randn(*col_vals.shape).astype(arr.dtype) * noise_std * np.abs(col_vals)
        arr[:, col_indices] += noise
    else:
        raise ValueError(f"Unsupported column_dim={column_dim}")


def clone_npz_arrays(npz_data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Return an independent array copy for one perturbation condition."""
    return {key: value.copy() for key, value in npz_data.items()}


def apply_suds_to_npz(
    npz_data: dict[str, np.ndarray],
    suds_data: dict[str, dict[str, Any]],
    mapping: dict[str, dict[str, Any]],
    *,
    degrade_noise_std: float = 0.03,
    prune_noise_std: float = 0.25,
    seed: int = 42,
) -> dict[str, Any]:
    """Apply SUDS ternary decisions via weight perturbation (SC precision model).

    In the SC photonic context, KEEP/DEGRADE/PRUNE control ADC bit width per
    compute column, not whether the NN feature channel exists:

      KEEP    — full 8-bit ADC (no noise)
      DEGRADE — 4-bit ADC equivalent (σ ≈ degrade_noise_std × weight_scale)
      PRUNE   — 2-bit ADC equivalent (σ ≈ prune_noise_std × weight_scale)

    All weights are preserved; only precision changes per-column.
    """
    rng = np.random.RandomState(seed)
    tau_low, tau_high = _suds_thresholds(suds_data)
    stats: dict[str, Any] = {
        "total_params_modified": 0,
        "total_pruned_columns": 0,
        "total_degraded_columns": 0,
        "total_kept_columns": 0,
        "per_layer": {},
    }

    for slack_name, map_info in mapping.items():
        npz_key = map_info["npz_key"]
        if npz_key not in npz_data:
            continue

        arr = npz_data[npz_key]
        column_dim = map_info["column_dim"]
        target_size = map_info["target_size"]
        slack_values = map_info["slack_values"]

        layer_suds = suds_data.get(slack_name, {})
        decisions = layer_suds.get("decisions", [])
        if not decisions:
            continue

        resampled = _resample_decisions(
            slack_values,
            decisions,
            target_size,
            tau_low=tau_low,
            tau_high=tau_high,
        )

        prune_mask = resampled == "PRUNE"
        degrade_mask = resampled == "DEGRADE"
        keep_mask = resampled == "KEEP"
        n_prune = int(prune_mask.sum())
        n_degrade = int(degrade_mask.sum())
        n_keep = int(keep_mask.sum())

        # Apply perturbation: PRUNE gets strongest noise, DEGRADE gets moderate
        if n_prune > 0:
            _add_column_noise(arr, column_dim, np.where(prune_mask)[0], prune_noise_std, rng)
        if n_degrade > 0:
            _add_column_noise(arr, column_dim, np.where(degrade_mask)[0], degrade_noise_std, rng)

        stats["total_params_modified"] += 1
        stats["total_pruned_columns"] += n_prune
        stats["total_degraded_columns"] += n_degrade
        stats["total_kept_columns"] += n_keep
        stats["per_layer"][slack_name] = {
            "npz_key": npz_key,
            "target_size": target_size,
            "pruned": n_prune,
            "degraded": n_degrade,
            "kept": n_keep,
        }

    total = stats["total_pruned_columns"] + stats["total_degraded_columns"] + stats["total_kept_columns"]
    stats["prune_ratio"] = stats["total_pruned_columns"] / max(1, total)
    stats["degrade_ratio"] = stats["total_degraded_columns"] / max(1, total)
    return stats


def apply_l1_binary_prune_to_npz(
    npz_data: dict[str, np.ndarray],
    mapping: dict[str, dict[str, Any]],
    sparsity: float,
    *,
    prune_noise_std: float = 0.25,
    seed: int = 42,
) -> dict[str, Any]:
    """Apply E2 L1 KEEP/PRUNE perturbation.

    Lowest-L1 columns are treated as least important and receive coarse
    PRUNE-tier noise. This is the local-importance baseline that SUDS must
    beat or honestly trade against.
    """
    rng = np.random.RandomState(seed)
    stats: dict[str, Any] = {
        "total_params_modified": 0,
        "total_pruned_columns": 0,
        "total_kept_columns": 0,
        "per_layer": {},
        "selection_signal": "column_l1_norm",
    }

    for slack_name, map_info in mapping.items():
        npz_key = map_info["npz_key"]
        if npz_key not in npz_data:
            continue

        arr = npz_data[npz_key]
        column_dim = map_info["column_dim"]
        target_size = map_info["target_size"]
        if target_size < 4:
            stats["total_kept_columns"] += target_size
            stats["per_layer"][slack_name] = {
                "npz_key": npz_key, "target_size": target_size,
                "pruned": 0, "kept": target_size, "skipped": True,
            }
            continue

        n_prune = int(round(target_size * sparsity))
        n_prune = min(max(n_prune, 0), target_size - 1)
        if n_prune <= 0:
            stats["total_kept_columns"] += target_size
            continue

        prune_idx = np.argsort(_column_l1(arr, column_dim))[:n_prune]
        _add_column_noise(arr, column_dim, prune_idx, prune_noise_std, rng)

        n_keep = target_size - n_prune
        stats["total_params_modified"] += 1
        stats["total_pruned_columns"] += n_prune
        stats["total_kept_columns"] += n_keep
        stats["per_layer"][slack_name] = {
            "npz_key": npz_key,
            "target_size": target_size,
            "pruned": int(n_prune),
            "kept": int(n_keep),
        }

    total = stats["total_pruned_columns"] + stats["total_kept_columns"]
    stats["prune_ratio"] = stats["total_pruned_columns"] / max(1, total)
    return stats


def apply_random_binary_prune_to_npz(
    npz_data: dict[str, np.ndarray],
    mapping: dict[str, dict[str, Any]],
    sparsity: float,
    *,
    prune_noise_std: float = 0.25,
    seed: int = 42,
) -> dict[str, Any]:
    """Apply E5 random KEEP/PRUNE with the same target sparsity as E2/E3."""
    rng = np.random.RandomState(seed)
    stats: dict[str, Any] = {
        "total_params_modified": 0,
        "total_pruned_columns": 0,
        "total_kept_columns": 0,
        "per_layer": {},
        "selection_signal": "random_columns",
    }

    for slack_name, map_info in mapping.items():
        npz_key = map_info["npz_key"]
        if npz_key not in npz_data:
            continue

        arr = npz_data[npz_key]
        column_dim = map_info["column_dim"]
        target_size = map_info["target_size"]
        if target_size < 4:
            stats["total_kept_columns"] += target_size
            stats["per_layer"][slack_name] = {
                "npz_key": npz_key, "target_size": target_size,
                "pruned": 0, "kept": target_size, "skipped": True,
            }
            continue

        n_prune = int(round(target_size * sparsity))
        n_prune = min(max(n_prune, 0), target_size - 1)
        if n_prune <= 0:
            stats["total_kept_columns"] += target_size
            continue

        prune_idx = rng.choice(target_size, size=n_prune, replace=False)
        _add_column_noise(arr, column_dim, prune_idx, prune_noise_std, rng)

        n_keep = target_size - n_prune
        stats["total_params_modified"] += 1
        stats["total_pruned_columns"] += n_prune
        stats["total_kept_columns"] += n_keep
        stats["per_layer"][slack_name] = {
            "npz_key": npz_key,
            "target_size": target_size,
            "pruned": int(n_prune),
            "kept": int(n_keep),
        }

    total = stats["total_pruned_columns"] + stats["total_kept_columns"]
    stats["prune_ratio"] = stats["total_pruned_columns"] / max(1, total)
    return stats


def apply_l1_ternary_proxy_to_npz(
    npz_data: dict[str, np.ndarray],
    mapping: dict[str, dict[str, Any]],
    *,
    keep_ratio: float,
    degrade_ratio: float,
    prune_ratio: float,
    degrade_noise_std: float = 0.03,
    prune_noise_std: float = 0.25,
    seed: int = 42,
) -> dict[str, Any]:
    """Apply E6 signal/amplitude proxy with SUDS-matched tier ratios.

    This baseline uses the same global KEEP/DEGRADE/PRUNE budget as SUDS, but
    chooses concrete columns only by L1 magnitude and applies that budget to
    every mapped layer.  It isolates whether ternary ADC tiering plus amplitude
    selection explains the effect without scheduler-derived slack.
    """
    rng = np.random.RandomState(seed)
    stats: dict[str, Any] = {
        "total_params_modified": 0,
        "total_pruned_columns": 0,
        "total_degraded_columns": 0,
        "total_kept_columns": 0,
        "per_layer": {},
        "budget_signal": "global_suds_tier_ratios",
        "selection_signal": "column_l1_norm",
    }

    for slack_name, map_info in mapping.items():
        npz_key = map_info["npz_key"]
        if npz_key not in npz_data:
            continue

        arr = npz_data[npz_key]
        column_dim = map_info["column_dim"]
        target_size = map_info["target_size"]
        if target_size < 4:
            stats["total_kept_columns"] += target_size
            stats["per_layer"][slack_name] = {
                "npz_key": npz_key, "target_size": target_size,
                "pruned": 0, "degraded": 0, "kept": target_size,
                "skipped": True,
            }
            continue

        n_prune = int(round(target_size * prune_ratio))
        n_degrade = int(round(target_size * degrade_ratio))
        n_prune = min(max(n_prune, 0), target_size - 1)
        n_degrade = min(max(n_degrade, 0), target_size - n_prune - 1)
        n_keep = target_size - n_prune - n_degrade

        order = np.argsort(_column_l1(arr, column_dim))
        prune_idx = order[:n_prune]
        degrade_idx = order[n_prune:n_prune + n_degrade]
        _add_column_noise(arr, column_dim, prune_idx, prune_noise_std, rng)
        _add_column_noise(arr, column_dim, degrade_idx, degrade_noise_std, rng)

        stats["total_params_modified"] += 1
        stats["total_pruned_columns"] += n_prune
        stats["total_degraded_columns"] += n_degrade
        stats["total_kept_columns"] += n_keep
        stats["per_layer"][slack_name] = {
            "npz_key": npz_key,
            "target_size": target_size,
            "pruned": int(n_prune),
            "degraded": int(n_degrade),
            "kept": int(n_keep),
            "target_keep_ratio": keep_ratio,
        }

    total = stats["total_pruned_columns"] + stats["total_degraded_columns"] + stats["total_kept_columns"]
    stats["prune_ratio"] = stats["total_pruned_columns"] / max(1, total)
    stats["degrade_ratio"] = stats["total_degraded_columns"] / max(1, total)
    return stats


def compute_mapped_suds_tier_ratios(
    suds_data: dict[str, dict[str, Any]],
    mapping: dict[str, dict[str, Any]],
) -> dict[str, float]:
    counts = {"keep": 0, "degrade": 0, "prune": 0}
    tau_low, tau_high = _suds_thresholds(suds_data)
    for slack_name, map_info in mapping.items():
        target_size = map_info["target_size"]
        slack_values = map_info["slack_values"]
        layer_suds = suds_data.get(slack_name, {})
        decisions = layer_suds.get("decisions", [])
        if not decisions:
            continue
        resampled = _resample_decisions(
            slack_values,
            decisions,
            target_size,
            tau_low=tau_low,
            tau_high=tau_high,
        )
        counts["keep"] += int((resampled == "KEEP").sum())
        counts["degrade"] += int((resampled == "DEGRADE").sum())
        counts["prune"] += int((resampled == "PRUNE").sum())
    total = sum(counts.values())
    return {
        "keep_ratio": counts["keep"] / max(1, total),
        "degrade_ratio": counts["degrade"] / max(1, total),
        "prune_ratio": counts["prune"] / max(1, total),
        "total_columns": total,
    }


def apply_suds_l1_overlay_to_npz(
    npz_data: dict[str, np.ndarray],
    suds_data: dict[str, dict[str, Any]],
    mapping: dict[str, dict[str, Any]],
    *,
    degrade_noise_std: float = 0.03,
    prune_noise_std: float = 0.25,
    seed: int = 42,
) -> dict[str, Any]:
    """Apply E7: SUDS tier budgets with L1 choosing concrete columns.

    SUDS allocates how many columns per layer fall into KEEP/DEGRADE/PRUNE.
    L1-norm then selects the specific columns inside that budget: lowest-L1
    columns are PRUNE, the next-lowest are DEGRADE, and the rest are KEEP.
    """
    rng = np.random.RandomState(seed)
    tau_low, tau_high = _suds_thresholds(suds_data)
    stats: dict[str, Any] = {
        "total_params_modified": 0,
        "total_pruned_columns": 0,
        "total_degraded_columns": 0,
        "total_kept_columns": 0,
        "per_layer": {},
        "budget_signal": "suds_slack_tier_counts",
        "selection_signal": "column_l1_norm",
    }

    for slack_name, map_info in mapping.items():
        npz_key = map_info["npz_key"]
        if npz_key not in npz_data:
            continue

        arr = npz_data[npz_key]
        column_dim = map_info["column_dim"]
        target_size = map_info["target_size"]
        slack_values = map_info["slack_values"]
        layer_suds = suds_data.get(slack_name, {})
        decisions = layer_suds.get("decisions", [])
        if not decisions:
            continue

        if target_size < 4:
            stats["total_kept_columns"] += target_size
            stats["per_layer"][slack_name] = {
                "npz_key": npz_key, "target_size": target_size,
                "pruned": 0, "degraded": 0, "kept": target_size,
                "skipped": True,
            }
            continue

        resampled = _resample_decisions(
            slack_values,
            decisions,
            target_size,
            tau_low=tau_low,
            tau_high=tau_high,
        )
        n_prune = int((resampled == "PRUNE").sum())
        n_degrade = int((resampled == "DEGRADE").sum())
        n_keep = max(0, target_size - n_prune - n_degrade)

        order = np.argsort(_column_l1(arr, column_dim))
        prune_idx = order[:n_prune]
        degrade_idx = order[n_prune:n_prune + n_degrade]

        _add_column_noise(arr, column_dim, prune_idx, prune_noise_std, rng)
        _add_column_noise(arr, column_dim, degrade_idx, degrade_noise_std, rng)

        stats["total_params_modified"] += 1
        stats["total_pruned_columns"] += n_prune
        stats["total_degraded_columns"] += n_degrade
        stats["total_kept_columns"] += n_keep
        stats["per_layer"][slack_name] = {
            "npz_key": npz_key,
            "target_size": target_size,
            "pruned": n_prune,
            "degraded": n_degrade,
            "kept": n_keep,
        }

    total = stats["total_pruned_columns"] + stats["total_degraded_columns"] + stats["total_kept_columns"]
    stats["prune_ratio"] = stats["total_pruned_columns"] / max(1, total)
    stats["degrade_ratio"] = stats["total_degraded_columns"] / max(1, total)
    return stats


def apply_suds_overflow_proxy_to_npz(
    npz_data: dict[str, np.ndarray],
    suds_data: dict[str, dict[str, Any]],
    mapping: dict[str, dict[str, Any]],
    *,
    degrade_noise_std: float = 0.03,
    prune_noise_std: float = 0.25,
    seed: int = 42,
) -> dict[str, Any]:
    """Apply E8: SUDS tier budgets with a signal-overflow proxy selector.

    SUDS supplies the layer-local KEEP/DEGRADE/PRUNE budget.  A HyAtten-like
    amplitude proxy then protects columns with larger peak/RMS weight energy:
    the lowest-risk columns receive PRUNE noise, the next-lowest receive
    DEGRADE noise, and high-risk columns remain KEEP.
    """
    rng = np.random.RandomState(seed)
    tau_low, tau_high = _suds_thresholds(suds_data)
    stats: dict[str, Any] = {
        "total_params_modified": 0,
        "total_pruned_columns": 0,
        "total_degraded_columns": 0,
        "total_kept_columns": 0,
        "per_layer": {},
        "budget_signal": "suds_slack_tier_counts",
        "selection_signal": "hyatten_like_column_overflow_proxy",
    }

    for slack_name, map_info in mapping.items():
        npz_key = map_info["npz_key"]
        if npz_key not in npz_data:
            continue

        arr = npz_data[npz_key]
        column_dim = map_info["column_dim"]
        target_size = map_info["target_size"]
        slack_values = map_info["slack_values"]
        layer_suds = suds_data.get(slack_name, {})
        decisions = layer_suds.get("decisions", [])
        if not decisions:
            continue

        if target_size < 4:
            stats["total_kept_columns"] += target_size
            stats["per_layer"][slack_name] = {
                "npz_key": npz_key, "target_size": target_size,
                "pruned": 0, "degraded": 0, "kept": target_size,
                "skipped": True,
            }
            continue

        resampled = _resample_decisions(
            slack_values,
            decisions,
            target_size,
            tau_low=tau_low,
            tau_high=tau_high,
        )
        n_prune = int((resampled == "PRUNE").sum())
        n_degrade = int((resampled == "DEGRADE").sum())
        n_keep = max(0, target_size - n_prune - n_degrade)

        order = np.argsort(_column_overflow_proxy(arr, column_dim))
        prune_idx = order[:n_prune]
        degrade_idx = order[n_prune:n_prune + n_degrade]

        _add_column_noise(arr, column_dim, prune_idx, prune_noise_std, rng)
        _add_column_noise(arr, column_dim, degrade_idx, degrade_noise_std, rng)

        stats["total_params_modified"] += 1
        stats["total_pruned_columns"] += n_prune
        stats["total_degraded_columns"] += n_degrade
        stats["total_kept_columns"] += n_keep
        stats["per_layer"][slack_name] = {
            "npz_key": npz_key,
            "target_size": target_size,
            "pruned": n_prune,
            "degraded": n_degrade,
            "kept": n_keep,
        }

    total = stats["total_pruned_columns"] + stats["total_degraded_columns"] + stats["total_kept_columns"]
    stats["prune_ratio"] = stats["total_pruned_columns"] / max(1, total)
    stats["degrade_ratio"] = stats["total_degraded_columns"] / max(1, total)
    return stats


def apply_binary_prune_to_npz(
    npz_data: dict[str, np.ndarray],
    suds_data: dict[str, dict[str, Any]],
    mapping: dict[str, dict[str, Any]],
    sparsity: float,
    *,
    prune_noise_std: float = 0.25,
    seed: int = 42,
) -> dict[str, Any]:
    """Apply binary KEEP/PRUNE via weight perturbation (not channel zeroing).

    KEEP columns: weights unchanged (full 8-bit ADC precision).
    PRUNE columns: weights get coarse quantization noise (~2-bit ADC,
    σ ≈ prune_noise_std × weight_scale), simulating column removal in SC fabric.

    In the SC photonic context, "pruning" a compute column means removing its
    contribution to the bitstream, reducing effective ADC resolution — not
    eliminating the NN feature channel.
    """
    rng = np.random.RandomState(seed)
    stats: dict[str, Any] = {
        "total_params_modified": 0,
        "total_pruned_columns": 0,
        "total_kept_columns": 0,
        "per_layer": {},
    }

    for slack_name, map_info in mapping.items():
        npz_key = map_info["npz_key"]
        if npz_key not in npz_data:
            continue

        arr = npz_data[npz_key]
        column_dim = map_info["column_dim"]
        target_size = map_info["target_size"]
        slack_values = map_info["slack_values"]

        # Skip layers with very few columns (stem conv, classifier)
        if target_size < 4:
            stats["per_layer"][slack_name] = {
                "npz_key": npz_key, "target_size": target_size,
                "pruned": 0, "kept": target_size, "skipped": True,
            }
            stats["total_kept_columns"] += target_size
            continue

        # Resample slack to match target_size
        src_slack = np.array(slack_values)
        if len(src_slack) != target_size:
            src_pos = np.linspace(0, 1, len(src_slack))
            tgt_pos = np.linspace(0, 1, target_size)
            resampled_slack = np.interp(tgt_pos, src_pos, src_slack)
        else:
            resampled_slack = src_slack.copy()

        # Lower slack = more critical → KEEP
        n_keep = max(1, int(target_size * (1.0 - sparsity)))
        # Only prune if we'd actually keep most channels (skip near-0 sparsity)
        n_prune = target_size - n_keep
        if n_prune <= 0:
            stats["total_kept_columns"] += target_size
            continue

        threshold_idx = np.argsort(resampled_slack)[n_keep]
        keep_threshold = resampled_slack[threshold_idx] if n_keep < target_size else float("inf")
        prune_mask = resampled_slack > keep_threshold
        n_prune = int(prune_mask.sum())
        n_keep_actual = target_size - n_prune

        if n_prune > 0:
            idx = np.where(prune_mask)[0]
            _add_column_noise(arr, column_dim, idx, prune_noise_std, rng)

        stats["total_params_modified"] += 1
        stats["total_pruned_columns"] += n_prune
        stats["total_kept_columns"] += n_keep_actual
        stats["per_layer"][slack_name] = {
            "npz_key": npz_key,
            "target_size": target_size,
            "pruned": n_prune,
            "kept": n_keep_actual,
        }

    total = stats["total_pruned_columns"] + stats["total_kept_columns"]
    stats["prune_ratio"] = stats["total_pruned_columns"] / max(1, total)
    return stats


def run_eval_mlx(
    model,
    dataset,
    *,
    batch_size: int,
    max_samples: int | None = None,
    workers: int = 0,
    prefetch_batches: int = 1,
) -> dict[str, Any]:
    """Simple MLX evaluation loop.  Returns top1/top5/elapsed."""
    import mlx.core as mx

    from accuracy.eval_mlx_imagenet_noise import (
        OpenCVImageNetDataset,
        _batch_iter,
        _resolve_total_samples,
    )

    model.eval()
    mx.eval(model.parameters())

    total_samples = 0
    top1_correct = 0
    top5_correct = 0
    t0 = time.perf_counter()

    for batch_idx, (images_np, targets_np) in enumerate(
        _batch_iter(
            dataset,
            batch_size=batch_size,
            max_samples=max_samples,
            workers=workers,
            prefetch_batches=prefetch_batches,
        )
    ):
        images_mx = mx.array(images_np)
        logits = model(images_mx)
        preds = mx.argmax(logits, axis=1)
        top1_batch = int(mx.sum(preds == mx.array(targets_np)).item())

        # Top-5
        topk_indices = mx.argpartition(-logits, kth=4, axis=1)[:, :5]
        targets_expanded = mx.array(targets_np).reshape(-1, 1)
        top5_batch = int(mx.sum(topk_indices == targets_expanded).item())

        mx.eval(preds)
        mx.eval(topk_indices)

        batch_samples = int(targets_np.shape[0])
        total_samples += batch_samples
        top1_correct += top1_batch
        top5_correct += top5_batch

        if total_samples % 500 == 0:
            elapsed = time.perf_counter() - t0
            total = _resolve_total_samples(dataset, max_samples=max_samples)
            pct = (total_samples / total * 100) if total else 0
            print(f"  [{datetime.now().strftime('%H:%M:%S')}] {total_samples}/{total} ({pct:.0f}%) "
                  f"top1={top1_correct/total_samples*100:.1f}%", flush=True)

    elapsed = time.perf_counter() - t0
    top1 = float(top1_correct / total_samples * 100.0) if total_samples > 0 else 0.0
    top5 = float(top5_correct / total_samples * 100.0) if total_samples > 0 else 0.0

    return {
        "top1": top1,
        "top5": top5,
        "processed_samples": total_samples,
        "elapsed_s": elapsed,
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="SUDS ternary weight perturbation evaluation (MLX)")
    p.add_argument("--imagenet_val", required=True, help="Path to ImageNet val root")
    p.add_argument("--model", default="mobilevit_s", help="Model key")
    p.add_argument("--device", default="mps", choices=["mps"],
                   help="Governed accelerator backend. MLX reports this as gpu/Metal; CPU fallback is forbidden.")
    p.add_argument("--weights_npz", default=None, help="Path to pre-exported NPZ weights")
    p.add_argument("--weights_dir", default="weights", help="Local PyTorch weights directory")
    p.add_argument("--mlx_weights_dir", default=None, help="MLX NPZ cache directory")
    p.add_argument("--slack_manifest", default=SLACK_MANIFEST_DEFAULT)
    p.add_argument("--tau_low", type=float, default=0.10, help="SUDS tau_low")
    p.add_argument("--tau_high", type=float, default=0.70, help="SUDS tau_high")
    p.add_argument("--binary_sparsity", type=float, default=0.30, help="E3 binary sparsity ratio")
    p.add_argument("--degrade_noise_std", type=float, default=0.003,
                   help="Noise std for DEGRADE columns (0=identity, ~0.003 for 4-bit SC)")
    p.add_argument("--prune_noise_std", type=float, default=0.05,
                   help="Noise std for PRUNE columns (~0.05 for 2-bit SC)")
    p.add_argument("--max_eval_samples", type=int, default=5000)
    p.add_argument("--eval_batch_size", type=int, default=64)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_json", default=None, help="Detailed results JSON path")
    p.add_argument("--skip_baseline", action="store_true")
    p.add_argument("--skip_binary", action="store_true")
    p.add_argument("--skip_suds", action="store_true")
    p.add_argument("--condition", default="all",
                   choices=[
                       "all",
                       "baseline",
                       "binary",
                       "suds",
                       "e0_dense",
                       "e2_l1",
                       "e3_slack",
                       "e4_suds",
                       "e5_random",
                       "e6_signal",
                       "e7_overlay",
                       "e8_overflow",
                   ],
                   help="Which condition(s) to run")
    p.add_argument("--input_size_override", type=int, default=None,
                   help="Override ImageNet resize/crop size (model weights unchanged)")
    return p


def print_report(results: dict[str, Any]) -> None:
    baseline = results.get("e0_dense", results.get("baseline", {}))
    baseline_top1 = baseline.get("top1")
    baseline_top5 = baseline.get("top5")

    print("\n" + "=" * 80)
    print("SUDS Evaluation Results (MLX)")
    print("=" * 80)

    label_map = {
        "e0_dense": "E0 Dense baseline (unmodified)",
        "e2_l1": f"E2 L1 binary KEEP/PRUNE (s={results.get('config', {}).get('binary_sparsity', 0.3)})",
        "e3_slack": f"E3 Slack binary KEEP/PRUNE (s={results.get('config', {}).get('binary_sparsity', 0.3)})",
        "e4_suds": (
            "E4 SUDS ternary "
            f"(tau={results.get('config', {}).get('tau_low', 0.10):.2f}, "
            f"{results.get('config', {}).get('tau_high', 0.70):.2f})"
        ),
        "e5_random": f"E5 Random binary KEEP/PRUNE (s={results.get('config', {}).get('binary_sparsity', 0.3)})",
        "e6_signal": "E6 L1 signal ternary, SUDS-matched tier budget",
        "e7_overlay": "E7 SUDS+L1 overlay",
        "e8_overflow": "E8 SUDS+signal-overflow proxy",
    }

    for cond_name in ["e0_dense", "e2_l1", "e3_slack", "e4_suds", "e5_random", "e6_signal", "e7_overlay", "e8_overflow"]:
        if cond_name not in results:
            continue
        r = results[cond_name]
        top1 = r.get("top1")
        top5 = r.get("top5")
        d1 = (top1 - baseline_top1) if baseline_top1 is not None and top1 is not None else None
        d5 = (top5 - baseline_top5) if baseline_top5 is not None and top5 is not None else None

        print(f"\n  {label_map.get(cond_name, cond_name)}:")
        print(f"    Top-1: {top1:.2f}%  |  Top-5: {top5:.2f}%")
        if d1 is not None and cond_name != "e0_dense":
            print(f"    ΔTop-1: {d1:+.2f}pp  |  ΔTop-5: {d5:+.2f}pp")
        if "elapsed_s" in r:
            print(f"    Time: {r['elapsed_s']:.1f}s  |  Samples: {r.get('processed_samples', '?')}")

        perturb_stats = r.get("perturb_stats")
        if perturb_stats:
            pr = perturb_stats.get("prune_ratio", 0)
            dr = perturb_stats.get("degrade_ratio", 0)
            kr = 1.0 - pr - dr
            print(f"    Params modified: {perturb_stats.get('total_params_modified', '?')}")
            print(f"    Columns - KEEP: {kr:.1%}  DEGRADE: {dr:.1%}  PRUNE: {pr:.1%}")

    print("\n" + "=" * 80)


def main():
    parser = build_parser()
    args = parser.parse_args()

    # Import MLX modules
    try:
        import mlx.core as mx
    except ImportError:
        raise SystemExit("MLX is required. Install: pip install mlx")
    metal_available = bool(mx.metal.is_available()) if hasattr(mx, "metal") else False
    mlx_default_device = str(mx.default_device())
    if args.device != "mps" or not metal_available or "gpu" not in mlx_default_device.lower():
        raise SystemExit(
            "Governed MLX/MPS execution is required; CPU fallback is forbidden. "
            f"device={args.device} metal_available={metal_available} default={mlx_default_device}"
        )

    from accuracy.mlx_mobilevit import (
        MLXMobileViT,
        ensure_mlx_weights_exported,
        resolve_mlx_weights_cache_path,
        resolve_default_mlx_weights_dir,
    )
    from accuracy.eval_mlx_imagenet_noise import (
        OpenCVImageNetDataset,
        _resolve_total_samples,
    )
    from exp_common.model_specs import MODEL_SPECS

    if args.model not in MODEL_SPECS:
        raise SystemExit(f"Unsupported model: {args.model}. Choices: {list(MODEL_SPECS)}")

    model_key = args.model
    spec = MODEL_SPECS[model_key]
    nominal_input_size = int(spec["input_size"])
    input_size = args.input_size_override if args.input_size_override is not None else nominal_input_size

    # ── Resolve NPZ weights path ──────────────────────────────────────────
    if args.weights_npz:
        npz_path = Path(args.weights_npz)
    else:
        mlx_weights_dir = args.mlx_weights_dir or resolve_default_mlx_weights_dir()
        npz_path = ensure_mlx_weights_exported(
            model_key,
            weights_dir=args.weights_dir,
            out_dir=mlx_weights_dir,
        )
    print(f"NPZ weights: {npz_path}")

    # ── Load slack + SUDS decisions ───────────────────────────────────────
    print(f"Loading slack manifest: {args.slack_manifest}")
    with open(args.slack_manifest, "r") as fh:
        raw_slack = json.load(fh)
    manifest_model_key = raw_slack.get("_global", {}).get("model_key")
    if manifest_model_key and manifest_model_key != model_key:
        raise SystemExit(
            "Transferred slack manifests are forbidden for P2 validation. "
            f"model={model_key} manifest_model_key={manifest_model_key} "
            f"manifest={args.slack_manifest}"
        )
    suds_data = apply_suds_decisions(raw_slack, args.tau_low, args.tau_high)
    global_stats = suds_data.get("_global", {})
    print(f"SUDS tiers: KEEP={global_stats.get('keep_ratio', 0):.1%} "
          f"DEGRADE={global_stats.get('degrade_ratio', 0):.1%} "
          f"PRUNE={global_stats.get('prune_ratio', 0):.1%} "
          f"({global_stats.get('total_columns', 0)} total cols)")

    # ── Load baseline NPZ ─────────────────────────────────────────────────
    base_npz = dict(np.load(npz_path))
    print(f"Loaded {len(base_npz)} arrays from NPZ")

    # ── Build slack → NPZ mapping ─────────────────────────────────────────
    mapping = build_slack_to_npz_mapping(base_npz, raw_slack, model_key=model_key)
    print(f"Mapped {len(mapping)} slack layers to NPZ weight arrays")
    mapped_suds_tiers = compute_mapped_suds_tier_ratios(suds_data, mapping)
    print(f"Mapped SUDS tiers: KEEP={mapped_suds_tiers['keep_ratio']:.1%} "
          f"DEGRADE={mapped_suds_tiers['degrade_ratio']:.1%} "
          f"PRUNE={mapped_suds_tiers['prune_ratio']:.1%} "
          f"({mapped_suds_tiers['total_columns']} mapped cols)")

    # ── Set up dataset ────────────────────────────────────────────────────
    dataset = OpenCVImageNetDataset(
        args.imagenet_val,
        manifest_path=None,
        resize_size=input_size + 32,
        center_crop_size=input_size,
        percentage=100.0,
        seed=args.seed,
        enable_mean_std=False,
        mean_std_mean=[0.0, 0.0, 0.0],
        mean_std_std=[1.0, 1.0, 1.0],
        input_color_order="bgr",   # cv2.imread produces BGR
        model_color_order="rgb",   # MobileViT expects RGB
        input_scale=1.0,
    )
    total_available = len(dataset)
    eval_samples = min(args.max_eval_samples, total_available)
    print(f"Dataset: {total_available} samples, evaluating {eval_samples}")
    try:
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=_PROJECT_ROOT,
            text=True,
        ).strip()
    except Exception:
        git_hash = "unknown"

    results: dict[str, Any] = {
        "config": {
            "model": model_key,
            "device": args.device,
            "mlx_default_device": mlx_default_device,
            "mlx_metal_available": metal_available,
            "git_hash": git_hash,
            "command": " ".join(sys.argv),
            "dataset_root": args.imagenet_val,
            "subset_policy": (
                f"max_eval_samples={args.max_eval_samples}; "
                "uses OpenCVImageNetDataset deterministic seed order"
            ),
            "nominal_input_size": nominal_input_size,
            "eval_input_size": input_size,
            "input_size_override": args.input_size_override,
            "max_eval_samples": args.max_eval_samples,
            "processed_sample_target": eval_samples,
            "seed": args.seed,
            "tau_low": args.tau_low,
            "tau_high": args.tau_high,
            "binary_sparsity": args.binary_sparsity,
            "npz_path": str(npz_path),
            "weights_dir": args.weights_dir,
            "slack_manifest": args.slack_manifest,
            "environment_note": "MLX backend on Apple Metal/MPS; CPU fallback forbidden.",
            "perturbation_isolation": "independent_weight_copy_per_condition",
            "mapped_suds_tiers": mapped_suds_tiers,
        },
        "suds_global": global_stats,
    }

    condition_alias = {
        "baseline": "e0_dense",
        "binary": "e3_slack",
        "suds": "e4_suds",
    }
    condition = condition_alias.get(args.condition, args.condition)

    run_baseline = condition in ("all", "e0_dense") and not args.skip_baseline
    run_e2_l1 = condition in ("all", "e2_l1") and not args.skip_binary
    run_e3_slack = condition in ("all", "e3_slack") and not args.skip_binary
    run_e4_suds = condition in ("all", "e4_suds") and not args.skip_suds
    run_e5_random = condition in ("all", "e5_random") and not args.skip_binary
    run_e6_signal = condition in ("all", "e6_signal") and not args.skip_suds
    run_e7_overlay = condition in ("all", "e7_overlay") and not args.skip_suds
    run_e8_overflow = condition in ("all", "e8_overflow") and not args.skip_suds

    # ── E0 Baseline ───────────────────────────────────────────────────────
    if run_baseline:
        print("\n── E0 Baseline (unmodified weights) ──")
        model = MLXMobileViT(model_key=model_key)
        model.load_weights(str(npz_path), strict=True)
        model.eval()
        mx.eval(model.parameters())
        result = run_eval_mlx(
            model, dataset,
            batch_size=args.eval_batch_size,
            max_samples=eval_samples,
            workers=args.workers,
        )
        results["e0_dense"] = result
        results["baseline"] = result  # legacy alias
        print(f"  Top-1: {result['top1']:.2f}%  Top-5: {result['top5']:.2f}%  "
              f"({result['processed_samples']} samples in {result['elapsed_s']:.1f}s)")
        del model

    # ── E2 L1 Binary ──────────────────────────────────────────────────────
    if run_e2_l1:
        print(f"\n── E2 L1 Binary KEEP/PRUNE (s={args.binary_sparsity}) ──")
        modified_npz = clone_npz_arrays(base_npz)
        perturb_stats = apply_l1_binary_prune_to_npz(
            modified_npz, mapping, sparsity=args.binary_sparsity,
            prune_noise_std=args.prune_noise_std, seed=args.seed,
        )
        print(f"  Pruned: {perturb_stats['total_pruned_columns']} columns "
              f"({perturb_stats['prune_ratio']:.1%}) "
              f"across {perturb_stats['total_params_modified']} params")

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tmp:
            np.savez_compressed(tmp.name, **modified_npz)
            tmp_path = tmp.name

        model = MLXMobileViT(model_key=model_key)
        model.load_weights(tmp_path, strict=True)
        model.eval()
        mx.eval(model.parameters())
        result = run_eval_mlx(
            model, dataset,
            batch_size=args.eval_batch_size,
            max_samples=eval_samples,
            workers=args.workers,
        )
        result["perturb_stats"] = perturb_stats
        results["e2_l1"] = result
        print(f"  Top-1: {result['top1']:.2f}%  Top-5: {result['top5']:.2f}%  "
              f"({result['processed_samples']} samples in {result['elapsed_s']:.1f}s)")
        os.unlink(tmp_path)
        del model
        del modified_npz

    # ── E3 Slack Binary ───────────────────────────────────────────────────
    if run_e3_slack:
        print(f"\n── E3 Slack Binary KEEP/PRUNE (s={args.binary_sparsity}) ──")
        modified_npz = clone_npz_arrays(base_npz)
        perturb_stats = apply_binary_prune_to_npz(
            modified_npz, suds_data, mapping, sparsity=args.binary_sparsity,
            prune_noise_std=args.prune_noise_std, seed=args.seed,
        )
        print(f"  Pruned: {perturb_stats['total_pruned_columns']} columns "
              f"({perturb_stats['prune_ratio']:.1%}) "
              f"across {perturb_stats['total_params_modified']} params")

        # Save to temp file and load model
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tmp:
            np.savez_compressed(tmp.name, **modified_npz)
            tmp_path = tmp.name

        model = MLXMobileViT(model_key=model_key)
        model.load_weights(tmp_path, strict=True)
        model.eval()
        mx.eval(model.parameters())
        result = run_eval_mlx(
            model, dataset,
            batch_size=args.eval_batch_size,
            max_samples=eval_samples,
            workers=args.workers,
        )
        result["perturb_stats"] = perturb_stats
        results["e3_slack"] = result
        results["binary"] = result  # legacy alias
        print(f"  Top-1: {result['top1']:.2f}%  Top-5: {result['top5']:.2f}%  "
              f"({result['processed_samples']} samples in {result['elapsed_s']:.1f}s)")
        os.unlink(tmp_path)
        del model
        del modified_npz

    # ── E4 SUDS Ternary ───────────────────────────────────────────────────
    if run_e4_suds:
        print(f"\n── E4 SUDS Ternary (tau=({args.tau_low:.2f}, {args.tau_high:.2f})) ──")
        modified_npz = clone_npz_arrays(base_npz)
        perturb_stats = apply_suds_to_npz(
            modified_npz, suds_data, mapping,
            degrade_noise_std=args.degrade_noise_std,
            prune_noise_std=args.prune_noise_std,
            seed=args.seed,
        )
        print(f"  PRUNE: {perturb_stats['total_pruned_columns']}  "
              f"DEGRADE: {perturb_stats['total_degraded_columns']}  "
              f"KEEP: {perturb_stats['total_kept_columns']}  "
              f"({perturb_stats['total_params_modified']} params)")

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tmp:
            np.savez_compressed(tmp.name, **modified_npz)
            tmp_path = tmp.name

        model = MLXMobileViT(model_key=model_key)
        model.load_weights(tmp_path, strict=True)
        model.eval()
        mx.eval(model.parameters())
        result = run_eval_mlx(
            model, dataset,
            batch_size=args.eval_batch_size,
            max_samples=eval_samples,
            workers=args.workers,
        )
        result["perturb_stats"] = perturb_stats
        results["e4_suds"] = result
        results["suds"] = result  # legacy alias
        print(f"  Top-1: {result['top1']:.2f}%  Top-5: {result['top5']:.2f}%  "
              f"({result['processed_samples']} samples in {result['elapsed_s']:.1f}s)")
        os.unlink(tmp_path)
        del model
        del modified_npz

    # ── E5 Random Binary ────────────────────────────────────────────────────
    if run_e5_random:
        print(f"\n── E5 Random Binary KEEP/PRUNE (s={args.binary_sparsity}) ──")
        modified_npz = clone_npz_arrays(base_npz)
        perturb_stats = apply_random_binary_prune_to_npz(
            modified_npz, mapping, sparsity=args.binary_sparsity,
            prune_noise_std=args.prune_noise_std, seed=args.seed,
        )
        print(f"  Pruned: {perturb_stats['total_pruned_columns']} columns "
              f"({perturb_stats['prune_ratio']:.1%}) "
              f"across {perturb_stats['total_params_modified']} params")

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tmp:
            np.savez_compressed(tmp.name, **modified_npz)
            tmp_path = tmp.name

        model = MLXMobileViT(model_key=model_key)
        model.load_weights(tmp_path, strict=True)
        model.eval()
        mx.eval(model.parameters())
        result = run_eval_mlx(
            model, dataset,
            batch_size=args.eval_batch_size,
            max_samples=eval_samples,
            workers=args.workers,
        )
        result["perturb_stats"] = perturb_stats
        results["e5_random"] = result
        print(f"  Top-1: {result['top1']:.2f}%  Top-5: {result['top5']:.2f}%  "
              f"({result['processed_samples']} samples in {result['elapsed_s']:.1f}s)")
        os.unlink(tmp_path)
        del model
        del modified_npz

    # ── E6 Signal/Amplitude Ternary Proxy ───────────────────────────────────
    if run_e6_signal:
        print(f"\n── E6 L1 Signal Ternary Proxy (SUDS-matched global tier budget) ──")
        modified_npz = clone_npz_arrays(base_npz)
        perturb_stats = apply_l1_ternary_proxy_to_npz(
            modified_npz, mapping,
            keep_ratio=float(mapped_suds_tiers.get("keep_ratio", 0.0)),
            degrade_ratio=float(mapped_suds_tiers.get("degrade_ratio", 0.0)),
            prune_ratio=float(mapped_suds_tiers.get("prune_ratio", 0.0)),
            degrade_noise_std=args.degrade_noise_std,
            prune_noise_std=args.prune_noise_std,
            seed=args.seed,
        )
        print(f"  PRUNE: {perturb_stats['total_pruned_columns']}  "
              f"DEGRADE: {perturb_stats['total_degraded_columns']}  "
              f"KEEP: {perturb_stats['total_kept_columns']}  "
              f"({perturb_stats['total_params_modified']} params)")

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tmp:
            np.savez_compressed(tmp.name, **modified_npz)
            tmp_path = tmp.name

        model = MLXMobileViT(model_key=model_key)
        model.load_weights(tmp_path, strict=True)
        model.eval()
        mx.eval(model.parameters())
        result = run_eval_mlx(
            model, dataset,
            batch_size=args.eval_batch_size,
            max_samples=eval_samples,
            workers=args.workers,
        )
        result["perturb_stats"] = perturb_stats
        results["e6_signal"] = result
        print(f"  Top-1: {result['top1']:.2f}%  Top-5: {result['top5']:.2f}%  "
              f"({result['processed_samples']} samples in {result['elapsed_s']:.1f}s)")
        os.unlink(tmp_path)
        del model
        del modified_npz

    # ── E7 SUDS+L1 Overlay ────────────────────────────────────────────────
    if run_e7_overlay:
        print(f"\n── E7 SUDS+L1 Overlay (tau=({args.tau_low:.2f}, {args.tau_high:.2f})) ──")
        modified_npz = clone_npz_arrays(base_npz)
        perturb_stats = apply_suds_l1_overlay_to_npz(
            modified_npz, suds_data, mapping,
            degrade_noise_std=args.degrade_noise_std,
            prune_noise_std=args.prune_noise_std,
            seed=args.seed,
        )
        print(f"  PRUNE: {perturb_stats['total_pruned_columns']}  "
              f"DEGRADE: {perturb_stats['total_degraded_columns']}  "
              f"KEEP: {perturb_stats['total_kept_columns']}  "
              f"({perturb_stats['total_params_modified']} params)")

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tmp:
            np.savez_compressed(tmp.name, **modified_npz)
            tmp_path = tmp.name

        model = MLXMobileViT(model_key=model_key)
        model.load_weights(tmp_path, strict=True)
        model.eval()
        mx.eval(model.parameters())
        result = run_eval_mlx(
            model, dataset,
            batch_size=args.eval_batch_size,
            max_samples=eval_samples,
            workers=args.workers,
        )
        result["perturb_stats"] = perturb_stats
        results["e7_overlay"] = result
        print(f"  Top-1: {result['top1']:.2f}%  Top-5: {result['top5']:.2f}%  "
              f"({result['processed_samples']} samples in {result['elapsed_s']:.1f}s)")
        os.unlink(tmp_path)
        del model
        del modified_npz

    # ── E8 SUDS+Signal-Overflow Proxy ─────────────────────────────────────
    if run_e8_overflow:
        print(f"\n── E8 SUDS+Signal-Overflow Proxy (tau=({args.tau_low:.2f}, {args.tau_high:.2f})) ──")
        modified_npz = clone_npz_arrays(base_npz)
        perturb_stats = apply_suds_overflow_proxy_to_npz(
            modified_npz, suds_data, mapping,
            degrade_noise_std=args.degrade_noise_std,
            prune_noise_std=args.prune_noise_std,
            seed=args.seed,
        )
        print(f"  PRUNE: {perturb_stats['total_pruned_columns']}  "
              f"DEGRADE: {perturb_stats['total_degraded_columns']}  "
              f"KEEP: {perturb_stats['total_kept_columns']}  "
              f"({perturb_stats['total_params_modified']} params)")

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tmp:
            np.savez_compressed(tmp.name, **modified_npz)
            tmp_path = tmp.name

        model = MLXMobileViT(model_key=model_key)
        model.load_weights(tmp_path, strict=True)
        model.eval()
        mx.eval(model.parameters())
        result = run_eval_mlx(
            model, dataset,
            batch_size=args.eval_batch_size,
            max_samples=eval_samples,
            workers=args.workers,
        )
        result["perturb_stats"] = perturb_stats
        results["e8_overflow"] = result
        print(f"  Top-1: {result['top1']:.2f}%  Top-5: {result['top5']:.2f}%  "
              f"({result['processed_samples']} samples in {result['elapsed_s']:.1f}s)")
        os.unlink(tmp_path)
        del model
        del modified_npz

    # ── Report ────────────────────────────────────────────────────────────
    print_report(results)

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as fh:
            json.dump(results, fh, indent=2, default=str)
        print(f"\nResults saved: {out_path}")


if __name__ == "__main__":
    main()
