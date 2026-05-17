"""MLX implementation of the bounded photonic perturbation proxy."""

from __future__ import annotations

from typing import Any, Callable

try:
    import mlx.core as mx
except Exception:  # pragma: no cover - exercised via mlx_available()
    mx = None

from accuracy.photonic_perturb_common import (
    normalize_channel_dim,
    resolve_keep_count,
    resolve_quant_max,
)


_MISSING = object()


def mlx_available() -> bool:
    """Return True when MLX is importable in the current interpreter."""
    return mx is not None


def require_mlx() -> None:
    """Raise a helpful error when MLX support was requested but is unavailable."""
    if mx is None:
        raise RuntimeError(
            "MLX is not installed in the active environment. "
            "Install `mlx` in the Apple Silicon repo interpreter before using the MLX path."
        )


def _is_mlx_array(value: object) -> bool:
    if mx is None:
        return False
    module_name = getattr(value.__class__, "__module__", "")
    return module_name.startswith("mlx")


def _normalize_channel_dim(
    tensor: Any,
    channel_dim: int | None,
) -> int | None:
    return normalize_channel_dim(getattr(tensor, "ndim", 0), channel_dim)


def compute_symmetric_scale(
    tensor: Any | None,
    bits: int | None,
    *,
    eps: float = 1e-8,
    channel_dim: int | None = None,
):
    """Compute a per-tensor or per-channel symmetric quantization scale."""
    if not _is_mlx_array(tensor):
        return None
    quant_max = resolve_quant_max(bits)
    if quant_max is None:
        return None
    channel_dim = _normalize_channel_dim(tensor, channel_dim)
    if channel_dim is None:
        max_value = mx.max(mx.abs(tensor))
    else:
        reduce_dims = tuple(dim for dim in range(tensor.ndim) if dim != channel_dim)
        if reduce_dims:
            max_value = mx.max(mx.abs(tensor), axis=reduce_dims, keepdims=True)
        else:
            max_value = mx.max(mx.abs(tensor))
    return mx.maximum(max_value / quant_max, eps)


def quantize_symmetric(
    tensor: Any | None,
    bits: int | None,
    *,
    eps: float = 1e-8,
    scale_override: Any | float | None = None,
    channel_dim: int | None = None,
):
    """Quantize a tensor symmetrically and return ``(quantized, scale)``."""
    if not _is_mlx_array(tensor):
        return tensor, None
    quant_max = resolve_quant_max(bits)
    if quant_max is None:
        return tensor, None
    if scale_override is None:
        scale = compute_symmetric_scale(
            tensor,
            bits,
            eps=eps,
            channel_dim=channel_dim,
        )
    else:
        scale = (
            scale_override.astype(tensor.dtype)
            if _is_mlx_array(scale_override)
            else mx.array(scale_override, dtype=tensor.dtype)
        )
        scale = mx.maximum(scale, eps)
    if scale is None:
        return tensor, None
    quantized = mx.clip(mx.round(tensor / scale), -quant_max, quant_max) * scale
    return quantized, scale


def add_gaussian_noise(
    tensor: Any | None,
    noise_std: float | None,
    scale: Any | None,
    *,
    channel_dim: int | None = None,
    channel_shared: bool = False,
):
    """Add mapping-induced Gaussian noise with sigma specified in LSB units."""
    if not _is_mlx_array(tensor):
        return tensor
    if noise_std is None or scale is None:
        return tensor
    std_value = float(noise_std)
    if std_value <= 0:
        return tensor
    if channel_shared:
        channel_dim = _normalize_channel_dim(tensor, channel_dim)
        if channel_dim is None:
            noise = mx.random.normal(shape=[1], dtype=tensor.dtype)
        else:
            noise_shape = [1] * tensor.ndim
            noise_shape[channel_dim] = tensor.shape[channel_dim]
            noise = mx.random.normal(shape=noise_shape, dtype=tensor.dtype)
    else:
        noise = mx.random.normal(shape=tensor.shape, dtype=tensor.dtype)
    return tensor + (noise * (scale * std_value))


def _shift_with_zero_padding(
    tensor: Any,
    *,
    shifts: int,
    channel_dim: int,
):
    """Shift along ``channel_dim`` without cyclic wraparound."""
    if shifts == 0:
        return tensor
    zero_shape = list(tensor.shape)
    if shifts > 0:
        zero_shape[channel_dim] = shifts
        body_slices = [slice(None)] * tensor.ndim
        body_slices[channel_dim] = slice(0, tensor.shape[channel_dim] - shifts)
        return mx.concatenate(
            [
                mx.zeros(zero_shape, dtype=tensor.dtype),
                tensor[tuple(body_slices)],
            ],
            axis=channel_dim,
        )
    shift_abs = abs(shifts)
    zero_shape[channel_dim] = shift_abs
    body_slices = [slice(None)] * tensor.ndim
    body_slices[channel_dim] = slice(shift_abs, None)
    return mx.concatenate(
        [
            tensor[tuple(body_slices)],
            mx.zeros(zero_shape, dtype=tensor.dtype),
        ],
        axis=channel_dim,
    )


def apply_crosstalk(
    tensor: Any | None,
    alpha: float | None,
    channel_dim: int | None,
    *,
    bidirectional: bool = True,
):
    """Apply neighbor-channel crosstalk along ``channel_dim``."""
    if not _is_mlx_array(tensor):
        return tensor
    if alpha is None:
        return tensor
    alpha_value = float(alpha)
    if alpha_value <= 0:
        return tensor
    channel_dim = _normalize_channel_dim(tensor, channel_dim)
    if channel_dim is None:
        channel_dim = tensor.ndim - 1
    if bidirectional:
        shifted_left = _shift_with_zero_padding(
            tensor,
            shifts=1,
            channel_dim=channel_dim,
        )
        shifted_right = _shift_with_zero_padding(
            tensor,
            shifts=-1,
            channel_dim=channel_dim,
        )
        return (
            tensor * (1 - alpha_value)
            + shifted_left * (alpha_value / 2.0)
            + shifted_right * (alpha_value / 2.0)
        )
    shifted = _shift_with_zero_padding(
        tensor,
        shifts=1,
        channel_dim=channel_dim,
    )
    return tensor * (1 - alpha_value) + shifted * alpha_value


def apply_det_prefix_mixing(
    tensor: Any | None,
    det_enabled: bool,
    det_prefix_error_mean: float | None,
    channel_dim: int | None,
):
    """Inject a deterministic DET prefix-mixing error along ``channel_dim``."""
    if not _is_mlx_array(tensor):
        return tensor
    if not det_enabled:
        return tensor
    if det_prefix_error_mean is None:
        return tensor
    mix = float(det_prefix_error_mean)
    if mix <= 0:
        return tensor
    resolved_dim = channel_dim if channel_dim is not None else -1
    shifted = mx.roll(tensor, 1, axis=resolved_dim)
    return tensor * (1.0 - mix) + shifted * mix


def apply_det_early_stop_policy(
    tensor: Any | None,
    *,
    det_enabled: bool,
    det_k: float | None,
    det_bsl_max: float | None,
    channel_dim: int | None,
    det_mode: str | None = None,
):
    """Apply the DET early-stop proxy using a keep-ratio on the channel axis."""
    if not _is_mlx_array(tensor):
        return tensor
    if not det_enabled:
        return tensor
    if det_k is None or det_bsl_max is None:
        return tensor
    bsl_max_value = float(det_bsl_max)
    if bsl_max_value <= 0.0:
        return tensor
    keep_ratio = max(0.0, min(1.0, float(det_k) / bsl_max_value))
    if keep_ratio >= 1.0:
        return tensor
    if keep_ratio <= 0.0:
        return mx.zeros_like(tensor)
    channel_dim = _normalize_channel_dim(tensor, channel_dim)
    if channel_dim is None:
        channel_dim = tensor.ndim - 1
    channels = int(tensor.shape[channel_dim])
    if channels <= 0:
        return tensor
    keep = resolve_keep_count(channels, keep_ratio)
    if keep >= channels:
        return tensor
    kept_slices = [slice(None)] * tensor.ndim
    kept_slices[channel_dim] = slice(0, keep)
    kept = tensor[tuple(kept_slices)]
    pad_shape = list(tensor.shape)
    pad_shape[channel_dim] = channels - keep
    mode = str(det_mode or "reorder").strip().lower()
    if mode == "replace" and keep > 0:
        tail_slices = [slice(None)] * tensor.ndim
        tail_slices[channel_dim] = slice(keep - 1, keep)
        tail_value = tensor[tuple(tail_slices)]
        pad = mx.ones(pad_shape, dtype=tensor.dtype) * tail_value
    else:
        pad = mx.zeros(pad_shape, dtype=tensor.dtype)
    return mx.concatenate([kept, pad], axis=channel_dim)


def apply_sparse_gating(
    tensor: Any | None,
    sparse_active_fraction: float | None,
    channel_dim: int | None,
    *,
    sparse_tau_global: float | None = None,
    return_stats: bool = False,
):
    """Apply the bounded sparse proxy using MLX array operators.

    ``sparse_active_fraction`` acts as a safety floor for tau-threshold gating:
    a configured 75% active point must not collapse to a much more destructive
    threshold just because a layer has a heavy-tailed activation distribution.
    """
    if not _is_mlx_array(tensor):
        return (tensor, None) if return_stats else tensor

    def _activity_stats(mask: Any, gate_mode: str) -> dict[str, Any]:
        active_elements = int(mx.sum(mask.astype(mx.int32)).item())
        total_elements = int(mask.size)
        return {
            "gate_mode": gate_mode,
            "active_elements": active_elements,
            "total_elements": total_elements,
            "activity_fraction": (
                active_elements / total_elements if total_elements > 0 else None
            ),
        }

    sparse_tau = None if sparse_tau_global is None else float(sparse_tau_global)
    if sparse_tau is not None and sparse_tau > 0.0:
        reference = mx.max(mx.abs(tensor))
        if float(reference.item()) <= 0.0:
            stats = {
                "gate_mode": "tau_threshold",
                "active_elements": int(tensor.size),
                "total_elements": int(tensor.size),
                "activity_fraction": 1.0,
            }
            return (tensor, stats) if return_stats else tensor
        threshold = reference * sparse_tau
        mask = mx.abs(tensor) >= threshold
        gate_mode = "tau_threshold"
        if sparse_active_fraction is not None:
            active_fraction = float(sparse_active_fraction)
            if 0.0 < active_fraction < 1.0:
                total = int(tensor.size)
                min_keep = resolve_keep_count(total, active_fraction)
                active = int(mx.sum(mask.astype(mx.int32)).item())
                if active < min_keep:
                    scores = mx.reshape(mx.abs(tensor), (-1,))
                    floor_threshold = mx.sort(scores)[-min_keep]
                    floor_mask = mx.abs(tensor) >= floor_threshold
                    mask = (mask.astype(mx.int32) + floor_mask.astype(mx.int32)) > 0
                    gate_mode = "tau_threshold_min_active"
        gated = tensor * mask.astype(tensor.dtype)
        stats = _activity_stats(mask, gate_mode)
        return (gated, stats) if return_stats else gated
    if sparse_active_fraction is None:
        return (tensor, None) if return_stats else tensor
    active_fraction = float(sparse_active_fraction)
    if active_fraction >= 1.0:
        stats = {
            "gate_mode": "topk_proxy",
            "active_elements": int(tensor.size),
            "total_elements": int(tensor.size),
            "activity_fraction": 1.0,
        }
        return (tensor, stats) if return_stats else tensor
    if active_fraction <= 0.0:
        gated = mx.zeros_like(tensor)
        stats = {
            "gate_mode": "topk_proxy",
            "active_elements": 0,
            "total_elements": int(tensor.size),
            "activity_fraction": 0.0,
        }
        return (gated, stats) if return_stats else gated

    channel_dim = _normalize_channel_dim(tensor, channel_dim)
    if channel_dim is None:
        total = int(tensor.size)
        keep = resolve_keep_count(total, active_fraction)
        if keep >= total:
            stats = {
                "gate_mode": "topk_proxy",
                "active_elements": total,
                "total_elements": total,
                "activity_fraction": 1.0,
            }
            return (tensor, stats) if return_stats else tensor
        scores = mx.reshape(mx.abs(tensor), (-1,))
        threshold = mx.sort(scores)[-keep]
        mask = scores >= threshold
        gated = mx.reshape(
            mx.reshape(tensor, (-1,)) * mask.astype(tensor.dtype),
            tensor.shape,
        )
        stats = _activity_stats(mask, "topk_proxy")
        return (gated, stats) if return_stats else gated

    moved = mx.moveaxis(mx.abs(tensor), channel_dim, 0)
    scores = mx.mean(mx.reshape(moved, (moved.shape[0], -1)), axis=1)
    channels = int(scores.size)
    keep = resolve_keep_count(channels, active_fraction)
    if keep >= channels:
        stats = {
            "gate_mode": "topk_proxy",
            "active_elements": int(tensor.size),
            "total_elements": int(tensor.size),
            "activity_fraction": 1.0,
        }
        return (tensor, stats) if return_stats else tensor
    threshold = mx.sort(scores)[-keep]
    channel_mask = (scores >= threshold).astype(tensor.dtype)
    mask_shape = [1] * tensor.ndim
    mask_shape[channel_dim] = channels
    expanded_mask = mx.reshape(channel_mask, mask_shape)
    gated = tensor * expanded_mask
    stats = _activity_stats(expanded_mask > 0, "topk_proxy")
    return (gated, stats) if return_stats else gated


def apply_perturb(
    tensor: Any | None,
    bits: int | None,
    legacy_sigma_lsb_or_alpha: float | None,
    alpha_or_channel_dim: float | int | None = None,
    channel_dim: object = _MISSING,
    *,
    gaussian_noise_std: float | None = None,
    scale_override: Any | float | None = None,
    bidirectional: bool = True,
    drift_lsb: float | None = None,
    noise_correlation: float | None = None,
    burst_error_prob: float | None = None,
    burst_error_scale_lsb: float | None = None,
    burst_span: int | None = None,
    det_enabled: bool = False,
    det_k: float | None = None,
    det_bsl_max: float | None = None,
    det_mode: str | None = None,
    det_prefix_error_mean: float | None = None,
    sparse_enabled: bool = False,
    sparse_active_fraction: float | None = None,
    sparse_tau_global: float | None = None,
    sparse_activity_callback: Callable[[dict[str, Any]], None] | None = None,
):
    """Apply the bounded photonic perturbation pipeline to an MLX tensor."""
    if not _is_mlx_array(tensor):
        return tensor
    if channel_dim is _MISSING:
        alpha = legacy_sigma_lsb_or_alpha
        resolved_channel_dim = alpha_or_channel_dim
    else:
        alpha = alpha_or_channel_dim
        resolved_channel_dim = channel_dim
        if gaussian_noise_std is None:
            gaussian_noise_std = legacy_sigma_lsb_or_alpha
    unsupported = []
    if noise_correlation is not None and float(noise_correlation) != 0.0:
        unsupported.append("noise_correlation")
    if burst_error_prob is not None and float(burst_error_prob) != 0.0:
        unsupported.append("burst_error_prob")
    if burst_error_scale_lsb is not None and float(burst_error_scale_lsb) != 0.0:
        unsupported.append("burst_error_scale_lsb")
    if unsupported:
        raise NotImplementedError(
            "Unsupported photonic perturbation options in the current bounded proxy: "
            + ", ".join(unsupported)
        )

    quantized, scale = quantize_symmetric(
        tensor,
        bits,
        scale_override=scale_override,
        channel_dim=resolved_channel_dim,
    )
    if scale is None:
        return tensor
    if sparse_enabled:
        if sparse_activity_callback is not None:
            gated, sparse_stats = apply_sparse_gating(
                quantized,
                sparse_active_fraction,
                resolved_channel_dim,
                sparse_tau_global=sparse_tau_global,
                return_stats=True,
            )
            if sparse_stats is not None:
                sparse_activity_callback(sparse_stats)
        else:
            gated = apply_sparse_gating(
                quantized,
                sparse_active_fraction,
                resolved_channel_dim,
                sparse_tau_global=sparse_tau_global,
            )
    else:
        gated = quantized
    det_applied = gated
    if det_k is not None and det_bsl_max is not None:
        det_applied = apply_det_early_stop_policy(
            gated,
            det_enabled=det_enabled,
            det_k=det_k,
            det_bsl_max=det_bsl_max,
            channel_dim=resolved_channel_dim,
            det_mode=det_mode,
        )
    noisy = add_gaussian_noise(
        det_applied,
        gaussian_noise_std,
        scale,
        channel_dim=resolved_channel_dim,
        channel_shared=True,
    )
    if _is_mlx_array(noisy) and drift_lsb is not None and float(drift_lsb) != 0.0:
        noisy = noisy + (scale * float(drift_lsb))
    if det_k is None or det_bsl_max is None:
        mixed = apply_det_prefix_mixing(
            noisy,
            det_enabled=det_enabled,
            det_prefix_error_mean=det_prefix_error_mean,
            channel_dim=resolved_channel_dim,
        )
    else:
        mixed = noisy
    return apply_crosstalk(
        mixed,
        alpha,
        resolved_channel_dim,
        bidirectional=bidirectional,
    )


__all__ = [
    "add_gaussian_noise",
    "apply_crosstalk",
    "apply_det_early_stop_policy",
    "apply_det_prefix_mixing",
    "apply_perturb",
    "apply_sparse_gating",
    "compute_symmetric_scale",
    "mlx_available",
    "quantize_symmetric",
    "require_mlx",
]
