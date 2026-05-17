"""Quantization + Gaussian-noise + crosstalk perturbations for photonic simulation."""

from __future__ import annotations

from typing import Any, Callable

import torch

from accuracy.photonic_perturb_common import (
    normalize_channel_dim,
    resolve_keep_count,
    resolve_quant_max,
)


_MISSING = object()


def _normalize_channel_dim(
    tensor: torch.Tensor, channel_dim: int | None
) -> int | None:
    return normalize_channel_dim(tensor.ndim, channel_dim)


def compute_symmetric_scale(
    tensor: torch.Tensor | None,
    bits: int | None,
    *,
    eps: float = 1e-8,
    channel_dim: int | None = None,
) -> torch.Tensor | None:
    """Compute per-tensor or per-channel symmetric quantization scale."""
    if not torch.is_tensor(tensor):
        return None
    quant_max = resolve_quant_max(bits)
    if quant_max is None:
        return None
    channel_dim = _normalize_channel_dim(tensor, channel_dim)
    if channel_dim is None:
        max_value = tensor.detach().abs().max()
    else:
        reduce_dims = [dim for dim in range(tensor.ndim) if dim != channel_dim]
        if reduce_dims:
            max_value = tensor.detach().abs().amax(dim=reduce_dims, keepdim=True)
        else:
            max_value = tensor.detach().abs().max()
    scale = (max_value / quant_max).clamp(min=eps)
    return scale


def quantize_symmetric(
    tensor: torch.Tensor | None,
    bits: int | None,
    *,
    eps: float = 1e-8,
    scale_override: torch.Tensor | float | None = None,
    channel_dim: int | None = None,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Quantize a tensor symmetrically and return (quantized, scale)."""
    if not torch.is_tensor(tensor):
        return tensor, None
    quant_max = resolve_quant_max(bits)
    if quant_max is None:
        return tensor, None
    if scale_override is None:
        scale = compute_symmetric_scale(
            tensor, bits, eps=eps, channel_dim=channel_dim
        )
    else:
        if torch.is_tensor(scale_override):
            scale = scale_override.to(device=tensor.device, dtype=tensor.dtype)
        else:
            scale = torch.tensor(scale_override, device=tensor.device, dtype=tensor.dtype)
        scale = scale.clamp(min=eps)
    if scale is None:
        return tensor, None
    quantized = torch.clamp(torch.round(tensor / scale), -quant_max, quant_max) * scale
    return quantized, scale


def add_gaussian_noise(
    tensor: torch.Tensor | None,
    noise_std: float | None,
    scale: torch.Tensor | None,
    *,
    channel_dim: int | None = None,
    channel_shared: bool = False,
) -> torch.Tensor | None:
    """Add mapping-induced Gaussian noise with sigma specified in LSB units."""
    if not torch.is_tensor(tensor):
        return tensor
    if noise_std is None or scale is None:
        return tensor
    std_value = float(noise_std)
    if std_value <= 0:
        return tensor
    if channel_shared:
        channel_dim = _normalize_channel_dim(tensor, channel_dim)
        if channel_dim is None:
            noise = torch.randn(
                [1],
                device=tensor.device,
                dtype=tensor.dtype,
            )
        else:
            noise_shape = [1] * tensor.ndim
            noise_shape[channel_dim] = tensor.shape[channel_dim]
            noise = torch.randn(
                noise_shape,
                device=tensor.device,
                dtype=tensor.dtype,
            )
    else:
        noise = torch.randn_like(tensor)
    noise = noise * (scale * std_value)
    return tensor + noise


def _shift_with_zero_padding(
    tensor: torch.Tensor,
    *,
    shifts: int,
    channel_dim: int,
) -> torch.Tensor:
    """Shift along channel_dim without introducing cyclic wraparound."""
    shifted = torch.roll(tensor, shifts=shifts, dims=channel_dim).clone()
    if shifts == 0:
        return shifted
    edge = [slice(None)] * tensor.ndim
    if shifts > 0:
        edge[channel_dim] = slice(0, shifts)
    else:
        edge[channel_dim] = slice(shifts, None)
    shifted[tuple(edge)] = 0
    return shifted


def apply_crosstalk(
    tensor: torch.Tensor | None,
    alpha: float | None,
    channel_dim: int | None,
    *,
    bidirectional: bool = True,
) -> torch.Tensor | None:
    """Apply neighbor-channel crosstalk along channel_dim.

    When ``bidirectional=True`` (default), both left and right neighbors
    contribute equally, each with weight ``alpha/2``.  When ``False``,
    only a single-direction roll is applied (legacy behaviour).
    """
    if not torch.is_tensor(tensor):
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
            + shifted_left * (alpha_value / 2)
            + shifted_right * (alpha_value / 2)
        )
    else:
        shifted = _shift_with_zero_padding(
            tensor,
            shifts=1,
            channel_dim=channel_dim,
        )
    return tensor * (1 - alpha_value) + shifted * alpha_value


def apply_det_prefix_mixing(
    tensor: torch.Tensor | None,
    det_enabled: bool,
    det_prefix_error_mean: float | None,
    channel_dim: int | None,
) -> torch.Tensor | None:
    """Inject a deterministic DET prefix-mixing error along channel_dim."""
    if not torch.is_tensor(tensor):
        return tensor
    if not det_enabled:
        return tensor
    if det_prefix_error_mean is None:
        return tensor
    mix = float(det_prefix_error_mean)
    if mix <= 0:
        return tensor
    if channel_dim is None:
        channel_dim = -1
    shifted = torch.roll(tensor, shifts=1, dims=channel_dim)
    return tensor * (1.0 - mix) + shifted * mix


def apply_det_early_stop_policy(
    tensor: torch.Tensor | None,
    *,
    det_enabled: bool,
    det_k: float | None,
    det_bsl_max: float | None,
    channel_dim: int | None,
    det_mode: str | None = None,
) -> torch.Tensor | None:
    """Apply a simple policy-faithful DET early-stop path from k/bsl_max.

    The accuracy harness cannot replay the full SC bitstream, but it can apply
    the actual control variable of the adaptive policy: per-layer effective
    prefix length k. We model early stop by limiting the effective contribution
    of the channel dimension according to k / bsl_max rather than injecting a
    precomputed scalar error surrogate.
    """
    if not torch.is_tensor(tensor):
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
        return torch.zeros_like(tensor)
    channel_dim = _normalize_channel_dim(tensor, channel_dim)
    if channel_dim is None:
        channel_dim = tensor.ndim - 1
    channels = int(tensor.shape[channel_dim])
    if channels <= 0:
        return tensor
    keep = resolve_keep_count(channels, keep_ratio)
    if keep >= channels:
        return tensor
    mode = str(det_mode or "reorder").strip().lower()
    kept = tensor.narrow(channel_dim, 0, keep)
    pad_shape = list(tensor.shape)
    pad_shape[channel_dim] = channels - keep
    if mode == "replace" and keep > 0:
        tail_value = kept.narrow(channel_dim, keep - 1, 1)
        repeat_shape = [1] * tensor.ndim
        repeat_shape[channel_dim] = channels - keep
        pad = tail_value.repeat(*repeat_shape)
    else:
        pad = torch.zeros(pad_shape, dtype=tensor.dtype, device=tensor.device)
    return torch.cat([kept, pad], dim=channel_dim)


def apply_sparse_gating(
    tensor: torch.Tensor | None,
    sparse_active_fraction: float | None,
    channel_dim: int | None,
    *,
    sparse_tau_global: float | None = None,
    return_stats: bool = False,
) -> torch.Tensor | None | tuple[torch.Tensor | None, dict[str, Any] | None]:
    """Apply the current sparse proxy used by the accuracy harness.

    If ``sparse_tau_global > 0``, apply a near-zero threshold gate that zeros
    activations whose magnitude falls below ``tau * max(abs(tensor))`` for the
    current tensor. When ``sparse_active_fraction`` is also configured, it is a
    safety floor: the threshold gate is relaxed to keep at least that fraction
    of the largest-magnitude elements. This avoids turning an intended sparse
    operating point such as 75% active into a destructive ~10% active gate.

    If ``sparse_tau_global`` is absent or non-positive, fall back to the legacy
    destructive top-k mask over channels/elements implied by
    ``sparse_active_fraction``. That path is retained for backwards
    compatibility with older proxy runs only.
    """
    if not torch.is_tensor(tensor):
        return (tensor, None) if return_stats else tensor

    def _activity_stats(mask: torch.Tensor, gate_mode: str) -> dict[str, Any]:
        active_elements = int(mask.sum().item())
        total_elements = int(mask.numel())
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
        reference = tensor.detach().abs().max()
        if reference.item() <= 0.0:
            stats = {
                "gate_mode": "tau_threshold",
                "active_elements": int(tensor.numel()),
                "total_elements": int(tensor.numel()),
                "activity_fraction": 1.0,
            }
            return (tensor, stats) if return_stats else tensor
        threshold = reference * sparse_tau
        mask = tensor.detach().abs() >= threshold
        gate_mode = "tau_threshold"
        if sparse_active_fraction is not None:
            active_fraction = float(sparse_active_fraction)
            if 0.0 < active_fraction < 1.0:
                total = int(mask.numel())
                min_keep = resolve_keep_count(total, active_fraction)
                active = int(mask.sum().item())
                if active < min_keep:
                    flat_scores = tensor.detach().abs().reshape(-1)
                    keep_idx = torch.topk(flat_scores, min_keep, sorted=False).indices
                    floor_mask = torch.zeros(total, device=tensor.device, dtype=torch.bool)
                    floor_mask.scatter_(0, keep_idx, True)
                    mask = torch.logical_or(mask.reshape(-1), floor_mask).view_as(mask)
                    gate_mode = "tau_threshold_min_active"
        gated = tensor * mask.to(dtype=tensor.dtype)
        stats = _activity_stats(mask, gate_mode)
        return (gated, stats) if return_stats else gated
    if sparse_active_fraction is None:
        return (tensor, None) if return_stats else tensor
    active_fraction = float(sparse_active_fraction)
    if active_fraction >= 1.0:
        stats = {
            "gate_mode": "topk_proxy",
            "active_elements": int(tensor.numel()),
            "total_elements": int(tensor.numel()),
            "activity_fraction": 1.0,
        }
        return (tensor, stats) if return_stats else tensor
    if active_fraction <= 0.0:
        gated = torch.zeros_like(tensor)
        stats = {
            "gate_mode": "topk_proxy",
            "active_elements": 0,
            "total_elements": int(tensor.numel()),
            "activity_fraction": 0.0,
        }
        return (gated, stats) if return_stats else gated

    channel_dim = _normalize_channel_dim(tensor, channel_dim)
    if channel_dim is None:
        total = int(tensor.numel())
        keep = resolve_keep_count(total, active_fraction)
        if keep >= total:
            stats = {
                "gate_mode": "topk_proxy",
                "active_elements": total,
                "total_elements": total,
                "activity_fraction": 1.0,
            }
            return (tensor, stats) if return_stats else tensor
        scores = tensor.detach().abs().reshape(-1)
        keep_idx = torch.topk(scores, keep, sorted=False).indices
        mask = torch.zeros(total, device=tensor.device, dtype=tensor.dtype)
        mask.scatter_(0, keep_idx, 1.0)
        gated = tensor * mask.view_as(tensor)
        stats = _activity_stats(mask.bool(), "topk_proxy")
        return (gated, stats) if return_stats else gated

    moved = tensor.detach().abs().movedim(channel_dim, 0)
    scores = moved.reshape(moved.shape[0], -1).mean(dim=1)
    channels = int(scores.numel())
    keep = resolve_keep_count(channels, active_fraction)
    if keep >= channels:
        stats = {
            "gate_mode": "topk_proxy",
            "active_elements": int(tensor.numel()),
            "total_elements": int(tensor.numel()),
            "activity_fraction": 1.0,
        }
        return (tensor, stats) if return_stats else tensor
    keep_idx = torch.topk(scores, keep, sorted=False).indices
    mask = torch.zeros(channels, device=tensor.device, dtype=tensor.dtype)
    mask.scatter_(0, keep_idx, 1.0)
    mask_shape = [1] * tensor.ndim
    mask_shape[channel_dim] = channels
    channel_mask = mask.view(mask_shape)
    gated = tensor * channel_mask
    stats = _activity_stats(channel_mask.expand_as(tensor).bool(), "topk_proxy")
    return (gated, stats) if return_stats else gated


def apply_perturb(
    tensor: torch.Tensor | None,
    bits: int | None,
    legacy_sigma_lsb_or_alpha: float | None,
    alpha_or_channel_dim: float | int | None = None,
    channel_dim: object = _MISSING,
    *,
    gaussian_noise_std: float | None = None,
    scale_override: torch.Tensor | float | None = None,
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
) -> torch.Tensor | None:
    """Apply quantization, Gaussian noise, and crosstalk to a tensor."""
    if not torch.is_tensor(tensor):
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

    # Order: quantize -> sparse gating -> DET early stop
    # -> channel-shared Gaussian drift -> static drift -> DET prefix mixing
    # -> output-side crosstalk.
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
    if torch.is_tensor(noisy) and drift_lsb is not None and float(drift_lsb) != 0.0:
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
    mixed = apply_crosstalk(
        mixed,
        alpha,
        resolved_channel_dim,
        bidirectional=bidirectional,
    )
    return mixed


__all__ = [
    "add_gaussian_noise",
    "apply_det_early_stop_policy",
    "apply_det_prefix_mixing",
    "apply_crosstalk",
    "apply_perturb",
    "apply_sparse_gating",
    "compute_symmetric_scale",
    "quantize_symmetric",
]
