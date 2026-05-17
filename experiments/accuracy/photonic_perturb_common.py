"""Shared scalar and shape helpers for photonic perturbation backends."""

from __future__ import annotations


def normalize_channel_dim(
    ndim: int,
    channel_dim: int | None,
) -> int | None:
    """Resolve a possibly-negative channel dim against tensor rank."""
    if channel_dim is None:
        return None
    if ndim <= 0:
        return None
    resolved = int(channel_dim)
    if resolved < 0:
        resolved += ndim
    if resolved < 0 or resolved >= ndim:
        return None
    return resolved


def resolve_quant_max(bits: int | None) -> int | None:
    """Return the symmetric integer quantization bound for ``bits``."""
    if bits is None:
        return None
    bits_value = int(bits)
    if bits_value <= 0:
        return None
    quant_max = 2 ** (bits_value - 1) - 1
    if quant_max <= 0:
        return None
    return quant_max


def resolve_keep_count(total: int, keep_ratio: float) -> int:
    """Map a keep ratio onto a bounded element count."""
    total_value = int(total)
    if total_value <= 0:
        return 0
    ratio_value = float(keep_ratio)
    return max(1, min(total_value, int(round(total_value * ratio_value))))


__all__ = [
    "normalize_channel_dim",
    "resolve_keep_count",
    "resolve_quant_max",
]
