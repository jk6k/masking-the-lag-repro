"""Bitwise multiplication primitives for stochastic streams."""

from __future__ import annotations

from .encoding import AND, XNOR, normalize_multiplier_mode


def _normalize_bit(value: int | bool) -> int:
    if value in {0, False}:
        return 0
    if value in {1, True}:
        return 1
    raise ValueError(f"Expected bit-like value 0/1, got {value!r}")


def xnor_bit(lhs: int | bool, rhs: int | bool) -> int:
    return 1 if _normalize_bit(lhs) == _normalize_bit(rhs) else 0


def and_bit(lhs: int | bool, rhs: int | bool) -> int:
    return _normalize_bit(lhs) & _normalize_bit(rhs)


def multiply_streams(
    lhs_stream: tuple[int, ...] | list[int],
    rhs_stream: tuple[int, ...] | list[int],
    *,
    multiplier_mode: str = XNOR,
) -> tuple[int, ...]:
    if len(lhs_stream) != len(rhs_stream):
        raise ValueError("lhs_stream and rhs_stream must have the same length.")
    mode = normalize_multiplier_mode(multiplier_mode)
    if mode == XNOR:
        return tuple(xnor_bit(lhs, rhs) for lhs, rhs in zip(lhs_stream, rhs_stream))
    if mode == AND:
        return tuple(and_bit(lhs, rhs) for lhs, rhs in zip(lhs_stream, rhs_stream))
    raise ValueError(f"Unsupported multiplier mode: {multiplier_mode!r}")
