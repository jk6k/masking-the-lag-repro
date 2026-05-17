"""Encoding and decode helpers for stochastic bitstreams."""

from __future__ import annotations

BIPOLAR = "bipolar"
UNIPOLAR = "unipolar"
XNOR = "xnor"
AND = "and"


def normalize_encoding_mode(mode: str | None) -> str:
    text = str(mode or BIPOLAR).strip().lower() or BIPOLAR
    if text not in {BIPOLAR, UNIPOLAR}:
        raise ValueError(f"Unsupported encoding mode: {mode!r}")
    return text


def normalize_multiplier_mode(mode: str | None) -> str:
    text = str(mode or XNOR).strip().lower() or XNOR
    if text not in {XNOR, AND}:
        raise ValueError(f"Unsupported multiplier mode: {mode!r}")
    return text


def _clamp_probability(probability: float) -> float:
    return max(0.0, min(1.0, float(probability)))


def value_to_probability(value: float, *, encoding_mode: str = BIPOLAR) -> float:
    mode = normalize_encoding_mode(encoding_mode)
    if mode == BIPOLAR:
        return _clamp_probability((float(value) + 1.0) / 2.0)
    return _clamp_probability(float(value))


def probability_to_value(probability: float, *, encoding_mode: str = BIPOLAR) -> float:
    mode = normalize_encoding_mode(encoding_mode)
    prob = _clamp_probability(probability)
    if mode == BIPOLAR:
        return (2.0 * prob) - 1.0
    return prob


def decode_scalar_product_from_count(
    count: int,
    *,
    stream_length: int,
    encoding_mode: str = BIPOLAR,
    multiplier_mode: str = XNOR,
) -> float:
    if stream_length <= 0:
        raise ValueError("stream_length must be positive.")
    mode = normalize_encoding_mode(encoding_mode)
    mult = normalize_multiplier_mode(multiplier_mode)
    mean = float(count) / float(stream_length)
    if mode == BIPOLAR and mult == XNOR:
        return (2.0 * mean) - 1.0
    if mode == UNIPOLAR and mult == AND:
        return mean
    raise ValueError(
        "Unsupported encoding/multiplier combination: "
        f"{mode}/{mult}. Expected bipolar/xnor or unipolar/and."
    )


def decode_dot_product_from_total_count(
    total_count: int,
    *,
    stream_length: int,
    vector_length: int,
    encoding_mode: str = BIPOLAR,
    multiplier_mode: str = XNOR,
) -> float:
    if stream_length <= 0:
        raise ValueError("stream_length must be positive.")
    if vector_length < 0:
        raise ValueError("vector_length must be non-negative.")
    mode = normalize_encoding_mode(encoding_mode)
    mult = normalize_multiplier_mode(multiplier_mode)
    if mode == BIPOLAR and mult == XNOR:
        return ((2.0 * float(total_count)) / float(stream_length)) - float(vector_length)
    if mode == UNIPOLAR and mult == AND:
        return float(total_count) / float(stream_length)
    raise ValueError(
        "Unsupported encoding/multiplier combination: "
        f"{mode}/{mult}. Expected bipolar/xnor or unipolar/and."
    )
