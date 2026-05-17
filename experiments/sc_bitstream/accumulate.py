"""Accumulation helpers for stochastic product streams."""

from __future__ import annotations

from typing import Iterable

from .encoding import (
    BIPOLAR,
    XNOR,
    decode_dot_product_from_total_count,
    decode_scalar_product_from_count,
    normalize_encoding_mode,
    normalize_multiplier_mode,
)


def bitcount(bits: Iterable[int | bool]) -> int:
    return sum(1 for bit in bits if bool(bit))


def pca_cycle_counts(product_streams: Iterable[tuple[int, ...] | list[int]]) -> tuple[int, ...]:
    streams = [tuple(stream) for stream in product_streams]
    if not streams:
        return tuple()
    stream_length = len(streams[0])
    if any(len(stream) != stream_length for stream in streams):
        raise ValueError("All product streams must have the same length.")
    return tuple(sum(int(stream[index]) for stream in streams) for index in range(stream_length))


def pca_total_count(product_streams: Iterable[tuple[int, ...] | list[int]]) -> int:
    return sum(pca_cycle_counts(product_streams))


def decode_product_counts(
    counts: int | tuple[int, ...] | list[int],
    *,
    stream_length: int,
    encoding_mode: str = BIPOLAR,
    multiplier_mode: str = XNOR,
) -> float | tuple[float, ...]:
    normalize_encoding_mode(encoding_mode)
    normalize_multiplier_mode(multiplier_mode)
    if isinstance(counts, int):
        return decode_scalar_product_from_count(
            counts,
            stream_length=stream_length,
            encoding_mode=encoding_mode,
            multiplier_mode=multiplier_mode,
        )
    return tuple(
        decode_scalar_product_from_count(
            int(count),
            stream_length=stream_length,
            encoding_mode=encoding_mode,
            multiplier_mode=multiplier_mode,
        )
        for count in counts
    )


def decode_total_dot_count(
    total_count: int,
    *,
    stream_length: int,
    vector_length: int,
    encoding_mode: str = BIPOLAR,
    multiplier_mode: str = XNOR,
) -> float:
    return decode_dot_product_from_total_count(
        total_count,
        stream_length=stream_length,
        vector_length=vector_length,
        encoding_mode=encoding_mode,
        multiplier_mode=multiplier_mode,
    )
