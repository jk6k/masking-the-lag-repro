"""Bounded scalar and vector kernels for true bitstream stochastic computing."""

from __future__ import annotations

from typing import Iterable

import numpy as np

from .accumulate import bitcount, decode_total_dot_count, pca_cycle_counts, pca_total_count
from .encoding import BIPOLAR, XNOR, normalize_encoding_mode, normalize_multiplier_mode
from .generators import generate_bitstream, generate_bitstreams
from .multiply import multiply_streams


def _validate_combination(encoding_mode: str, multiplier_mode: str) -> None:
    mode = normalize_encoding_mode(encoding_mode)
    mult = normalize_multiplier_mode(multiplier_mode)
    supported = {(BIPOLAR, XNOR), ("unipolar", "and")}
    if (mode, mult) not in supported:
        raise ValueError(
            "Unsupported encoding/multiplier combination: "
            f"{mode}/{mult}. Expected bipolar/xnor or unipolar/and."
        )


def _materialize_stream_tuple(
    streams: Iterable[tuple[int, ...] | list[int]],
) -> tuple[tuple[int, ...], ...]:
    if isinstance(streams, tuple) and all(isinstance(stream, tuple) for stream in streams):
        return streams
    return tuple(tuple(stream) for stream in streams)


def _materialize_stream_array(
    streams: np.ndarray | Iterable[tuple[int, ...] | list[int]],
    *,
    stream_length: int,
) -> np.ndarray:
    resolved_stream_length = int(stream_length)
    if isinstance(streams, np.ndarray):
        array = streams.astype(np.uint8, copy=False)
    else:
        tupled = _materialize_stream_tuple(streams)
        if not tupled:
            return np.empty((0, resolved_stream_length), dtype=np.uint8)
        array = np.asarray(tupled, dtype=np.uint8)
    if array.ndim == 1 and array.size == 0:
        return np.empty((0, resolved_stream_length), dtype=np.uint8)
    if array.ndim != 2:
        raise ValueError("Stream arrays must be rank-2.")
    if int(array.shape[1]) != resolved_stream_length:
        raise ValueError("Stream length mismatch while estimating dot product from streams.")
    return array


def _count_total_product_bits_from_arrays(
    lhs_array: np.ndarray,
    rhs_array: np.ndarray,
    *,
    multiplier_mode: str,
    include_per_element_counts: bool,
) -> tuple[int, tuple[int, ...]]:
    if lhs_array.shape != rhs_array.shape:
        raise ValueError("lhs_streams and rhs_streams must have the same shape.")
    if lhs_array.size == 0:
        return 0, tuple()
    if normalize_multiplier_mode(multiplier_mode) == XNOR:
        per_element_counts_array = np.count_nonzero(lhs_array == rhs_array, axis=1)
    else:
        per_element_counts_array = np.count_nonzero(
            np.logical_and(lhs_array, rhs_array),
            axis=1,
        )
    total_count = int(np.sum(per_element_counts_array, dtype=np.int64))
    if not include_per_element_counts:
        return total_count, tuple()
    return total_count, tuple(int(count) for count in per_element_counts_array.tolist())


def _count_total_product_bits(
    lhs_tuple: tuple[tuple[int, ...], ...],
    rhs_tuple: tuple[tuple[int, ...], ...],
    *,
    stream_length: int,
    multiplier_mode: str,
    include_per_element_counts: bool,
) -> tuple[int, tuple[int, ...]]:
    resolved_stream_length = int(stream_length)
    for lhs_stream, rhs_stream in zip(lhs_tuple, rhs_tuple):
        if len(lhs_stream) != len(rhs_stream):
            raise ValueError("All stream pairs must have equal length.")
        if len(lhs_stream) != resolved_stream_length:
            raise ValueError(
                "Stream length mismatch while estimating dot product from streams."
            )
    if not lhs_tuple:
        return 0, tuple()
    lhs_array = _materialize_stream_array(lhs_tuple, stream_length=stream_length)
    rhs_array = _materialize_stream_array(rhs_tuple, stream_length=stream_length)
    return _count_total_product_bits_from_arrays(
        lhs_array,
        rhs_array,
        multiplier_mode=multiplier_mode,
        include_per_element_counts=include_per_element_counts,
    )


def estimate_scalar_product(
    lhs_value: float,
    rhs_value: float,
    *,
    stream_length: int,
    generator: str = "bernoulli",
    encoding_mode: str = BIPOLAR,
    multiplier_mode: str = XNOR,
    seed: int | None = 0,
    phase: int = 0,
) -> dict[str, object]:
    _validate_combination(encoding_mode, multiplier_mode)
    lhs_stream = generate_bitstream(
        lhs_value,
        stream_length=stream_length,
        generator=generator,
        encoding_mode=encoding_mode,
        seed=seed,
        phase=phase,
    )
    rhs_stream = generate_bitstream(
        rhs_value,
        stream_length=stream_length,
        generator=generator,
        encoding_mode=encoding_mode,
        seed=None if seed is None else seed + 104729,
        phase=phase + 1,
    )
    product_stream = multiply_streams(
        lhs_stream,
        rhs_stream,
        multiplier_mode=multiplier_mode,
    )
    product_count = bitcount(product_stream)
    estimated_value = decode_total_dot_count(
        product_count,
        stream_length=stream_length,
        vector_length=1,
        encoding_mode=encoding_mode,
        multiplier_mode=multiplier_mode,
    )
    exact_value = float(lhs_value) * float(rhs_value)
    return {
        "lhs_stream": lhs_stream,
        "rhs_stream": rhs_stream,
        "product_stream": product_stream,
        "product_count": product_count,
        "estimated_value": estimated_value,
        "exact_value": exact_value,
        "abs_error": abs(estimated_value - exact_value),
    }


def estimate_dot_product(
    lhs_values: Iterable[float],
    rhs_values: Iterable[float],
    *,
    stream_length: int,
    generator: str = "bernoulli",
    encoding_mode: str = BIPOLAR,
    multiplier_mode: str = XNOR,
    seed: int | None = 0,
    phase: int = 0,
    compute_exact_value: bool = True,
) -> dict[str, object]:
    _validate_combination(encoding_mode, multiplier_mode)
    lhs_list = tuple(float(value) for value in lhs_values)
    rhs_list = tuple(float(value) for value in rhs_values)
    if len(lhs_list) != len(rhs_list):
        raise ValueError("lhs_values and rhs_values must have the same length.")
    lhs_streams = generate_bitstreams(
        lhs_list,
        stream_length=stream_length,
        generator=generator,
        encoding_mode=encoding_mode,
        seed=seed,
        phase=phase,
    )
    rhs_streams = generate_bitstreams(
        rhs_list,
        stream_length=stream_length,
        generator=generator,
        encoding_mode=encoding_mode,
        seed=None if seed is None else seed + 130363,
        phase=phase + 1,
    )
    return estimate_dot_product_from_streams(
        lhs_streams,
        rhs_streams,
        stream_length=stream_length,
        encoding_mode=encoding_mode,
        multiplier_mode=multiplier_mode,
        lhs_values=lhs_list,
        rhs_values=rhs_list,
        include_debug_artifacts=True,
        compute_exact_value=compute_exact_value,
    )


def estimate_dot_product_value(
    lhs_values: Iterable[float],
    rhs_values: Iterable[float],
    *,
    stream_length: int,
    generator: str = "bernoulli",
    encoding_mode: str = BIPOLAR,
    multiplier_mode: str = XNOR,
    seed: int | None = 0,
    phase: int = 0,
) -> float:
    _validate_combination(encoding_mode, multiplier_mode)
    lhs_list = tuple(float(value) for value in lhs_values)
    rhs_list = tuple(float(value) for value in rhs_values)
    if len(lhs_list) != len(rhs_list):
        raise ValueError("lhs_values and rhs_values must have the same length.")
    lhs_streams = generate_bitstreams(
        lhs_list,
        stream_length=stream_length,
        generator=generator,
        encoding_mode=encoding_mode,
        seed=seed,
        phase=phase,
    )
    rhs_streams = generate_bitstreams(
        rhs_list,
        stream_length=stream_length,
        generator=generator,
        encoding_mode=encoding_mode,
        seed=None if seed is None else seed + 130363,
        phase=phase + 1,
    )
    return estimate_dot_product_value_from_streams(
        lhs_streams,
        rhs_streams,
        stream_length=stream_length,
        encoding_mode=encoding_mode,
        multiplier_mode=multiplier_mode,
    )


def estimate_dot_product_value_from_streams(
    lhs_streams: Iterable[tuple[int, ...] | list[int]],
    rhs_streams: Iterable[tuple[int, ...] | list[int]],
    *,
    stream_length: int,
    encoding_mode: str = BIPOLAR,
    multiplier_mode: str = XNOR,
) -> float:
    return estimate_dot_product_value_from_stream_arrays(
        lhs_streams,
        rhs_streams,
        stream_length=stream_length,
        encoding_mode=encoding_mode,
        multiplier_mode=multiplier_mode,
    )


def estimate_dot_product_value_from_stream_arrays(
    lhs_stream_arrays: np.ndarray | Iterable[tuple[int, ...] | list[int]],
    rhs_stream_arrays: np.ndarray | Iterable[tuple[int, ...] | list[int]],
    *,
    stream_length: int,
    encoding_mode: str = BIPOLAR,
    multiplier_mode: str = XNOR,
) -> float:
    _validate_combination(encoding_mode, multiplier_mode)
    lhs_array = _materialize_stream_array(lhs_stream_arrays, stream_length=stream_length)
    rhs_array = _materialize_stream_array(rhs_stream_arrays, stream_length=stream_length)
    total_count, _ = _count_total_product_bits_from_arrays(
        lhs_array,
        rhs_array,
        multiplier_mode=multiplier_mode,
        include_per_element_counts=False,
    )
    return decode_total_dot_count(
        total_count,
        stream_length=stream_length,
        vector_length=int(lhs_array.shape[0]),
        encoding_mode=encoding_mode,
        multiplier_mode=multiplier_mode,
    )


def estimate_dot_product_from_streams(
    lhs_streams: Iterable[tuple[int, ...] | list[int]],
    rhs_streams: Iterable[tuple[int, ...] | list[int]],
    *,
    stream_length: int,
    encoding_mode: str = BIPOLAR,
    multiplier_mode: str = XNOR,
    lhs_values: Iterable[float] | None = None,
    rhs_values: Iterable[float] | None = None,
    include_debug_artifacts: bool = False,
    compute_exact_value: bool = True,
) -> dict[str, object]:
    _validate_combination(encoding_mode, multiplier_mode)
    lhs_tuple = _materialize_stream_tuple(lhs_streams)
    rhs_tuple = _materialize_stream_tuple(rhs_streams)
    if len(lhs_tuple) != len(rhs_tuple):
        raise ValueError("lhs_streams and rhs_streams must have the same length.")

    product_streams: tuple[tuple[int, ...], ...] = tuple()
    if include_debug_artifacts:
        product_streams = tuple(
            multiply_streams(lhs_stream, rhs_stream, multiplier_mode=multiplier_mode)
            for lhs_stream, rhs_stream in zip(lhs_tuple, rhs_tuple)
        )
        per_element_counts = tuple(bitcount(stream) for stream in product_streams)
        cycle_counts = pca_cycle_counts(product_streams)
        total_count = pca_total_count(product_streams)
    else:
        total_count, per_element_counts = _count_total_product_bits(
            lhs_tuple,
            rhs_tuple,
            stream_length=stream_length,
            multiplier_mode=multiplier_mode,
            include_per_element_counts=True,
        )
        cycle_counts = tuple()

    estimated_value = decode_total_dot_count(
        total_count,
        stream_length=stream_length,
        vector_length=len(lhs_tuple),
        encoding_mode=encoding_mode,
        multiplier_mode=multiplier_mode,
    )
    exact_value = None
    if compute_exact_value and lhs_values is not None and rhs_values is not None:
        lhs_list = tuple(float(value) for value in lhs_values)
        rhs_list = tuple(float(value) for value in rhs_values)
        if len(lhs_list) != len(lhs_tuple) or len(rhs_list) != len(rhs_tuple):
            raise ValueError("Value lists must align with stream lists.")
        exact_value = sum(lhs * rhs for lhs, rhs in zip(lhs_list, rhs_list))
    return {
        "lhs_streams": lhs_tuple,
        "rhs_streams": rhs_tuple,
        "product_streams": product_streams,
        "per_element_counts": per_element_counts,
        "cycle_counts": cycle_counts,
        "total_count": total_count,
        "estimated_value": estimated_value,
        "exact_value": exact_value,
        "abs_error": (
            None if exact_value is None else abs(estimated_value - exact_value)
        ),
    }
