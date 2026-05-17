"""Bitstream-level stochastic computing helpers."""

from .accumulate import bitcount, decode_product_counts, pca_cycle_counts, pca_total_count
from .calibration import (
    aggregate_smoke_summary_envelopes,
    run_stream_length_sweep,
    write_smoke_calibration_pack_csv,
)
from .encoding import (
    decode_dot_product_from_total_count,
    decode_scalar_product_from_count,
    normalize_encoding_mode,
    normalize_multiplier_mode,
    probability_to_value,
    value_to_probability,
)
from .generators import (
    generate_bitstream,
    generate_bitstreams,
    run_stream_correlation_probe,
    run_stream_correlation_scenarios,
    summarize_stream_correlations,
)
from .kernels import estimate_dot_product, estimate_scalar_product
from .multiply import and_bit, multiply_streams, xnor_bit

__all__ = [
    "and_bit",
    "aggregate_smoke_summary_envelopes",
    "bitcount",
    "decode_dot_product_from_total_count",
    "decode_product_counts",
    "decode_scalar_product_from_count",
    "estimate_dot_product",
    "estimate_scalar_product",
    "generate_bitstream",
    "generate_bitstreams",
    "multiply_streams",
    "normalize_encoding_mode",
    "normalize_multiplier_mode",
    "pca_cycle_counts",
    "pca_total_count",
    "probability_to_value",
    "run_stream_correlation_probe",
    "run_stream_correlation_scenarios",
    "run_stream_length_sweep",
    "summarize_stream_correlations",
    "value_to_probability",
    "write_smoke_calibration_pack_csv",
    "xnor_bit",
]
