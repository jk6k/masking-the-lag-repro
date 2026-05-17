"""Shared semantics helpers for the future bitstream accuracy lane."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

BITSTREAM_MODEL_LEVEL_MEASURED_TRUTH_CLASS = "bitstream_model_level_measured"
BITSTREAM_BRIDGE_MEASUREMENT_TRUTH_CLASS = "bridge_only_nonbitstream_measured"
BITSTREAM_LIMITED_SURFACE_PILOT_TRUTH_CLASS = "bitstream_limited_surface_pilot"
PROMOTABLE_BITSTREAM_MEASUREMENT_TRUTH_CLASSES = frozenset(
    {BITSTREAM_MODEL_LEVEL_MEASURED_TRUTH_CLASS}
)


@dataclass(frozen=True)
class BitstreamSemanticsConfig:
    execution_semantics: str
    encoding_mode: str
    multiplier_mode: str
    accumulator_mode: str
    stream_length: int
    generator: str
    calibration_source: str | None = None
    sign_mapping: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def normalize_bitstream_semantics(payload: dict[str, Any] | None) -> BitstreamSemanticsConfig:
    cfg = payload or {}
    execution_semantics = str(cfg.get("execution_semantics") or "bitstream").strip().lower()
    if execution_semantics != "bitstream":
        raise ValueError(
            "bitstream slice semantics must use execution_semantics='bitstream'."
        )
    encoding_mode = str(cfg.get("encoding_mode") or "bipolar").strip().lower()
    multiplier_mode = str(cfg.get("multiplier_mode") or "xnor").strip().lower()
    accumulator_mode = str(cfg.get("accumulator_mode") or "bitcount").strip().lower()
    generator = str(cfg.get("generator") or "bernoulli").strip().lower()
    stream_length = int(cfg.get("stream_length") or 0)
    if stream_length <= 0:
        raise ValueError("stream_length must be positive for bitstream semantics.")
    return BitstreamSemanticsConfig(
        execution_semantics=execution_semantics,
        encoding_mode=encoding_mode,
        multiplier_mode=multiplier_mode,
        accumulator_mode=accumulator_mode,
        stream_length=stream_length,
        generator=generator,
        calibration_source=(
            str(cfg.get("calibration_source")).strip()
            if cfg.get("calibration_source") not in {None, ""}
            else None
        ),
        sign_mapping=(
            str(cfg.get("sign_mapping")).strip()
            if cfg.get("sign_mapping") not in {None, ""}
            else None
        ),
    )
