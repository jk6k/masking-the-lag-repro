#!/usr/bin/env python3
"""Stamp measured accuracy CSV rows with explicit bitstream semantics."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from accuracy.bitstream_semantics import (
    BITSTREAM_BRIDGE_MEASUREMENT_TRUTH_CLASS,
    BitstreamSemanticsConfig,
    normalize_bitstream_semantics,
)

CONTRACT_FIELDS = [
    "execution_semantics",
    "bitstream_generator",
    "bitstream_stream_length",
    "bitstream_encoding_mode",
    "bitstream_multiplier_mode",
    "bitstream_accumulator_mode",
    "bitstream_calibration_source",
    "bitstream_sign_mapping",
    "bitstream_measurement_truth_class",
    "accuracy_measurement_contract_note",
]


def _read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = [dict(row) for row in reader]
    return fieldnames, rows


def _write_rows(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _normalize_filters(filters: dict[str, str] | None) -> dict[str, str]:
    normalized = dict(filters or {})
    if normalized:
        return {str(key): str(value) for key, value in normalized.items()}
    return {"baseline": "false"}


def _row_matches_filters(row: dict[str, Any], filters: dict[str, str]) -> bool:
    for key, expected in filters.items():
        observed = str(row.get(key) or "").strip().lower()
        if observed != str(expected).strip().lower():
            return False
    return True


def annotate_accuracy_rows(
    rows: list[dict[str, Any]],
    *,
    semantics_cfg: BitstreamSemanticsConfig,
    row_filters: dict[str, str] | None = None,
    contract_note: str = "",
    measurement_truth_class: str = BITSTREAM_BRIDGE_MEASUREMENT_TRUTH_CLASS,
    extra_fields: dict[str, Any] | None = None,
) -> tuple[list[str], list[dict[str, Any]], int]:
    filters = _normalize_filters(row_filters)
    annotated_rows: list[dict[str, Any]] = []
    matched_count = 0
    normalized_extra_fields = {
        str(key): value for key, value in (extra_fields or {}).items() if str(key).strip()
    }

    fieldnames: list[str] = list(rows[0].keys()) if rows else []
    for field in CONTRACT_FIELDS:
        if field not in fieldnames:
            fieldnames.append(field)
    for field in normalized_extra_fields:
        if field not in fieldnames:
            fieldnames.append(field)

    for row in rows:
        updated = dict(row)
        if _row_matches_filters(updated, filters):
            matched_count += 1
            updated["execution_semantics"] = semantics_cfg.execution_semantics
            updated["bitstream_generator"] = semantics_cfg.generator
            updated["bitstream_stream_length"] = str(semantics_cfg.stream_length)
            updated["bitstream_encoding_mode"] = semantics_cfg.encoding_mode
            updated["bitstream_multiplier_mode"] = semantics_cfg.multiplier_mode
            updated["bitstream_accumulator_mode"] = semantics_cfg.accumulator_mode
            updated["bitstream_calibration_source"] = semantics_cfg.calibration_source or ""
            updated["bitstream_sign_mapping"] = semantics_cfg.sign_mapping or ""
            updated["bitstream_measurement_truth_class"] = measurement_truth_class
            updated["accuracy_measurement_contract_note"] = contract_note
            for field, value in normalized_extra_fields.items():
                updated[field] = value
        annotated_rows.append(updated)

    if matched_count <= 0:
        raise ValueError(f"No accuracy rows matched filters: {filters}")
    return fieldnames, annotated_rows, matched_count


def _parse_match_filters(raw_filters: list[str]) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for item in raw_filters:
        if "=" not in item:
            raise ValueError(f"Invalid --match filter: {item!r}")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Invalid --match filter: {item!r}")
        parsed[key] = value.strip()
    return parsed


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Annotate measured accuracy CSV rows with bitstream semantics fields.",
    )
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--stream_length", required=True, type=int)
    parser.add_argument("--generator", required=True)
    parser.add_argument("--encoding_mode", default="bipolar")
    parser.add_argument("--multiplier_mode", default="xnor")
    parser.add_argument("--accumulator_mode", default="bitcount")
    parser.add_argument("--execution_semantics", default="bitstream")
    parser.add_argument("--calibration_source", default="")
    parser.add_argument("--sign_mapping", default="")
    parser.add_argument("--contract_note", default="")
    parser.add_argument(
        "--match",
        action="append",
        default=[],
        help="Field filter in key=value form. Defaults to baseline=false when omitted.",
    )
    return parser


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()

    semantics_cfg = normalize_bitstream_semantics(
        {
            "execution_semantics": args.execution_semantics,
            "encoding_mode": args.encoding_mode,
            "multiplier_mode": args.multiplier_mode,
            "accumulator_mode": args.accumulator_mode,
            "stream_length": args.stream_length,
            "generator": args.generator,
            "calibration_source": args.calibration_source,
            "sign_mapping": args.sign_mapping,
        }
    )
    _, rows = _read_rows(Path(args.input_csv))
    filters = _parse_match_filters(args.match)
    output_fieldnames, annotated_rows, matched_count = annotate_accuracy_rows(
        rows,
        semantics_cfg=semantics_cfg,
        row_filters=filters,
        contract_note=args.contract_note,
    )
    _write_rows(Path(args.output_csv), output_fieldnames, annotated_rows)
    print(
        f"Annotated {matched_count} measured accuracy rows with bitstream semantics: {args.output_csv}"
    )


if __name__ == "__main__":
    main()
