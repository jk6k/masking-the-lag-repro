"""Merge repaired accuracy rows into a canonical accuracy CSV.

The merge is append-safe and deterministic:
- preserve all base rows;
- expand the header with any new metadata columns from the overlay;
- replace existing rows when the logical accuracy context key matches;
- optionally filter overlay rows by experiment ID.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, Any]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _norm_text(value: Any) -> str:
    return str(value or "").strip()


def _norm_numeric_token(value: Any) -> str:
    text = _norm_text(value)
    if not text:
        return ""
    try:
        number = float(text)
    except ValueError:
        return text
    if number.is_integer():
        return str(int(number))
    return format(number, ".15g")


def _infer_baseline_token(row: dict[str, Any]) -> str:
    explicit = _norm_text(row.get("baseline")).lower()
    if explicit in {"true", "false"}:
        return explicit
    note = _norm_text(row.get("notes")).lower()
    if note == "baseline_fp32":
        return "true"
    if note in {"baseline_quant", "config_conditioned_sim", "mtl_sim"}:
        return "false"
    return explicit


def _row_key(row: dict[str, Any]) -> tuple[str, ...]:
    return (
        _norm_text(row.get("run_id")),
        _norm_text(row.get("model")).lower(),
        _infer_baseline_token(row),
        _norm_text(row.get("seed")),
        _norm_numeric_token(row.get("quant_bits")),
        _norm_numeric_token(row.get("noise_sigma_lsb")),
        _norm_numeric_token(row.get("crosstalk_alpha")),
        _norm_numeric_token(row.get("drift_lsb")),
        _norm_numeric_token(row.get("noise_correlation")),
        _norm_numeric_token(row.get("burst_error_prob")),
        _norm_numeric_token(row.get("burst_error_scale_lsb")),
        _norm_numeric_token(row.get("burst_span")),
    )


def _filter_overlay_rows(
    rows: list[dict[str, Any]],
    *,
    include_experiments: set[str],
) -> list[dict[str, Any]]:
    if not include_experiments:
        return rows
    out: list[dict[str, Any]] = []
    for row in rows:
        exp = _norm_text(row.get("experiment_id")).upper()
        if exp in include_experiments:
            out.append(row)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge repaired accuracy rows into a canonical accuracy CSV.")
    parser.add_argument("--base_csv", required=True, help="Existing canonical accuracy CSV.")
    parser.add_argument("--overlay_csv", required=True, help="Repair/batch accuracy CSV to merge.")
    parser.add_argument("--out_csv", required=True, help="Merged output CSV.")
    parser.add_argument(
        "--include_experiments",
        default="",
        help="Optional comma list of experiment IDs to keep from the overlay (e.g. E3,E4).",
    )
    parser.add_argument(
        "--report_json",
        default="",
        help="Optional JSON summary path.",
    )
    args = parser.parse_args()

    base_path = Path(args.base_csv)
    overlay_path = Path(args.overlay_csv)
    out_path = Path(args.out_csv)
    report_path = Path(args.report_json) if str(args.report_json).strip() else None

    base_fields, base_rows = _read_csv(base_path)
    overlay_fields, overlay_rows = _read_csv(overlay_path)
    include_experiments = {
        token.strip().upper()
        for token in str(args.include_experiments or "").split(",")
        if token.strip()
    }
    overlay_rows = _filter_overlay_rows(
        overlay_rows,
        include_experiments=include_experiments,
    )

    merged_fields = list(base_fields)
    for field in overlay_fields:
        if field not in merged_fields:
            merged_fields.append(field)

    merged_rows = [dict(row) for row in base_rows]
    key_to_index = {_row_key(row): idx for idx, row in enumerate(merged_rows)}

    replaced = 0
    inserted = 0
    for row in overlay_rows:
        key = _row_key(row)
        row_copy = dict(row)
        if key in key_to_index:
            merged_rows[key_to_index[key]] = row_copy
            replaced += 1
        else:
            key_to_index[key] = len(merged_rows)
            merged_rows.append(row_copy)
            inserted += 1

    _write_csv(out_path, merged_fields, merged_rows)

    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "base_csv": str(base_path),
            "overlay_csv": str(overlay_path),
            "out_csv": str(out_path),
            "base_rows": len(base_rows),
            "overlay_rows_kept": len(overlay_rows),
            "inserted_rows": inserted,
            "replaced_rows": replaced,
            "merged_rows": len(merged_rows),
            "include_experiments": sorted(include_experiments),
        }
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "out_csv": str(out_path),
                "base_rows": len(base_rows),
                "overlay_rows_kept": len(overlay_rows),
                "inserted_rows": inserted,
                "replaced_rows": replaced,
                "merged_rows": len(merged_rows),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
