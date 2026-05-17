#!/usr/bin/env python3
"""Build the appendix-style ablation table for the fuller implementation lane."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

try:
    from .path_policy import MAIN_PROJECT_REPORT_DATA_DIR
except ImportError:
    from path_policy import MAIN_PROJECT_REPORT_DATA_DIR  # type: ignore


DEFAULT_INPUT = (
    MAIN_PROJECT_REPORT_DATA_DIR / "fuller_ablation_summary_20260319_fullerexp_v1.csv"
)
DEFAULT_OUTPUT = (
    MAIN_PROJECT_REPORT_DATA_DIR / "fuller_ablation_appendix_table_20260319_fullerexp_v1.csv"
)

OUTPUT_FIELDS = [
    "mechanism_variant",
    "latency_ms",
    "energy_j",
    "avg_power_w",
    "acc_top1",
    "acc_drop_pp",
    "delta_latency_pct",
    "delta_energy_pct",
    "delta_acc_drop_pp",
]

ABLATION_SUMMARY_FIELDS = [
    "mechanism_variant",
    "latency_ms",
    "energy_j",
    "avg_power_w",
    "acc_top1",
    "acc_drop_pp",
    "stage_cycles",
    "bubble_cycles",
    "utilization_avg",
]


def _to_float(value: object) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _pct_delta(value: float | None, ref: float | None) -> float | None:
    if value is None or ref is None or ref == 0.0:
        return None
    return ((value - ref) / ref) * 100.0


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in OUTPUT_FIELDS})


def build_appendix_table(input_csv: Path) -> list[dict[str, object]]:
    if not input_csv.is_file():
        raise SystemExit(f"Missing ablation summary csv: {input_csv}")

    with input_csv.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit(f"No rows in ablation summary csv: {input_csv}")

    baseline_row = None
    for row in rows:
        if str(row.get("mechanism_variant") or "").strip().upper() == "ASTRA":
            baseline_row = row
            break
    if baseline_row is None:
        raise SystemExit("Ablation summary is missing the ASTRA baseline row.")

    ref_latency = _to_float(baseline_row.get("latency_ms"))
    ref_energy = _to_float(baseline_row.get("energy_j"))
    ref_drop = _to_float(baseline_row.get("acc_drop_pp"))

    output_rows: list[dict[str, object]] = []
    for row in rows:
        latency = _to_float(row.get("latency_ms"))
        energy = _to_float(row.get("energy_j"))
        acc_drop = _to_float(row.get("acc_drop_pp"))
        output_rows.append(
            {
                "mechanism_variant": row.get("mechanism_variant", ""),
                "latency_ms": latency,
                "energy_j": energy,
                "avg_power_w": _to_float(row.get("avg_power_w")),
                "acc_top1": _to_float(row.get("acc_top1")),
                "acc_drop_pp": acc_drop,
                "delta_latency_pct": _pct_delta(latency, ref_latency),
                "delta_energy_pct": _pct_delta(energy, ref_energy),
                "delta_acc_drop_pp": (
                    None
                    if acc_drop is None or ref_drop is None
                    else acc_drop - ref_drop
                ),
            }
        )
    return output_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the fuller ablation appendix table.")
    parser.add_argument("--input_csv", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out_csv", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    rows = build_appendix_table(args.input_csv)
    _write_csv(args.out_csv, rows)
    print(f"[fuller-ablation-appendix] wrote {args.out_csv}")


if __name__ == "__main__":
    main()
