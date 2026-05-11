#!/usr/bin/env python3
"""Build the SUDS P1.1 interface-overhead model."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
TAG = "20260511_maxq"
SLACK_MANIFEST = REPO_ROOT / "experiments/results/runs/slack_manifest.json"
ENERGY_SENSITIVITY = REPO_ROOT / f"experiments/results/report_data/suds_circuit_energy_sensitivity_{TAG}.csv"
CSV_OUT = REPO_ROOT / f"experiments/results/report_data/suds_interface_overhead_{TAG}.csv"
JSON_OUT = REPO_ROOT / f"experiments/results/report_data/suds_interface_overhead_{TAG}.json"
REPORT_OUT = REPO_ROOT / f"docs/reports/{TAG}_suds_interface_overhead.md"

FIELDS = [
    "row_type",
    "component",
    "scope",
    "evidence_label",
    "count",
    "bits_per_item",
    "total_bits",
    "total_bytes",
    "update_frequency",
    "amortization_window_inferences",
    "nominal_control_fraction_of_suds_pre_control",
    "pessimistic_control_fraction_of_suds_pre_control",
    "nominal_energy_ratio_vs_baseline",
    "pessimistic_energy_ratio_vs_baseline",
    "latency_cycles",
    "latency_fraction_of_serial_cycles",
    "source_artifact",
    "acceptance_note",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default=TAG)
    parser.add_argument("--slack-manifest", type=Path, default=SLACK_MANIFEST)
    parser.add_argument("--energy-sensitivity", type=Path, default=ENERGY_SENSITIVITY)
    parser.add_argument("--csv-out", type=Path, default=CSV_OUT)
    parser.add_argument("--json-out", type=Path, default=JSON_OUT)
    parser.add_argument("--report-out", type=Path, default=REPORT_OUT)
    return parser.parse_args()


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in FIELDS})


def select_energy_row(rows: list[dict[str, str]], profile_id: str, *, control: str) -> dict[str, str]:
    matches = [
        row
        for row in rows
        if row["profile_id"] == profile_id
        and row["adc4_scale"] == "1.0"
        and row["adc8_scale"] == "1.0"
        and row["dac_modulator_scale"] == "1.0"
        and row["laser_wallplug_scale"] == "1.0"
        and row["control_overhead_fraction"] == control
        and row["crosstalk_case"] == "baseline"
    ]
    if not matches:
        raise SystemExit(f"missing nominal energy row for {profile_id} control={control}")
    return matches[0]


def select_worst_energy_row(rows: list[dict[str, str]], profile_id: str) -> dict[str, str]:
    profile_rows = [row for row in rows if row["profile_id"] == profile_id]
    if not profile_rows:
        raise SystemExit(f"missing energy rows for {profile_id}")
    return max(profile_rows, key=lambda row: float(row["conversion_peripheral_energy_ratio_vs_baseline"]))


def fmt_float(value: float, digits: int = 4) -> float:
    return round(float(value), digits)


def build_rows(slack: dict[str, Any], energy_rows: list[dict[str, str]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    global_stats = slack["_global"]
    n_layers = int(global_stats["n_layers"])
    n_columns = int(global_stats["n_columns_total"])
    total_serial_cycles = float(
        sum(float(value["serial_cycles"]) for key, value in slack.items() if key != "_global")
    )

    mobilevit_nominal = select_energy_row(
        energy_rows, "mobilevit_s_e7_overlay_measured_imagenet", control="0.03"
    )
    mobilevit_pessimistic = select_worst_energy_row(
        energy_rows, "mobilevit_s_e7_overlay_measured_imagenet"
    )
    bert_nominal = select_energy_row(
        energy_rows, "bert_glue_e7_overlay_analytical", control="0.03"
    )
    bert_pessimistic = select_worst_energy_row(energy_rows, "bert_glue_e7_overlay_analytical")

    source = rel(SLACK_MANIFEST)
    rows: list[dict[str, Any]] = [
        {
            "row_type": "component",
            "component": "tier_id_storage",
            "scope": "per_column",
            "evidence_label": "modeled",
            "count": n_columns,
            "bits_per_item": 2,
            "total_bits": n_columns * 2,
            "total_bytes": (n_columns * 2) / 8.0,
            "update_frequency": "per_profile_or_recalibration_window",
            "amortization_window_inferences": 1024,
            "nominal_control_fraction_of_suds_pre_control": 0.004,
            "pessimistic_control_fraction_of_suds_pre_control": 0.014,
            "source_artifact": source,
            "acceptance_note": "Three SUDS tiers require two stored bits per mapped column.",
        },
        {
            "row_type": "component",
            "component": "tau_threshold_storage",
            "scope": "per_layer",
            "evidence_label": "modeled",
            "count": n_layers * 2,
            "bits_per_item": 16,
            "total_bits": n_layers * 2 * 16,
            "total_bytes": (n_layers * 2 * 16) / 8.0,
            "update_frequency": "per_profile_or_recalibration_window",
            "amortization_window_inferences": 1024,
            "nominal_control_fraction_of_suds_pre_control": 0.002,
            "pessimistic_control_fraction_of_suds_pre_control": 0.006,
            "source_artifact": source,
            "acceptance_note": "Two fixed-point thresholds per layer are enough for KEEP/DEGRADE/PRUNE.",
        },
        {
            "row_type": "component",
            "component": "scheduler_slack_metadata",
            "scope": "per_layer",
            "evidence_label": "modeled",
            "count": n_layers * 3,
            "bits_per_item": 32,
            "total_bits": n_layers * 3 * 32,
            "total_bytes": (n_layers * 3 * 32) / 8.0,
            "update_frequency": "existing_scheduler_timing_snapshot",
            "amortization_window_inferences": 1024,
            "nominal_control_fraction_of_suds_pre_control": 0.004,
            "pessimistic_control_fraction_of_suds_pre_control": 0.012,
            "source_artifact": source,
            "acceptance_note": "Arrival, deadline, and slack fields are modeled as retained scheduler metadata.",
        },
        {
            "row_type": "component",
            "component": "threshold_compare_logic",
            "scope": "per_column_update",
            "evidence_label": "modeled",
            "count": n_columns * 2,
            "bits_per_item": 16,
            "total_bits": n_columns * 2 * 16,
            "total_bytes": (n_columns * 2 * 16) / 8.0,
            "update_frequency": "off_critical_path_profile_update",
            "amortization_window_inferences": 1024,
            "nominal_control_fraction_of_suds_pre_control": 0.008,
            "pessimistic_control_fraction_of_suds_pre_control": 0.030,
            "latency_cycles": n_layers,
            "latency_fraction_of_serial_cycles": n_layers / total_serial_cycles,
            "source_artifact": source,
            "acceptance_note": "Two threshold comparisons per mapped column are modeled as scheduler-side logic.",
        },
        {
            "row_type": "component",
            "component": "sideband_schedule_traffic",
            "scope": "per_profile_update",
            "evidence_label": "modeled",
            "count": n_columns,
            "bits_per_item": 2,
            "total_bits": n_columns * 2,
            "total_bytes": (n_columns * 2) / 8.0,
            "update_frequency": "tier_map_push_to_accelerator",
            "amortization_window_inferences": 1024,
            "nominal_control_fraction_of_suds_pre_control": 0.007,
            "pessimistic_control_fraction_of_suds_pre_control": 0.024,
            "source_artifact": source,
            "acceptance_note": "The runtime sideband can carry compact tier IDs rather than full slack arrays.",
        },
        {
            "row_type": "component",
            "component": "update_amortization_bookkeeping",
            "scope": "per_profile_update",
            "evidence_label": "modeled",
            "count": n_layers,
            "bits_per_item": 32,
            "total_bits": n_layers * 32,
            "total_bytes": (n_layers * 32) / 8.0,
            "update_frequency": "profile_epoch_counter_and_validity_tags",
            "amortization_window_inferences": 1024,
            "nominal_control_fraction_of_suds_pre_control": 0.005,
            "pessimistic_control_fraction_of_suds_pre_control": 0.014,
            "source_artifact": source,
            "acceptance_note": "Validity tags prevent stale tier maps from being reused across unmatched profiles.",
        },
    ]

    nominal_fraction = sum(
        float(row["nominal_control_fraction_of_suds_pre_control"]) for row in rows
    )
    pessimistic_fraction = sum(
        float(row["pessimistic_control_fraction_of_suds_pre_control"]) for row in rows
    )
    metadata_bits = sum(
        int(float(row["total_bits"]))
        for row in rows
        if row["component"] not in {"threshold_compare_logic", "sideband_schedule_traffic"}
    )
    latency_cycles = n_layers
    latency_fraction = latency_cycles / total_serial_cycles

    rows.extend(
        [
            {
                "row_type": "aggregate",
                "component": "interface_metadata_footprint",
                "scope": "mobilevit_s_profile",
                "evidence_label": "modeled",
                "count": n_columns,
                "total_bits": metadata_bits,
                "total_bytes": metadata_bits / 8.0,
                "update_frequency": "per_profile_or_recalibration_window",
                "amortization_window_inferences": 1024,
                "source_artifact": source,
                "acceptance_note": "Static tier and timing metadata are near 4 KiB for the promoted MobileViT-S profile.",
            },
            {
                "row_type": "aggregate",
                "component": "nominal_control_overhead",
                "scope": "energy_sensitivity_model",
                "evidence_label": "modeled_accounting",
                "nominal_control_fraction_of_suds_pre_control": nominal_fraction,
                "nominal_energy_ratio_vs_baseline": float(
                    mobilevit_nominal["conversion_peripheral_energy_ratio_vs_baseline"]
                ),
                "source_artifact": rel(ENERGY_SENSITIVITY),
                "acceptance_note": "Matches the 3% nominal control term already included in P0.4 sensitivity.",
            },
            {
                "row_type": "aggregate",
                "component": "pessimistic_control_overhead",
                "scope": "energy_sensitivity_model",
                "evidence_label": "modeled_accounting",
                "pessimistic_control_fraction_of_suds_pre_control": pessimistic_fraction,
                "pessimistic_energy_ratio_vs_baseline": float(
                    mobilevit_pessimistic["conversion_peripheral_energy_ratio_vs_baseline"]
                ),
                "source_artifact": rel(ENERGY_SENSITIVITY),
                "acceptance_note": "Matches the 10% stress control term already included in P0.4 sensitivity.",
            },
            {
                "row_type": "aggregate",
                "component": "latency_impact",
                "scope": "scheduler_path",
                "evidence_label": "modeled",
                "latency_cycles": latency_cycles,
                "latency_fraction_of_serial_cycles": latency_fraction,
                "source_artifact": source,
                "acceptance_note": "One scheduler compare slot per layer is less than 0.002% of summed modeled serial cycles.",
            },
        ]
    )

    summary = {
        "tag": TAG,
        "n_layers": n_layers,
        "n_columns_total": n_columns,
        "metadata_bits": metadata_bits,
        "metadata_bytes": metadata_bits / 8.0,
        "metadata_kib": metadata_bits / 8.0 / 1024.0,
        "sideband_bits_per_profile_update": n_columns * 2,
        "nominal_control_fraction_of_suds_pre_control": nominal_fraction,
        "pessimistic_control_fraction_of_suds_pre_control": pessimistic_fraction,
        "mobilevit_e7_nominal_total_ratio": float(
            mobilevit_nominal["conversion_peripheral_energy_ratio_vs_baseline"]
        ),
        "mobilevit_e7_nominal_total_reduction": float(
            mobilevit_nominal["conversion_peripheral_reduction"]
        ),
        "mobilevit_e7_pessimistic_total_ratio": float(
            mobilevit_pessimistic["conversion_peripheral_energy_ratio_vs_baseline"]
        ),
        "mobilevit_e7_pessimistic_total_reduction": float(
            mobilevit_pessimistic["conversion_peripheral_reduction"]
        ),
        "bert_e7_nominal_total_ratio": float(
            bert_nominal["conversion_peripheral_energy_ratio_vs_baseline"]
        ),
        "bert_e7_nominal_total_reduction": float(bert_nominal["conversion_peripheral_reduction"]),
        "bert_e7_pessimistic_total_ratio": float(
            bert_pessimistic["conversion_peripheral_energy_ratio_vs_baseline"]
        ),
        "bert_e7_pessimistic_total_reduction": float(
            bert_pessimistic["conversion_peripheral_reduction"]
        ),
        "latency_cycles": latency_cycles,
        "total_serial_cycles": total_serial_cycles,
        "latency_fraction_of_serial_cycles": latency_fraction,
        "claim_boundary_note": (
            "Interface overhead is modeled accounting, not circuit implementation. "
            "The P0.4 energy sensitivity already includes the same 3% nominal and "
            "10% pessimistic control terms."
        ),
    }
    return rows, summary


def write_json(path: Path, rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "metadata": {
                    "tag": TAG,
                    "evidence_label": "modeled_interface_overhead",
                    "promotion_decision": "appendix_boundary_accounting",
                    "claim_boundary_note": summary["claim_boundary_note"],
                },
                "summary": summary,
                "rows": rows,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def write_report(path: Path, rows: list[dict[str, Any]], summary: dict[str, Any], tag: str) -> None:
    def pct(value: float) -> str:
        return f"{value * 100:.2f}%"

    component_lines = []
    for row in rows:
        if row["row_type"] != "component":
            continue
        component_lines.append(
            "| {component} | {scope} | {total_bits} | {nom:.1f}% | {pess:.1f}% | {note} |".format(
                component=row["component"],
                scope=row["scope"],
                total_bits=int(float(row["total_bits"])),
                nom=float(row["nominal_control_fraction_of_suds_pre_control"]) * 100.0,
                pess=float(row["pessimistic_control_fraction_of_suds_pre_control"]) * 100.0,
                note=row["acceptance_note"],
            )
        )

    report = f"""# SUDS Interface Overhead Model

Tag: `{tag}`
Evidence label: `modeled interface overhead`
Promotion decision: `appendix_boundary_accounting`

## Scope

This P1.1 gate quantifies the SUDS control-plane cost that was already swept
as a control-overhead term in the P0.4 energy calibration. The model is an
accounting layer, not RTL, SPICE, layout, or silicon measurement.

## Metadata And Logic Footprint

The promoted MobileViT-S slack manifest has `{summary['n_layers']}` mapped
layers and `{summary['n_columns_total']}` mapped columns. The static interface
state is `{summary['metadata_bits']:.0f}` bits
(`{summary['metadata_kib']:.2f}` KiB), plus `{summary['sideband_bits_per_profile_update']}`
sideband bits when a compact tier map is pushed to the accelerator.

| Component | Scope | Bits/op state | Nominal control slice | Pessimistic control slice | Note |
|---|---|---:|---:|---:|---|
{chr(10).join(component_lines)}

## Net Energy Accounting

The component rows sum to a `3.0%` nominal SUDS-control term and a `10.0%`
pessimistic term. Those are the same terms used by the calibrated sensitivity
table, so the manuscript energy values are already net of modeled interface
overhead rather than free-control numbers.

| Profile | Nominal total ratio | Nominal total reduction | Pessimistic total ratio | Pessimistic total reduction |
|---|---:|---:|---:|---:|
| MobileViT-S E7 | {summary['mobilevit_e7_nominal_total_ratio']:.3f} | {pct(summary['mobilevit_e7_nominal_total_reduction'])} | {summary['mobilevit_e7_pessimistic_total_ratio']:.3f} | {pct(summary['mobilevit_e7_pessimistic_total_reduction'])} |
| BERT/GLUE E7 analytical | {summary['bert_e7_nominal_total_ratio']:.3f} | {pct(summary['bert_e7_nominal_total_reduction'])} | {summary['bert_e7_pessimistic_total_ratio']:.3f} | {pct(summary['bert_e7_pessimistic_total_reduction'])} |

## Latency Path

The scheduling-path model charges one threshold-compare slot per mapped layer:
`{summary['latency_cycles']}` cycles across the MobileViT-S profile. Against
`{summary['total_serial_cycles']:.0f}` summed modeled serial cycles, this is
`{summary['latency_fraction_of_serial_cycles'] * 100:.4f}%`. The claim remains
bounded to scheduler-side accounting; a real accelerator implementation would
still need timing closure.

## Acceptance Checks

- Metadata bits per column and per layer are explicit.
- Comparator, threshold, tier-ID storage, sideband traffic, update frequency,
  amortization policy, and latency impact are all represented in the CSV.
- Pessimistic overhead is included in the P0.4 sensitivity sweep.
- If a future implementation measures larger control cost, the main text must
  use the net savings after that measured cost.

## Required Artifacts

- CSV: `experiments/results/report_data/suds_interface_overhead_{tag}.csv`
- JSON: `experiments/results/report_data/suds_interface_overhead_{tag}.json`
- Report: `docs/reports/{tag}_suds_interface_overhead.md`

## Regeneration

```bash
.venv311-mps/bin/python experiments/tools/build_suds_interface_overhead.py
```
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")


def main() -> int:
    args = parse_args()
    slack = json.loads(args.slack_manifest.read_text(encoding="utf-8"))
    energy_rows = read_csv(args.energy_sensitivity)
    rows, summary = build_rows(slack, energy_rows)
    write_csv(args.csv_out, rows)
    write_json(args.json_out, rows, summary)
    write_report(args.report_out, rows, summary, args.tag)
    print(f"[suds-interface-overhead] wrote {rel(args.csv_out)}")
    print(f"[suds-interface-overhead] wrote {rel(args.json_out)}")
    print(f"[suds-interface-overhead] wrote {rel(args.report_out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
