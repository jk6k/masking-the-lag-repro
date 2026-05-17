#!/usr/bin/env python3
"""Validate FULLER v2 engineering-smoke result surfaces and pre-execution readiness."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

try:
    from .fuller_v2_runtime_smoke_surface import (
        DEFAULT_PROGRAM_CONTRACT,
        build_lane_annotation_fields_from_row,
        required_result_surface_fields,
        supported_result_surface_fields,
    )
except ImportError:
    from fuller_v2_runtime_smoke_surface import (  # type: ignore
        DEFAULT_PROGRAM_CONTRACT,
        build_lane_annotation_fields_from_row,
        required_result_surface_fields,
        supported_result_surface_fields,
    )


def _load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    return False


def _resolve_eval_row(
    rows: list[dict[str, str]],
    *,
    eval_run_id: str | None = None,
) -> dict[str, str] | None:
    normalized_run_id = str(eval_run_id or "").strip()
    if normalized_run_id:
        for row in rows:
            if str(row.get("run_id") or "").strip() != normalized_run_id:
                continue
            if str(row.get("baseline") or "").strip().lower() == "true":
                continue
            return row
    for row in reversed(rows):
        if str(row.get("baseline") or "").strip().lower() == "true":
            continue
        return row
    return None


def _write_output(path: Path | None, payload: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _synthetic_runtime_row(lane_id: str) -> dict[str, Any]:
    sparse_fraction = "0.75" if lane_id in {"SPARSE", "FULLER"} else ""
    return {
        "run_id": f"synthetic_{lane_id.lower()}_acc_s0",
        "baseline": "false",
        "model": "mobilevit_s",
        "top1": "0.0",
        "top1_delta": "0.0",
        "latency_ms_per_sample": "1000.0",
        "measured_pass_elapsed_s": "512.0",
        "measured_processed_samples": "512",
        "sparse_measured_activity_fraction": sparse_fraction,
    }


def check_completed_result_surface(
    *,
    lane_id: str,
    annotated_csv: Path,
    eval_run_id: str | None = None,
    contract_path: Path = DEFAULT_PROGRAM_CONTRACT,
) -> dict[str, Any]:
    rows = _load_rows(annotated_csv) if annotated_csv.exists() else []
    target_row = _resolve_eval_row(rows, eval_run_id=eval_run_id)
    required_fields = required_result_surface_fields(
        variant_id=lane_id,
        contract_path=contract_path,
    )
    missing_fields: list[str] = []
    if target_row is None:
        missing_fields.append("completed_quantized_row_missing")
    else:
        for field in required_fields:
            if _is_missing(target_row.get(field)):
                missing_fields.append(field)
    payload = {
        "lane_id": lane_id,
        "result_surface_status": "complete" if not missing_fields else "incomplete",
        "missing_required_fields_json": json.dumps(
            missing_fields,
            ensure_ascii=False,
            sort_keys=True,
        ),
        "source_row_path": str(annotated_csv.resolve()),
        "ready_for_next_lane": not missing_fields,
    }
    return payload


def audit_pre_execution_surface(
    *,
    lane_id: str,
    config_snapshot: Path,
    baseline_reference_csv: Path | None = None,
    baseline_reference_summary_json: Path | None = None,
    contract_path: Path = DEFAULT_PROGRAM_CONTRACT,
) -> dict[str, Any]:
    supported_fields = supported_result_surface_fields(
        cfg_path=config_snapshot,
        variant_id=lane_id,
        contract_path=contract_path,
    )
    required_fields = required_result_surface_fields(
        variant_id=lane_id,
        contract_path=contract_path,
    )
    missing_fields = [
        field for field in required_fields if field not in supported_fields
    ]

    if lane_id != "ASTRA":
        if baseline_reference_csv is None or not baseline_reference_csv.exists():
            missing_fields.append("baseline_reference_csv")
        if (
            baseline_reference_summary_json is None
            or not baseline_reference_summary_json.exists()
        ):
            missing_fields.append("baseline_reference_summary_json")

    helper_probe_error = ""
    try:
        build_lane_annotation_fields_from_row(
            cfg_path=config_snapshot,
            raw_row=_synthetic_runtime_row(lane_id),
            raw_results_csv=baseline_reference_csv,
            variant_id=lane_id,
        )
    except Exception as exc:  # pragma: no cover - exact message varies with config
        helper_probe_error = str(exc)
        missing_fields.append("annotation_helper_probe_failed")

    deduped_missing: list[str] = []
    for item in missing_fields:
        if item not in deduped_missing:
            deduped_missing.append(item)
    payload = {
        "lane_id": lane_id,
        "result_surface_status": "complete" if not deduped_missing else "incomplete",
        "missing_required_fields_json": json.dumps(
            deduped_missing,
            ensure_ascii=False,
            sort_keys=True,
        ),
        "source_row_path": str(config_snapshot.resolve()),
        "ready_for_next_lane": not deduped_missing,
    }
    if helper_probe_error:
        payload["annotation_helper_probe_error"] = helper_probe_error
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check FULLER v2 runtime-smoke result surfaces."
    )
    parser.add_argument("--lane_id", required=True)
    parser.add_argument(
        "--mode",
        choices=["postrun", "pre_execution"],
        default="postrun",
    )
    parser.add_argument("--annotated_csv", type=Path, default=None)
    parser.add_argument("--eval_run_id", default=None)
    parser.add_argument("--config_snapshot", type=Path, default=None)
    parser.add_argument("--baseline_reference_csv", type=Path, default=None)
    parser.add_argument("--baseline_reference_summary_json", type=Path, default=None)
    parser.add_argument("--contract", type=Path, default=DEFAULT_PROGRAM_CONTRACT)
    parser.add_argument("--output_json", type=Path, default=None)
    args = parser.parse_args()

    lane_id = str(args.lane_id).strip().upper()
    if args.mode == "postrun":
        if args.annotated_csv is None:
            raise SystemExit("--annotated_csv is required for --mode postrun")
        payload = check_completed_result_surface(
            lane_id=lane_id,
            annotated_csv=args.annotated_csv,
            eval_run_id=args.eval_run_id,
            contract_path=args.contract,
        )
    else:
        if args.config_snapshot is None:
            raise SystemExit("--config_snapshot is required for --mode pre_execution")
        payload = audit_pre_execution_surface(
            lane_id=lane_id,
            config_snapshot=args.config_snapshot,
            baseline_reference_csv=args.baseline_reference_csv,
            baseline_reference_summary_json=args.baseline_reference_summary_json,
            contract_path=args.contract,
        )

    _write_output(args.output_json, payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True))
    if not payload["ready_for_next_lane"]:
        raise SystemExit(
            f"{lane_id} result surface is incomplete: {payload['missing_required_fields_json']}"
        )


if __name__ == "__main__":
    main()
