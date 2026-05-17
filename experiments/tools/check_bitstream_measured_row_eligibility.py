#!/usr/bin/env python3
"""Check whether an observed bitstream accuracy row is promotable measured evidence."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[2]
EXPERIMENTS_ROOT = ROOT_DIR / "experiments"
if str(EXPERIMENTS_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_ROOT))

from accuracy.bitstream_semantics import (  # noqa: E402
    BITSTREAM_MODEL_LEVEL_MEASURED_TRUTH_CLASS,
)
from accuracy.bitstream_conv_semantics import (  # noqa: E402
    CONV_FOCUSED_CLAIM_SURFACE_STATUS,
    CONV_FOCUSED_MEASURED_PACKAGE_STATUS_AUTHORIZED,
    CONV_MEASURED_CLOSURE_STATUS_MEASURED_CLOSED,
    LEGACY_ALL_TARGET_MEASURED_ANCHOR_RUN_ID,
    resolve_conv_evidence_manifest,
    resolve_conv_focused_measured_package,
    validate_conv_focused_measured_package,
)


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _to_int(raw: Any) -> int | None:
    text = str(raw or "").strip()
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        return None


def _row_matches(
    row: dict[str, str],
    *,
    run_id: str | None,
    measurement_window: str | None,
) -> bool:
    if run_id and str(row.get("run_id") or "").strip() != run_id:
        return False
    if measurement_window and str(row.get("measurement_window") or "").strip() != measurement_window:
        return False
    return True


def select_target_row(
    rows: list[dict[str, str]],
    *,
    run_id: str | None = None,
    measurement_window: str | None = "quantized_eval_pass",
) -> dict[str, str]:
    matched = [
        row
        for row in rows
        if str(row.get("baseline") or "").strip().lower() != "true"
        and _row_matches(row, run_id=run_id, measurement_window=measurement_window)
    ]
    if not matched:
        raise ValueError(
            "No non-baseline bitstream row matched the requested filters: "
            f"run_id={run_id!r}, measurement_window={measurement_window!r}"
        )
    if len(matched) > 1:
        raise ValueError(
            "Multiple non-baseline bitstream rows matched the requested filters; "
            "pass --run_id or a more specific measurement window."
        )
    return matched[0]


def _default_conv_contract_metadata() -> dict[str, Any]:
    manifest = resolve_conv_evidence_manifest(
        model_key="mobilevit_s",
        ops_path=ROOT_DIR / "experiments" / "mtl_model" / "ops" / "ops_mobilevit_s.json",
    )
    package = resolve_conv_focused_measured_package()
    package_validation = validate_conv_focused_measured_package(
        conv_evidence_manifest=manifest,
        conv_focused_measured_package=package,
    )
    return {
        "manifest": manifest,
        "package": package,
        "package_validation": package_validation,
    }


def assess_row_eligibility(
    row: dict[str, str],
    *,
    conv_contract_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    execution_semantics = str(row.get("execution_semantics") or "").strip()
    truth_class = str(row.get("bitstream_measurement_truth_class") or "").strip()
    claim_surface_status = str(row.get("bitstream_runtime_claim_surface_status") or "").strip()
    authorization_note = str(row.get("bitstream_truth_class_authorization_note") or "").strip()
    authorization_status = str(row.get("bitstream_truth_class_authorization_status") or "").strip()
    conv_measured_closure_status = str(
        row.get("bitstream_conv_measured_closure_status") or ""
    ).strip()
    conv_measured_package_status = str(
        row.get("bitstream_conv_measured_package_status") or ""
    ).strip()
    conv_measured_package_path = str(
        row.get("bitstream_conv_measured_package_path") or ""
    ).strip()
    conv_measured_package_sha256 = str(
        row.get("bitstream_conv_measured_package_sha256") or ""
    ).strip()
    conv_evidence_manifest_path = str(
        row.get("bitstream_conv_evidence_manifest_path") or ""
    ).strip()
    conv_evidence_manifest_sha256 = str(
        row.get("bitstream_conv_evidence_manifest_sha256") or ""
    ).strip()
    conv_target_set_sha256 = str(row.get("bitstream_conv_target_set_sha256") or "").strip()
    active_count = _to_int(row.get("bitstream_runtime_active_target_module_count"))
    targetable_count = _to_int(row.get("bitstream_runtime_targetable_module_count"))
    source_run_id = str(row.get("source_run_id") or row.get("run_id") or "").strip()
    full_target_coverage = (
        active_count is not None
        and targetable_count is not None
        and targetable_count > 0
        and active_count == targetable_count
    )
    strong_conv_closure_requested = (
        conv_measured_closure_status == CONV_MEASURED_CLOSURE_STATUS_MEASURED_CLOSED
        or bool(conv_measured_package_status)
        or bool(conv_measured_package_path)
        or bool(conv_measured_package_sha256)
        or bool(conv_evidence_manifest_path)
        or bool(conv_target_set_sha256)
    )

    blockers: list[str] = []
    if execution_semantics != "bitstream":
        blockers.append("execution_semantics_not_bitstream")
    if strong_conv_closure_requested:
        if claim_surface_status not in {
            "full_model_claim_surface_runtime",
            CONV_FOCUSED_CLAIM_SURFACE_STATUS,
        }:
            blockers.append("claim_surface_not_promotable")
    else:
        if not full_target_coverage:
            blockers.append("target_coverage_incomplete")
        if claim_surface_status != "full_model_claim_surface_runtime":
            blockers.append("claim_surface_not_promotable")
    if truth_class != BITSTREAM_MODEL_LEVEL_MEASURED_TRUTH_CLASS:
        blockers.append("truth_class_not_model_level_measured")
    if authorization_status and authorization_status != "authorized":
        blockers.append("authorization_not_satisfied")
    if not authorization_status and not authorization_note:
        blockers.append("authorization_missing")
    if strong_conv_closure_requested:
        resolved_contract = conv_contract_metadata or _default_conv_contract_metadata()
        expected_manifest = resolved_contract.get("manifest") or {}
        expected_package = resolved_contract.get("package") or {}
        package_validation = resolved_contract.get("package_validation") or {}
        expected_target_set_sha256 = str(
            (((expected_manifest.get("manifest") or {}).get("conv_focused_target_set") or {}).get("target_set_sha256"))
            or ""
        ).strip()
        expected_manifest_path = str(expected_manifest.get("manifest_path") or "").strip()
        expected_manifest_sha256 = str(expected_manifest.get("manifest_sha256") or "").strip()
        expected_package_path = str(expected_package.get("package_path") or "").strip()
        expected_package_sha256 = str(expected_package.get("package_sha256") or "").strip()
        if conv_measured_closure_status != CONV_MEASURED_CLOSURE_STATUS_MEASURED_CLOSED:
            blockers.append("conv_measured_closure_not_declared")
        if not conv_evidence_manifest_path or not conv_evidence_manifest_sha256:
            blockers.append("conv_evidence_manifest_not_bound")
        if expected_manifest_path and conv_evidence_manifest_path != expected_manifest_path:
            blockers.append("conv_evidence_manifest_not_bound")
        if expected_manifest_sha256 and conv_evidence_manifest_sha256 != expected_manifest_sha256:
            blockers.append("conv_evidence_manifest_sha_mismatch")
        if conv_measured_package_status != CONV_FOCUSED_MEASURED_PACKAGE_STATUS_AUTHORIZED:
            blockers.append("conv_measured_package_not_authorized")
        if not conv_measured_package_path:
            blockers.append("conv_measured_package_not_bound")
        if expected_package_path and conv_measured_package_path != expected_package_path:
            blockers.append("conv_measured_package_not_bound")
        if not conv_measured_package_sha256:
            blockers.append("conv_measured_package_sha_mismatch")
        if expected_package_sha256 and conv_measured_package_sha256 != expected_package_sha256:
            blockers.append("conv_measured_package_sha_mismatch")
        if not conv_target_set_sha256:
            blockers.append("conv_target_set_not_bound")
        if expected_target_set_sha256 and conv_target_set_sha256 != expected_target_set_sha256:
            blockers.append("conv_target_set_not_bound")
        for blocker in package_validation.get("package_blockers") or []:
            blocker_text = str(blocker).strip()
            if blocker_text:
                blockers.append(blocker_text)
        if source_run_id == LEGACY_ALL_TARGET_MEASURED_ANCHOR_RUN_ID:
            blockers.append("legacy_all_target_row_not_authorized_for_conv_closure")

    deduped_blockers: list[str] = []
    for blocker in blockers:
        if blocker not in deduped_blockers:
            deduped_blockers.append(blocker)

    return {
        "run_id": str(row.get("run_id") or "").strip(),
        "measurement_window": str(row.get("measurement_window") or "").strip(),
        "execution_semantics": execution_semantics,
        "bitstream_measurement_truth_class": truth_class,
        "bitstream_runtime_claim_surface_status": claim_surface_status,
        "bitstream_truth_class_authorization_note": authorization_note,
        "bitstream_truth_class_authorization_status": authorization_status,
        "bitstream_conv_measured_closure_status": conv_measured_closure_status,
        "bitstream_conv_measured_package_status": conv_measured_package_status,
        "bitstream_conv_measured_package_path": conv_measured_package_path,
        "bitstream_conv_measured_package_sha256": conv_measured_package_sha256,
        "bitstream_conv_evidence_manifest_path": conv_evidence_manifest_path,
        "bitstream_conv_evidence_manifest_sha256": conv_evidence_manifest_sha256,
        "bitstream_conv_target_set_sha256": conv_target_set_sha256,
        "bitstream_runtime_active_target_module_count": active_count,
        "bitstream_runtime_targetable_module_count": targetable_count,
        "full_target_coverage": full_target_coverage,
        "strong_conv_closure_requested": strong_conv_closure_requested,
        "promotable_measured_row_eligible": not deduped_blockers,
        "blockers": deduped_blockers,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_markdown(path: Path, payload: dict[str, Any], *, input_csv: Path) -> None:
    lines = [
        "# Bitstream Measured Row Eligibility",
        "",
        f"- input_csv: `{input_csv}`",
        f"- run_id: `{payload['run_id']}`",
        f"- measurement_window: `{payload['measurement_window']}`",
        f"- execution_semantics: `{payload['execution_semantics']}`",
        f"- bitstream_measurement_truth_class: `{payload['bitstream_measurement_truth_class']}`",
        f"- bitstream_runtime_claim_surface_status: `{payload['bitstream_runtime_claim_surface_status']}`",
        f"- bitstream_truth_class_authorization_status: `{payload['bitstream_truth_class_authorization_status']}`",
        f"- strong_conv_closure_requested: `{payload['strong_conv_closure_requested']}`",
        f"- bitstream_conv_measured_closure_status: `{payload['bitstream_conv_measured_closure_status']}`",
        f"- bitstream_conv_measured_package_status: `{payload['bitstream_conv_measured_package_status']}`",
        f"- bitstream_conv_measured_package_path: `{payload['bitstream_conv_measured_package_path']}`",
        f"- bitstream_conv_measured_package_sha256: `{payload['bitstream_conv_measured_package_sha256']}`",
        f"- bitstream_runtime_active_target_module_count: `{payload['bitstream_runtime_active_target_module_count']}`",
        f"- bitstream_runtime_targetable_module_count: `{payload['bitstream_runtime_targetable_module_count']}`",
        f"- full_target_coverage: `{payload['full_target_coverage']}`",
        f"- promotable_measured_row_eligible: `{payload['promotable_measured_row_eligible']}`",
    ]
    blockers = payload.get("blockers") or []
    if blockers:
        lines.extend(["", "## Blockers", ""])
        lines.extend(f"- `{blocker}`" for blocker in blockers)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Check whether a measured bitstream accuracy row is promotable evidence.",
    )
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--run_id", default=None)
    parser.add_argument("--measurement_window", default="quantized_eval_pass")
    parser.add_argument("--output_json", default=None)
    parser.add_argument("--output_md", default=None)
    return parser


def main() -> int:
    parser = build_argparser()
    args = parser.parse_args()
    input_csv = Path(args.input_csv)
    rows = _read_rows(input_csv)
    target_row = select_target_row(
        rows,
        run_id=args.run_id,
        measurement_window=args.measurement_window,
    )
    payload = assess_row_eligibility(
        target_row,
        conv_contract_metadata=_default_conv_contract_metadata(),
    )
    if args.output_json:
        _write_json(Path(args.output_json), payload)
    if args.output_md:
        _write_markdown(Path(args.output_md), payload, input_csv=input_csv)
    print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
