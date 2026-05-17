#!/usr/bin/env python3
"""Validate the current FULLER phase2 modeling surface."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

try:
    from repo_python_bootstrap import maybe_reexec_for_module
except ImportError:
    def maybe_reexec_for_module(_module: str, *, anchor: Path | None = None) -> None:
        return None

maybe_reexec_for_module("yaml", anchor=Path(__file__))

import yaml

try:
    from .build_fuller_phase2_modeling_surface import (
        DEFAULT_CONTRACT,
        HANDOFF_FIELDS,
        ROOT,
        RUNNER_SCHEMA_FIELDS,
        _build_lane_lookups,
        _current_candidate_experiments,
        _effective_required_fields,
        _load_csv,
        _load_json,
        _load_yaml,
        _normalize_variant_lookup,
        _resolve_path,
        _ensure_matching_bundles,
    )
except ImportError:
    from build_fuller_phase2_modeling_surface import (  # type: ignore
        DEFAULT_CONTRACT,
        HANDOFF_FIELDS,
        ROOT,
        RUNNER_SCHEMA_FIELDS,
        _build_lane_lookups,
        _current_candidate_experiments,
        _effective_required_fields,
        _load_csv,
        _load_json,
        _load_yaml,
        _normalize_variant_lookup,
        _resolve_path,
        _ensure_matching_bundles,
    )


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _load_contract(contract_path: Path, root_dir: Path) -> dict[str, Any]:
    resolved_contract_path = contract_path if contract_path.is_absolute() else root_dir / contract_path
    return _load_yaml(resolved_contract_path)


def _parse_json_field(raw: str, field_name: str) -> list[str]:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid JSON in {field_name}: {raw!r}") from exc
    if not isinstance(payload, list):
        raise SystemExit(f"Expected JSON list in {field_name}")
    return [str(item) for item in payload]


def check_phase2_modeling_surface(
    contract_path: Path = DEFAULT_CONTRACT,
    *,
    root_dir: Path = ROOT,
) -> dict[str, Any]:
    contract = _load_contract(contract_path, root_dir)
    sources = contract.get("sources") or {}
    outputs = contract.get("outputs") or {}
    coordination = contract.get("coordination") or {}
    variants = contract.get("variants") or []

    canonical_bundle = _load_yaml(_resolve_path(root_dir, sources["phase1_canonical_bundle"]))
    runtime_smoke_bundle = _load_yaml(_resolve_path(root_dir, sources["phase1_runtime_smoke_bundle"]))
    canonical_lookup, _ = _normalize_variant_lookup(canonical_bundle)
    overlay_lookup, _ = _normalize_variant_lookup(runtime_smoke_bundle)
    _ensure_matching_bundles(canonical_lookup, overlay_lookup)

    preflight_payload = _load_json(_resolve_path(root_dir, sources["phase1_preflight_json"]))
    preflight_rows, _ = _build_lane_lookups(preflight_payload)
    current_candidate_rows = _load_csv(_resolve_path(root_dir, sources["phase1_current_runtime_smoke_candidate_csv"]))
    current_candidate_experiment_ids = _current_candidate_experiments(current_candidate_rows)

    gate_matrix_csv = _resolve_path(root_dir, outputs["gate_matrix_csv"])
    gate_matrix_json = _resolve_path(root_dir, outputs["gate_matrix_json"])
    handoff_contract_csv = _resolve_path(root_dir, outputs["handoff_contract_csv"])
    handoff_contract_json = _resolve_path(root_dir, outputs["handoff_contract_json"])
    modeling_manifest_json = _resolve_path(root_dir, outputs["modeling_manifest_json"])
    modeling_decision_md = _resolve_path(root_dir, outputs["modeling_decision_md"])
    governance_gate_matrix_md = _resolve_path(root_dir, outputs["governance_gate_matrix_md"])
    phase4_handoff_md = _resolve_path(root_dir, outputs["phase4_handoff_md"])

    for path in (
        gate_matrix_csv,
        gate_matrix_json,
        handoff_contract_csv,
        handoff_contract_json,
        modeling_manifest_json,
        modeling_decision_md,
        governance_gate_matrix_md,
        phase4_handoff_md,
    ):
        if not path.exists():
            raise SystemExit(f"Missing phase2 output: {path}")

    gate_csv_rows = list(csv.DictReader(gate_matrix_csv.open("r", newline="", encoding="utf-8")))
    gate_json_payload = _load_json(gate_matrix_json)
    gate_json_rows = gate_json_payload.get("rows")
    if not isinstance(gate_json_rows, list):
        raise SystemExit("Gate matrix JSON must expose rows")
    handoff_csv_rows = list(csv.DictReader(handoff_contract_csv.open("r", newline="", encoding="utf-8")))
    handoff_json_payload = _load_json(handoff_contract_json)
    handoff_json_rows = handoff_json_payload.get("rows")
    if not isinstance(handoff_json_rows, list):
        raise SystemExit("Handoff contract JSON must expose rows")

    expected_variant_ids = [str(item.get("variant_id") or "").strip().upper() for item in variants]
    if len(gate_csv_rows) != 7 or len(gate_json_rows) != 7:
        raise SystemExit("Gate matrix must contain exactly 7 rows")
    if len(handoff_csv_rows) != 7 or len(handoff_json_rows) != 7:
        raise SystemExit("Handoff contract must contain exactly 7 rows")
    gate_variant_ids = [str(row.get("variant_id") or "").strip().upper() for row in gate_csv_rows]
    if gate_variant_ids != expected_variant_ids:
        raise SystemExit(f"Gate matrix variants do not match contract order: {gate_variant_ids}")
    handoff_variant_ids = [str(row.get("variant_id") or "").strip().upper() for row in handoff_csv_rows]
    if handoff_variant_ids != expected_variant_ids:
        raise SystemExit(f"Handoff contract variants do not match contract order: {handoff_variant_ids}")

    if "FLOW" in gate_variant_ids:
        raise SystemExit("HOPS public naming regressed to FLOW")
    if "HOPS" not in gate_variant_ids:
        raise SystemExit("HOPS row missing from gate matrix")

    gate_by_variant = {str(row["variant_id"]).upper(): row for row in gate_csv_rows}
    handoff_by_variant = {str(row["variant_id"]).upper(): row for row in handoff_csv_rows}
    contract_by_variant = {str(row["variant_id"]).upper(): row for row in variants}

    for variant_id in expected_variant_ids:
        contract_variant = contract_by_variant[variant_id]
        gate_row = gate_by_variant[variant_id]
        handoff_row = handoff_by_variant[variant_id]
        overlay_variant = overlay_lookup[variant_id]
        preflight_row = preflight_rows.get(variant_id)
        if preflight_row is None:
            raise SystemExit(f"Missing preflight row for {variant_id}")

        if str(gate_row.get("claim_boundary") or "").strip() == "":
            raise SystemExit(f"Gate matrix claim boundary missing for {variant_id}")
        if str(gate_row.get("forbidden_claims") or "").strip() == "":
            raise SystemExit(f"Gate matrix forbidden claims missing for {variant_id}")

        parsed_required_fields = _parse_json_field(
            str(gate_row.get("required_summary_fields_json") or ""),
            f"{variant_id}.required_summary_fields_json",
        )
        missing_runner_fields = [field for field in parsed_required_fields if field not in RUNNER_SCHEMA_FIELDS]
        if missing_runner_fields:
            raise SystemExit(f"{variant_id} references fields not in current runner schema: {missing_runner_fields}")
        expected_required_fields = _effective_required_fields(contract_variant)
        if parsed_required_fields != expected_required_fields:
            raise SystemExit(
                f"Gate matrix required fields drift for {variant_id}: "
                f"{parsed_required_fields!r} != {expected_required_fields!r}"
            )

        expected_blockers = [str(item) for item in preflight_row.get("lane_blockers") or []]
        observed_blockers = [
            item.strip()
            for item in str(gate_row.get("phase3_entry_blockers") or "").split(";")
            if item.strip()
        ]
        if observed_blockers != expected_blockers:
            raise SystemExit(
                f"Gate matrix blockers drift for {variant_id}: {observed_blockers!r} != {expected_blockers!r}"
            )
        if str(gate_row.get("accuracy_contract_status") or "") != str(preflight_row.get("accuracy_status") or ""):
            raise SystemExit(f"Accuracy contract status drift for {variant_id}")
        if str(gate_row.get("runtime_smoke_gate_status") or "") != str(
            (preflight_row.get("runtime_smoke_gate") or {}).get("status") or ""
        ):
            raise SystemExit(f"Runtime-smoke gate drift for {variant_id}")
        if str(gate_row.get("analysis_grade_gate_status") or "") != str(
            (preflight_row.get("analysis_grade_gate") or {}).get("status") or ""
        ):
            raise SystemExit(f"Analysis-grade gate drift for {variant_id}")

        expected_config_path = str(
            (
                _resolve_path(root_dir, runtime_smoke_bundle["paths"]["generated_config_dir"])
                / f"{overlay_variant['config_stub']}.yaml"
            ).resolve()
        )
        if str(handoff_row.get("expected_generated_config_path") or "") != expected_config_path:
            raise SystemExit(f"Expected generated config path drift for {variant_id}")

        required_outputs = _parse_json_field(
            str(handoff_row.get("required_outputs_json") or ""),
            f"{variant_id}.required_outputs_json",
        )
        if "{launch_progress_root}/manifest.json" not in required_outputs:
            raise SystemExit(f"{variant_id} handoff is missing progress manifest requirement")
        if variant_id in {"HOPS", "FULLER"} and not any(
            item.endswith("/flow_buffer_trace.csv") for item in required_outputs
        ):
            raise SystemExit(f"{variant_id} handoff missing flow buffer trace output")
        if variant_id in {"PHY", "FULLER"} and not any(item.endswith("/phy_budget.json") for item in required_outputs):
            raise SystemExit(f"{variant_id} handoff missing phy budget output")
        if variant_id == "DET" and not any(item.endswith("/det_prefix_error.csv") for item in required_outputs):
            raise SystemExit("DET handoff missing det_prefix_error.csv")

        required_manifest_fields = _parse_json_field(
            str(handoff_row.get("required_manifest_fields_json") or ""),
            f"{variant_id}.required_manifest_fields_json",
        )
        if "runtime_guardrail" not in required_manifest_fields:
            raise SystemExit(f"{variant_id} handoff manifest fields missing runtime_guardrail")
        if str(handoff_row.get("required_device") or "") != str((contract.get("governance") or {}).get("required_device") or ""):
            raise SystemExit(f"{variant_id} handoff device requirement drift")
        if str(handoff_row.get("authorization_required") or "").lower() != "true":
            raise SystemExit(f"{variant_id} handoff must require authorization")
        if str(handoff_row.get("cpu_fallback_forbidden") or "").lower() != "true":
            raise SystemExit(f"{variant_id} handoff must forbid cpu fallback")
        if str(handoff_row.get("archived_row_relabel_forbidden") or "").lower() != "true":
            raise SystemExit(f"{variant_id} handoff must forbid archived-row relabel")

    readme_text = _read_text(_resolve_path(root_dir, coordination["active_readme_md"]))
    expected_pointer = str(coordination.get("expected_active_pointer") or "")
    forbidden_pointer = str(coordination.get("forbidden_legacy_pointer") or "")
    if forbidden_pointer and forbidden_pointer in readme_text:
        raise SystemExit("Active coordination README still points at the superseded true_sc plan")
    if expected_pointer not in readme_text:
        raise SystemExit("Active coordination README does not point at the fuller plan")

    plan_text = _read_text(_resolve_path(root_dir, coordination["active_phase_plan_md"]))
    if "`Phase2 Modeling`: `completed_2026-04-21`" not in plan_text:
        raise SystemExit("Active phase plan does not mark Phase2 as completed")

    manifest_payload = _load_json(modeling_manifest_json)
    if manifest_payload.get("phase2_status") != "completed":
        raise SystemExit("Modeling manifest phase2_status must be completed")

    modeling_decision_text = _read_text(modeling_decision_md)
    for needle in (
        "This does not mean:",
        "`analysis_grade` is enabled",
        "`benchmark_claim_ready=True`",
        "`E1-E6` already have current measured rows",
    ):
        if needle not in modeling_decision_text:
            raise SystemExit(f"Modeling decision note missing required phrase: {needle}")

    return {
        "status": "pass",
        "variant_ids": expected_variant_ids,
        "current_candidate_experiment_ids": sorted(current_candidate_experiment_ids),
        "gate_matrix_csv": str(gate_matrix_csv.resolve()),
        "handoff_contract_csv": str(handoff_contract_csv.resolve()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the FULLER phase2 modeling surface.")
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    args = parser.parse_args()
    payload = check_phase2_modeling_surface(args.contract)
    print(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
