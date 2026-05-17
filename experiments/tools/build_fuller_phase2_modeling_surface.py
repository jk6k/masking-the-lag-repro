#!/usr/bin/env python3
"""Build the current FULLER phase2 modeling surface from phase1 artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
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
    from .phase1_runner import MASTER_FIELDS, PHASE1_SUMMARY_FIELDS
except ImportError:
    from phase1_runner import MASTER_FIELDS, PHASE1_SUMMARY_FIELDS  # type: ignore


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONTRACT = ROOT / "configs" / "fuller_phase2_modeling_contract_20260421.yaml"

GATE_MATRIX_FIELDS = [
    "variant_id",
    "internal_experiment_id",
    "public_module_stack",
    "module_role",
    "phase2_modeling_status",
    "required_summary_fields_json",
    "accuracy_contract_status",
    "runtime_smoke_gate_status",
    "analysis_grade_gate_status",
    "phase3_entry_status",
    "phase3_entry_blockers",
    "phase4_handoff_status",
    "claim_boundary",
    "forbidden_claims",
]

HANDOFF_FIELDS = [
    "variant_id",
    "internal_experiment_id",
    "config_stub",
    "expected_generated_config_path",
    "required_outputs_json",
    "required_manifest_fields_json",
    "required_summary_fields_json",
    "required_device",
    "long_run_wrapper_json",
    "authorization_required",
    "archived_row_relabel_forbidden",
    "cpu_fallback_forbidden",
    "notes",
]

RUNNER_SCHEMA_FIELDS = set(PHASE1_SUMMARY_FIELDS) | set(MASTER_FIELDS)
HOPS_OPTIONAL_FIELDS = {
    "flow_buffer_peak_cycles",
    "flow_buffer_peak_frac",
    "flow_residency_hit_rate",
    "flow_control_backpressure",
    "flow_eviction_count",
    "flow_admission_stalls",
}


def _resolve_path(root_dir: Path, raw_path: str | Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return root_dir / path


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise SystemExit(f"Expected YAML mapping in {path}")
    return payload


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise SystemExit(f"Expected JSON object in {path}")
    return payload


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def _normalize_variant_lookup(bundle_payload: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], Path]:
    variants = bundle_payload.get("variants") or []
    if not isinstance(variants, list):
        raise SystemExit("Bundle variants must be a list")
    generated_config_dir = bundle_payload.get("paths", {}).get("generated_config_dir")
    if not generated_config_dir:
        raise SystemExit("Bundle missing paths.generated_config_dir")
    lookup: dict[str, dict[str, Any]] = {}
    for raw_item in variants:
        if not isinstance(raw_item, dict):
            raise SystemExit("Bundle variants must contain mappings")
        variant_id = str(raw_item.get("variant_id") or "").strip().upper()
        internal_experiment_id = str(raw_item.get("internal_experiment_id") or "").strip().upper()
        if not variant_id or not internal_experiment_id:
            raise SystemExit("Bundle variants must define variant_id and internal_experiment_id")
        lookup[variant_id] = {
            "variant_id": variant_id,
            "internal_experiment_id": internal_experiment_id,
            "public_module_stack": list(raw_item.get("public_module_stack") or []),
            "config_stub": str(raw_item.get("config_stub") or "").strip(),
            "switches": dict(raw_item.get("switches") or {}),
        }
    return lookup, Path(str(generated_config_dir))


def _ensure_matching_bundles(
    canonical_lookup: dict[str, dict[str, Any]],
    overlay_lookup: dict[str, dict[str, Any]],
) -> None:
    if set(canonical_lookup) != set(overlay_lookup):
        raise SystemExit("Canonical and runtime-smoke bundles expose different variant ids")
    for variant_id, canonical_row in canonical_lookup.items():
        overlay_row = overlay_lookup[variant_id]
        for key in ("internal_experiment_id", "public_module_stack", "config_stub", "switches"):
            if canonical_row.get(key) != overlay_row.get(key):
                raise SystemExit(
                    f"Canonical/runtime-smoke mismatch for {variant_id} field {key}: "
                    f"{canonical_row.get(key)!r} != {overlay_row.get(key)!r}"
                )


def _effective_required_fields(variant_cfg: dict[str, Any]) -> list[str]:
    required = [str(item) for item in variant_cfg.get("required_summary_fields") or []]
    optional_if_present = [str(item) for item in variant_cfg.get("summary_field_candidates_if_present") or []]
    resolved = list(dict.fromkeys(required))
    for field in optional_if_present:
        if field in PHASE1_SUMMARY_FIELDS and field not in resolved:
            resolved.append(field)
    missing = [field for field in resolved if field not in RUNNER_SCHEMA_FIELDS]
    if missing:
        raise SystemExit(
            f"Variant {variant_cfg.get('variant_id')} references fields not exported by phase1_runner: {missing}"
        )
    return resolved


def _join_semicolon(values: list[str]) -> str:
    return "; ".join(item for item in values if str(item).strip())


def _build_lane_lookups(preflight_payload: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], dict[str, list[str]]]:
    lane_rows = preflight_payload.get("lane_rows") or []
    if not isinstance(lane_rows, list):
        raise SystemExit("Preflight payload lane_rows must be a list")
    by_variant: dict[str, dict[str, Any]] = {}
    by_experiment: dict[str, list[str]] = {}
    for raw_row in lane_rows:
        if not isinstance(raw_row, dict):
            raise SystemExit("Preflight lane rows must be mappings")
        variant_id = str(raw_row.get("variant_id") or "").strip().upper()
        if variant_id:
            by_variant[variant_id] = raw_row
        experiment_id = str(raw_row.get("internal_experiment_id") or raw_row.get("experiment_id") or "").strip().upper()
        if experiment_id:
            by_experiment.setdefault(experiment_id, []).append(variant_id or experiment_id)
    return by_variant, by_experiment


def _current_candidate_experiments(rows: list[dict[str, str]]) -> set[str]:
    experiments: set[str] = set()
    for row in rows:
        experiment_id = str(row.get("internal_experiment_id") or row.get("experiment_id") or "").strip().upper()
        if experiment_id:
            experiments.add(experiment_id)
    return experiments


def _build_gate_rows(
    *,
    contract: dict[str, Any],
    overlay_lookup: dict[str, dict[str, Any]],
    preflight_rows: dict[str, dict[str, Any]],
    current_candidate_experiment_ids: set[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for variant_cfg in contract.get("variants") or []:
        variant_id = str(variant_cfg.get("variant_id") or "").strip().upper()
        if variant_id not in overlay_lookup:
            raise SystemExit(f"Contract variant {variant_id} missing from runtime-smoke bundle")
        bundle_variant = overlay_lookup[variant_id]
        lane_row = preflight_rows.get(variant_id)
        if lane_row is None:
            raise SystemExit(f"Missing preflight lane row for {variant_id}")
        required_summary_fields = _effective_required_fields(variant_cfg)
        phase3_blockers = [str(item) for item in lane_row.get("lane_blockers") or []]
        experiment_id = str(bundle_variant["internal_experiment_id"]).upper()
        current_evidence_present = experiment_id in current_candidate_experiment_ids
        phase2_modeling_status = (
            "current_modeling_surface_ready_with_anchor"
            if variant_id == "ASTRA" and current_evidence_present
            else "current_modeling_surface_ready_evidence_pending"
        )
        phase4_handoff_status = (
            "contract_ready_waiting_for_phase3_context_closure"
            if variant_id == "ASTRA" and phase3_blockers
            else "contract_ready_waiting_for_phase3_collection"
        )
        rows.append(
            {
                "variant_id": variant_id,
                "internal_experiment_id": experiment_id,
                "public_module_stack": list(bundle_variant["public_module_stack"]),
                "module_role": str(variant_cfg.get("module_role") or "").strip(),
                "phase2_modeling_status": phase2_modeling_status,
                "required_summary_fields_json": required_summary_fields,
                "accuracy_contract_status": str(lane_row.get("accuracy_status") or "").strip(),
                "runtime_smoke_gate_status": str(
                    (lane_row.get("runtime_smoke_gate") or {}).get("status") or ""
                ).strip(),
                "analysis_grade_gate_status": str(
                    (lane_row.get("analysis_grade_gate") or {}).get("status") or ""
                ).strip(),
                "phase3_entry_status": "blocked" if phase3_blockers else "ready",
                "phase3_entry_blockers": phase3_blockers,
                "phase4_handoff_status": phase4_handoff_status,
                "claim_boundary": str(variant_cfg.get("claim_boundary") or "").strip(),
                "forbidden_claims": [str(item) for item in variant_cfg.get("forbidden_claims") or []],
            }
        )
    return rows


def _build_handoff_rows(
    *,
    contract: dict[str, Any],
    overlay_lookup: dict[str, dict[str, Any]],
    overlay_generated_dir: Path,
    gate_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    governance = contract.get("governance") or {}
    handoff_cfg = contract.get("handoff") or {}
    base_required_outputs = [str(item) for item in handoff_cfg.get("base_required_outputs") or []]
    required_manifest_fields = [str(item) for item in handoff_cfg.get("required_manifest_fields") or []]
    gate_lookup = {str(row["variant_id"]).upper(): row for row in gate_rows}

    rows: list[dict[str, Any]] = []
    for variant_cfg in contract.get("variants") or []:
        variant_id = str(variant_cfg.get("variant_id") or "").strip().upper()
        bundle_variant = overlay_lookup[variant_id]
        gate_row = gate_lookup[variant_id]
        required_outputs = list(base_required_outputs)
        required_outputs.extend(str(item) for item in variant_cfg.get("handoff_extra_outputs") or [])
        required_outputs = list(dict.fromkeys(required_outputs))
        notes = (
            f"{variant_cfg.get('module_role')} handoff; "
            f"phase3 blockers={_join_semicolon(gate_row['phase3_entry_blockers']) or 'none'}"
        )
        rows.append(
            {
                "variant_id": variant_id,
                "internal_experiment_id": str(bundle_variant["internal_experiment_id"]).upper(),
                "config_stub": str(bundle_variant["config_stub"]).strip(),
                "expected_generated_config_path": str(
                    (overlay_generated_dir / f"{bundle_variant['config_stub']}.yaml").resolve()
                ),
                "required_outputs_json": required_outputs,
                "required_manifest_fields_json": required_manifest_fields,
                "required_summary_fields_json": gate_row["required_summary_fields_json"],
                "required_device": str(governance.get("required_device") or "").strip(),
                "long_run_wrapper_json": [str(item) for item in governance.get("long_run_wrapper") or []],
                "authorization_required": bool(governance.get("authorization_required_before_long_run")),
                "archived_row_relabel_forbidden": bool(governance.get("archived_row_relabel_forbidden")),
                "cpu_fallback_forbidden": bool(governance.get("cpu_fallback_forbidden")),
                "notes": notes,
            }
        )
    return rows


def _csv_gate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    csv_rows: list[dict[str, Any]] = []
    for row in rows:
        csv_rows.append(
            {
                "variant_id": row["variant_id"],
                "internal_experiment_id": row["internal_experiment_id"],
                "public_module_stack": json.dumps(row["public_module_stack"], ensure_ascii=False),
                "module_role": row["module_role"],
                "phase2_modeling_status": row["phase2_modeling_status"],
                "required_summary_fields_json": json.dumps(row["required_summary_fields_json"], ensure_ascii=False),
                "accuracy_contract_status": row["accuracy_contract_status"],
                "runtime_smoke_gate_status": row["runtime_smoke_gate_status"],
                "analysis_grade_gate_status": row["analysis_grade_gate_status"],
                "phase3_entry_status": row["phase3_entry_status"],
                "phase3_entry_blockers": _join_semicolon(row["phase3_entry_blockers"]),
                "phase4_handoff_status": row["phase4_handoff_status"],
                "claim_boundary": row["claim_boundary"],
                "forbidden_claims": _join_semicolon(row["forbidden_claims"]),
            }
        )
    return csv_rows


def _csv_handoff_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    csv_rows: list[dict[str, Any]] = []
    for row in rows:
        csv_rows.append(
            {
                "variant_id": row["variant_id"],
                "internal_experiment_id": row["internal_experiment_id"],
                "config_stub": row["config_stub"],
                "expected_generated_config_path": row["expected_generated_config_path"],
                "required_outputs_json": json.dumps(row["required_outputs_json"], ensure_ascii=False),
                "required_manifest_fields_json": json.dumps(
                    row["required_manifest_fields_json"], ensure_ascii=False
                ),
                "required_summary_fields_json": json.dumps(
                    row["required_summary_fields_json"], ensure_ascii=False
                ),
                "required_device": row["required_device"],
                "long_run_wrapper_json": json.dumps(row["long_run_wrapper_json"], ensure_ascii=False),
                "authorization_required": str(bool(row["authorization_required"])).lower(),
                "archived_row_relabel_forbidden": str(
                    bool(row["archived_row_relabel_forbidden"])
                ).lower(),
                "cpu_fallback_forbidden": str(bool(row["cpu_fallback_forbidden"])).lower(),
                "notes": row["notes"],
            }
        )
    return csv_rows


def _render_modeling_decision_note(
    *,
    contract: dict[str, Any],
    gate_rows: list[dict[str, Any]],
    current_candidate_experiment_ids: set[str],
) -> str:
    active_pointer = str((contract.get("coordination") or {}).get("expected_active_pointer") or "")
    lines = [
        "# FULLER Phase2 Modeling Decision Note",
        "",
        "Date: `2026-04-21`",
        "Status: `phase2_completed`",
        "Scope: `active fuller phase2 / current modeled interpretation and governance gates`",
        "",
        "## Decision",
        "",
        "Phase2 is now complete for the current fuller public stack. In this repository, that means the active",
        "fuller architecture has one current modeled interpretation, one explicit governance gate matrix, and one",
        "phase4 handoff contract that phase3/4 can execute against without reconstructing the architecture from archive notes.",
        "",
        "This does not mean:",
        "",
        "- `analysis_grade` is enabled",
        "- `benchmark_claim_ready=True`",
        "- `E1-E6` already have current measured rows",
        "- `Phase3` or `Phase4` are complete",
        "",
        "## Current Truth",
        "",
        f"- active phase plan: `{active_pointer}`",
        "- current evidence input remains bounded to the `2026-04-21` fuller phase1 family",
        "- archive modeling notes remain read-only historical inputs",
        "- only `ASTRA/E0` has a current runtime-smoke candidate input; the other public variants remain evidence-pending",
        "",
        "## Lane Posture",
        "",
    ]
    for row in gate_rows:
        lines.append(
            f"- `{row['variant_id']}` (`{row['internal_experiment_id']}`) "
            f"role=`{row['module_role']}` modeling=`{row['phase2_modeling_status']}` "
            f"phase3=`{row['phase3_entry_status']}` blockers=`{_join_semicolon(row['phase3_entry_blockers'])}`"
        )
    lines.extend(
        [
            "",
            "## HOPS Surface Clarification",
            "",
        ]
    )
    if HOPS_OPTIONAL_FIELDS & set(PHASE1_SUMMARY_FIELDS):
        lines.extend(
            [
                "The current phase1 summary schema already exports HOPS buffer/residency/backpressure fields, so the",
                "phase2 required-field set includes those summary fields directly. `flow_buffer_trace.csv` remains the",
                "lane-specific handoff output for the detailed per-layer buffer surface.",
            ]
        )
    else:
        lines.extend(
            [
                "The current phase1 summary schema does not export the full HOPS buffer/residency/backpressure surface,",
                "so phase2 treats `timeline_summary.csv` and `flow_buffer_trace.csv` as the controlling handoff outputs",
                "for those diagnostics.",
            ]
        )
    lines.extend(
        [
            "",
            "## Governance Invariants",
            "",
            "- `runtime_smoke` remains the only enabled execution-evidence tier in the current lane",
            "- all accelerator-backed follow-on execution remains `mps` only",
            "- long runs remain under `caffeinate -dimsu`",
            "- archived rows remain historical input only and may not be relabeled as fresh evidence",
        ]
    )
    return "\n".join(lines) + "\n"


def _render_governance_gate_matrix_note(gate_rows: list[dict[str, Any]]) -> str:
    lines = [
        "# FULLER Phase2 Governance Gate Matrix",
        "",
        "Date: `2026-04-21`",
        "Status: `current_phase2_gate_surface`",
        "",
        "## Lane Table",
        "",
        "| variant | internal | role | accuracy | runtime_smoke | analysis_grade | phase3 | blockers |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in gate_rows:
        lines.append(
            f"| `{row['variant_id']}` | `{row['internal_experiment_id']}` | `{row['module_role']}` | "
            f"`{row['accuracy_contract_status']}` | `{row['runtime_smoke_gate_status']}` | "
            f"`{row['analysis_grade_gate_status']}` | `{row['phase3_entry_status']}` | "
            f"`{_join_semicolon(row['phase3_entry_blockers'])}` |"
        )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "This table is the current phase2 truth surface. It does not inherit archive-era `postopt7` measured",
            "closure language. The current blockers are:",
            "",
            "- `ASTRA`: `context_match_incomplete`",
            "- `MESO/HOPS/DET/SPARSE/PHY/FULLER`: `missing_current_accuracy_row`",
            "",
            "Accordingly, phase2 is complete at the modeled-control-surface level, while phase3 and phase4 remain",
            "blocked on current lane evidence collection.",
        ]
    )
    return "\n".join(lines) + "\n"


def _render_phase4_handoff_note(
    *,
    contract: dict[str, Any],
    handoff_rows: list[dict[str, Any]],
) -> str:
    governance = contract.get("governance") or {}
    lines = [
        "# FULLER Phase2 Phase4 Evidence Handoff Contract",
        "",
        "Date: `2026-04-21`",
        "Status: `current_phase4_handoff_contract`",
        "",
        "## Path Tokens",
        "",
        "- `{phase1_run_output_dir}`: the lane-local output directory written by `phase1_runner.py`",
        "- `{launch_progress_root}`: the lane-local launch/prep progress directory that owns `progress/manifest.json`",
        "",
        "## Governance Fields",
        "",
        f"- `required_device={governance.get('required_device')}`",
        f"- `long_run_wrapper={json.dumps(governance.get('long_run_wrapper') or [], ensure_ascii=False)}`",
        f"- `cpu_fallback_forbidden={str(bool(governance.get('cpu_fallback_forbidden'))).lower()}`",
        f"- `archived_row_relabel_forbidden={str(bool(governance.get('archived_row_relabel_forbidden'))).lower()}`",
        f"- `analysis_grade_enabled={str(bool(governance.get('analysis_grade_enabled'))).lower()}`",
        f"- `authorization_required_before_long_run={str(bool(governance.get('authorization_required_before_long_run'))).lower()}`",
        "",
        "## Lane Requirements",
        "",
    ]
    for row in handoff_rows:
        lines.append(
            f"- `{row['variant_id']}` (`{row['internal_experiment_id']}`) config=`{row['config_stub']}` "
            f"generated_config=`{row['expected_generated_config_path']}`"
        )
        lines.append(f"  required outputs: `{', '.join(row['required_outputs_json'])}`")
        lines.append(f"  required manifest fields: `{', '.join(row['required_manifest_fields_json'])}`")
        lines.append(f"  required summary fields: `{', '.join(row['required_summary_fields_json'])}`")
    lines.extend(
        [
            "",
            "The HOPS/FULLER buffer handoff surface is currently emitted as `flow_buffer_trace.csv`, which is the repo's",
            "current concrete equivalent of the requested per-layer buffer output.",
        ]
    )
    return "\n".join(lines) + "\n"


def build_phase2_modeling_surface(
    contract_path: Path = DEFAULT_CONTRACT,
    *,
    root_dir: Path = ROOT,
) -> dict[str, Any]:
    resolved_contract_path = contract_path if contract_path.is_absolute() else root_dir / contract_path
    contract = _load_yaml(resolved_contract_path)
    sources = contract.get("sources") or {}
    outputs = contract.get("outputs") or {}

    canonical_bundle = _load_yaml(_resolve_path(root_dir, sources["phase1_canonical_bundle"]))
    runtime_smoke_bundle = _load_yaml(_resolve_path(root_dir, sources["phase1_runtime_smoke_bundle"]))
    canonical_lookup, canonical_generated_dir = _normalize_variant_lookup(canonical_bundle)
    overlay_lookup, overlay_generated_dir = _normalize_variant_lookup(runtime_smoke_bundle)
    _ensure_matching_bundles(canonical_lookup, overlay_lookup)

    preflight_payload = _load_json(_resolve_path(root_dir, sources["phase1_preflight_json"]))
    preflight_rows, _ = _build_lane_lookups(preflight_payload)
    _ = _load_csv(_resolve_path(root_dir, sources["phase1_accuracy_contract_csv"]))
    _ = _load_csv(_resolve_path(root_dir, sources["phase1_runtime_smoke_lane_status_csv"]))
    current_candidate_rows = _load_csv(_resolve_path(root_dir, sources["phase1_current_runtime_smoke_candidate_csv"]))
    current_candidate_experiment_ids = _current_candidate_experiments(current_candidate_rows)

    gate_rows = _build_gate_rows(
        contract=contract,
        overlay_lookup=overlay_lookup,
        preflight_rows=preflight_rows,
        current_candidate_experiment_ids=current_candidate_experiment_ids,
    )
    handoff_rows = _build_handoff_rows(
        contract=contract,
        overlay_lookup=overlay_lookup,
        overlay_generated_dir=_resolve_path(root_dir, overlay_generated_dir),
        gate_rows=gate_rows,
    )

    gate_matrix_csv = _resolve_path(root_dir, outputs["gate_matrix_csv"])
    gate_matrix_json = _resolve_path(root_dir, outputs["gate_matrix_json"])
    handoff_contract_csv = _resolve_path(root_dir, outputs["handoff_contract_csv"])
    handoff_contract_json = _resolve_path(root_dir, outputs["handoff_contract_json"])
    modeling_manifest_json = _resolve_path(root_dir, outputs["modeling_manifest_json"])
    modeling_decision_md = _resolve_path(root_dir, outputs["modeling_decision_md"])
    governance_gate_matrix_md = _resolve_path(root_dir, outputs["governance_gate_matrix_md"])
    phase4_handoff_md = _resolve_path(root_dir, outputs["phase4_handoff_md"])

    _write_csv(gate_matrix_csv, GATE_MATRIX_FIELDS, _csv_gate_rows(gate_rows))
    _write_csv(handoff_contract_csv, HANDOFF_FIELDS, _csv_handoff_rows(handoff_rows))
    _write_json(
        gate_matrix_json,
        {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "rows": gate_rows,
        },
    )
    _write_json(
        handoff_contract_json,
        {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "rows": handoff_rows,
        },
    )
    _write_text(
        modeling_decision_md,
        _render_modeling_decision_note(
            contract=contract,
            gate_rows=gate_rows,
            current_candidate_experiment_ids=current_candidate_experiment_ids,
        ),
    )
    _write_text(governance_gate_matrix_md, _render_governance_gate_matrix_note(gate_rows))
    _write_text(
        phase4_handoff_md,
        _render_phase4_handoff_note(contract=contract, handoff_rows=handoff_rows),
    )

    manifest_payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "phase2_status": "completed",
        "contract_path": str(resolved_contract_path.resolve()),
        "source_artifacts": {
            key: str(_resolve_path(root_dir, value).resolve())
            for key, value in sources.items()
        },
        "historical_inputs": [
            str(_resolve_path(root_dir, item).resolve()) for item in contract.get("historical_inputs") or []
        ],
        "generated_outputs": {
            "gate_matrix_csv": str(gate_matrix_csv.resolve()),
            "gate_matrix_json": str(gate_matrix_json.resolve()),
            "handoff_contract_csv": str(handoff_contract_csv.resolve()),
            "handoff_contract_json": str(handoff_contract_json.resolve()),
            "modeling_manifest_json": str(modeling_manifest_json.resolve()),
            "modeling_decision_md": str(modeling_decision_md.resolve()),
            "governance_gate_matrix_md": str(governance_gate_matrix_md.resolve()),
            "phase4_handoff_md": str(phase4_handoff_md.resolve()),
        },
        "active_phase_plan_md": str(
            _resolve_path(root_dir, (contract.get("coordination") or {}).get("active_phase_plan_md")).resolve()
        ),
        "row_counts": {
            "gate_matrix": len(gate_rows),
            "handoff_contract": len(handoff_rows),
        },
    }
    _write_json(modeling_manifest_json, manifest_payload)

    return manifest_payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the FULLER phase2 modeling surface.")
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    args = parser.parse_args()
    payload = build_phase2_modeling_surface(args.contract)
    print(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
