#!/usr/bin/env python3
"""Validate the fuller-accelerator preimplementation contract and emit a readiness report."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import yaml

try:
    from .path_policy import MAIN_PROJECT_REPORT_DATA_DIR, assert_main_project_path, resolve_repo_path
except ImportError:
    from path_policy import MAIN_PROJECT_REPORT_DATA_DIR, assert_main_project_path, resolve_repo_path  # type: ignore


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONTRACT = ROOT / "configs" / "fuller_optimized_accelerator_preimplementation_contract_20260319.yaml"
DEFAULT_OUT_DIR = MAIN_PROJECT_REPORT_DATA_DIR
CHECK_FIELDS = ["check_id", "status", "detail"]
INCOMPLETE_TRACEABILITY_STATUS = "bounded_architecture_reasonable_claim_traceability_incomplete"
RESTORED_TRACEABILITY_STATUS = "bounded_claim_traceability_restored_active_library_20260321"


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise SystemExit(f"Expected YAML mapping in {path}")
    return data


def _require_mapping(cfg: dict[str, Any], key: str) -> dict[str, Any]:
    value = cfg.get(key)
    if not isinstance(value, dict):
        raise SystemExit(f"Missing required mapping: {key}")
    return value


def _require_list(cfg: dict[str, Any], key: str) -> list[str]:
    value = cfg.get(key)
    if not isinstance(value, list) or not value:
        raise SystemExit(f"Missing required list: {key}")
    return [str(item) for item in value]


def _read_list(cfg: dict[str, Any], key: str) -> list[str]:
    value = cfg.get(key)
    if not isinstance(value, list):
        raise SystemExit(f"Missing required list: {key}")
    return [str(item) for item in value]


def _record(rows: list[dict[str, str]], check_id: str, status: str, detail: str) -> None:
    rows.append({"check_id": check_id, "status": status, "detail": detail})


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _count_csv_rows(path: Path) -> int:
    return len(_read_csv(path))


def evaluate_contract(contract: dict[str, Any]) -> tuple[list[dict[str, str]], dict[str, Any]]:
    checks: list[dict[str, str]] = []

    meta = _require_mapping(contract, "meta")
    paper_route = _require_mapping(contract, "paper_route")
    fuller_target = _require_mapping(contract, "fuller_target")
    wave1 = _require_mapping(contract, "wave_1_flow_pascal")
    wave2 = _require_mapping(contract, "wave_2_meso_redesign")
    wave3 = _require_mapping(contract, "wave_3_phy_support")
    architecture_traceability = _require_mapping(contract, "architecture_traceability")
    stop_rules = _require_mapping(contract, "stop_rules")
    launch_state = _require_mapping(contract, "launch_state")

    expected_route = "bounded local baseline + narrow HOPS paper"
    _record(
        checks,
        "paper_route.active_safe_route",
        "pass" if str(paper_route.get("active_safe_route")) == expected_route else "fail",
        f"active_safe_route={paper_route.get('active_safe_route')!r}",
    )
    _record(
        checks,
        "fuller_target.active_layers",
        "pass" if [str(x) for x in fuller_target.get("active_layers", [])] == ["HOPS", "MESO", "PHY"] else "fail",
        f"active_layers={fuller_target.get('active_layers')!r}",
    )
    _record(
        checks,
        "fuller_target.deferred_hooks",
        "pass" if [str(x) for x in fuller_target.get("deferred_hooks", [])] == ["DET", "SPARSE"] else "fail",
        f"deferred_hooks={fuller_target.get('deferred_hooks')!r}",
    )
    retired_hooks_note = assert_main_project_path(
        resolve_repo_path(str(fuller_target.get("retired_hooks_note_md") or "")),
        arg_name="fuller_target.retired_hooks_note_md",
    )
    _record(
        checks,
        "fuller_target.retired_hooks",
        "pass" if [str(x) for x in fuller_target.get("retired_hooks", [])] == ["DET", "SPARSE"] else "fail",
        f"retired_hooks={fuller_target.get('retired_hooks')!r}",
    )
    _record(
        checks,
        "fuller_target.retired_hooks_note_md",
        "pass" if retired_hooks_note.exists() else "fail",
        str(retired_hooks_note),
    )

    baseline_csv = assert_main_project_path(
        resolve_repo_path(str(wave1.get("baseline_summary_csv") or "")),
        arg_name="wave_1_flow_pascal.baseline_summary_csv",
    )
    measured_baseline_csv_raw = str(wave1.get("measured_baseline_csv") or "").strip()
    measured_baseline_csv = (
        assert_main_project_path(
            resolve_repo_path(measured_baseline_csv_raw),
            arg_name="wave_1_flow_pascal.measured_baseline_csv",
        )
        if measured_baseline_csv_raw
        else None
    )
    adapter_summary_csv = assert_main_project_path(
        resolve_repo_path(str(wave1.get("adapter_summary_csv") or "")),
        arg_name="wave_1_flow_pascal.adapter_summary_csv",
    )
    adapter_report_md = assert_main_project_path(
        resolve_repo_path(str(wave1.get("adapter_report_md") or "")),
        arg_name="wave_1_flow_pascal.adapter_report_md",
    )
    stage_mapping_md = assert_main_project_path(
        resolve_repo_path(str(wave1.get("execution_glue_stage_mapping_md") or "")),
        arg_name="wave_1_flow_pascal.execution_glue_stage_mapping_md",
    )
    rerun_contract_yaml = assert_main_project_path(
        resolve_repo_path(str(wave1.get("execution_glue_rerun_contract_yaml") or "")),
        arg_name="wave_1_flow_pascal.execution_glue_rerun_contract_yaml",
    )
    metrics_template_csv = assert_main_project_path(
        resolve_repo_path(str(wave1.get("execution_glue_metrics_template_csv") or "")),
        arg_name="wave_1_flow_pascal.execution_glue_metrics_template_csv",
    )
    flow_execution_freeze_note_raw = str(wave1.get("flow_execution_freeze_note_md") or "").strip()
    flow_execution_freeze_note = (
        assert_main_project_path(
            resolve_repo_path(flow_execution_freeze_note_raw),
            arg_name="wave_1_flow_pascal.flow_execution_freeze_note_md",
        )
        if flow_execution_freeze_note_raw
        else None
    )
    runtime_blocker_note = assert_main_project_path(
        resolve_repo_path(str(wave1.get("runtime_blocker_note_md") or "")),
        arg_name="wave_1_flow_pascal.runtime_blocker_note_md",
    )
    runtime_blocker_log = assert_main_project_path(
        resolve_repo_path(str(wave1.get("runtime_blocker_log") or "")),
        arg_name="wave_1_flow_pascal.runtime_blocker_log",
    )
    for label, path in (
        ("wave1.baseline_summary_csv", baseline_csv),
        ("wave1.measured_baseline_csv", measured_baseline_csv),
        ("wave1.adapter_summary_csv", adapter_summary_csv),
        ("wave1.adapter_report_md", adapter_report_md),
        ("wave1.execution_glue_stage_mapping_md", stage_mapping_md),
        ("wave1.execution_glue_rerun_contract_yaml", rerun_contract_yaml),
        ("wave1.execution_glue_metrics_template_csv", metrics_template_csv),
        ("wave1.flow_execution_freeze_note_md", flow_execution_freeze_note),
        ("wave1.runtime_blocker_note_md", runtime_blocker_note),
        ("wave1.runtime_blocker_log", runtime_blocker_log),
    ):
        if path is None:
            continue
        _record(checks, label, "pass" if path.exists() else "fail", str(path))

    adapter_rows = _read_csv(adapter_summary_csv)
    required_row_roles = set(_require_list(wave1, "required_row_roles"))
    observed_roles = {str(row.get("row_role") or "") for row in adapter_rows}
    _record(
        checks,
        "wave1.required_row_roles",
        "pass" if required_row_roles.issubset(observed_roles) else "fail",
        f"required={sorted(required_row_roles)} observed={sorted(observed_roles)}",
    )
    flow_rows = [row for row in adapter_rows if row.get("row_role") == "flow_candidate"]
    if len(flow_rows) != 1:
        raise SystemExit(f"Expected exactly one flow_candidate row in {adapter_summary_csv}")
    flow_row = flow_rows[0]
    expected_workload = str(wave1.get("workload_id") or "").strip()
    expected_model = str(wave1.get("model") or "").strip()
    _record(
        checks,
        "wave1.flow_candidate_identity",
        "pass"
        if flow_row.get("workload_id") == expected_workload and flow_row.get("model") == expected_model
        else "fail",
        f"workload={flow_row.get('workload_id')!r} model={flow_row.get('model')!r}",
    )
    _record(
        checks,
        "wave1.current_state",
        "pass"
        if str(wave1.get("current_state")) in {
            "adapter_surface_ready_pending_flow_metrics",
            "execution_glue_ready_pending_flow_metrics",
            "execution_glue_ready_blocked_by_mps_runtime",
            "unsandboxed_mps_verified_pending_baseline_materialization",
            "unsandboxed_mps_verified_baseline_measured_pending_flow_metrics",
            "unsandboxed_mps_verified_baseline_measured_flow_execution_frozen_pending_flow_metrics",
            "paired_flow_ready",
        }
        else "fail",
        f"current_state={wave1.get('current_state')!r}",
    )
    current_state = str(wave1.get("current_state") or "")
    if current_state == "execution_glue_ready_blocked_by_mps_runtime":
        blocker_text = runtime_blocker_log.read_text(encoding="utf-8") if runtime_blocker_log.exists() else ""
        _record(
            checks,
            "wave1.mps_runtime_blocker_log",
            "pass" if "MPS-only policy blocked execution" in blocker_text else "fail",
            blocker_text.strip()[:240] or str(runtime_blocker_log),
        )
    if current_state == "unsandboxed_mps_verified_baseline_measured_pending_flow_metrics":
        measured_rows = _read_csv(measured_baseline_csv) if measured_baseline_csv is not None else []
        measured_ok = (
            len(measured_rows) == 1
            and measured_rows[0].get("row_status") == "measured"
            and measured_rows[0].get("workload_id") == expected_workload
            and measured_rows[0].get("model") == expected_model
        )
        _record(
            checks,
            "wave1.measured_baseline_row",
            "pass" if measured_ok else "fail",
            repr(measured_rows[0] if measured_rows else {}),
        )
    if current_state == "unsandboxed_mps_verified_baseline_measured_flow_execution_frozen_pending_flow_metrics":
        measured_rows = _read_csv(measured_baseline_csv) if measured_baseline_csv is not None else []
        measured_ok = (
            len(measured_rows) == 1
            and measured_rows[0].get("row_status") == "measured"
            and measured_rows[0].get("workload_id") == expected_workload
            and measured_rows[0].get("model") == expected_model
        )
        _record(
            checks,
            "wave1.measured_baseline_row",
            "pass" if measured_ok else "fail",
            repr(measured_rows[0] if measured_rows else {}),
        )
        rerun_contract = _load_yaml(rerun_contract_yaml)
        rerun_run = _require_mapping(rerun_contract, "run")
        flow_execute_command = str(rerun_run.get("flow_execute_command") or "").strip()
        flow_console_log = str(rerun_run.get("flow_console_log") or "").strip()
        flow_results_csv = str(rerun_run.get("flow_results_csv") or "").strip()
        flow_contract_ok = (
            "eval_cvnets_segmentation_noise.py" in flow_execute_command
            and "caffeinate" in flow_execute_command
            and expected_workload in flow_execute_command
            and flow_console_log
            and flow_results_csv
        )
        _record(
            checks,
            "wave1.flow_execution_contract",
            "pass" if flow_contract_ok else "fail",
            f"flow_execute_command={flow_execute_command!r} flow_console_log={flow_console_log!r} flow_results_csv={flow_results_csv!r}",
        )

    meso_report = resolve_repo_path(str(wave2.get("meso_status_report_md") or ""))
    meso_note = resolve_repo_path(str(wave2.get("redesign_scope_note") or ""))
    meso_impl_note = resolve_repo_path(str(wave2.get("implementation_note_md") or ""))
    phy_report = resolve_repo_path(str(wave3.get("phy_scope_report_md") or ""))
    phy_note = resolve_repo_path(str(wave3.get("support_boundary_note") or ""))
    for label, path in (
        ("wave2.meso_status_report_md", meso_report),
        ("wave2.redesign_scope_note", meso_note),
        ("wave2.implementation_note_md", meso_impl_note),
        ("wave3.phy_scope_report_md", phy_report),
        ("wave3.support_boundary_note", phy_note),
    ):
        _record(checks, label, "pass" if path.exists() else "fail", str(path))

    _record(
        checks,
        "wave2.required_explicit_terms",
        "pass" if len(_require_list(wave2, "required_explicit_terms")) >= 5 else "fail",
        f"terms={wave2.get('required_explicit_terms')!r}",
    )
    _record(
        checks,
        "wave3.forbidden_scope",
        "pass"
        if set(_require_list(wave3, "forbidden_scope")) >= {"standalone_headline_gain", "hardware_validated_oracle"}
        else "fail",
        f"forbidden_scope={wave3.get('forbidden_scope')!r}",
    )
    _record(
        checks,
        "stop_rules.forbidden_reopens",
        "pass" if _require_list(stop_rules, "forbidden_reopens") == ["DET", "SPARSE"] else "fail",
        f"forbidden_reopens={stop_rules.get('forbidden_reopens')!r}",
    )
    forbidden_actions = set(_require_list(stop_rules, "forbidden_actions"))
    _record(
        checks,
        "stop_rules.forbidden_actions",
        "pass"
        if {
            "global_multitask_phase1_rewrite",
            "five_mechanism_simultaneous_launch",
            "claim_promotion_before_traceability_restore",
            "claim_promotion_beyond_bounded_traceability_surface",
        }.issubset(forbidden_actions)
        else "fail",
        f"forbidden_actions={sorted(forbidden_actions)!r}",
    )

    final_blockers = _read_list(launch_state, "final_blockers")
    bounded_collection_ready = bool(launch_state.get("bounded_collection_ready"))
    bounded_paper_writing_ready = bool(launch_state.get("bounded_paper_writing_ready"))
    large_scale_launch_ready = bool(launch_state.get("large_scale_launch_ready"))
    flow_ready = str(flow_row.get("row_status") or "") == "ready_candidate"
    meso_ready = str(wave2.get("current_state") or "") == "explicit_cost_model_implemented"
    review_note = resolve_repo_path(str(architecture_traceability.get("review_note_md") or ""))
    review_status = str(architecture_traceability.get("review_status") or "")
    claim_traceability_ready = bool(architecture_traceability.get("claim_traceability_ready"))
    missing_primary_artifacts = _read_list(architecture_traceability, "missing_primary_local_reference_artifacts")
    claim_blockers = _read_list(architecture_traceability, "claim_blockers")
    _record(
        checks,
        "architecture_traceability.review_note_md",
        "pass" if review_note.exists() else "fail",
        str(review_note),
    )
    _record(
        checks,
        "architecture_traceability.review_status",
        "pass"
        if review_status in {INCOMPLETE_TRACEABILITY_STATUS, RESTORED_TRACEABILITY_STATUS}
        else "fail",
        f"review_status={review_status!r}",
    )
    missing_artifact_states: list[str] = []
    if review_status == INCOMPLETE_TRACEABILITY_STATUS:
        all_missing_absent = True
        for raw in missing_primary_artifacts:
            path = resolve_repo_path(raw)
            is_missing = not path.exists()
            all_missing_absent = all_missing_absent and is_missing
            missing_artifact_states.append(f"{raw}::{ 'missing' if is_missing else 'present' }")
        _record(
            checks,
            "architecture_traceability.missing_primary_local_reference_artifacts",
            "pass" if all_missing_absent else "fail",
            "; ".join(missing_artifact_states),
        )
        _record(
            checks,
            "architecture_traceability.claim_blockers",
            "pass" if (not claim_traceability_ready and bool(claim_blockers)) else "fail",
            f"claim_traceability_ready={claim_traceability_ready} claim_blockers={claim_blockers!r}",
        )
    elif review_status == RESTORED_TRACEABILITY_STATUS:
        reference_root = assert_main_project_path(
            resolve_repo_path(str(architecture_traceability.get("active_local_reference_root") or "")),
            arg_name="architecture_traceability.active_local_reference_root",
        )
        reference_index = assert_main_project_path(
            resolve_repo_path(str(architecture_traceability.get("active_local_reference_index_csv") or "")),
            arg_name="architecture_traceability.active_local_reference_index_csv",
        )
        reference_validation = assert_main_project_path(
            resolve_repo_path(str(architecture_traceability.get("active_local_reference_validation_md") or "")),
            arg_name="architecture_traceability.active_local_reference_validation_md",
        )
        quarantine_manifest = assert_main_project_path(
            resolve_repo_path(
                str(architecture_traceability.get("active_local_reference_quarantine_manifest_csv") or "")
            ),
            arg_name="architecture_traceability.active_local_reference_quarantine_manifest_csv",
        )
        retired_manifest = assert_main_project_path(
            resolve_repo_path(
                str(architecture_traceability.get("active_local_reference_retired_manifest_csv") or "")
            ),
            arg_name="architecture_traceability.active_local_reference_retired_manifest_csv",
        )
        for label, path, predicate in (
            ("architecture_traceability.active_local_reference_root", reference_root, reference_root.is_dir),
            ("architecture_traceability.active_local_reference_index_csv", reference_index, reference_index.is_file),
            ("architecture_traceability.active_local_reference_validation_md", reference_validation, reference_validation.is_file),
            (
                "architecture_traceability.active_local_reference_quarantine_manifest_csv",
                quarantine_manifest,
                quarantine_manifest.is_file,
            ),
            (
                "architecture_traceability.active_local_reference_retired_manifest_csv",
                retired_manifest,
                retired_manifest.is_file,
            ),
        ):
            _record(checks, label, "pass" if predicate() else "fail", str(path))

        expected_counts = _require_mapping(architecture_traceability, "active_reference_expected_counts")
        active_indexed_pdfs = int(expected_counts.get("active_indexed_pdfs") or 0)
        quarantined_entries = int(expected_counts.get("quarantined_entries") or 0)
        retired_entries = int(expected_counts.get("retired_entries") or 0)
        observed_counts = {
            "active_indexed_pdfs": _count_csv_rows(reference_index),
            "quarantined_entries": _count_csv_rows(quarantine_manifest),
            "retired_entries": _count_csv_rows(retired_manifest),
        }
        counts_ok = (
            observed_counts["active_indexed_pdfs"] == active_indexed_pdfs
            and observed_counts["quarantined_entries"] == quarantined_entries
            and observed_counts["retired_entries"] == retired_entries
        )
        _record(
            checks,
            "architecture_traceability.active_reference_expected_counts",
            "pass" if counts_ok else "fail",
            f"expected={expected_counts!r} observed={observed_counts!r}",
        )

        validation_text = reference_validation.read_text(encoding="utf-8") if reference_validation.exists() else ""
        validation_fragments = [
            f"Active indexed PDFs: `{active_indexed_pdfs}`",
            f"Quarantined historical mismatches: `{quarantined_entries}`",
        ]
        _record(
            checks,
            "architecture_traceability.active_reference_validation_state",
            "pass" if all(fragment in validation_text for fragment in validation_fragments) else "fail",
            validation_text.strip()[:240] or str(reference_validation),
        )
        bounded_claims = set(_require_list(architecture_traceability, "bounded_claims"))
        claim_guardrails = set(_require_list(architecture_traceability, "claim_guardrails"))
        _record(
            checks,
            "architecture_traceability.bounded_claims",
            "pass"
            if {
                "ASTRA substrate anchor",
                "HOPS primary bounded paper route",
                "MESO support-only explicit cost/topology realism",
                "PHY support-only bounded realism and calibration",
                "MobileViT workload anchor",
            }.issubset(bounded_claims)
            else "fail",
            f"bounded_claims={sorted(bounded_claims)!r}",
        )
        _record(
            checks,
            "architecture_traceability.claim_guardrails",
            "pass"
            if {
                "DET and SPARSE are retired from the active architecture",
                "MESO is not a standalone headline claim",
                "PHY is not a hardware-validated oracle",
                "No generalization beyond the bounded MobileViT route",
            }.issubset(claim_guardrails)
            else "fail",
            f"claim_guardrails={sorted(claim_guardrails)!r}",
        )
        _record(
            checks,
            "architecture_traceability.missing_primary_local_reference_artifacts",
            "pass" if not missing_primary_artifacts else "fail",
            repr(missing_primary_artifacts),
        )
        _record(
            checks,
            "architecture_traceability.claim_blockers",
            "pass" if (claim_traceability_ready and not claim_blockers) else "fail",
            f"claim_traceability_ready={claim_traceability_ready} claim_blockers={claim_blockers!r}",
        )
    else:
        _record(
            checks,
            "architecture_traceability.missing_primary_local_reference_artifacts",
            "fail",
            f"unsupported review_status={review_status!r}",
        )
        _record(
            checks,
            "architecture_traceability.claim_blockers",
            "fail",
            f"unsupported review_status={review_status!r}",
        )
    bounded_collection_coherent = (
        (flow_ready and meso_ready and bounded_collection_ready and not final_blockers)
        or ((not flow_ready or not meso_ready) and (not bounded_collection_ready) and bool(final_blockers))
    )
    _record(
        checks,
        "launch_state.bounded_collection_coherence",
        "pass" if bounded_collection_coherent else "fail",
        (
            f"flow_ready={flow_ready} meso_ready={meso_ready} "
            f"bounded_collection_ready={bounded_collection_ready} final_blockers={final_blockers!r}"
        ),
    )
    _record(
        checks,
        "launch_state.final_blockers",
        "pass"
        if (bounded_collection_ready and not final_blockers) or ((not bounded_collection_ready) and bool(final_blockers))
        else "fail",
        f"bounded_collection_ready={bounded_collection_ready} final_blockers={final_blockers!r}",
    )
    bounded_paper_writing_coherent = (
        bounded_paper_writing_ready
        == (bounded_collection_ready and claim_traceability_ready and not final_blockers)
    )
    _record(
        checks,
        "launch_state.bounded_paper_writing_ready",
        "pass" if bounded_paper_writing_coherent else "fail",
        (
            f"bounded_paper_writing_ready={bounded_paper_writing_ready} "
            f"bounded_collection_ready={bounded_collection_ready} "
            f"claim_traceability_ready={claim_traceability_ready} "
            f"final_blockers={final_blockers!r}"
        ),
    )
    _record(
        checks,
        "launch_state.large_scale_launch_ready",
        "pass" if not large_scale_launch_ready else "fail",
        f"large_scale_launch_ready={large_scale_launch_ready!r}",
    )
    if meso_ready:
        meso_text = meso_report.read_text(encoding="utf-8") if meso_report.exists() else ""
        _record(
            checks,
            "wave2.explicit_cost_model_report",
            "pass"
            if "implemented explicit-cost model" in meso_text and "cost_model_mode=explicit_topology_v1" in meso_text
            else "fail",
            meso_text.strip()[:240] or str(meso_report),
        )

    summary = {
        "tag": str(meta.get("tag") or "unknown"),
        "paper_route": paper_route,
        "fuller_target": fuller_target,
        "wave1_state": str(wave1.get("current_state") or ""),
        "flow_row_status": str(flow_row.get("row_status") or ""),
        "wave2_state": str(wave2.get("current_state") or ""),
        "wave3_state": str(wave3.get("current_state") or ""),
        "preimplementation_pack_complete": bool(launch_state.get("preimplementation_pack_complete")),
        "bounded_collection_ready": bounded_collection_ready,
        "bounded_paper_writing_ready": bounded_paper_writing_ready,
        "large_scale_launch_ready": large_scale_launch_ready,
        "final_blockers": final_blockers,
        "review_status": review_status,
        "claim_traceability_ready": claim_traceability_ready,
        "claim_blockers": claim_blockers,
        "overall_ok": all(row["status"] == "pass" for row in checks),
    }
    return checks, summary


def _write_report(path: Path, checks: list[dict[str, str]], summary: dict[str, Any], contract_path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Fuller Preimplementation Readiness Report",
        "",
        "Scope",
        f"- Contract: `{contract_path}`",
        f"- Tag: `{summary['tag']}`",
        f"- Active route: `{summary['paper_route']['active_safe_route']}`",
        f"- Fuller target: `{summary['fuller_target']['name']}`",
        "",
        "State",
        f"- Preimplementation pack complete: `{summary['preimplementation_pack_complete']}`",
        f"- Bounded collection ready: `{summary['bounded_collection_ready']}`",
        f"- Bounded paper writing ready: `{summary['bounded_paper_writing_ready']}`",
        f"- Large-scale launch ready: `{summary['large_scale_launch_ready']}`",
        f"- Review status: `{summary['review_status']}`",
        f"- Claim traceability ready: `{summary['claim_traceability_ready']}`",
        f"- Wave 1 state: `{summary['wave1_state']}` with HOPS row status `{summary['flow_row_status']}`",
        f"- Wave 2 state: `{summary['wave2_state']}`",
        f"- Wave 3 state: `{summary['wave3_state']}`",
        "",
        "Final blockers",
    ]
    if summary["final_blockers"]:
        for blocker in summary["final_blockers"]:
            lines.append(f"- `{blocker}`")
    else:
        lines.append("- `none`")
    lines.extend(["", "Claim blockers"])
    if summary["claim_blockers"]:
        for blocker in summary["claim_blockers"]:
            lines.append(f"- `{blocker}`")
    else:
        lines.append("- `none`")
    lines.extend(["", "Checks"])
    for row in checks:
        lines.append(f"- `{row['status']}` `{row['check_id']}`: {row['detail']}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Check fuller preimplementation readiness.")
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    contract_path = assert_main_project_path(resolve_repo_path(args.contract), arg_name="--contract")
    out_dir = assert_main_project_path(resolve_repo_path(args.out_dir), arg_name="--out_dir")
    contract = _load_yaml(contract_path)
    checks, summary = evaluate_contract(contract)
    tag = summary["tag"]
    report_path = out_dir / f"fuller_preimplementation_readiness_{tag}.md"
    _write_report(report_path, checks, summary, contract_path)
    overall = "OK" if summary["overall_ok"] else "FAIL"
    print(f"[fuller-preimpl] overall={overall} report={report_path}")
    if not summary["overall_ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
