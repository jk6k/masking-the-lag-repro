#!/usr/bin/env python3
"""Materialize the current FULLER analysis-grade replay gate and plan surfaces."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from .fuller_experiment_program_common import (
        ANALYSIS_GRADE_MAINLINE_LANES,
        _active_target_module_keys,
        _analysis_grade_baseline_reference_run_id,
        _analysis_grade_baseline_source_step_id,
        _analysis_grade_canonical_baseline_lane,
        _analysis_grade_claim_lane_scope,
        _analysis_grade_lane_pass_mode,
        _analysis_grade_lane_sample_budget,
        _analysis_grade_redirected_support_lanes,
        _host_tuning_profile_path,
        DEFAULT_CONTRACT,
        HOST_TUNING_PROVENANCE_FIELDS,
        LEGACY_REPLACEMENT_POLICY,
        ROOT,
        command_runtime_policy_from_bundle,
        _full_eval_manifest_path,
        _governance,
        _phase1_run_id,
        _resolve_path,
        host_tuning_provenance_fields,
        resolve_fuller_accuracy_policy_bundle,
        runtime_health_gate_from_policy,
        _write_csv,
        _write_json,
        _write_text,
        load_program_context,
    )
except ImportError:
    from fuller_experiment_program_common import (  # type: ignore
        ANALYSIS_GRADE_MAINLINE_LANES,
        _active_target_module_keys,
        _analysis_grade_baseline_reference_run_id,
        _analysis_grade_baseline_source_step_id,
        _analysis_grade_canonical_baseline_lane,
        _analysis_grade_claim_lane_scope,
        _analysis_grade_lane_pass_mode,
        _analysis_grade_lane_sample_budget,
        _analysis_grade_redirected_support_lanes,
        _host_tuning_profile_path,
        DEFAULT_CONTRACT,
        HOST_TUNING_PROVENANCE_FIELDS,
        LEGACY_REPLACEMENT_POLICY,
        ROOT,
        command_runtime_policy_from_bundle,
        _full_eval_manifest_path,
        _governance,
        _phase1_run_id,
        _resolve_path,
        host_tuning_provenance_fields,
        resolve_fuller_accuracy_policy_bundle,
        runtime_health_gate_from_policy,
        _write_csv,
        _write_json,
        _write_text,
        load_program_context,
    )

DATE_TAG = "20260423"
RUNTIME_GOVERNANCE_STATUS_CSV = (
    ROOT / "experiments" / "results" / "report_data" / "fuller_runtime_lane_governance_status_matrix_20260423.csv"
)
ANALYSIS_OUTPUT_ROOT = (
    ROOT / "experiments" / "results" / "report_data" / f"{DATE_TAG}_fuller_analysis_grade_replay"
)

GATE_FIELDS = [
    "lane_id",
    "internal_experiment_id",
    "engineering_smoke_status",
    "result_surface_status",
    "inherited_runtime_blockers_json",
    "analysis_grade_gate_status",
    "analysis_grade_gate_blockers_json",
    "required_seeds_json",
    "required_pass_mode",
    "required_manifest_path",
    *HOST_TUNING_PROVENANCE_FIELDS,
    "phase4_eligible",
    "claim_tier",
    "materialized_step_id",
    "next_action",
]

PLAN_FIELDS = [
    "step_id",
    "lane_id",
    "internal_experiment_id",
    "package_kind",
    "command_json",
    "checker_kind",
    "checker_command_json",
    "progress_root",
    "required_device",
    "launch_wrapper_json",
    "serial_group",
    "resume_enabled",
    "stop_on_failure",
    "execution_authorized",
    "current_run_policy",
    "sample_budget",
    "slice_manifest_path",
    "pass_mode",
    "host_tuning_profile_json",
    "host_id",
    *HOST_TUNING_PROVENANCE_FIELDS,
    "baseline_source_step_id",
    "baseline_reference_csv",
    "baseline_reference_run_id",
    "baseline_reference_summary_json",
    "phase4_eligible",
    "legacy_replacement_policy",
]


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _rows_by_lane(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    payload: dict[str, dict[str, str]] = {}
    for row in rows:
        lane_id = str(row.get("lane_id") or "").strip().upper()
        if lane_id:
            payload[lane_id] = row
    return payload


def _lane_output_root(lane_id: str) -> Path:
    return ANALYSIS_OUTPUT_ROOT / str(lane_id).strip().lower()


def _lane_results_csv(lane_id: str) -> Path:
    return _lane_output_root(lane_id) / "raw_accuracy.csv"


def _lane_annotated_csv(lane_id: str) -> Path:
    return _lane_output_root(lane_id) / "annotated_accuracy.csv"


def _lane_progress_root(lane_id: str) -> Path:
    return _lane_output_root(lane_id) / "progress"


def _lane_prepared_phase1_root(lane_id: str) -> Path:
    return _lane_output_root(lane_id) / "prepared_phase1_configs"


def _lane_prepared_eligibility_root(lane_id: str) -> Path:
    return _lane_output_root(lane_id) / "prepared_eligibility"


def _gate_note(rows: list[dict[str, Any]]) -> str:
    ready_rows = [
        row for row in rows if str(row["analysis_grade_gate_status"]).startswith("ready_pending")
    ]
    lines = [
        "# FULLER Analysis-Grade Replay Gate",
        "",
        "Date: `2026-04-23`",
        "Status: `analysis_grade_gate_materialized_current`",
        "",
        "## Current Gate",
        "",
        "- `analysis_grade_replay` is now materialized as a redesigned current gated family.",
        "- `ASTRA` remains the canonical full-manifest paired baseline lane.",
        "- `MESO/HOPS/DET/SPARSE/FULLER` are staged as full-manifest `quantized_only` replays that reuse the ASTRA baseline reference instead of restaging paired baselines lane-by-lane.",
        "- `PHY` is no longer part of the main claim-tier analysis-grade lane family and is redirected to `realism_calibration_support`.",
        "",
        "## Lane Status",
        "",
    ]
    lines.extend(
        f"- `{row['lane_id']}` gate=`{row['analysis_grade_gate_status']}` blockers=`{row['analysis_grade_gate_blockers_json']}`"
        for row in rows
    )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            f"- ready_lanes: `{len(ready_rows)}/{len(rows)}`",
            "- full-dataset replay remains `mps` only and must launch under `caffeinate -dimsu`.",
            "- claim-tier evidence still requires ASTRA canonical completion plus later phase4 intake of the redesigned mainline replay family.",
        ]
    )
    return "\n".join(lines) + "\n"


def _materialization_note(
    rows: list[dict[str, Any]],
    *,
    full_manifest: Path,
    redirected_supports: dict[str, str],
) -> str:
    projected_seed_jobs = len(rows) * 3
    projected_eval_passes = sum(6 if str(row["pass_mode"]) == "paired" else 3 for row in rows)
    lines = [
        "# FULLER Analysis-Grade Replay Materialization Plan",
        "",
        "Date: `2026-04-23`",
        "Status: `analysis_grade_plan_redesigned_current`",
        "",
        "## Plan Shape",
        "",
        f"- lane_count: `{len(rows)}`",
        "- canonical_baseline_lane: `ASTRA`",
        "- required_seeds: `0,1,2`",
        f"- full_eval_manifest: `{full_manifest}`",
        f"- projected_seed_jobs: `{projected_seed_jobs}`",
        f"- projected_eval_passes: `{projected_eval_passes}`",
        f"- redirected_support_lanes: `{redirected_supports}`",
        "",
        "## Materialized Steps",
        "",
    ]
    lines.extend(
        f"- `{row['step_id']}` lane=`{row['lane_id']}` pass_mode=`{row['pass_mode']}` baseline_source=`{row['baseline_source_step_id'] or 'self'}`"
        for row in rows
    )
    lines.extend(
        [
            "",
            "## Gate",
            "",
            "- ASTRA is the only canonical paired baseline lane in this family.",
            "- MESO/HOPS/DET/SPARSE/FULLER reuse the ASTRA baseline reference and therefore cut the projected full-manifest pass count in half versus the old seven-lane paired template.",
            "- PHY support/calibration work is intentionally removed from this main claim-tier queue and handled by `realism_calibration_support`.",
        ]
    )
    return "\n".join(lines) + "\n"


def build_fuller_analysis_grade_replay_surface(
    contract_path: Path = DEFAULT_CONTRACT,
    *,
    root_dir: Path = ROOT,
) -> dict[str, Any]:
    ctx = load_program_context(contract_path, root_dir=root_dir)
    outputs = ctx.contract.get("outputs") or {}
    governance = _governance(ctx)
    required_device = str(governance.get("required_device") or "").strip()
    launch_wrapper = [str(item) for item in governance.get("launch_wrapper") or []]
    runtime_rows = _rows_by_lane(_load_csv(RUNTIME_GOVERNANCE_STATUS_CSV))
    full_manifest = _full_eval_manifest_path(ctx)
    target_module_keys = _active_target_module_keys(ctx)
    python_bin = ROOT / ".venv311-mps" / "bin" / "python"
    runner = ROOT / "experiments" / "tools" / "run_config_conditioned_accuracy_matrix.py"
    checker = ROOT / "experiments" / "tools" / "check_fuller_analysis_grade_replay_surface.py"

    claim_lanes = _analysis_grade_claim_lane_scope(ctx)
    canonical_lane = _analysis_grade_canonical_baseline_lane(ctx)
    baseline_reference_csv = _lane_results_csv(canonical_lane)
    baseline_reference_run_id = _analysis_grade_baseline_reference_run_id(ctx)
    redirected_supports = _analysis_grade_redirected_support_lanes(ctx)
    host_tuning_profile_json = str(_host_tuning_profile_path(ctx))
    limited_surface_scope = str(
        (ctx.contract.get("engineering_smoke") or {}).get("limited_runtime_surface_scope")
        or "limited_linear_attention_pilot"
    )

    gate_rows: list[dict[str, Any]] = []
    plan_rows: list[dict[str, Any]] = []

    def _gate_value(gate: dict[str, Any], key: str, default: float | int) -> str:
        value = gate.get(key)
        if value in (None, ""):
            return str(default)
        return str(value)

    for lane_id in claim_lanes:
        runtime_row = runtime_rows.get(lane_id)
        if runtime_row is None:
            raise SystemExit(f"Missing runtime governance row for {lane_id}")
        engineering_smoke_status = str(runtime_row.get("runtime_smoke_status") or "").strip()
        result_surface_status = str(runtime_row.get("result_surface_status") or "").strip()
        internal_experiment_id = ctx.runtime_bundle_lookup[lane_id]["internal_experiment_id"]
        inherited_runtime_blockers = json.loads(
            str(runtime_row.get("analysis_grade_blockers_json") or "[]")
        )
        required_pass_mode = _analysis_grade_lane_pass_mode(ctx, lane_id)
        policy_bundle = resolve_fuller_accuracy_policy_bundle(
            ctx,
            experiment_family_id="analysis_grade_replay",
            lane_id=lane_id,
            pass_mode=required_pass_mode,
        )
        policy_provenance = host_tuning_provenance_fields(policy_bundle)
        command_policy = command_runtime_policy_from_bundle(
            policy_bundle,
            pass_mode=required_pass_mode,
        )
        runtime_policy = (
            dict((policy_bundle.get("pass_policies") or {}).get("quantized_eval_pass") or {})
            if required_pass_mode == "paired"
            else command_policy
        )
        runtime_health_gate = runtime_health_gate_from_policy(runtime_policy or command_policy)

        if engineering_smoke_status == "complete_clean_pass" and result_surface_status == "complete":
            if lane_id == canonical_lane:
                gate_status = "ready_pending_explicit_authorization"
                gate_blockers = ["explicit_authorization_required"]
                next_action = "continue_or_authorize_canonical_astra_baseline"
            else:
                gate_status = "ready_pending_astra_baseline_reference"
                gate_blockers = ["canonical_astra_baseline_reference_required"]
                next_action = "complete_astra_then_launch_quantized_only_replay"
        else:
            gate_status = "blocked_on_engineering_smoke_prereq"
            gate_blockers = ["engineering_smoke_incomplete"]
            next_action = "repair_engineering_smoke_first"

        step_id = f"analysis_grade_replay__{lane_id}__current"
        gate_rows.append(
            {
                "lane_id": lane_id,
                "internal_experiment_id": internal_experiment_id,
                "engineering_smoke_status": engineering_smoke_status,
                "result_surface_status": result_surface_status,
                "inherited_runtime_blockers_json": inherited_runtime_blockers,
                "analysis_grade_gate_status": gate_status,
                "analysis_grade_gate_blockers_json": gate_blockers,
                "required_seeds_json": [0, 1, 2],
                "required_pass_mode": required_pass_mode,
                "required_manifest_path": str(full_manifest.resolve()),
                **policy_provenance,
                "phase4_eligible": True,
                "claim_tier": "analysis_grade",
                "materialized_step_id": step_id,
                "next_action": next_action,
            }
        )

        command = [
            str(python_bin),
            str(runner),
            "--run_ids",
            _phase1_run_id(ctx, lane_id),
            "--results_csv",
            str(_lane_results_csv(lane_id)),
            "--annotated_results_csv",
            str(_lane_annotated_csv(lane_id)),
            "--prepared_phase1_config_root",
            str(_lane_prepared_phase1_root(lane_id)),
            "--prepared_eligibility_report_root",
            str(_lane_prepared_eligibility_root(lane_id)),
            "--progress_root",
            str(_lane_progress_root(lane_id)),
            "--progress_heartbeat_interval_seconds",
            _gate_value(runtime_health_gate, "progress_heartbeat_interval_seconds", 15.0),
            "--stall_timeout_seconds",
            _gate_value(runtime_health_gate, "stall_timeout_seconds", 300.0),
            "--prelaunch_runtime_smoke_samples",
            "16",
            "--prelaunch_min_samples_per_hour",
            "30.0",
            "--prelaunch_max_seconds_per_sample",
            "120.0",
            "--pathological_min_samples_per_hour",
            _gate_value(runtime_health_gate, "pathological_min_samples_per_hour", 12.0),
            "--pathological_max_seconds_per_sample",
            _gate_value(runtime_health_gate, "pathological_max_seconds_per_sample", 600.0),
            "--pathological_max_eta_current_rate_seconds",
            _gate_value(runtime_health_gate, "pathological_max_eta_current_rate_seconds", 604800.0),
            "--pathological_min_processed_samples",
            _gate_value(runtime_health_gate, "pathological_min_processed_samples", 32),
            "--pathological_min_elapsed_seconds",
            _gate_value(runtime_health_gate, "pathological_min_elapsed_seconds", 900.0),
            "--accuracy_backend",
            "mlx",
            "--models",
            "mobilevit_s",
            "--device",
            required_device,
            "--host_tuning_profile_json",
            host_tuning_profile_json,
            "--host_id",
            str(policy_bundle.get("host_id") or ""),
            "--experiment_family_id",
            "analysis_grade_replay",
            "--lane_id",
            lane_id,
            "--workers",
            str(int(command_policy.get("workers") or 0)),
            "--eval_batch_size",
            str(int(command_policy.get("eval_batch_size") or 0)),
            "--seeds",
            "0,1,2",
            "--evidence_tier",
            "analysis_grade",
            "--manifest_override",
            str(full_manifest),
            "--pass_mode",
            required_pass_mode,
            "--annotation_measurement_truth_class",
            "bridge_only_nonbitstream_measured",
            "--bitstream_measurement_truth_class",
            "bitstream_limited_surface_pilot",
            "--annotation_contract_note",
            "analysis_grade_replay_pending_phase4_intake",
            "--bitstream_contract_note",
            "analysis_grade_replay_pending_phase4_intake",
            "--enable_bitstream_pilot",
            "--bitstream_surface_scope",
            limited_surface_scope,
            "--bitstream_target_module_keys",
            target_module_keys,
            "--resume",
        ]
        if lane_id in ANALYSIS_GRADE_MAINLINE_LANES:
            command.extend(
                [
                    "--baseline_reference_csv",
                    str(baseline_reference_csv),
                    "--baseline_reference_run_id",
                    baseline_reference_run_id,
                ]
            )

        checker_command = [
            str(python_bin),
            str(checker),
            "--mode",
            "postrun",
            "--lane_id",
            lane_id,
            "--annotated_csv",
            str(_lane_annotated_csv(lane_id)),
            "--eval_run_id_prefix",
            f"{_phase1_run_id(ctx, lane_id)}_acc_s",
        ]
        plan_rows.append(
            {
                "step_id": step_id,
                "lane_id": lane_id,
                "internal_experiment_id": internal_experiment_id,
                "package_kind": (
                    "analysis_grade_canonical_paired_baseline"
                    if lane_id == canonical_lane
                    else "analysis_grade_quantized_only_replay"
                ),
                "command_json": command,
                "checker_kind": "analysis_grade_surface_contract",
                "checker_command_json": checker_command,
                "progress_root": str(_lane_progress_root(lane_id)),
                "required_device": required_device,
                "launch_wrapper_json": launch_wrapper,
                "serial_group": "fuller_analysis_grade_replay",
                "resume_enabled": True,
                "stop_on_failure": True,
                "execution_authorized": False,
                "current_run_policy": (
                    "analysis_grade_canonical_baseline_current"
                    if lane_id == canonical_lane
                    else "analysis_grade_full_manifest_quantized_only_reuse"
                ),
                "sample_budget": _analysis_grade_lane_sample_budget(ctx, lane_id),
                "slice_manifest_path": str(full_manifest.resolve()),
                "pass_mode": required_pass_mode,
                "host_tuning_profile_json": host_tuning_profile_json,
                "host_id": str(policy_bundle.get("host_id") or ""),
                **policy_provenance,
                "baseline_source_step_id": (
                    "" if lane_id == canonical_lane else _analysis_grade_baseline_source_step_id()
                ),
                "baseline_reference_csv": (
                    "" if lane_id == canonical_lane else str(baseline_reference_csv)
                ),
                "baseline_reference_run_id": (
                    "" if lane_id == canonical_lane else baseline_reference_run_id
                ),
                "baseline_reference_summary_json": "",
                "phase4_eligible": True,
                "legacy_replacement_policy": LEGACY_REPLACEMENT_POLICY,
            }
        )

    gate_csv = _resolve_path(root_dir, outputs["analysis_grade_replay_gate_matrix_csv"])
    gate_json = _resolve_path(root_dir, outputs["analysis_grade_replay_gate_matrix_json"])
    plan_csv = _resolve_path(root_dir, outputs["analysis_grade_replay_materialization_plan_csv"])
    plan_json = _resolve_path(root_dir, outputs["analysis_grade_replay_materialization_plan_json"])
    gate_md = _resolve_path(root_dir, outputs["analysis_grade_replay_gate_md"])
    plan_md = _resolve_path(root_dir, outputs["analysis_grade_replay_materialization_md"])

    _write_csv(gate_csv, GATE_FIELDS, gate_rows)
    _write_json(
        gate_json,
        {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "rows": gate_rows,
        },
    )
    _write_csv(plan_csv, PLAN_FIELDS, plan_rows)
    _write_json(
        plan_json,
        {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "rows": plan_rows,
        },
    )
    _write_text(gate_md, _gate_note(gate_rows))
    _write_text(
        plan_md,
        _materialization_note(
            plan_rows,
            full_manifest=full_manifest,
            redirected_supports=redirected_supports,
        ),
    )
    return {
        "status": "pass",
        "gate_csv": str(gate_csv.resolve()),
        "plan_csv": str(plan_csv.resolve()),
        "lane_count": len(gate_rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Materialize the FULLER analysis-grade replay gate and plan surfaces."
    )
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    args = parser.parse_args()
    payload = build_fuller_analysis_grade_replay_surface(args.contract)
    print(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
