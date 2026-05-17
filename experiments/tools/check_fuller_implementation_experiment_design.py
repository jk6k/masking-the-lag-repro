#!/usr/bin/env python3
"""Validate the governed fuller-implementation experiment design package."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from repo_python_bootstrap import maybe_reexec_for_module
except ImportError:
    def maybe_reexec_for_module(_module: str, *, anchor: Path | None = None) -> None:
        """Allow local test execution when the repo bootstrap shim is unavailable."""
        return None

maybe_reexec_for_module("yaml", anchor=Path(__file__))

import yaml

try:
    from .path_policy import MAIN_PROJECT_REPORT_DATA_DIR, MAIN_PROJECT_REPORT_FIG_DIR, assert_main_project_path, resolve_repo_path
except ImportError:
    from path_policy import MAIN_PROJECT_REPORT_DATA_DIR, MAIN_PROJECT_REPORT_FIG_DIR, assert_main_project_path, resolve_repo_path  # type: ignore


DEFAULT_CONTRACT = ROOT / "configs" / "fuller_implementation_experiment_design_contract_20260319.yaml"
DEFAULT_OUT_DIR = MAIN_PROJECT_REPORT_DATA_DIR
NOISE_SUMMARY_REQUIRED_FIELDS = [
    "model",
    "profile",
    "sweep_resolution",
    "crosstalk_alpha",
    "gaussian_noise_std",
    "accuracy_backend",
    "engine",
    "parity_status",
    "parity_report_ref",
    "acc_top1",
    "acc_drop_pp",
    "latency_ms",
    "energy_j",
    "accuracy_results_csv",
    "accuracy_source_run_ids",
    "accuracy_seeds",
    "phase1_run_id",
    "phase1_run_dir",
    "accuracy_launch_policy",
    "phase1_launch_policy",
]

NOISE_MODEL_SUFFIXES = {
    "mobilevit_s": "s",
    "mobilevit_xs": "xs",
    "mobilevit_xxs": "xxs",
}


def _noise_model_suffix(model: str) -> str:
    suffix = NOISE_MODEL_SUFFIXES.get(str(model).strip())
    if suffix is None:
        raise SystemExit(f"Unsupported governed noise model: {model!r}")
    return suffix


def _noise_family_id(*, model: str, sweep_resolution: str) -> str:
    return f"NOISE_IMAGENET_MOBILEVIT_{_noise_model_suffix(model).upper()}_{str(sweep_resolution).upper()}"


def _noise_artifact_id(*, model: str, sweep_resolution: str) -> str:
    return f"noise_accuracy_summary_{_noise_model_suffix(model)}_{str(sweep_resolution).lower()}"


def _support_noise_specs(noise_sweep_policy: dict[str, Any]) -> tuple[str, list[tuple[str, str]]]:
    supporting_sparse_models = [str(item) for item in noise_sweep_policy.get("supporting_sparse_models") or []]
    supporting_dense_models = [str(item) for item in noise_sweep_policy.get("supporting_dense_models") or []]
    if supporting_sparse_models == ["mobilevit_xs", "mobilevit_xxs"]:
        return "sparse", [(model, "sparse") for model in supporting_sparse_models]
    if supporting_dense_models == ["mobilevit_xs", "mobilevit_xxs"]:
        return "dense", [(model, "dense") for model in supporting_dense_models]
    raise SystemExit(
        "noise_sweep_policy must declare either supporting_sparse_models "
        "or supporting_dense_models as ['mobilevit_xs', 'mobilevit_xxs']"
    )


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise SystemExit(f"Expected YAML mapping in {path}")
    return payload


def _require_mapping(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise SystemExit(f"Missing required mapping: {key}")
    return value


def _require_list(payload: dict[str, Any], key: str, *, allow_empty: bool = False) -> list[str]:
    value = payload.get(key)
    if not isinstance(value, list) or (not value and not allow_empty):
        raise SystemExit(f"Missing required list: {key}")
    return [str(item) for item in value]


def _record(rows: list[dict[str, str]], check_id: str, status: str, detail: str) -> None:
    rows.append({"check_id": check_id, "status": status, "detail": detail})


def _is_valid_figure_output_dir(path: Path) -> bool:
    if path == MAIN_PROJECT_REPORT_FIG_DIR:
        return True
    return path.name == "report_figures"


def _validate_csv(path: Path, expected_fields: list[str]) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        rows = list(reader)
    missing = [field for field in expected_fields if field not in fieldnames]
    if missing:
        raise SystemExit(f"Missing CSV columns in {path}: {missing}")
    return rows


def evaluate_contract(contract: dict[str, Any]) -> tuple[list[dict[str, str]], dict[str, Any]]:
    checks: list[dict[str, str]] = []

    meta = _require_mapping(contract, "meta")
    governed_route = _require_mapping(contract, "governed_route")
    fuller_design = _require_mapping(contract, "fuller_design")
    objectives = _require_mapping(contract, "objectives")
    comparison_policy = _require_mapping(contract, "comparison_policy")
    proxy_readiness_policy = _require_mapping(contract, "proxy_readiness_policy")
    noise_sweep_policy = _require_mapping(contract, "noise_sweep_policy")
    artifacts = _require_mapping(contract, "artifacts")
    schema = _require_mapping(contract, "schema")
    stop_rules = _require_mapping(contract, "stop_rules")

    _record(
        checks,
        "governed_route.active_safe_route",
        "pass" if str(governed_route.get("active_safe_route") or "") == "bounded local fuller reentry mainline" else "fail",
        str(governed_route.get("active_safe_route") or ""),
    )
    _record(
        checks,
        "governed_route.retained_claim_promotion_allowed",
        "pass" if governed_route.get("retained_claim_promotion_allowed") is True else "fail",
        repr(governed_route.get("retained_claim_promotion_allowed")),
    )
    live_state_note = assert_main_project_path(
        resolve_repo_path(str(governed_route.get("fuller_live_state_note_md") or "")),
        arg_name="governed_route.fuller_live_state_note_md",
    )
    _record(
        checks,
        "governed_route.fuller_live_state_note_md",
        "pass" if live_state_note.exists() else "fail",
        str(live_state_note),
    )

    _record(
        checks,
        "fuller_design.substrate_role",
        "pass" if str(fuller_design.get("substrate_role") or "") == "ASTRA-style baseline execution substrate" else "fail",
        str(fuller_design.get("substrate_role") or ""),
    )
    _record(
        checks,
        "fuller_design.primary_and_support_lanes",
        "pass"
        if str(fuller_design.get("primary_lane") or "") == "HOPS"
        and _require_list(fuller_design, "support_lanes") == ["MESO", "DET", "SPARSE", "PHY"]
        else "fail",
        repr(
            {
                "primary_lane": fuller_design.get("primary_lane"),
                "support_lanes": fuller_design.get("support_lanes"),
            }
        ),
    )
    _record(
        checks,
        "fuller_design.deferred_hooks",
        "pass" if _require_list(fuller_design, "deferred_hooks", allow_empty=True) == [] else "fail",
        repr(fuller_design.get("deferred_hooks")),
    )
    _record(
        checks,
        "fuller_design.retired_hooks",
        "pass" if _require_list(fuller_design, "retired_hooks", allow_empty=True) == [] else "fail",
        repr(fuller_design.get("retired_hooks")),
    )
    retired_hooks_note = assert_main_project_path(
        resolve_repo_path(str(fuller_design.get("retired_hooks_note_md") or "")),
        arg_name="fuller_design.retired_hooks_note_md",
    )
    _record(
        checks,
        "fuller_design.retired_hooks_note_md",
        "pass" if retired_hooks_note.exists() else "fail",
        str(retired_hooks_note),
    )
    active_template = assert_main_project_path(
        resolve_repo_path(str(fuller_design.get("active_template_yaml") or "")),
        arg_name="fuller_design.active_template_yaml",
    )
    _record(checks, "fuller_design.active_template_yaml", "pass" if active_template.exists() else "fail", str(active_template))
    template_cfg = _load_yaml(active_template)
    template_run_cfg = template_cfg.get("run") if isinstance(template_cfg.get("run"), dict) else {}
    realism_cfg = template_cfg.get("realism") if isinstance(template_cfg.get("realism"), dict) else {}
    integrated_cfg = (
        template_cfg.get("integrated_system_costs")
        if isinstance(template_cfg.get("integrated_system_costs"), dict)
        else {}
    )
    flow_cfg = template_cfg.get("flow") if isinstance(template_cfg.get("flow"), dict) else {}
    meso_cfg = template_cfg.get("meso") if isinstance(template_cfg.get("meso"), dict) else {}
    phy_cfg = template_cfg.get("phy") if isinstance(template_cfg.get("phy"), dict) else {}
    _record(
        checks,
        "fuller_design.active_experiment_id",
        "pass"
        if str(fuller_design.get("active_experiment_id") or "") == "FULLER_REENTRY_V1"
        and str(template_run_cfg.get("experiment_id") or "") == str(fuller_design.get("active_experiment_id") or "")
        else "fail",
        repr(
            {
                "contract": fuller_design.get("active_experiment_id"),
                "template": template_run_cfg.get("experiment_id"),
            }
        ),
    )
    _record(
        checks,
        "template.realism_boundary",
        "pass"
        if str(realism_cfg.get("target_class") or "") == "realistic_accelerator_proxy"
        and str(realism_cfg.get("device_comparison_scope") or "") == "contextual_only"
        and realism_cfg.get("benchmark_equivalence") is False
        else "fail",
        repr(realism_cfg),
    )
    _record(
        checks,
        "template.realism_evidence_surface",
        "pass"
        if all(
            str(value or "").strip()
            for value in (
                integrated_cfg.get("onchip_comm_evidence_type"),
                integrated_cfg.get("control_sched_evidence_type"),
                integrated_cfg.get("host_staging_evidence_type"),
                integrated_cfg.get("calibration_monitoring_evidence_type"),
                integrated_cfg.get("uncertainty_method"),
                flow_cfg.get("evidence_type"),
                meso_cfg.get("evidence_type"),
                phy_cfg.get("evidence_type"),
            )
        )
        else "fail",
        repr(
            {
                "integrated_system_costs": integrated_cfg,
                "flow": flow_cfg,
                "meso": meso_cfg,
                "phy": phy_cfg,
            }
        ),
    )

    core_objectives = set(_require_list(objectives, "core"))
    _record(
        checks,
        "objectives.core",
        "pass"
        if core_objectives
        >= {
            "device_real_machine_comparison",
            "noise_and_crosstalk_accuracy_impact",
            "scale_aware_noise_robustness_support",
            "ablation_study",
            "multi_panel_ablation_evidence",
            "latency_and_energy_breakdown_figures",
        }
        else "fail",
        repr(sorted(core_objectives)),
    )
    slide_extensions = set(_require_list(objectives, "slide_inspired_extensions"))
    _record(
        checks,
        "objectives.slide_extensions",
        "pass"
        if slide_extensions
        >= {
            "throughput_scaling_vs_batch_and_sequence",
            "workload_and_lightweight_model_diversity_after_core_closure",
        }
        else "fail",
        repr(sorted(slide_extensions)),
    )

    _record(
        checks,
        "comparison_policy.cpu_gpu_boundary",
        "pass"
        if comparison_policy.get("local_cpu_comparison_allowed") is True
        and comparison_policy.get("local_gpu_mps_comparison_allowed") is True
        and comparison_policy.get("local_cuda_allowed") is False
        and comparison_policy.get("external_cuda_host_required") is False
        else "fail",
        repr(
            {
                "local_cpu_comparison_allowed": comparison_policy.get("local_cpu_comparison_allowed"),
                "local_gpu_mps_comparison_allowed": comparison_policy.get("local_gpu_mps_comparison_allowed"),
                "local_cuda_allowed": comparison_policy.get("local_cuda_allowed"),
                "external_cuda_host_required": comparison_policy.get("external_cuda_host_required"),
            }
        ),
    )
    _record(
        checks,
        "comparison_policy.real_device_metrics",
        "pass"
        if set(_require_list(comparison_policy, "required_real_device_metrics")) >= {"latency_ms", "avg_power_w", "energy_j"}
        else "fail",
        repr(comparison_policy.get("required_real_device_metrics")),
    )
    _record(
        checks,
        "comparison_policy.host_metadata",
        "pass"
        if set(_require_list(comparison_policy, "required_host_metadata")) >= {"host_name", "device_model", "framework", "precision_mode"}
        else "fail",
        repr(comparison_policy.get("required_host_metadata")),
    )
    _record(
        checks,
        "proxy_readiness.current_target_state",
        "pass"
        if str(proxy_readiness_policy.get("current_class") or "") == "realistic_accelerator_proxy"
        and str(proxy_readiness_policy.get("target_class") or "") == "realistic_accelerator_proxy"
        and proxy_readiness_policy.get("promotion_ready") is True
        else "fail",
        repr(
            {
                "current_class": proxy_readiness_policy.get("current_class"),
                "target_class": proxy_readiness_policy.get("target_class"),
                "promotion_ready": proxy_readiness_policy.get("promotion_ready"),
            }
        ),
    )
    support_mode, support_specs = _support_noise_specs(noise_sweep_policy)
    noise_policy_ok = (
        str(noise_sweep_policy.get("primary_dense_model") or "") == "mobilevit_s"
        and int(noise_sweep_policy.get("dense_grid_min_points_per_axis") or 0) >= 5
        and int(noise_sweep_policy.get("dense_grid_preferred_points_per_axis") or 0) >= 7
    )
    if support_mode == "sparse":
        noise_policy_ok = (
            noise_policy_ok
            and int(noise_sweep_policy.get("sparse_profile_target_count") or 0) >= 4
            and str(noise_sweep_policy.get("gaussian_noise_mode") or "") == "bounded_profile_slices_not_full_3d_grid"
        )
    else:
        noise_policy_ok = (
            noise_policy_ok
            and str(noise_sweep_policy.get("gaussian_noise_mode") or "") == "full_2d_grid_for_all_governed_models"
        )
    _record(
        checks,
        "noise_sweep_policy.model_resolution_split",
        "pass" if noise_policy_ok else "fail",
        repr({"support_mode": support_mode, **noise_sweep_policy}),
    )

    design_note = assert_main_project_path(resolve_repo_path(str(artifacts.get("design_note_md") or "")), arg_name="artifacts.design_note_md")
    matrix_csv = assert_main_project_path(resolve_repo_path(str(artifacts.get("experiment_matrix_csv") or "")), arg_name="artifacts.experiment_matrix_csv")
    data_contract_csv = assert_main_project_path(resolve_repo_path(str(artifacts.get("data_contract_csv") or "")), arg_name="artifacts.data_contract_csv")
    figure_contract_csv = assert_main_project_path(resolve_repo_path(str(artifacts.get("figure_contract_csv") or "")), arg_name="artifacts.figure_contract_csv")
    for label, path in (
        ("artifacts.design_note_md", design_note),
        ("artifacts.experiment_matrix_csv", matrix_csv),
        ("artifacts.data_contract_csv", data_contract_csv),
        ("artifacts.figure_contract_csv", figure_contract_csv),
    ):
        _record(checks, label, "pass" if path.exists() else "fail", str(path))

    matrix_rows = _validate_csv(matrix_csv, _require_list(schema, "experiment_matrix_fields"))
    observed_families = {str(row.get("family_id") or "") for row in matrix_rows}
    required_core_families = set(_require_mapping(contract, "required_experiment_families")["core"])
    _record(
        checks,
        "experiment_matrix.required_core_families",
        "pass" if required_core_families.issubset(observed_families) else "fail",
        f"observed={sorted(observed_families)}",
    )

    by_family = {str(row.get("family_id") or ""): row for row in matrix_rows}
    cpu_row = by_family.get("DEVCOMP_CPU_LOCAL", {})
    gpu_row = by_family.get("DEVCOMP_GPU_LOCAL_MPS", {})
    noise_s_dense_row = by_family.get("NOISE_IMAGENET_MOBILEVIT_S_DENSE", {})
    ablation_row = by_family.get("ABLATION_CORE_STACK", {})
    breakdown_row = by_family.get("BREAKDOWN_STAGE_ENERGY", {})
    scaling_row = by_family.get("SCALING_BATCH_SEQ", {})
    _record(
        checks,
        "experiment_matrix.cpu_row",
        "pass"
        if str(cpu_row.get("execution_surface") or "") == "local_host_cpu"
        and "cpu" in str(cpu_row.get("device_policy") or "").lower()
        else "fail",
        repr(cpu_row),
    )
    _record(
        checks,
        "experiment_matrix.gpu_row",
        "pass"
        if str(gpu_row.get("execution_surface") or "") == "local_apple_gpu_mps"
        and "mps" in str(gpu_row.get("device_policy") or "")
        and "no_local_cuda" in str(gpu_row.get("device_policy") or "")
        else "fail",
        repr(gpu_row),
    )
    _record(
        checks,
        "experiment_matrix.noise_s_dense_anchor",
        "pass"
        if str(noise_s_dense_row.get("workload_scope") or "") == "W0_mobilevit_imagenet"
        and str(noise_s_dense_row.get("model_scope") or "") == "mobilevit_s"
        and "dense" in str(noise_s_dense_row.get("parameter_values") or "")
        else "fail",
        repr(noise_s_dense_row),
    )
    for model, sweep_resolution in support_specs:
        family_id = _noise_family_id(model=model, sweep_resolution=sweep_resolution)
        row = by_family.get(family_id, {})
        _record(
            checks,
            f"experiment_matrix.{_noise_model_suffix(model)}_{sweep_resolution}_anchor",
            "pass"
            if str(row.get("workload_scope") or "") == "W0_mobilevit_imagenet"
            and str(row.get("model_scope") or "") == model
            and str(sweep_resolution) in str(row.get("parameter_values") or "")
            else "fail",
            repr(row),
        )
    _record(
        checks,
        "experiment_matrix.ablation_ladder",
        "pass"
        if str(ablation_row.get("baseline_pairing") or "") == "ASTRA->HOPS->HOPS+MESO->HOPS+PHY->FULLER"
        and "ablation_appendix_table_csv" in str(ablation_row.get("required_outputs") or "")
        else "fail",
        repr(
            {
                "baseline_pairing": ablation_row.get("baseline_pairing"),
                "required_outputs": ablation_row.get("required_outputs"),
            }
        ),
    )
    _record(
        checks,
        "experiment_matrix.breakdown_outputs",
        "pass"
        if "stage_latency_breakdown_csv" in str(breakdown_row.get("required_outputs") or "")
        and "stage_energy_breakdown_csv" in str(breakdown_row.get("required_outputs") or "")
        else "fail",
        str(breakdown_row.get("required_outputs") or ""),
    )
    _record(
        checks,
        "experiment_matrix.scaling_family",
        "pass"
        if "batch_size" in str(scaling_row.get("parameter_values") or "")
        and "sequence_length" in str(scaling_row.get("parameter_values") or "")
        else "fail",
        str(scaling_row.get("parameter_values") or ""),
    )

    data_rows = _validate_csv(data_contract_csv, _require_list(schema, "data_contract_fields"))
    observed_artifacts = {str(row.get("artifact_id") or "") for row in data_rows}
    required_data_artifacts = set(_require_list(contract, "required_data_artifacts"))
    _record(
        checks,
        "data_contract.required_artifacts",
        "pass" if required_data_artifacts.issubset(observed_artifacts) else "fail",
        f"observed={sorted(observed_artifacts)}",
    )
    data_by_id = {str(row.get("artifact_id") or ""): row for row in data_rows}
    _record(
        checks,
        "data_contract.device_metric_fields",
        "pass"
        if all(
            metric in str(data_by_id[artifact_id].get("required_fields") or "")
            for artifact_id in ("cpu_real_device_metrics", "gpu_real_device_metrics")
            for metric in ("latency_ms", "avg_power_w", "energy_j")
        )
        else "fail",
        repr(
            {
                "cpu": data_by_id.get("cpu_real_device_metrics"),
                "gpu": data_by_id.get("gpu_real_device_metrics"),
            }
        ),
    )
    proxy_model_fields = _require_list(proxy_readiness_policy, "required_model_fields")
    proxy_device_fields = _require_list(proxy_readiness_policy, "required_device_fields")
    _record(
        checks,
        "data_contract.proxy_model_fields",
        "pass"
        if all(
            field in str(data_by_id["fuller_slice_model_summary"].get("required_fields") or "")
            for field in proxy_model_fields
        )
        else "fail",
        repr(data_by_id.get("fuller_slice_model_summary")),
    )
    _record(
        checks,
        "data_contract.device_provenance_fields",
        "pass"
        if all(
            field in str(data_by_id[artifact_id].get("required_fields") or "")
            for artifact_id in ("cpu_real_device_metrics", "gpu_real_device_metrics")
            for field in proxy_device_fields
        )
        else "fail",
        repr(
            {
                "cpu": data_by_id.get("cpu_real_device_metrics"),
                "gpu": data_by_id.get("gpu_real_device_metrics"),
            }
        ),
    )
    expected_noise_artifacts = [
        _noise_artifact_id(model="mobilevit_s", sweep_resolution="dense"),
        *[_noise_artifact_id(model=model, sweep_resolution=sweep_resolution) for model, sweep_resolution in support_specs],
    ]
    _record(
        checks,
        "data_contract.noise_fields",
        "pass"
        if all(
            metric in str(data_by_id[artifact_id].get("required_fields") or "")
            for artifact_id in expected_noise_artifacts
            for metric in NOISE_SUMMARY_REQUIRED_FIELDS
        )
        else "fail",
        repr({artifact_id: data_by_id.get(artifact_id) for artifact_id in expected_noise_artifacts}),
    )
    _record(
        checks,
        "data_contract.ablation_appendix_table",
        "pass"
        if all(
            metric in str(data_by_id["ablation_appendix_table"].get("required_fields") or "")
            for metric in ("delta_latency_pct", "delta_energy_pct", "delta_acc_drop_pp")
        )
        else "fail",
        repr(data_by_id.get("ablation_appendix_table")),
    )

    figure_rows = _validate_csv(figure_contract_csv, _require_list(schema, "figure_contract_fields"))
    observed_figure_ids = {str(row.get("figure_id") or "") for row in figure_rows}
    required_figure_ids = set(_require_list(contract, "required_figure_ids"))
    _record(
        checks,
        "figure_contract.required_figure_ids",
        "pass" if required_figure_ids.issubset(observed_figure_ids) else "fail",
        f"observed={sorted(observed_figure_ids)}",
    )
    figure_by_id = {str(row.get("figure_id") or ""): row for row in figure_rows}
    noise_figure = figure_by_id.get("FigFuller_NoiseAccuracy", {})
    ablation_figure = figure_by_id.get("FigFuller_Ablation", {})
    required_noise_tokens = [
        "_s_dense",
        *[
            f"_{_noise_model_suffix(model)}_{sweep_resolution}"
            for model, sweep_resolution in support_specs
        ],
    ]
    must_have_text = str(noise_figure.get("must_have") or "").lower()
    _record(
        checks,
        "figure_contract.noise_accuracy_scale_support",
        "pass"
        if all(token in str(noise_figure.get("data_or_semantic_source") or "") for token in required_noise_tokens)
        and "dense" in must_have_text
        and (support_mode != "sparse" or "sparse" in must_have_text)
        else "fail",
        repr(noise_figure),
    )
    _record(
        checks,
        "figure_contract.ablation_multi_panel",
        "pass"
        if "multi-panel" in str(ablation_figure.get("must_have") or "").lower()
        and "single metric only" in str(ablation_figure.get("must_not_have") or "").lower()
        and "ablation_appendix" in str(ablation_figure.get("data_or_semantic_source") or "")
        else "fail",
        repr(ablation_figure),
    )
    figure_dirs: set[Path] = set()
    for row in figure_rows:
        figure_id = str(row.get("figure_id") or "")
        planned_output_dir = assert_main_project_path(
            resolve_repo_path(str(row.get("planned_output_dir") or "")),
            arg_name=f"figure_contract[{figure_id}].planned_output_dir",
        )
        figure_dirs.add(planned_output_dir)
        _record(
            checks,
            f"figure_contract.{figure_id}.planned_output_dir",
            "pass" if _is_valid_figure_output_dir(planned_output_dir) else "fail",
            str(planned_output_dir),
        )
    _record(
        checks,
        "figure_contract.shared_planned_output_dir",
        "pass" if len(figure_dirs) == 1 else "fail",
        repr(sorted(str(path) for path in figure_dirs)),
    )

    forbidden_actions = set(_require_list(stop_rules, "forbidden_actions"))
    _record(
        checks,
        "stop_rules.forbidden_actions",
        "pass"
        if forbidden_actions
        >= {
            "local_cuda_probe_on_this_mac",
            "broad_cross_mechanism_matrix_launch",
            "freeze_replacement_in_this_design_thread",
            "retained_claim_promotion_from_fuller_design_only",
            "gpu_claim_without_host_metadata",
        }
        else "fail",
        repr(sorted(forbidden_actions)),
    )
    _record(
        checks,
        "stop_rules.forbidden_reopens",
        "pass" if _require_list(stop_rules, "forbidden_reopens", allow_empty=True) == [] else "fail",
        repr(stop_rules.get("forbidden_reopens")),
    )

    summary = {
        "tag": str(meta.get("tag") or "unknown"),
        "overall_ok": all(row["status"] == "pass" for row in checks),
        "experiment_family_count": len(matrix_rows),
        "data_artifact_count": len(data_rows),
        "figure_count": len(figure_rows),
    }
    return checks, summary


def _write_report(path: Path, *, contract_path: Path, summary: dict[str, Any], checks: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Fuller Implementation Experiment Design Status",
        "",
        "Scope",
        f"- Contract: `{contract_path}`",
        f"- Overall status: `{summary['overall_ok']}`",
        f"- Experiment families: `{summary['experiment_family_count']}`",
        f"- Data artifacts: `{summary['data_artifact_count']}`",
        f"- Figures: `{summary['figure_count']}`",
        "",
        "Checks",
    ]
    for row in checks:
        lines.append(f"- `{row['status']}` `{row['check_id']}`: {row['detail']}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Check the governed fuller-implementation experiment design package.")
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    contract_path = assert_main_project_path(resolve_repo_path(args.contract), arg_name="--contract")
    out_dir = assert_main_project_path(resolve_repo_path(args.out_dir), arg_name="--out_dir")
    contract = _load_yaml(contract_path)
    checks, summary = evaluate_contract(contract)
    status_path = assert_main_project_path(resolve_repo_path(_require_mapping(contract, "artifacts")["status_report_md"]), arg_name="artifacts.status_report_md")
    report_path = out_dir / status_path.name
    _write_report(report_path, contract_path=contract_path, summary=summary, checks=checks)
    overall = "OK" if summary["overall_ok"] else "FAIL"
    print(f"[fuller-implementation-design] overall={overall} report={report_path}")
    if not summary["overall_ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
