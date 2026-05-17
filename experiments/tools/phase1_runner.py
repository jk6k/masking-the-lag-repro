"""Phase-1 runner to align paper experiments with the fixed E0–E6 schema."""

from __future__ import annotations

import argparse
from collections import deque
import csv
import hashlib
import json
import math
import os
import random
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

ROOT_DIR = Path(__file__).resolve().parents[1]


def _detect_git_hash() -> str:
    """Auto-detect git commit hash; return 'nogit' on failure."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
            cwd=str(ROOT_DIR),
        )
        if result.returncode == 0:
            return result.stdout.strip() or "nogit"
    except Exception:
        pass
    return "nogit"


if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from exp_common.phy_link_budget import compute_link_budget  # noqa: E402
from exp_common.det_prefix import compute_prefix_error_stats  # noqa: E402
from exp_common.det_policy import (  # noqa: E402
    build_det_policy_signature,
    resolve_det_policy_label,
    resolve_det_runtime_metadata,
)
from exp_common.io_utils import resolve_workspace_path  # noqa: E402
from exp_common.meso_cost_model import (  # noqa: E402
    compute_meso_cost_model,
    resolve_meso_load_scale,
)
from exp_common.realism_proxy_calibration import apply_realism_proxy_profile  # noqa: E402
from hpat_model import ELEMENTWISE_TYPES, summarize_ops  # noqa: E402
from accuracy.bitstream_conv_semantics import (  # noqa: E402
    CONV_FIDELITY_STAGE_MEASURED_CLOSED,
    CONV_FIDELITY_STAGE_RUNTIME_MODELED,
    CONV_FOCUSED_MEASURED_PACKAGE_STATUS_AUTHORIZED,
    CONV_HARDWARE_EVIDENCE_UNCLOSED,
    CONV_MEASURED_CLOSURE_STATUS_MEASURED_CLOSED,
    CONV_MEASURED_CLOSURE_STATUS_RUNTIME_MODELED,
    CONV_MEASURED_CLOSURE_STATUS_UNCLOSED,
    LEGACY_ALL_TARGET_MEASURED_ANCHOR_RUN_ID,
    resolve_conv_evidence_manifest,
    resolve_conv_focused_measured_package,
    validate_conv_focused_measured_package,
)
from accuracy.bitstream_semantics import (  # noqa: E402
    BITSTREAM_MODEL_LEVEL_MEASURED_TRUTH_CLASS,
)
from accuracy.bitstream_truth_authorization import (  # noqa: E402
    BITSTREAM_MODEL_LEVEL_MEASURED_AUTHORIZED_STATUS,
    assess_truth_class_authorization,
    resolve_truth_class_authorization_note,
)

RUNTIME_SMOKE_TIER = "runtime_smoke"
ANALYSIS_GRADE_TIER = "analysis_grade"
from accuracy.mlx_mobilevit import (  # noqa: E402
    BITSTREAM_RUNTIME_FAMILY_POLICY_SOURCE,
    DEFAULT_BITSTREAM_RUNTIME_STREAM_REUSE_POLICY,
)
from sc_bitstream.generators import resolve_generator_default_policy  # noqa: E402
from tools.fuller_phase1_registry import (  # noqa: E402
    default_fuller_phase1_variants,
    default_variant_descriptor_for_experiment,
)


EXPERIMENT_SWITCH_MATRIX: dict[str, dict[str, bool]] = {
    str(item["internal_experiment_id"]).upper(): dict(item["switches"])
    for item in default_fuller_phase1_variants()
}

FLOW_SCHEDULER_MODE_V1 = "explicit_scheduler_v1"
FLOW_SCHEDULER_MODE_V2 = "explicit_scheduler_v2"
FLOW_SCHEDULER_MODE_V3 = "elastic_residency_v3"
FLOW_REUSE_POLICIES = {"none", "operand_pair", "operand_factored"}

PAPER_MODULES = [
    "Workload Mapper",
    "DET-Aware BtoS Frontend",
    "MESO Broadcast Fabric",
    "HOPS Scheduler",
    "Photonic Compute Cluster",
    "O/E + ADC/PCA Backend",
    "PHY Closure Manager",
]

GEMM_COMPONENT_FIELDS = [
    "energy_mj_load_x",
    "energy_mj_load_y",
    "energy_mj_oe",
    "energy_mj_adc_pca",
    "energy_mj_detect",
    "energy_mj_laser",
    "energy_mj_mem",
    "energy_mj_static",
]

OPS_FIELDS = [
    "name",
    "type",
    "estimator_mode",
    "true_sc_claim_state",
    "true_sc_claim_surface_role",
    "true_sc_claim_surface_reason",
    "model_abstraction_kind",
    "model_abstraction_status",
    "model_abstraction_reason",
    "model_abstraction_boundary_json",
    "conv_lowering_kernel",
    "conv_lowering_stride",
    "conv_lowering_groups",
    "conv_native_class",
    "conv_fidelity_stage",
    "conv_fidelity_blockers",
    "conv_runtime_semantics_json",
    "m",
    "d",
    "n",
    "elements",
    "tiles",
    "latency_ms",
    "energy_mj",
    "energy_mj_load_x",
    "energy_mj_load_y",
    "energy_mj_detect",
    "energy_mj_oe",
    "energy_mj_adc_pca",
    "energy_mj_laser",
    "energy_mj_mem",
    "energy_mj_static",
    "energy_mj_elementwise",
    "energy_mj_bitstream_accumulator",
    "bitstream_stream_length",
    "bitstream_effective_stream_length",
    "bitstream_effective_stream_length_scale",
    "bitstream_effective_stream_length_scale_provenance",
    "bitstream_parallel_passes",
    "bitstream_parallel_outputs",
    "bitstream_parallel_outputs_provenance",
    "bitstream_cycles_per_stream_bit",
    "bitstream_cycles_per_stream_bit_provenance",
    "bitstream_accumulator_energy_pj",
    "bitstream_accumulator_energy_pj_provenance",
    "bitstream_elementwise_parallelism_factor",
    "bitstream_elementwise_parallelism_provenance",
    "bitstream_generator",
    "generator_stream_state_policy_json",
    "bitstream_capture_manifest_csv",
    "bitstream_calibration_applied",
    "bitstream_calibration_summary_json",
    "bitstream_calibration_reason",
    "bitstream_datapath_stages_json",
    "power_w",
]

PHASE1_SUMMARY_FIELDS = [
    "run_id",
    "variant_id",
    "experiment_id",
    "internal_experiment_id",
    "public_module_stack",
    "active_switch_set",
    "workload_id",
    "model",
    "ops_path",
    "input_size",
    "batch_size",
    "execution_semantics",
    "execution_semantics_default",
    "execution_semantics_origin",
    "bitstream_enabled",
    "bitstream_encoding_mode",
    "bitstream_multiplier_mode",
    "bitstream_stream_length",
    "bitstream_generator",
    "generator_stream_state_policy",
    "runtime_stream_reuse_policy",
    "bitstream_accumulator_mode",
    "bitstream_calibration_source",
    "bitstream_capture_manifest_csv",
    "bitstream_effective_stream_length",
    "bitstream_effective_stream_length_scale",
    "bitstream_effective_stream_length_scale_provenance",
    "bitstream_parallel_outputs",
    "bitstream_parallel_outputs_provenance",
    "bitstream_cycles_per_stream_bit",
    "bitstream_cycles_per_stream_bit_provenance",
    "bitstream_accumulator_energy_pj",
    "bitstream_accumulator_energy_pj_provenance",
    "bitstream_elementwise_parallelism_factor",
    "bitstream_elementwise_parallelism_provenance",
    "bitstream_calibration_applied",
    "bitstream_calibration_summary_json",
    "bitstream_calibration_reason",
    "bitstream_calibration_median_relative_error",
    "bitstream_calibration_capture_row_count",
    "bitstream_calibration_replay_row_count",
    "bitstream_datapath_stage_summary",
    "model_abstraction_boundary_kind",
    "model_abstraction_boundary_status",
    "model_abstraction_boundary_reason",
    "model_abstraction_boundary_json",
    "conv2d_gemm_lowered_approximation_op_count",
    "conv2d_native_runtime_modeled_op_count",
    "conv_fidelity_stage",
    "conv_fidelity_blockers",
    "conv_evidence_manifest_path",
    "conv_evidence_manifest_sha256",
    "conv_measured_package_path",
    "conv_measured_package_sha256",
    "conv_measured_closure_status",
    "workload_fidelity_class",
    "workload_fidelity_status",
    "workload_fidelity_reason",
    "workload_fidelity_blockers",
    "fidelity_ready",
    "fidelity_blockers",
    "sc_summary_trust_posture",
    "true_sc_summary_claim_state",
    "true_sc_claim_state_inventory",
    "true_sc_claim_surface_status",
    "true_sc_claim_surface_inventory",
    "true_sc_claim_surface_native_op_count",
    "true_sc_claim_surface_governed_support_op_count",
    "true_sc_support_out_of_surface_op_count",
    "true_sc_out_of_claim_surface_op_count",
    "true_sc_native_op_count",
    "true_sc_governed_not_true_sc_op_count",
    "true_sc_out_of_surface_op_count",
    "sc_calibration_state",
    "sc_generator_policy_status",
    "sc_generator_policy_reason",
    "sc_support_classes",
    "sc_native_op_count",
    "sc_governed_support_op_count",
    "sc_unsupported_op_count",
    "estimation_model_coverage_status",
    "estimation_model_coverage_reason",
    "estimation_model_support_boundary",
    "estimation_model_supported_op_count",
    "estimation_model_unsupported_op_count",
    "estimation_model_ready_status",
    "estimation_model_ready",
    "estimation_model_ready_reason",
    "estimation_model_ready_blockers",
    "sc_fail_mode",
    "sc_fail_closed_triggered",
    "sc_default_status",
    "dark_launch_enabled",
    "dark_launch_candidate_label",
    "dark_launch_comparator_label",
    "dark_launch_comparator_execution_semantics",
    "dark_launch_comparator_trust_posture",
    "core_latency_ms",
    "latency_ms",
    "system_latency_lower_ms",
    "system_latency_upper_ms",
    "core_energy_j",
    "energy_j",
    "system_energy_lower_j",
    "system_energy_upper_j",
    "avg_power_w",
    "core_avg_power_w",
    "tops_w",
    "energy_upperbound_j",
    "energy_countbased_j",
    "energy_breakdown_conversion_control_j",
    "energy_breakdown_memory_move_j",
    "energy_breakdown_oe_j",
    "energy_breakdown_adc_pca_j",
    "energy_breakdown_laser_optical_j",
    "energy_breakdown_other_static_j",
    "integrated_onchip_comm_j",
    "integrated_control_sched_j",
    "integrated_host_staging_j",
    "integrated_calibration_monitoring_j",
    "integrated_hidden_system_cost_j",
    "integrated_hidden_system_cost_lower_j",
    "integrated_hidden_system_cost_upper_j",
    "integrated_onchip_comm_ms",
    "integrated_control_sched_ms",
    "integrated_host_staging_ms",
    "integrated_calibration_monitoring_ms",
    "integrated_hidden_system_latency_ms",
    "integrated_hidden_system_latency_lower_ms",
    "integrated_hidden_system_latency_upper_ms",
    "integrated_system_cost_mode",
    "integrated_onchip_comm_evidence",
    "integrated_control_sched_evidence",
    "integrated_host_staging_evidence",
    "integrated_calibration_monitoring_evidence",
    "integrated_system_cost_evidence",
    "integrated_system_cost_calibration_source",
    "integrated_system_cost_uncertainty_method",
    "flow_timeline_evidence",
    "flow_timeline_calibration_source",
    "meso_cost_evidence",
    "meso_cost_calibration_source",
    "phy_support_evidence",
    "phy_support_calibration_source",
    "accuracy_coupling_evidence",
    "accuracy_coupling_metric",
    "accuracy_coupling_source",
    "accuracy_coupling_reason",
    "accuracy_measurement_contract_status",
    "accuracy_measurement_contract_reason",
    "accuracy_measurement_contract_source",
    "accuracy_measurement_contract_truth_class",
    "accuracy_measurement_contract_authorization_note",
    "accuracy_measurement_contract_authorization_status",
    "accuracy_measurement_contract_conv_measured_package_path",
    "accuracy_measurement_contract_conv_measured_package_sha256",
    "accuracy_measurement_contract_required_truth_class",
    "accuracy_measurement_contract_required_fields_json",
    "accuracy_measurement_contract_observed_fields_json",
    "accuracy_measurement_contract_violations_json",
    "accuracy_evidence_tier",
    "analysis_grade_ready",
    "analysis_grade_blockers",
    "realism_class",
    "proxy_promotion_ready",
    "proxy_upgrade_blockers",
    "benchmark_claim_ready",
    "device_comparison_scope",
    "benchmark_equivalence",
    "comparison_boundary",
    "latency_boundary",
    "energy_boundary",
    "power_boundary",
    "retained_energy_mode",
    "acc_ref_top1",
    "acc_top1",
    "acc_drop_pp",
    "accuracy_source_csv",
    "accuracy_baseline_row_id",
    "accuracy_target_row_id",
    "accuracy_baseline_source_run_id",
    "accuracy_target_source_run_id",
    "accuracy_target_split",
    "accuracy_target_notes",
    "pass_delta",
    "det_policy",
    "det_k_signature",
    "det_runtime_enabled",
    "det_quality_gate_enabled",
    "det_quality_gate_policy",
    "det_quality_gate_status",
    "det_quality_gate_reason",
    "det_quality_gate_fallback_policy",
    "det_quality_gate_require_measured_accuracy",
    "det_quality_gate_measured_accuracy_ready",
    "det_quality_gate_max_prefix_error_mean",
    "det_quality_gate_max_prefix_error_p95",
    "det_prefix_error_mean",
    "det_prefix_error_p95",
    "avg_effective_bsl",
    "duty_cycle_avg",
    "sparse_active_fraction",
    "sparse_scale_source",
    "sparse_measured_activity_fraction",
    "det_saved_j",
    "det_overhead_j",
    "det_net_gain_j",
    "pass_det_net_gain",
    "stage_cycles",
    "bubble_cycles",
    "utilization_avg",
    "flow_model_mode",
    "flow_buffer_depth",
    "flow_overlap_efficiency",
    "flow_staging_cost_scale",
    "flow_sync_penalty_scale",
    "flow_reuse_policy",
    "flow_prefetch_window",
    "flow_control_group_size",
    "flow_effective_overlap",
    "flow_reuse_gain",
    "flow_control_relief",
    "flow_buffer_peak_cycles",
    "flow_buffer_peak_frac",
    "flow_tile_rows",
    "flow_tile_cols",
    "flow_prefetch_credits",
    "flow_execute_credits",
    "flow_control_issue_width",
    "flow_admission_policy",
    "flow_eviction_policy",
    "flow_service_policy",
    "flow_reuse_residency_budget",
    "flow_broadcast_stability_window",
    "flow_prefetch_distance",
    "flow_exception_lane_policy",
    "flow_admission_stalls",
    "flow_prefetch_hits",
    "flow_prefetch_drops",
    "flow_residency_hit_rate",
    "flow_control_backpressure",
    "flow_eviction_count",
    "hops_scheduler_mode",
    "fanout",
    "topology_dimension",
    "serializers_saved",
    "serializer_energy_j",
    "broadcast_driver_energy_j",
    "fabric_control_overhead_j",
    "extra_buffering_overhead_j",
    "explicit_total_cost_j",
    "explicit_total_savings_j",
    "meso_cost_model_mode",
    "net_energy_gain_j",
    "N_wdm",
    "PP_crosstalk_db",
    "P_laser_dbm",
    "P_laser_mw",
    "phy_link_budget_status",
    "gaussian_noise_std_ref",
    "crosstalk_alpha_ref",
    "p1_align_method",
    "p1_align_fit_error",
    "phy_penalty_table_version",
]

MASTER_FIELDS = [
    "run_id",
    "variant_id",
    "experiment_id",
    "internal_experiment_id",
    "public_module_stack",
    "workload_id",
    "model",
    "split",
    "seed",
    "git_hash",
    "date",
    "meso",
    "flow",
    "det",
    "sparse",
    "phy",
    "det_mode",
    "sparse_mode",
    "phy_mode",
    "execution_semantics",
    "execution_semantics_default",
    "execution_semantics_origin",
    "bitstream_enabled",
    "bitstream_encoding_mode",
    "bitstream_multiplier_mode",
    "bitstream_stream_length",
    "bitstream_generator",
    "generator_stream_state_policy",
    "runtime_stream_reuse_policy",
    "bitstream_accumulator_mode",
    "bitstream_calibration_source",
    "bitstream_capture_manifest_csv",
    "bitstream_effective_stream_length",
    "bitstream_effective_stream_length_scale",
    "bitstream_effective_stream_length_scale_provenance",
    "bitstream_parallel_outputs",
    "bitstream_parallel_outputs_provenance",
    "bitstream_cycles_per_stream_bit",
    "bitstream_cycles_per_stream_bit_provenance",
    "bitstream_accumulator_energy_pj",
    "bitstream_accumulator_energy_pj_provenance",
    "bitstream_elementwise_parallelism_factor",
    "bitstream_elementwise_parallelism_provenance",
    "bitstream_calibration_applied",
    "bitstream_calibration_summary_json",
    "bitstream_calibration_reason",
    "bitstream_calibration_median_relative_error",
    "bitstream_calibration_capture_row_count",
    "bitstream_calibration_replay_row_count",
    "bitstream_datapath_stage_summary",
    "model_abstraction_boundary_kind",
    "model_abstraction_boundary_status",
    "model_abstraction_boundary_reason",
    "model_abstraction_boundary_json",
    "conv2d_gemm_lowered_approximation_op_count",
    "conv2d_native_runtime_modeled_op_count",
    "conv_fidelity_stage",
    "conv_fidelity_blockers",
    "conv_evidence_manifest_path",
    "conv_evidence_manifest_sha256",
    "conv_measured_package_path",
    "conv_measured_package_sha256",
    "conv_measured_closure_status",
    "workload_fidelity_class",
    "workload_fidelity_status",
    "workload_fidelity_reason",
    "workload_fidelity_blockers",
    "fidelity_ready",
    "fidelity_blockers",
    "sc_summary_trust_posture",
    "true_sc_summary_claim_state",
    "true_sc_claim_state_inventory",
    "true_sc_claim_surface_status",
    "true_sc_claim_surface_inventory",
    "true_sc_claim_surface_native_op_count",
    "true_sc_claim_surface_governed_support_op_count",
    "true_sc_support_out_of_surface_op_count",
    "true_sc_out_of_claim_surface_op_count",
    "true_sc_native_op_count",
    "true_sc_governed_not_true_sc_op_count",
    "true_sc_out_of_surface_op_count",
    "sc_calibration_state",
    "sc_generator_policy_status",
    "sc_generator_policy_reason",
    "sc_support_classes",
    "sc_native_op_count",
    "sc_governed_support_op_count",
    "sc_unsupported_op_count",
    "estimation_model_coverage_status",
    "estimation_model_coverage_reason",
    "estimation_model_support_boundary",
    "estimation_model_supported_op_count",
    "estimation_model_unsupported_op_count",
    "estimation_model_ready_status",
    "estimation_model_ready",
    "estimation_model_ready_reason",
    "estimation_model_ready_blockers",
    "sc_fail_mode",
    "sc_fail_closed_triggered",
    "sc_default_status",
    "dark_launch_enabled",
    "dark_launch_candidate_label",
    "dark_launch_comparator_label",
    "dark_launch_comparator_execution_semantics",
    "dark_launch_comparator_trust_posture",
    "acc_ref_top1",
    "acc_top1",
    "acc_drop_pp",
    "accuracy_source_csv",
    "accuracy_baseline_row_id",
    "accuracy_target_row_id",
    "accuracy_baseline_source_run_id",
    "accuracy_target_source_run_id",
    "accuracy_target_split",
    "accuracy_target_notes",
    "delta_pp_budget",
    "pass_delta",
    "core_latency_ms",
    "latency_ms",
    "system_latency_lower_ms",
    "system_latency_upper_ms",
    "throughput_images_s",
    "throughput_tokens_s",
    "speedup_vs_E0",
    "core_energy_j",
    "energy_j",
    "system_energy_lower_j",
    "system_energy_upper_j",
    "avg_power_w",
    "core_avg_power_w",
    "tops_w",
    "energy_breakdown_conversion_control_j",
    "energy_breakdown_memory_move_j",
    "energy_breakdown_oe_j",
    "energy_breakdown_adc_pca_j",
    "energy_breakdown_laser_optical_j",
    "energy_breakdown_other_static_j",
    "integrated_onchip_comm_j",
    "integrated_control_sched_j",
    "integrated_host_staging_j",
    "integrated_calibration_monitoring_j",
    "integrated_hidden_system_cost_j",
    "integrated_hidden_system_cost_lower_j",
    "integrated_hidden_system_cost_upper_j",
    "integrated_onchip_comm_ms",
    "integrated_control_sched_ms",
    "integrated_host_staging_ms",
    "integrated_calibration_monitoring_ms",
    "integrated_hidden_system_latency_ms",
    "integrated_hidden_system_latency_lower_ms",
    "integrated_hidden_system_latency_upper_ms",
    "integrated_system_cost_mode",
    "integrated_onchip_comm_evidence",
    "integrated_control_sched_evidence",
    "integrated_host_staging_evidence",
    "integrated_calibration_monitoring_evidence",
    "integrated_system_cost_evidence",
    "integrated_system_cost_calibration_source",
    "integrated_system_cost_uncertainty_method",
    "flow_timeline_evidence",
    "flow_timeline_calibration_source",
    "meso_cost_evidence",
    "meso_cost_calibration_source",
    "phy_support_evidence",
    "phy_support_calibration_source",
    "accuracy_coupling_evidence",
    "accuracy_coupling_metric",
    "accuracy_coupling_source",
    "accuracy_coupling_reason",
    "accuracy_measurement_contract_status",
    "accuracy_measurement_contract_reason",
    "accuracy_measurement_contract_source",
    "accuracy_measurement_contract_truth_class",
    "accuracy_measurement_contract_authorization_note",
    "accuracy_measurement_contract_authorization_status",
    "accuracy_measurement_contract_conv_measured_package_path",
    "accuracy_measurement_contract_conv_measured_package_sha256",
    "accuracy_measurement_contract_required_truth_class",
    "accuracy_measurement_contract_required_fields_json",
    "accuracy_measurement_contract_observed_fields_json",
    "accuracy_measurement_contract_violations_json",
    "accuracy_evidence_tier",
    "analysis_grade_ready",
    "analysis_grade_blockers",
    "realism_class",
    "proxy_promotion_ready",
    "proxy_upgrade_blockers",
    "benchmark_claim_ready",
    "device_comparison_scope",
    "benchmark_equivalence",
    "comparison_boundary",
    "latency_boundary",
    "energy_boundary",
    "power_boundary",
    "retained_energy_mode",
    "energy_upperbound_j",
    "energy_countbased_j",
    "bsl_max",
    "det_policy",
    "det_k_signature",
    "det_runtime_enabled",
    "det_quality_gate_enabled",
    "det_quality_gate_policy",
    "det_quality_gate_status",
    "det_quality_gate_reason",
    "det_quality_gate_fallback_policy",
    "det_quality_gate_require_measured_accuracy",
    "det_quality_gate_measured_accuracy_ready",
    "det_quality_gate_max_prefix_error_mean",
    "det_quality_gate_max_prefix_error_p95",
    "det_prefix_error_mean",
    "det_prefix_error_p95",
    "avg_effective_bsl",
    "k_i",
    "tau_i",
    "duty_cycle_avg",
    "sparse_scale_source",
    "sparse_measured_activity_fraction",
    "det_overhead_j",
    "det_saved_j",
    "det_net_gain_j",
    "pass_det_net_gain",
    "stage_cycles",
    "bubble_cycles",
    "utilization_avg",
    "flow_model_mode",
    "flow_buffer_depth",
    "flow_overlap_efficiency",
    "flow_staging_cost_scale",
    "flow_sync_penalty_scale",
    "flow_reuse_policy",
    "flow_prefetch_window",
    "flow_control_group_size",
    "flow_effective_overlap",
    "flow_reuse_gain",
    "flow_control_relief",
    "flow_buffer_peak_cycles",
    "flow_buffer_peak_frac",
    "flow_tile_rows",
    "flow_tile_cols",
    "flow_prefetch_credits",
    "flow_execute_credits",
    "flow_control_issue_width",
    "flow_admission_policy",
    "flow_eviction_policy",
    "flow_service_policy",
    "flow_reuse_residency_budget",
    "flow_broadcast_stability_window",
    "flow_prefetch_distance",
    "flow_exception_lane_policy",
    "flow_admission_stalls",
    "flow_prefetch_hits",
    "flow_prefetch_drops",
    "flow_residency_hit_rate",
    "flow_control_backpressure",
    "flow_eviction_count",
    "fanout",
    "topology_dimension",
    "serializers_saved",
    "serializer_energy_j",
    "broadcast_driver_energy_j",
    "fabric_control_overhead_j",
    "extra_buffering_overhead_j",
    "explicit_total_cost_j",
    "explicit_total_savings_j",
    "meso_cost_model_mode",
    "net_energy_gain_j",
    "N_wdm",
    "ER_db",
    "BER_target",
    "Loss_path_db",
    "PP_crosstalk_db",
    "P_laser_dbm",
    "P_laser_mw",
    "gaussian_noise_std_ref",
    "crosstalk_alpha_ref",
    "p1_align_method",
    "p1_align_fit_error",
    "phy_penalty_table_version",
    "s_wg_min",
    "P_thermal_tuning",
]

TIMELINE_SUMMARY_FIELDS = [
    "model",
    "execution_semantics",
    "bitstream_enabled",
    "bitstream_capture_manifest_csv",
    "sc_summary_trust_posture",
    "true_sc_summary_claim_state",
    "estimation_model_ready_status",
    "estimation_model_ready",
    "core_latency_ms",
    "latency_ms",
    "integrated_hidden_system_latency_ms",
    "stage_cycles",
    "bubble_cycles",
    "utilization_avg",
    "flow_buffer_peak_cycles",
    "flow_buffer_peak_frac",
    "flow_admission_stalls",
    "flow_prefetch_hits",
    "flow_prefetch_drops",
    "flow_residency_hit_rate",
    "flow_control_backpressure",
    "flow_eviction_count",
]

DARK_LAUNCH_FIELDS = [
    "model",
    "candidate_label",
    "candidate_execution_semantics",
    "candidate_execution_semantics_origin",
    "candidate_summary_trust_posture",
    "candidate_true_sc_summary_claim_state",
    "candidate_default_status",
    "candidate_latency_ms",
    "candidate_energy_j",
    "candidate_support_classes",
    "candidate_native_op_count",
    "candidate_governed_support_op_count",
    "candidate_unsupported_op_count",
    "comparator_label",
    "comparator_execution_semantics",
    "comparator_summary_trust_posture",
    "comparator_true_sc_summary_claim_state",
    "comparator_latency_ms",
    "comparator_energy_j",
    "coverage_delta_rows",
    "latency_delta_ms",
    "energy_delta_j",
    "compatibility_status",
]

DEFAULT_INTEGRATED_SYSTEM_COSTS = {
    "mode": "integrated_minimal_v1",
    "onchip_scale_vs_move": 0.10,
    "control_sched_scale_vs_conv": 0.10,
    "host_staging_scale_vs_conv": 0.05,
    "calibration_monitoring_scale_vs_thermal": 0.50,
    "onchip_latency_scale_vs_io": 0.10,
    "control_sched_latency_scale_vs_control": 0.10,
    "host_staging_latency_scale_vs_io": 0.05,
    "calibration_monitoring_latency_scale_vs_core": 0.02,
    "uncertainty_lower_scale": 0.75,
    "uncertainty_upper_scale": 1.25,
    "host_staging_flow_only": True,
    "calibration_monitoring_phy_or_noise_only": True,
    "onchip_comm_evidence_type": "heuristic_proxy",
    "control_sched_evidence_type": "heuristic_proxy",
    "host_staging_evidence_type": "heuristic_proxy",
    "calibration_monitoring_evidence_type": "heuristic_proxy",
    "calibration_source": None,
    "uncertainty_method": "scale_band",
}


def _resolve_execution_semantics_cfg(
    cfg: dict[str, Any],
    *,
    override_semantics: str | None = None,
) -> dict[str, Any]:
    """Resolve execution semantics for the active estimator lane."""
    raw_cfg = cfg.get("bitstream") or {}
    if raw_cfg is None:
        raw_cfg = {}
    if not isinstance(raw_cfg, dict):
        raise SystemExit("Expected optional 'bitstream' config to be a mapping.")

    requested_semantics = str(
        override_semantics if override_semantics is not None else raw_cfg.get("execution_semantics") or ""
    ).strip().lower()
    default_semantics = str(raw_cfg.get("default_execution_semantics") or "proxy").strip().lower()
    if default_semantics not in {"proxy", "bitstream"}:
        raise SystemExit(f"Unsupported default execution semantics: {default_semantics!r}")
    raw_bitstream_enabled = _to_bool(raw_cfg.get("enabled"), False)
    if requested_semantics not in {"", "proxy", "bitstream"}:
        raise SystemExit(f"Unsupported execution semantics: {requested_semantics!r}")
    capture_manifest_csv = (
        str(raw_cfg.get("capture_manifest_csv")).strip()
        if raw_cfg.get("capture_manifest_csv") not in {None, ""}
        else None
    )
    if requested_semantics:
        resolved_semantics = requested_semantics
        origin = "cli_override" if override_semantics is not None else "config_explicit"
    elif raw_bitstream_enabled:
        resolved_semantics = "bitstream"
        origin = "legacy_enabled"
    else:
        resolved_semantics = default_semantics
        origin = "default_policy"

    if resolved_semantics == "proxy":
        return {
            "execution_semantics": "proxy",
            "execution_semantics_default": default_semantics,
            "execution_semantics_origin": origin,
            "bitstream_enabled": False,
            "bitstream_encoding_mode": None,
            "bitstream_multiplier_mode": None,
            "bitstream_stream_length": None,
            "bitstream_generator": None,
            "bitstream_accumulator_mode": None,
            "bitstream_calibration_source": None,
            "bitstream_capture_manifest_csv": capture_manifest_csv,
        }
    stream_length = _to_int(raw_cfg.get("stream_length"), None)
    if stream_length is None or stream_length <= 0:
        raise SystemExit(
            "Bitstream execution semantics require a positive bitstream.stream_length."
        )
    return {
        "execution_semantics": "bitstream",
        "execution_semantics_default": default_semantics,
        "execution_semantics_origin": origin,
        "bitstream_enabled": True,
        "bitstream_encoding_mode": str(raw_cfg.get("encoding_mode") or "bipolar").strip().lower(),
        "bitstream_multiplier_mode": str(raw_cfg.get("multiplier_mode") or "xnor").strip().lower(),
        "bitstream_stream_length": stream_length,
        "bitstream_generator": str(raw_cfg.get("generator") or "bernoulli").strip().lower(),
        "bitstream_accumulator_mode": str(
            raw_cfg.get("accumulator_mode") or "bitcount"
        ).strip().lower(),
        "bitstream_calibration_source": (
            str(raw_cfg.get("calibration_source")).strip()
            if raw_cfg.get("calibration_source") not in {None, ""}
            else None
        ),
        "bitstream_capture_manifest_csv": capture_manifest_csv,
    }


def _resolve_dark_launch_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    raw_cfg = cfg.get("dark_launch") or {}
    if raw_cfg is None:
        raw_cfg = {}
    if not isinstance(raw_cfg, dict):
        raise SystemExit("Expected optional 'dark_launch' config to be a mapping.")

    enabled = _to_bool(raw_cfg.get("enabled"), False)
    comparator_semantics = str(
        raw_cfg.get("comparator_execution_semantics") or "proxy"
    ).strip().lower() or "proxy"
    if comparator_semantics not in {"proxy", "bitstream"}:
        raise SystemExit(
            f"Unsupported dark-launch comparator semantics: {comparator_semantics!r}"
        )
    return {
        "enabled": enabled,
        "candidate_label": str(raw_cfg.get("candidate_label") or "sc_default_candidate").strip()
        or "sc_default_candidate",
        "comparator_label": str(raw_cfg.get("comparator_label") or "historical_proxy").strip()
        or "historical_proxy",
        "comparator_execution_semantics": comparator_semantics,
    }


def _resolve_sc_trust_contract_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    bitstream_cfg = cfg.get("bitstream") or {}
    trust_cfg = (bitstream_cfg.get("trust_contract") or {}) if isinstance(bitstream_cfg, dict) else {}
    if trust_cfg is None:
        trust_cfg = {}
    if not isinstance(trust_cfg, dict):
        raise SystemExit("Expected optional 'bitstream.trust_contract' config to be a mapping.")
    fail_mode = str(trust_cfg.get("fail_mode") or "fail_closed_to_comparator").strip().lower()
    if fail_mode not in {"fail_closed_to_comparator", "fail_open_emit_default"}:
        raise SystemExit(f"Unsupported SC trust-contract fail_mode: {fail_mode!r}")
    return {"fail_mode": fail_mode}


def _load_generator_policy_rows(path: Path) -> list[dict[str, Any]]:
    try:
        with path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            return list(reader)
    except OSError:
        return []


def _resolve_generator_policy_for_model(
    cfg: dict[str, Any],
    execution_semantics_cfg: dict[str, Any],
    *,
    model: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    resolved_cfg = dict(execution_semantics_cfg)
    policy_meta = {
        "sc_generator_policy_status": "",
        "sc_generator_policy_reason": "",
        "sc_generator_policy_source": "",
    }
    if resolved_cfg.get("execution_semantics") != "bitstream":
        return resolved_cfg, policy_meta

    bitstream_cfg = cfg.get("bitstream") or {}
    if not isinstance(bitstream_cfg, dict):
        return resolved_cfg, policy_meta
    matrix_csv = bitstream_cfg.get("generator_policy_matrix_csv")
    if matrix_csv in {None, ""}:
        return resolved_cfg, policy_meta

    matrix_path = _resolve_existing_path(str(matrix_csv))
    rows = _load_generator_policy_rows(matrix_path)
    if not rows:
        policy_meta["sc_generator_policy_status"] = "out_of_band"
        policy_meta["sc_generator_policy_reason"] = "generator_policy_matrix_missing_or_empty"
        policy_meta["sc_generator_policy_source"] = str(matrix_path)
        return resolved_cfg, policy_meta

    policy = resolve_generator_default_policy(
        rows,
        workload_class=model,
        stream_length=resolved_cfg.get("bitstream_stream_length"),
    )
    policy_meta["sc_generator_policy_status"] = str(policy.get("policy_state") or "")
    policy_meta["sc_generator_policy_reason"] = (
        "machine_readable_policy_selected_generator"
        if policy.get("repository_default_generator")
        else "machine_readable_policy_requires_explicit_comparator_or_override"
    )
    policy_meta["sc_generator_policy_source"] = str(matrix_path)
    selected_generator = str(policy.get("repository_default_generator") or "").strip().lower()
    if selected_generator:
        resolved_cfg["bitstream_generator"] = selected_generator
    return resolved_cfg, policy_meta


def _merge_execution_semantics_into_estimator_config(
    estimator_config: dict[str, Any],
    execution_semantics_cfg: dict[str, Any],
) -> dict[str, Any]:
    """Propagate the runner's execution-semantics selection into estimator config."""
    merged = dict(estimator_config)
    merged["execution_semantics"] = execution_semantics_cfg["execution_semantics"]

    bitstream_cfg = dict(merged.get("bitstream") or {})
    bitstream_cfg["enabled"] = execution_semantics_cfg["bitstream_enabled"]
    bitstream_cfg["execution_semantics"] = execution_semantics_cfg["execution_semantics"]

    field_map = {
        "bitstream_encoding_mode": "encoding_mode",
        "bitstream_multiplier_mode": "multiplier_mode",
        "bitstream_stream_length": "stream_length",
        "bitstream_generator": "generator",
        "bitstream_accumulator_mode": "accumulator_mode",
        "bitstream_calibration_source": "calibration_source",
        "bitstream_capture_manifest_csv": "capture_manifest_csv",
    }
    for source_key, target_key in field_map.items():
        value = execution_semantics_cfg.get(source_key)
        if value is None:
            continue
        bitstream_cfg[target_key] = value

    merged["bitstream"] = bitstream_cfg
    return merged

FLOW_BUFFER_TRACE_FIELDS = [
    "layer_id",
    "upstream_stage",
    "downstream_stage",
    "upstream_cycles",
    "downstream_cycles",
    "buffer_depth",
    "effective_buffer_depth",
    "buffer_capacity_cycles",
    "occupancy_cycles",
    "occupancy_frac",
    "scheduler_mode",
    "reuse_policy",
    "prefetch_window",
    "control_group_size",
    "admission_stalls",
    "prefetch_hits",
    "prefetch_drops",
    "residency_hit_rate",
    "control_backpressure",
    "eviction_count",
]

P0_P1_ALIGNMENT_FIELDS = [
    "delta_p_db",
    "p_laser_dbm_eff",
    "gaussian_noise_std_pred",
    "crosstalk_alpha_pred",
]


def _to_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _to_float(value: Any, default: float | None = None) -> float | None:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


_EVIDENCE_RANK = {
    "missing": 0,
    "optimistic_estimate": 1,
    "heuristic_proxy": 2,
    "calibrated_model": 3,
    "measured": 4,
}


def _canonical_evidence_type(value: Any, *, default: str = "missing") -> str:
    raw = str(value or "").strip().lower()
    if not raw:
        return default
    aliases = {
        "config_fixed_heuristic": "heuristic_proxy",
        "parametric_envelope": "heuristic_proxy",
        "literature_calibrated": "calibrated_model",
        "device_calibrated": "calibrated_model",
        "modeled_timeline": "heuristic_proxy",
        "retained_model_calibrated": "calibrated_model",
        "bounded_envelope_calibrated": "calibrated_model",
        "contextual_device_calibrated": "calibrated_model",
    }
    canonical = aliases.get(raw, raw)
    if canonical not in _EVIDENCE_RANK:
        return default
    return canonical


def _worst_evidence_type(values: list[str], *, default: str = "missing") -> str:
    if not values:
        return default
    return min(
        (_canonical_evidence_type(value, default=default) for value in values),
        key=lambda item: _EVIDENCE_RANK.get(item, -1),
    )


def _evidence_is_at_least(value: str, minimum: str) -> bool:
    return _EVIDENCE_RANK.get(_canonical_evidence_type(value), -1) >= _EVIDENCE_RANK.get(
        _canonical_evidence_type(minimum),
        -1,
    )


def _to_int(value: Any, default: int | None = None) -> int | None:
    if value is None:
        return default
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _cfg_value(mapping: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key not in mapping:
            continue
        value = mapping.get(key)
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        return value
    return None


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _resolve_sparse_tau_reference(sparse_cfg: dict[str, Any]) -> float | None:
    tau_default = _to_float(sparse_cfg.get("tau_global"), None)
    tau_by_layer = sparse_cfg.get("tau_by_layer")
    tau_candidates: list[float] = []
    if tau_default is not None and tau_default > 0.0:
        tau_candidates.append(float(tau_default))
    if isinstance(tau_by_layer, dict):
        for value in tau_by_layer.values():
            tau_value = _to_float(value, None)
            if tau_value is not None and tau_value > 0.0:
                tau_candidates.append(float(tau_value))
    if not tau_candidates:
        return None
    return sum(tau_candidates) / len(tau_candidates)


def _normalize_sparse_tau_curve(curve_payload: Any) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    if isinstance(curve_payload, dict):
        items = curve_payload.items()
    elif isinstance(curve_payload, list):
        items = enumerate(curve_payload)
    else:
        return points

    for key, value in items:
        tau_value: float | None = None
        active_value: float | None = None
        if isinstance(value, dict):
            tau_value = _to_float(_cfg_value(value, "tau", "tau_global", "x"), None)
            active_value = _to_float(
                _cfg_value(
                    value,
                    "active_fraction",
                    "duty_cycle",
                    "activity_fraction",
                    "y",
                ),
                None,
            )
        elif isinstance(value, (list, tuple)) and len(value) >= 2:
            tau_value = _to_float(value[0], None)
            active_value = _to_float(value[1], None)
        elif not isinstance(key, int):
            tau_value = _to_float(key, None)
            active_value = _to_float(value, None)
        if tau_value is None or active_value is None:
            continue
        points.append((float(tau_value), _clamp(float(active_value), 0.0, 1.0)))
    points.sort(key=lambda item: item[0])
    return points


def _estimate_sparse_scale_from_tau(sparse_cfg: dict[str, Any]) -> float | None:
    tau_reference = _resolve_sparse_tau_reference(sparse_cfg)
    tau_requested = _to_bool(sparse_cfg.get("use_tau_for_gating"), False)
    if not tau_requested and tau_reference is None:
        return None
    if tau_reference is None or tau_reference <= 0.0:
        return None

    curve_points = _normalize_sparse_tau_curve(sparse_cfg.get("tau_to_active_curve"))
    if curve_points:
        if tau_reference <= curve_points[0][0]:
            return curve_points[0][1]
        if tau_reference >= curve_points[-1][0]:
            return curve_points[-1][1]
        for (left_tau, left_active), (right_tau, right_active) in zip(
            curve_points,
            curve_points[1:],
        ):
            if left_tau <= tau_reference <= right_tau:
                if math.isclose(left_tau, right_tau):
                    return left_active
                ratio = (tau_reference - left_tau) / (right_tau - left_tau)
                return left_active + ratio * (right_active - left_active)

    min_active_fraction = _to_float(sparse_cfg.get("min_active_fraction"), 0.0) or 0.0
    return _clamp(1.0 - tau_reference, min_active_fraction, 1.0)


def _safe_ratio(numer: float, denom: float) -> float | None:
    if denom == 0:
        return None
    return numer / denom


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _dump_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=True)


def _dump_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _apply_realism_profile_if_present(
    cfg: dict[str, Any],
    *,
    cfg_path: Path,
) -> dict[str, Any]:
    realism_cfg = cfg.get("realism") or {}
    profile_value = realism_cfg.get("calibration_profile_yaml") or realism_cfg.get(
        "calibration_profile"
    )
    if not profile_value:
        return cfg
    profile_path = _resolve_existing_path(str(profile_value))
    profile = _load_yaml(profile_path)
    merged = apply_realism_proxy_profile(cfg, profile)
    merged_realism = dict(merged.get("realism") or {})
    merged_realism["resolved_calibration_profile_yaml"] = str(profile_path)
    merged["realism"] = merged_realism
    return merged


def _write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fields})


def _append_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if not exists:
            writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fields})


def _read_csv_header(path: Path) -> list[str]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        return next(reader, [])


def _load_rows_with_schema_recovery(
    path: Path, target_fields: list[str]
) -> tuple[list[dict[str, Any]], list[str]]:
    """Load rows and recover tail columns when old headers are a strict prefix.

    Legacy `master_metrics.csv` files may have fewer header columns than current
    `MASTER_FIELDS` while newer rows were appended using the newer schema.
    `csv.DictReader` puts such overflow values in row[None]. If old header is
    a prefix of the target schema, map these overflow values to missing tail
    fields in order so historical rows are salvageable.
    """
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        source_fields = list(reader.fieldnames or [])
        prefix_compatible = source_fields == target_fields[: len(source_fields)]
        tail_fields = target_fields[len(source_fields) :] if prefix_compatible else []

        recovered_rows: list[dict[str, Any]] = []
        for row in reader:
            recovered = {k: row.get(k) for k in target_fields}
            extras = row.get(None)
            if isinstance(extras, list) and extras and tail_fields:
                for field, value in zip(tail_fields, extras):
                    if str(recovered.get(field) or "").strip() == "":
                        recovered[field] = value
            recovered_rows.append(recovered)
    return recovered_rows, source_fields


def _append_csv_schema_aware(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    """Append rows while guarding against schema drift in existing CSV files."""
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)

    if not path.exists():
        _append_csv(path, fields, rows)
        return

    header = _read_csv_header(path)
    if header == fields:
        _append_csv(path, fields, rows)
        return

    recovered_rows, source_fields = _load_rows_with_schema_recovery(path, fields)
    recovered_rows.extend(rows)
    _write_csv(path, fields, recovered_rows)
    print(
        "[phase1-runner] Rewrote global master with canonical schema "
        f"({len(source_fields)} -> {len(fields)} columns): {path}"
    )


def _load_latency_baseline(path: Path | None) -> dict[str, float]:
    if path is None or not path.exists():
        return {}
    out: dict[str, float] = {}
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            model = str(row.get("model") or "").strip()
            if not model:
                continue
            latency = (
                _to_float(row.get("latency_ms"))
                or _to_float(row.get("total_latency_ms"))
                or _to_float(row.get("latency"))
            )
            if latency and latency > 0:
                out[model] = latency
    return out


def _load_accuracy_rows(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _resolve_runtime_manifest_metadata(
    *,
    model_key: str | None = None,
    ops_path: str | Path | None = None,
) -> dict[str, Any]:
    manifest_path = (
        Path(ops_path).expanduser()
        if ops_path not in {None, ""}
        else (ROOT_DIR / "mtl_model" / "ops" / f"ops_{model_key}.json")
    )
    try:
        raw_text = manifest_path.read_text(encoding="utf-8")
        payload = json.loads(raw_text)
    except (OSError, json.JSONDecodeError, TypeError):
        return {
            "manifest_path": str(manifest_path),
            "manifest_sha256": "",
            "required_families": [],
        }
    ops = payload.get("ops")
    if not isinstance(ops, list):
        return {
            "manifest_path": str(manifest_path.resolve()),
            "manifest_sha256": hashlib.sha256(raw_text.encode("utf-8")).hexdigest(),
            "required_families": [],
        }
    required_families = sorted(
        {
            str(row.get("type") or "").strip()
            for row in ops
            if isinstance(row, dict) and str(row.get("type") or "").strip()
        }
    )
    return {
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": hashlib.sha256(raw_text.encode("utf-8")).hexdigest(),
        "required_families": required_families,
    }


def _match_model(row: dict[str, Any], model: str) -> bool:
    return str(row.get("model") or "").strip().lower() == model.strip().lower()


def _normalize_accuracy_split(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in {"eval", "holdout", "calib"}:
        return text
    return ""


def _workload_base(value: Any) -> str:
    text = str(value or "").strip().lower()
    for suffix in ("_eval", "_holdout", "_calib"):
        if text.endswith(suffix):
            return text[: -len(suffix)]
    return text


def _infer_accuracy_row_split(row: dict[str, Any]) -> str:
    split = _normalize_accuracy_split(row.get("split"))
    if split:
        return split
    workload = str(row.get("workload") or row.get("workload_id") or "").strip().lower()
    for suffix, label in (("_eval", "eval"), ("_holdout", "holdout"), ("_calib", "calib")):
        if workload.endswith(suffix):
            return label
    return ""


def _normalize_accuracy_source_run_id(row: dict[str, Any]) -> str:
    source_run_id = str(row.get("source_run_id") or "").strip()
    if source_run_id:
        return source_run_id
    run_id = str(row.get("run_id") or "").strip()
    return re.sub(r"_acc_s\d+$", "", run_id)


def _parse_analysis_grade_blockers(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value or "").strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except (TypeError, ValueError, json.JSONDecodeError):
        parsed = None
    if isinstance(parsed, list):
        return [str(item).strip() for item in parsed if str(item).strip()]
    return [item.strip() for item in re.split(r"[;,]", text) if item.strip()]


def _resolve_accuracy_evidence_tier(
    *,
    accuracy_provenance: dict[str, Any],
    accuracy_measurement_contract_metadata: dict[str, Any] | None,
) -> str:
    explicit_tier = str(
        accuracy_provenance.get("accuracy_target_accuracy_evidence_tier") or ""
    ).strip()
    if explicit_tier in {RUNTIME_SMOKE_TIER, ANALYSIS_GRADE_TIER}:
        return explicit_tier
    target_semantics = str(
        accuracy_provenance.get("accuracy_target_execution_semantics") or ""
    ).strip()
    contract_truth_class = str(
        (accuracy_measurement_contract_metadata or {}).get(
            "accuracy_measurement_contract_truth_class"
        )
        or ""
    ).strip()
    if target_semantics == "bitstream" or contract_truth_class:
        return RUNTIME_SMOKE_TIER
    return ""


def _normalize_path_for_compare(value: Any) -> str:
    return str(value or "").strip().replace("\\", "/").lower()


def _config_snapshot_matches(row_value: Any, target_run_id: Any) -> bool:
    row_path = _normalize_path_for_compare(row_value)
    run_id = str(target_run_id or "").strip()
    if not row_path or not run_id:
        return False
    expected_suffix = f"/experiments/results/runs/{run_id}/config_snapshot.yaml".lower()
    return row_path.endswith(expected_suffix)


def _score_context_match(
    row_value: Any,
    target_value: Any,
    *,
    tol: float = 1e-9,
) -> int:
    if target_value is None or str(target_value).strip() == "":
        return 0
    if row_value is None or str(row_value).strip() == "":
        return 0
    row_num = _to_float(row_value, None)
    target_num = _to_float(target_value, None)
    if row_num is not None and target_num is not None:
        return 1 if abs(row_num - target_num) <= tol else -1
    return 1 if str(row_value).strip().lower() == str(target_value).strip().lower() else -1


def _score_accuracy_candidate(
    row: dict[str, Any],
    accuracy_context: dict[str, Any] | None,
) -> tuple[int, int, int, int, int, int, int, int, int, int, int, float]:
    if not accuracy_context:
        return (
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            _to_float(row.get("top1"), -1e9) or -1e9,
        )

    row_workload = str(row.get("workload") or row.get("workload_id") or "").strip()
    row_split = _infer_accuracy_row_split(row)
    ctx_run_id = accuracy_context.get("run_id")
    ctx_split = _normalize_accuracy_split(accuracy_context.get("split"))
    ctx_workload = str(accuracy_context.get("workload") or "").strip()
    ctx_workload_base = _workload_base(ctx_workload)

    workload_exact = _score_context_match(row_workload, ctx_workload)
    workload_base = _score_context_match(_workload_base(row_workload), ctx_workload_base)

    return (
        _score_context_match(_normalize_accuracy_source_run_id(row), ctx_run_id),
        _score_context_match(row_split, ctx_split),
        workload_exact,
        workload_base,
        _score_context_match(row.get("experiment_id"), accuracy_context.get("experiment_id")),
        _score_context_match(row.get("seed"), accuracy_context.get("seed")),
        _score_context_match(row.get("det_policy"), accuracy_context.get("det_policy")),
        _score_context_match(row.get("det_k_signature"), accuracy_context.get("det_k_signature")),
        _score_context_match(row.get("det_k_global"), accuracy_context.get("det_k_global")),
        _score_context_match(row.get("sparse_tau_global"), accuracy_context.get("sparse_tau_global")),
        _score_context_match(row.get("sparse_active_fraction"), accuracy_context.get("sparse_active_fraction")),
        _to_float(row.get("top1"), -1e9) or -1e9,
    )


def _accuracy_context_mismatches(
    row: dict[str, Any] | None,
    accuracy_context: dict[str, Any] | None,
    *,
    expected_baseline: bool | None,
) -> list[str]:
    if row is None:
        return ["missing_row"]
    if not accuracy_context:
        return []

    mismatches: list[str] = []
    if expected_baseline is not None and _to_bool(row.get("baseline"), False) != expected_baseline:
        mismatches.append("baseline")

    row_source_run_id = _normalize_accuracy_source_run_id(row)
    ctx_run_id = str(accuracy_context.get("run_id") or "").strip()
    if ctx_run_id and row_source_run_id.lower() != ctx_run_id.lower():
        mismatches.append("run_id")

    row_split = _infer_accuracy_row_split(row)
    ctx_split = _normalize_accuracy_split(accuracy_context.get("split"))
    if ctx_split and row_split != ctx_split:
        mismatches.append("split")

    row_workload = str(row.get("workload") or row.get("workload_id") or "").strip()
    ctx_workload = str(accuracy_context.get("workload") or "").strip()
    if ctx_workload and _workload_base(row_workload) != _workload_base(ctx_workload):
        mismatches.append("workload")

    for key in (
        "experiment_id",
        "seed",
        "det_policy",
        "det_k_signature",
        "det_k_global",
        "sparse_tau_global",
        "sparse_active_fraction",
        "gaussian_noise_std_ref",
        "crosstalk_alpha_ref",
    ):
        target_value = accuracy_context.get(key)
        if target_value is None or str(target_value).strip() == "":
            continue
        if _score_context_match(row.get(key), target_value) != 1:
            mismatches.append(key)

    if ctx_run_id and not _config_snapshot_matches(row.get("config_snapshot"), ctx_run_id):
        mismatches.append("config_snapshot")
    return mismatches


def _require_accuracy_context_match(
    row: dict[str, Any] | None,
    accuracy_context: dict[str, Any] | None,
    *,
    expected_baseline: bool | None,
    row_label: str,
    model: str,
) -> None:
    mismatches = _accuracy_context_mismatches(
        row,
        accuracy_context,
        expected_baseline=expected_baseline,
    )
    if mismatches:
        raise ValueError(
            f"Accuracy {row_label} row for model={model} does not match required context: "
            f"{', '.join(mismatches)}. row={row}"
        )


def _pick_accuracy_row(
    *,
    rows: list[dict[str, Any]],
    model: str,
    accuracy_context: dict[str, Any] | None = None,
    baseline: bool | None = None,
    quant_bits: float | None = None,
    crosstalk_alpha: float | None = None,
    gaussian_noise_std: float | None = None,
    tol: float = 1e-9,
) -> dict[str, Any] | None:
    candidates: list[dict[str, Any]] = []
    for row in rows:
        if not _match_model(row, model):
            continue
        if baseline is not None:
            row_baseline = _to_bool(row.get("baseline"), False)
            # Backward-compatible inference for accuracy CSVs without an explicit
            # "baseline" column (our eval script writes "notes" instead).
            if str(row.get("baseline") or "").strip() == "":
                note = str(row.get("notes") or "").strip().lower()
                if note == "baseline_fp32":
                    row_baseline = True
                elif note in {"baseline_quant", "hpat_sim"}:
                    row_baseline = False
            if row_baseline != baseline:
                continue
        if quant_bits is not None:
            bits = _to_float(row.get("quant_bits"), None)
            if bits is None or abs(bits - quant_bits) > tol:
                continue
        if crosstalk_alpha is not None:
            alpha = _to_float(row.get("crosstalk_alpha"), None)
            if alpha is None or abs(alpha - crosstalk_alpha) > tol:
                continue
        if gaussian_noise_std is not None:
            gaussian = _to_float(row.get("gaussian_noise_std"), None)
            if gaussian is None or abs(gaussian - gaussian_noise_std) > tol:
                continue
        candidates.append(row)
    if not candidates:
        return None
    return max(candidates, key=lambda row: _score_accuracy_candidate(row, accuracy_context))


def _resolve_accuracy_for_model(
    *,
    model: str,
    accuracy_cfg: dict[str, Any],
    noise_cfg: dict[str, Any],
    sc_det_cfg: dict[str, Any],
    accuracy_rows: list[dict[str, Any]],
    accuracy_context: dict[str, Any] | None = None,
    return_provenance: bool = False,
) -> tuple[float | None, float | None, float | None] | tuple[
    float | None,
    float | None,
    float | None,
    dict[str, Any],
]:
    acc_ref_top1 = _to_float(accuracy_cfg.get("acc_ref_top1"), None)
    acc_top1 = _to_float(accuracy_cfg.get("acc_top1"), None)
    acc_drop_pp = _to_float(accuracy_cfg.get("acc_drop_pp"), None)
    require_context_match = _to_bool(accuracy_cfg.get("require_context_match"), False)
    baseline_row: dict[str, Any] | None = None
    target_row: dict[str, Any] | None = None

    if accuracy_rows and (acc_ref_top1 is None or acc_top1 is None or acc_drop_pp is None):
        baseline_row = _pick_accuracy_row(
            rows=accuracy_rows,
            model=model,
            accuracy_context=accuracy_context,
            baseline=True,
        )
        if baseline_row is None:
            baseline_row = _pick_accuracy_row(
                rows=accuracy_rows,
                model=model,
                accuracy_context=accuracy_context,
                quant_bits=_to_float(sc_det_cfg.get("quant_bits"), None),
                gaussian_noise_std=0.0,
                crosstalk_alpha=0.0,
            )
        if acc_ref_top1 is None and baseline_row is not None:
            acc_ref_top1 = _to_float(baseline_row.get("top1"), None)

        if _to_bool(noise_cfg.get("enabled"), False):
            target_sigma = _to_float(
                _cfg_value(noise_cfg, "gaussian_noise_sigma_lsb", "sigma_lsb"),
                0.0,
            ) or 0.0
            target_alpha = _to_float(noise_cfg.get("crosstalk_alpha"), 0.0) or 0.0
            target_gaussian = _to_float(noise_cfg.get("gaussian_noise_std"), 0.0) or 0.0
            target_bits = _to_float(
                noise_cfg.get("quant_bits", sc_det_cfg.get("quant_bits")),
                None,
            )
        else:
            target_alpha = 0.0
            target_gaussian = None
            target_bits = _to_float(sc_det_cfg.get("quant_bits"), None)

        target_row = _pick_accuracy_row(
            rows=accuracy_rows,
            model=model,
            accuracy_context=accuracy_context,
            baseline=False,
            quant_bits=target_bits,
            crosstalk_alpha=target_alpha,
            gaussian_noise_std=target_gaussian,
        )
        if target_row is None:
            target_row = _pick_accuracy_row(
                rows=accuracy_rows,
                model=model,
                accuracy_context=accuracy_context,
                quant_bits=target_bits,
                crosstalk_alpha=target_alpha,
                gaussian_noise_std=target_gaussian,
            )
        if target_row is None and _to_bool(noise_cfg.get("enabled"), False):
            raise ValueError(
                "Missing accuracy row for "
                f"model={model} quant_bits={target_bits} "
                f"crosstalk_alpha={target_alpha} gaussian_noise_std={target_gaussian}."
            )
        if target_row is None and baseline_row is not None:
            target_row = baseline_row

        if require_context_match:
            _require_accuracy_context_match(
                baseline_row,
                accuracy_context,
                expected_baseline=True,
                row_label="baseline",
                model=model,
            )
            _require_accuracy_context_match(
                target_row,
                accuracy_context,
                expected_baseline=False,
                row_label="target",
                model=model,
            )

        if acc_top1 is None and target_row is not None:
            acc_top1 = _to_float(target_row.get("top1"), None)

        if acc_drop_pp is None and acc_ref_top1 is not None and acc_top1 is not None:
            acc_drop_pp = acc_ref_top1 - acc_top1
        elif acc_drop_pp is None and target_row is not None:
            top1_delta = _to_float(target_row.get("top1_delta"), None)
            if top1_delta is not None:
                acc_drop_pp = -top1_delta

    if acc_drop_pp is None and acc_ref_top1 is not None and acc_top1 is not None:
        acc_drop_pp = acc_ref_top1 - acc_top1
    elif acc_drop_pp is None:
        # Fallback path: infer from top1_delta only when ref/top1 are unavailable.
        target_row = _pick_accuracy_row(
            rows=accuracy_rows,
            model=model,
            accuracy_context=accuracy_context,
        )
        if target_row is not None:
            top1_delta = _to_float(target_row.get("top1_delta"), None)
            if top1_delta is not None:
                acc_drop_pp = -top1_delta

    if not return_provenance:
        return acc_ref_top1, acc_top1, acc_drop_pp

    source_csv = accuracy_cfg.get("source_csv") or accuracy_cfg.get("csv")
    resolved_source_csv = (
        str(resolve_workspace_path(source_csv).resolve())
        if source_csv
        else None
    )
    provenance = {
        "accuracy_source_csv": resolved_source_csv,
        "accuracy_baseline_row_id": baseline_row.get("run_id") if baseline_row else None,
        "accuracy_target_row_id": target_row.get("run_id") if target_row else None,
        "accuracy_baseline_source_run_id": baseline_row.get("source_run_id") if baseline_row else None,
        "accuracy_target_source_run_id": target_row.get("source_run_id") if target_row else None,
        "accuracy_target_execution_semantics": (
            target_row.get("execution_semantics") if target_row else None
        ),
        "accuracy_target_bitstream_generator": (
            target_row.get("bitstream_generator") if target_row else None
        ),
        "accuracy_target_bitstream_stream_length": (
            target_row.get("bitstream_stream_length") if target_row else None
        ),
        "accuracy_target_bitstream_runtime_stream_reuse_policy": (
            target_row.get("bitstream_runtime_stream_reuse_policy") if target_row else None
        ),
        "accuracy_target_bitstream_measurement_truth_class": (
            target_row.get("bitstream_measurement_truth_class") if target_row else None
        ),
        "accuracy_target_bitstream_runtime_claim_surface_status": (
            target_row.get("bitstream_runtime_claim_surface_status") if target_row else None
        ),
        "accuracy_target_bitstream_runtime_required_operator_families_json": (
            target_row.get("bitstream_runtime_required_operator_families_json")
            if target_row
            else None
        ),
        "accuracy_target_bitstream_runtime_covered_operator_families_json": (
            target_row.get("bitstream_runtime_covered_operator_families_json")
            if target_row
            else None
        ),
        "accuracy_target_bitstream_runtime_supported_operator_families_json": (
            target_row.get("bitstream_runtime_supported_operator_families_json")
            if target_row
            else None
        ),
        "accuracy_target_bitstream_runtime_missing_operator_families_json": (
            target_row.get("bitstream_runtime_missing_operator_families_json")
            if target_row
            else None
        ),
        "accuracy_target_bitstream_runtime_family_policy_source": (
            target_row.get("bitstream_runtime_family_policy_source") if target_row else None
        ),
        "accuracy_target_bitstream_runtime_manifest_sha256": (
            target_row.get("bitstream_runtime_manifest_sha256") if target_row else None
        ),
        "accuracy_target_bitstream_conv_measured_package_path": (
            target_row.get("bitstream_conv_measured_package_path") if target_row else None
        ),
        "accuracy_target_bitstream_conv_measured_package_sha256": (
            target_row.get("bitstream_conv_measured_package_sha256") if target_row else None
        ),
        "accuracy_target_bitstream_truth_class_authorization_note": (
            target_row.get("bitstream_truth_class_authorization_note") if target_row else None
        ),
        "accuracy_target_bitstream_truth_class_authorization_status": (
            target_row.get("bitstream_truth_class_authorization_status") if target_row else None
        ),
        "accuracy_target_accuracy_evidence_tier": (
            target_row.get("accuracy_evidence_tier") if target_row else None
        ),
        "accuracy_target_analysis_grade_ready": (
            target_row.get("analysis_grade_ready") if target_row else None
        ),
        "accuracy_target_analysis_grade_blockers": (
            target_row.get("analysis_grade_blockers") if target_row else None
        ),
        "accuracy_target_split": (
            target_row.get("split")
            if target_row and str(target_row.get("split") or "").strip()
            else (accuracy_context or {}).get("split")
        ),
        "accuracy_target_notes": target_row.get("notes") if target_row else None,
        "sparse_measured_activity_fraction": (
            _to_float(target_row.get("sparse_measured_activity_fraction"), None)
            if target_row
            else None
        ),
        "sparse_measured_zero_fraction": (
            _to_float(target_row.get("sparse_measured_zero_fraction"), None)
            if target_row
            else None
        ),
    }
    return acc_ref_top1, acc_top1, acc_drop_pp, provenance


def _resolve_accuracy_measurement_contract_metadata(
    *,
    accuracy_cfg: dict[str, Any],
    accuracy_provenance: dict[str, Any],
    active_execution_semantics: str = "proxy",
    active_bitstream_generator: str | None = None,
    active_bitstream_stream_length: int | None = None,
    active_runtime_stream_reuse_policy: str | None = None,
    active_model_key: str | None = None,
    active_ops_path: str | Path | None = None,
) -> dict[str, Any]:
    """Resolve whether measured accuracy evidence satisfies the bitstream contract."""
    coupling_cfg = accuracy_cfg.get("coupling") or accuracy_cfg.get("accuracy_coupling") or {}
    if not isinstance(coupling_cfg, dict):
        coupling_cfg = {}
    contract_cfg = (
        accuracy_cfg.get("measurement_contract")
        or accuracy_cfg.get("accuracy_measurement_contract")
        or {}
    )
    if not isinstance(contract_cfg, dict):
        contract_cfg = {}

    explicit_evidence = (
        coupling_cfg.get("evidence_type")
        or accuracy_cfg.get("coupling_evidence_type")
        or accuracy_cfg.get("accuracy_coupling_evidence_type")
    )
    explicit_evidence = (
        _canonical_evidence_type(explicit_evidence, default="missing")
        if explicit_evidence is not None
        else ""
    )

    def _has_observed_value(value: Any) -> bool:
        return value not in {None, ""}

    def _resolve_observed_value(
        *,
        accuracy_key: str,
        coupling_key: str,
        provenance_key: str,
    ) -> Any:
        provenance_value = accuracy_provenance.get(provenance_key)
        if _has_observed_value(provenance_value):
            return provenance_value
        # Measurement-contract config describes the expected evidence contract.
        # It must not be allowed to overwrite what the measured CSV row actually
        # declares; only explicit measured coupling metadata can stand in when no
        # annotated measured row is available.
        if (
            explicit_evidence == "measured"
            and accuracy_cfg.get(accuracy_key) not in {None, ""}
        ):
            return accuracy_cfg.get(accuracy_key)
        if explicit_evidence == "measured":
            if coupling_cfg.get(coupling_key) not in {None, ""}:
                return coupling_cfg.get(coupling_key)
            if accuracy_cfg.get(coupling_key) not in {None, ""}:
                return accuracy_cfg.get(coupling_key)
        return None

    def _normalize_json_list(value: Any) -> list[str] | None:
        if value in {None, ""}:
            return None
        if isinstance(value, list):
            return sorted(str(item) for item in value if str(item).strip())
        try:
            decoded = json.loads(value)
        except (TypeError, ValueError, json.JSONDecodeError):
            return None
        if not isinstance(decoded, list):
            return None
        return sorted(str(item) for item in decoded if str(item).strip())

    contract_source = str(
        contract_cfg.get("source")
        or accuracy_cfg.get("measurement_contract_source")
        or (
            coupling_cfg.get("source")
            or accuracy_cfg.get("coupling_source")
            if explicit_evidence == "measured"
            else ""
        )
        or accuracy_provenance.get("accuracy_source_csv")
        or ""
    ).strip()
    declared_semantics = str(
        _resolve_observed_value(
            accuracy_key="measurement_contract_execution_semantics",
            coupling_key="execution_semantics",
            provenance_key="accuracy_target_execution_semantics",
        )
        or ""
    ).strip().lower()
    declared_generator = str(
        _resolve_observed_value(
            accuracy_key="measurement_contract_bitstream_generator",
            coupling_key="bitstream_generator",
            provenance_key="accuracy_target_bitstream_generator",
        )
        or ""
    ).strip().lower()
    declared_stream_length = _to_float(
        _resolve_observed_value(
            accuracy_key="measurement_contract_bitstream_stream_length",
            coupling_key="bitstream_stream_length",
            provenance_key="accuracy_target_bitstream_stream_length",
        ),
        None,
    )
    declared_truth_class = str(
        _resolve_observed_value(
            accuracy_key="measurement_contract_bitstream_measurement_truth_class",
            coupling_key="bitstream_measurement_truth_class",
            provenance_key="accuracy_target_bitstream_measurement_truth_class",
        )
        or ""
    ).strip()
    declared_runtime_stream_reuse_policy = str(
        _resolve_observed_value(
            accuracy_key="measurement_contract_bitstream_runtime_stream_reuse_policy",
            coupling_key="bitstream_runtime_stream_reuse_policy",
            provenance_key="accuracy_target_bitstream_runtime_stream_reuse_policy",
        )
        or ""
    ).strip()
    declared_runtime_claim_surface_status = str(
        _resolve_observed_value(
            accuracy_key="measurement_contract_bitstream_runtime_claim_surface_status",
            coupling_key="bitstream_runtime_claim_surface_status",
            provenance_key="accuracy_target_bitstream_runtime_claim_surface_status",
        )
        or ""
    ).strip()
    declared_runtime_required_families = _normalize_json_list(
        _resolve_observed_value(
            accuracy_key="measurement_contract_bitstream_runtime_required_operator_families_json",
            coupling_key="bitstream_runtime_required_operator_families_json",
            provenance_key="accuracy_target_bitstream_runtime_required_operator_families_json",
        )
    )
    declared_runtime_covered_families = _normalize_json_list(
        _resolve_observed_value(
            accuracy_key="measurement_contract_bitstream_runtime_covered_operator_families_json",
            coupling_key="bitstream_runtime_covered_operator_families_json",
            provenance_key="accuracy_target_bitstream_runtime_covered_operator_families_json",
        )
        or _resolve_observed_value(
            accuracy_key="measurement_contract_bitstream_runtime_supported_operator_families_json",
            coupling_key="bitstream_runtime_supported_operator_families_json",
            provenance_key="accuracy_target_bitstream_runtime_supported_operator_families_json",
        )
    )
    declared_runtime_missing_families = _normalize_json_list(
        _resolve_observed_value(
            accuracy_key="measurement_contract_bitstream_runtime_missing_operator_families_json",
            coupling_key="bitstream_runtime_missing_operator_families_json",
            provenance_key="accuracy_target_bitstream_runtime_missing_operator_families_json",
        )
    )
    declared_runtime_family_policy_source = str(
        _resolve_observed_value(
            accuracy_key="measurement_contract_bitstream_runtime_family_policy_source",
            coupling_key="bitstream_runtime_family_policy_source",
            provenance_key="accuracy_target_bitstream_runtime_family_policy_source",
        )
        or ""
    ).strip()
    declared_runtime_manifest_sha256 = str(
        _resolve_observed_value(
            accuracy_key="measurement_contract_bitstream_runtime_manifest_sha256",
            coupling_key="bitstream_runtime_manifest_sha256",
            provenance_key="accuracy_target_bitstream_runtime_manifest_sha256",
        )
        or ""
    ).strip()
    declared_conv_evidence_manifest_path = str(
        _resolve_observed_value(
            accuracy_key="measurement_contract_bitstream_conv_evidence_manifest_path",
            coupling_key="bitstream_conv_evidence_manifest_path",
            provenance_key="accuracy_target_bitstream_conv_evidence_manifest_path",
        )
        or ""
    ).strip()
    if not declared_conv_evidence_manifest_path:
        declared_conv_evidence_manifest_path = str(
            contract_cfg.get("bitstream_conv_evidence_manifest_path")
            or accuracy_cfg.get("measurement_contract_bitstream_conv_evidence_manifest_path")
            or ""
        ).strip()
    declared_conv_evidence_manifest_sha256 = str(
        _resolve_observed_value(
            accuracy_key="measurement_contract_bitstream_conv_evidence_manifest_sha256",
            coupling_key="bitstream_conv_evidence_manifest_sha256",
            provenance_key="accuracy_target_bitstream_conv_evidence_manifest_sha256",
        )
        or ""
    ).strip()
    if not declared_conv_evidence_manifest_sha256:
        declared_conv_evidence_manifest_sha256 = str(
            contract_cfg.get("bitstream_conv_evidence_manifest_sha256")
            or accuracy_cfg.get("measurement_contract_bitstream_conv_evidence_manifest_sha256")
            or ""
        ).strip()
    declared_conv_measured_closure_status = str(
        _resolve_observed_value(
            accuracy_key="measurement_contract_bitstream_conv_measured_closure_status",
            coupling_key="bitstream_conv_measured_closure_status",
            provenance_key="accuracy_target_bitstream_conv_measured_closure_status",
        )
        or ""
    ).strip()
    if not declared_conv_measured_closure_status:
        declared_conv_measured_closure_status = str(
            contract_cfg.get("bitstream_conv_measured_closure_status")
            or accuracy_cfg.get("measurement_contract_bitstream_conv_measured_closure_status")
            or ""
        ).strip()
    declared_conv_measured_package_status = str(
        _resolve_observed_value(
            accuracy_key="measurement_contract_bitstream_conv_measured_package_status",
            coupling_key="bitstream_conv_measured_package_status",
            provenance_key="accuracy_target_bitstream_conv_measured_package_status",
        )
        or ""
    ).strip()
    if not declared_conv_measured_package_status:
        declared_conv_measured_package_status = str(
            contract_cfg.get("bitstream_conv_measured_package_status")
            or accuracy_cfg.get("measurement_contract_bitstream_conv_measured_package_status")
            or ""
        ).strip()
    declared_conv_measured_package_path = str(
        _resolve_observed_value(
            accuracy_key="measurement_contract_bitstream_conv_measured_package_path",
            coupling_key="bitstream_conv_measured_package_path",
            provenance_key="accuracy_target_bitstream_conv_measured_package_path",
        )
        or ""
    ).strip()
    if not declared_conv_measured_package_path:
        declared_conv_measured_package_path = str(
            contract_cfg.get("bitstream_conv_measured_package_path")
            or accuracy_cfg.get("measurement_contract_bitstream_conv_measured_package_path")
            or ""
        ).strip()
    declared_conv_measured_package_sha256 = str(
        _resolve_observed_value(
            accuracy_key="measurement_contract_bitstream_conv_measured_package_sha256",
            coupling_key="bitstream_conv_measured_package_sha256",
            provenance_key="accuracy_target_bitstream_conv_measured_package_sha256",
        )
        or ""
    ).strip()
    if not declared_conv_measured_package_sha256:
        declared_conv_measured_package_sha256 = str(
            contract_cfg.get("bitstream_conv_measured_package_sha256")
            or accuracy_cfg.get("measurement_contract_bitstream_conv_measured_package_sha256")
            or ""
        ).strip()
    declared_conv_target_set_sha256 = str(
        _resolve_observed_value(
            accuracy_key="measurement_contract_bitstream_conv_target_set_sha256",
            coupling_key="bitstream_conv_target_set_sha256",
            provenance_key="accuracy_target_bitstream_conv_target_set_sha256",
        )
        or ""
    ).strip()
    if not declared_conv_target_set_sha256:
        declared_conv_target_set_sha256 = str(
            contract_cfg.get("bitstream_conv_target_set_sha256")
            or accuracy_cfg.get("measurement_contract_bitstream_conv_target_set_sha256")
            or ""
        ).strip()
    declared_authorization_note = str(
        _resolve_observed_value(
            accuracy_key="measurement_contract_bitstream_truth_class_authorization_note",
            coupling_key="bitstream_truth_class_authorization_note",
            provenance_key="accuracy_target_bitstream_truth_class_authorization_note",
        )
        or ""
    ).strip()
    declared_authorization_status = str(
        _resolve_observed_value(
            accuracy_key="measurement_contract_bitstream_truth_class_authorization_status",
            coupling_key="bitstream_truth_class_authorization_status",
            provenance_key="accuracy_target_bitstream_truth_class_authorization_status",
        )
        or ""
    ).strip()
    expected_authorization_run_id = str(
        accuracy_provenance.get("accuracy_target_source_run_id")
        or accuracy_provenance.get("accuracy_target_row_id")
        or ""
    ).strip()
    resolved_authorization_note = resolve_truth_class_authorization_note(
        declared_authorization_note,
        search_roots=(ROOT_DIR.parent, ROOT_DIR),
    )
    authorization_assessment = assess_truth_class_authorization(
        resolved_authorization_note,
        expected_run_id=expected_authorization_run_id or None,
    )

    active_semantics = str(active_execution_semantics or "proxy").strip().lower()
    required_truth_class = (
        BITSTREAM_MODEL_LEVEL_MEASURED_TRUTH_CLASS if active_semantics == "bitstream" else ""
    )
    required_fields: list[str] = []
    observed_fields: dict[str, Any] = {}
    violations: dict[str, Any] = {"missing": [], "mismatched": {}}

    def _record_required(field: str, observed: Any, expected: Any) -> None:
        required_fields.append(field)
        observed_fields[field] = observed
        if observed is None or observed == "":
            violations["missing"].append(field)
            return
        if observed != expected:
            violations["mismatched"][field] = {
                "expected": expected,
                "observed": observed,
            }

    if active_semantics == "bitstream":
        _record_required("execution_semantics", declared_semantics, "bitstream")
        if active_bitstream_generator:
            _record_required(
                "bitstream_generator",
                declared_generator,
                str(active_bitstream_generator).strip().lower(),
            )
        if active_bitstream_stream_length is not None:
            observed_stream_length = (
                int(declared_stream_length) if declared_stream_length is not None else None
            )
            _record_required(
                "bitstream_stream_length",
                observed_stream_length,
                int(active_bitstream_stream_length),
            )
            _record_required(
                "bitstream_measurement_truth_class",
                declared_truth_class,
                BITSTREAM_MODEL_LEVEL_MEASURED_TRUTH_CLASS,
            )
        _record_required(
            "bitstream_runtime_stream_reuse_policy",
            declared_runtime_stream_reuse_policy,
            str(
                active_runtime_stream_reuse_policy
                or DEFAULT_BITSTREAM_RUNTIME_STREAM_REUSE_POLICY
            ),
        )
        runtime_manifest_metadata = _resolve_runtime_manifest_metadata(
            model_key=active_model_key,
            ops_path=active_ops_path,
        )
        expected_runtime_required_families = list(
            runtime_manifest_metadata.get("required_families") or []
        )
        if expected_runtime_required_families:
            _record_required(
                "bitstream_runtime_claim_surface_status",
                declared_runtime_claim_surface_status,
                "full_model_claim_surface_runtime",
            )
            _record_required(
                "bitstream_runtime_required_operator_families_json",
                declared_runtime_required_families,
                expected_runtime_required_families,
            )
            _record_required(
                "bitstream_runtime_covered_operator_families_json",
                declared_runtime_covered_families,
                expected_runtime_required_families,
            )
            _record_required(
                "bitstream_runtime_missing_operator_families_json",
                declared_runtime_missing_families,
                [],
            )
            _record_required(
                "bitstream_runtime_family_policy_source",
                declared_runtime_family_policy_source,
                BITSTREAM_RUNTIME_FAMILY_POLICY_SOURCE,
            )
            _record_required(
                "bitstream_runtime_manifest_sha256",
                declared_runtime_manifest_sha256,
                str(runtime_manifest_metadata.get("manifest_sha256") or ""),
            )
        if active_model_key and active_ops_path:
            conv_manifest_metadata = resolve_conv_evidence_manifest(
                model_key=str(active_model_key),
                ops_path=str(active_ops_path),
            )
            conv_package_metadata = resolve_conv_focused_measured_package()
            conv_manifest_payload = conv_manifest_metadata.get("manifest") or {}
            if (conv_manifest_payload.get("rows") or []) and (
                declared_conv_measured_closure_status
                == CONV_MEASURED_CLOSURE_STATUS_MEASURED_CLOSED
            ):
                _record_required(
                    "bitstream_conv_evidence_manifest_path",
                    declared_conv_evidence_manifest_path,
                    str(conv_manifest_metadata.get("manifest_path") or ""),
                )
                _record_required(
                    "bitstream_conv_evidence_manifest_sha256",
                    declared_conv_evidence_manifest_sha256,
                    str(conv_manifest_metadata.get("manifest_sha256") or ""),
                )
                _record_required(
                    "bitstream_conv_measured_closure_status",
                    declared_conv_measured_closure_status,
                    CONV_MEASURED_CLOSURE_STATUS_MEASURED_CLOSED,
                )
                _record_required(
                    "bitstream_conv_measured_package_status",
                    declared_conv_measured_package_status,
                    CONV_FOCUSED_MEASURED_PACKAGE_STATUS_AUTHORIZED,
                )
                _record_required(
                    "bitstream_conv_measured_package_path",
                    declared_conv_measured_package_path,
                    str(conv_package_metadata.get("package_path") or ""),
                )
                _record_required(
                    "bitstream_conv_measured_package_sha256",
                    declared_conv_measured_package_sha256,
                    str(conv_package_metadata.get("package_sha256") or ""),
                )
                _record_required(
                    "bitstream_conv_target_set_sha256",
                    declared_conv_target_set_sha256,
                    str(
                        ((conv_manifest_payload.get("conv_focused_target_set") or {}).get(
                            "target_set_sha256"
                        ))
                        or ""
                    ),
                )
        if declared_truth_class == BITSTREAM_MODEL_LEVEL_MEASURED_TRUTH_CLASS:
            _record_required(
                "bitstream_truth_class_authorization_status",
                declared_authorization_status,
                BITSTREAM_MODEL_LEVEL_MEASURED_AUTHORIZED_STATUS,
            )
            observed_fields["bitstream_truth_class_authorization_note"] = (
                str(resolved_authorization_note) if resolved_authorization_note is not None else ""
            )
            required_fields.append("bitstream_truth_class_authorization_note")
            if resolved_authorization_note is None:
                violations["missing"].append("bitstream_truth_class_authorization_note")
            elif not bool(authorization_assessment["authorized"]):
                violations["mismatched"]["bitstream_truth_class_authorization_note"] = {
                    "expected": {
                        "marker_present": True,
                        "authorized_run_id": expected_authorization_run_id,
                    },
                    "observed": {
                        "authorization_note": str(resolved_authorization_note),
                        "authorization_status": str(authorization_assessment["status"]),
                        "authorized_run_id": str(
                            authorization_assessment.get("authorized_run_id") or ""
                        ),
                    },
                }

    if active_semantics != "bitstream":
        status = "not_required"
        reason = "non_bitstream_candidate"
    elif not contract_source:
        status = "missing_source"
        reason = "no_measured_accuracy_source"
    elif violations["missing"] == ["bitstream_measurement_truth_class"] and not violations["mismatched"]:
        status = "unsatisfied"
        reason = "bitstream_measurement_truth_class_missing"
    elif (
        set(violations["missing"] + list(violations["mismatched"].keys()))
        <= {
            "bitstream_truth_class_authorization_note",
            "bitstream_truth_class_authorization_status",
        }
        and (violations["missing"] or violations["mismatched"])
    ):
        status = "unsatisfied"
        reason = "bitstream_measurement_truth_class_authorization_unsatisfied"
    elif list(violations["mismatched"].keys()) == ["bitstream_measurement_truth_class"] and not violations["missing"]:
        status = "unsatisfied"
        reason = "bitstream_measurement_truth_class_not_promotable"
    elif violations["missing"] and violations["mismatched"]:
        status = "unsatisfied"
        reason = "bitstream_measurement_contract_missing_and_mismatched_fields"
    elif violations["missing"]:
        status = "unsatisfied"
        reason = "bitstream_measurement_contract_missing_fields"
    elif violations["mismatched"]:
        status = "unsatisfied"
        reason = "bitstream_measurement_contract_mismatched_fields"
    else:
        status = "satisfied"
        reason = "bitstream_measurement_contract_satisfied"

    return {
        "accuracy_measurement_contract_status": status,
        "accuracy_measurement_contract_reason": reason,
        "accuracy_measurement_contract_source": contract_source,
        "accuracy_measurement_contract_truth_class": declared_truth_class,
        "accuracy_measurement_contract_authorization_note": (
            str(authorization_assessment.get("resolved_path") or "")
        ),
        "accuracy_measurement_contract_authorization_status": declared_authorization_status,
        "accuracy_measurement_contract_source_run_id": expected_authorization_run_id,
        "accuracy_measurement_contract_conv_evidence_manifest_path": (
            declared_conv_evidence_manifest_path
        ),
        "accuracy_measurement_contract_conv_evidence_manifest_sha256": (
            declared_conv_evidence_manifest_sha256
        ),
        "accuracy_measurement_contract_conv_measured_closure_status": (
            declared_conv_measured_closure_status
        ),
        "accuracy_measurement_contract_conv_measured_package_status": (
            declared_conv_measured_package_status
        ),
        "accuracy_measurement_contract_conv_measured_package_path": (
            declared_conv_measured_package_path
        ),
        "accuracy_measurement_contract_conv_measured_package_sha256": (
            declared_conv_measured_package_sha256
        ),
        "accuracy_measurement_contract_conv_target_set_sha256": (
            declared_conv_target_set_sha256
        ),
        "accuracy_measurement_contract_required_truth_class": required_truth_class,
        "accuracy_measurement_contract_required_fields_json": json.dumps(
            required_fields,
            ensure_ascii=False,
        ),
        "accuracy_measurement_contract_observed_fields_json": json.dumps(
            observed_fields,
            ensure_ascii=False,
            sort_keys=True,
        ),
        "accuracy_measurement_contract_violations_json": json.dumps(
            violations,
            ensure_ascii=False,
            sort_keys=True,
        ),
        "accuracy_measurement_contract_satisfied": status in {"not_required", "satisfied"},
    }


def _resolve_accuracy_coupling_metadata(
    *,
    accuracy_cfg: dict[str, Any],
    accuracy_provenance: dict[str, Any],
    acc_top1: float | None,
    acc_drop_pp: float | None,
    active_execution_semantics: str = "proxy",
    active_bitstream_generator: str | None = None,
    active_bitstream_stream_length: int | None = None,
    active_runtime_stream_reuse_policy: str | None = None,
    accuracy_measurement_contract_metadata: dict[str, Any] | None = None,
) -> dict[str, str]:
    """Resolve the evidence surface linking reported accuracy to the active lane."""
    coupling_cfg = accuracy_cfg.get("coupling") or accuracy_cfg.get("accuracy_coupling") or {}
    if not isinstance(coupling_cfg, dict):
        coupling_cfg = {}
    resolved_contract = accuracy_measurement_contract_metadata or _resolve_accuracy_measurement_contract_metadata(
        accuracy_cfg=accuracy_cfg,
        accuracy_provenance=accuracy_provenance,
        active_execution_semantics=active_execution_semantics,
        active_bitstream_generator=active_bitstream_generator,
        active_bitstream_stream_length=active_bitstream_stream_length,
        active_runtime_stream_reuse_policy=active_runtime_stream_reuse_policy,
    )
    contract_satisfied = bool(
        resolved_contract.get("accuracy_measurement_contract_satisfied", False)
    )
    contract_reason = str(
        resolved_contract.get("accuracy_measurement_contract_reason") or ""
    ).strip()

    def _contract_failure_reason() -> str:
        if resolved_contract.get("accuracy_measurement_contract_status") == "missing_source":
            return "no_accuracy_coupling_source"
        if contract_reason == "bitstream_measurement_truth_class_missing":
            return "measured_accuracy_truth_class_missing"
        if contract_reason == "bitstream_measurement_truth_class_authorization_unsatisfied":
            return "measured_accuracy_truth_class_authorization_unsatisfied"
        if contract_reason == "bitstream_measurement_truth_class_not_promotable":
            return "measured_accuracy_truth_class_not_promotable"
        return "measured_accuracy_semantics_mismatch"

    explicit_evidence = (
        coupling_cfg.get("evidence_type")
        or accuracy_cfg.get("coupling_evidence_type")
        or accuracy_cfg.get("accuracy_coupling_evidence_type")
    )
    if explicit_evidence is not None:
        evidence = _canonical_evidence_type(explicit_evidence, default="missing")
        if evidence == "measured" and not contract_satisfied:
            return {
                "accuracy_coupling_evidence": "missing",
                "accuracy_coupling_metric": str(
                    coupling_cfg.get("metric")
                    or accuracy_cfg.get("coupling_metric")
                    or "unspecified"
                ).strip(),
                "accuracy_coupling_source": str(
                    coupling_cfg.get("source")
                    or accuracy_cfg.get("coupling_source")
                    or resolved_contract.get("accuracy_measurement_contract_source")
                    or accuracy_provenance.get("accuracy_source_csv")
                    or ""
                ).strip(),
                "accuracy_coupling_reason": (
                    _contract_failure_reason()
                ),
            }
        return {
            "accuracy_coupling_evidence": evidence,
            "accuracy_coupling_metric": str(
                coupling_cfg.get("metric")
                or accuracy_cfg.get("coupling_metric")
                or "unspecified"
            ).strip(),
            "accuracy_coupling_source": str(
                coupling_cfg.get("source")
                or accuracy_cfg.get("coupling_source")
                or accuracy_provenance.get("accuracy_source_csv")
                or ""
            ).strip(),
            "accuracy_coupling_reason": str(
                coupling_cfg.get("reason")
                or accuracy_cfg.get("coupling_reason")
                or "config_explicit"
            ).strip(),
        }

    accuracy_source_csv = str(accuracy_provenance.get("accuracy_source_csv") or "").strip()
    if accuracy_source_csv and (acc_top1 is not None or acc_drop_pp is not None):
        if not contract_satisfied:
            return {
                "accuracy_coupling_evidence": "missing",
                "accuracy_coupling_metric": "top1" if acc_top1 is not None else "acc_drop_pp",
                "accuracy_coupling_source": accuracy_source_csv,
                "accuracy_coupling_reason": _contract_failure_reason(),
            }
        metric = "top1"
        if acc_top1 is None and acc_drop_pp is not None:
            metric = "acc_drop_pp"
        return {
            "accuracy_coupling_evidence": "measured",
            "accuracy_coupling_metric": metric,
            "accuracy_coupling_source": accuracy_source_csv,
            "accuracy_coupling_reason": "measured_accuracy_row_semantics_matched",
        }

    return {
        "accuracy_coupling_evidence": "missing",
        "accuracy_coupling_metric": "",
        "accuracy_coupling_source": "",
        "accuracy_coupling_reason": "no_accuracy_coupling_source",
    }


def _build_accuracy_context(
    *,
    run_id: str,
    experiment_id: str,
    out_dir: Path,
    run_cfg: dict[str, Any],
    data_cfg: dict[str, Any],
    switches: dict[str, bool],
    sc_det_cfg: dict[str, Any],
    sparse_cfg: dict[str, Any],
    p1_align_resolved: dict[str, Any],
    accuracy_cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    workload_id = str(data_cfg.get("workload_id") or data_cfg.get("workload") or "")
    accuracy_cfg = accuracy_cfg or {}
    context_run_id = str(accuracy_cfg.get("context_run_id") or run_id or "").strip()
    config_snapshot_path = (out_dir.parent / context_run_id / "config_snapshot.yaml").resolve()
    sparse_active_fraction = sparse_cfg.get("active_fraction")
    if sparse_active_fraction is None:
        sparse_active_fraction = _estimate_sparse_scale_from_tau(sparse_cfg)
    return {
        "split": data_cfg.get("split"),
        "workload": workload_id,
        "run_id": context_run_id,
        "config_snapshot": str(config_snapshot_path),
        "seed": run_cfg.get("seed"),
        "experiment_id": experiment_id,
        "det_policy": resolve_det_policy_label(sc_det_cfg, switches),
        "det_k_signature": build_det_policy_signature(sc_det_cfg, switches),
        "det_k_global": ((sc_det_cfg.get("early_stop") or {}).get("k_global")),
        "sparse_tau_global": sparse_cfg.get("tau_global"),
        "sparse_active_fraction": sparse_active_fraction,
        "gaussian_noise_std_ref": p1_align_resolved["gaussian_noise_std_ref"],
        "crosstalk_alpha_ref": p1_align_resolved["crosstalk_alpha_ref"],
    }


def _stable_random_seed(*parts: Any) -> int:
    payload = "|".join(str(part) for part in parts)
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def _apply_stochastic_accuracy_uncertainty(
    *,
    acc_ref_top1: float | None,
    acc_top1: float | None,
    acc_drop_pp: float | None,
    model: str,
    run_cfg: dict[str, Any],
    accuracy_cfg: dict[str, Any],
    noise_cfg: dict[str, Any],
) -> tuple[float | None, float | None, float | None, dict[str, Any] | None]:
    stochastic_cfg = accuracy_cfg.get("stochastic_uncertainty") or {}
    if not _to_bool(stochastic_cfg.get("enabled"), False):
        return acc_ref_top1, acc_top1, acc_drop_pp, {"applied": False, "reason": "disabled"}

    base_drop = acc_drop_pp
    if base_drop is None and acc_ref_top1 is not None and acc_top1 is not None:
        base_drop = acc_ref_top1 - acc_top1
    if base_drop is None:
        return acc_ref_top1, acc_top1, acc_drop_pp, {
            "applied": False,
            "reason": "missing_base_drop",
        }

    gaussian_noise_std = _to_float(noise_cfg.get("gaussian_noise_std"), 0.0) or 0.0
    crosstalk_alpha = _to_float(noise_cfg.get("crosstalk_alpha"), 0.0) or 0.0
    std_pp = _to_float(stochastic_cfg.get("std_pp"), 0.0) or 0.0
    min_std_pp = _to_float(stochastic_cfg.get("min_std_pp"), 0.0) or 0.0
    noise_scale = _to_float(stochastic_cfg.get("noise_scale"), 0.0) or 0.0
    min_drop_pp = _to_float(stochastic_cfg.get("min_drop_pp"), 0.0) or 0.0

    effective_std_pp = max(
        std_pp + noise_scale * (abs(gaussian_noise_std) + abs(crosstalk_alpha)),
        min_std_pp,
    )
    stable_seed = _stable_random_seed(
        run_cfg.get("run_id"),
        run_cfg.get("seed"),
        model,
        gaussian_noise_std,
        crosstalk_alpha,
    )
    rng = random.Random(stable_seed)
    sampled_delta_pp = rng.gauss(0.0, effective_std_pp)
    adjusted_drop_pp = max(min_drop_pp, base_drop + sampled_delta_pp)

    if acc_ref_top1 is not None:
        adjusted_top1 = _clamp(acc_ref_top1 - adjusted_drop_pp, 0.0, acc_ref_top1)
    elif acc_top1 is not None:
        adjusted_top1 = max(0.0, acc_top1 - (adjusted_drop_pp - base_drop))
    else:
        adjusted_top1 = None

    trace = {
        "applied": True,
        "seed": stable_seed,
        "base_drop_pp": base_drop,
        "sampled_delta_pp": sampled_delta_pp,
        "effective_std_pp": effective_std_pp,
        "adjusted_drop_pp": adjusted_drop_pp,
        "gaussian_noise_std": gaussian_noise_std,
        "crosstalk_alpha": crosstalk_alpha,
        "min_drop_pp": min_drop_pp,
    }
    return acc_ref_top1, adjusted_top1, adjusted_drop_pp, trace


def _calibrate_p1_alignment(
    *,
    p1_align_cfg: dict[str, Any],
    accuracy_rows: list[dict[str, Any]],
    model_keys: list[str],
    quant_bits_default: float | None,
    delta_pp_budget: float | None,
) -> dict[str, Any]:
    sigma_ref = _to_float(
        _cfg_value(p1_align_cfg, "gaussian_noise_sigma_lsb_ref", "sigma_lsb_ref"),
        0.0,
    ) or 0.0
    gaussian_ref = sigma_ref
    alpha_ref = _to_float(p1_align_cfg.get("crosstalk_alpha_ref"), 0.0) or 0.0
    fit_error = _to_float(p1_align_cfg.get("fit_error"), 0.0) or 0.0
    method = "config_fixed"
    selection_rule = "manual_config"

    auto_enabled = _to_bool(p1_align_cfg.get("auto_from_accuracy_csv"), True)
    target_drop = _to_float(
        p1_align_cfg.get("target_acc_drop_pp"),
        delta_pp_budget if delta_pp_budget is not None else 1.0,
    )
    quant_bits = _to_float(p1_align_cfg.get("quant_bits"), quant_bits_default)

    if auto_enabled and accuracy_rows and target_drop is not None:
        model_set = {m.strip().lower() for m in model_keys if m}
        grid_stats: dict[tuple[float, float], list[float]] = {}
        for row in accuracy_rows:
            model = str(row.get("model") or "").strip().lower()
            if model_set and model not in model_set:
                continue
            row_bits = _to_float(row.get("quant_bits"), None)
            if quant_bits is not None and (row_bits is None or abs(row_bits - quant_bits) > 1e-9):
                continue
            gaussian = _to_float(row.get("gaussian_noise_std"), None)
            alpha = _to_float(row.get("crosstalk_alpha"), None)
            top1_delta = _to_float(row.get("top1_delta"), None)
            if gaussian is None or alpha is None or top1_delta is None:
                continue
            drop = max(0.0, -top1_delta)
            grid_stats.setdefault((gaussian, alpha), []).append(drop)

        if grid_stats:
            best: tuple[float, float, float] | None = None
            for (gaussian, alpha), drops in grid_stats.items():
                mean_drop = sum(drops) / len(drops)
                score = abs(mean_drop - target_drop)
                if best is None or score < best[2]:
                    best = (gaussian, alpha, score)
            if best is not None:
                gaussian_ref, alpha_ref, fit_error = best
                method = "auto_from_accuracy_csv"
                selection_rule = "closest_mean_acc_drop_to_target"

    return {
        "gaussian_noise_std_ref": gaussian_ref,
        "sigma_lsb_ref": gaussian_ref,
        "crosstalk_alpha_ref": alpha_ref,
        "fit_error": fit_error,
        "method": method,
        "selection_rule": selection_rule,
        "target_acc_drop_pp": target_drop,
        "points_db": p1_align_cfg.get("p1_alignment_points_db") or [],
    }


def _resolve_run_id(run_cfg: dict[str, Any], experiment_id: str) -> str:
    run_id = str(run_cfg.get("run_id") or "").strip()
    if not run_id or "YYYY" in run_id:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        suffix = experiment_id.lower() if experiment_id else "phase1"
        run_id = f"{stamp}_{suffix}"
    return run_id


def _load_ops(path: Path) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, list):
        return path.stem, data, {}
    if isinstance(data, dict):
        ops = data.get("ops", [])
        model = data.get("model") or path.stem
        meta = {k: v for k, v in data.items() if k != "ops"}
        return model, ops, meta
    raise ValueError(f"Unsupported ops format: {path}")


def _infer_sequence_length_reference(
    *,
    model: str,
    meta: dict[str, Any],
    data_cfg: dict[str, Any],
) -> tuple[float | None, str]:
    for key in (
        "sequence_length_ref",
        "base_sequence_length",
        "sequence_length_base",
        "baseline_sequence_length",
        "l_ref",
    ):
        value = _to_float(data_cfg.get(key), None)
        if value is not None and value > 0:
            return value, f"data:{key}"

    meta_sequence = _to_float(meta.get("sequence_length", meta.get("l")), None)
    if meta_sequence is not None and meta_sequence > 0:
        return meta_sequence, "ops_meta"

    workload_id = str(data_cfg.get("workload_id") or data_cfg.get("workload") or "").strip().lower()
    model_key = str(model).strip().lower()
    if "imagenet" in workload_id and model_key.startswith("mobilevit"):
        # Current ImageNet MobileViT scaling configs sweep around the fixed
        # paper-time token baseline of 197. Use that as an explicit proxy so
        # sequence sweeps affect modeled latency/energy instead of only the
        # token-throughput denominator.
        return 197.0, "heuristic:imagenet_mobilevit_197"

    return None, "none"


def _resolve_workload_shape(
    *,
    model: str,
    meta: dict[str, Any],
    data_cfg: dict[str, Any],
) -> dict[str, Any]:
    base_batch_size = max(1, _to_int(meta.get("batch_size"), 1) or 1)
    batch_size = max(1, _to_int(data_cfg.get("batch_size"), base_batch_size) or base_batch_size)
    batch_scale = batch_size / max(base_batch_size, 1)

    requested_sequence_length = _to_float(
        data_cfg.get("sequence_length", data_cfg.get("l")),
        None,
    )
    base_sequence_length, sequence_scale_source = _infer_sequence_length_reference(
        model=model,
        meta=meta,
        data_cfg=data_cfg,
    )

    sequence_scale = 1.0
    sequence_modeled = False
    if (
        requested_sequence_length is not None
        and requested_sequence_length > 0
        and base_sequence_length is not None
        and base_sequence_length > 0
    ):
        sequence_scale = requested_sequence_length / base_sequence_length
        sequence_modeled = True

    workload_scale = max(batch_scale * sequence_scale, 0.0)
    if workload_scale <= 0:
        workload_scale = 1.0

    return {
        "batch_size": batch_size,
        "base_batch_size": base_batch_size,
        "batch_scale": batch_scale,
        "sequence_length": requested_sequence_length,
        "base_sequence_length": base_sequence_length,
        "sequence_scale": sequence_scale,
        "sequence_scale_source": sequence_scale_source,
        "sequence_modeled": sequence_modeled,
        "workload_scale": workload_scale,
    }


def _scale_count(value: Any, scale: float) -> Any:
    numeric = _to_float(value, None)
    if numeric is None:
        return value
    scaled = numeric * scale
    if numeric > 0 and scale > 0:
        return max(1, int(round(scaled)))
    return int(round(scaled))


def _is_sequence_sensitive_op(op: dict[str, Any]) -> bool:
    name = str(op.get("name") or "").strip().lower()
    if not name:
        return False
    return ".global_rep." in name


def _apply_workload_scale_to_ops(
    ops: list[dict[str, Any]],
    *,
    batch_scale: float,
    sequence_scale: float,
    sequence_modeled: bool,
) -> list[dict[str, Any]]:
    if (
        abs(batch_scale - 1.0) <= 1e-12
        and (not sequence_modeled or abs(sequence_scale - 1.0) <= 1e-12)
    ):
        return [dict(op) for op in ops]

    scaled_ops: list[dict[str, Any]] = []
    for op in ops:
        updated = dict(op)
        op_scale = batch_scale
        if sequence_modeled and _is_sequence_sensitive_op(op):
            op_scale *= sequence_scale
        if "m" in updated:
            updated["m"] = _scale_count(updated.get("m"), op_scale)
        if "elements" in updated:
            updated["elements"] = _scale_count(updated.get("elements"), op_scale)
        scaled_ops.append(updated)
    return scaled_ops


def _filter_ops_paths(ops_paths: list[Path], model_keys: list[str]) -> list[Path]:
    if not model_keys:
        return ops_paths
    wanted = {str(model).strip().lower() for model in model_keys if str(model).strip()}
    kept: list[Path] = []
    for ops_path in ops_paths:
        model, _, _ = _load_ops(ops_path)
        if str(model).strip().lower() in wanted:
            kept.append(ops_path)
    if not kept:
        raise SystemExit(
            f"No ops JSON files in the selected ops_dir match models.keys={sorted(wanted)}"
        )
    return kept


def _write_ops_csv(path: Path, op_results: list[dict[str, Any]]) -> None:
    _write_csv(path, OPS_FIELDS, op_results)


def _resolve_existing_path(path_like: str | Path) -> Path:
    raw = Path(path_like)
    if raw.is_absolute():
        return raw
    candidate_in_experiments = ROOT_DIR / raw
    if candidate_in_experiments.exists():
        return candidate_in_experiments
    candidate_in_repo = ROOT_DIR.parent / raw
    if candidate_in_repo.exists():
        return candidate_in_repo
    return candidate_in_experiments


def _resolve_public_variant_surface(
    *,
    run_cfg: dict[str, Any],
    experiment_id: str,
) -> dict[str, Any]:
    descriptor = default_variant_descriptor_for_experiment(experiment_id)
    explicit_variant_id = str(run_cfg.get("variant_id") or "").strip().upper()
    if explicit_variant_id:
        descriptor["variant_id"] = explicit_variant_id
    explicit_stack = run_cfg.get("public_module_stack")
    if isinstance(explicit_stack, list) and explicit_stack:
        descriptor["public_module_stack"] = [str(item) for item in explicit_stack if str(item).strip()]
    descriptor["internal_experiment_id"] = (
        str(run_cfg.get("internal_experiment_id") or descriptor["internal_experiment_id"] or experiment_id)
        .strip()
        .upper()
    )
    return descriptor


def _active_switch_set(switches: dict[str, bool]) -> str:
    enabled = [key for key in ("meso", "flow", "det", "sparse", "phy") if bool(switches.get(key))]
    return ",".join(enabled) if enabled else "none"


def _resolve_switches(cfg: dict[str, Any], experiment_id: str) -> dict[str, bool]:
    defaults = EXPERIMENT_SWITCH_MATRIX.get(
        experiment_id,
        {"meso": False, "flow": False, "det": False, "sparse": False, "phy": False},
    )
    switches_cfg = cfg.get("switches") or {}
    resolved = dict(defaults)
    for key in ("meso", "flow", "det", "sparse", "phy"):
        section_enabled = _to_bool((cfg.get(key) or {}).get("enabled"), defaults[key])
        resolved[key] = section_enabled
        if key in switches_cfg:
            resolved[key] = _to_bool(switches_cfg.get(key), resolved[key])
    return resolved


def _sync_section_enabled(cfg: dict[str, Any], switches: dict[str, bool]) -> None:
    for key in ("meso", "flow", "sparse", "phy"):
        section = cfg.get(key) or {}
        section["enabled"] = switches[key]
        cfg[key] = section
    sc_det = cfg.get("sc_det") or {}
    early_stop = sc_det.get("early_stop") or {}
    early_stop["enabled"] = switches["det"]
    sc_det["early_stop"] = early_stop
    cfg["sc_det"] = sc_det


def _compute_bsl_scale(sc_det_cfg: dict[str, Any], det_enabled: bool) -> tuple[float, float]:
    bsl_max = float(sc_det_cfg.get("bsl_max") or 1.0)
    early_stop = sc_det_cfg.get("early_stop") or {}
    if not det_enabled or not _to_bool(early_stop.get("enabled")):
        return 1.0, bsl_max
    k_global = early_stop.get("k_global")
    k_by_layer = early_stop.get("k_by_layer")
    effective_k = None
    if isinstance(k_global, (int, float)):
        effective_k = float(k_global)
    elif isinstance(k_by_layer, dict) and k_by_layer:
        values = [float(v) for v in k_by_layer.values() if v is not None]
        if values:
            effective_k = sum(values) / len(values)
    if effective_k is None:
        effective_k = bsl_max
    det_mode = str(sc_det_cfg.get("det_mode") or "reorder").strip().lower()
    if det_mode == "replace":
        mode_scale = _to_float(sc_det_cfg.get("replace_mode_scale"), 1.0) or 1.0
    else:
        mode_scale = _to_float(sc_det_cfg.get("reorder_mode_scale"), 1.0) or 1.0
    effective_k *= mode_scale
    effective_k = _clamp(effective_k, 1.0, bsl_max)
    return effective_k / max(bsl_max, 1.0), effective_k


def _compute_sparse_scale(
    sparse_cfg: dict[str, Any],
    sparse_enabled: bool,
    *,
    measured_activity_fraction: float | None = None,
) -> tuple[float, str]:
    if not sparse_enabled:
        return 1.0, "disabled"
    if measured_activity_fraction is not None:
        return (
            _clamp(float(measured_activity_fraction), 0.0, 1.0),
            "measured_accuracy_row",
        )
    tau_estimate = _estimate_sparse_scale_from_tau(sparse_cfg)
    if tau_estimate is not None:
        return _clamp(float(tau_estimate), 0.0, 1.0), "tau_threshold_estimate"
    active_fraction = sparse_cfg.get("active_fraction")
    if active_fraction is None and "sparsity" in sparse_cfg:
        active_fraction = 1.0 - float(sparse_cfg.get("sparsity") or 0.0)
    if active_fraction is None:
        return 1.0, "default_full_activity"
    return _clamp(float(active_fraction), 0.0, 1.0), "proxy_active_fraction"


def _scale_op_results(
    op_results: list[dict[str, Any]],
    *,
    bsl_max: float,
    bsl_scale: float,
    det_enabled: bool,
    k_by_layer: dict[str, Any] | None,
    sparse_scale: float,
    meso_load_scale: float,
) -> list[dict[str, Any]]:
    scaled: list[dict[str, Any]] = []
    for idx, op in enumerate(op_results, start=1):
        op_type = op.get("type")
        is_elementwise = op_type in ELEMENTWISE_TYPES
        local_bsl_scale = bsl_scale
        if det_enabled and isinstance(k_by_layer, dict) and k_by_layer:
            layer_name = str(op.get("name") or "")
            per_layer_k = _to_float(k_by_layer.get(layer_name), None)
            if per_layer_k is None:
                per_layer_k = _to_float(k_by_layer.get(str(idx)), None)
            if per_layer_k is not None:
                local_bsl_scale = _clamp(per_layer_k / max(bsl_max, 1.0), 0.0, 1e9)
        scale = sparse_scale if is_elementwise else (local_bsl_scale * sparse_scale)
        scale = _clamp(scale, 0.0, 1e9)

        updated = dict(op)
        if is_elementwise:
            updated["latency_ms"] = (updated.get("latency_ms") or 0.0) * scale
            updated["energy_mj_elementwise"] = (
                updated.get("energy_mj_elementwise") or 0.0
            ) * scale
            updated["energy_mj"] = updated["energy_mj_elementwise"]
        else:
            for key in GEMM_COMPONENT_FIELDS:
                updated[key] = (updated.get(key) or 0.0) * scale
            updated["energy_mj_load_x"] *= meso_load_scale
            updated["energy_mj_load_y"] *= meso_load_scale
            updated["energy_mj_detect"] = (
                (updated.get("energy_mj_oe") or 0.0)
                + (updated.get("energy_mj_adc_pca") or 0.0)
            )
            updated["energy_mj"] = sum(
                updated.get(k) or 0.0
                for k in (
                    "energy_mj_load_x",
                    "energy_mj_load_y",
                    "energy_mj_oe",
                    "energy_mj_adc_pca",
                    "energy_mj_laser",
                    "energy_mj_mem",
                    "energy_mj_static",
                )
            )
            updated["latency_ms"] = (updated.get("latency_ms") or 0.0) * scale

        latency_ms = updated.get("latency_ms") or 0.0
        energy_mj = updated.get("energy_mj") or 0.0
        updated["power_w"] = (
            (energy_mj / 1e3) / (latency_ms / 1e3) if latency_ms > 0 else None
        )
        scaled.append(updated)
    return scaled


def _summarize_scaled_ops(
    op_results: list[dict[str, Any]],
    *,
    meso_overhead_mj: float,
    meso_overhead_ms: float,
) -> dict[str, Any]:
    totals = {
        "total_latency_ms": 0.0,
        "total_energy_mj": 0.0,
        "energy_mj_load_x": 0.0,
        "energy_mj_load_y": 0.0,
        "energy_mj_detect": 0.0,
        "energy_mj_oe": 0.0,
        "energy_mj_adc_pca": 0.0,
        "energy_mj_laser": 0.0,
        "energy_mj_mem": 0.0,
        "energy_mj_static": 0.0,
        "energy_mj_elementwise": 0.0,
    }
    for op in op_results:
        totals["total_latency_ms"] += float(op.get("latency_ms") or 0.0)
        totals["total_energy_mj"] += float(op.get("energy_mj") or 0.0)
        totals["energy_mj_load_x"] += float(op.get("energy_mj_load_x") or 0.0)
        totals["energy_mj_load_y"] += float(op.get("energy_mj_load_y") or 0.0)
        totals["energy_mj_detect"] += float(op.get("energy_mj_detect") or 0.0)
        totals["energy_mj_oe"] += float(op.get("energy_mj_oe") or 0.0)
        totals["energy_mj_adc_pca"] += float(op.get("energy_mj_adc_pca") or 0.0)
        totals["energy_mj_laser"] += float(op.get("energy_mj_laser") or 0.0)
        totals["energy_mj_mem"] += float(op.get("energy_mj_mem") or 0.0)
        totals["energy_mj_static"] += float(op.get("energy_mj_static") or 0.0)
        totals["energy_mj_elementwise"] += float(op.get("energy_mj_elementwise") or 0.0)

    totals["total_energy_mj"] += meso_overhead_mj
    totals["energy_mj_static"] += meso_overhead_mj
    totals["total_latency_ms"] += meso_overhead_ms

    latency_s = totals["total_latency_ms"] / 1e3
    energy_j = totals["total_energy_mj"] / 1e3
    totals["total_power_w"] = _safe_ratio(energy_j, latency_s) if latency_s > 0 else None
    totals["energy_j_total"] = energy_j
    totals["energy_j_conversion_control"] = (
        (totals["energy_mj_load_x"] + totals["energy_mj_load_y"]) / 1e3
    )
    totals["energy_j_memory_move"] = totals["energy_mj_mem"] / 1e3
    totals["energy_j_oe"] = totals["energy_mj_oe"] / 1e3
    totals["energy_j_adc_pca"] = totals["energy_mj_adc_pca"] / 1e3
    totals["energy_j_laser_optical"] = totals["energy_mj_laser"] / 1e3
    totals["energy_j_other_static"] = (
        (totals["energy_mj_static"] + totals["energy_mj_elementwise"]) / 1e3
    )
    return totals


def _extract_bitstream_estimator_metadata(estimator_summary: dict[str, Any]) -> dict[str, Any]:
    """Lift auditable bitstream calibration metadata from estimator outputs."""
    model_abstraction_boundary = estimator_summary.get("model_abstraction_boundary") or {}
    if not isinstance(model_abstraction_boundary, dict):
        model_abstraction_boundary = {}
    return {
        "bitstream_effective_stream_length": estimator_summary.get(
            "bitstream_effective_stream_length"
        ),
        "bitstream_effective_stream_length_scale": estimator_summary.get(
            "bitstream_effective_stream_length_scale"
        ),
        "bitstream_effective_stream_length_scale_provenance": estimator_summary.get(
            "bitstream_effective_stream_length_scale_provenance"
        ),
        "bitstream_parallel_outputs": estimator_summary.get("bitstream_parallel_outputs"),
        "bitstream_parallel_outputs_provenance": estimator_summary.get(
            "bitstream_parallel_outputs_provenance"
        ),
        "bitstream_cycles_per_stream_bit": estimator_summary.get(
            "bitstream_cycles_per_stream_bit"
        ),
        "bitstream_cycles_per_stream_bit_provenance": estimator_summary.get(
            "bitstream_cycles_per_stream_bit_provenance"
        ),
        "bitstream_accumulator_energy_pj": estimator_summary.get(
            "bitstream_accumulator_energy_pj"
        ),
        "bitstream_accumulator_energy_pj_provenance": estimator_summary.get(
            "bitstream_accumulator_energy_pj_provenance"
        ),
        "bitstream_elementwise_parallelism_factor": json.dumps(
            estimator_summary.get("bitstream_elementwise_parallelism_factor") or {},
            ensure_ascii=False,
            sort_keys=True,
        ),
        "bitstream_elementwise_parallelism_provenance": json.dumps(
            estimator_summary.get("bitstream_elementwise_parallelism_provenance") or {},
            ensure_ascii=False,
            sort_keys=True,
        ),
        "bitstream_calibration_applied": estimator_summary.get("bitstream_calibration_applied"),
        "bitstream_calibration_summary_json": estimator_summary.get(
            "bitstream_calibration_summary_json"
        ),
        "bitstream_calibration_reason": estimator_summary.get("bitstream_calibration_reason"),
        "generator_stream_state_policy": json.dumps(
            estimator_summary.get("generator_stream_state_policy")
            or estimator_summary.get("bitstream_stream_state_policy")
            or {},
            ensure_ascii=False,
            sort_keys=True,
        ),
        "bitstream_calibration_median_relative_error": estimator_summary.get(
            "bitstream_calibration_median_relative_error"
        ),
        "bitstream_calibration_capture_row_count": estimator_summary.get(
            "bitstream_calibration_capture_row_count"
        ),
        "bitstream_calibration_replay_row_count": estimator_summary.get(
            "bitstream_calibration_replay_row_count"
        ),
        "bitstream_datapath_stage_summary": json.dumps(
            estimator_summary.get("bitstream_datapath_stage_summary") or {},
            ensure_ascii=False,
            sort_keys=True,
        ),
        "model_abstraction_boundary_kind": model_abstraction_boundary.get(
            "model_abstraction_boundary_kind"
        ),
        "model_abstraction_boundary_status": model_abstraction_boundary.get(
            "model_abstraction_boundary_status"
        ),
        "model_abstraction_boundary_reason": model_abstraction_boundary.get(
            "model_abstraction_boundary_reason"
        ),
        "model_abstraction_boundary_json": json.dumps(
            model_abstraction_boundary,
            ensure_ascii=False,
            sort_keys=True,
        ),
        "conv2d_gemm_lowered_approximation_op_count": model_abstraction_boundary.get(
            "conv2d_gemm_lowered_approximation_op_count"
        ),
        "conv2d_native_runtime_modeled_op_count": model_abstraction_boundary.get(
            "conv2d_native_runtime_modeled_op_count"
        ),
    }


def _extract_workload_fidelity_metadata(estimator_summary: dict[str, Any]) -> dict[str, Any]:
    blockers = estimator_summary.get("workload_fidelity_blockers") or []
    if isinstance(blockers, str):
        try:
            decoded = json.loads(blockers)
        except (TypeError, ValueError, json.JSONDecodeError):
            decoded = [blockers] if blockers else []
        blockers = decoded if isinstance(decoded, list) else []
    if not isinstance(blockers, list):
        blockers = []
    normalized_blockers = [str(item) for item in blockers if str(item).strip()]
    native_claim_eligible = _to_bool(
        estimator_summary.get("workload_native_claim_eligible"),
        not normalized_blockers,
    )
    return {
        "workload_fidelity_class": str(
            estimator_summary.get("workload_fidelity_class")
            or ("native_workload_fidelity" if native_claim_eligible else "approximate_workload_fidelity")
        ),
        "workload_fidelity_status": str(
            estimator_summary.get("workload_fidelity_status")
            or ("native_ready" if native_claim_eligible else "approximate")
        ),
        "workload_fidelity_reason": str(
            estimator_summary.get("workload_fidelity_reason")
            or ("all_required_workload_models_are_native" if native_claim_eligible else "workload_fidelity_blocked")
        ),
        "workload_fidelity_blockers": json.dumps(
            normalized_blockers,
            ensure_ascii=False,
        ),
        "workload_native_claim_eligible": native_claim_eligible,
    }


def _resolve_conv_measured_closure_metadata(
    *,
    conv_evidence_manifest: dict[str, Any],
    accuracy_measurement_contract_metadata: dict[str, Any] | None,
) -> dict[str, Any]:
    resolved_contract = accuracy_measurement_contract_metadata or {}
    measured_contract_status = str(
        resolved_contract.get("accuracy_measurement_contract_status") or ""
    ).strip()
    measured_truth_class = str(
        resolved_contract.get("accuracy_measurement_contract_truth_class") or ""
    ).strip()
    authorization_status = str(
        resolved_contract.get("accuracy_measurement_contract_authorization_status") or ""
    ).strip()
    declared_status = str(
        resolved_contract.get("accuracy_measurement_contract_conv_measured_closure_status")
        or CONV_MEASURED_CLOSURE_STATUS_RUNTIME_MODELED
    ).strip()
    declared_manifest_path = str(
        resolved_contract.get("accuracy_measurement_contract_conv_evidence_manifest_path")
        or ""
    ).strip()
    declared_manifest_sha256 = str(
        resolved_contract.get("accuracy_measurement_contract_conv_evidence_manifest_sha256")
        or ""
    ).strip()
    declared_package_status = str(
        resolved_contract.get("accuracy_measurement_contract_conv_measured_package_status")
        or ""
    ).strip()
    declared_package_path = str(
        resolved_contract.get("accuracy_measurement_contract_conv_measured_package_path") or ""
    ).strip()
    declared_package_sha256 = str(
        resolved_contract.get("accuracy_measurement_contract_conv_measured_package_sha256") or ""
    ).strip()
    declared_target_set_sha256 = str(
        resolved_contract.get("accuracy_measurement_contract_conv_target_set_sha256") or ""
    ).strip()
    source_run_id = str(
        resolved_contract.get("accuracy_measurement_contract_source_run_id") or ""
    ).strip()
    expected_target_set_sha256 = str(
        ((conv_evidence_manifest.get("manifest") or {}).get("conv_focused_target_set") or {}).get(
            "target_set_sha256"
        )
        or ""
    ).strip()
    expected_manifest_path = str(conv_evidence_manifest.get("manifest_path") or "").strip()
    expected_manifest_sha256 = str(conv_evidence_manifest.get("manifest_sha256") or "").strip()
    conv_package_metadata = resolve_conv_focused_measured_package()
    expected_package_path = str(conv_package_metadata.get("package_path") or "").strip()
    expected_package_sha256 = str(conv_package_metadata.get("package_sha256") or "").strip()
    package_validation = validate_conv_focused_measured_package(
        conv_evidence_manifest=conv_evidence_manifest,
        conv_focused_measured_package=conv_package_metadata,
    )
    package_measured_run_id = str(package_validation.get("measured_run_id") or "").strip()

    blockers: list[str] = []
    if declared_status != CONV_MEASURED_CLOSURE_STATUS_MEASURED_CLOSED:
        return {
            "conv_measured_closure_ready": False,
            "conv_measured_closure_status": CONV_MEASURED_CLOSURE_STATUS_RUNTIME_MODELED,
            "conv_measured_closure_blockers": [],
        }

    ready = True
    if measured_contract_status != "satisfied":
        ready = False
        blockers.append("conv_measured_contract_unsatisfied")
    if measured_truth_class != BITSTREAM_MODEL_LEVEL_MEASURED_TRUTH_CLASS:
        ready = False
        blockers.append("conv_measured_truth_class_unsatisfied")
    if authorization_status != BITSTREAM_MODEL_LEVEL_MEASURED_AUTHORIZED_STATUS:
        ready = False
        blockers.append("conv_measured_authorization_unsatisfied")
    if package_measured_run_id == LEGACY_ALL_TARGET_MEASURED_ANCHOR_RUN_ID:
        ready = False
        blockers.append("legacy_all_target_row_not_authorized_for_conv_closure")
    elif not package_measured_run_id and source_run_id == LEGACY_ALL_TARGET_MEASURED_ANCHOR_RUN_ID:
        ready = False
        blockers.append("legacy_all_target_row_not_authorized_for_conv_closure")
    if expected_manifest_path and declared_manifest_path != expected_manifest_path:
        ready = False
        blockers.append("conv_evidence_manifest_not_bound")
    if expected_manifest_sha256 and declared_manifest_sha256 != expected_manifest_sha256:
        ready = False
        blockers.append("conv_evidence_manifest_sha_mismatch")
    if declared_package_status != CONV_FOCUSED_MEASURED_PACKAGE_STATUS_AUTHORIZED:
        ready = False
        blockers.append("conv_measured_package_not_authorized")
    if expected_package_path and declared_package_path != expected_package_path:
        ready = False
        blockers.append("conv_measured_package_not_bound")
    if expected_package_sha256 and declared_package_sha256 != expected_package_sha256:
        ready = False
        blockers.append("conv_measured_package_sha_mismatch")
    if expected_target_set_sha256 and declared_target_set_sha256 != expected_target_set_sha256:
        ready = False
        blockers.append("conv_target_set_not_bound")
    for blocker in package_validation.get("package_blockers") or []:
        blocker_text = str(blocker).strip()
        if blocker_text:
            ready = False
            blockers.append(blocker_text)

    deduped_blockers: list[str] = []
    for blocker in blockers:
        if blocker not in deduped_blockers:
            deduped_blockers.append(blocker)
    return {
        "conv_measured_closure_ready": ready,
        "conv_measured_closure_status": (
            CONV_MEASURED_CLOSURE_STATUS_MEASURED_CLOSED
            if ready
            else (
                CONV_MEASURED_CLOSURE_STATUS_UNCLOSED
                if declared_status == CONV_MEASURED_CLOSURE_STATUS_MEASURED_CLOSED
                else CONV_MEASURED_CLOSURE_STATUS_RUNTIME_MODELED
            )
        ),
        "conv_measured_closure_blockers": deduped_blockers,
    }


def _resolve_conv_evidence_metadata(
    *,
    op_results: list[dict[str, Any]] | None,
    active_model_key: str | None,
    active_ops_path: str | Path | None,
    accuracy_measurement_contract_metadata: dict[str, Any] | None,
) -> dict[str, Any]:
    conv_rows = [
        row
        for row in (op_results or [])
        if str(row.get("type") or "").strip().lower() == "conv2d"
    ]
    if not conv_rows:
        return {
            "conv_present": False,
            "conv_fidelity_stage": "",
            "conv_fidelity_blockers": json.dumps([], ensure_ascii=False),
            "conv_evidence_manifest_path": "",
            "conv_evidence_manifest_sha256": "",
            "conv_measured_package_path": "",
            "conv_measured_package_sha256": "",
            "conv_measured_closure_status": "",
            "conv_premeasurement_contract_ready": True,
        }

    conv_evidence_manifest = resolve_conv_evidence_manifest(
        model_key=str(active_model_key or ""),
        ops_path=str(active_ops_path or ""),
    )
    conv_package_metadata = resolve_conv_focused_measured_package()
    package_validation = validate_conv_focused_measured_package(
        conv_evidence_manifest=conv_evidence_manifest,
        conv_focused_measured_package=conv_package_metadata,
    )
    manifest_payload = conv_evidence_manifest.get("manifest") or {}
    manifest_rows = {
        str(row.get("module_key") or ""): row
        for row in (manifest_payload.get("rows") or [])
        if isinstance(row, dict) and str(row.get("module_key") or "").strip()
    }
    premeasurement_blockers: list[str] = []
    for row in conv_rows:
        if str(row.get("model_abstraction_kind") or "") != "conv2d_native_runtime_path_modeled":
            premeasurement_blockers.append("conv_premeasurement_contract_unsatisfied")
        if not str(row.get("conv_native_class") or "").strip():
            premeasurement_blockers.append("conv_premeasurement_contract_unsatisfied")
        if not str(row.get("conv_runtime_semantics_json") or "").strip():
            premeasurement_blockers.append("conv_premeasurement_contract_unsatisfied")
        if str(row.get("name") or "").strip() not in manifest_rows:
            premeasurement_blockers.append("conv_provenance_artifact_missing")
    premeasurement_blockers = sorted(set(premeasurement_blockers))
    closure_metadata = _resolve_conv_measured_closure_metadata(
        conv_evidence_manifest=conv_evidence_manifest,
        accuracy_measurement_contract_metadata=accuracy_measurement_contract_metadata,
    )
    fidelity_blockers = list(premeasurement_blockers)
    if not closure_metadata["conv_measured_closure_ready"]:
        fidelity_blockers.append(CONV_HARDWARE_EVIDENCE_UNCLOSED)
        fidelity_stage = CONV_FIDELITY_STAGE_RUNTIME_MODELED
    else:
        fidelity_stage = CONV_FIDELITY_STAGE_MEASURED_CLOSED
    for blocker in closure_metadata["conv_measured_closure_blockers"]:
        if blocker != CONV_HARDWARE_EVIDENCE_UNCLOSED:
            fidelity_blockers.append(blocker)
    fidelity_blockers = sorted(set(fidelity_blockers))
    return {
        "conv_present": True,
        "conv_fidelity_stage": fidelity_stage,
        "conv_fidelity_blockers": json.dumps(fidelity_blockers, ensure_ascii=False),
        "conv_evidence_manifest_path": str(conv_evidence_manifest.get("manifest_path") or ""),
        "conv_evidence_manifest_sha256": str(
            conv_evidence_manifest.get("manifest_sha256") or ""
        ),
        "conv_measured_package_path": str(package_validation.get("package_path") or ""),
        "conv_measured_package_sha256": str(package_validation.get("package_sha256") or ""),
        "conv_measured_closure_status": closure_metadata["conv_measured_closure_status"],
        "conv_premeasurement_contract_ready": not premeasurement_blockers,
    }


def _build_fidelity_metadata(
    *,
    active_execution_semantics: str = "proxy",
    op_results: list[dict[str, Any]] | None = None,
    active_model_key: str | None = None,
    active_ops_path: str | Path | None = None,
    workload_fidelity_metadata: dict[str, Any] | None,
    bitstream_estimator_metadata: dict[str, Any] | None,
    accuracy_measurement_contract_metadata: dict[str, Any] | None,
) -> dict[str, Any]:
    blockers: list[str] = []
    active_semantics = str(active_execution_semantics or "proxy").strip().lower()
    resolved_workload = workload_fidelity_metadata or {}
    raw_workload_blockers = resolved_workload.get("workload_fidelity_blockers") or "[]"
    try:
        workload_blockers = json.loads(raw_workload_blockers)
    except (TypeError, ValueError, json.JSONDecodeError):
        workload_blockers = []
    for blocker in workload_blockers:
        blocker_text = str(blocker).strip()
        if blocker_text and blocker_text not in blockers:
            blockers.append(blocker_text)

    resolved_contract = accuracy_measurement_contract_metadata or {}
    try:
        violations = json.loads(
            resolved_contract.get("accuracy_measurement_contract_violations_json") or "{}"
        )
    except (TypeError, ValueError, json.JSONDecodeError):
        violations = {}
    missing_fields = {str(item) for item in (violations.get("missing") or []) if str(item).strip()}
    mismatched_fields = {
        str(item) for item in (violations.get("mismatched") or {}).keys() if str(item).strip()
    }
    if active_semantics == "bitstream":
        runtime_family_fields = {
            "bitstream_runtime_claim_surface_status",
            "bitstream_runtime_required_operator_families_json",
            "bitstream_runtime_covered_operator_families_json",
            "bitstream_runtime_missing_operator_families_json",
            "bitstream_runtime_family_policy_source",
            "bitstream_runtime_manifest_sha256",
        }
        if runtime_family_fields & (missing_fields | mismatched_fields):
            blockers.append("required_runtime_family_missing")
        if "bitstream_runtime_stream_reuse_policy" in missing_fields | mismatched_fields:
            blockers.append("runtime_reuse_policy_not_bound")

    resolved_scalars = bitstream_estimator_metadata or {}
    if active_semantics == "bitstream":
        scalar_provenance_fields = [
            "bitstream_parallel_outputs_provenance",
            "bitstream_cycles_per_stream_bit_provenance",
            "bitstream_accumulator_energy_pj_provenance",
            "bitstream_effective_stream_length_scale_provenance",
        ]
        scalar_provenances: set[str] = set()
        scalar_provenance_missing = False
        for field in scalar_provenance_fields:
            provenance = str(resolved_scalars.get(field) or "").strip()
            if not provenance:
                scalar_provenance_missing = True
                continue
            scalar_provenances.add(provenance)
        try:
            elementwise_provenance = json.loads(
                resolved_scalars.get("bitstream_elementwise_parallelism_provenance") or "{}"
            )
        except (TypeError, ValueError, json.JSONDecodeError):
            elementwise_provenance = {}
        if not isinstance(elementwise_provenance, dict):
            elementwise_provenance = {}
        for key in ("activation", "norm"):
            provenance = str(elementwise_provenance.get(key) or "").strip()
            if not provenance:
                scalar_provenance_missing = True
                continue
            scalar_provenances.add(provenance)
        if scalar_provenance_missing or "implicit_default" in scalar_provenances:
            blockers.append("critical_scalar_provenance_implicit")

    conv_evidence_metadata = _resolve_conv_evidence_metadata(
        op_results=op_results,
        active_model_key=active_model_key,
        active_ops_path=active_ops_path,
        accuracy_measurement_contract_metadata=accuracy_measurement_contract_metadata,
    )
    if bool(conv_evidence_metadata.get("conv_present")):
        if (
            str(conv_evidence_metadata.get("conv_fidelity_stage") or "")
            == CONV_FIDELITY_STAGE_MEASURED_CLOSED
        ):
            blockers = [
                blocker
                for blocker in blockers
                if blocker != CONV_HARDWARE_EVIDENCE_UNCLOSED
            ]
        try:
            conv_blockers = json.loads(
                conv_evidence_metadata.get("conv_fidelity_blockers") or "[]"
            )
        except (TypeError, ValueError, json.JSONDecodeError):
            conv_blockers = []
        for blocker in conv_blockers:
            blocker_text = str(blocker).strip()
            if blocker_text:
                blockers.append(blocker_text)

    deduped_blockers: list[str] = []
    seen: set[str] = set()
    for blocker in blockers:
        if blocker in seen:
            continue
        seen.add(blocker)
        deduped_blockers.append(blocker)
    return {
        "fidelity_ready": not deduped_blockers,
        "fidelity_blockers": json.dumps(deduped_blockers, ensure_ascii=False),
        "conv_fidelity_stage": conv_evidence_metadata.get("conv_fidelity_stage") or "",
        "conv_fidelity_blockers": conv_evidence_metadata.get("conv_fidelity_blockers") or "[]",
        "conv_evidence_manifest_path": conv_evidence_metadata.get(
            "conv_evidence_manifest_path"
        )
        or "",
        "conv_evidence_manifest_sha256": conv_evidence_metadata.get(
            "conv_evidence_manifest_sha256"
        )
        or "",
        "conv_measured_package_path": conv_evidence_metadata.get(
            "conv_measured_package_path"
        )
        or "",
        "conv_measured_package_sha256": conv_evidence_metadata.get(
            "conv_measured_package_sha256"
        )
        or "",
        "conv_measured_closure_status": conv_evidence_metadata.get(
            "conv_measured_closure_status"
        )
        or "",
    }


def _extract_sc_default_metadata(estimator_summary: dict[str, Any]) -> dict[str, Any]:
    support_class_counts = estimator_summary.get("support_class_counts")
    if not isinstance(support_class_counts, dict):
        support_class_counts = estimator_summary.get("support_class_inventory")
    if not isinstance(support_class_counts, dict):
        support_class_counts = {}
    support_classes = (
        estimator_summary.get("sc_support_classes")
        or [key for key, value in support_class_counts.items() if int(value or 0) > 0]
        or estimator_summary.get("support_class_inventory_json")
        or []
    )
    if isinstance(support_classes, str):
        try:
            decoded = json.loads(support_classes)
        except (TypeError, ValueError):
            decoded = None
        support_classes = decoded if isinstance(decoded, list) else ([support_classes] if support_classes else [])
    support_classes = sorted({str(item) for item in support_classes if str(item).strip()})
    estimator_mode = str(estimator_summary.get("estimator_mode") or "").strip().lower()
    default_trust = "proxy_reference" if estimator_mode == "proxy" else "out_of_band"
    default_calibration = "proxy_reference" if estimator_mode == "proxy" else "uncalibrated"
    trust_inventory = estimator_summary.get("trust_posture_inventory") or []
    if isinstance(trust_inventory, str):
        try:
            trust_inventory = json.loads(trust_inventory)
        except (TypeError, ValueError):
            trust_inventory = [trust_inventory] if trust_inventory else []
    if not isinstance(trust_inventory, list):
        trust_inventory = []
    summary_trust = str(
        estimator_summary.get("sc_summary_trust_posture")
        or estimator_summary.get("summary_trust_posture")
        or estimator_summary.get("trust_posture")
        or default_trust
    )
    if summary_trust == default_trust and trust_inventory:
        if "out_of_band" in trust_inventory:
            summary_trust = "out_of_band"
        elif "default_with_supporting_assumptions" in trust_inventory:
            summary_trust = "default_with_supporting_assumptions"
        elif "trusted_default" in trust_inventory:
            summary_trust = "trusted_default"
    calibration_state = estimator_summary.get("sc_calibration_state")
    if not calibration_state:
        calibration_applied = _to_bool(
            estimator_summary.get("bitstream_calibration_applied"),
            False,
        )
        if estimator_mode == "proxy":
            calibration_state = "proxy_reference"
        elif not calibration_applied:
            calibration_state = "uncalibrated"
        elif summary_trust == "out_of_band":
            calibration_state = "calibrated_out_of_band"
        else:
            calibration_state = "calibrated_in_band"
    return {
        "sc_summary_trust_posture": summary_trust,
        "sc_calibration_state": str(calibration_state or default_calibration),
        "sc_generator_policy_status": str(
            estimator_summary.get("sc_generator_policy_status") or ""
        ),
        "sc_generator_policy_reason": str(
            estimator_summary.get("sc_generator_policy_reason") or ""
        ),
        "sc_support_classes": json.dumps(support_classes, ensure_ascii=False),
        "sc_native_op_count": int(
            estimator_summary.get("sc_native_op_count")
            or support_class_counts.get("sc_native_bitstream")
            or 0
        ),
        "sc_governed_support_op_count": int(
            estimator_summary.get("sc_governed_support_op_count")
            or support_class_counts.get(
                "sc_governed_electronic_support"
            )
            or 0
        ),
        "sc_unsupported_op_count": int(
            estimator_summary.get("sc_unsupported_op_count")
            or support_class_counts.get("sc_default_unsupported")
            or 0
        ),
        "estimation_model_coverage_status": str(
            estimator_summary.get("estimation_model_coverage_status")
            or (
                "incomplete_estimation_model"
                if int(
                    estimator_summary.get("sc_unsupported_op_count")
                    or support_class_counts.get("sc_default_unsupported")
                    or 0
                )
                > 0
                else "complete_estimation_model"
            )
        ),
        "estimation_model_coverage_reason": str(
            estimator_summary.get("estimation_model_coverage_reason")
            or (
                "unsupported_operator_models_present"
                if int(
                    estimator_summary.get("sc_unsupported_op_count")
                    or support_class_counts.get("sc_default_unsupported")
                    or 0
                )
                > 0
                else "all_ops_have_native_or_governed_models"
            )
        ),
        "estimation_model_support_boundary": str(
            estimator_summary.get("estimation_model_support_boundary")
            or (
                "native_plus_governed_support"
                if int(
                    estimator_summary.get("sc_governed_support_op_count")
                    or support_class_counts.get("sc_governed_electronic_support")
                    or 0
                )
                > 0
                else "native_only"
            )
        ),
        "estimation_model_supported_op_count": int(
            estimator_summary.get("estimation_model_supported_op_count")
            or (
                int(
                    estimator_summary.get("sc_native_op_count")
                    or support_class_counts.get("sc_native_bitstream")
                    or 0
                )
                + int(
                    estimator_summary.get("sc_governed_support_op_count")
                    or support_class_counts.get("sc_governed_electronic_support")
                    or 0
                )
            )
        ),
        "estimation_model_unsupported_op_count": int(
            estimator_summary.get("estimation_model_unsupported_op_count")
            or estimator_summary.get("sc_unsupported_op_count")
            or support_class_counts.get("sc_default_unsupported")
            or 0
        ),
    }


def _derive_estimation_model_readiness(
    *,
    sc_default_metadata: dict[str, Any],
    realism_assessment: dict[str, Any],
    workload_fidelity_metadata: dict[str, Any] | None = None,
    fidelity_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    blockers: list[str] = []
    coverage_status = str(
        sc_default_metadata.get("estimation_model_coverage_status") or ""
    ).strip()
    if coverage_status != "complete_estimation_model":
        blockers.append("operator_models_incomplete")
    integrated_system_cost_evidence = str(
        realism_assessment.get("integrated_system_cost_evidence") or ""
    ).strip()
    if integrated_system_cost_evidence in {"", "missing", "disabled"}:
        blockers.append("integrated_system_costs_incomplete")
    resolved_fidelity = fidelity_metadata or {}
    if not _to_bool(resolved_fidelity.get("fidelity_ready"), True):
        try:
            fidelity_blockers = json.loads(resolved_fidelity.get("fidelity_blockers") or "[]")
        except (TypeError, ValueError, json.JSONDecodeError):
            fidelity_blockers = []
        for blocker in fidelity_blockers:
            blocker_text = str(blocker).strip()
            if blocker_text and blocker_text not in blockers:
                blockers.append(blocker_text)
    else:
        resolved_workload_fidelity = workload_fidelity_metadata or {}
        if not _to_bool(
            resolved_workload_fidelity.get("workload_native_claim_eligible"),
            True,
        ):
            for blocker in json.loads(
                resolved_workload_fidelity.get("workload_fidelity_blockers") or "[]"
            ):
                blocker_text = str(blocker).strip()
                if blocker_text and blocker_text not in blockers:
                    blockers.append(blocker_text)
    return {
        "estimation_model_ready_status": (
            "ready_for_required_estimation"
            if not blockers
            else "not_ready_for_required_estimation"
        ),
        "estimation_model_ready": not blockers,
        "estimation_model_ready_reason": (
            "complete_modeling_available_for_required_estimation"
            if not blockers
            else ";".join(blockers)
        ),
        "estimation_model_ready_blockers": ";".join(blockers),
    }


def _extract_true_sc_claim_metadata(estimator_summary: dict[str, Any]) -> dict[str, Any]:
    true_sc_claim_state_inventory = estimator_summary.get("true_sc_claim_state_inventory") or {}
    if not isinstance(true_sc_claim_state_inventory, dict):
        true_sc_claim_state_inventory = {}
    true_sc_claim_surface_inventory = (
        estimator_summary.get("true_sc_claim_surface_inventory") or {}
    )
    if not isinstance(true_sc_claim_surface_inventory, dict):
        true_sc_claim_surface_inventory = {}
    estimator_mode = str(estimator_summary.get("estimator_mode") or "").strip().lower()
    workload_native_claim_eligible = _to_bool(
        estimator_summary.get("workload_native_claim_eligible"),
        True,
    )
    true_sc_summary_claim_state = str(
        estimator_summary.get("true_sc_summary_claim_state") or ""
    ).strip()
    if not true_sc_summary_claim_state:
        if int(true_sc_claim_state_inventory.get("governed_support_not_true_sc", 0)) > 0:
            true_sc_summary_claim_state = "governed_support_not_true_sc"
        elif int(true_sc_claim_state_inventory.get("true_sc_out_of_surface", 0)) > 0:
            true_sc_summary_claim_state = "true_sc_out_of_surface"
        elif not workload_native_claim_eligible:
            true_sc_summary_claim_state = "true_sc_native_blocked_by_workload_fidelity"
        elif estimator_mode == "bitstream":
            true_sc_summary_claim_state = "true_sc_native"
        else:
            true_sc_summary_claim_state = "true_sc_out_of_surface"
    surface_native_count = int(
        estimator_summary.get("true_sc_claim_surface_native_op_count")
        or true_sc_claim_surface_inventory.get("in_claim_surface")
        or 0
    )
    surface_governed_count = int(
        estimator_summary.get("true_sc_claim_surface_governed_support_op_count") or 0
    )
    support_out_count = int(
        estimator_summary.get("true_sc_support_out_of_surface_op_count")
        or true_sc_claim_surface_inventory.get("support_out_of_claim_surface")
        or 0
    )
    out_of_claim_count = int(
        estimator_summary.get("true_sc_out_of_claim_surface_op_count")
        or true_sc_claim_surface_inventory.get("out_of_claim_surface")
        or 0
    )
    true_sc_claim_surface_status = str(
        estimator_summary.get("true_sc_claim_surface_status") or ""
    ).strip()
    if not true_sc_claim_surface_status:
        if surface_governed_count > 0:
            true_sc_claim_surface_status = "claim_surface_blocked_by_governed_support"
        elif surface_native_count <= 0:
            true_sc_claim_surface_status = "no_true_sc_claim_surface"
        elif support_out_count > 0 or out_of_claim_count > 0:
            true_sc_claim_surface_status = "limited_true_sc_surface_with_out_of_surface_support"
        elif not workload_native_claim_eligible:
            true_sc_claim_surface_status = "claim_surface_blocked_by_workload_fidelity"
        else:
            true_sc_claim_surface_status = "full_true_sc_claim_surface"
    return {
        "true_sc_summary_claim_state": true_sc_summary_claim_state,
        "true_sc_claim_state_inventory": json.dumps(
            true_sc_claim_state_inventory,
            ensure_ascii=False,
            sort_keys=True,
        ),
        "true_sc_claim_surface_status": true_sc_claim_surface_status,
        "true_sc_claim_surface_inventory": json.dumps(
            true_sc_claim_surface_inventory,
            ensure_ascii=False,
            sort_keys=True,
        ),
        "true_sc_claim_surface_native_op_count": surface_native_count,
        "true_sc_claim_surface_governed_support_op_count": surface_governed_count,
        "true_sc_support_out_of_surface_op_count": support_out_count,
        "true_sc_out_of_claim_surface_op_count": out_of_claim_count,
        "true_sc_native_op_count": int(
            estimator_summary.get("true_sc_native_op_count")
            or true_sc_claim_state_inventory.get("true_sc_native")
            or 0
        ),
        "true_sc_governed_not_true_sc_op_count": int(
            estimator_summary.get("true_sc_governed_not_true_sc_op_count")
            or true_sc_claim_state_inventory.get("governed_support_not_true_sc")
            or 0
        ),
        "true_sc_out_of_surface_op_count": int(
            estimator_summary.get("true_sc_out_of_surface_op_count")
            or true_sc_claim_state_inventory.get("true_sc_out_of_surface")
            or 0
        ),
    }


def _derive_sc_default_status(
    *,
    trust_posture: str,
    fail_mode: str,
    dark_launch_enabled: bool,
) -> tuple[str, bool]:
    if trust_posture == "trusted_default":
        return "trusted_default", False
    if trust_posture == "default_with_supporting_assumptions":
        return "default_with_supporting_assumptions", False
    if fail_mode == "fail_closed_to_comparator":
        return ("comparator_only" if dark_launch_enabled else "out_of_band_no_comparator"), True
    return "out_of_band_emitted", False


def _resolve_switch_gated_evidence_surface(
    *,
    enabled: bool,
    cfg: dict[str, Any],
    default_evidence: str = "heuristic_proxy",
) -> tuple[str, str]:
    if not enabled:
        return "disabled", ""
    return (
        _canonical_evidence_type(cfg.get("evidence_type"), default=default_evidence),
        str(cfg.get("calibration_source") or ""),
    )


def _default_flow_diagnostics() -> dict[str, Any]:
    return {
        "flow_admission_stalls": 0,
        "flow_prefetch_hits": 0,
        "flow_prefetch_drops": 0,
        "flow_residency_hit_rate": 0.0,
        "flow_control_backpressure": 0.0,
        "flow_eviction_count": 0,
    }


def _resolve_flow_summary_fields(
    *,
    flow_cfg: dict[str, Any],
    flow_enabled: bool,
    flow_buffer_peak_cycles: int,
    flow_buffer_peak_frac: float,
    diagnostics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    scheduler_controls = _resolve_flow_scheduler_controls(
        flow_cfg=flow_cfg,
        flow_enabled=flow_enabled,
    )
    flow_diagnostics = _default_flow_diagnostics()
    if diagnostics:
        flow_diagnostics.update(diagnostics)
    if not flow_enabled:
        return {
            "flow_model_mode": "disabled",
            "flow_buffer_depth": 0,
            "flow_overlap_efficiency": 0.0,
            "flow_staging_cost_scale": 0.0,
            "flow_sync_penalty_scale": 0.0,
            "flow_reuse_policy": "disabled",
            "flow_prefetch_window": 0,
            "flow_control_group_size": 0,
            "flow_effective_overlap": 0.0,
            "flow_reuse_gain": 0.0,
            "flow_control_relief": 0.0,
            "flow_buffer_peak_cycles": 0,
            "flow_buffer_peak_frac": 0.0,
            "flow_tile_rows": 0,
            "flow_tile_cols": 0,
            "flow_prefetch_credits": 0,
            "flow_execute_credits": 0,
            "flow_control_issue_width": 0,
            "flow_admission_policy": "disabled",
            "flow_eviction_policy": "disabled",
            "flow_service_policy": "disabled",
            "flow_reuse_residency_budget": 0,
            "flow_broadcast_stability_window": 0,
            "flow_prefetch_distance": 0,
            "flow_exception_lane_policy": "disabled",
            **flow_diagnostics,
        }
    return {
        "flow_model_mode": scheduler_controls["scheduler_mode"],
        "flow_buffer_depth": scheduler_controls["buffer_depth"],
        "flow_overlap_efficiency": scheduler_controls["overlap_efficiency"],
        "flow_staging_cost_scale": scheduler_controls["staging_cost_scale"],
        "flow_sync_penalty_scale": scheduler_controls["sync_penalty_scale"],
        "flow_reuse_policy": scheduler_controls["reuse_policy"],
        "flow_prefetch_window": scheduler_controls["prefetch_window"],
        "flow_control_group_size": scheduler_controls["control_group_size"],
        "flow_effective_overlap": scheduler_controls["effective_overlap"],
        "flow_reuse_gain": scheduler_controls["reuse_gain"],
        "flow_control_relief": scheduler_controls["control_relief"],
        "flow_buffer_peak_cycles": flow_buffer_peak_cycles,
        "flow_buffer_peak_frac": flow_buffer_peak_frac,
        "flow_tile_rows": scheduler_controls["tile_rows"],
        "flow_tile_cols": scheduler_controls["tile_cols"],
        "flow_prefetch_credits": scheduler_controls["prefetch_credits"],
        "flow_execute_credits": scheduler_controls["execute_credits"],
        "flow_control_issue_width": scheduler_controls["control_issue_width"],
        "flow_admission_policy": scheduler_controls["admission_policy"],
        "flow_eviction_policy": scheduler_controls["eviction_policy"],
        "flow_service_policy": scheduler_controls["service_policy"],
        "flow_reuse_residency_budget": scheduler_controls["reuse_residency_budget"],
        "flow_broadcast_stability_window": scheduler_controls["broadcast_stability_window"],
        "flow_prefetch_distance": scheduler_controls["prefetch_distance"],
        "flow_exception_lane_policy": scheduler_controls["exception_lane_policy"],
        **flow_diagnostics,
    }


def _build_model_governance_summary(
    summaries: list[dict[str, Any]],
    dark_launch_rows: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Preserve per-model governance semantics in run metadata."""
    dark_launch_by_model: dict[str, dict[str, Any]] = {}
    for row in dark_launch_rows:
        model = str(row.get("model") or "").strip()
        if not model:
            continue
        dark_launch_by_model[model] = {
            "dark_launch_candidate_label": row.get("candidate_label"),
            "dark_launch_comparator_label": row.get("comparator_label"),
            "dark_launch_comparator_execution_semantics": row.get(
                "comparator_execution_semantics"
            ),
            "dark_launch_comparator_trust_posture": row.get(
                "comparator_summary_trust_posture"
            ),
            "dark_launch_coverage_delta_rows": row.get("coverage_delta_rows"),
            "dark_launch_compatibility_status": row.get("compatibility_status"),
        }

    model_summary: dict[str, dict[str, Any]] = {}
    for row in summaries:
        model = str(row.get("model") or "").strip()
        if not model:
            continue
        model_summary[model] = {
            "execution_semantics": row.get("execution_semantics"),
            "execution_semantics_default": row.get("execution_semantics_default"),
            "execution_semantics_origin": row.get("execution_semantics_origin"),
            "bitstream_enabled": row.get("bitstream_enabled"),
            "bitstream_generator": row.get("bitstream_generator"),
            "bitstream_stream_length": row.get("bitstream_stream_length"),
            "generator_stream_state_policy": row.get("generator_stream_state_policy"),
            "runtime_stream_reuse_policy": row.get("runtime_stream_reuse_policy"),
            "fidelity_ready": row.get("fidelity_ready"),
            "fidelity_blockers": row.get("fidelity_blockers"),
            "sc_summary_trust_posture": row.get("sc_summary_trust_posture"),
            "true_sc_summary_claim_state": row.get("true_sc_summary_claim_state"),
            "true_sc_claim_state_inventory": row.get("true_sc_claim_state_inventory"),
            "true_sc_claim_surface_status": row.get("true_sc_claim_surface_status"),
            "true_sc_claim_surface_inventory": row.get("true_sc_claim_surface_inventory"),
            "true_sc_claim_surface_native_op_count": row.get(
                "true_sc_claim_surface_native_op_count"
            ),
            "true_sc_claim_surface_governed_support_op_count": row.get(
                "true_sc_claim_surface_governed_support_op_count"
            ),
            "true_sc_support_out_of_surface_op_count": row.get(
                "true_sc_support_out_of_surface_op_count"
            ),
            "true_sc_out_of_claim_surface_op_count": row.get(
                "true_sc_out_of_claim_surface_op_count"
            ),
            "true_sc_native_op_count": row.get("true_sc_native_op_count"),
            "true_sc_governed_not_true_sc_op_count": row.get(
                "true_sc_governed_not_true_sc_op_count"
            ),
            "true_sc_out_of_surface_op_count": row.get("true_sc_out_of_surface_op_count"),
            "sc_calibration_state": row.get("sc_calibration_state"),
            "sc_generator_policy_status": row.get("sc_generator_policy_status"),
            "sc_generator_policy_reason": row.get("sc_generator_policy_reason"),
            "sc_default_status": row.get("sc_default_status"),
            "sc_support_classes": row.get("sc_support_classes"),
            "estimation_model_coverage_status": row.get(
                "estimation_model_coverage_status"
            ),
            "estimation_model_coverage_reason": row.get(
                "estimation_model_coverage_reason"
            ),
            "estimation_model_support_boundary": row.get(
                "estimation_model_support_boundary"
            ),
            "estimation_model_supported_op_count": row.get(
                "estimation_model_supported_op_count"
            ),
            "estimation_model_unsupported_op_count": row.get(
                "estimation_model_unsupported_op_count"
            ),
            "estimation_model_ready_status": row.get(
                "estimation_model_ready_status"
            ),
            "estimation_model_ready": row.get("estimation_model_ready"),
            "estimation_model_ready_reason": row.get(
                "estimation_model_ready_reason"
            ),
            "estimation_model_ready_blockers": row.get(
                "estimation_model_ready_blockers"
            ),
            "accuracy_measurement_contract_status": row.get(
                "accuracy_measurement_contract_status"
            ),
            "accuracy_measurement_contract_reason": row.get(
                "accuracy_measurement_contract_reason"
            ),
            "accuracy_measurement_contract_source": row.get(
                "accuracy_measurement_contract_source"
            ),
            "accuracy_measurement_contract_truth_class": row.get(
                "accuracy_measurement_contract_truth_class"
            ),
            "accuracy_measurement_contract_authorization_note": row.get(
                "accuracy_measurement_contract_authorization_note"
            ),
            "accuracy_measurement_contract_authorization_status": row.get(
                "accuracy_measurement_contract_authorization_status"
            ),
            "accuracy_measurement_contract_required_truth_class": row.get(
                "accuracy_measurement_contract_required_truth_class"
            ),
            "accuracy_measurement_contract_required_fields_json": row.get(
                "accuracy_measurement_contract_required_fields_json"
            ),
            "accuracy_measurement_contract_observed_fields_json": row.get(
                "accuracy_measurement_contract_observed_fields_json"
            ),
            "accuracy_measurement_contract_violations_json": row.get(
                "accuracy_measurement_contract_violations_json"
            ),
            "accuracy_evidence_tier": row.get("accuracy_evidence_tier"),
            "analysis_grade_ready": row.get("analysis_grade_ready"),
            "analysis_grade_blockers": row.get("analysis_grade_blockers"),
        }
        model_summary[model].update(dark_launch_by_model.get(model, {}))
    return model_summary


def _resolve_reporting_boundaries(execution_semantics: str) -> dict[str, str]:
    semantics = str(execution_semantics or "proxy").strip().lower()
    if semantics == "bitstream":
        return {
            "comparison_boundary": "sc_default_candidate_core_plus_governed_support_overheads",
            "latency_boundary": "sc_default_candidate_core_plus_governed_support_latency",
            "energy_boundary": "sc_default_candidate_core_plus_governed_support_energy",
            "power_boundary": "derived_system_energy_over_system_latency",
        }
    return {
        "comparison_boundary": "modeled_endpoint_core_plus_proxy_system_overheads",
        "latency_boundary": "core_timeline_plus_proxy_hidden_system_latency",
        "energy_boundary": "core_energy_plus_proxy_hidden_system_energy",
        "power_boundary": "derived_system_energy_over_system_latency",
    }


def _resolve_integrated_system_cost_cfg(raw_cfg: dict[str, Any] | None) -> dict[str, Any]:
    cfg = dict(DEFAULT_INTEGRATED_SYSTEM_COSTS)
    if not isinstance(raw_cfg, dict):
        cfg["mode"] = "disabled"
        return cfg
    cfg.update(raw_cfg)
    enabled = raw_cfg.get("enabled")
    if enabled is not None and not _to_bool(enabled, True):
        cfg["mode"] = "disabled"
    return cfg


def _compute_integrated_system_costs(
    *,
    conversion_control_j: float,
    memory_move_j: float,
    thermal_energy_j: float,
    core_latency_ms: float = 0.0,
    stage_cycles_payload: str | None = None,
    sample_rate_gsps: float = 1.0,
    flow_enabled: bool,
    phy_enabled: bool,
    noise_enabled: bool,
    cost_cfg: dict[str, Any] | None,
) -> dict[str, float | str]:
    resolved_cfg = _resolve_integrated_system_cost_cfg(cost_cfg)
    mode = str(resolved_cfg.get("mode") or "disabled").strip() or "disabled"
    if mode.lower() == "disabled":
        return {
            "integrated_onchip_comm_j": 0.0,
            "integrated_control_sched_j": 0.0,
            "integrated_host_staging_j": 0.0,
            "integrated_calibration_monitoring_j": 0.0,
            "integrated_hidden_system_cost_j": 0.0,
            "integrated_hidden_system_cost_lower_j": 0.0,
            "integrated_hidden_system_cost_upper_j": 0.0,
            "integrated_onchip_comm_ms": 0.0,
            "integrated_control_sched_ms": 0.0,
            "integrated_host_staging_ms": 0.0,
            "integrated_calibration_monitoring_ms": 0.0,
            "integrated_hidden_system_latency_ms": 0.0,
            "integrated_hidden_system_latency_lower_ms": 0.0,
            "integrated_hidden_system_latency_upper_ms": 0.0,
            "integrated_system_cost_mode": "disabled",
        }

    def _cycles_to_ms(stage_names: tuple[str, ...]) -> float:
        if not stage_cycles_payload or sample_rate_gsps <= 0:
            return 0.0
        try:
            stage_cycles = json.loads(stage_cycles_payload)
        except (TypeError, ValueError, json.JSONDecodeError):
            return 0.0
        total_cycles = 0
        for stage_name in stage_names:
            total_cycles += int(_to_float(stage_cycles.get(stage_name), 0.0) or 0)
        return total_cycles / (sample_rate_gsps * 1e9) * 1e3

    total_proxy_j = max(conversion_control_j + memory_move_j + thermal_energy_j, 1e-12)
    memory_share = memory_move_j / total_proxy_j
    conversion_share = conversion_control_j / total_proxy_j
    io_latency_ms = _cycles_to_ms(("fetch_map", "writeback"))
    control_latency_ms = _cycles_to_ms(("btos", "serialize_drive", "pca_adc"))
    io_latency_base_ms = (
        io_latency_ms
        if io_latency_ms > 0
        else core_latency_ms * _clamp(memory_share, 0.10, 0.80)
    )
    control_latency_base_ms = (
        control_latency_ms
        if control_latency_ms > 0
        else core_latency_ms * _clamp(conversion_share, 0.05, 0.60)
    )

    onchip = max(
        0.0,
        memory_move_j
        * max(0.0, _to_float(resolved_cfg.get("onchip_scale_vs_move"), 0.0) or 0.0),
    )
    onchip_ms = max(
        0.0,
        io_latency_base_ms
        * max(0.0, _to_float(resolved_cfg.get("onchip_latency_scale_vs_io"), 0.0) or 0.0),
    )
    control_sched = max(
        0.0,
        conversion_control_j
        * max(0.0, _to_float(resolved_cfg.get("control_sched_scale_vs_conv"), 0.0) or 0.0),
    )
    control_sched_ms = max(
        0.0,
        control_latency_base_ms
        * max(
            0.0,
            _to_float(resolved_cfg.get("control_sched_latency_scale_vs_control"), 0.0) or 0.0,
        ),
    )

    host_staging_enabled = flow_enabled or not _to_bool(
        resolved_cfg.get("host_staging_flow_only"),
        True,
    )
    host_staging = (
        max(
            0.0,
            conversion_control_j
            * max(0.0, _to_float(resolved_cfg.get("host_staging_scale_vs_conv"), 0.0) or 0.0),
        )
        if host_staging_enabled
        else 0.0
    )
    host_staging_ms = (
        max(
            0.0,
            io_latency_base_ms
            * max(0.0, _to_float(resolved_cfg.get("host_staging_latency_scale_vs_io"), 0.0) or 0.0),
        )
        if host_staging_enabled
        else 0.0
    )

    calibration_gate = phy_enabled or noise_enabled or not _to_bool(
        resolved_cfg.get("calibration_monitoring_phy_or_noise_only"),
        True,
    )
    calibration_monitoring = (
        max(
            0.0,
            thermal_energy_j
            * max(
                0.0,
                _to_float(resolved_cfg.get("calibration_monitoring_scale_vs_thermal"), 0.0)
                or 0.0,
            ),
        )
        if calibration_gate
        else 0.0
    )
    calibration_monitoring_ms = (
        max(
            0.0,
            core_latency_ms
            * max(
                0.0,
                _to_float(
                    resolved_cfg.get("calibration_monitoring_latency_scale_vs_core"),
                    0.0,
                )
                or 0.0,
            ),
        )
        if calibration_gate
        else 0.0
    )

    total = onchip + control_sched + host_staging + calibration_monitoring
    total_ms = onchip_ms + control_sched_ms + host_staging_ms + calibration_monitoring_ms
    lower_scale = max(
        0.0,
        _to_float(resolved_cfg.get("uncertainty_lower_scale"), 0.75) or 0.75,
    )
    upper_scale = max(
        lower_scale,
        _to_float(resolved_cfg.get("uncertainty_upper_scale"), 1.25) or 1.25,
    )
    return {
        "integrated_onchip_comm_j": onchip,
        "integrated_control_sched_j": control_sched,
        "integrated_host_staging_j": host_staging,
        "integrated_calibration_monitoring_j": calibration_monitoring,
        "integrated_hidden_system_cost_j": total,
        "integrated_hidden_system_cost_lower_j": total * lower_scale,
        "integrated_hidden_system_cost_upper_j": total * upper_scale,
        "integrated_onchip_comm_ms": onchip_ms,
        "integrated_control_sched_ms": control_sched_ms,
        "integrated_host_staging_ms": host_staging_ms,
        "integrated_calibration_monitoring_ms": calibration_monitoring_ms,
        "integrated_hidden_system_latency_ms": total_ms,
        "integrated_hidden_system_latency_lower_ms": total_ms * lower_scale,
        "integrated_hidden_system_latency_upper_ms": total_ms * upper_scale,
        "integrated_system_cost_mode": mode,
    }


def _build_realism_assessment(
    *,
    integrated_system_cost_cfg: dict[str, Any],
    integrated_system_cost_mode: str,
    flow_cfg: dict[str, Any],
    meso_cfg: dict[str, Any],
    phy_cfg: dict[str, Any],
    realism_cfg: dict[str, Any],
    accuracy_source_csv: str | None,
    accuracy_provenance: dict[str, Any] | None = None,
    switches: dict[str, bool],
    accuracy_coupling_metadata: dict[str, Any] | None = None,
    accuracy_measurement_contract_metadata: dict[str, Any] | None = None,
    workload_fidelity_metadata: dict[str, Any] | None = None,
    fidelity_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    integrated_onchip_comm_evidence = _canonical_evidence_type(
        integrated_system_cost_cfg.get("onchip_comm_evidence_type"),
        default="heuristic_proxy",
    )
    integrated_control_sched_evidence = _canonical_evidence_type(
        integrated_system_cost_cfg.get("control_sched_evidence_type"),
        default="heuristic_proxy",
    )
    integrated_host_staging_evidence = _canonical_evidence_type(
        integrated_system_cost_cfg.get("host_staging_evidence_type"),
        default="heuristic_proxy",
    )
    integrated_calibration_monitoring_evidence = _canonical_evidence_type(
        integrated_system_cost_cfg.get("calibration_monitoring_evidence_type"),
        default="heuristic_proxy",
    )
    integrated_system_cost_evidence = _worst_evidence_type(
        [
            integrated_onchip_comm_evidence,
            integrated_control_sched_evidence,
            integrated_host_staging_evidence,
            integrated_calibration_monitoring_evidence,
        ],
        default="missing",
    )
    flow_timeline_evidence, flow_timeline_calibration_source = _resolve_switch_gated_evidence_surface(
        enabled=bool(switches.get("flow")),
        cfg=flow_cfg,
    )
    meso_cost_evidence, meso_cost_calibration_source = _resolve_switch_gated_evidence_surface(
        enabled=bool(switches.get("meso")),
        cfg=meso_cfg,
    )
    phy_support_evidence, phy_support_calibration_source = _resolve_switch_gated_evidence_surface(
        enabled=bool(switches.get("phy")),
        cfg=phy_cfg,
    )
    resolved_accuracy_coupling = accuracy_coupling_metadata or {}
    accuracy_coupling_evidence = _canonical_evidence_type(
        resolved_accuracy_coupling.get("accuracy_coupling_evidence"),
        default=("measured" if str(accuracy_source_csv or "").strip() else "missing"),
    )
    accuracy_coupling_metric = str(
        resolved_accuracy_coupling.get("accuracy_coupling_metric") or ""
    ).strip()
    accuracy_coupling_source = str(
        resolved_accuracy_coupling.get("accuracy_coupling_source")
        or accuracy_source_csv
        or ""
    ).strip()
    accuracy_coupling_reason = str(
        resolved_accuracy_coupling.get("accuracy_coupling_reason") or ""
    ).strip()
    resolved_accuracy_contract = accuracy_measurement_contract_metadata or {}
    accuracy_measurement_contract_status = str(
        resolved_accuracy_contract.get("accuracy_measurement_contract_status") or ""
    ).strip()
    accuracy_measurement_contract_reason = str(
        resolved_accuracy_contract.get("accuracy_measurement_contract_reason") or ""
    ).strip()
    accuracy_measurement_contract_source = str(
        resolved_accuracy_contract.get("accuracy_measurement_contract_source") or ""
    ).strip()
    accuracy_measurement_contract_truth_class = str(
        resolved_accuracy_contract.get("accuracy_measurement_contract_truth_class") or ""
    ).strip()
    accuracy_measurement_contract_authorization_note = str(
        resolved_accuracy_contract.get("accuracy_measurement_contract_authorization_note") or ""
    ).strip()
    accuracy_measurement_contract_authorization_status = str(
        resolved_accuracy_contract.get("accuracy_measurement_contract_authorization_status") or ""
    ).strip()
    accuracy_measurement_contract_conv_measured_package_path = str(
        resolved_accuracy_contract.get("accuracy_measurement_contract_conv_measured_package_path")
        or ""
    ).strip()
    accuracy_measurement_contract_conv_measured_package_sha256 = str(
        resolved_accuracy_contract.get("accuracy_measurement_contract_conv_measured_package_sha256")
        or ""
    ).strip()
    accuracy_measurement_contract_required_truth_class = str(
        resolved_accuracy_contract.get("accuracy_measurement_contract_required_truth_class")
        or ""
    ).strip()
    accuracy_measurement_contract_required_fields_json = str(
        resolved_accuracy_contract.get("accuracy_measurement_contract_required_fields_json")
        or ""
    ).strip()
    accuracy_measurement_contract_observed_fields_json = str(
        resolved_accuracy_contract.get("accuracy_measurement_contract_observed_fields_json")
        or ""
    ).strip()
    accuracy_measurement_contract_violations_json = str(
        resolved_accuracy_contract.get("accuracy_measurement_contract_violations_json")
        or ""
    ).strip()
    accuracy_evidence_tier = _resolve_accuracy_evidence_tier(
        accuracy_provenance=accuracy_provenance or {},
        accuracy_measurement_contract_metadata=resolved_accuracy_contract,
    )
    analysis_grade_ready = _to_bool(
        (accuracy_provenance or {}).get("accuracy_target_analysis_grade_ready"),
        False,
    )
    analysis_grade_blockers = _parse_analysis_grade_blockers(
        (accuracy_provenance or {}).get("accuracy_target_analysis_grade_blockers")
    )
    if accuracy_evidence_tier != ANALYSIS_GRADE_TIER:
        analysis_grade_ready = False
        if accuracy_evidence_tier == RUNTIME_SMOKE_TIER:
            if "runtime_smoke_only" not in analysis_grade_blockers:
                analysis_grade_blockers.insert(0, "runtime_smoke_only")
        elif "missing_accuracy_evidence_tier" not in analysis_grade_blockers:
            analysis_grade_blockers.insert(0, "missing_accuracy_evidence_tier")
    elif not analysis_grade_ready and "analysis_grade_row_not_ready" not in analysis_grade_blockers:
        analysis_grade_blockers.append("analysis_grade_row_not_ready")
    target_class = str(realism_cfg.get("target_class") or "realistic_accelerator_proxy").strip()
    device_comparison_scope = str(
        realism_cfg.get("device_comparison_scope") or "contextual_only"
    ).strip() or "contextual_only"
    benchmark_equivalence = _to_bool(
        realism_cfg.get("benchmark_equivalence"),
        False,
    )
    benchmark_claim_eligible = benchmark_equivalence and device_comparison_scope in {
        "benchmark_equivalent",
        "strict_benchmark_equivalent",
    }
    blockers: list[str] = []
    if str(integrated_system_cost_mode).strip().lower() == "disabled":
        blockers.append("hidden_system_costs_missing")
    elif not _evidence_is_at_least(integrated_system_cost_evidence, "calibrated_model"):
        blockers.append("hidden_system_costs_not_calibrated")
    if switches.get("flow") and not _evidence_is_at_least(flow_timeline_evidence, "calibrated_model"):
        blockers.append("flow_timeline_not_calibrated")
    if switches.get("meso") and not _evidence_is_at_least(meso_cost_evidence, "calibrated_model"):
        blockers.append("meso_cost_surface_not_calibrated")
    if switches.get("phy") and not _evidence_is_at_least(phy_support_evidence, "calibrated_model"):
        blockers.append("phy_envelope_not_calibrated")
    if accuracy_coupling_evidence == "missing":
        blockers.append("accuracy_coupling_missing")
    elif accuracy_coupling_evidence != "measured":
        blockers.append("accuracy_coupling_not_measured")
    resolved_fidelity = fidelity_metadata or {}
    if not _to_bool(resolved_fidelity.get("fidelity_ready"), True):
        raw_fidelity_blockers = resolved_fidelity.get("fidelity_blockers") or "[]"
        try:
            fidelity_blockers = json.loads(raw_fidelity_blockers)
        except (TypeError, ValueError, json.JSONDecodeError):
            fidelity_blockers = []
        for blocker in fidelity_blockers:
            blocker_text = str(blocker).strip()
            if blocker_text and blocker_text not in blockers:
                blockers.append(blocker_text)

    realism_class = (
        "idealized_estimator"
        if str(integrated_system_cost_mode).strip().lower() == "disabled"
        else ("realistic_accelerator_proxy" if not blockers else "bounded_system_model")
    )
    return {
        "target_class": target_class,
        "integrated_onchip_comm_evidence": integrated_onchip_comm_evidence,
        "integrated_control_sched_evidence": integrated_control_sched_evidence,
        "integrated_host_staging_evidence": integrated_host_staging_evidence,
        "integrated_calibration_monitoring_evidence": integrated_calibration_monitoring_evidence,
        "integrated_system_cost_evidence": integrated_system_cost_evidence,
        "integrated_system_cost_calibration_source": str(
            integrated_system_cost_cfg.get("calibration_source") or ""
        ),
        "integrated_system_cost_uncertainty_method": str(
            integrated_system_cost_cfg.get("uncertainty_method") or ""
        ),
        "flow_timeline_evidence": flow_timeline_evidence,
        "flow_timeline_calibration_source": flow_timeline_calibration_source,
        "meso_cost_evidence": meso_cost_evidence,
        "meso_cost_calibration_source": meso_cost_calibration_source,
        "phy_support_evidence": phy_support_evidence,
        "phy_support_calibration_source": phy_support_calibration_source,
        "accuracy_coupling_evidence": accuracy_coupling_evidence,
        "accuracy_coupling_metric": accuracy_coupling_metric,
        "accuracy_coupling_source": accuracy_coupling_source,
        "accuracy_coupling_reason": accuracy_coupling_reason,
        "accuracy_measurement_contract_status": accuracy_measurement_contract_status,
        "accuracy_measurement_contract_reason": accuracy_measurement_contract_reason,
        "accuracy_measurement_contract_source": accuracy_measurement_contract_source,
        "accuracy_measurement_contract_truth_class": accuracy_measurement_contract_truth_class,
        "accuracy_measurement_contract_authorization_note": (
            accuracy_measurement_contract_authorization_note
        ),
        "accuracy_measurement_contract_authorization_status": (
            accuracy_measurement_contract_authorization_status
        ),
        "accuracy_measurement_contract_conv_measured_package_path": (
            accuracy_measurement_contract_conv_measured_package_path
        ),
        "accuracy_measurement_contract_conv_measured_package_sha256": (
            accuracy_measurement_contract_conv_measured_package_sha256
        ),
        "accuracy_measurement_contract_required_truth_class": (
            accuracy_measurement_contract_required_truth_class
        ),
        "accuracy_measurement_contract_required_fields_json": (
            accuracy_measurement_contract_required_fields_json
        ),
        "accuracy_measurement_contract_observed_fields_json": (
            accuracy_measurement_contract_observed_fields_json
        ),
        "accuracy_measurement_contract_violations_json": (
            accuracy_measurement_contract_violations_json
        ),
        "accuracy_evidence_tier": accuracy_evidence_tier,
        "analysis_grade_ready": analysis_grade_ready,
        "analysis_grade_blockers": json.dumps(
            analysis_grade_blockers,
            ensure_ascii=False,
        ),
        "fidelity_ready": _to_bool(resolved_fidelity.get("fidelity_ready"), not blockers),
        "fidelity_blockers": str(resolved_fidelity.get("fidelity_blockers") or "[]"),
        "conv_fidelity_stage": str(resolved_fidelity.get("conv_fidelity_stage") or ""),
        "conv_fidelity_blockers": str(
            resolved_fidelity.get("conv_fidelity_blockers") or "[]"
        ),
        "conv_evidence_manifest_path": str(
            resolved_fidelity.get("conv_evidence_manifest_path") or ""
        ),
        "conv_evidence_manifest_sha256": str(
            resolved_fidelity.get("conv_evidence_manifest_sha256") or ""
        ),
        "conv_measured_package_path": str(
            resolved_fidelity.get("conv_measured_package_path") or ""
        ),
        "conv_measured_package_sha256": str(
            resolved_fidelity.get("conv_measured_package_sha256") or ""
        ),
        "conv_measured_closure_status": str(
            resolved_fidelity.get("conv_measured_closure_status") or ""
        ),
        "realism_class": realism_class,
        "proxy_promotion_ready": not blockers and realism_class == target_class,
        "proxy_upgrade_blockers": ";".join(blockers),
        "benchmark_claim_ready": benchmark_claim_eligible
        and not blockers
        and realism_class == target_class,
        "device_comparison_scope": device_comparison_scope,
        "benchmark_equivalence": benchmark_equivalence,
    }


def _write_phy_sweep(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "model",
        "duty_cycle",
        "sparse_scale_source",
        "wdm_channels_n",
        "loss_path_db",
        "pp_crosstalk_db",
        "p_laser_dbm",
        "p_laser_mw",
        "phy_penalty_table_version",
    ]
    _write_csv(path, fields, rows)


def _estimate_total_ops(ops: list[dict[str, Any]]) -> float:
    total = 0.0
    for op in ops:
        op_type = str(op.get("type") or "gemm").lower()
        if op_type in ELEMENTWISE_TYPES:
            total += float(op.get("elements") or 0.0)
        else:
            m = float(op.get("m") or 0.0)
            d = float(op.get("d") or 0.0)
            n = float(op.get("n") or 0.0)
            total += 2.0 * m * d * n
    return total


def _serialize_layer_param(value: Any) -> str | float | int | None:
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    if isinstance(value, (int, float, str)):
        return value
    return None


def _per_layer_value(mapping: Any, layer_id: str, index: int, default: Any) -> Any:
    if isinstance(mapping, dict):
        if layer_id in mapping:
            return mapping[layer_id]
        if str(index) in mapping:
            return mapping[str(index)]
    return default


def _build_per_layer_accuracy_rows(
    *,
    model: str,
    scaled_ops: list[dict[str, Any]],
    effective_k: float,
    sc_det_cfg: dict[str, Any],
    sparse_cfg: dict[str, Any],
    accuracy_cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    tau_default = sparse_cfg.get("tau_global")
    tau_by_layer = sparse_cfg.get("tau_by_layer")
    k_by_layer = (sc_det_cfg.get("early_stop") or {}).get("k_by_layer")
    delta_by_layer = accuracy_cfg.get("per_layer_delta_pp") or {}
    rows = []
    for idx, op in enumerate(scaled_ops, start=1):
        layer_name = str(op.get("name") or f"layer_{idx}")
        layer_id = f"{model}:{layer_name}"
        rows.append(
            {
                "layer_id": layer_id,
                "k_i": _per_layer_value(k_by_layer, layer_id, idx, effective_k),
                "tau_i": _per_layer_value(tau_by_layer, layer_id, idx, tau_default),
                "delta_acc_layer_pp": _per_layer_value(
                    delta_by_layer, layer_id, idx, None
                ),
            }
        )
    return rows


def _split_cycles(total_cycles: int, stage_ratios: list[tuple[str, float]]) -> list[tuple[str, int]]:
    remaining = total_cycles
    rows: list[tuple[str, int]] = []
    for idx, (name, ratio) in enumerate(stage_ratios):
        if idx == len(stage_ratios) - 1:
            cycles = max(0, remaining)
        else:
            cycles = max(0, int(round(total_cycles * ratio)))
            cycles = min(cycles, remaining)
            remaining -= cycles
        rows.append((name, cycles))
    return rows


def _simulate_elastic_residency_v3(
    *,
    op: dict[str, Any],
    is_elementwise: bool,
    scheduler_controls: dict[str, Any],
) -> dict[str, Any]:
    tile_rows = max(1, int(scheduler_controls["tile_rows"]))
    tile_cols = max(1, int(scheduler_controls["tile_cols"]))
    tile_area = max(1, tile_rows * tile_cols)
    buffer_depth = max(1, int(scheduler_controls["buffer_depth"]))
    prefetch_credits = max(1, int(scheduler_controls["prefetch_credits"]))
    execute_credits = max(1, int(scheduler_controls["execute_credits"]))
    control_issue_width = max(1, int(scheduler_controls["control_issue_width"]))
    reuse_residency_budget = max(1, int(scheduler_controls["reuse_residency_budget"]))
    prefetch_distance = max(1, int(scheduler_controls["prefetch_distance"]))
    broadcast_stability_window = max(1, int(scheduler_controls["broadcast_stability_window"]))
    reuse_policy = str(scheduler_controls["reuse_policy"])
    admission_policy = str(scheduler_controls["admission_policy"])
    eviction_policy = str(scheduler_controls["eviction_policy"])
    service_policy = str(scheduler_controls["service_policy"])
    exception_lane_policy = str(scheduler_controls["exception_lane_policy"])
    reuse_gain = float(scheduler_controls["reuse_gain"])
    control_relief = float(scheduler_controls["control_relief"])
    overlap_hidden = float(scheduler_controls["effective_overlap"])

    if is_elementwise:
        e_compute = _to_float(op.get("energy_mj_elementwise"), None)
        if e_compute is None:
            e_compute = _to_float(op.get("energy_mj"), 0.0) or 0.0
        e_mem = max(0.0, _to_float(op.get("energy_mj_mem"), 0.0) or 0.0)
        total = max(1e-9, float(e_compute) + e_mem)
        mem_ratio = e_mem / total
        base_work_units = 32.0
    else:
        e_load = max(0.0, _to_float(op.get("energy_mj_load_x"), 0.0) or 0.0) + max(
            0.0, _to_float(op.get("energy_mj_load_y"), 0.0) or 0.0
        )
        e_mem = max(0.0, _to_float(op.get("energy_mj_mem"), 0.0) or 0.0)
        e_static = max(0.0, _to_float(op.get("energy_mj_static"), 0.0) or 0.0)
        e_oe = max(0.0, _to_float(op.get("energy_mj_oe"), 0.0) or 0.0)
        e_adc = max(0.0, _to_float(op.get("energy_mj_adc_pca"), 0.0) or 0.0)
        e_laser = max(0.0, _to_float(op.get("energy_mj_laser"), 0.0) or 0.0)
        total = max(1e-9, e_load + e_mem + e_static + e_oe + e_adc + e_laser)
        mem_ratio = (e_load + e_mem) / total
        base_work_units = 96.0

    logical_tiles = int(
        _clamp(
            math.ceil(base_work_units * (0.70 + 0.80 * mem_ratio) / tile_area),
            4,
            64,
        )
    )
    operand_span = max(
        1,
        min(
            logical_tiles,
            int(
                math.ceil(
                    logical_tiles
                    / max(1.0, 1.0 + prefetch_distance * (0.40 + 0.60 * reuse_gain))
                )
            ),
        ),
    )
    tiles = [
        {
            "tile_id": tile_id,
            "operand_id": tile_id % operand_span,
        }
        for tile_id in range(logical_tiles)
    ]
    if service_policy == "reuse_first" or admission_policy == "reuse_first":
        tiles.sort(key=lambda row: (row["operand_id"], row["tile_id"]))
    elif service_policy == "critical_path_first":
        tiles.sort(key=lambda row: (row["tile_id"] % max(1, broadcast_stability_window), row["tile_id"]))

    pending: deque[dict[str, int]] = deque(tiles)
    prefetch_queue: deque[dict[str, int]] = deque()
    execute_queue: deque[dict[str, int]] = deque()
    residency_queue: deque[int] = deque()
    residency_set: set[int] = set()

    rounds = 0
    completed = 0
    inflight_tiles = 0
    inflight_peak = 0
    admission_stalls = 0
    prefetch_hits = 0
    prefetch_drops = 0
    residency_hits = 0
    control_backpressure_events = 0
    eviction_count = 0
    max_rounds = logical_tiles * max(6, prefetch_distance + execute_credits + control_issue_width) + 32

    while (pending or prefetch_queue or execute_queue) and rounds < max_rounds:
        rounds += 1

        retired = min(execute_credits, len(execute_queue))
        for _ in range(retired):
            execute_queue.popleft()
            completed += 1
            inflight_tiles = max(0, inflight_tiles - 1)

        advanced_prefetch: deque[dict[str, int]] = deque()
        while prefetch_queue:
            item = prefetch_queue.popleft()
            item["ready_in"] -= 1
            if item["ready_in"] <= 0:
                execute_queue.append(item)
            else:
                advanced_prefetch.append(item)
        prefetch_queue = advanced_prefetch

        issue_budget = control_issue_width
        if admission_policy == "conservative":
            issue_budget = max(1, control_issue_width - 1)
        elif admission_policy == "greedy":
            issue_budget = control_issue_width + 1

        issued = 0
        stalled_this_round = False
        while pending and issued < issue_budget:
            next_tile = pending[0]
            operand_id = int(next_tile["operand_id"])
            residency_hit = reuse_policy != "none" and operand_id in residency_set
            buffer_limit = buffer_depth + (1 if admission_policy == "greedy" else 0)
            if exception_lane_policy == "spill":
                buffer_limit += max(0, prefetch_credits - 1)
            if residency_hit:
                buffer_limit += 1

            direct_execute_ok = residency_hit and len(execute_queue) < max(1, execute_credits * 2)
            prefetch_ok = len(prefetch_queue) < prefetch_credits
            if inflight_tiles >= buffer_limit or (not prefetch_ok and not direct_execute_ok):
                admission_stalls += 1
                stalled_this_round = True
                break

            pending.popleft()
            issued += 1
            inflight_tiles += 1
            inflight_peak = max(inflight_peak, inflight_tiles)

            if residency_hit:
                residency_hits += 1
            elif reuse_policy != "none":
                if len(residency_queue) >= reuse_residency_budget and residency_queue:
                    evicted = residency_queue.popleft()
                    residency_set.discard(evicted)
                    eviction_count += 1
                if len(residency_queue) < reuse_residency_budget:
                    residency_queue.append(operand_id)
                    residency_set.add(operand_id)

            ready_latency = 1 if residency_hit else prefetch_distance
            if not residency_hit:
                ready_latency = max(
                    1,
                    ready_latency - min(broadcast_stability_window - 1, int(round(control_relief * 2.0))),
                )

            if direct_execute_ok:
                execute_queue.append({"operand_id": operand_id, "ready_in": 0})
                prefetch_hits += 1
            elif prefetch_ok:
                prefetch_queue.append({"operand_id": operand_id, "ready_in": ready_latency})
                prefetch_hits += 1
            else:
                prefetch_drops += 1
                inflight_tiles = max(0, inflight_tiles - 1)
                stalled_this_round = True
                break

        if pending and (stalled_this_round or issued < min(issue_budget, len(pending))):
            control_backpressure_events += 1

    if pending or prefetch_queue or execute_queue:
        fallback_flush = len(pending) + len(prefetch_queue) + len(execute_queue)
        completed += max(0, len(execute_queue))
        prefetch_drops += max(0, len(pending))
        control_backpressure_events += 1 if fallback_flush > 0 else 0

    residency_hit_rate = residency_hits / max(1, logical_tiles)
    prefetch_success_rate = prefetch_hits / max(1, prefetch_hits + prefetch_drops)
    queue_efficiency = completed / max(1, rounds * execute_credits)
    admission_pressure = admission_stalls / max(1, logical_tiles)
    control_backpressure = control_backpressure_events / max(1, rounds)
    eviction_pressure = eviction_count / max(1, logical_tiles)
    overlap_boost = _clamp(
        0.45 * overlap_hidden
        + 0.20 * prefetch_success_rate
        + 0.20 * residency_hit_rate
        + 0.15 * queue_efficiency,
        0.0,
        1.0,
    )
    queue_pressure = _clamp(
        0.50 * admission_pressure + 0.35 * control_backpressure + 0.15 * eviction_pressure,
        0.0,
        1.0,
    )
    effective_buffer_frac = inflight_peak / max(1, buffer_depth + prefetch_credits)

    return {
        "logical_tiles": logical_tiles,
        "operand_span": operand_span,
        "admission_stalls": admission_stalls,
        "prefetch_hits": prefetch_hits,
        "prefetch_drops": prefetch_drops,
        "residency_hit_rate": residency_hit_rate,
        "control_backpressure": control_backpressure,
        "eviction_count": eviction_count,
        "prefetch_success_rate": prefetch_success_rate,
        "queue_efficiency": queue_efficiency,
        "admission_pressure": admission_pressure,
        "eviction_pressure": eviction_pressure,
        "overlap_boost": overlap_boost,
        "queue_pressure": queue_pressure,
        "effective_buffer_frac": effective_buffer_frac,
    }


def _component_weighted_stage_ratios(
    *,
    op: dict[str, Any],
    is_elementwise: bool,
    flow_enabled: bool,
    det_enabled: bool,
    flow_cfg: dict[str, Any],
) -> tuple[list[tuple[str, float]], float, float, dict[str, Any]]:
    """Build stage ratios from per-op component energy instead of fixed constants."""
    scheduler_controls = _resolve_flow_scheduler_controls(
        flow_cfg=flow_cfg,
        flow_enabled=flow_enabled,
    )
    if flow_enabled:
        staging_cost_scale = float(scheduler_controls["staging_cost_scale"])
        sync_penalty_scale = float(scheduler_controls["sync_penalty_scale"])
    else:
        staging_cost_scale = max(0.0, _to_float(flow_cfg.get("staging_cost_scale"), 1.0) or 1.0)
        sync_penalty_scale = max(0.0, _to_float(flow_cfg.get("sync_penalty_scale"), 1.0) or 1.0)
    scheduler_mode = str(scheduler_controls["scheduler_mode"])
    scheduler_state = dict(scheduler_controls)
    scheduler_state.update(
        {
            "admission_stalls": 0,
            "prefetch_hits": 0,
            "prefetch_drops": 0,
            "residency_hit_rate": 0.0,
            "control_backpressure": 0.0,
            "eviction_count": 0,
            "prefetch_success_rate": 0.0,
            "queue_efficiency": 0.0,
            "admission_pressure": 0.0,
            "eviction_pressure": 0.0,
            "queue_pressure": 0.0,
            "effective_buffer_frac": 0.0,
        }
    )
    if flow_enabled and scheduler_mode == FLOW_SCHEDULER_MODE_V3:
        scheduler_runtime = _simulate_elastic_residency_v3(
            op=op,
            is_elementwise=is_elementwise,
            scheduler_controls=scheduler_controls,
        )
        scheduler_state.update(scheduler_runtime)
        scheduler_state["effective_overlap"] = _clamp(
            0.55 * float(scheduler_controls["effective_overlap"])
            + 0.45 * float(scheduler_runtime["overlap_boost"]),
            0.0,
            1.0,
        )
        max_effective_buffer_depth = int(
            max(
                scheduler_controls["buffer_depth"],
                scheduler_controls["buffer_depth"] + scheduler_controls["prefetch_credits"],
            )
        )
        scheduler_state["effective_buffer_depth"] = int(
            _clamp(
                math.ceil(
                    max(
                        1.0,
                        (scheduler_controls["buffer_depth"] + scheduler_controls["prefetch_credits"])
                        * min(1.0, float(scheduler_runtime["effective_buffer_frac"]) + 0.15),
                    )
                ),
                scheduler_controls["buffer_depth"],
                max_effective_buffer_depth,
            )
        )
    overlap_hidden = float(scheduler_state["effective_overlap"])
    reuse_gain = float(scheduler_state["reuse_gain"])
    control_relief = float(scheduler_state["control_relief"])
    oversubscribe_penalty = float(scheduler_state["oversubscribe_penalty"])

    if is_elementwise:
        e_compute = _to_float(op.get("energy_mj_elementwise"), None)
        if e_compute is None:
            e_compute = _to_float(op.get("energy_mj"), 0.0) or 0.0
        e_compute = max(0.0, float(e_compute))
        e_mem = max(0.0, _to_float(op.get("energy_mj_mem"), 0.0) or 0.0)

        total = e_compute + e_mem
        mem_ratio = (e_mem / total) if total > 0 else 0.20
        base_bubble = _clamp((0.06 + 0.20 * mem_ratio) * sync_penalty_scale, 0.04, 0.34)
        if flow_enabled and scheduler_mode == FLOW_SCHEDULER_MODE_V3:
            queue_efficiency = float(scheduler_state["queue_efficiency"])
            admission_pressure = float(scheduler_state["admission_pressure"])
            control_backpressure = float(scheduler_state["control_backpressure"])
            residency_hit_rate = float(scheduler_state["residency_hit_rate"])
            prefetch_success_rate = float(scheduler_state["prefetch_success_rate"])
            queue_pressure = float(scheduler_state["queue_pressure"])
            bubble_scale = (
                1.0
                - 0.66 * overlap_hidden
                - 0.14 * queue_efficiency
                - 0.10 * residency_hit_rate
                - 0.08 * prefetch_success_rate
            )
            bubble_scale += 0.12 * queue_pressure + 0.08 * control_backpressure + 0.06 * oversubscribe_penalty
            bubble_ratio = _clamp(base_bubble * bubble_scale, 0.02, 0.24)
            fetch_scale = staging_cost_scale * max(
                0.68,
                1.0
                - 0.12 * residency_hit_rate
                - 0.08 * queue_efficiency
                + 0.10 * admission_pressure
                + 0.06 * control_backpressure,
            )
            writeback_scale = staging_cost_scale * max(
                0.74,
                1.0 - 0.04 * queue_efficiency - 0.04 * residency_hit_rate + 0.04 * control_backpressure,
            )
        elif flow_enabled and scheduler_mode == FLOW_SCHEDULER_MODE_V2:
            reuse_tail = reuse_gain * 0.45
            bubble_scale = 1.0 - 0.68 * overlap_hidden - 0.10 * reuse_tail - 0.10 * control_relief
            bubble_scale += 0.08 * oversubscribe_penalty
            bubble_ratio = _clamp(base_bubble * bubble_scale, 0.02, 0.22)
            fetch_scale = staging_cost_scale * max(
                0.72,
                1.0 - 0.08 * reuse_tail - 0.06 * control_relief + 0.05 * oversubscribe_penalty,
            )
            writeback_scale = staging_cost_scale * max(
                0.76,
                1.0 - 0.04 * reuse_tail - 0.06 * control_relief,
            )
        else:
            if flow_enabled:
                bubble_ratio = _clamp(base_bubble * (1.0 - 0.90 * overlap_hidden), 0.03, 0.24)
            else:
                bubble_ratio = _clamp(base_bubble, 0.06, 0.34)
            fetch_scale = staging_cost_scale if flow_enabled else 1.0
            writeback_scale = staging_cost_scale if flow_enabled else 1.0

        base_stage_weights = [
            ("fetch_map", e_mem * 0.60 + e_compute * 0.08 + 1e-9),
            ("electronic_compute", e_compute + 1e-9),
            ("writeback", e_mem * 0.40 + e_compute * 0.06 + 1e-9),
        ]
        stage_weights = [
            ("fetch_map", base_stage_weights[0][1] * fetch_scale),
            ("electronic_compute", base_stage_weights[1][1]),
            ("writeback", base_stage_weights[2][1] * writeback_scale),
        ]
    else:
        e_load_x = max(0.0, _to_float(op.get("energy_mj_load_x"), 0.0) or 0.0)
        e_load_y = max(0.0, _to_float(op.get("energy_mj_load_y"), 0.0) or 0.0)
        e_load = e_load_x + e_load_y
        e_oe = max(0.0, _to_float(op.get("energy_mj_oe"), 0.0) or 0.0)
        e_adc = max(0.0, _to_float(op.get("energy_mj_adc_pca"), 0.0) or 0.0)
        e_laser = max(0.0, _to_float(op.get("energy_mj_laser"), 0.0) or 0.0)
        e_mem = max(0.0, _to_float(op.get("energy_mj_mem"), 0.0) or 0.0)
        e_static = max(0.0, _to_float(op.get("energy_mj_static"), 0.0) or 0.0)

        total = e_load + e_oe + e_adc + e_laser + e_mem + e_static
        mem_ratio = (e_mem / total) if total > 0 else 0.18
        det_bonus = 0.02 if det_enabled else 0.0
        base_bubble = _clamp((0.08 + 0.22 * mem_ratio - det_bonus) * sync_penalty_scale, 0.05, 0.38)
        if flow_enabled and scheduler_mode == FLOW_SCHEDULER_MODE_V3:
            queue_efficiency = float(scheduler_state["queue_efficiency"])
            admission_pressure = float(scheduler_state["admission_pressure"])
            control_backpressure = float(scheduler_state["control_backpressure"])
            residency_hit_rate = float(scheduler_state["residency_hit_rate"])
            prefetch_success_rate = float(scheduler_state["prefetch_success_rate"])
            eviction_pressure = float(scheduler_state["eviction_pressure"])
            queue_pressure = float(scheduler_state["queue_pressure"])
            bubble_scale = (
                1.0
                - 0.72 * overlap_hidden
                - 0.18 * queue_efficiency
                - 0.16 * residency_hit_rate
                - 0.10 * prefetch_success_rate
            )
            bubble_scale += (
                0.16 * admission_pressure
                + 0.12 * control_backpressure
                + 0.08 * eviction_pressure
                + 0.08 * oversubscribe_penalty
                + 0.06 * queue_pressure
            )
            bubble_ratio = _clamp(base_bubble * bubble_scale, 0.02, 0.24)
            fetch_scale = staging_cost_scale * max(
                0.52,
                1.0
                - 0.18 * residency_hit_rate
                - 0.12 * prefetch_success_rate
                + 0.14 * admission_pressure
                + 0.08 * control_backpressure,
            )
            btos_scale = staging_cost_scale * max(
                0.58,
                1.0
                - 0.12 * residency_hit_rate
                - 0.10 * queue_efficiency
                + 0.10 * control_backpressure
                + 0.06 * oversubscribe_penalty,
            )
            serialize_scale = staging_cost_scale * max(
                0.60,
                1.0
                - 0.12 * queue_efficiency
                - 0.10 * prefetch_success_rate
                + 0.10 * control_backpressure
                + 0.08 * oversubscribe_penalty,
            )
            writeback_scale = staging_cost_scale * max(
                0.68,
                1.0 - 0.06 * queue_efficiency - 0.06 * residency_hit_rate + 0.05 * control_backpressure,
            )
        elif flow_enabled and scheduler_mode == FLOW_SCHEDULER_MODE_V2:
            bubble_scale = 1.0 - 0.80 * overlap_hidden - 0.24 * reuse_gain - 0.12 * control_relief
            bubble_scale += 0.10 * oversubscribe_penalty
            bubble_ratio = _clamp(base_bubble * bubble_scale, 0.02, 0.22)
            fetch_scale = staging_cost_scale * max(
                0.55,
                1.0 - 0.24 * reuse_gain - 0.10 * control_relief + 0.08 * oversubscribe_penalty,
            )
            btos_scale = staging_cost_scale * max(
                0.60,
                1.0 - 0.16 * reuse_gain - 0.08 * control_relief + 0.08 * oversubscribe_penalty,
            )
            serialize_scale = staging_cost_scale * max(
                0.62,
                1.0 - 0.14 * reuse_gain - 0.10 * control_relief + 0.10 * oversubscribe_penalty,
            )
            writeback_scale = staging_cost_scale * max(
                0.70,
                1.0 - 0.06 * reuse_gain - 0.08 * control_relief,
            )
        else:
            if flow_enabled:
                bubble_ratio = _clamp(base_bubble * (1.0 - 0.95 * overlap_hidden), 0.03, 0.24)
            else:
                bubble_ratio = _clamp(base_bubble, 0.06, 0.36)
            fetch_scale = staging_cost_scale if flow_enabled else 1.0
            btos_scale = staging_cost_scale if flow_enabled else 1.0
            serialize_scale = staging_cost_scale if flow_enabled else 1.0
            writeback_scale = staging_cost_scale if flow_enabled else 1.0

        base_stage_weights = [
            ("fetch_map", e_mem * 0.45 + e_static * 0.05 + 1e-9),
            ("btos", e_load * (0.56 if det_enabled else 0.48) + 1e-9),
            ("serialize_drive", e_load * (0.44 if det_enabled else 0.52) + e_static * 0.03 + 1e-9),
            ("oag_compute", e_laser + e_static * 0.10 + 1e-9),
            ("pca_adc", e_oe + e_adc + 1e-9),
            ("writeback", e_mem * 0.55 + e_static * 0.04 + 1e-9),
        ]
        stage_weights = [
            ("fetch_map", base_stage_weights[0][1] * fetch_scale),
            ("btos", base_stage_weights[1][1] * btos_scale),
            ("serialize_drive", base_stage_weights[2][1] * serialize_scale),
            ("oag_compute", base_stage_weights[3][1]),
            ("pca_adc", base_stage_weights[4][1]),
            ("writeback", base_stage_weights[5][1] * writeback_scale),
        ]

    base_stage_total = sum(weight for _, weight in base_stage_weights)
    stage_total = sum(weight for _, weight in stage_weights)
    service_scale = stage_total / base_stage_total if base_stage_total > 0 else 1.0
    if stage_total <= 0:
        # Defensive fallback (should rarely happen with +1e-9 floor).
        uniform = 1.0 / max(1, len(stage_weights))
        stage_ratios = [(name, uniform * (1.0 - bubble_ratio)) for name, _ in stage_weights]
    else:
        scale = (1.0 - bubble_ratio) / stage_total
        stage_ratios = [(name, weight * scale) for name, weight in stage_weights]
    stage_ratios.append(("bubble", bubble_ratio))
    return stage_ratios, bubble_ratio, service_scale, scheduler_state


def _build_per_layer_timeline_rows(
    *,
    model: str,
    scaled_ops: list[dict[str, Any]],
    sample_rate_gsps: float,
    flow_enabled: bool,
    det_enabled: bool,
    flow_cfg: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = []
    buffer_rows = []

    for idx, op in enumerate(scaled_ops, start=1):
        layer_name = str(op.get("name") or f"layer_{idx}")
        layer_id = f"{model}:{layer_name}"
        base_latency_s = float(op.get("latency_ms") or 0.0) / 1e3
        active_cycles = max(1, int(round(base_latency_s * sample_rate_gsps * 1e9)))

        op_type = str(op.get("type") or "gemm").lower()
        is_elementwise = op_type in ELEMENTWISE_TYPES
        stage_ratios, bubble_ratio, service_scale, scheduler_controls = _component_weighted_stage_ratios(
            op=op,
            is_elementwise=is_elementwise,
            flow_enabled=flow_enabled,
            det_enabled=det_enabled,
            flow_cfg=flow_cfg,
        )
        active_cycles = max(active_cycles, int(round(active_cycles * service_scale)))
        total_cycles = max(active_cycles, int(round(active_cycles / max(1e-6, 1.0 - bubble_ratio))))
        split_rows = _split_cycles(total_cycles, stage_ratios)

        non_bubble_rows = [(stage, cycles) for stage, cycles in split_rows if stage != "bubble"]
        buffer_depth = int(scheduler_controls["buffer_depth"])
        effective_buffer_depth = int(scheduler_controls["effective_buffer_depth"])
        buffer_capacity = max(
            1,
            int(round(active_cycles * max(1, effective_buffer_depth) / max(1, len(non_bubble_rows)))),
        )
        occupancy_cycles = 0
        for (upstream_stage, upstream_cycles), (downstream_stage, downstream_cycles) in zip(non_bubble_rows, non_bubble_rows[1:]):
            if flow_enabled and scheduler_controls["scheduler_mode"] == FLOW_SCHEDULER_MODE_V3:
                queue_reserve = int(
                    round(
                        buffer_capacity
                        * (
                            0.12 * scheduler_controls["residency_hit_rate"]
                            + 0.10 * scheduler_controls["prefetch_success_rate"]
                            + 0.06 * scheduler_controls["queue_efficiency"]
                        )
                    )
                )
                control_drain = int(
                    round(
                        buffer_capacity
                        * (
                            0.10 * scheduler_controls["control_backpressure"]
                            + 0.08 * scheduler_controls["admission_pressure"]
                        )
                    )
                )
                eviction_drain = int(round(buffer_capacity * 0.06 * scheduler_controls["eviction_pressure"]))
                occupancy_cycles = _clamp(
                    occupancy_cycles
                    + downstream_cycles
                    - upstream_cycles
                    + queue_reserve
                    - control_drain
                    - eviction_drain,
                    0,
                    buffer_capacity,
                )
            elif flow_enabled and scheduler_controls["scheduler_mode"] == FLOW_SCHEDULER_MODE_V2:
                prefetch_reserve = int(
                    round(buffer_capacity * 0.10 * max(0, scheduler_controls["prefetch_window"] - 1))
                )
                reuse_residency = int(round(buffer_capacity * 0.18 * scheduler_controls["reuse_gain"]))
                control_drain = int(round(buffer_capacity * 0.10 * scheduler_controls["control_relief"]))
                occupancy_cycles = _clamp(
                    occupancy_cycles
                    + downstream_cycles
                    - upstream_cycles
                    + prefetch_reserve
                    + reuse_residency
                    - control_drain,
                    0,
                    buffer_capacity,
                )
            else:
                occupancy_cycles = _clamp(
                    occupancy_cycles + downstream_cycles - upstream_cycles,
                    0,
                    buffer_capacity,
                )
            buffer_rows.append(
                {
                    "layer_id": layer_id,
                    "upstream_stage": upstream_stage,
                    "downstream_stage": downstream_stage,
                    "upstream_cycles": upstream_cycles,
                    "downstream_cycles": downstream_cycles,
                    "buffer_depth": buffer_depth,
                    "effective_buffer_depth": effective_buffer_depth,
                    "buffer_capacity_cycles": buffer_capacity,
                    "occupancy_cycles": int(round(occupancy_cycles)),
                    "occupancy_frac": (occupancy_cycles / buffer_capacity) if buffer_capacity > 0 else 0.0,
                    "scheduler_mode": scheduler_controls["scheduler_mode"],
                    "reuse_policy": scheduler_controls["reuse_policy"],
                    "prefetch_window": scheduler_controls["prefetch_window"],
                    "control_group_size": scheduler_controls["control_group_size"],
                    "admission_stalls": scheduler_controls["admission_stalls"],
                    "prefetch_hits": scheduler_controls["prefetch_hits"],
                    "prefetch_drops": scheduler_controls["prefetch_drops"],
                    "residency_hit_rate": scheduler_controls["residency_hit_rate"],
                    "control_backpressure": scheduler_controls["control_backpressure"],
                    "eviction_count": scheduler_controls["eviction_count"],
                }
            )

        for stage, cycles in split_rows:
            scheduler_mode = str(scheduler_controls["scheduler_mode"])
            if stage == "bubble":
                utilization = 0.0
            elif stage == "oag_compute":
                if flow_enabled and scheduler_mode == FLOW_SCHEDULER_MODE_V3:
                    base_util = (
                        0.91
                        + 0.02 * scheduler_controls["queue_efficiency"]
                        + 0.02 * scheduler_controls["residency_hit_rate"]
                        - 0.03 * scheduler_controls["control_backpressure"]
                    )
                    utilization = _clamp(base_util - 0.07 * bubble_ratio, 0.58, 0.98)
                elif flow_enabled and scheduler_mode == FLOW_SCHEDULER_MODE_V2:
                    base_util = 0.91 + 0.02 * scheduler_controls["control_relief"]
                    utilization = _clamp(base_util - 0.07 * bubble_ratio, 0.60, 0.97)
                else:
                    utilization = _clamp((0.90 if flow_enabled else 0.80) - 0.08 * bubble_ratio, 0.60, 0.95)
            elif stage == "electronic_compute":
                base_util = 0.88
                if flow_enabled and scheduler_mode == FLOW_SCHEDULER_MODE_V3:
                    base_util += (
                        0.02 * scheduler_controls["queue_efficiency"]
                        + 0.02 * scheduler_controls["residency_hit_rate"]
                        - 0.02 * scheduler_controls["control_backpressure"]
                    )
                elif flow_enabled and scheduler_mode == FLOW_SCHEDULER_MODE_V2:
                    base_util += 0.02 * scheduler_controls["reuse_gain"]
                utilization = _clamp(base_util - 0.05 * bubble_ratio, 0.65, 0.94)
            elif stage in {"fetch_map", "writeback"}:
                base_util = 0.70 if not flow_enabled else 0.80
                if flow_enabled and scheduler_mode == FLOW_SCHEDULER_MODE_V3:
                    base_util += (
                        0.04 * scheduler_controls["residency_hit_rate"]
                        + 0.03 * scheduler_controls["prefetch_success_rate"]
                        - 0.03 * scheduler_controls["control_backpressure"]
                        - 0.02 * scheduler_controls["admission_pressure"]
                    )
                elif flow_enabled and scheduler_mode == FLOW_SCHEDULER_MODE_V2:
                    base_util += (
                        0.04 * scheduler_controls["reuse_gain"]
                        + 0.03 * scheduler_controls["control_relief"]
                        - 0.02 * scheduler_controls["oversubscribe_penalty"]
                    )
                utilization = _clamp(base_util - 0.25 * bubble_ratio, 0.45, 0.92)
            else:
                base_util = 0.74 if not flow_enabled else 0.83
                if flow_enabled and scheduler_mode == FLOW_SCHEDULER_MODE_V3:
                    base_util += (
                        0.03 * scheduler_controls["effective_overlap"]
                        + 0.02 * scheduler_controls["queue_efficiency"]
                        - 0.02 * scheduler_controls["control_backpressure"]
                    )
                elif flow_enabled and scheduler_mode == FLOW_SCHEDULER_MODE_V2:
                    base_util += (
                        0.03 * scheduler_controls["effective_overlap"]
                        + 0.02 * scheduler_controls["control_relief"]
                    )
                utilization = _clamp(base_util - 0.20 * bubble_ratio, 0.50, 0.95)
            rows.append(
                {
                    "layer_id": layer_id,
                    "stage": stage,
                    "cycles": cycles,
                    "utilization": utilization,
                }
            )
    return rows, buffer_rows


def _resolve_flow_scheduler_controls(
    *,
    flow_cfg: dict[str, Any],
    flow_enabled: bool,
) -> dict[str, Any]:
    buffer_depth = max(0, int(_to_float(flow_cfg.get("buffer_depth"), 2) or 0))
    overlap_efficiency = _clamp(_to_float(flow_cfg.get("overlap_efficiency"), 0.75) or 0.75, 0.0, 1.0)
    staging_cost_scale = max(0.0, _to_float(flow_cfg.get("staging_cost_scale"), 1.0) or 1.0)
    sync_penalty_scale = max(0.0, _to_float(flow_cfg.get("sync_penalty_scale"), 1.0) or 1.0)
    if not flow_enabled:
        return {
            "scheduler_mode": "disabled",
            "buffer_depth": 0,
            "effective_buffer_depth": 0,
            "overlap_efficiency": 0.0,
            "staging_cost_scale": 0.0,
            "sync_penalty_scale": 0.0,
            "reuse_policy": "disabled",
            "prefetch_window": 0,
            "control_group_size": 0,
            "buffer_headroom": 0.0,
            "effective_overlap": 0.0,
            "reuse_gain": 0.0,
            "control_relief": 0.0,
            "oversubscribe_penalty": 0.0,
            "tile_rows": 0,
            "tile_cols": 0,
            "prefetch_credits": 0,
            "execute_credits": 0,
            "control_issue_width": 0,
            "admission_policy": "disabled",
            "eviction_policy": "disabled",
            "service_policy": "disabled",
            "reuse_residency_budget": 0,
            "broadcast_stability_window": 0,
            "prefetch_distance": 0,
            "exception_lane_policy": "disabled",
        }

    raw_scheduler_mode = str(flow_cfg.get("scheduler_mode") or "").strip().lower()
    scheduler_mode = FLOW_SCHEDULER_MODE_V2
    if raw_scheduler_mode in {"legacy", "v1", FLOW_SCHEDULER_MODE_V1}:
        scheduler_mode = FLOW_SCHEDULER_MODE_V1
    elif raw_scheduler_mode in {"v3", FLOW_SCHEDULER_MODE_V3}:
        scheduler_mode = FLOW_SCHEDULER_MODE_V3

    reuse_policy = str(flow_cfg.get("reuse_policy") or "operand_factored").strip().lower()
    if reuse_policy not in FLOW_REUSE_POLICIES:
        reuse_policy = "operand_factored"
    prefetch_window = max(1, int(_to_float(flow_cfg.get("prefetch_window"), 2) or 1))
    control_group_size = max(1, int(_to_float(flow_cfg.get("control_group_size"), 4) or 1))
    tile_rows = max(1, int(_to_float(flow_cfg.get("tile_rows"), 4) or 1))
    tile_cols = max(1, int(_to_float(flow_cfg.get("tile_cols"), 4) or 1))
    prefetch_credits = max(
        1,
        int(_to_float(flow_cfg.get("prefetch_credits"), max(1, prefetch_window)) or 1),
    )
    execute_credits = max(
        1,
        int(_to_float(flow_cfg.get("execute_credits"), max(1, min(4, buffer_depth + 1))) or 1),
    )
    control_issue_width = max(
        1,
        int(_to_float(flow_cfg.get("control_issue_width"), control_group_size) or 1),
    )
    admission_policy = str(
        flow_cfg.get("admission_policy")
        or ("reuse_first" if reuse_policy != "none" else "conservative")
    ).strip().lower()
    if admission_policy not in {"conservative", "greedy", "reuse_first"}:
        admission_policy = "reuse_first" if reuse_policy != "none" else "conservative"
    eviction_policy = str(
        flow_cfg.get("eviction_policy")
        or ("pinned_operand" if reuse_policy != "none" else "fifo")
    ).strip().lower()
    if eviction_policy not in {"fifo", "reuse_distance", "pinned_operand"}:
        eviction_policy = "pinned_operand" if reuse_policy != "none" else "fifo"
    service_policy = str(flow_cfg.get("service_policy") or "critical_path_first").strip().lower()
    if service_policy not in {"age_first", "reuse_first", "critical_path_first"}:
        service_policy = "critical_path_first"
    reuse_residency_budget = max(
        1,
        int(
            _to_float(
                flow_cfg.get("reuse_residency_budget"),
                buffer_depth + max(0, prefetch_window - 1),
            )
            or 1
        ),
    )
    broadcast_stability_window = max(
        1,
        int(_to_float(flow_cfg.get("broadcast_stability_window"), max(1, control_group_size // 2)) or 1),
    )
    prefetch_distance = max(
        1,
        int(_to_float(flow_cfg.get("prefetch_distance"), prefetch_window) or 1),
    )
    exception_lane_policy = str(flow_cfg.get("exception_lane_policy") or "defer").strip().lower()
    if exception_lane_policy not in {"defer", "spill"}:
        exception_lane_policy = "defer"

    if scheduler_mode == FLOW_SCHEDULER_MODE_V1:
        buffer_headroom = (buffer_depth / (buffer_depth + 1.0)) if buffer_depth > 0 else 0.0
        return {
            "scheduler_mode": scheduler_mode,
            "buffer_depth": buffer_depth,
            "effective_buffer_depth": buffer_depth,
            "overlap_efficiency": overlap_efficiency,
            "staging_cost_scale": staging_cost_scale,
            "sync_penalty_scale": sync_penalty_scale,
            "reuse_policy": "none",
            "prefetch_window": 0,
            "control_group_size": 1,
            "buffer_headroom": buffer_headroom,
            "effective_overlap": overlap_efficiency * buffer_headroom,
            "reuse_gain": 0.0,
            "control_relief": 0.0,
            "oversubscribe_penalty": 0.0,
            "tile_rows": 0,
            "tile_cols": 0,
            "prefetch_credits": 0,
            "execute_credits": 0,
            "control_issue_width": 1,
            "admission_policy": "disabled",
            "eviction_policy": "disabled",
            "service_policy": "disabled",
            "reuse_residency_budget": 0,
            "broadcast_stability_window": 0,
            "prefetch_distance": 0,
            "exception_lane_policy": "disabled",
        }

    prefetch_gain = prefetch_window / (prefetch_window + 1.0)
    buffer_headroom = (buffer_depth / (buffer_depth + prefetch_window)) if buffer_depth > 0 else 0.0
    reuse_base = {
        "none": 0.0,
        "operand_pair": 0.45,
        "operand_factored": 0.75,
    }[reuse_policy]
    reuse_support = 0.20 + 0.80 * buffer_headroom
    reuse_gain = _clamp(reuse_base * prefetch_gain * reuse_support, 0.0, 1.0)
    control_relief = _clamp(math.log2(control_group_size + 1.0) / math.log2(9.0), 0.0, 1.0)
    effective_overlap = _clamp(
        overlap_efficiency
        * (0.60 * buffer_headroom + 0.25 * prefetch_gain + 0.15 * control_relief),
        0.0,
        1.0,
    )
    active_buffer_depth = buffer_depth + max(0, min(prefetch_window - 1, buffer_depth))
    oversubscribe_penalty = 0.0
    buffer_guard = max(1, buffer_depth)
    if prefetch_window > buffer_guard:
        oversubscribe_penalty = (prefetch_window - buffer_guard) / (prefetch_window + buffer_guard)
    return {
        "scheduler_mode": scheduler_mode,
        "buffer_depth": buffer_depth,
        "effective_buffer_depth": active_buffer_depth,
        "overlap_efficiency": overlap_efficiency,
        "staging_cost_scale": staging_cost_scale,
        "sync_penalty_scale": sync_penalty_scale,
        "reuse_policy": reuse_policy,
        "prefetch_window": prefetch_window,
        "control_group_size": control_group_size,
        "buffer_headroom": buffer_headroom,
        "effective_overlap": effective_overlap,
        "reuse_gain": reuse_gain,
        "control_relief": control_relief,
        "oversubscribe_penalty": oversubscribe_penalty,
        "tile_rows": tile_rows if scheduler_mode == FLOW_SCHEDULER_MODE_V3 else 0,
        "tile_cols": tile_cols if scheduler_mode == FLOW_SCHEDULER_MODE_V3 else 0,
        "prefetch_credits": prefetch_credits if scheduler_mode == FLOW_SCHEDULER_MODE_V3 else 0,
        "execute_credits": execute_credits if scheduler_mode == FLOW_SCHEDULER_MODE_V3 else 0,
        "control_issue_width": control_issue_width if scheduler_mode == FLOW_SCHEDULER_MODE_V3 else control_group_size,
        "admission_policy": admission_policy if scheduler_mode == FLOW_SCHEDULER_MODE_V3 else "disabled",
        "eviction_policy": eviction_policy if scheduler_mode == FLOW_SCHEDULER_MODE_V3 else "disabled",
        "service_policy": service_policy if scheduler_mode == FLOW_SCHEDULER_MODE_V3 else "disabled",
        "reuse_residency_budget": (
            reuse_residency_budget if scheduler_mode == FLOW_SCHEDULER_MODE_V3 else 0
        ),
        "broadcast_stability_window": (
            broadcast_stability_window if scheduler_mode == FLOW_SCHEDULER_MODE_V3 else 0
        ),
        "prefetch_distance": prefetch_distance if scheduler_mode == FLOW_SCHEDULER_MODE_V3 else 0,
        "exception_lane_policy": (
            exception_lane_policy if scheduler_mode == FLOW_SCHEDULER_MODE_V3 else "disabled"
        ),
    }


def _aggregate_timeline_for_model(
    *, model: str, per_layer_timeline_rows: list[dict[str, Any]]
) -> tuple[str, int, float]:
    stage_totals: dict[str, int] = {}
    weighted_util = 0.0
    total_cycles = 0
    for row in per_layer_timeline_rows:
        layer_id = str(row.get("layer_id") or "")
        if not layer_id.startswith(f"{model}:"):
            continue
        stage = str(row.get("stage") or "")
        cycles = int(_to_float(row.get("cycles"), 0.0) or 0)
        util = _to_float(row.get("utilization"), 0.0) or 0.0
        stage_totals[stage] = stage_totals.get(stage, 0) + cycles
        weighted_util += util * cycles
        total_cycles += cycles
    bubble_cycles = int(stage_totals.get("bubble", 0))
    utilization_avg = weighted_util / total_cycles if total_cycles > 0 else 0.0
    return json.dumps(stage_totals, ensure_ascii=False, sort_keys=True), bubble_cycles, utilization_avg


def _aggregate_buffer_trace_for_model(
    *,
    model: str,
    per_layer_buffer_rows: list[dict[str, Any]],
) -> tuple[int, float]:
    peak_cycles = 0
    peak_frac = 0.0
    for row in per_layer_buffer_rows:
        layer_id = str(row.get("layer_id") or "")
        if not layer_id.startswith(f"{model}:"):
            continue
        peak_cycles = max(peak_cycles, int(_to_float(row.get("occupancy_cycles"), 0.0) or 0))
        peak_frac = max(peak_frac, float(_to_float(row.get("occupancy_frac"), 0.0) or 0.0))
    return peak_cycles, peak_frac


def _aggregate_flow_diagnostics_for_model(
    *,
    model: str,
    per_layer_buffer_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    layer_diagnostics: dict[str, dict[str, float]] = {}
    for row in per_layer_buffer_rows:
        layer_id = str(row.get("layer_id") or "")
        if not layer_id.startswith(f"{model}:"):
            continue
        diagnostics = layer_diagnostics.setdefault(
            layer_id,
            {
                "admission_stalls": 0.0,
                "prefetch_hits": 0.0,
                "prefetch_drops": 0.0,
                "residency_hit_rate": 0.0,
                "control_backpressure": 0.0,
                "eviction_count": 0.0,
            },
        )
        diagnostics["admission_stalls"] = max(
            diagnostics["admission_stalls"],
            float(_to_float(row.get("admission_stalls"), 0.0) or 0.0),
        )
        diagnostics["prefetch_hits"] = max(
            diagnostics["prefetch_hits"],
            float(_to_float(row.get("prefetch_hits"), 0.0) or 0.0),
        )
        diagnostics["prefetch_drops"] = max(
            diagnostics["prefetch_drops"],
            float(_to_float(row.get("prefetch_drops"), 0.0) or 0.0),
        )
        diagnostics["residency_hit_rate"] = max(
            diagnostics["residency_hit_rate"],
            float(_to_float(row.get("residency_hit_rate"), 0.0) or 0.0),
        )
        diagnostics["control_backpressure"] = max(
            diagnostics["control_backpressure"],
            float(_to_float(row.get("control_backpressure"), 0.0) or 0.0),
        )
        diagnostics["eviction_count"] = max(
            diagnostics["eviction_count"],
            float(_to_float(row.get("eviction_count"), 0.0) or 0.0),
        )

    if not layer_diagnostics:
        return _default_flow_diagnostics()

    layer_rows = list(layer_diagnostics.values())
    return {
        "flow_admission_stalls": int(round(sum(row["admission_stalls"] for row in layer_rows))),
        "flow_prefetch_hits": int(round(sum(row["prefetch_hits"] for row in layer_rows))),
        "flow_prefetch_drops": int(round(sum(row["prefetch_drops"] for row in layer_rows))),
        "flow_residency_hit_rate": sum(row["residency_hit_rate"] for row in layer_rows) / len(layer_rows),
        "flow_control_backpressure": sum(row["control_backpressure"] for row in layer_rows) / len(layer_rows),
        "flow_eviction_count": int(round(sum(row["eviction_count"] for row in layer_rows))),
    }


def _timeline_latency_ms_from_stage_cycles(stage_cycles_payload: str, sample_rate_gsps: float) -> float:
    total_cycles = sum(int(value) for value in json.loads(stage_cycles_payload).values())
    if sample_rate_gsps <= 0:
        return 0.0
    return total_cycles / (sample_rate_gsps * 1e9) * 1e3


def _compute_meso_break_even_metrics(
    *,
    meso_cfg: dict[str, Any],
    meso_enabled: bool,
    latency_s: float,
) -> dict[str, float | int | str | bool]:
    return compute_meso_cost_model(
        meso_cfg=meso_cfg,
        meso_enabled=meso_enabled,
        latency_s=latency_s,
    )


def _build_p0_p1_alignment_rows(
    *,
    points_db: list[Any],
    p_laser_dbm_ref: float | None,
    gaussian_noise_std_ref: float,
    crosstalk_alpha_ref: float,
    p1_align_cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    sigma_step = _to_float(
        _cfg_value(
            p1_align_cfg,
            "gaussian_noise_sigma_lsb_per_3db",
            "sigma_lsb_per_3db",
        ),
        0.1,
    ) or 0.1
    alpha_step = _to_float(p1_align_cfg.get("crosstalk_alpha_per_3db"), 0.02) or 0.02
    for raw in points_db:
        delta = _to_float(raw, None)
        if delta is None:
            continue
        severity = max(0.0, -delta) / 3.0
        p_eff = p_laser_dbm_ref + delta if p_laser_dbm_ref is not None else None
        rows.append(
            {
                "delta_p_db": delta,
                "p_laser_dbm_eff": p_eff,
                "gaussian_noise_std_pred": gaussian_noise_std_ref + sigma_step * severity,
                "crosstalk_alpha_pred": crosstalk_alpha_ref + alpha_step * severity,
            }
        )
    return rows


def _build_per_layer_phy_rows(
    *,
    model: str,
    scaled_ops: list[dict[str, Any]],
    sparse_scale: float,
    sparse_enabled: bool,
    phy_cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    n_wdm = int(_to_float(phy_cfg.get("wdm_channels_n"), 16) or 16)
    margin_db = _to_float(phy_cfg.get("margin_db"), 0.0) or 0.0
    duty_cycle = sparse_scale if sparse_enabled else 1.0
    for idx, op in enumerate(scaled_ops, start=1):
        layer_name = str(op.get("name") or f"layer_{idx}")
        layer_id = f"{model}:{layer_name}"
        op_type = str(op.get("type") or "")
        if op_type in ELEMENTWISE_TYPES:
            active_channels = 0
            layer_duty = 0.0
            margin_eff = margin_db
        else:
            layer_duty = duty_cycle
            active_channels = max(1, int(round(n_wdm * layer_duty)))
            margin_eff = margin_db + 10.0 * math.log10(max(layer_duty, 1e-6))
        rows.append(
            {
                "layer_id": layer_id,
                "active_channels": active_channels,
                "duty_cycle": layer_duty,
                "margin_eff_db": margin_eff,
            }
        )
    return rows


def _build_calibration_log_rows(
    *,
    cfg: dict[str, Any],
    effective_k: float,
    sparse_scale: float,
    fit_error: float | None,
) -> list[dict[str, Any]]:
    data_cfg = cfg.get("data") or {}
    sc_det_cfg = cfg.get("sc_det") or {}
    sparse_cfg = cfg.get("sparse") or {}
    p1_align = cfg.get("p1_align") or {}

    calib_manifest = data_cfg.get("calib_manifest_csv")
    eval_manifest = data_cfg.get("eval_manifest_csv")
    rows = []

    # B7 fix: each entry records its own fit_error (only p1_align uses the
    # global fit_error; DET and SPARSE get None as placeholder).
    k_grid = (sc_det_cfg.get("early_stop") or {}).get("k_grid") or []
    rows.append(
        {
            "calib_manifest": calib_manifest,
            "eval_manifest": eval_manifest,
            "scan_grid": json.dumps({"k_grid": k_grid}, ensure_ascii=False),
            "selected_value": effective_k,
            "selection_rule": "minimum_k_under_delta_pp_budget",
            "objective": "accuracy_constrained_energy_reduction",
            "fit_error": None,
        }
    )

    tau_mode = sparse_cfg.get("tau_mode")
    tau_payload = {
        "tau_mode": tau_mode,
        "tau_global": sparse_cfg.get("tau_global"),
        "tau_by_layer": sparse_cfg.get("tau_by_layer"),
    }
    rows.append(
        {
            "calib_manifest": calib_manifest,
            "eval_manifest": eval_manifest,
            "scan_grid": json.dumps(tau_payload, ensure_ascii=False, sort_keys=True),
            "selected_value": sparse_scale,
            "selection_rule": "delta_pp_constrained_duty_cycle",
            "objective": "sparse_energy_saving",
            "fit_error": None,
        }
    )

    rows.append(
        {
            "calib_manifest": calib_manifest,
            "eval_manifest": eval_manifest,
            "scan_grid": json.dumps(
                {"p1_alignment_points_db": p1_align.get("p1_alignment_points_db") or []},
                ensure_ascii=False,
            ),
            "selected_value": json.dumps(
                {
                    "gaussian_noise_std_ref": p1_align.get("gaussian_noise_std_ref"),
                    "crosstalk_alpha_ref": p1_align.get("crosstalk_alpha_ref"),
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            "selection_rule": "min_abs_delta_acc_m0_vs_m2",
            "objective": "p0_p1_alignment",
            "fit_error": fit_error,
        }
    )
    return rows


def _collect_det_prefix_k_grid(sc_det_cfg: dict[str, Any], effective_k: float) -> list[int]:
    early_stop = sc_det_cfg.get("early_stop") or {}
    prefix_cfg = sc_det_cfg.get("prefix_error") or {}
    raw_grid = prefix_cfg.get("k_grid")
    if not isinstance(raw_grid, list):
        raw_grid = early_stop.get("k_grid")

    k_candidates: list[int] = []
    if isinstance(raw_grid, list):
        for raw in raw_grid:
            k_val = _to_float(raw, None)
            if k_val is not None:
                k_candidates.append(int(round(k_val)))

    for raw in (early_stop.get("k_global"), effective_k):
        k_val = _to_float(raw, None)
        if k_val is not None:
            k_candidates.append(int(round(k_val)))

    k_by_layer = early_stop.get("k_by_layer")
    if isinstance(k_by_layer, dict):
        for raw in k_by_layer.values():
            k_val = _to_float(raw, None)
            if k_val is not None:
                k_candidates.append(int(round(k_val)))

    return sorted({k for k in k_candidates if k > 0})


def _write_det_prefix_error_table(
    *,
    out_dir: Path,
    sc_det_cfg: dict[str, Any],
    effective_k: float,
) -> Path | None:
    prefix_cfg = sc_det_cfg.get("prefix_error") or {}
    if not _to_bool(prefix_cfg.get("enabled"), True):
        return None

    bsl_max = int(_to_float(sc_det_cfg.get("bsl_max"), 129.0) or 129.0)
    bsl_max = max(1, bsl_max)
    k_grid = _collect_det_prefix_k_grid(sc_det_cfg, effective_k)
    if not k_grid:
        k_grid = [bsl_max]

    num_prob_points = int(_to_float(prefix_cfg.get("num_prob_points"), 129.0) or 129)
    p_min = _to_float(prefix_cfg.get("p_min"), 1e-3)
    p_max = _to_float(prefix_cfg.get("p_max"), 1.0 - 1e-3)
    if p_min is None:
        p_min = 1e-3
    if p_max is None:
        p_max = 1.0 - 1e-3

    rows = compute_prefix_error_stats(
        bsl_max=bsl_max,
        k_grid=k_grid,
        num_prob_points=num_prob_points,
        p_min=p_min,
        p_max=p_max,
        det_mode=str(sc_det_cfg.get("det_mode") or "reorder"),
        phase_shift=int(_to_float(prefix_cfg.get("phase_shift"), 0.0) or 0.0),
        scramble_seed=int(_to_float(prefix_cfg.get("scramble_seed"), 0.0) or 0.0),
        enforce_monotonic=_to_bool(prefix_cfg.get("enforce_monotonic"), False),
    )
    out_path = out_dir / "det_prefix_error.csv"
    _write_csv(
        out_path,
        [
            "k",
            "bsl_max",
            "prefix_error_mean",
            "prefix_error_p95",
            "relative_error_mean",
            "energy_saved_pct",
            "num_prob_points",
        ],
        rows,
    )
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase-1 experiment runner (E0–E6).")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to phase1 config YAML (see configs/phase1_template.yaml).",
    )
    parser.add_argument(
        "--execution-semantics",
        choices=("proxy", "bitstream"),
        default=None,
        help="Optional explicit execution-semantics override for candidate or rollback runs.",
    )
    parser.add_argument(
        "--device",
        choices=("mps",),
        default="mps",
        help="Governed local accelerator declaration. This repository allows local MPS runs only.",
    )
    args = parser.parse_args()

    cfg_path = Path(args.config).resolve()
    cfg = _load_yaml(cfg_path)

    run_cfg = cfg.get("run") or {}
    experiment_id = str(run_cfg.get("experiment_id") or "E0").upper()
    run_cfg["experiment_id"] = experiment_id
    switches = _resolve_switches(cfg, experiment_id)
    _sync_section_enabled(cfg, switches)

    run_id = _resolve_run_id(run_cfg, experiment_id)
    run_cfg["run_id"] = run_id
    cfg["run"] = run_cfg
    cfg["switches"] = switches
    public_variant_surface = _resolve_public_variant_surface(
        run_cfg=run_cfg,
        experiment_id=experiment_id,
    )
    cfg = _apply_realism_profile_if_present(cfg, cfg_path=cfg_path)
    execution_semantics_cfg = _resolve_execution_semantics_cfg(
        cfg,
        override_semantics=args.execution_semantics,
    )
    dark_launch_cfg = _resolve_dark_launch_cfg(cfg)
    sc_trust_contract_cfg = _resolve_sc_trust_contract_cfg(cfg)

    outputs_cfg = cfg.get("outputs") or {}
    out_root = resolve_workspace_path(outputs_cfg.get("out_dir") or "results/runs", anchor=ROOT_DIR)
    out_dir = out_root / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    accuracy_cfg = cfg.get("accuracy") or {}

    if _to_bool(outputs_cfg.get("save_config_snapshot"), True):
        _dump_yaml(out_dir / "config_snapshot.yaml", cfg)
        context_run_id = str(accuracy_cfg.get("context_run_id") or run_id or "").strip()
        if context_run_id and context_run_id != run_id:
            _dump_yaml(out_root / context_run_id / "config_snapshot.yaml", cfg)

    phy_cfg = cfg.get("phy") or {}
    sc_det_cfg = cfg.get("sc_det") or {}
    sparse_cfg = cfg.get("sparse") or {}
    meso_cfg = cfg.get("meso") or {}
    flow_cfg = cfg.get("flow") or {}
    data_cfg = cfg.get("data") or {}
    layout_cfg = cfg.get("layout_thermal") or {}
    p1_align_cfg = cfg.get("p1_align") or {}
    energy_model_cfg = cfg.get("energy_model") or {}
    integrated_system_cost_cfg = _resolve_integrated_system_cost_cfg(
        cfg.get("integrated_system_costs") or cfg.get("integrated_system_cost")
    )
    realism_cfg = cfg.get("realism") or {}
    noise_cfg = cfg.get("noise_injection") or {}
    sigma_lsb = _cfg_value(noise_cfg, "gaussian_noise_sigma_lsb", "sigma_lsb")
    if sigma_lsb is not None:
        noise_cfg["sigma_lsb"] = sigma_lsb
    sigma_lsb_ref = _cfg_value(
        p1_align_cfg,
        "gaussian_noise_sigma_lsb_ref",
        "sigma_lsb_ref",
    )
    if sigma_lsb_ref is not None:
        p1_align_cfg["sigma_lsb_ref"] = sigma_lsb_ref
    sigma_lsb_step = _cfg_value(
        p1_align_cfg,
        "gaussian_noise_sigma_lsb_per_3db",
        "sigma_lsb_per_3db",
    )
    if sigma_lsb_step is not None:
        p1_align_cfg["sigma_lsb_per_3db"] = sigma_lsb_step

    det_runtime_metadata = resolve_det_runtime_metadata(sc_det_cfg, switches)
    det_runtime_enabled = _to_bool(det_runtime_metadata.get("det_runtime_enabled"), False)
    bsl_scale, effective_k = _compute_bsl_scale(sc_det_cfg, det_runtime_enabled)
    default_sparse_scale, default_sparse_scale_source = _compute_sparse_scale(
        sparse_cfg,
        switches["sparse"],
    )
    sparse_scale = default_sparse_scale
    sparse_scale_source = default_sparse_scale_source

    phy_result = None
    phy_budget_path = out_dir / "phy_budget.json" if switches["phy"] else None
    phy_sweep_path = out_dir / "phy_sweep.csv" if switches["phy"] else None
    phy_budget_rows: list[dict[str, Any]] = []
    phy_sweep_rows: list[dict[str, Any]] = []

    estimator_cfg = cfg.get("mtl") or cfg.get("hpat") or {}
    estimator_config_path = _resolve_existing_path(
        estimator_cfg.get("config_path") or "mtl_model/mtl_config_asic.yaml"
    )
    estimator_config = _load_yaml(estimator_config_path)

    if switches["phy"]:
        # B1 fix: When PHY is enabled, we store P_laser_mw for power-based
        # laser energy calculation later (E_laser = P_laser_W × latency_s),
        # instead of feeding it to hpat_model's tile-count model which would
        # grossly overestimate laser energy.
        # We set laser_power_mw to 0 here so the tile-based model excludes
        # laser; the real laser energy is added in the per-model loop below.
        energy_cfg = estimator_config.get("energy") or {}
        energy_cfg["laser_power_mw"] = 0.0  # suppress tile-based laser
        estimator_config["energy"] = energy_cfg

    base_estimator_config = dict(estimator_config)
    if isinstance(estimator_config.get("bitstream"), dict):
        base_estimator_config["bitstream"] = dict(estimator_config.get("bitstream") or {})
    estimator_config = _merge_execution_semantics_into_estimator_config(
        base_estimator_config,
        execution_semantics_cfg,
    )

    ops_dir = _resolve_existing_path(estimator_cfg.get("ops_dir") or "mtl_model/ops")
    ops_paths = sorted(ops_dir.glob("*.json"))
    if not ops_paths:
        raise SystemExit(f"No ops JSON files found in {ops_dir}")

    meso_load_scale = resolve_meso_load_scale(
        meso_cfg=meso_cfg,
        meso_enabled=switches["meso"],
    )
    meso_overhead_mj = (
        _to_float(meso_cfg.get("broadcast_overhead_mj"), 0.0) or 0.0
        if switches["meso"]
        else 0.0
    )
    meso_overhead_ms = (
        _to_float(meso_cfg.get("broadcast_overhead_ms"), 0.0) or 0.0
        if switches["meso"]
        else 0.0
    )
    mode = str(energy_model_cfg.get("energy_model_mode") or "UpperBound").lower()
    if mode not in {"upperbound", "countbased"}:
        mode = "upperbound"
    retained_energy_mode = "CountBased" if mode == "countbased" else "UpperBound"
    upperbound_scale = _to_float(energy_model_cfg.get("upperbound_scale"), 1.0) or 1.0
    countbased_scale = _to_float(energy_model_cfg.get("countbased_scale"), 0.9) or 0.9
    upperbound_scale = max(upperbound_scale, countbased_scale)

    thermal_tuning_mw = _to_float(
        layout_cfg.get("p_thermal_tuning_mw", layout_cfg.get("P_thermal_tuning")),
        0.0,
    ) or 0.0
    s_wg_min = _to_float(layout_cfg.get("s_wg_min"), None)

    delta_pp_budget = _to_float(
        accuracy_cfg.get("delta_pp_budget", sc_det_cfg.get("delta_pp")),
        None,
    )

    accuracy_rows: list[dict[str, Any]] = []
    accuracy_source = accuracy_cfg.get("source_csv") or accuracy_cfg.get("csv")
    if accuracy_source:
        accuracy_source_path = _resolve_existing_path(accuracy_source)
        accuracy_rows = _load_accuracy_rows(accuracy_source_path)

    models_cfg = cfg.get("models") or {}
    model_keys_cfg = models_cfg.get("keys")
    if isinstance(model_keys_cfg, list):
        model_keys = [str(x).strip() for x in model_keys_cfg if str(x).strip()]
    else:
        model_keys = []
    ops_paths = _filter_ops_paths(ops_paths, model_keys)

    p1_align_resolved = _calibrate_p1_alignment(
        p1_align_cfg=p1_align_cfg,
        accuracy_rows=accuracy_rows,
        model_keys=model_keys,
        quant_bits_default=_to_float(sc_det_cfg.get("quant_bits"), None),
        delta_pp_budget=delta_pp_budget,
    )
    p1_align_cfg["gaussian_noise_std_ref"] = p1_align_resolved["gaussian_noise_std_ref"]
    p1_align_cfg["crosstalk_alpha_ref"] = p1_align_resolved["crosstalk_alpha_ref"]
    p1_align_cfg["fit_error"] = p1_align_resolved["fit_error"]
    cfg["p1_align"] = p1_align_cfg

    baseline_ref_cfg = cfg.get("baseline_ref") or {}
    baseline_path = baseline_ref_cfg.get("e0_latency_csv")
    baseline_map = {}
    if baseline_path:
        base_path = _resolve_existing_path(baseline_path)
        baseline_map = _load_latency_baseline(base_path)

    summaries: list[dict[str, Any]] = []
    master_rows: list[dict[str, Any]] = []
    dark_launch_rows: list[dict[str, Any]] = []
    per_layer_accuracy_rows: list[dict[str, Any]] = []
    per_layer_timeline_rows: list[dict[str, Any]] = []
    per_layer_buffer_rows: list[dict[str, Any]] = []
    per_layer_phy_rows: list[dict[str, Any]] = []

    ops_subdir = str(estimator_cfg.get("ops_output_subdir") or "mtl_ops")
    ops_out_dir = out_dir / ops_subdir
    sample_rate_gsps = _to_float(
        (estimator_config.get("photonic") or {}).get("sample_rate_gsps"),
        1.0,
    ) or 1.0

    for ops_path in ops_paths:
        model, ops, meta = _load_ops(ops_path)
        model_execution_semantics_cfg, sc_generator_policy_meta = (
            _resolve_generator_policy_for_model(
                cfg,
                execution_semantics_cfg,
                model=model,
            )
        )
        model_estimator_config = _merge_execution_semantics_into_estimator_config(
            base_estimator_config,
            model_execution_semantics_cfg,
        )
        workload_id = str(data_cfg.get("workload_id") or data_cfg.get("workload") or "")
        workload_shape = _resolve_workload_shape(
            model=model,
            meta=meta,
            data_cfg=data_cfg,
        )
        workload_ops = _apply_workload_scale_to_ops(
            ops,
            batch_scale=float(workload_shape["batch_scale"]),
            sequence_scale=float(workload_shape["sequence_scale"]),
            sequence_modeled=bool(workload_shape["sequence_modeled"]),
        )
        op_results, estimator_summary = summarize_ops(workload_ops, model_estimator_config)
        bitstream_estimator_metadata = _extract_bitstream_estimator_metadata(estimator_summary)
        workload_fidelity_metadata = _extract_workload_fidelity_metadata(estimator_summary)
        sc_default_metadata = _extract_sc_default_metadata(estimator_summary)
        true_sc_claim_metadata = _extract_true_sc_claim_metadata(estimator_summary)
        if sc_generator_policy_meta["sc_generator_policy_status"]:
            sc_default_metadata["sc_generator_policy_status"] = sc_generator_policy_meta[
                "sc_generator_policy_status"
            ]
            sc_default_metadata["sc_generator_policy_reason"] = sc_generator_policy_meta[
                "sc_generator_policy_reason"
            ]
        sc_default_status, sc_fail_closed_triggered = _derive_sc_default_status(
            trust_posture=sc_default_metadata["sc_summary_trust_posture"],
            fail_mode=sc_trust_contract_cfg["fail_mode"],
            dark_launch_enabled=dark_launch_cfg["enabled"],
        )
        accuracy_context = _build_accuracy_context(
            run_id=run_id,
            experiment_id=experiment_id,
            out_dir=out_dir,
            run_cfg=run_cfg,
            data_cfg=data_cfg,
            switches=switches,
            sc_det_cfg=sc_det_cfg,
            sparse_cfg=sparse_cfg,
            p1_align_resolved=p1_align_resolved,
            accuracy_cfg=accuracy_cfg,
        )

        acc_ref_top1, acc_top1, acc_drop_pp, accuracy_provenance = _resolve_accuracy_for_model(
            model=model,
            accuracy_cfg=accuracy_cfg,
            noise_cfg=noise_cfg,
            sc_det_cfg=sc_det_cfg,
            accuracy_rows=accuracy_rows,
            accuracy_context=accuracy_context,
            return_provenance=True,
        )
        acc_ref_top1, acc_top1, acc_drop_pp, _ = _apply_stochastic_accuracy_uncertainty(
            acc_ref_top1=acc_ref_top1,
            acc_top1=acc_top1,
            acc_drop_pp=acc_drop_pp,
            model=model,
            run_cfg=run_cfg,
            accuracy_cfg=accuracy_cfg,
            noise_cfg=noise_cfg,
        )
        accuracy_measurement_contract_metadata = _resolve_accuracy_measurement_contract_metadata(
            accuracy_cfg=accuracy_cfg,
            accuracy_provenance=accuracy_provenance,
            active_execution_semantics=model_execution_semantics_cfg["execution_semantics"],
            active_bitstream_generator=model_execution_semantics_cfg["bitstream_generator"],
            active_bitstream_stream_length=model_execution_semantics_cfg[
                "bitstream_stream_length"
            ],
            active_runtime_stream_reuse_policy=DEFAULT_BITSTREAM_RUNTIME_STREAM_REUSE_POLICY,
            active_model_key=model,
            active_ops_path=ops_path,
        )
        fidelity_metadata = _build_fidelity_metadata(
            active_execution_semantics=model_execution_semantics_cfg["execution_semantics"],
            op_results=op_results,
            active_model_key=model,
            active_ops_path=ops_path,
            workload_fidelity_metadata=workload_fidelity_metadata,
            bitstream_estimator_metadata=bitstream_estimator_metadata,
            accuracy_measurement_contract_metadata=accuracy_measurement_contract_metadata,
        )
        accuracy_coupling_metadata = _resolve_accuracy_coupling_metadata(
            accuracy_cfg=accuracy_cfg,
            accuracy_provenance=accuracy_provenance,
            acc_top1=acc_top1,
            acc_drop_pp=acc_drop_pp,
            active_execution_semantics=model_execution_semantics_cfg["execution_semantics"],
            active_bitstream_generator=model_execution_semantics_cfg["bitstream_generator"],
            active_bitstream_stream_length=model_execution_semantics_cfg[
                "bitstream_stream_length"
            ],
            active_runtime_stream_reuse_policy=DEFAULT_BITSTREAM_RUNTIME_STREAM_REUSE_POLICY,
            accuracy_measurement_contract_metadata=accuracy_measurement_contract_metadata,
        )
        sparse_scale, sparse_scale_source = _compute_sparse_scale(
            sparse_cfg,
            switches["sparse"],
            measured_activity_fraction=accuracy_provenance.get(
                "sparse_measured_activity_fraction"
            ),
        )
        phy_result = None
        if switches["phy"]:
            phy_result = compute_link_budget(phy_cfg, duty_cycle=sparse_scale)
            phy_budget_rows.append(
                {
                    "model": model,
                    "duty_cycle": sparse_scale,
                    "sparse_scale_source": sparse_scale_source,
                    **phy_result,
                }
            )
            sweep = phy_cfg.get("sweep_wdm_channels") or []
            if isinstance(sweep, list) and sweep:
                for n in sweep:
                    updated = dict(phy_cfg)
                    updated["wdm_channels_n"] = int(n)
                    phy_sweep_rows.append(
                        {
                            "model": model,
                            "duty_cycle": sparse_scale,
                            "sparse_scale_source": sparse_scale_source,
                            **compute_link_budget(updated, duty_cycle=sparse_scale),
                        }
                    )

        k_by_layer = (sc_det_cfg.get("early_stop") or {}).get("k_by_layer")
        bsl_max_value = float(sc_det_cfg.get("bsl_max") or 1.0)
        scaled_ops = _scale_op_results(
            op_results,
            bsl_max=bsl_max_value,
            bsl_scale=bsl_scale,
            det_enabled=det_runtime_enabled,
            k_by_layer=k_by_layer,
            sparse_scale=sparse_scale,
            meso_load_scale=meso_load_scale,
        )
        scaled_summary = _summarize_scaled_ops(
            scaled_ops,
            meso_overhead_mj=meso_overhead_mj * float(workload_shape["workload_scale"]),
            meso_overhead_ms=meso_overhead_ms * float(workload_shape["workload_scale"]),
        )
        _write_ops_csv(ops_out_dir / f"{model}_ops.csv", scaled_ops)

        base_latency_ms = float(scaled_summary["total_latency_ms"])
        latency_ms = base_latency_ms
        latency_s = latency_ms / 1e3
        base_energy_j = float(scaled_summary["energy_j_total"])

        det_saved_j = 0.0
        if det_runtime_enabled and bsl_scale > 0:
            # B2 fix: Only count energy components that actually scale with
            # BSL length. Exclude mem and static which are BSL-independent.
            affected_mj = (
                float(scaled_summary["energy_mj_load_x"])
                + float(scaled_summary["energy_mj_load_y"])
                + float(scaled_summary["energy_mj_oe"])
                + float(scaled_summary["energy_mj_adc_pca"])
                + float(scaled_summary["energy_mj_laser"])
            )
            det_saved_j = max(0.0, affected_mj / 1e3 * ((1.0 / bsl_scale) - 1.0))

        det_overhead_cfg = sc_det_cfg.get("overhead") or {}
        det_overhead_j = (
            (_to_float(det_overhead_cfg.get("seq_gen_j"), 0.0) or 0.0)
            + (_to_float(det_overhead_cfg.get("counter_j"), 0.0) or 0.0)
            + (_to_float(det_overhead_cfg.get("ctrl_j"), 0.0) or 0.0)
            + (_to_float(sc_det_cfg.get("det_overhead_j"), 0.0) or 0.0)
        )

        pass_delta = (
            (acc_drop_pp <= delta_pp_budget)
            if acc_drop_pp is not None and delta_pp_budget is not None
            else None
        )

        if not det_runtime_enabled:
            det_saved_j = 0.0
            det_overhead_j = 0.0
        det_net_gain_j = det_saved_j - det_overhead_j
        pass_det_net_gain = (
            (det_net_gain_j > 0)
            and (pass_delta is True or pass_delta is None)
            if det_runtime_enabled
            else None
        )

        batch_size = int(workload_shape["batch_size"])
        sequence_length = (
            float(workload_shape["sequence_length"])
            if workload_shape["sequence_modeled"] and workload_shape["sequence_length"] is not None
            else None
        )
        baseline_latency = baseline_map.get(model)

        total_ops = _estimate_total_ops(workload_ops)
        tops_w = None

        tau_by_layer = sparse_cfg.get("tau_by_layer")
        k_i = _serialize_layer_param(k_by_layer) or effective_k
        tau_i = _serialize_layer_param(tau_by_layer) or sparse_cfg.get("tau_global")
        avg_effective_bsl = effective_k if det_runtime_enabled else sc_det_cfg.get("bsl_max")

        model_accuracy_rows = _build_per_layer_accuracy_rows(
            model=model,
            scaled_ops=scaled_ops,
            effective_k=effective_k,
            sc_det_cfg=sc_det_cfg,
            sparse_cfg=sparse_cfg,
            accuracy_cfg=accuracy_cfg,
        )
        model_timeline_rows, model_buffer_rows = _build_per_layer_timeline_rows(
            model=model,
            scaled_ops=scaled_ops,
            sample_rate_gsps=sample_rate_gsps,
            flow_enabled=switches["flow"],
            det_enabled=det_runtime_enabled,
            flow_cfg=flow_cfg,
        )
        model_phy_rows = _build_per_layer_phy_rows(
            model=model,
            scaled_ops=scaled_ops,
            sparse_scale=sparse_scale,
            sparse_enabled=switches["sparse"],
            phy_cfg=phy_cfg,
        )
        stage_cycles, bubble_cycles, utilization_avg = _aggregate_timeline_for_model(
            model=model,
            per_layer_timeline_rows=model_timeline_rows,
        )
        flow_buffer_peak_cycles, flow_buffer_peak_frac = _aggregate_buffer_trace_for_model(
            model=model,
            per_layer_buffer_rows=model_buffer_rows,
        )
        flow_diagnostics = _aggregate_flow_diagnostics_for_model(
            model=model,
            per_layer_buffer_rows=model_buffer_rows,
        )
        core_latency_ms = _timeline_latency_ms_from_stage_cycles(stage_cycles, sample_rate_gsps)
        latency_ms = core_latency_ms
        latency_s = latency_ms / 1e3
        thermal_energy_j = thermal_tuning_mw / 1000.0 * latency_s

        core_energy_upperbound_j = base_energy_j * upperbound_scale + thermal_energy_j
        core_energy_countbased_j = base_energy_j * countbased_scale + thermal_energy_j
        upperbound_breakdown_conversion = (
            scaled_summary["energy_j_conversion_control"] * upperbound_scale
        )
        upperbound_breakdown_memory = scaled_summary["energy_j_memory_move"] * upperbound_scale
        countbased_breakdown_conversion = (
            scaled_summary["energy_j_conversion_control"] * countbased_scale
        )
        countbased_breakdown_memory = scaled_summary["energy_j_memory_move"] * countbased_scale
        if mode == "countbased":
            selected_scale = countbased_scale
            core_energy_j = core_energy_countbased_j
        else:
            selected_scale = upperbound_scale
            core_energy_j = core_energy_upperbound_j
        core_avg_power_w = _safe_ratio(core_energy_j, latency_s) if latency_s > 0 else None

        breakdown_conversion = scaled_summary["energy_j_conversion_control"] * selected_scale
        breakdown_memory = scaled_summary["energy_j_memory_move"] * selected_scale
        breakdown_oe = scaled_summary["energy_j_oe"] * selected_scale
        breakdown_adc = scaled_summary["energy_j_adc_pca"] * selected_scale
        breakdown_laser = scaled_summary["energy_j_laser_optical"] * selected_scale
        _p_laser_mw = phy_result.get("p_laser_mw") if phy_result else None
        if phy_result is not None and _p_laser_mw is not None:
            phy_laser_energy_j = (_p_laser_mw / 1000.0) * latency_s
            breakdown_laser = phy_laser_energy_j
            core_energy_j = (
                core_energy_j
                - scaled_summary["energy_j_laser_optical"] * selected_scale
                + phy_laser_energy_j
            )
            core_energy_upperbound_j = (
                base_energy_j * upperbound_scale
                + thermal_energy_j
                - scaled_summary["energy_j_laser_optical"] * upperbound_scale
                + phy_laser_energy_j
            )
            core_energy_countbased_j = (
                base_energy_j * countbased_scale
                + thermal_energy_j
                - scaled_summary["energy_j_laser_optical"] * countbased_scale
                + phy_laser_energy_j
            )
            if mode == "countbased":
                core_energy_j = core_energy_countbased_j
            else:
                core_energy_j = core_energy_upperbound_j
            core_avg_power_w = _safe_ratio(core_energy_j, latency_s) if latency_s > 0 else None

        meso_metrics = _compute_meso_break_even_metrics(
            meso_cfg=meso_cfg,
            meso_enabled=switches["meso"],
            latency_s=latency_s,
        )
        upperbound_breakdown_conversion = max(
            0.0,
            upperbound_breakdown_conversion - float(meso_metrics["net_energy_gain_j"]),
        )
        countbased_breakdown_conversion = max(
            0.0,
            countbased_breakdown_conversion - float(meso_metrics["net_energy_gain_j"]),
        )
        core_energy_upperbound_j = max(
            0.0,
            core_energy_upperbound_j - float(meso_metrics["net_energy_gain_j"]),
        )
        core_energy_countbased_j = max(
            0.0,
            core_energy_countbased_j - float(meso_metrics["net_energy_gain_j"]),
        )
        if mode == "countbased":
            core_energy_j = core_energy_countbased_j
        else:
            core_energy_j = core_energy_upperbound_j
        core_avg_power_w = _safe_ratio(core_energy_j, latency_s) if latency_s > 0 else None

        upperbound_integrated_costs = _compute_integrated_system_costs(
            conversion_control_j=upperbound_breakdown_conversion,
            memory_move_j=upperbound_breakdown_memory,
            thermal_energy_j=thermal_energy_j,
            core_latency_ms=core_latency_ms,
            stage_cycles_payload=stage_cycles,
            sample_rate_gsps=sample_rate_gsps,
            flow_enabled=switches["flow"],
            phy_enabled=switches["phy"],
            noise_enabled=_to_bool(noise_cfg.get("enabled"), False),
            cost_cfg=integrated_system_cost_cfg,
        )
        countbased_integrated_costs = _compute_integrated_system_costs(
            conversion_control_j=countbased_breakdown_conversion,
            memory_move_j=countbased_breakdown_memory,
            thermal_energy_j=thermal_energy_j,
            core_latency_ms=core_latency_ms,
            stage_cycles_payload=stage_cycles,
            sample_rate_gsps=sample_rate_gsps,
            flow_enabled=switches["flow"],
            phy_enabled=switches["phy"],
            noise_enabled=_to_bool(noise_cfg.get("enabled"), False),
            cost_cfg=integrated_system_cost_cfg,
        )
        energy_upperbound_j = core_energy_upperbound_j + float(
            upperbound_integrated_costs["integrated_hidden_system_cost_j"]
        )
        energy_countbased_j = core_energy_countbased_j + float(
            countbased_integrated_costs["integrated_hidden_system_cost_j"]
        )
        if mode == "countbased":
            selected_integrated_costs = countbased_integrated_costs
            selected_breakdown_conversion = countbased_breakdown_conversion
            selected_breakdown_memory = countbased_breakdown_memory
            energy_j = energy_countbased_j
        else:
            selected_integrated_costs = upperbound_integrated_costs
            selected_breakdown_conversion = upperbound_breakdown_conversion
            selected_breakdown_memory = upperbound_breakdown_memory
            energy_j = energy_upperbound_j
        breakdown_conversion = selected_breakdown_conversion
        breakdown_memory = selected_breakdown_memory
        hidden_system_latency_ms = float(
            selected_integrated_costs["integrated_hidden_system_latency_ms"]
        )
        latency_ms = core_latency_ms + hidden_system_latency_ms
        latency_s = latency_ms / 1e3
        system_latency_lower_ms = core_latency_ms + float(
            selected_integrated_costs["integrated_hidden_system_latency_lower_ms"]
        )
        system_latency_upper_ms = core_latency_ms + float(
            selected_integrated_costs["integrated_hidden_system_latency_upper_ms"]
        )
        system_energy_lower_j = core_energy_j + float(
            selected_integrated_costs["integrated_hidden_system_cost_lower_j"]
        )
        system_energy_upper_j = core_energy_j + float(
            selected_integrated_costs["integrated_hidden_system_cost_upper_j"]
        )
        avg_power_w = _safe_ratio(energy_j, latency_s) if latency_s > 0 else None

        throughput_images_s = None
        throughput_tokens_s = None
        if latency_s > 0:
            if "imagenet" in workload_id.lower():
                throughput_images_s = batch_size / latency_s
            if sequence_length is not None:
                throughput_tokens_s = (batch_size * sequence_length) / latency_s
        if experiment_id == "E0":
            speedup_vs_e0 = 1.0
        elif baseline_latency:
            speedup_vs_e0 = baseline_latency / latency_ms if latency_ms > 0 else None
        else:
            speedup_vs_e0 = None
        if avg_power_w and avg_power_w > 0 and latency_s > 0 and total_ops > 0:
            tops_w = (total_ops / latency_s) / avg_power_w / 1e12
        breakdown_other = (
            scaled_summary["energy_j_other_static"] * selected_scale + thermal_energy_j
        )
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        split = str(data_cfg.get("split") or "eval")
        git_hash = str(run_cfg.get("git_hash") or "nogit")
        # B11 fix: auto-detect git hash if placeholder
        if git_hash == "nogit":
            git_hash = _detect_git_hash()

        n_wdm = int(_to_float(phy_cfg.get("wdm_channels_n"), 0) or 0)
        er_db = _to_float(phy_cfg.get("er_db"), None)
        ber_target = _to_float(phy_cfg.get("ber_target"), None)
        loss_path_db = phy_result.get("loss_path_db") if phy_result else None
        pp_crosstalk_db = phy_result.get("pp_crosstalk_db") if phy_result else None
        p_laser_dbm = phy_result.get("p_laser_dbm") if phy_result else None
        p_laser_mw = phy_result.get("p_laser_mw") if phy_result else None
        phy_penalty_table_version = (
            phy_result.get("phy_penalty_table_version") if phy_result else None
        ) or (
            (phy_cfg.get("crosstalk") or {}).get("phy_penalty_table_version")
            or phy_cfg.get("phy_penalty_table_version")
        )
        realism_assessment = _build_realism_assessment(
            integrated_system_cost_cfg=integrated_system_cost_cfg,
            integrated_system_cost_mode=str(selected_integrated_costs["integrated_system_cost_mode"]),
            flow_cfg=flow_cfg,
            meso_cfg=meso_cfg,
            phy_cfg=phy_cfg,
            realism_cfg=realism_cfg,
            accuracy_source_csv=accuracy_provenance.get("accuracy_source_csv"),
            accuracy_provenance=accuracy_provenance,
            switches=switches,
            accuracy_coupling_metadata=accuracy_coupling_metadata,
            accuracy_measurement_contract_metadata=accuracy_measurement_contract_metadata,
            workload_fidelity_metadata=workload_fidelity_metadata,
            fidelity_metadata=fidelity_metadata,
        )
        estimation_model_readiness = _derive_estimation_model_readiness(
            sc_default_metadata=sc_default_metadata,
            realism_assessment=realism_assessment,
            workload_fidelity_metadata=workload_fidelity_metadata,
            fidelity_metadata=fidelity_metadata,
        )
        reporting_boundaries = _resolve_reporting_boundaries(
            model_execution_semantics_cfg["execution_semantics"]
        )
        dark_launch_comparator_metadata = {
            "sc_summary_trust_posture": "",
        }
        if dark_launch_cfg["enabled"]:
            comparator_execution_semantics_cfg = _resolve_execution_semantics_cfg(
                cfg,
                override_semantics=dark_launch_cfg["comparator_execution_semantics"],
            )
            comparator_execution_semantics_cfg, _ = _resolve_generator_policy_for_model(
                cfg,
                comparator_execution_semantics_cfg,
                model=model,
            )
            if (
                comparator_execution_semantics_cfg["execution_semantics"]
                == model_execution_semantics_cfg["execution_semantics"]
            ):
                raise SystemExit(
                    "dark_launch comparator semantics must differ from the candidate semantics."
                )
            comparator_estimator_config = _merge_execution_semantics_into_estimator_config(
                base_estimator_config,
                comparator_execution_semantics_cfg,
            )
            comparator_op_results, comparator_estimator_summary = summarize_ops(
                workload_ops,
                comparator_estimator_config,
            )
            comparator_scaled_ops = _scale_op_results(
                comparator_op_results,
                bsl_max=bsl_max_value,
                bsl_scale=bsl_scale,
                det_enabled=det_runtime_enabled,
                k_by_layer=k_by_layer,
                sparse_scale=sparse_scale,
                meso_load_scale=meso_load_scale,
            )
            comparator_scaled_summary = _summarize_scaled_ops(
                comparator_scaled_ops,
                meso_overhead_mj=meso_overhead_mj * float(workload_shape["workload_scale"]),
                meso_overhead_ms=meso_overhead_ms * float(workload_shape["workload_scale"]),
            )
            comparator_timeline_rows, _ = _build_per_layer_timeline_rows(
                model=model,
                scaled_ops=comparator_scaled_ops,
                sample_rate_gsps=sample_rate_gsps,
                flow_enabled=switches["flow"],
                det_enabled=det_runtime_enabled,
                flow_cfg=flow_cfg,
            )
            comparator_stage_cycles, _, _ = _aggregate_timeline_for_model(
                model=model,
                per_layer_timeline_rows=comparator_timeline_rows,
            )
            comparator_core_latency_ms = _timeline_latency_ms_from_stage_cycles(
                comparator_stage_cycles,
                sample_rate_gsps,
            )
            comparator_core_energy_j = (
                float(comparator_scaled_summary["energy_j_total"]) * selected_scale + thermal_energy_j
            )
            comparator_core_energy_j = max(
                0.0,
                comparator_core_energy_j - float(meso_metrics["net_energy_gain_j"]),
            )
            if phy_result is not None and _p_laser_mw is not None:
                comparator_core_energy_j = (
                    comparator_core_energy_j
                    - float(comparator_scaled_summary["energy_j_laser_optical"]) * selected_scale
                    + phy_laser_energy_j
                )
            dark_launch_comparator_metadata = _extract_sc_default_metadata(
                comparator_estimator_summary
            )
            dark_launch_rows.append(
                {
                    "model": model,
                    "candidate_label": dark_launch_cfg["candidate_label"],
                    "candidate_execution_semantics": model_execution_semantics_cfg[
                        "execution_semantics"
                    ],
                    "candidate_execution_semantics_origin": model_execution_semantics_cfg[
                        "execution_semantics_origin"
                    ],
                    "candidate_summary_trust_posture": sc_default_metadata[
                        "sc_summary_trust_posture"
                    ],
                    "candidate_true_sc_summary_claim_state": true_sc_claim_metadata[
                        "true_sc_summary_claim_state"
                    ],
                    "candidate_default_status": sc_default_status,
                    "candidate_latency_ms": core_latency_ms,
                    "candidate_energy_j": core_energy_j,
                    "candidate_support_classes": sc_default_metadata["sc_support_classes"],
                    "candidate_native_op_count": sc_default_metadata["sc_native_op_count"],
                    "candidate_governed_support_op_count": sc_default_metadata[
                        "sc_governed_support_op_count"
                    ],
                    "candidate_unsupported_op_count": sc_default_metadata[
                        "sc_unsupported_op_count"
                    ],
                    "comparator_label": dark_launch_cfg["comparator_label"],
                    "comparator_execution_semantics": comparator_execution_semantics_cfg[
                        "execution_semantics"
                    ],
                    "comparator_summary_trust_posture": dark_launch_comparator_metadata[
                        "sc_summary_trust_posture"
                    ],
                    "comparator_true_sc_summary_claim_state": _extract_true_sc_claim_metadata(
                        comparator_estimator_summary
                    )["true_sc_summary_claim_state"],
                    "comparator_latency_ms": comparator_core_latency_ms,
                    "comparator_energy_j": comparator_core_energy_j,
                    "coverage_delta_rows": len(op_results) - len(comparator_op_results),
                    "latency_delta_ms": core_latency_ms - comparator_core_latency_ms,
                    "energy_delta_j": core_energy_j - comparator_core_energy_j,
                    "compatibility_status": (
                        "schema_compatible"
                        if len(op_results) == len(comparator_op_results)
                        else "coverage_mismatch"
                    ),
                }
            )

        flow_summary_fields = _resolve_flow_summary_fields(
            flow_cfg=flow_cfg,
            flow_enabled=switches["flow"],
            flow_buffer_peak_cycles=flow_buffer_peak_cycles,
            flow_buffer_peak_frac=flow_buffer_peak_frac,
            diagnostics=flow_diagnostics,
        )
        summary_row = {
            "run_id": run_id,
            "variant_id": public_variant_surface["variant_id"],
            "experiment_id": experiment_id,
            "internal_experiment_id": public_variant_surface["internal_experiment_id"],
            "public_module_stack": json.dumps(
                public_variant_surface["public_module_stack"],
                ensure_ascii=False,
            ),
            "active_switch_set": _active_switch_set(switches),
            "workload_id": workload_id,
            "model": model,
            "ops_path": str(ops_path),
            "input_size": meta.get("input_size"),
            "batch_size": batch_size,
            "execution_semantics": model_execution_semantics_cfg["execution_semantics"],
            "execution_semantics_default": model_execution_semantics_cfg[
                "execution_semantics_default"
            ],
            "execution_semantics_origin": model_execution_semantics_cfg[
                "execution_semantics_origin"
            ],
            "bitstream_enabled": model_execution_semantics_cfg["bitstream_enabled"],
            "bitstream_encoding_mode": model_execution_semantics_cfg["bitstream_encoding_mode"],
            "bitstream_multiplier_mode": model_execution_semantics_cfg["bitstream_multiplier_mode"],
            "bitstream_stream_length": model_execution_semantics_cfg["bitstream_stream_length"],
            "bitstream_generator": model_execution_semantics_cfg["bitstream_generator"],
            "generator_stream_state_policy": bitstream_estimator_metadata[
                "generator_stream_state_policy"
            ],
            "runtime_stream_reuse_policy": (
                DEFAULT_BITSTREAM_RUNTIME_STREAM_REUSE_POLICY
                if model_execution_semantics_cfg["execution_semantics"] == "bitstream"
                else ""
            ),
            "bitstream_accumulator_mode": model_execution_semantics_cfg["bitstream_accumulator_mode"],
            "bitstream_calibration_source": model_execution_semantics_cfg["bitstream_calibration_source"],
            "bitstream_capture_manifest_csv": model_execution_semantics_cfg["bitstream_capture_manifest_csv"],
            "bitstream_effective_stream_length": bitstream_estimator_metadata[
                "bitstream_effective_stream_length"
            ],
            "bitstream_effective_stream_length_scale": bitstream_estimator_metadata[
                "bitstream_effective_stream_length_scale"
            ],
            "bitstream_effective_stream_length_scale_provenance": bitstream_estimator_metadata[
                "bitstream_effective_stream_length_scale_provenance"
            ],
            "bitstream_parallel_outputs": bitstream_estimator_metadata[
                "bitstream_parallel_outputs"
            ],
            "bitstream_parallel_outputs_provenance": bitstream_estimator_metadata[
                "bitstream_parallel_outputs_provenance"
            ],
            "bitstream_cycles_per_stream_bit": bitstream_estimator_metadata[
                "bitstream_cycles_per_stream_bit"
            ],
            "bitstream_cycles_per_stream_bit_provenance": bitstream_estimator_metadata[
                "bitstream_cycles_per_stream_bit_provenance"
            ],
            "bitstream_accumulator_energy_pj": bitstream_estimator_metadata[
                "bitstream_accumulator_energy_pj"
            ],
            "bitstream_accumulator_energy_pj_provenance": bitstream_estimator_metadata[
                "bitstream_accumulator_energy_pj_provenance"
            ],
            "bitstream_elementwise_parallelism_factor": bitstream_estimator_metadata[
                "bitstream_elementwise_parallelism_factor"
            ],
            "bitstream_elementwise_parallelism_provenance": bitstream_estimator_metadata[
                "bitstream_elementwise_parallelism_provenance"
            ],
            "bitstream_calibration_applied": bitstream_estimator_metadata[
                "bitstream_calibration_applied"
            ],
            "bitstream_calibration_summary_json": bitstream_estimator_metadata[
                "bitstream_calibration_summary_json"
            ],
            "bitstream_calibration_reason": bitstream_estimator_metadata[
                "bitstream_calibration_reason"
            ],
            "bitstream_calibration_median_relative_error": bitstream_estimator_metadata[
                "bitstream_calibration_median_relative_error"
            ],
            "bitstream_calibration_capture_row_count": bitstream_estimator_metadata[
                "bitstream_calibration_capture_row_count"
            ],
            "bitstream_calibration_replay_row_count": bitstream_estimator_metadata[
                "bitstream_calibration_replay_row_count"
            ],
            "bitstream_datapath_stage_summary": bitstream_estimator_metadata[
                "bitstream_datapath_stage_summary"
            ],
            "model_abstraction_boundary_kind": bitstream_estimator_metadata[
                "model_abstraction_boundary_kind"
            ],
            "model_abstraction_boundary_status": bitstream_estimator_metadata[
                "model_abstraction_boundary_status"
            ],
            "model_abstraction_boundary_reason": bitstream_estimator_metadata[
                "model_abstraction_boundary_reason"
            ],
            "model_abstraction_boundary_json": bitstream_estimator_metadata[
                "model_abstraction_boundary_json"
            ],
            "conv2d_gemm_lowered_approximation_op_count": bitstream_estimator_metadata[
                "conv2d_gemm_lowered_approximation_op_count"
            ],
            "conv2d_native_runtime_modeled_op_count": bitstream_estimator_metadata[
                "conv2d_native_runtime_modeled_op_count"
            ],
            "conv_fidelity_stage": realism_assessment["conv_fidelity_stage"],
            "conv_fidelity_blockers": realism_assessment["conv_fidelity_blockers"],
            "conv_evidence_manifest_path": realism_assessment[
                "conv_evidence_manifest_path"
            ],
            "conv_evidence_manifest_sha256": realism_assessment[
                "conv_evidence_manifest_sha256"
            ],
            "conv_measured_package_path": realism_assessment[
                "conv_measured_package_path"
            ],
            "conv_measured_package_sha256": realism_assessment[
                "conv_measured_package_sha256"
            ],
            "conv_measured_closure_status": realism_assessment[
                "conv_measured_closure_status"
            ],
            "workload_fidelity_class": workload_fidelity_metadata["workload_fidelity_class"],
            "workload_fidelity_status": workload_fidelity_metadata["workload_fidelity_status"],
            "workload_fidelity_reason": workload_fidelity_metadata["workload_fidelity_reason"],
            "workload_fidelity_blockers": workload_fidelity_metadata[
                "workload_fidelity_blockers"
            ],
            "fidelity_ready": realism_assessment["fidelity_ready"],
            "fidelity_blockers": realism_assessment["fidelity_blockers"],
            "sc_summary_trust_posture": sc_default_metadata["sc_summary_trust_posture"],
            "true_sc_summary_claim_state": true_sc_claim_metadata[
                "true_sc_summary_claim_state"
            ],
            "true_sc_claim_state_inventory": true_sc_claim_metadata[
                "true_sc_claim_state_inventory"
            ],
            "true_sc_claim_surface_status": true_sc_claim_metadata[
                "true_sc_claim_surface_status"
            ],
            "true_sc_claim_surface_inventory": true_sc_claim_metadata[
                "true_sc_claim_surface_inventory"
            ],
            "true_sc_claim_surface_native_op_count": true_sc_claim_metadata[
                "true_sc_claim_surface_native_op_count"
            ],
            "true_sc_claim_surface_governed_support_op_count": true_sc_claim_metadata[
                "true_sc_claim_surface_governed_support_op_count"
            ],
            "true_sc_support_out_of_surface_op_count": true_sc_claim_metadata[
                "true_sc_support_out_of_surface_op_count"
            ],
            "true_sc_out_of_claim_surface_op_count": true_sc_claim_metadata[
                "true_sc_out_of_claim_surface_op_count"
            ],
            "true_sc_native_op_count": true_sc_claim_metadata["true_sc_native_op_count"],
            "true_sc_governed_not_true_sc_op_count": true_sc_claim_metadata[
                "true_sc_governed_not_true_sc_op_count"
            ],
            "true_sc_out_of_surface_op_count": true_sc_claim_metadata[
                "true_sc_out_of_surface_op_count"
            ],
            "sc_calibration_state": sc_default_metadata["sc_calibration_state"],
            "sc_generator_policy_status": sc_default_metadata["sc_generator_policy_status"],
            "sc_generator_policy_reason": sc_default_metadata["sc_generator_policy_reason"],
            "sc_support_classes": sc_default_metadata["sc_support_classes"],
            "sc_native_op_count": sc_default_metadata["sc_native_op_count"],
            "sc_governed_support_op_count": sc_default_metadata[
                "sc_governed_support_op_count"
            ],
            "sc_unsupported_op_count": sc_default_metadata["sc_unsupported_op_count"],
            "estimation_model_coverage_status": sc_default_metadata[
                "estimation_model_coverage_status"
            ],
            "estimation_model_coverage_reason": sc_default_metadata[
                "estimation_model_coverage_reason"
            ],
            "estimation_model_support_boundary": sc_default_metadata[
                "estimation_model_support_boundary"
            ],
            "estimation_model_supported_op_count": sc_default_metadata[
                "estimation_model_supported_op_count"
            ],
            "estimation_model_unsupported_op_count": sc_default_metadata[
                "estimation_model_unsupported_op_count"
            ],
            "estimation_model_ready_status": estimation_model_readiness[
                "estimation_model_ready_status"
            ],
            "estimation_model_ready": estimation_model_readiness[
                "estimation_model_ready"
            ],
            "estimation_model_ready_reason": estimation_model_readiness[
                "estimation_model_ready_reason"
            ],
            "estimation_model_ready_blockers": estimation_model_readiness[
                "estimation_model_ready_blockers"
            ],
            "sc_fail_mode": sc_trust_contract_cfg["fail_mode"],
            "sc_fail_closed_triggered": sc_fail_closed_triggered,
            "sc_default_status": sc_default_status,
            "dark_launch_enabled": dark_launch_cfg["enabled"],
            "dark_launch_candidate_label": dark_launch_cfg["candidate_label"],
            "dark_launch_comparator_label": dark_launch_cfg["comparator_label"],
            "dark_launch_comparator_execution_semantics": dark_launch_cfg[
                "comparator_execution_semantics"
            ]
            if dark_launch_cfg["enabled"]
            else None,
            "dark_launch_comparator_trust_posture": dark_launch_comparator_metadata[
                "sc_summary_trust_posture"
            ],
            "core_latency_ms": core_latency_ms,
            "latency_ms": latency_ms,
            "system_latency_lower_ms": system_latency_lower_ms,
            "system_latency_upper_ms": system_latency_upper_ms,
            "core_energy_j": core_energy_j,
            "energy_j": energy_j,
            "system_energy_lower_j": system_energy_lower_j,
            "system_energy_upper_j": system_energy_upper_j,
            "avg_power_w": avg_power_w,
            "core_avg_power_w": core_avg_power_w,
            "tops_w": tops_w,
            "energy_upperbound_j": energy_upperbound_j,
            "energy_countbased_j": energy_countbased_j,
            "energy_breakdown_conversion_control_j": breakdown_conversion,
            "energy_breakdown_memory_move_j": breakdown_memory,
            "energy_breakdown_oe_j": breakdown_oe,
            "energy_breakdown_adc_pca_j": breakdown_adc,
            "energy_breakdown_laser_optical_j": breakdown_laser,
            "energy_breakdown_other_static_j": breakdown_other,
            "integrated_onchip_comm_j": selected_integrated_costs["integrated_onchip_comm_j"],
            "integrated_control_sched_j": selected_integrated_costs["integrated_control_sched_j"],
            "integrated_host_staging_j": selected_integrated_costs["integrated_host_staging_j"],
            "integrated_calibration_monitoring_j": selected_integrated_costs["integrated_calibration_monitoring_j"],
            "integrated_hidden_system_cost_j": selected_integrated_costs["integrated_hidden_system_cost_j"],
            "integrated_hidden_system_cost_lower_j": selected_integrated_costs["integrated_hidden_system_cost_lower_j"],
            "integrated_hidden_system_cost_upper_j": selected_integrated_costs["integrated_hidden_system_cost_upper_j"],
            "integrated_onchip_comm_ms": selected_integrated_costs["integrated_onchip_comm_ms"],
            "integrated_control_sched_ms": selected_integrated_costs["integrated_control_sched_ms"],
            "integrated_host_staging_ms": selected_integrated_costs["integrated_host_staging_ms"],
            "integrated_calibration_monitoring_ms": selected_integrated_costs["integrated_calibration_monitoring_ms"],
            "integrated_hidden_system_latency_ms": selected_integrated_costs["integrated_hidden_system_latency_ms"],
            "integrated_hidden_system_latency_lower_ms": selected_integrated_costs["integrated_hidden_system_latency_lower_ms"],
            "integrated_hidden_system_latency_upper_ms": selected_integrated_costs["integrated_hidden_system_latency_upper_ms"],
            "integrated_system_cost_mode": selected_integrated_costs["integrated_system_cost_mode"],
            "integrated_onchip_comm_evidence": realism_assessment["integrated_onchip_comm_evidence"],
            "integrated_control_sched_evidence": realism_assessment["integrated_control_sched_evidence"],
            "integrated_host_staging_evidence": realism_assessment["integrated_host_staging_evidence"],
            "integrated_calibration_monitoring_evidence": realism_assessment["integrated_calibration_monitoring_evidence"],
            "integrated_system_cost_evidence": realism_assessment["integrated_system_cost_evidence"],
            "integrated_system_cost_calibration_source": realism_assessment["integrated_system_cost_calibration_source"],
            "integrated_system_cost_uncertainty_method": realism_assessment["integrated_system_cost_uncertainty_method"],
            "flow_timeline_evidence": realism_assessment["flow_timeline_evidence"],
            "flow_timeline_calibration_source": realism_assessment["flow_timeline_calibration_source"],
            "meso_cost_evidence": realism_assessment["meso_cost_evidence"],
            "meso_cost_calibration_source": realism_assessment["meso_cost_calibration_source"],
            "phy_support_evidence": realism_assessment["phy_support_evidence"],
            "phy_support_calibration_source": realism_assessment["phy_support_calibration_source"],
            "accuracy_coupling_evidence": realism_assessment["accuracy_coupling_evidence"],
            "accuracy_coupling_metric": realism_assessment["accuracy_coupling_metric"],
            "accuracy_coupling_source": realism_assessment["accuracy_coupling_source"],
            "accuracy_coupling_reason": realism_assessment["accuracy_coupling_reason"],
            "accuracy_measurement_contract_status": realism_assessment[
                "accuracy_measurement_contract_status"
            ],
            "accuracy_measurement_contract_reason": realism_assessment[
                "accuracy_measurement_contract_reason"
            ],
            "accuracy_measurement_contract_source": realism_assessment[
                "accuracy_measurement_contract_source"
            ],
            "accuracy_measurement_contract_truth_class": realism_assessment[
                "accuracy_measurement_contract_truth_class"
            ],
            "accuracy_measurement_contract_authorization_note": realism_assessment[
                "accuracy_measurement_contract_authorization_note"
            ],
            "accuracy_measurement_contract_authorization_status": realism_assessment[
                "accuracy_measurement_contract_authorization_status"
            ],
            "accuracy_measurement_contract_conv_measured_package_path": realism_assessment[
                "accuracy_measurement_contract_conv_measured_package_path"
            ],
            "accuracy_measurement_contract_conv_measured_package_sha256": realism_assessment[
                "accuracy_measurement_contract_conv_measured_package_sha256"
            ],
            "accuracy_measurement_contract_required_truth_class": realism_assessment[
                "accuracy_measurement_contract_required_truth_class"
            ],
            "accuracy_measurement_contract_required_fields_json": realism_assessment[
                "accuracy_measurement_contract_required_fields_json"
            ],
            "accuracy_measurement_contract_observed_fields_json": realism_assessment[
                "accuracy_measurement_contract_observed_fields_json"
            ],
            "accuracy_measurement_contract_violations_json": realism_assessment[
                "accuracy_measurement_contract_violations_json"
            ],
            "accuracy_evidence_tier": realism_assessment["accuracy_evidence_tier"],
            "analysis_grade_ready": realism_assessment["analysis_grade_ready"],
            "analysis_grade_blockers": realism_assessment["analysis_grade_blockers"],
            "realism_class": realism_assessment["realism_class"],
            "proxy_promotion_ready": realism_assessment["proxy_promotion_ready"],
            "proxy_upgrade_blockers": realism_assessment["proxy_upgrade_blockers"],
            "benchmark_claim_ready": realism_assessment["benchmark_claim_ready"],
            "device_comparison_scope": realism_assessment["device_comparison_scope"],
            "benchmark_equivalence": realism_assessment["benchmark_equivalence"],
            "comparison_boundary": reporting_boundaries["comparison_boundary"],
            "latency_boundary": reporting_boundaries["latency_boundary"],
            "energy_boundary": reporting_boundaries["energy_boundary"],
            "power_boundary": reporting_boundaries["power_boundary"],
            "retained_energy_mode": retained_energy_mode,
            "acc_ref_top1": acc_ref_top1,
            "acc_top1": acc_top1,
            "acc_drop_pp": acc_drop_pp,
            "accuracy_source_csv": accuracy_provenance.get("accuracy_source_csv"),
            "accuracy_baseline_row_id": accuracy_provenance.get("accuracy_baseline_row_id"),
            "accuracy_target_row_id": accuracy_provenance.get("accuracy_target_row_id"),
            "accuracy_baseline_source_run_id": accuracy_provenance.get("accuracy_baseline_source_run_id"),
            "accuracy_target_source_run_id": accuracy_provenance.get("accuracy_target_source_run_id"),
            "accuracy_target_split": accuracy_provenance.get("accuracy_target_split"),
            "accuracy_target_notes": accuracy_provenance.get("accuracy_target_notes"),
            "pass_delta": pass_delta,
            "det_policy": det_runtime_metadata.get("det_policy"),
            "det_k_signature": det_runtime_metadata.get("det_k_signature"),
            "det_runtime_enabled": det_runtime_enabled,
            "det_quality_gate_enabled": det_runtime_metadata.get("det_quality_gate_enabled"),
            "det_quality_gate_policy": det_runtime_metadata.get("det_quality_gate_policy"),
            "det_quality_gate_status": det_runtime_metadata.get("det_quality_gate_status"),
            "det_quality_gate_reason": det_runtime_metadata.get("det_quality_gate_reason"),
            "det_quality_gate_fallback_policy": det_runtime_metadata.get(
                "det_quality_gate_fallback_policy"
            ),
            "det_quality_gate_require_measured_accuracy": det_runtime_metadata.get(
                "det_quality_gate_require_measured_accuracy"
            ),
            "det_quality_gate_measured_accuracy_ready": det_runtime_metadata.get(
                "det_quality_gate_measured_accuracy_ready"
            ),
            "det_quality_gate_max_prefix_error_mean": det_runtime_metadata.get(
                "det_quality_gate_max_prefix_error_mean"
            ),
            "det_quality_gate_max_prefix_error_p95": det_runtime_metadata.get(
                "det_quality_gate_max_prefix_error_p95"
            ),
            "det_prefix_error_mean": det_runtime_metadata.get("det_prefix_error_mean"),
            "det_prefix_error_p95": det_runtime_metadata.get("det_prefix_error_p95"),
            "avg_effective_bsl": avg_effective_bsl,
            "duty_cycle_avg": sparse_scale if switches["sparse"] else 1.0,
            "sparse_active_fraction": sparse_scale if switches["sparse"] else 1.0,
            "sparse_scale_source": sparse_scale_source,
            "sparse_measured_activity_fraction": accuracy_provenance.get(
                "sparse_measured_activity_fraction"
            ),
            "det_saved_j": det_saved_j,
            "det_overhead_j": det_overhead_j,
            "det_net_gain_j": det_net_gain_j,
            "pass_det_net_gain": pass_det_net_gain,
            "stage_cycles": stage_cycles,
            "bubble_cycles": bubble_cycles,
            "utilization_avg": utilization_avg,
            **flow_summary_fields,
            "hops_scheduler_mode": (
                flow_summary_fields["flow_model_mode"] if switches["flow"] else "disabled"
            ),
            "fanout": meso_metrics["fanout"],
            "topology_dimension": meso_metrics["topology_dimension"],
            "serializers_saved": meso_metrics["serializers_saved"],
            "serializer_energy_j": meso_metrics["serializer_energy_j"],
            "broadcast_driver_energy_j": meso_metrics["broadcast_driver_energy_j"],
            "fabric_control_overhead_j": meso_metrics["fabric_control_overhead_j"],
            "extra_buffering_overhead_j": meso_metrics["extra_buffering_overhead_j"],
            "explicit_total_cost_j": meso_metrics["explicit_total_cost_j"],
            "explicit_total_savings_j": meso_metrics["explicit_total_savings_j"],
            "meso_cost_model_mode": meso_metrics["cost_model_mode"],
            "net_energy_gain_j": meso_metrics["net_energy_gain_j"],
            "N_wdm": n_wdm,
            "PP_crosstalk_db": pp_crosstalk_db,
            "P_laser_dbm": p_laser_dbm,
            "P_laser_mw": p_laser_mw,
            "phy_link_budget_status": (
                "ready" if switches["phy"] and phy_result is not None else "disabled"
            ),
            "gaussian_noise_std_ref": p1_align_resolved["gaussian_noise_std_ref"],
            "crosstalk_alpha_ref": p1_align_resolved["crosstalk_alpha_ref"],
            "p1_align_method": p1_align_resolved["method"],
            "p1_align_fit_error": p1_align_resolved["fit_error"],
            "phy_penalty_table_version": phy_penalty_table_version,
        }
        summaries.append(summary_row)

        master_rows.append(
            {
                "run_id": run_id,
                "variant_id": public_variant_surface["variant_id"],
                "experiment_id": experiment_id,
                "internal_experiment_id": public_variant_surface["internal_experiment_id"],
                "public_module_stack": json.dumps(
                    public_variant_surface["public_module_stack"],
                    ensure_ascii=False,
                ),
                "workload_id": workload_id,
                "model": model,
                "split": split,
                "seed": run_cfg.get("seed"),
                "git_hash": git_hash,
                "date": now,
                "meso": switches["meso"],
                "flow": switches["flow"],
                "det": switches["det"],
                "sparse": switches["sparse"],
                "phy": switches["phy"],
                "det_mode": sc_det_cfg.get("det_mode"),
                "sparse_mode": sparse_cfg.get("tau_mode"),
                "phy_mode": phy_cfg.get("closure"),
                "execution_semantics_default": model_execution_semantics_cfg[
                    "execution_semantics_default"
                ],
                "execution_semantics_origin": model_execution_semantics_cfg[
                    "execution_semantics_origin"
                ],
                "execution_semantics": model_execution_semantics_cfg["execution_semantics"],
                "bitstream_enabled": model_execution_semantics_cfg["bitstream_enabled"],
                "bitstream_encoding_mode": model_execution_semantics_cfg["bitstream_encoding_mode"],
                "bitstream_multiplier_mode": model_execution_semantics_cfg["bitstream_multiplier_mode"],
                "bitstream_stream_length": model_execution_semantics_cfg["bitstream_stream_length"],
                "bitstream_generator": model_execution_semantics_cfg["bitstream_generator"],
                "generator_stream_state_policy": bitstream_estimator_metadata[
                    "generator_stream_state_policy"
                ],
                "runtime_stream_reuse_policy": (
                    DEFAULT_BITSTREAM_RUNTIME_STREAM_REUSE_POLICY
                    if model_execution_semantics_cfg["execution_semantics"] == "bitstream"
                    else ""
                ),
                "bitstream_accumulator_mode": model_execution_semantics_cfg["bitstream_accumulator_mode"],
                "bitstream_calibration_source": model_execution_semantics_cfg["bitstream_calibration_source"],
                "bitstream_capture_manifest_csv": model_execution_semantics_cfg["bitstream_capture_manifest_csv"],
                "bitstream_effective_stream_length": bitstream_estimator_metadata[
                    "bitstream_effective_stream_length"
                ],
                "bitstream_effective_stream_length_scale": bitstream_estimator_metadata[
                    "bitstream_effective_stream_length_scale"
                ],
                "bitstream_effective_stream_length_scale_provenance": bitstream_estimator_metadata[
                    "bitstream_effective_stream_length_scale_provenance"
                ],
                "bitstream_parallel_outputs": bitstream_estimator_metadata[
                    "bitstream_parallel_outputs"
                ],
                "bitstream_parallel_outputs_provenance": bitstream_estimator_metadata[
                    "bitstream_parallel_outputs_provenance"
                ],
                "bitstream_cycles_per_stream_bit": bitstream_estimator_metadata[
                    "bitstream_cycles_per_stream_bit"
                ],
                "bitstream_cycles_per_stream_bit_provenance": bitstream_estimator_metadata[
                    "bitstream_cycles_per_stream_bit_provenance"
                ],
                "bitstream_accumulator_energy_pj": bitstream_estimator_metadata[
                    "bitstream_accumulator_energy_pj"
                ],
                "bitstream_accumulator_energy_pj_provenance": bitstream_estimator_metadata[
                    "bitstream_accumulator_energy_pj_provenance"
                ],
                "bitstream_elementwise_parallelism_factor": bitstream_estimator_metadata[
                    "bitstream_elementwise_parallelism_factor"
                ],
                "bitstream_elementwise_parallelism_provenance": bitstream_estimator_metadata[
                    "bitstream_elementwise_parallelism_provenance"
                ],
                "bitstream_calibration_applied": bitstream_estimator_metadata[
                    "bitstream_calibration_applied"
                ],
                "bitstream_calibration_summary_json": bitstream_estimator_metadata[
                    "bitstream_calibration_summary_json"
                ],
                "bitstream_calibration_reason": bitstream_estimator_metadata[
                    "bitstream_calibration_reason"
                ],
                "bitstream_calibration_median_relative_error": bitstream_estimator_metadata[
                    "bitstream_calibration_median_relative_error"
                ],
                "bitstream_calibration_capture_row_count": bitstream_estimator_metadata[
                    "bitstream_calibration_capture_row_count"
                ],
                "bitstream_calibration_replay_row_count": bitstream_estimator_metadata[
                    "bitstream_calibration_replay_row_count"
                ],
                "bitstream_datapath_stage_summary": bitstream_estimator_metadata[
                    "bitstream_datapath_stage_summary"
                ],
                "model_abstraction_boundary_kind": bitstream_estimator_metadata[
                    "model_abstraction_boundary_kind"
                ],
                "model_abstraction_boundary_status": bitstream_estimator_metadata[
                    "model_abstraction_boundary_status"
                ],
                "model_abstraction_boundary_reason": bitstream_estimator_metadata[
                    "model_abstraction_boundary_reason"
                ],
                "model_abstraction_boundary_json": bitstream_estimator_metadata[
                    "model_abstraction_boundary_json"
                ],
                "conv2d_gemm_lowered_approximation_op_count": bitstream_estimator_metadata[
                    "conv2d_gemm_lowered_approximation_op_count"
                ],
                "conv2d_native_runtime_modeled_op_count": bitstream_estimator_metadata[
                    "conv2d_native_runtime_modeled_op_count"
                ],
                "conv_fidelity_stage": realism_assessment["conv_fidelity_stage"],
                "conv_fidelity_blockers": realism_assessment["conv_fidelity_blockers"],
                "conv_evidence_manifest_path": realism_assessment[
                    "conv_evidence_manifest_path"
                ],
                "conv_evidence_manifest_sha256": realism_assessment[
                    "conv_evidence_manifest_sha256"
                ],
                "conv_measured_package_path": realism_assessment[
                    "conv_measured_package_path"
                ],
                "conv_measured_package_sha256": realism_assessment[
                    "conv_measured_package_sha256"
                ],
                "conv_measured_closure_status": realism_assessment[
                    "conv_measured_closure_status"
                ],
                "workload_fidelity_class": workload_fidelity_metadata[
                    "workload_fidelity_class"
                ],
                "workload_fidelity_status": workload_fidelity_metadata[
                    "workload_fidelity_status"
                ],
                "workload_fidelity_reason": workload_fidelity_metadata[
                    "workload_fidelity_reason"
                ],
                "workload_fidelity_blockers": workload_fidelity_metadata[
                    "workload_fidelity_blockers"
                ],
                "fidelity_ready": realism_assessment["fidelity_ready"],
                "fidelity_blockers": realism_assessment["fidelity_blockers"],
                "sc_summary_trust_posture": sc_default_metadata["sc_summary_trust_posture"],
                "true_sc_summary_claim_state": true_sc_claim_metadata[
                    "true_sc_summary_claim_state"
                ],
                "true_sc_claim_state_inventory": true_sc_claim_metadata[
                    "true_sc_claim_state_inventory"
                ],
                "true_sc_claim_surface_status": true_sc_claim_metadata[
                    "true_sc_claim_surface_status"
                ],
                "true_sc_claim_surface_inventory": true_sc_claim_metadata[
                    "true_sc_claim_surface_inventory"
                ],
                "true_sc_claim_surface_native_op_count": true_sc_claim_metadata[
                    "true_sc_claim_surface_native_op_count"
                ],
                "true_sc_claim_surface_governed_support_op_count": true_sc_claim_metadata[
                    "true_sc_claim_surface_governed_support_op_count"
                ],
                "true_sc_support_out_of_surface_op_count": true_sc_claim_metadata[
                    "true_sc_support_out_of_surface_op_count"
                ],
                "true_sc_out_of_claim_surface_op_count": true_sc_claim_metadata[
                    "true_sc_out_of_claim_surface_op_count"
                ],
                "true_sc_native_op_count": true_sc_claim_metadata[
                    "true_sc_native_op_count"
                ],
                "true_sc_governed_not_true_sc_op_count": true_sc_claim_metadata[
                    "true_sc_governed_not_true_sc_op_count"
                ],
                "true_sc_out_of_surface_op_count": true_sc_claim_metadata[
                    "true_sc_out_of_surface_op_count"
                ],
                "sc_calibration_state": sc_default_metadata["sc_calibration_state"],
                "sc_generator_policy_status": sc_default_metadata["sc_generator_policy_status"],
                "sc_generator_policy_reason": sc_default_metadata["sc_generator_policy_reason"],
                "sc_support_classes": sc_default_metadata["sc_support_classes"],
                "sc_native_op_count": sc_default_metadata["sc_native_op_count"],
                "sc_governed_support_op_count": sc_default_metadata[
                    "sc_governed_support_op_count"
                ],
                "sc_unsupported_op_count": sc_default_metadata["sc_unsupported_op_count"],
                "estimation_model_coverage_status": sc_default_metadata[
                    "estimation_model_coverage_status"
                ],
                "estimation_model_coverage_reason": sc_default_metadata[
                    "estimation_model_coverage_reason"
                ],
                "estimation_model_support_boundary": sc_default_metadata[
                    "estimation_model_support_boundary"
                ],
                "estimation_model_supported_op_count": sc_default_metadata[
                    "estimation_model_supported_op_count"
                ],
                "estimation_model_unsupported_op_count": sc_default_metadata[
                    "estimation_model_unsupported_op_count"
                ],
                "estimation_model_ready_status": estimation_model_readiness[
                    "estimation_model_ready_status"
                ],
                "estimation_model_ready": estimation_model_readiness[
                    "estimation_model_ready"
                ],
                "estimation_model_ready_reason": estimation_model_readiness[
                    "estimation_model_ready_reason"
                ],
                "estimation_model_ready_blockers": estimation_model_readiness[
                    "estimation_model_ready_blockers"
                ],
                "sc_fail_mode": sc_trust_contract_cfg["fail_mode"],
                "sc_fail_closed_triggered": sc_fail_closed_triggered,
                "sc_default_status": sc_default_status,
                "dark_launch_enabled": dark_launch_cfg["enabled"],
                "dark_launch_candidate_label": dark_launch_cfg["candidate_label"],
                "dark_launch_comparator_label": dark_launch_cfg["comparator_label"],
                "dark_launch_comparator_execution_semantics": dark_launch_cfg[
                    "comparator_execution_semantics"
                ]
                if dark_launch_cfg["enabled"]
                else None,
                "dark_launch_comparator_trust_posture": dark_launch_comparator_metadata[
                    "sc_summary_trust_posture"
                ],
                "acc_ref_top1": acc_ref_top1,
                "acc_top1": acc_top1,
                "acc_drop_pp": acc_drop_pp,
                "accuracy_source_csv": accuracy_provenance.get("accuracy_source_csv"),
                "accuracy_baseline_row_id": accuracy_provenance.get("accuracy_baseline_row_id"),
                "accuracy_target_row_id": accuracy_provenance.get("accuracy_target_row_id"),
                "accuracy_baseline_source_run_id": accuracy_provenance.get("accuracy_baseline_source_run_id"),
                "accuracy_target_source_run_id": accuracy_provenance.get("accuracy_target_source_run_id"),
                "accuracy_target_split": accuracy_provenance.get("accuracy_target_split"),
                "accuracy_target_notes": accuracy_provenance.get("accuracy_target_notes"),
                "delta_pp_budget": delta_pp_budget,
                "pass_delta": pass_delta,
                "core_latency_ms": core_latency_ms,
                "latency_ms": latency_ms,
                "system_latency_lower_ms": system_latency_lower_ms,
                "system_latency_upper_ms": system_latency_upper_ms,
                "throughput_images_s": throughput_images_s,
                "throughput_tokens_s": throughput_tokens_s,
                "speedup_vs_E0": speedup_vs_e0,
                "core_energy_j": core_energy_j,
                "energy_j": energy_j,
                "system_energy_lower_j": system_energy_lower_j,
                "system_energy_upper_j": system_energy_upper_j,
                "avg_power_w": avg_power_w,
                "core_avg_power_w": core_avg_power_w,
                "tops_w": tops_w,
                "energy_breakdown_conversion_control_j": breakdown_conversion,
                "energy_breakdown_memory_move_j": breakdown_memory,
                "energy_breakdown_oe_j": breakdown_oe,
                "energy_breakdown_adc_pca_j": breakdown_adc,
                "energy_breakdown_laser_optical_j": breakdown_laser,
                "energy_breakdown_other_static_j": breakdown_other,
                "integrated_onchip_comm_j": selected_integrated_costs["integrated_onchip_comm_j"],
                "integrated_control_sched_j": selected_integrated_costs["integrated_control_sched_j"],
                "integrated_host_staging_j": selected_integrated_costs["integrated_host_staging_j"],
                "integrated_calibration_monitoring_j": selected_integrated_costs["integrated_calibration_monitoring_j"],
                "integrated_hidden_system_cost_j": selected_integrated_costs["integrated_hidden_system_cost_j"],
                "integrated_hidden_system_cost_lower_j": selected_integrated_costs["integrated_hidden_system_cost_lower_j"],
                "integrated_hidden_system_cost_upper_j": selected_integrated_costs["integrated_hidden_system_cost_upper_j"],
                "integrated_onchip_comm_ms": selected_integrated_costs["integrated_onchip_comm_ms"],
                "integrated_control_sched_ms": selected_integrated_costs["integrated_control_sched_ms"],
                "integrated_host_staging_ms": selected_integrated_costs["integrated_host_staging_ms"],
                "integrated_calibration_monitoring_ms": selected_integrated_costs["integrated_calibration_monitoring_ms"],
                "integrated_hidden_system_latency_ms": selected_integrated_costs["integrated_hidden_system_latency_ms"],
                "integrated_hidden_system_latency_lower_ms": selected_integrated_costs["integrated_hidden_system_latency_lower_ms"],
                "integrated_hidden_system_latency_upper_ms": selected_integrated_costs["integrated_hidden_system_latency_upper_ms"],
                "integrated_system_cost_mode": selected_integrated_costs["integrated_system_cost_mode"],
                "integrated_onchip_comm_evidence": realism_assessment["integrated_onchip_comm_evidence"],
                "integrated_control_sched_evidence": realism_assessment["integrated_control_sched_evidence"],
                "integrated_host_staging_evidence": realism_assessment["integrated_host_staging_evidence"],
                "integrated_calibration_monitoring_evidence": realism_assessment["integrated_calibration_monitoring_evidence"],
                "integrated_system_cost_evidence": realism_assessment["integrated_system_cost_evidence"],
                "integrated_system_cost_calibration_source": realism_assessment["integrated_system_cost_calibration_source"],
                "integrated_system_cost_uncertainty_method": realism_assessment["integrated_system_cost_uncertainty_method"],
                "flow_timeline_evidence": realism_assessment["flow_timeline_evidence"],
                "flow_timeline_calibration_source": realism_assessment["flow_timeline_calibration_source"],
                "meso_cost_evidence": realism_assessment["meso_cost_evidence"],
                "meso_cost_calibration_source": realism_assessment["meso_cost_calibration_source"],
                "phy_support_evidence": realism_assessment["phy_support_evidence"],
                "phy_support_calibration_source": realism_assessment["phy_support_calibration_source"],
                "accuracy_coupling_evidence": realism_assessment["accuracy_coupling_evidence"],
                "accuracy_coupling_metric": realism_assessment["accuracy_coupling_metric"],
                "accuracy_coupling_source": realism_assessment["accuracy_coupling_source"],
                "accuracy_coupling_reason": realism_assessment["accuracy_coupling_reason"],
                "accuracy_measurement_contract_status": realism_assessment[
                    "accuracy_measurement_contract_status"
                ],
                "accuracy_measurement_contract_reason": realism_assessment[
                    "accuracy_measurement_contract_reason"
                ],
                "accuracy_measurement_contract_source": realism_assessment[
                    "accuracy_measurement_contract_source"
                ],
                "accuracy_measurement_contract_truth_class": realism_assessment[
                    "accuracy_measurement_contract_truth_class"
                ],
                "accuracy_measurement_contract_authorization_note": realism_assessment[
                    "accuracy_measurement_contract_authorization_note"
                ],
            "accuracy_measurement_contract_authorization_status": realism_assessment[
                "accuracy_measurement_contract_authorization_status"
            ],
            "accuracy_measurement_contract_conv_measured_package_path": realism_assessment[
                "accuracy_measurement_contract_conv_measured_package_path"
            ],
            "accuracy_measurement_contract_conv_measured_package_sha256": realism_assessment[
                "accuracy_measurement_contract_conv_measured_package_sha256"
            ],
            "accuracy_measurement_contract_required_truth_class": realism_assessment[
                "accuracy_measurement_contract_required_truth_class"
            ],
                "accuracy_measurement_contract_required_fields_json": realism_assessment[
                    "accuracy_measurement_contract_required_fields_json"
                ],
                "accuracy_measurement_contract_observed_fields_json": realism_assessment[
                    "accuracy_measurement_contract_observed_fields_json"
                ],
                "accuracy_measurement_contract_violations_json": realism_assessment[
                    "accuracy_measurement_contract_violations_json"
                ],
                "accuracy_evidence_tier": realism_assessment["accuracy_evidence_tier"],
                "analysis_grade_ready": realism_assessment["analysis_grade_ready"],
                "analysis_grade_blockers": realism_assessment["analysis_grade_blockers"],
                "realism_class": realism_assessment["realism_class"],
                "proxy_promotion_ready": realism_assessment["proxy_promotion_ready"],
                "proxy_upgrade_blockers": realism_assessment["proxy_upgrade_blockers"],
                "benchmark_claim_ready": realism_assessment["benchmark_claim_ready"],
                "device_comparison_scope": realism_assessment["device_comparison_scope"],
                "benchmark_equivalence": realism_assessment["benchmark_equivalence"],
                "comparison_boundary": reporting_boundaries["comparison_boundary"],
                "latency_boundary": reporting_boundaries["latency_boundary"],
                "energy_boundary": reporting_boundaries["energy_boundary"],
                "power_boundary": reporting_boundaries["power_boundary"],
                "retained_energy_mode": retained_energy_mode,
                "energy_upperbound_j": energy_upperbound_j,
                "energy_countbased_j": energy_countbased_j,
                "bsl_max": sc_det_cfg.get("bsl_max"),
                "det_policy": det_runtime_metadata.get("det_policy"),
                "det_k_signature": det_runtime_metadata.get("det_k_signature"),
                "det_runtime_enabled": det_runtime_enabled,
                "det_quality_gate_enabled": det_runtime_metadata.get("det_quality_gate_enabled"),
                "det_quality_gate_policy": det_runtime_metadata.get("det_quality_gate_policy"),
                "det_quality_gate_status": det_runtime_metadata.get("det_quality_gate_status"),
                "det_quality_gate_reason": det_runtime_metadata.get("det_quality_gate_reason"),
                "det_quality_gate_fallback_policy": det_runtime_metadata.get(
                    "det_quality_gate_fallback_policy"
                ),
                "det_quality_gate_require_measured_accuracy": det_runtime_metadata.get(
                    "det_quality_gate_require_measured_accuracy"
                ),
                "det_quality_gate_measured_accuracy_ready": det_runtime_metadata.get(
                    "det_quality_gate_measured_accuracy_ready"
                ),
                "det_quality_gate_max_prefix_error_mean": det_runtime_metadata.get(
                    "det_quality_gate_max_prefix_error_mean"
                ),
                "det_quality_gate_max_prefix_error_p95": det_runtime_metadata.get(
                    "det_quality_gate_max_prefix_error_p95"
                ),
                "det_prefix_error_mean": det_runtime_metadata.get("det_prefix_error_mean"),
                "det_prefix_error_p95": det_runtime_metadata.get("det_prefix_error_p95"),
                "avg_effective_bsl": avg_effective_bsl,
                "k_i": k_i,
                "tau_i": tau_i,
                "duty_cycle_avg": sparse_scale if switches["sparse"] else 1.0,
                "sparse_active_fraction": sparse_scale if switches["sparse"] else 1.0,
                "sparse_scale_source": sparse_scale_source,
                "sparse_measured_activity_fraction": accuracy_provenance.get(
                    "sparse_measured_activity_fraction"
                ),
                "det_overhead_j": det_overhead_j,
                "det_saved_j": det_saved_j,
                "det_net_gain_j": det_net_gain_j,
                "pass_det_net_gain": pass_det_net_gain,
                "stage_cycles": stage_cycles,
                "bubble_cycles": bubble_cycles,
                "utilization_avg": utilization_avg,
                **flow_summary_fields,
                "hops_scheduler_mode": (
                    flow_summary_fields["flow_model_mode"] if switches["flow"] else "disabled"
                ),
                "fanout": meso_metrics["fanout"],
                "topology_dimension": meso_metrics["topology_dimension"],
                "serializers_saved": meso_metrics["serializers_saved"],
                "serializer_energy_j": meso_metrics["serializer_energy_j"],
                "broadcast_driver_energy_j": meso_metrics["broadcast_driver_energy_j"],
                "fabric_control_overhead_j": meso_metrics["fabric_control_overhead_j"],
                "extra_buffering_overhead_j": meso_metrics["extra_buffering_overhead_j"],
                "explicit_total_cost_j": meso_metrics["explicit_total_cost_j"],
                "explicit_total_savings_j": meso_metrics["explicit_total_savings_j"],
                "meso_cost_model_mode": meso_metrics["cost_model_mode"],
                "net_energy_gain_j": meso_metrics["net_energy_gain_j"],
                "N_wdm": n_wdm,
                "ER_db": er_db,
                "BER_target": ber_target,
                "Loss_path_db": loss_path_db,
                "PP_crosstalk_db": pp_crosstalk_db,
                "P_laser_dbm": p_laser_dbm,
                "P_laser_mw": p_laser_mw,
                "phy_link_budget_status": (
                    "ready" if switches["phy"] and phy_result is not None else "disabled"
                ),
                "gaussian_noise_std_ref": p1_align_resolved["gaussian_noise_std_ref"],
                "crosstalk_alpha_ref": p1_align_resolved["crosstalk_alpha_ref"],
                "p1_align_method": p1_align_resolved["method"],
                "p1_align_fit_error": p1_align_resolved["fit_error"],
                "phy_penalty_table_version": phy_penalty_table_version,
                "s_wg_min": s_wg_min,
                "P_thermal_tuning": thermal_tuning_mw,
            }
        )

        per_layer_accuracy_rows.extend(model_accuracy_rows)
        per_layer_timeline_rows.extend(model_timeline_rows)
        per_layer_buffer_rows.extend(model_buffer_rows)
        per_layer_phy_rows.extend(model_phy_rows)

    summary_path = out_dir / "phase1_summary.csv"
    _write_csv(summary_path, PHASE1_SUMMARY_FIELDS, summaries)

    master_path = out_dir / "master_metrics.csv"
    _write_csv(master_path, MASTER_FIELDS, master_rows)

    timeline_summary_rows = [
        {
            "model": row.get("model"),
            "execution_semantics": row.get("execution_semantics"),
            "bitstream_enabled": row.get("bitstream_enabled"),
            "bitstream_capture_manifest_csv": row.get("bitstream_capture_manifest_csv"),
            "sc_summary_trust_posture": row.get("sc_summary_trust_posture"),
            "true_sc_summary_claim_state": row.get("true_sc_summary_claim_state"),
            "estimation_model_ready_status": row.get("estimation_model_ready_status"),
            "estimation_model_ready": row.get("estimation_model_ready"),
            "core_latency_ms": row.get("core_latency_ms"),
            "latency_ms": row.get("latency_ms"),
            "integrated_hidden_system_latency_ms": row.get("integrated_hidden_system_latency_ms"),
            "stage_cycles": row.get("stage_cycles"),
            "bubble_cycles": row.get("bubble_cycles"),
            "utilization_avg": row.get("utilization_avg"),
            "flow_buffer_peak_cycles": row.get("flow_buffer_peak_cycles"),
            "flow_buffer_peak_frac": row.get("flow_buffer_peak_frac"),
            "flow_admission_stalls": row.get("flow_admission_stalls"),
            "flow_prefetch_hits": row.get("flow_prefetch_hits"),
            "flow_prefetch_drops": row.get("flow_prefetch_drops"),
            "flow_residency_hit_rate": row.get("flow_residency_hit_rate"),
            "flow_control_backpressure": row.get("flow_control_backpressure"),
            "flow_eviction_count": row.get("flow_eviction_count"),
        }
        for row in summaries
    ]
    timeline_summary_path = out_dir / "timeline_summary.csv"
    _write_csv(timeline_summary_path, TIMELINE_SUMMARY_FIELDS, timeline_summary_rows)
    dark_launch_summary_path = None
    if dark_launch_rows:
        dark_launch_summary_path = out_dir / "dark_launch_comparison.csv"
        _write_csv(dark_launch_summary_path, DARK_LAUNCH_FIELDS, dark_launch_rows)

    p_laser_dbm_ref = None
    for row in summaries:
        p_laser_dbm_ref = _to_float(row.get("P_laser_dbm"), None)
        if p_laser_dbm_ref is not None:
            break
    if p_laser_dbm_ref is None:
        p_laser_dbm_ref = _to_float(p1_align_cfg.get("p_laser_dbm_ref"), None)
    if p_laser_dbm_ref is None:
        p_laser_dbm_ref = _to_float(phy_cfg.get("p_sensitivity_dbm"), None)

    p0_p1_alignment_rows = _build_p0_p1_alignment_rows(
        points_db=p1_align_resolved.get("points_db") or [],
        p_laser_dbm_ref=p_laser_dbm_ref,
        gaussian_noise_std_ref=float(p1_align_resolved["gaussian_noise_std_ref"]),
        crosstalk_alpha_ref=float(p1_align_resolved["crosstalk_alpha_ref"]),
        p1_align_cfg=p1_align_cfg,
    )
    p0_p1_alignment_path = out_dir / "p0_p1_alignment.csv"
    _write_csv(p0_p1_alignment_path, P0_P1_ALIGNMENT_FIELDS, p0_p1_alignment_rows)

    if _to_bool(outputs_cfg.get("save_layer_tables"), True):
        _write_csv(
            out_dir / "per_layer_accuracy.csv",
            ["layer_id", "k_i", "tau_i", "delta_acc_layer_pp"],
            per_layer_accuracy_rows,
        )
        _write_csv(
            out_dir / "per_layer_timeline.csv",
            ["layer_id", "stage", "cycles", "utilization"],
            per_layer_timeline_rows,
        )
        _write_csv(
            out_dir / "flow_buffer_trace.csv",
            FLOW_BUFFER_TRACE_FIELDS,
            per_layer_buffer_rows,
        )
        _write_csv(
            out_dir / "per_layer_phy.csv",
            ["layer_id", "active_channels", "duty_cycle", "margin_eff_db"],
            per_layer_phy_rows,
        )

    if switches["phy"] and phy_budget_path is not None:
        _dump_json(phy_budget_path, phy_budget_rows)
        if phy_sweep_rows and phy_sweep_path is not None:
            _write_phy_sweep(phy_sweep_path, phy_sweep_rows)

    calibration_log_path = None
    if _to_bool(outputs_cfg.get("save_calibration_log"), True):
        fit_error = _to_float(p1_align_cfg.get("fit_error"), None)
        calibration_rows = _build_calibration_log_rows(
            cfg=cfg,
            effective_k=effective_k,
            sparse_scale=(
                json.dumps(
                    {
                        "default_sparse_scale": default_sparse_scale,
                        "default_sparse_scale_source": default_sparse_scale_source,
                        "measured_activity_by_model": {
                            row["model"]: row["duty_cycle"]
                            for row in phy_budget_rows
                        }
                        if phy_budget_rows
                        else None,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                if switches["sparse"]
                else default_sparse_scale
            ),
            fit_error=fit_error,
        )
        calibration_log_path = out_dir / "calibration_log.csv"
        _write_csv(
            calibration_log_path,
            [
                "calib_manifest",
                "eval_manifest",
                "scan_grid",
                "selected_value",
                "selection_rule",
                "objective",
                "fit_error",
            ],
            calibration_rows,
        )

    det_prefix_error_path = None
    if switches["det"]:
        det_prefix_error_path = _write_det_prefix_error_table(
            out_dir=out_dir,
            sc_det_cfg=sc_det_cfg,
            effective_k=effective_k,
        )

    append_master = _to_bool(outputs_cfg.get("append_master"), True)
    global_master_path = outputs_cfg.get("master_csv") or "results/master_metrics.csv"
    global_master_resolved = resolve_workspace_path(global_master_path, anchor=ROOT_DIR)
    if append_master:
        _append_csv_schema_aware(global_master_resolved, MASTER_FIELDS, master_rows)

    metadata = {
        "run": run_cfg,
        "switches": switches,
        "det_runtime": det_runtime_metadata,
        "execution_semantics": execution_semantics_cfg["execution_semantics"],
        "execution_semantics_default": execution_semantics_cfg["execution_semantics_default"],
        "execution_semantics_origin": execution_semantics_cfg["execution_semantics_origin"],
        "bitstream": {
            **execution_semantics_cfg,
            "generator_policy_matrix_csv": str(
                (cfg.get("bitstream") or {}).get("generator_policy_matrix_csv") or ""
            ),
        },
        "sc_trust_contract": sc_trust_contract_cfg,
        "dark_launch": {
            **dark_launch_cfg,
            "comparison_csv": str(dark_launch_summary_path) if dark_launch_summary_path else None,
        },
        "realism": realism_cfg,
        "paper_modules": PAPER_MODULES,
        "accelerator_name": "Masking-the-Lag",
        "estimator_backend": "mtl-parametric",
        "local_device_policy": args.device,
        "config_path": str(cfg_path),
        "summary_csv": str(summary_path),
        "master_metrics_csv": str(master_path),
        "global_master_metrics_csv": str(global_master_resolved) if append_master else None,
        "ops_dir": str(ops_out_dir),
        "phy_budget": str(phy_budget_path) if switches["phy"] and phy_budget_path else None,
        "phy_sweep": str(phy_sweep_path) if switches["phy"] and phy_sweep_rows and phy_sweep_path else None,
        "per_layer_accuracy_csv": str(out_dir / "per_layer_accuracy.csv"),
        "per_layer_timeline_csv": str(out_dir / "per_layer_timeline.csv"),
        "per_layer_phy_csv": str(out_dir / "per_layer_phy.csv"),
        "timeline_summary_csv": str(timeline_summary_path),
        "dark_launch_comparison_csv": str(dark_launch_summary_path) if dark_launch_summary_path else None,
        "p0_p1_alignment_csv": str(p0_p1_alignment_path),
        "calibration_log_csv": str(calibration_log_path) if calibration_log_path else None,
        "det_prefix_error_csv": str(det_prefix_error_path) if det_prefix_error_path else None,
        "energy_model_mode": mode,
        "retained_energy_mode": retained_energy_mode,
        "integrated_system_cost_mode": integrated_system_cost_cfg.get("mode"),
        "comparison_boundary": _resolve_reporting_boundaries(
            execution_semantics_cfg["execution_semantics"]
        )["comparison_boundary"],
        "latency_boundary": _resolve_reporting_boundaries(
            execution_semantics_cfg["execution_semantics"]
        )["latency_boundary"],
        "energy_boundary": _resolve_reporting_boundaries(
            execution_semantics_cfg["execution_semantics"]
        )["energy_boundary"],
        "power_boundary": _resolve_reporting_boundaries(
            execution_semantics_cfg["execution_semantics"]
        )["power_boundary"],
        "realism_assessment": summaries[0]
        if len(summaries) == 1
        else [
            {
                "model": row.get("model"),
                "realism_class": row.get("realism_class"),
                "proxy_promotion_ready": row.get("proxy_promotion_ready"),
                "proxy_upgrade_blockers": row.get("proxy_upgrade_blockers"),
                "fidelity_ready": row.get("fidelity_ready"),
                "fidelity_blockers": row.get("fidelity_blockers"),
                "benchmark_claim_ready": row.get("benchmark_claim_ready"),
                "integrated_system_cost_evidence": row.get("integrated_system_cost_evidence"),
                "flow_timeline_evidence": row.get("flow_timeline_evidence"),
                "meso_cost_evidence": row.get("meso_cost_evidence"),
                "phy_support_evidence": row.get("phy_support_evidence"),
                "accuracy_coupling_evidence": row.get("accuracy_coupling_evidence"),
                "accuracy_coupling_metric": row.get("accuracy_coupling_metric"),
                "accuracy_coupling_source": row.get("accuracy_coupling_source"),
                "accuracy_coupling_reason": row.get("accuracy_coupling_reason"),
                "accuracy_measurement_contract_status": row.get(
                    "accuracy_measurement_contract_status"
                ),
                "accuracy_measurement_contract_reason": row.get(
                    "accuracy_measurement_contract_reason"
                ),
                "accuracy_measurement_contract_source": row.get(
                    "accuracy_measurement_contract_source"
                ),
                "accuracy_evidence_tier": row.get("accuracy_evidence_tier"),
                "analysis_grade_ready": row.get("analysis_grade_ready"),
                "analysis_grade_blockers": row.get("analysis_grade_blockers"),
                "true_sc_claim_surface_status": row.get("true_sc_claim_surface_status"),
                "true_sc_claim_surface_governed_support_op_count": row.get(
                    "true_sc_claim_surface_governed_support_op_count"
                ),
                "true_sc_support_out_of_surface_op_count": row.get(
                    "true_sc_support_out_of_surface_op_count"
                ),
                "device_comparison_scope": row.get("device_comparison_scope"),
                "benchmark_equivalence": row.get("benchmark_equivalence"),
            }
            for row in summaries
        ],
        "model_governance_summary": _build_model_governance_summary(
            summaries,
            dark_launch_rows,
        ),
        "dark_launch_assessment": dark_launch_rows[0]
        if len(dark_launch_rows) == 1
        else dark_launch_rows,
        "p1_align": p1_align_resolved,
    }
    _dump_json(out_dir / "run_metadata.json", metadata)
    print(f"Saved Phase-1 summary: {summary_path}")
    print(f"Saved master metrics: {master_path}")


if __name__ == "__main__":
    main()
