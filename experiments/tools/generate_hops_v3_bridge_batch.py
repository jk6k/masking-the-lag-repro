#!/usr/bin/env python3
"""Generate reintegration-aware HOPS v3 bridge configs for 2026-04-20.

This stays inside the proven 4x4/8x8 spill neighborhood and emits both
bounded E2 proxy configs and the paired reintegration replay overlays so the
manager can schedule reruns without touching shared launchers.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from copy import deepcopy
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROXY_TEMPLATE_PATH = REPO_ROOT / "configs" / "phase1_true_sc_e0_e6_canonical_template_20260418.yaml"
DEFAULT_REINTEGRATION_TEMPLATE_PATH = REPO_ROOT / "configs" / "fuller_det_sparse_reentry_slice_template_20260331.yaml"
DEFAULT_PROXY_OUT_DIR = REPO_ROOT / "experiments" / "results" / "generated_configs" / "20260420_hops_v3_bridge_batch"
DEFAULT_REINTEGRATION_OUT_DIR = (
    REPO_ROOT / "experiments" / "results" / "generated_configs" / "20260420_hops_v3_bridge_reintegration_batch"
)
DEFAULT_RUN_PREFIX = "20260420"
DEFAULT_COMPARISON_PATH = DEFAULT_PROXY_OUT_DIR / "reintegration_comparison.csv"
DEFAULT_BRIDGE_SCORECARD_PATH = DEFAULT_PROXY_OUT_DIR / "bridge_scorecard.csv"
DEFAULT_BRIDGE_PLAN_PATH = DEFAULT_PROXY_OUT_DIR / "bridge_plan.csv"
DEFAULT_MANAGER_EVAL_SURFACE_PATH = DEFAULT_PROXY_OUT_DIR / "manager_eval_surface.csv"
DEFAULT_REVIEW_INTAKE_SURFACE_PATH = DEFAULT_PROXY_OUT_DIR / "review_intake_surface.csv"
DEFAULT_PRIORITY_CROSSWALK_PATH = DEFAULT_PROXY_OUT_DIR / "priority_execution_crosswalk.csv"
DEFAULT_PRIORITY1_PACKET_PATH = DEFAULT_PROXY_OUT_DIR / "priority1_execution_packet.md"
DEFAULT_PACKET_INDEX_PATH = DEFAULT_PROXY_OUT_DIR / "priority_execution_packet_index.md"
DEFAULT_MANAGER_SLOT_RECOMMENDATION_PATH = DEFAULT_PROXY_OUT_DIR / "manager_slot_recommendation_20260421.md"
DEFAULT_MPS_PYTHON = "./.venv311-mps/bin/python"
DEFAULT_PHASE1_RUNNER = "experiments/tools/phase1_runner.py"
DEFAULT_CAFFEINATE_PREFIX = "caffeinate -dimsu"
SUPPRESSED_PCT_LIMIT = 1_000_000.0
PROXY_HEDGE_CONFIG_ID = "hopsv3_b4p3e4_8x8_spill"
PRIMARY_WINNER_CONFIG_ID = "hopsv3_b5p4e5_4x4_spill"
PROXY_HEDGE_SPEC = {
    "buffer_depth": 4,
    "prefetch_credits": 3,
    "execute_credits": 4,
    "tile_rows": 8,
    "tile_cols": 8,
    "control_issue_width": 5,
    "prefetch_distance": 2,
    "exception_lane_policy": "spill",
}

REFERENCE_PATHS = {
    "reintegration_note": REPO_ROOT / "docs" / "reports" / "20260420_hops_v3_batch2_reintegration_note.md",
    "batch2_results": (
        REPO_ROOT / "experiments" / "results" / "generated_configs" / "20260420_hops_v3_sweep_batch2" / "results_summary.csv"
    ),
    "primary_reintegration_results": (
        REPO_ROOT
        / "experiments"
        / "results"
        / "generated_configs"
        / "20260420_hops_v3_reintegration_batch"
        / "results_summary.csv"
    ),
    "tile8_reintegration_results": (
        REPO_ROOT
        / "experiments"
        / "results"
        / "generated_configs"
        / "20260420_hops_v3_reintegration_batch_tile8"
        / "results_summary.csv"
    ),
    "hopsv2_reference": (
        REPO_ROOT
        / "experiments"
        / "results"
        / "generated_configs"
        / "20260420_hops_v2_sweep_batch1"
        / "hopsv2_ref.yaml"
    ),
}

GENERATOR_SCRIPT_PATH = Path(__file__).resolve()


BRIDGE_SPECS = [
    {
        "priority": 1,
        "config_id": "hopsv3_bridge_b4p3e4_8x8_pd1_spill",
        "purpose": "8x8 spill hedge with shorter prefetch distance to isolate replay eagerness",
        "anchor": "8x8 hedge with lighter prefetch pacing",
        "proxy_hypothesis": "Keeps the safe 8x8 geometry and credit pair but shortens prefetch distance to test whether the remaining FULLER tax is driven by replay eagerness.",
        "replay_goal": "Lowest-risk 8x8 factor probe focused on hidden-latency and energy trimming before changing control issuance or credits.",
        "decision_bucket": "safe_prefetch_probe",
        "recommended_next_action": "Run first as the narrowest 8x8 perturbation; if E2 stays pressure-free, replay all three reintegration lanes and compare FULLER hidden latency and energy against the current hedge.",
        "file_name": "hopsv3_bridge_b4p3e4_8x8_pd1_spill.yaml",
        "buffer_depth": 4,
        "prefetch_credits": 3,
        "execute_credits": 4,
        "tile_rows": 8,
        "tile_cols": 8,
        "control_issue_width": 5,
        "prefetch_distance": 1,
        "exception_lane_policy": "spill",
    },
    {
        "priority": 2,
        "config_id": "hopsv3_bridge_b4p3e4_8x8_cw4_spill",
        "purpose": "8x8 spill hedge with narrower issue width to isolate control pacing",
        "anchor": "8x8 hedge with lighter control issuance",
        "proxy_hypothesis": "Keeps the safe 8x8 geometry and credit pair but narrows control issuance to test whether replay cost is still being driven by control pacing even on the zero-stall hedge.",
        "replay_goal": "Second 8x8 factor probe focused on control pacing after the prefetch-isolation run.",
        "decision_bucket": "safe_control_probe",
        "recommended_next_action": "Run after the prefetch-isolation probe if FULLER still carries the hedge tax; it isolates control-width impact without changing credits or geometry.",
        "file_name": "hopsv3_bridge_b4p3e4_8x8_cw4_spill.yaml",
        "buffer_depth": 4,
        "prefetch_credits": 3,
        "execute_credits": 4,
        "tile_rows": 8,
        "tile_cols": 8,
        "control_issue_width": 4,
        "prefetch_distance": 2,
        "exception_lane_policy": "spill",
    },
    {
        "priority": 3,
        "config_id": "hopsv3_bridge_b4p3e4_8x8_cw4pd1_spill",
        "purpose": "8x8 spill hedge with the older gentler issue and prefetch pacing",
        "anchor": "8x8 hedge retaining spill while relaxing control pacing",
        "proxy_hypothesis": "Combines the two gentlest 8x8 pacing relaxations after the single-factor probes so the manager can test whether both levers together are needed to trim the hedge tax.",
        "replay_goal": "Joint 8x8 comparator after the factorized probes; use it to confirm whether pacing effects compound before lifting credits or shrinking tiles.",
        "decision_bucket": "joint_safe_probe",
        "recommended_next_action": "Use after the single-factor 8x8 probes if both stay safe but neither closes the FULLER gap alone; it tests the combined pacing relaxation before credit lifting or any 4x4 rescue.",
        "file_name": "hopsv3_bridge_b4p3e4_8x8_cw4pd1_spill.yaml",
        "buffer_depth": 4,
        "prefetch_credits": 3,
        "execute_credits": 4,
        "tile_rows": 8,
        "tile_cols": 8,
        "control_issue_width": 4,
        "prefetch_distance": 1,
        "exception_lane_policy": "spill",
    },
    {
        "priority": 4,
        "config_id": "hopsv3_bridge_b4p4e5_8x8_spill",
        "purpose": "8x8 spill bridge that lifts credits without adding the batch-2 extra buffer",
        "anchor": "8x8 hedge nudged toward the 4x4 primary credit pair",
        "proxy_hypothesis": "Tests whether credit lifting alone is still safe on the reintegration-friendly 8x8 geometry once the pacing levers have been decomposed.",
        "replay_goal": "Higher-risk 8x8-local follow-up after the factorized pacing probes, mainly to see whether extra credits buy back latency without reopening pressure.",
        "decision_bucket": "higher_risk_8x8_credit_lift",
        "recommended_next_action": "Keep as the last 8x8-local probe; run it only after the three lower-risk 8x8 points if the hedge surface stays safe but still leaves too much latency tax.",
        "file_name": "hopsv3_bridge_b4p4e5_8x8_spill.yaml",
        "buffer_depth": 4,
        "prefetch_credits": 4,
        "execute_credits": 5,
        "tile_rows": 8,
        "tile_cols": 8,
        "control_issue_width": 5,
        "prefetch_distance": 2,
        "exception_lane_policy": "spill",
    },
    {
        "priority": 5,
        "config_id": "hopsv3_bridge_b4p4e5_4x4_cw4_spill",
        "purpose": "4x4 spill bridge with narrower issue width to ease reintegration pressure",
        "anchor": "4x4 bridge with lighter control issuance",
        "proxy_hypothesis": "Keeps the bridge credit pair but narrows control issuance so 4x4 replay pressure may collapse before tile geometry changes.",
        "replay_goal": "Primary 4x4 rescue attempt for MESO/PHY/FULLER after the 8x8-local ladder is exhausted.",
        "decision_bucket": "primary_4x4_rescue",
        "recommended_next_action": "Only enter after the full 8x8-local ladder stays pressure-safe yet still misses the FULLER gate; this is the first 4x4 rescue attempt if a tighter tile is still required.",
        "file_name": "hopsv3_bridge_b4p4e5_4x4_cw4_spill.yaml",
        "buffer_depth": 4,
        "prefetch_credits": 4,
        "execute_credits": 5,
        "tile_rows": 4,
        "tile_cols": 4,
        "control_issue_width": 4,
        "prefetch_distance": 2,
        "exception_lane_policy": "spill",
    },
    {
        "priority": 6,
        "config_id": "hopsv3_bridge_b4p4e5_4x4_pd1_spill",
        "purpose": "4x4 spill bridge with shorter prefetch distance to reduce replay eagerness",
        "anchor": "4x4 bridge with lighter prefetch pacing",
        "proxy_hypothesis": "Reduces prefetch eagerness while keeping the bridge credit envelope to test whether reintegration stalls are pacing-driven.",
        "replay_goal": "Secondary 4x4 rescue attempt focused on replay pacing rather than control issuance once the 8x8-local probes are exhausted.",
        "decision_bucket": "pacing_probe",
        "recommended_next_action": "Use after the control-width 4x4 rescue attempt if the lane still needs a 4x4 rescue; it isolates prefetch pacing as the remaining reintegration lever.",
        "file_name": "hopsv3_bridge_b4p4e5_4x4_pd1_spill.yaml",
        "buffer_depth": 4,
        "prefetch_credits": 4,
        "execute_credits": 5,
        "tile_rows": 4,
        "tile_cols": 4,
        "control_issue_width": 5,
        "prefetch_distance": 1,
        "exception_lane_policy": "spill",
    },
    {
        "priority": 7,
        "config_id": "hopsv3_bridge_b4p4e5_4x4_spill",
        "purpose": "historical 4x4 spill bridge point: batch-1 spill credits with batch-2 spill policy",
        "anchor": "between batch-2 4x4 primary and 8x8 hedge",
        "proxy_hypothesis": "Restores the older 4x4 spill budget as the direct midpoint between the stalled 4x4 winner and the safe 8x8 hedge.",
        "replay_goal": "Final 4x4 midpoint probe if both the 8x8-local ladder and the targeted 4x4 rescues still miss.",
        "decision_bucket": "historical_midpoint",
        "recommended_next_action": "Use as the explicit midpoint replay only after the factorized 8x8 ladder and targeted 4x4 probes; it checks whether the older spill envelope is the real reintegration pressure boundary.",
        "file_name": "hopsv3_bridge_b4p4e5_4x4_spill.yaml",
        "buffer_depth": 4,
        "prefetch_credits": 4,
        "execute_credits": 5,
        "tile_rows": 4,
        "tile_cols": 4,
        "control_issue_width": 5,
        "prefetch_distance": 2,
        "exception_lane_policy": "spill",
    },
]


REINTEGRATION_VARIANTS = [
    {
        "variant_order": 1,
        "variant_id": "FLOW_MESO",
        "experiment_id": "E1",
        "run_suffix": "flow_meso",
        "file_suffix": "flow_meso",
        "notes_suffix": "FLOW_MESO lane",
        "variant_goal": "Recover toward the historical E1 latency envelope without any admission stalls or control backpressure.",
        "switches": {"meso": True, "flow": True, "det": False, "sparse": False, "phy": False},
    },
    {
        "variant_order": 2,
        "variant_id": "FLOW_PHY",
        "experiment_id": "E5",
        "run_suffix": "flow_phy",
        "file_suffix": "flow_phy",
        "notes_suffix": "FLOW_PHY lane",
        "variant_goal": "Recover toward the historical E5 latency envelope without any admission stalls or control backpressure.",
        "switches": {"meso": False, "flow": True, "det": False, "sparse": False, "phy": True},
    },
    {
        "variant_order": 3,
        "variant_id": "FULLER",
        "experiment_id": "FULLER_REENTRY_V1",
        "run_suffix": "fuller",
        "file_suffix": "fuller",
        "notes_suffix": "FULLER reentry lane",
        "variant_goal": "Clear replay pressure while reducing FULLER latency, hidden latency, and energy versus the current 8x8 hedge replay.",
        "switches": {"meso": True, "flow": True, "det": True, "sparse": True, "phy": True},
    },
]


MEASURED_REINTEGRATION_COMPARISON_FIELDNAMES = [
    "variant_order",
    "variant_id",
    "experiment_id",
    "bridge_gate",
    "historical_run_id",
    "stalled_run_id",
    "hedge_run_id",
    "historical_bubble_cycles",
    "stalled_bubble_cycles",
    "hedge_bubble_cycles",
    "historical_utilization_avg",
    "stalled_utilization_avg",
    "hedge_utilization_avg",
    "historical_latency_ms",
    "stalled_latency_ms",
    "hedge_latency_ms",
    "historical_hidden_latency_ms",
    "stalled_hidden_latency_ms",
    "hedge_hidden_latency_ms",
    "historical_energy_j",
    "stalled_energy_j",
    "hedge_energy_j",
    "historical_flow_admission_stalls",
    "stalled_flow_admission_stalls",
    "hedge_flow_admission_stalls",
    "historical_flow_control_backpressure",
    "stalled_flow_control_backpressure",
    "hedge_flow_control_backpressure",
    "historical_flow_residency_hit_rate",
    "stalled_flow_residency_hit_rate",
    "hedge_flow_residency_hit_rate",
    "historical_flow_prefetch_hits",
    "stalled_flow_prefetch_hits",
    "hedge_flow_prefetch_hits",
    "historical_flow_eviction_count",
    "stalled_flow_eviction_count",
    "hedge_flow_eviction_count",
    "stalled_vs_historical_bubble_cycles_pct",
    "hedge_vs_stalled_bubble_cycles_pct",
    "hedge_vs_historical_bubble_cycles_pct",
    "stalled_vs_historical_utilization_pp",
    "hedge_vs_stalled_utilization_pp",
    "hedge_vs_historical_utilization_pp",
    "stalled_vs_historical_latency_pct",
    "hedge_vs_stalled_latency_pct",
    "hedge_vs_historical_latency_pct",
    "stalled_vs_historical_hidden_latency_pct",
    "hedge_vs_stalled_hidden_latency_pct",
    "hedge_vs_historical_hidden_latency_pct",
    "stalled_vs_historical_hidden_latency_suppressed",
    "hedge_vs_historical_hidden_latency_suppressed",
    "stalled_vs_historical_energy_pct",
    "hedge_vs_stalled_energy_pct",
    "hedge_vs_historical_energy_pct",
]


REINTEGRATION_PACKAGE_COMMON_FIELDNAMES = [
    "config_id",
    "priority",
    "purpose",
    "entry_surface_kind",
    "entry_surface_path",
    "entry_surface_section",
    "anchor",
    "proxy_hypothesis",
    "replay_goal",
    "decision_bucket",
    "recommended_next_action",
    "proxy_run_id",
    "proxy_config_path",
    "proxy_launch_command",
    "proxy_review_artifact_path",
    "proxy_review_row_selector",
    "proxy_review_artifact_note",
    "proxy_acceptance_gate",
    "proxy_flow_admission_stalls_max",
    "proxy_flow_control_backpressure_max",
    "proxy_hedge_run_id",
    "proxy_hedge_latency_ms",
    "proxy_hedge_flow_admission_stalls",
    "proxy_hedge_flow_control_backpressure",
    "proxy_replay_promotion_note",
    "proxy_hedge_snapshot",
    "proxy_spec_delta_vs_hedge",
    "buffer_depth",
    "prefetch_credits",
    "execute_credits",
    "tile_rows",
    "tile_cols",
    "control_issue_width",
    "prefetch_distance",
    "exception_lane_policy",
]

REINTEGRATION_PACKAGE_VARIANT_SUFFIXES = [
    "run_id",
    "config_path",
    "launch_command",
    "review_artifact_path",
    "review_row_selector",
    "review_artifact_note",
    "bridge_gate",
    "historical_run_id",
    "stalled_run_id",
    "hedge_run_id",
    "flow_admission_stalls_max",
    "flow_control_backpressure_max",
    "latency_target_lt_ms",
    "hidden_latency_target_lt_ms",
    "energy_target_lt_j",
    "current_hedge_snapshot",
]

REINTEGRATION_PACKAGE_FIELDNAMES = REINTEGRATION_PACKAGE_COMMON_FIELDNAMES + [
    f"{variant['file_suffix']}_{suffix}"
    for variant in REINTEGRATION_VARIANTS
    for suffix in REINTEGRATION_PACKAGE_VARIANT_SUFFIXES
]


BRIDGE_SCORECARD_COMMON_FIELDNAMES = [
    "priority",
    "config_id",
    "purpose",
    "anchor",
    "proxy_hypothesis",
    "replay_goal",
    "decision_bucket",
    "recommended_next_action",
    "proxy_acceptance_gate",
    "proxy_flow_admission_stalls_max",
    "proxy_flow_control_backpressure_max",
    "proxy_hedge_run_id",
    "proxy_hedge_latency_ms",
    "proxy_hedge_flow_admission_stalls",
    "proxy_hedge_flow_control_backpressure",
    "proxy_replay_promotion_note",
    "proxy_hedge_snapshot",
    "proxy_spec_delta_vs_hedge",
    "proxy_run_id",
    "proxy_config_path",
    "buffer_depth",
    "prefetch_credits",
    "execute_credits",
    "tile_rows",
    "tile_cols",
    "control_issue_width",
    "prefetch_distance",
    "exception_lane_policy",
]

BRIDGE_SCORECARD_VARIANT_SUFFIXES = [
    "goal",
    "run_id",
    "config_path",
    "bridge_gate",
    "historical_run_id",
    "stalled_run_id",
    "hedge_run_id",
    "flow_admission_stalls_max",
    "flow_control_backpressure_max",
    "latency_target_lt_ms",
    "hidden_latency_target_lt_ms",
    "energy_target_lt_j",
    "historical_latency_ms",
    "stalled_latency_ms",
    "hedge_latency_ms",
    "hedge_vs_stalled_latency_pct",
    "hedge_vs_historical_latency_pct",
    "historical_hidden_latency_ms",
    "stalled_hidden_latency_ms",
    "hedge_hidden_latency_ms",
    "hedge_vs_stalled_hidden_latency_pct",
    "hedge_vs_historical_hidden_latency_pct",
    "hedge_vs_historical_hidden_latency_suppressed",
    "historical_energy_j",
    "stalled_energy_j",
    "hedge_energy_j",
    "hedge_vs_stalled_energy_pct",
    "hedge_vs_historical_energy_pct",
    "stalled_flow_admission_stalls",
    "hedge_flow_admission_stalls",
    "stalled_flow_control_backpressure",
    "hedge_flow_control_backpressure",
]

BRIDGE_SCORECARD_FIELDNAMES = BRIDGE_SCORECARD_COMMON_FIELDNAMES + [
    f"{variant['file_suffix']}_{suffix}"
    for variant in REINTEGRATION_VARIANTS
    for suffix in BRIDGE_SCORECARD_VARIANT_SUFFIXES
]


BRIDGE_PLAN_FIELDNAMES = [
    "launch_order",
    "stage",
    "priority",
    "config_id",
    "variant_order",
    "variant_id",
    "experiment_id",
    "run_id",
    "depends_on_run_id",
    "config_path",
    "launch_command",
    "purpose",
    "anchor",
    "decision_bucket",
    "hypothesis",
    "success_signal",
    "recommended_next_action",
    "buffer_depth",
    "prefetch_credits",
    "execute_credits",
    "tile_rows",
    "tile_cols",
    "control_issue_width",
    "prefetch_distance",
    "exception_lane_policy",
    "proxy_acceptance_gate",
    "proxy_hedge_run_id",
    "proxy_hedge_latency_ms",
    "proxy_hedge_flow_admission_stalls",
    "proxy_hedge_flow_control_backpressure",
    "proxy_replay_promotion_note",
    "proxy_hedge_snapshot",
    "proxy_spec_delta_vs_hedge",
    "bridge_gate",
    "flow_admission_stalls_max",
    "flow_control_backpressure_max",
    "latency_target_lt_ms",
    "hidden_latency_target_lt_ms",
    "energy_target_lt_j",
    "historical_run_id",
    "stalled_run_id",
    "hedge_run_id",
    "historical_latency_ms",
    "stalled_latency_ms",
    "hedge_latency_ms",
    "historical_hidden_latency_ms",
    "stalled_hidden_latency_ms",
    "hedge_hidden_latency_ms",
    "historical_energy_j",
    "stalled_energy_j",
    "hedge_energy_j",
    "stalled_flow_admission_stalls",
    "hedge_flow_admission_stalls",
    "stalled_flow_control_backpressure",
    "hedge_flow_control_backpressure",
    "stalled_vs_historical_latency_pct",
    "hedge_vs_stalled_latency_pct",
    "hedge_vs_historical_latency_pct",
    "stalled_vs_historical_hidden_latency_pct",
    "hedge_vs_stalled_hidden_latency_pct",
    "hedge_vs_historical_hidden_latency_pct",
    "stalled_vs_historical_hidden_latency_suppressed",
    "hedge_vs_historical_hidden_latency_suppressed",
    "stalled_vs_historical_energy_pct",
    "hedge_vs_stalled_energy_pct",
    "hedge_vs_historical_energy_pct",
]


MANAGER_EVAL_SURFACE_FIELDNAMES = [
    "config_id",
    "priority",
    "purpose",
    "entry_surface_kind",
    "entry_surface_path",
    "entry_surface_section",
    "anchor",
    "decision_bucket",
    "recommended_next_action",
    "proxy_run_id",
    "proxy_config_path",
    "proxy_launch_command",
    "proxy_review_artifact_path",
    "proxy_review_row_selector",
    "proxy_review_artifact_note",
    "proxy_acceptance_gate",
    "proxy_flow_admission_stalls_max",
    "proxy_flow_control_backpressure_max",
    "proxy_hedge_run_id",
    "proxy_hedge_latency_ms",
    "proxy_hedge_flow_admission_stalls",
    "proxy_hedge_flow_control_backpressure",
    "proxy_replay_promotion_note",
    "proxy_hedge_snapshot",
    "proxy_spec_delta_vs_hedge",
    "fuller_run_id",
    "fuller_config_path",
    "fuller_launch_command",
    "fuller_review_artifact_path",
    "fuller_review_row_selector",
    "fuller_review_artifact_note",
    "fuller_bridge_gate",
    "fuller_historical_run_id",
    "fuller_stalled_run_id",
    "fuller_hedge_run_id",
    "fuller_flow_admission_stalls_max",
    "fuller_flow_control_backpressure_max",
    "fuller_latency_target_lt_ms",
    "fuller_hidden_latency_target_lt_ms",
    "fuller_energy_target_lt_j",
    "fuller_historical_latency_ms",
    "fuller_historical_hidden_latency_ms",
    "fuller_historical_energy_j",
    "fuller_stalled_latency_ms",
    "fuller_stalled_hidden_latency_ms",
    "fuller_stalled_energy_j",
    "fuller_hedge_latency_ms",
    "fuller_hedge_hidden_latency_ms",
    "fuller_hedge_energy_j",
    "fuller_hedge_vs_stalled_latency_pct",
    "fuller_hedge_vs_historical_latency_pct",
    "fuller_hedge_vs_stalled_hidden_latency_pct",
    "fuller_hedge_vs_historical_hidden_latency_pct",
    "fuller_hedge_vs_historical_hidden_latency_suppressed",
    "fuller_hedge_vs_stalled_energy_pct",
    "fuller_hedge_vs_historical_energy_pct",
    "fuller_current_hedge_snapshot",
]


REVIEW_INTAKE_SURFACE_FIELDNAMES = [
    "launch_order",
    "stage",
    "priority",
    "config_id",
    "branch_purpose",
    "branch_delta_vs_hedge",
    "entry_surface_kind",
    "entry_surface_path",
    "entry_surface_section",
    "variant_id",
    "experiment_id",
    "run_id",
    "depends_on_run_id",
    "config_path",
    "launch_command",
    "review_artifact_path",
    "review_row_selector",
    "review_artifact_note",
    "gate_type",
    "gate_summary",
    "success_signal",
    "flow_admission_stalls_max",
    "flow_control_backpressure_max",
    "latency_target_lt_ms",
    "hidden_latency_target_lt_ms",
    "energy_target_lt_j",
    "baseline_context",
    "on_pass",
    "on_fail",
]

LAUNCH_PLAN_FIELDNAMES = [
    "launch_order",
    "stage",
    "priority",
    "config_id",
    "branch_purpose",
    "branch_delta_vs_hedge",
    "entry_surface_kind",
    "entry_surface_path",
    "entry_surface_section",
    "variant_id",
    "experiment_id",
    "run_id",
    "depends_on_run_id",
    "config_path",
    "launch_command",
    "review_artifact_path",
    "review_row_selector",
    "review_artifact_note",
    "hypothesis",
    "success_signal",
]

PRIORITY_EXECUTION_CROSSWALK_FIELDNAMES = [
    "priority",
    "config_id",
    "purpose",
    "branch_purpose",
    "decision_bucket",
    "entry_surface_kind",
    "entry_surface_path",
    "entry_surface_section",
    "source_row_span",
    "branch_delta_vs_hedge",
    "recommended_next_action",
    "flow_meso_current_hedge_snapshot",
    "flow_phy_current_hedge_snapshot",
    "fuller_current_hedge_snapshot",
    "proxy_launch_row",
    "proxy_run_id",
    "proxy_experiment_id",
    "proxy_config_path",
    "proxy_launch_command",
    "proxy_depends_on_run_id",
    "proxy_review_artifact_path",
    "proxy_review_row_selector",
    "proxy_review_artifact_note",
    "proxy_gate_summary",
    "proxy_success_signal",
    "proxy_on_pass",
    "proxy_on_fail",
    "flow_meso_launch_row",
    "flow_meso_run_id",
    "flow_meso_experiment_id",
    "flow_meso_config_path",
    "flow_meso_launch_command",
    "flow_meso_depends_on_run_id",
    "flow_meso_review_artifact_path",
    "flow_meso_review_row_selector",
    "flow_meso_review_artifact_note",
    "flow_meso_gate_summary",
    "flow_meso_success_signal",
    "flow_meso_on_pass",
    "flow_meso_on_fail",
    "flow_phy_launch_row",
    "flow_phy_run_id",
    "flow_phy_experiment_id",
    "flow_phy_config_path",
    "flow_phy_launch_command",
    "flow_phy_depends_on_run_id",
    "flow_phy_review_artifact_path",
    "flow_phy_review_row_selector",
    "flow_phy_review_artifact_note",
    "flow_phy_gate_summary",
    "flow_phy_success_signal",
    "flow_phy_on_pass",
    "flow_phy_on_fail",
    "fuller_launch_row",
    "fuller_run_id",
    "fuller_experiment_id",
    "fuller_config_path",
    "fuller_launch_command",
    "fuller_depends_on_run_id",
    "fuller_review_artifact_path",
    "fuller_review_row_selector",
    "fuller_review_artifact_note",
    "fuller_gate_summary",
    "fuller_success_signal",
    "fuller_on_pass",
    "fuller_on_fail",
]


PROXY_ACCEPTANCE_GATE = "Keep flow_admission_stalls=0 and flow_control_backpressure=0.0 in isolated E2."
PRESSURE_FREE_FLOW_ADMISSION_STALLS_MAX = 0.0
PRESSURE_FREE_FLOW_CONTROL_BACKPRESSURE_MAX = 0.0
REVIEW_ARTIFACT_NOTE = "Expected batch summary table after the run completes; inspect the row matching run_id."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate reintegration-aware HOPS v3 bridge configs.")
    parser.add_argument("--proxy-template", type=Path, default=DEFAULT_PROXY_TEMPLATE_PATH)
    parser.add_argument("--reintegration-template", type=Path, default=DEFAULT_REINTEGRATION_TEMPLATE_PATH)
    parser.add_argument("--proxy-out-dir", type=Path, default=DEFAULT_PROXY_OUT_DIR)
    parser.add_argument("--reintegration-out-dir", type=Path, default=DEFAULT_REINTEGRATION_OUT_DIR)
    parser.add_argument("--run-prefix", default=DEFAULT_RUN_PREFIX)
    parser.add_argument("--summary-path", type=Path)
    parser.add_argument("--package-path", type=Path)
    parser.add_argument("--comparison-path", type=Path, default=DEFAULT_COMPARISON_PATH)
    parser.add_argument("--bridge-scorecard-path", type=Path, default=DEFAULT_BRIDGE_SCORECARD_PATH)
    parser.add_argument("--bridge-plan-path", type=Path, default=DEFAULT_BRIDGE_PLAN_PATH)
    parser.add_argument("--manager-eval-surface-path", type=Path, default=DEFAULT_MANAGER_EVAL_SURFACE_PATH)
    parser.add_argument("--review-intake-surface-path", type=Path, default=DEFAULT_REVIEW_INTAKE_SURFACE_PATH)
    parser.add_argument("--priority-crosswalk-path", type=Path, default=DEFAULT_PRIORITY_CROSSWALK_PATH)
    parser.add_argument("--priority1-packet-path", type=Path, default=DEFAULT_PRIORITY1_PACKET_PATH)
    parser.add_argument("--packet-index-path", type=Path, default=DEFAULT_PACKET_INDEX_PATH)
    parser.add_argument(
        "--manager-slot-recommendation-path",
        type=Path,
        default=DEFAULT_MANAGER_SLOT_RECOMMENDATION_PATH,
    )
    parser.add_argument("--launch-plan-path", type=Path)
    return parser.parse_args()


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def build_review_artifact_path(stage: str, proxy_out_dir: Path, reintegration_out_dir: Path) -> str:
    if stage == "proxy":
        return str(proxy_out_dir / "results_summary.csv")
    return str(reintegration_out_dir / "results_summary.csv")


def build_review_row_selector(run_id: str) -> str:
    return f"run_id={run_id}"


def build_review_artifact_fields(run_id: str, config_path: str) -> dict[str, str]:
    return {
        "review_artifact_path": str(Path(config_path).with_name("results_summary.csv")),
        "review_row_selector": build_review_row_selector(run_id),
        "review_artifact_note": REVIEW_ARTIFACT_NOTE,
    }


def render_proxy_baseline_context(
    proxy_hedge_run_id: object,
    proxy_hedge_latency_ms: object,
    proxy_hedge_flow_admission_stalls: object,
    proxy_hedge_flow_control_backpressure: object,
) -> str:
    return (
        f"proxy hedge `{proxy_hedge_run_id}` with latency_ms={proxy_hedge_latency_ms}, "
        f"flow_admission_stalls={proxy_hedge_flow_admission_stalls}, "
        f"flow_control_backpressure={proxy_hedge_flow_control_backpressure}"
    )


def render_reintegration_baseline_context(
    historical_run_id: object,
    stalled_run_id: object,
    hedge_run_id: object,
) -> str:
    return f"historical `{historical_run_id}`; stalled `{stalled_run_id}`; current safe hedge `{hedge_run_id}`"


def build_expected_review_baseline_context(
    package_row: dict[str, object],
    *,
    stage: str,
    variant_id: str,
) -> str:
    if stage == "proxy":
        return render_proxy_baseline_context(
            package_row.get("proxy_hedge_run_id", ""),
            package_row.get("proxy_hedge_latency_ms", ""),
            package_row.get("proxy_hedge_flow_admission_stalls", ""),
            package_row.get("proxy_hedge_flow_control_backpressure", ""),
        )

    prefix_by_variant = {
        "FLOW_MESO": "flow_meso",
        "FLOW_PHY": "flow_phy",
        "FULLER": "fuller",
    }
    prefix = prefix_by_variant.get(variant_id)
    if prefix is None:
        raise ValueError(f"Unsupported variant_id for baseline_context rendering: {variant_id!r}")

    return render_reintegration_baseline_context(
        package_row.get(f"{prefix}_historical_run_id", package_row.get("historical_run_id", "")),
        package_row.get(f"{prefix}_stalled_run_id", package_row.get("stalled_run_id", "")),
        package_row.get(f"{prefix}_hedge_run_id", package_row.get("hedge_run_id", "")),
    )


def get_expected_review_threshold(
    package_row: dict[str, object],
    *,
    variant_id: str,
    field_name: str,
) -> object:
    if field_name in package_row:
        return package_row[field_name]

    prefix_by_variant = {
        "FLOW_MESO": "flow_meso",
        "FLOW_PHY": "flow_phy",
        "FULLER": "fuller",
    }
    prefix = prefix_by_variant.get(variant_id)
    if prefix is None:
        raise ValueError(f"Unsupported variant_id for threshold lookup: {variant_id!r}")
    return package_row.get(f"{prefix}_{field_name}", "")


def build_expected_review_contract_fields(
    package_row: dict[str, object],
    *,
    stage: str,
    variant_id: str,
    next_run_id: str,
) -> dict[str, str]:
    if stage == "proxy":
        if not next_run_id:
            raise ValueError("Proxy review contract requires next_run_id")
        return {
            "gate_type": "proxy_pressure_gate",
            "gate_summary": (
                "E2 must keep flow_admission_stalls<=0 and flow_control_backpressure<=0.0 "
                "before any reintegration replay is authorized."
            ),
            "baseline_context": build_expected_review_baseline_context(
                package_row,
                stage=stage,
                variant_id=variant_id,
            ),
            "on_pass": (
                f"Start this branch replay ladder with `{next_run_id}` and continue in "
                "FLOW_MESO -> FLOW_PHY -> FULLER order."
            ),
            "on_fail": "Stop this branch at proxy and do not spend reintegration replay on this config.",
        }

    if variant_id == "FULLER":
        return {
            "gate_type": "fuller_reintegration_gate",
            "gate_summary": (
                "Replay must keep flow_admission_stalls<=0 and flow_control_backpressure<=0.0 "
                f"while also beating latency_ms<{get_expected_review_threshold(package_row, variant_id=variant_id, field_name='latency_target_lt_ms')}, "
                f"integrated_hidden_system_latency_ms<{get_expected_review_threshold(package_row, variant_id=variant_id, field_name='hidden_latency_target_lt_ms')}, "
                f"energy_j<{get_expected_review_threshold(package_row, variant_id=variant_id, field_name='energy_target_lt_j')}."
            ),
            "baseline_context": build_expected_review_baseline_context(
                package_row,
                stage=stage,
                variant_id=variant_id,
            ),
            "on_pass": (
                "Escalate to manager review: this candidate is the only HOPS branch that clears the "
                "current hedge-defined FULLER gate."
            ),
            "on_fail": (
                "Mark this candidate NOT_READY and stop the branch; keep the default Package B recommendation "
                "unless the manager explicitly authorizes another HOPS hedge."
            ),
        }

    if not next_run_id:
        raise ValueError(f"Replay review contract for {variant_id!r} requires next_run_id")
    return {
        "gate_type": "reintegration_pressure_plus_latency_gate",
        "gate_summary": (
            "Replay must keep flow_admission_stalls<=0 and flow_control_backpressure<=0.0 "
            f"while beating latency_ms<{get_expected_review_threshold(package_row, variant_id=variant_id, field_name='latency_target_lt_ms')}."
        ),
        "baseline_context": build_expected_review_baseline_context(
            package_row,
            stage=stage,
            variant_id=variant_id,
        ),
        "on_pass": f"Continue this branch with `{next_run_id}`.",
        "on_fail": "Stop this branch here and do not continue to later replay lanes.",
    }


def set_path(obj: dict, path: list[str], value) -> None:
    cursor = obj
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = value


def set_section_enabled(cfg: dict, section_name: str, enabled: bool) -> None:
    section = cfg.get(section_name) or {}
    section["enabled"] = enabled
    cfg[section_name] = section


def build_proxy_config(template: dict, spec: dict, run_prefix: str) -> dict:
    cfg = deepcopy(template)
    run_id = f"{run_prefix}_{spec['config_id']}"

    set_path(cfg, ["run", "run_id"], run_id)
    set_path(cfg, ["run", "experiment_id"], "E2")
    set_path(
        cfg,
        ["run", "notes"],
        "HOPS v3 reintegration-aware bridge sweep; 8x8-first spill ladder with bounded 4x4 fallback for manager-scheduled replay",
    )
    set_path(cfg, ["switches", "meso"], False)
    set_path(cfg, ["switches", "flow"], True)
    set_path(cfg, ["switches", "det"], False)
    set_path(cfg, ["switches", "sparse"], False)
    set_path(cfg, ["switches", "phy"], False)

    set_path(cfg, ["accuracy", "require_context_match"], False)
    set_path(
        cfg,
        ["accuracy", "measurement_contract", "note"],
        "bridge_proxy_hops_v3_reintegration_aware_sweep; do_not_treat_as_measured_promotion_surface",
    )

    set_path(cfg, ["flow", "enabled"], True)
    set_path(cfg, ["flow", "evidence_type"], "heuristic_proxy")
    set_path(cfg, ["flow", "calibration_source"], "")
    set_path(cfg, ["flow", "latency_scale"], 0.85)
    set_path(cfg, ["flow", "scheduler_mode"], "elastic_residency_v3")
    set_path(cfg, ["flow", "reuse_policy"], "operand_factored")
    set_path(cfg, ["flow", "prefetch_window"], 2)
    set_path(cfg, ["flow", "control_group_size"], 4)
    set_path(cfg, ["flow", "tile_rows"], spec["tile_rows"])
    set_path(cfg, ["flow", "tile_cols"], spec["tile_cols"])
    set_path(cfg, ["flow", "prefetch_credits"], spec["prefetch_credits"])
    set_path(cfg, ["flow", "execute_credits"], spec["execute_credits"])
    set_path(cfg, ["flow", "control_issue_width"], spec["control_issue_width"])
    set_path(cfg, ["flow", "admission_policy"], "reuse_first")
    set_path(cfg, ["flow", "eviction_policy"], "pinned_operand")
    set_path(cfg, ["flow", "service_policy"], "critical_path_first")
    set_path(cfg, ["flow", "reuse_residency_budget"], 5)
    set_path(cfg, ["flow", "broadcast_stability_window"], 2)
    set_path(cfg, ["flow", "prefetch_distance"], spec["prefetch_distance"])
    set_path(cfg, ["flow", "exception_lane_policy"], spec["exception_lane_policy"])
    set_path(cfg, ["flow", "buffer_depth"], spec["buffer_depth"])
    return cfg


def build_reintegration_config(
    template: dict,
    proxy_cfg: dict,
    spec: dict,
    variant: dict,
    run_prefix: str,
) -> tuple[str, dict]:
    cfg = deepcopy(template)
    run_id = f"{run_prefix}_{spec['config_id']}_{variant['run_suffix']}"

    cfg["run"]["run_id"] = run_id
    cfg["run"]["experiment_id"] = variant["experiment_id"]
    cfg["run"]["notes"] = (
        f"HOPS v3 bridge replay from {spec['config_id']} into {variant['notes_suffix']}"
    )
    cfg["switches"] = deepcopy(variant["switches"])
    cfg["flow"] = deepcopy(proxy_cfg["flow"])
    cfg["flow"]["enabled"] = True
    cfg["accuracy"]["context_run_id"] = run_id
    cfg["accuracy"]["require_context_match"] = False

    set_section_enabled(cfg, "flow", variant["switches"]["flow"])
    set_section_enabled(cfg, "meso", variant["switches"]["meso"])
    set_section_enabled(cfg, "sparse", variant["switches"]["sparse"])
    set_section_enabled(cfg, "phy", variant["switches"]["phy"])

    sc_det = cfg.get("sc_det") or {}
    early_stop = sc_det.get("early_stop") or {}
    early_stop["enabled"] = variant["switches"]["det"]
    sc_det["early_stop"] = early_stop
    cfg["sc_det"] = sc_det

    return run_id, cfg


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def require_row(rows: list[dict[str, str]], key: str, value: str) -> dict[str, str]:
    for row in rows:
        if row.get(key) == value:
            return row
    raise KeyError(f"Missing row where {key}={value!r}")


def as_float(row: dict[str, str], key: str) -> float | None:
    value = row.get(key)
    if value in (None, ""):
        return None
    return float(value)


def pct_delta(candidate: float | None, baseline: float | None) -> float | None:
    if candidate is None or baseline in (None, 0.0):
        return None
    return ((candidate / baseline) - 1.0) * 100.0


def pp_delta(candidate: float | None, baseline: float | None) -> float | None:
    if candidate is None or baseline is None:
        return None
    return (candidate - baseline) * 100.0


def fmt_metric(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if abs(value - round(value)) < 1e-9:
        return f"{int(round(value)):,}"
    return f"{value:.{digits}f}"


def fmt_pct(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    return f"{value:+.{digits}f}%"


def format_date_tag_from_path(path: Path) -> str:
    raw_tag = path.stem.rsplit("_", 1)[-1]
    if len(raw_tag) == 8 and raw_tag.isdigit():
        return f"{raw_tag[:4]}-{raw_tag[4:6]}-{raw_tag[6:]}"
    return raw_tag


def is_suppressed_pct_delta(value: float | None, limit: float = SUPPRESSED_PCT_LIMIT) -> bool:
    return value is not None and abs(value) >= limit


def fmt_pct_or_suppressed(value: float | None, digits: int = 2, limit: float = SUPPRESSED_PCT_LIMIT) -> str:
    if value is None:
        return "n/a"
    if is_suppressed_pct_delta(value, limit):
        return "suppressed (historical baseline ~0)"
    return fmt_pct(value, digits=digits)


def baseline_from_pct_delta(
    candidate: float | None,
    delta_pct: float | None,
    *,
    suppressed_zero_limit: float | None = None,
) -> float | None:
    if candidate is None or delta_pct is None or delta_pct <= -100.0:
        return None
    if suppressed_zero_limit is not None and is_suppressed_pct_delta(delta_pct, suppressed_zero_limit):
        return 0.0
    return candidate / (1.0 + (delta_pct / 100.0))


def baseline_from_pp_delta(candidate: float | None, delta_pp: float | None) -> float | None:
    if candidate is None or delta_pp is None:
        return None
    return candidate - (delta_pp / 100.0)


def fmt_baseline_metric(value: float | None, digits: int = 6, near_zero: float = 1e-9) -> str:
    if value is None:
        return "n/a"
    if abs(value) < near_zero:
        return "~0"
    return fmt_metric(value, digits=digits)


def build_launch_command(config_path: str | Path) -> str:
    return f"{DEFAULT_CAFFEINATE_PREFIX} {DEFAULT_MPS_PYTHON} {DEFAULT_PHASE1_RUNNER} --config {config_path}"


def classify_bridge_family(spec_or_row: dict[str, object]) -> str:
    tile_rows = int(spec_or_row["tile_rows"])
    tile_cols = int(spec_or_row["tile_cols"])
    if tile_rows == 8 and tile_cols == 8:
        return "8x8_local"
    if tile_rows == 4 and tile_cols == 4:
        return "4x4_rescue"
    return "mixed_geometry"


def family_priority_rank(spec_or_row: dict[str, object]) -> int:
    family = classify_bridge_family(spec_or_row)
    priority = int(spec_or_row["priority"])
    if family == "8x8_local":
        return priority
    if family == "4x4_rescue":
        return priority - 4
    return priority


def build_lineage_hash(payload: dict[str, object]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:12]


def bridge_gate_for_variant(variant_id: str) -> str:
    if variant_id == "FULLER":
        return (
            "FULLER is still decisive; a bridge point must stay pressure-free and beat the current 8x8 hedge "
            "on latency, hidden latency, and energy together."
        )
    return "Remain pressure-free and beat the current 8x8 hedge latency without reopening replay backpressure."


def build_replay_target_thresholds(variant_id: str, comparison_row: dict[str, object]) -> dict[str, object]:
    thresholds: dict[str, object] = {
        "flow_admission_stalls_max": PRESSURE_FREE_FLOW_ADMISSION_STALLS_MAX,
        "flow_control_backpressure_max": PRESSURE_FREE_FLOW_CONTROL_BACKPRESSURE_MAX,
        "latency_target_lt_ms": comparison_row.get("hedge_latency_ms", ""),
        "hidden_latency_target_lt_ms": "",
        "energy_target_lt_j": "",
    }
    if variant_id == "FULLER":
        thresholds["hidden_latency_target_lt_ms"] = comparison_row.get("hedge_hidden_latency_ms", "")
        thresholds["energy_target_lt_j"] = comparison_row.get("hedge_energy_j", "")
    return thresholds


def render_current_hedge_snapshot(comparison_row: dict[str, object]) -> str:
    if not comparison_row:
        return ""

    hidden_vs_historical = comparison_row.get("hedge_vs_historical_hidden_latency_pct")
    if comparison_row.get("hedge_vs_historical_hidden_latency_suppressed"):
        hidden_vs_historical_text = "suppressed vs historical"
    else:
        hidden_vs_historical_text = f"{fmt_pct(hidden_vs_historical)} vs historical"

    return (
        f"lat {fmt_metric(comparison_row.get('hedge_latency_ms'), 6)} ms "
        f"({fmt_pct(comparison_row.get('hedge_vs_stalled_latency_pct'))} vs stalled, "
        f"{fmt_pct(comparison_row.get('hedge_vs_historical_latency_pct'))} vs historical); "
        f"hidden {fmt_metric(comparison_row.get('hedge_hidden_latency_ms'), 6)} ms "
        f"({fmt_pct(comparison_row.get('hedge_vs_stalled_hidden_latency_pct'))} vs stalled, "
        f"{hidden_vs_historical_text}); "
        f"energy {fmt_metric(comparison_row.get('hedge_energy_j'), 9)} J "
        f"({fmt_pct(comparison_row.get('hedge_vs_stalled_energy_pct'))} vs stalled, "
        f"{fmt_pct(comparison_row.get('hedge_vs_historical_energy_pct'))} vs historical); "
        f"stalls {fmt_metric(comparison_row.get('hedge_flow_admission_stalls'))}; "
        f"backpressure {fmt_metric(comparison_row.get('hedge_flow_control_backpressure'), 6)}"
    )


def build_expected_replay_hedge_snapshot_fields(
    comparison_rows: list[dict[str, object]],
) -> dict[str, str]:
    comparison_by_variant = {str(row.get("variant_id", "")): row for row in comparison_rows}
    snapshot_fields: dict[str, str] = {}
    for variant in REINTEGRATION_VARIANTS:
        prefix = str(variant["file_suffix"])
        variant_id = str(variant["variant_id"])
        snapshot_fields[f"{prefix}_current_hedge_snapshot"] = render_current_hedge_snapshot(
            comparison_by_variant.get(variant_id, {})
        )
    return snapshot_fields


def render_proxy_hedge_snapshot(batch2_row: dict[str, str]) -> str:
    return (
        f"lat {fmt_metric(as_float(batch2_row, 'latency_ms'), 6)} ms; "
        f"hidden {fmt_metric(as_float(batch2_row, 'integrated_hidden_system_latency_ms'), 6)} ms; "
        f"bubble {fmt_metric(as_float(batch2_row, 'bubble_cycles'))}; "
        f"util {fmt_metric(as_float(batch2_row, 'utilization_avg'), 6)}; "
        f"stalls {fmt_metric(as_float(batch2_row, 'flow_admission_stalls'))}; "
        f"backpressure {fmt_metric(as_float(batch2_row, 'flow_control_backpressure'), 6)}"
    )


def render_proxy_spec_delta_vs_hedge(spec_row: dict[str, object]) -> str:
    deltas: list[str] = []

    hedge_tile = f"{PROXY_HEDGE_SPEC['tile_rows']}x{PROXY_HEDGE_SPEC['tile_cols']}"
    row_tile = f"{spec_row['tile_rows']}x{spec_row['tile_cols']}"
    if row_tile != hedge_tile:
        deltas.append(f"tile {hedge_tile}->{row_tile}")

    hedge_credits = f"{PROXY_HEDGE_SPEC['prefetch_credits']}/{PROXY_HEDGE_SPEC['execute_credits']}"
    row_credits = f"{spec_row['prefetch_credits']}/{spec_row['execute_credits']}"
    if row_credits != hedge_credits:
        deltas.append(f"credits {hedge_credits}->{row_credits}")

    if spec_row["control_issue_width"] != PROXY_HEDGE_SPEC["control_issue_width"]:
        deltas.append(
            f"issue_width {PROXY_HEDGE_SPEC['control_issue_width']}->{spec_row['control_issue_width']}"
        )

    if spec_row["prefetch_distance"] != PROXY_HEDGE_SPEC["prefetch_distance"]:
        deltas.append(
            f"prefetch_distance {PROXY_HEDGE_SPEC['prefetch_distance']}->{spec_row['prefetch_distance']}"
        )

    if spec_row["buffer_depth"] != PROXY_HEDGE_SPEC["buffer_depth"]:
        deltas.append(f"buffer_depth {PROXY_HEDGE_SPEC['buffer_depth']}->{spec_row['buffer_depth']}")

    if spec_row["exception_lane_policy"] != PROXY_HEDGE_SPEC["exception_lane_policy"]:
        deltas.append(
            "lane_policy "
            f"{PROXY_HEDGE_SPEC['exception_lane_policy']}->{spec_row['exception_lane_policy']}"
        )

    if not deltas:
        return "Matches the current safe 8x8 spill hedge."
    return "; ".join(deltas) + "; other hedge knobs unchanged"


def build_proxy_hedge_context(batch2_rows: list[dict[str, str]]) -> dict[str, object]:
    empty_context = {
        "proxy_hedge_run_id": "",
        "proxy_hedge_latency_ms": "",
        "proxy_hedge_flow_admission_stalls": "",
        "proxy_hedge_flow_control_backpressure": "",
        "proxy_replay_promotion_note": "",
        "proxy_hedge_snapshot": "",
    }
    if not batch2_rows:
        return empty_context

    hedge_row = require_row(batch2_rows, "config_id", PROXY_HEDGE_CONFIG_ID)
    return {
        "proxy_hedge_run_id": hedge_row.get("run_id", ""),
        "proxy_hedge_latency_ms": as_float(hedge_row, "latency_ms"),
        "proxy_hedge_flow_admission_stalls": as_float(hedge_row, "flow_admission_stalls"),
        "proxy_hedge_flow_control_backpressure": as_float(hedge_row, "flow_control_backpressure"),
        "proxy_replay_promotion_note": (
            "Pressure-free E2 is mandatory. Use the current zero-pressure 8x8 hedge as the replay-spend "
            "reference before scheduling FLOW_MESO/FLOW_PHY/FULLER."
        ),
        "proxy_hedge_snapshot": render_proxy_hedge_snapshot(hedge_row),
    }


def build_measured_reintegration_comparison_rows(
    primary_replay_rows: list[dict[str, str]],
    tile8_replay_rows: list[dict[str, str]],
) -> list[dict[str, object]]:
    comparison_rows: list[dict[str, object]] = []
    for variant in sorted(REINTEGRATION_VARIANTS, key=lambda item: item["variant_order"]):
        primary_row = require_row(primary_replay_rows, "variant_id", variant["variant_id"])
        tile8_row = require_row(tile8_replay_rows, "variant_id", variant["variant_id"])

        historical_bubble_cycles = baseline_from_pct_delta(
            as_float(tile8_row, "bubble_cycles"),
            as_float(tile8_row, "bubble_delta_vs_historical_pct"),
        )
        historical_utilization = baseline_from_pp_delta(
            as_float(tile8_row, "utilization_avg"),
            as_float(tile8_row, "util_delta_vs_historical_pp"),
        )
        historical_latency = baseline_from_pct_delta(
            as_float(tile8_row, "latency_ms"),
            as_float(tile8_row, "latency_delta_vs_historical_pct"),
        )
        stalled_hidden_vs_historical = as_float(primary_row, "hidden_latency_delta_vs_historical_pct")
        hedge_hidden_vs_historical = as_float(tile8_row, "hidden_latency_delta_vs_historical_pct")
        stalled_hidden_vs_historical_suppressed = is_suppressed_pct_delta(stalled_hidden_vs_historical)
        hedge_hidden_vs_historical_suppressed = is_suppressed_pct_delta(hedge_hidden_vs_historical)
        historical_hidden_latency = baseline_from_pct_delta(
            as_float(tile8_row, "integrated_hidden_system_latency_ms"),
            hedge_hidden_vs_historical,
            suppressed_zero_limit=SUPPRESSED_PCT_LIMIT,
        )
        historical_energy = baseline_from_pct_delta(
            as_float(tile8_row, "energy_j"),
            as_float(tile8_row, "energy_delta_vs_historical_pct"),
        )

        comparison_rows.append(
            {
                "variant_order": variant["variant_order"],
                "variant_id": variant["variant_id"],
                "experiment_id": variant["experiment_id"],
                "bridge_gate": bridge_gate_for_variant(variant["variant_id"]),
                "historical_run_id": tile8_row["historical_run_id"],
                "stalled_run_id": primary_row["run_id"],
                "hedge_run_id": tile8_row["run_id"],
                "historical_bubble_cycles": historical_bubble_cycles,
                "stalled_bubble_cycles": as_float(primary_row, "bubble_cycles"),
                "hedge_bubble_cycles": as_float(tile8_row, "bubble_cycles"),
                "historical_utilization_avg": historical_utilization,
                "stalled_utilization_avg": as_float(primary_row, "utilization_avg"),
                "hedge_utilization_avg": as_float(tile8_row, "utilization_avg"),
                "historical_latency_ms": historical_latency,
                "stalled_latency_ms": as_float(primary_row, "latency_ms"),
                "hedge_latency_ms": as_float(tile8_row, "latency_ms"),
                "historical_hidden_latency_ms": historical_hidden_latency,
                "stalled_hidden_latency_ms": as_float(primary_row, "integrated_hidden_system_latency_ms"),
                "hedge_hidden_latency_ms": as_float(tile8_row, "integrated_hidden_system_latency_ms"),
                "historical_energy_j": historical_energy,
                "stalled_energy_j": as_float(primary_row, "energy_j"),
                "hedge_energy_j": as_float(tile8_row, "energy_j"),
                "historical_flow_admission_stalls": None,
                "stalled_flow_admission_stalls": as_float(primary_row, "flow_admission_stalls"),
                "hedge_flow_admission_stalls": as_float(tile8_row, "flow_admission_stalls"),
                "historical_flow_control_backpressure": None,
                "stalled_flow_control_backpressure": as_float(primary_row, "flow_control_backpressure"),
                "hedge_flow_control_backpressure": as_float(tile8_row, "flow_control_backpressure"),
                "historical_flow_residency_hit_rate": None,
                "stalled_flow_residency_hit_rate": as_float(primary_row, "flow_residency_hit_rate"),
                "hedge_flow_residency_hit_rate": as_float(tile8_row, "flow_residency_hit_rate"),
                "historical_flow_prefetch_hits": None,
                "stalled_flow_prefetch_hits": as_float(primary_row, "flow_prefetch_hits"),
                "hedge_flow_prefetch_hits": as_float(tile8_row, "flow_prefetch_hits"),
                "historical_flow_eviction_count": None,
                "stalled_flow_eviction_count": as_float(primary_row, "flow_eviction_count"),
                "hedge_flow_eviction_count": as_float(tile8_row, "flow_eviction_count"),
                "stalled_vs_historical_bubble_cycles_pct": as_float(primary_row, "bubble_delta_vs_historical_pct"),
                "hedge_vs_stalled_bubble_cycles_pct": pct_delta(
                    as_float(tile8_row, "bubble_cycles"),
                    as_float(primary_row, "bubble_cycles"),
                ),
                "hedge_vs_historical_bubble_cycles_pct": as_float(tile8_row, "bubble_delta_vs_historical_pct"),
                "stalled_vs_historical_utilization_pp": as_float(primary_row, "util_delta_vs_historical_pp"),
                "hedge_vs_stalled_utilization_pp": pp_delta(
                    as_float(tile8_row, "utilization_avg"),
                    as_float(primary_row, "utilization_avg"),
                ),
                "hedge_vs_historical_utilization_pp": as_float(tile8_row, "util_delta_vs_historical_pp"),
                "stalled_vs_historical_latency_pct": as_float(primary_row, "latency_delta_vs_historical_pct"),
                "hedge_vs_stalled_latency_pct": pct_delta(
                    as_float(tile8_row, "latency_ms"),
                    as_float(primary_row, "latency_ms"),
                ),
                "hedge_vs_historical_latency_pct": as_float(tile8_row, "latency_delta_vs_historical_pct"),
                "stalled_vs_historical_hidden_latency_pct": (
                    None if stalled_hidden_vs_historical_suppressed else stalled_hidden_vs_historical
                ),
                "hedge_vs_stalled_hidden_latency_pct": pct_delta(
                    as_float(tile8_row, "integrated_hidden_system_latency_ms"),
                    as_float(primary_row, "integrated_hidden_system_latency_ms"),
                ),
                "hedge_vs_historical_hidden_latency_pct": (
                    None if hedge_hidden_vs_historical_suppressed else hedge_hidden_vs_historical
                ),
                "stalled_vs_historical_hidden_latency_suppressed": stalled_hidden_vs_historical_suppressed,
                "hedge_vs_historical_hidden_latency_suppressed": hedge_hidden_vs_historical_suppressed,
                "stalled_vs_historical_energy_pct": as_float(primary_row, "energy_delta_vs_historical_pct"),
                "hedge_vs_stalled_energy_pct": pct_delta(
                    as_float(tile8_row, "energy_j"),
                    as_float(primary_row, "energy_j"),
                ),
                "hedge_vs_historical_energy_pct": as_float(tile8_row, "energy_delta_vs_historical_pct"),
            }
        )
    return comparison_rows


def build_reintegration_package_rows(
    proxy_manifest_rows: list[dict[str, object]],
    reintegration_manifest_rows: list[dict[str, object]],
    comparison_rows: list[dict[str, object]],
    proxy_hedge_context: dict[str, object],
    *,
    priority1_packet_path: Path,
    packet_index_path: Path,
) -> list[dict[str, object]]:
    replays_by_config: dict[str, dict[str, dict[str, object]]] = {}
    for replay_row in reintegration_manifest_rows:
        replays_by_config.setdefault(str(replay_row["config_id"]), {})[str(replay_row["variant_id"])] = replay_row

    comparison_by_variant = {str(row["variant_id"]): row for row in comparison_rows}
    replay_hedge_snapshot_fields = build_expected_replay_hedge_snapshot_fields(comparison_rows)
    package_rows: list[dict[str, object]] = []
    for proxy_row in sorted(proxy_manifest_rows, key=lambda item: int(item["priority"])):
        config_id = str(proxy_row["config_id"])
        priority = int(proxy_row["priority"])
        replays = replays_by_config.get(config_id, {})
        branch_routing_fields = build_branch_routing_fields(
            priority,
            config_id,
            branch_purpose=str(proxy_row["purpose"]),
            branch_delta_vs_hedge=render_proxy_spec_delta_vs_hedge(proxy_row),
            priority1_packet_path=priority1_packet_path,
            packet_index_path=packet_index_path,
        )
        package_row: dict[str, object] = {
            "config_id": config_id,
            "priority": proxy_row["priority"],
            "purpose": proxy_row["purpose"],
            "entry_surface_kind": branch_routing_fields["entry_surface_kind"],
            "entry_surface_path": branch_routing_fields["entry_surface_path"],
            "entry_surface_section": branch_routing_fields["entry_surface_section"],
            "anchor": proxy_row["anchor"],
            "proxy_hypothesis": proxy_row["proxy_hypothesis"],
            "replay_goal": proxy_row["replay_goal"],
            "decision_bucket": proxy_row["decision_bucket"],
            "recommended_next_action": proxy_row["recommended_next_action"],
            "proxy_run_id": proxy_row["run_id"],
            "proxy_config_path": proxy_row["file_path"],
            "proxy_launch_command": build_launch_command(str(proxy_row["file_path"])),
            **{
                f"proxy_{key}": value
                for key, value in build_review_artifact_fields(
                    str(proxy_row["run_id"]),
                    str(proxy_row["file_path"]),
                ).items()
            },
            "proxy_acceptance_gate": PROXY_ACCEPTANCE_GATE,
            "proxy_flow_admission_stalls_max": PRESSURE_FREE_FLOW_ADMISSION_STALLS_MAX,
            "proxy_flow_control_backpressure_max": PRESSURE_FREE_FLOW_CONTROL_BACKPRESSURE_MAX,
            "proxy_hedge_run_id": proxy_hedge_context.get("proxy_hedge_run_id", ""),
            "proxy_hedge_latency_ms": proxy_hedge_context.get("proxy_hedge_latency_ms", ""),
            "proxy_hedge_flow_admission_stalls": proxy_hedge_context.get("proxy_hedge_flow_admission_stalls", ""),
            "proxy_hedge_flow_control_backpressure": proxy_hedge_context.get(
                "proxy_hedge_flow_control_backpressure",
                "",
            ),
            "proxy_replay_promotion_note": proxy_hedge_context.get("proxy_replay_promotion_note", ""),
            "proxy_hedge_snapshot": proxy_hedge_context.get("proxy_hedge_snapshot", ""),
            "proxy_spec_delta_vs_hedge": branch_routing_fields["branch_delta_vs_hedge"],
            "buffer_depth": proxy_row["buffer_depth"],
            "prefetch_credits": proxy_row["prefetch_credits"],
            "execute_credits": proxy_row["execute_credits"],
            "tile_rows": proxy_row["tile_rows"],
            "tile_cols": proxy_row["tile_cols"],
            "control_issue_width": proxy_row["control_issue_width"],
            "prefetch_distance": proxy_row["prefetch_distance"],
            "exception_lane_policy": proxy_row["exception_lane_policy"],
        }

        for variant in sorted(REINTEGRATION_VARIANTS, key=lambda item: item["variant_order"]):
            prefix = str(variant["file_suffix"])
            replay_row = replays.get(str(variant["variant_id"]), {})
            comparison_row = comparison_by_variant.get(str(variant["variant_id"]), {})
            package_row[f"{prefix}_run_id"] = replay_row.get("run_id", "")
            package_row[f"{prefix}_config_path"] = replay_row.get("file_path", "")
            package_row[f"{prefix}_launch_command"] = (
                build_launch_command(str(replay_row["file_path"])) if replay_row.get("file_path") else ""
            )
            review_fields = (
                build_review_artifact_fields(str(replay_row["run_id"]), str(replay_row["file_path"]))
                if replay_row.get("file_path")
                else {}
            )
            for key, value in review_fields.items():
                package_row[f"{prefix}_{key}"] = value
            package_row[f"{prefix}_bridge_gate"] = comparison_row.get("bridge_gate", "")
            package_row[f"{prefix}_historical_run_id"] = comparison_row.get("historical_run_id", "")
            package_row[f"{prefix}_stalled_run_id"] = comparison_row.get("stalled_run_id", "")
            package_row[f"{prefix}_hedge_run_id"] = comparison_row.get("hedge_run_id", "")
            for threshold_key, threshold_value in build_replay_target_thresholds(
                str(variant["variant_id"]),
                comparison_row,
            ).items():
                package_row[f"{prefix}_{threshold_key}"] = threshold_value
        package_row.update(replay_hedge_snapshot_fields)

        package_rows.append(package_row)
    return package_rows


def build_bridge_scorecard_rows(
    package_rows: list[dict[str, object]],
    comparison_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    comparison_by_variant = {str(row["variant_id"]): row for row in comparison_rows}

    scorecard_rows: list[dict[str, object]] = []
    for package_row in sorted(package_rows, key=lambda item: int(item["priority"])):
        scorecard_row: dict[str, object] = {
            "priority": package_row["priority"],
            "config_id": package_row["config_id"],
            "purpose": package_row["purpose"],
            "anchor": package_row["anchor"],
            "proxy_hypothesis": package_row["proxy_hypothesis"],
            "replay_goal": package_row["replay_goal"],
            "decision_bucket": package_row["decision_bucket"],
            "recommended_next_action": package_row["recommended_next_action"],
            "proxy_acceptance_gate": PROXY_ACCEPTANCE_GATE,
            "proxy_flow_admission_stalls_max": PRESSURE_FREE_FLOW_ADMISSION_STALLS_MAX,
            "proxy_flow_control_backpressure_max": PRESSURE_FREE_FLOW_CONTROL_BACKPRESSURE_MAX,
            "proxy_hedge_run_id": package_row.get("proxy_hedge_run_id", ""),
            "proxy_hedge_latency_ms": package_row.get("proxy_hedge_latency_ms", ""),
            "proxy_hedge_flow_admission_stalls": package_row.get("proxy_hedge_flow_admission_stalls", ""),
            "proxy_hedge_flow_control_backpressure": package_row.get("proxy_hedge_flow_control_backpressure", ""),
            "proxy_replay_promotion_note": package_row.get("proxy_replay_promotion_note", ""),
            "proxy_hedge_snapshot": package_row.get("proxy_hedge_snapshot", ""),
            "proxy_spec_delta_vs_hedge": package_row.get("proxy_spec_delta_vs_hedge", ""),
            "proxy_run_id": package_row["proxy_run_id"],
            "proxy_config_path": package_row["proxy_config_path"],
            "buffer_depth": package_row["buffer_depth"],
            "prefetch_credits": package_row["prefetch_credits"],
            "execute_credits": package_row["execute_credits"],
            "tile_rows": package_row["tile_rows"],
            "tile_cols": package_row["tile_cols"],
            "control_issue_width": package_row["control_issue_width"],
            "prefetch_distance": package_row["prefetch_distance"],
            "exception_lane_policy": package_row["exception_lane_policy"],
        }
        for variant in sorted(REINTEGRATION_VARIANTS, key=lambda item: item["variant_order"]):
            prefix = str(variant["file_suffix"])
            comparison_row = comparison_by_variant.get(str(variant["variant_id"]), {})
            scorecard_row[f"{prefix}_goal"] = variant["variant_goal"]
            scorecard_row[f"{prefix}_run_id"] = package_row.get(f"{prefix}_run_id", "")
            scorecard_row[f"{prefix}_config_path"] = package_row.get(f"{prefix}_config_path", "")
            scorecard_row[f"{prefix}_bridge_gate"] = comparison_row.get("bridge_gate", "")
            scorecard_row[f"{prefix}_historical_run_id"] = comparison_row.get("historical_run_id", "")
            scorecard_row[f"{prefix}_stalled_run_id"] = comparison_row.get("stalled_run_id", "")
            scorecard_row[f"{prefix}_hedge_run_id"] = comparison_row.get("hedge_run_id", "")
            for threshold_key, threshold_value in build_replay_target_thresholds(
                str(variant["variant_id"]),
                comparison_row,
            ).items():
                scorecard_row[f"{prefix}_{threshold_key}"] = threshold_value
            scorecard_row[f"{prefix}_historical_latency_ms"] = comparison_row.get("historical_latency_ms", "")
            scorecard_row[f"{prefix}_stalled_latency_ms"] = comparison_row.get("stalled_latency_ms", "")
            scorecard_row[f"{prefix}_hedge_latency_ms"] = comparison_row.get("hedge_latency_ms", "")
            scorecard_row[f"{prefix}_hedge_vs_stalled_latency_pct"] = comparison_row.get(
                "hedge_vs_stalled_latency_pct", ""
            )
            scorecard_row[f"{prefix}_hedge_vs_historical_latency_pct"] = comparison_row.get(
                "hedge_vs_historical_latency_pct", ""
            )
            scorecard_row[f"{prefix}_historical_hidden_latency_ms"] = comparison_row.get(
                "historical_hidden_latency_ms", ""
            )
            scorecard_row[f"{prefix}_stalled_hidden_latency_ms"] = comparison_row.get(
                "stalled_hidden_latency_ms", ""
            )
            scorecard_row[f"{prefix}_hedge_hidden_latency_ms"] = comparison_row.get("hedge_hidden_latency_ms", "")
            scorecard_row[f"{prefix}_hedge_vs_stalled_hidden_latency_pct"] = comparison_row.get(
                "hedge_vs_stalled_hidden_latency_pct", ""
            )
            scorecard_row[f"{prefix}_hedge_vs_historical_hidden_latency_pct"] = comparison_row.get(
                "hedge_vs_historical_hidden_latency_pct", ""
            )
            scorecard_row[f"{prefix}_hedge_vs_historical_hidden_latency_suppressed"] = comparison_row.get(
                "hedge_vs_historical_hidden_latency_suppressed", ""
            )
            scorecard_row[f"{prefix}_historical_energy_j"] = comparison_row.get("historical_energy_j", "")
            scorecard_row[f"{prefix}_stalled_energy_j"] = comparison_row.get("stalled_energy_j", "")
            scorecard_row[f"{prefix}_hedge_energy_j"] = comparison_row.get("hedge_energy_j", "")
            scorecard_row[f"{prefix}_hedge_vs_stalled_energy_pct"] = comparison_row.get(
                "hedge_vs_stalled_energy_pct", ""
            )
            scorecard_row[f"{prefix}_hedge_vs_historical_energy_pct"] = comparison_row.get(
                "hedge_vs_historical_energy_pct", ""
            )
            scorecard_row[f"{prefix}_stalled_flow_admission_stalls"] = comparison_row.get(
                "stalled_flow_admission_stalls", ""
            )
            scorecard_row[f"{prefix}_hedge_flow_admission_stalls"] = comparison_row.get(
                "hedge_flow_admission_stalls", ""
            )
            scorecard_row[f"{prefix}_stalled_flow_control_backpressure"] = comparison_row.get(
                "stalled_flow_control_backpressure", ""
            )
            scorecard_row[f"{prefix}_hedge_flow_control_backpressure"] = comparison_row.get(
                "hedge_flow_control_backpressure", ""
            )
        scorecard_rows.append(scorecard_row)

    return scorecard_rows


def build_bridge_plan_rows(
    launch_plan_rows: list[dict[str, object]],
    package_rows: list[dict[str, object]],
    comparison_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    package_by_config = {str(row["config_id"]): row for row in package_rows}
    comparison_by_variant = {str(row["variant_id"]): row for row in comparison_rows}
    variant_order_by_id = {str(row["variant_id"]): row["variant_order"] for row in REINTEGRATION_VARIANTS}

    bridge_plan_rows: list[dict[str, object]] = []
    for launch_row in launch_plan_rows:
        config_id = str(launch_row["config_id"])
        variant_id = str(launch_row["variant_id"])
        package_row = package_by_config[config_id]
        comparison_row = comparison_by_variant.get(variant_id, {})
        if launch_row["stage"] == "proxy":
            target_thresholds = {
                "flow_admission_stalls_max": PRESSURE_FREE_FLOW_ADMISSION_STALLS_MAX,
                "flow_control_backpressure_max": PRESSURE_FREE_FLOW_CONTROL_BACKPRESSURE_MAX,
                "latency_target_lt_ms": "",
                "hidden_latency_target_lt_ms": "",
                "energy_target_lt_j": "",
            }
        else:
            target_thresholds = build_replay_target_thresholds(variant_id, comparison_row)

        bridge_plan_rows.append(
            {
                "launch_order": launch_row["launch_order"],
                "stage": launch_row["stage"],
                "priority": launch_row["priority"],
                "config_id": config_id,
                "variant_order": variant_order_by_id.get(variant_id, ""),
                "variant_id": variant_id,
                "experiment_id": launch_row["experiment_id"],
                "run_id": launch_row["run_id"],
                "depends_on_run_id": launch_row["depends_on_run_id"],
                "config_path": launch_row["config_path"],
                "launch_command": launch_row["launch_command"],
                "purpose": package_row["purpose"],
                "anchor": package_row["anchor"],
                "decision_bucket": package_row["decision_bucket"],
                "hypothesis": launch_row["hypothesis"],
                "success_signal": launch_row["success_signal"],
                "recommended_next_action": package_row["recommended_next_action"],
                "buffer_depth": package_row["buffer_depth"],
                "prefetch_credits": package_row["prefetch_credits"],
                "execute_credits": package_row["execute_credits"],
                "tile_rows": package_row["tile_rows"],
                "tile_cols": package_row["tile_cols"],
                "control_issue_width": package_row["control_issue_width"],
                "prefetch_distance": package_row["prefetch_distance"],
                "exception_lane_policy": package_row["exception_lane_policy"],
                "proxy_acceptance_gate": PROXY_ACCEPTANCE_GATE,
                "proxy_hedge_run_id": package_row.get("proxy_hedge_run_id", ""),
                "proxy_hedge_latency_ms": package_row.get("proxy_hedge_latency_ms", ""),
                "proxy_hedge_flow_admission_stalls": package_row.get("proxy_hedge_flow_admission_stalls", ""),
                "proxy_hedge_flow_control_backpressure": package_row.get(
                    "proxy_hedge_flow_control_backpressure",
                    "",
                ),
                "proxy_replay_promotion_note": package_row.get("proxy_replay_promotion_note", ""),
                "proxy_hedge_snapshot": package_row.get("proxy_hedge_snapshot", ""),
                "proxy_spec_delta_vs_hedge": package_row.get("proxy_spec_delta_vs_hedge", ""),
                "bridge_gate": comparison_row.get("bridge_gate", ""),
                "flow_admission_stalls_max": target_thresholds["flow_admission_stalls_max"],
                "flow_control_backpressure_max": target_thresholds["flow_control_backpressure_max"],
                "latency_target_lt_ms": target_thresholds["latency_target_lt_ms"],
                "hidden_latency_target_lt_ms": target_thresholds["hidden_latency_target_lt_ms"],
                "energy_target_lt_j": target_thresholds["energy_target_lt_j"],
                "historical_run_id": comparison_row.get("historical_run_id", ""),
                "stalled_run_id": comparison_row.get("stalled_run_id", ""),
                "hedge_run_id": comparison_row.get("hedge_run_id", ""),
                "historical_latency_ms": comparison_row.get("historical_latency_ms", ""),
                "stalled_latency_ms": comparison_row.get("stalled_latency_ms", ""),
                "hedge_latency_ms": comparison_row.get("hedge_latency_ms", ""),
                "historical_hidden_latency_ms": comparison_row.get("historical_hidden_latency_ms", ""),
                "stalled_hidden_latency_ms": comparison_row.get("stalled_hidden_latency_ms", ""),
                "hedge_hidden_latency_ms": comparison_row.get("hedge_hidden_latency_ms", ""),
                "historical_energy_j": comparison_row.get("historical_energy_j", ""),
                "stalled_energy_j": comparison_row.get("stalled_energy_j", ""),
                "hedge_energy_j": comparison_row.get("hedge_energy_j", ""),
                "stalled_flow_admission_stalls": comparison_row.get("stalled_flow_admission_stalls", ""),
                "hedge_flow_admission_stalls": comparison_row.get("hedge_flow_admission_stalls", ""),
                "stalled_flow_control_backpressure": comparison_row.get("stalled_flow_control_backpressure", ""),
                "hedge_flow_control_backpressure": comparison_row.get("hedge_flow_control_backpressure", ""),
                "stalled_vs_historical_latency_pct": comparison_row.get("stalled_vs_historical_latency_pct", ""),
                "hedge_vs_stalled_latency_pct": comparison_row.get("hedge_vs_stalled_latency_pct", ""),
                "hedge_vs_historical_latency_pct": comparison_row.get("hedge_vs_historical_latency_pct", ""),
                "stalled_vs_historical_hidden_latency_pct": comparison_row.get(
                    "stalled_vs_historical_hidden_latency_pct", ""
                ),
                "hedge_vs_stalled_hidden_latency_pct": comparison_row.get("hedge_vs_stalled_hidden_latency_pct", ""),
                "hedge_vs_historical_hidden_latency_pct": comparison_row.get(
                    "hedge_vs_historical_hidden_latency_pct", ""
                ),
                "stalled_vs_historical_hidden_latency_suppressed": comparison_row.get(
                    "stalled_vs_historical_hidden_latency_suppressed", ""
                ),
                "hedge_vs_historical_hidden_latency_suppressed": comparison_row.get(
                    "hedge_vs_historical_hidden_latency_suppressed", ""
                ),
                "stalled_vs_historical_energy_pct": comparison_row.get("stalled_vs_historical_energy_pct", ""),
                "hedge_vs_stalled_energy_pct": comparison_row.get("hedge_vs_stalled_energy_pct", ""),
                "hedge_vs_historical_energy_pct": comparison_row.get("hedge_vs_historical_energy_pct", ""),
            }
        )

    return bridge_plan_rows


def build_manager_eval_surface_rows(
    package_rows: list[dict[str, object]],
    comparison_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    fuller_comparison = next((row for row in comparison_rows if row.get("variant_id") == "FULLER"), {})

    manager_rows: list[dict[str, object]] = []
    for package_row in sorted(package_rows, key=lambda item: int(item["priority"])):
        manager_rows.append(
            {
                "config_id": package_row["config_id"],
                "priority": package_row["priority"],
                "purpose": package_row["purpose"],
                "entry_surface_kind": package_row.get("entry_surface_kind", ""),
                "entry_surface_path": package_row.get("entry_surface_path", ""),
                "entry_surface_section": package_row.get("entry_surface_section", ""),
                "anchor": package_row["anchor"],
                "decision_bucket": package_row["decision_bucket"],
                "recommended_next_action": package_row["recommended_next_action"],
                "proxy_run_id": package_row["proxy_run_id"],
                "proxy_config_path": package_row["proxy_config_path"],
                "proxy_launch_command": package_row.get("proxy_launch_command", ""),
                "proxy_review_artifact_path": package_row.get("proxy_review_artifact_path", ""),
                "proxy_review_row_selector": package_row.get("proxy_review_row_selector", ""),
                "proxy_review_artifact_note": package_row.get("proxy_review_artifact_note", ""),
                "proxy_acceptance_gate": package_row["proxy_acceptance_gate"],
                "proxy_flow_admission_stalls_max": package_row["proxy_flow_admission_stalls_max"],
                "proxy_flow_control_backpressure_max": package_row["proxy_flow_control_backpressure_max"],
                "proxy_hedge_run_id": package_row.get("proxy_hedge_run_id", ""),
                "proxy_hedge_latency_ms": package_row.get("proxy_hedge_latency_ms", ""),
                "proxy_hedge_flow_admission_stalls": package_row.get("proxy_hedge_flow_admission_stalls", ""),
                "proxy_hedge_flow_control_backpressure": package_row.get(
                    "proxy_hedge_flow_control_backpressure",
                    "",
                ),
                "proxy_replay_promotion_note": package_row.get("proxy_replay_promotion_note", ""),
                "proxy_hedge_snapshot": package_row.get("proxy_hedge_snapshot", ""),
                "proxy_spec_delta_vs_hedge": package_row.get("proxy_spec_delta_vs_hedge", ""),
                "fuller_run_id": package_row.get("fuller_run_id", ""),
                "fuller_config_path": package_row.get("fuller_config_path", ""),
                "fuller_launch_command": package_row.get("fuller_launch_command", ""),
                "fuller_review_artifact_path": package_row.get("fuller_review_artifact_path", ""),
                "fuller_review_row_selector": package_row.get("fuller_review_row_selector", ""),
                "fuller_review_artifact_note": package_row.get("fuller_review_artifact_note", ""),
                "fuller_bridge_gate": package_row.get("fuller_bridge_gate", ""),
                "fuller_historical_run_id": package_row.get("fuller_historical_run_id", ""),
                "fuller_stalled_run_id": package_row.get("fuller_stalled_run_id", ""),
                "fuller_hedge_run_id": package_row.get("fuller_hedge_run_id", ""),
                "fuller_flow_admission_stalls_max": package_row.get("fuller_flow_admission_stalls_max", ""),
                "fuller_flow_control_backpressure_max": package_row.get(
                    "fuller_flow_control_backpressure_max",
                    "",
                ),
                "fuller_latency_target_lt_ms": package_row.get("fuller_latency_target_lt_ms", ""),
                "fuller_hidden_latency_target_lt_ms": package_row.get("fuller_hidden_latency_target_lt_ms", ""),
                "fuller_energy_target_lt_j": package_row.get("fuller_energy_target_lt_j", ""),
                "fuller_historical_latency_ms": fuller_comparison.get("historical_latency_ms", ""),
                "fuller_historical_hidden_latency_ms": fuller_comparison.get("historical_hidden_latency_ms", ""),
                "fuller_historical_energy_j": fuller_comparison.get("historical_energy_j", ""),
                "fuller_stalled_latency_ms": fuller_comparison.get("stalled_latency_ms", ""),
                "fuller_stalled_hidden_latency_ms": fuller_comparison.get("stalled_hidden_latency_ms", ""),
                "fuller_stalled_energy_j": fuller_comparison.get("stalled_energy_j", ""),
                "fuller_hedge_latency_ms": fuller_comparison.get("hedge_latency_ms", ""),
                "fuller_hedge_hidden_latency_ms": fuller_comparison.get("hedge_hidden_latency_ms", ""),
                "fuller_hedge_energy_j": fuller_comparison.get("hedge_energy_j", ""),
                "fuller_hedge_vs_stalled_latency_pct": fuller_comparison.get("hedge_vs_stalled_latency_pct", ""),
                "fuller_hedge_vs_historical_latency_pct": fuller_comparison.get(
                    "hedge_vs_historical_latency_pct",
                    "",
                ),
                "fuller_hedge_vs_stalled_hidden_latency_pct": fuller_comparison.get(
                    "hedge_vs_stalled_hidden_latency_pct",
                    "",
                ),
                "fuller_hedge_vs_historical_hidden_latency_pct": fuller_comparison.get(
                    "hedge_vs_historical_hidden_latency_pct",
                    "",
                ),
                "fuller_hedge_vs_historical_hidden_latency_suppressed": fuller_comparison.get(
                    "hedge_vs_historical_hidden_latency_suppressed",
                    "",
                ),
                "fuller_hedge_vs_stalled_energy_pct": fuller_comparison.get("hedge_vs_stalled_energy_pct", ""),
                "fuller_hedge_vs_historical_energy_pct": fuller_comparison.get(
                    "hedge_vs_historical_energy_pct",
                    "",
                ),
                "fuller_current_hedge_snapshot": package_row.get("fuller_current_hedge_snapshot", ""),
            }
        )
    return manager_rows


def build_review_intake_surface_rows(
    launch_plan_rows: list[dict[str, object]],
    bridge_plan_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    bridge_by_run_id = {str(row["run_id"]): row for row in bridge_plan_rows}
    ordered_launch_rows = sorted(launch_plan_rows, key=lambda item: int(item["launch_order"]))
    next_row_by_run_id: dict[str, dict[str, object]] = {}
    for index, launch_row in enumerate(ordered_launch_rows[:-1]):
        next_row_by_run_id[str(launch_row["run_id"])] = ordered_launch_rows[index + 1]

    intake_rows: list[dict[str, object]] = []
    for launch_row in ordered_launch_rows:
        bridge_row = bridge_by_run_id[str(launch_row["run_id"])]
        stage = str(launch_row["stage"])
        variant_id = str(launch_row["variant_id"])
        next_row = next_row_by_run_id.get(str(launch_row["run_id"]))
        review_contract = build_expected_review_contract_fields(
            bridge_row,
            stage=stage,
            variant_id=variant_id,
            next_run_id="" if next_row is None else str(next_row["run_id"]),
        )

        intake_rows.append(
            {
                "launch_order": launch_row["launch_order"],
                "stage": launch_row["stage"],
                "priority": launch_row["priority"],
                "config_id": launch_row["config_id"],
                "branch_purpose": launch_row["branch_purpose"],
                "branch_delta_vs_hedge": launch_row["branch_delta_vs_hedge"],
                "entry_surface_kind": launch_row["entry_surface_kind"],
                "entry_surface_path": launch_row["entry_surface_path"],
                "entry_surface_section": launch_row["entry_surface_section"],
                "variant_id": launch_row["variant_id"],
                "experiment_id": launch_row["experiment_id"],
                "run_id": launch_row["run_id"],
                "depends_on_run_id": launch_row["depends_on_run_id"],
                "config_path": launch_row["config_path"],
                "launch_command": launch_row["launch_command"],
                "review_artifact_path": launch_row["review_artifact_path"],
                "review_row_selector": launch_row["review_row_selector"],
                "review_artifact_note": launch_row["review_artifact_note"],
                "gate_type": review_contract["gate_type"],
                "gate_summary": review_contract["gate_summary"],
                "success_signal": launch_row["success_signal"],
                "flow_admission_stalls_max": bridge_row["flow_admission_stalls_max"],
                "flow_control_backpressure_max": bridge_row["flow_control_backpressure_max"],
                "latency_target_lt_ms": bridge_row["latency_target_lt_ms"],
                "hidden_latency_target_lt_ms": bridge_row["hidden_latency_target_lt_ms"],
                "energy_target_lt_j": bridge_row["energy_target_lt_j"],
                "baseline_context": review_contract["baseline_context"],
                "on_pass": review_contract["on_pass"],
                "on_fail": review_contract["on_fail"],
            }
        )

    return intake_rows


def priority_packet_index_section_heading(priority: int, config_id: str) -> str:
    return f"### Priority {priority}: {config_id}"


def priority_entry_surface_kind(priority: int) -> str:
    return "standalone_priority_packet" if priority == 1 else "shared_priority_index"


def priority_entry_surface_path(
    priority: int,
    *,
    priority1_packet_path: Path,
    packet_index_path: Path,
) -> Path:
    return priority1_packet_path if priority == 1 else packet_index_path


def priority_entry_surface_section(priority: int, config_id: str) -> str:
    return "standalone packet root" if priority == 1 else priority_packet_index_section_heading(priority, config_id)


def build_branch_routing_fields(
    priority: int,
    config_id: str,
    *,
    branch_purpose: str,
    branch_delta_vs_hedge: str,
    priority1_packet_path: Path,
    packet_index_path: Path,
) -> dict[str, object]:
    return {
        "branch_purpose": branch_purpose,
        "branch_delta_vs_hedge": branch_delta_vs_hedge,
        "entry_surface_kind": priority_entry_surface_kind(priority),
        "entry_surface_path": str(
            priority_entry_surface_path(
                priority,
                priority1_packet_path=priority1_packet_path,
                packet_index_path=packet_index_path,
            )
        ),
        "entry_surface_section": priority_entry_surface_section(priority, config_id),
    }


def build_priority_execution_crosswalk_rows(
    package_rows: list[dict[str, object]],
    review_intake_surface_rows: list[dict[str, object]],
    *,
    priority1_packet_path: Path,
    packet_index_path: Path,
) -> list[dict[str, object]]:
    def build_stage_fields(prefix: str, step: dict[str, object]) -> dict[str, object]:
        return {
            f"{prefix}_launch_row": step["launch_order"],
            f"{prefix}_run_id": step["run_id"],
            f"{prefix}_experiment_id": step["experiment_id"],
            f"{prefix}_config_path": step["config_path"],
            f"{prefix}_launch_command": step["launch_command"],
            f"{prefix}_depends_on_run_id": step["depends_on_run_id"],
            f"{prefix}_review_artifact_path": step["review_artifact_path"],
            f"{prefix}_review_row_selector": step["review_row_selector"],
            f"{prefix}_review_artifact_note": step["review_artifact_note"],
            f"{prefix}_gate_summary": step["gate_summary"],
            f"{prefix}_success_signal": step["success_signal"],
            f"{prefix}_on_pass": step["on_pass"],
            f"{prefix}_on_fail": step["on_fail"],
        }

    crosswalk_rows: list[dict[str, object]] = []
    for branch_row in sorted(package_rows, key=lambda item: int(item["priority"])):
        priority = int(branch_row["priority"])
        branch_steps = collect_priority_branch_steps(priority, review_intake_surface_rows)
        proxy_step = next(step for step in branch_steps if str(step["stage"]) == "proxy")
        flow_meso_step = next(step for step in branch_steps if str(step["variant_id"]) == "FLOW_MESO")
        flow_phy_step = next(step for step in branch_steps if str(step["variant_id"]) == "FLOW_PHY")
        fuller_step = next(step for step in branch_steps if str(step["variant_id"]) == "FULLER")
        branch_routing_fields = build_branch_routing_fields(
            priority,
            str(branch_row["config_id"]),
            branch_purpose=str(branch_row["purpose"]),
            branch_delta_vs_hedge=str(branch_row["proxy_spec_delta_vs_hedge"]),
            priority1_packet_path=priority1_packet_path,
            packet_index_path=packet_index_path,
        )

        row = {
            "priority": priority,
            "config_id": branch_row["config_id"],
            "purpose": branch_row["purpose"],
            "decision_bucket": branch_row["decision_bucket"],
            "source_row_span": f"{branch_steps[0]['launch_order']}-{branch_steps[-1]['launch_order']}",
            "recommended_next_action": branch_row["recommended_next_action"],
            "flow_meso_current_hedge_snapshot": branch_row.get("flow_meso_current_hedge_snapshot", ""),
            "flow_phy_current_hedge_snapshot": branch_row.get("flow_phy_current_hedge_snapshot", ""),
            "fuller_current_hedge_snapshot": branch_row.get("fuller_current_hedge_snapshot", ""),
        }
        row.update(branch_routing_fields)
        row.update(build_stage_fields("proxy", proxy_step))
        row.update(build_stage_fields("flow_meso", flow_meso_step))
        row.update(build_stage_fields("flow_phy", flow_phy_step))
        row.update(build_stage_fields("fuller", fuller_step))
        crosswalk_rows.append(row)
    return crosswalk_rows


def validate_generated_surfaces(
    package_rows: list[dict[str, object]],
    comparison_rows: list[dict[str, object]],
    launch_plan_rows: list[dict[str, object]],
    review_intake_surface_rows: list[dict[str, object]],
    priority_crosswalk_rows: list[dict[str, object]],
    manager_eval_surface_rows: list[dict[str, object]],
) -> None:
    expected_priorities = [str(spec["priority"]) for spec in sorted(BRIDGE_SPECS, key=lambda item: item["priority"])]
    expected_stage_contract = [
        ("proxy", "", "E2"),
        ("reintegration", "FLOW_MESO", "E1"),
        ("reintegration", "FLOW_PHY", "E5"),
        ("reintegration", "FULLER", "FULLER_REENTRY_V1"),
    ]
    launch_review_shared_fields = [
        "launch_order",
        "stage",
        "priority",
        "config_id",
        "branch_purpose",
        "branch_delta_vs_hedge",
        "entry_surface_kind",
        "entry_surface_path",
        "entry_surface_section",
        "variant_id",
        "experiment_id",
        "run_id",
        "depends_on_run_id",
        "config_path",
        "launch_command",
        "review_artifact_path",
        "review_row_selector",
        "review_artifact_note",
        "success_signal",
    ]
    manager_projection_fields = [
        "config_id",
        "priority",
        "purpose",
        "entry_surface_kind",
        "entry_surface_path",
        "entry_surface_section",
        "anchor",
        "decision_bucket",
        "recommended_next_action",
        "proxy_run_id",
        "proxy_config_path",
        "proxy_launch_command",
        "proxy_review_artifact_path",
        "proxy_review_row_selector",
        "proxy_review_artifact_note",
        "proxy_acceptance_gate",
        "proxy_flow_admission_stalls_max",
        "proxy_flow_control_backpressure_max",
        "proxy_hedge_run_id",
        "proxy_hedge_latency_ms",
        "proxy_hedge_flow_admission_stalls",
        "proxy_hedge_flow_control_backpressure",
        "proxy_replay_promotion_note",
        "proxy_hedge_snapshot",
        "proxy_spec_delta_vs_hedge",
        "fuller_run_id",
        "fuller_config_path",
        "fuller_launch_command",
        "fuller_review_artifact_path",
        "fuller_review_row_selector",
        "fuller_review_artifact_note",
        "fuller_bridge_gate",
        "fuller_historical_run_id",
        "fuller_stalled_run_id",
        "fuller_hedge_run_id",
        "fuller_flow_admission_stalls_max",
        "fuller_flow_control_backpressure_max",
        "fuller_latency_target_lt_ms",
        "fuller_hidden_latency_target_lt_ms",
        "fuller_energy_target_lt_j",
        "fuller_current_hedge_snapshot",
    ]
    manager_fuller_comparison_fields = [
        ("fuller_historical_latency_ms", "historical_latency_ms"),
        ("fuller_historical_hidden_latency_ms", "historical_hidden_latency_ms"),
        ("fuller_historical_energy_j", "historical_energy_j"),
        ("fuller_stalled_latency_ms", "stalled_latency_ms"),
        ("fuller_stalled_hidden_latency_ms", "stalled_hidden_latency_ms"),
        ("fuller_stalled_energy_j", "stalled_energy_j"),
        ("fuller_hedge_latency_ms", "hedge_latency_ms"),
        ("fuller_hedge_hidden_latency_ms", "hedge_hidden_latency_ms"),
        ("fuller_hedge_energy_j", "hedge_energy_j"),
        ("fuller_hedge_vs_stalled_latency_pct", "hedge_vs_stalled_latency_pct"),
        ("fuller_hedge_vs_historical_latency_pct", "hedge_vs_historical_latency_pct"),
        ("fuller_hedge_vs_stalled_hidden_latency_pct", "hedge_vs_stalled_hidden_latency_pct"),
        ("fuller_hedge_vs_historical_hidden_latency_pct", "hedge_vs_historical_hidden_latency_pct"),
        ("fuller_hedge_vs_historical_hidden_latency_suppressed", "hedge_vs_historical_hidden_latency_suppressed"),
        ("fuller_hedge_vs_stalled_energy_pct", "hedge_vs_stalled_energy_pct"),
        ("fuller_hedge_vs_historical_energy_pct", "hedge_vs_historical_energy_pct"),
    ]

    def fail(message: str) -> None:
        raise ValueError(f"Generated bridge surface validation failed: {message}")

    def normalize_priority(value: object) -> str:
        return str(value)

    sorted_launch_rows = sorted(launch_plan_rows, key=lambda item: int(item["launch_order"]))
    sorted_review_rows = sorted(review_intake_surface_rows, key=lambda item: int(item["launch_order"]))
    sorted_package_rows = sorted(package_rows, key=lambda item: int(item["priority"]))
    sorted_crosswalk_rows = sorted(priority_crosswalk_rows, key=lambda item: int(item["priority"]))
    sorted_manager_rows = sorted(manager_eval_surface_rows, key=lambda item: int(item["priority"]))

    expected_launch_orders = [str(index) for index in range(1, len(sorted_launch_rows) + 1)]
    if [str(row["launch_order"]) for row in sorted_launch_rows] != expected_launch_orders:
        fail("launch_plan_rows launch_order is not contiguous from 1..N")
    if [str(row["launch_order"]) for row in sorted_review_rows] != expected_launch_orders:
        fail("review_intake_surface_rows launch_order is not contiguous from 1..N")
    if len(sorted_launch_rows) != len(sorted_review_rows):
        fail("launch_plan_rows and review_intake_surface_rows length mismatch")

    for launch_row, review_row in zip(sorted_launch_rows, sorted_review_rows):
        for field in launch_review_shared_fields:
            if launch_row.get(field, "") != review_row.get(field, ""):
                fail(
                    f"launch/review mismatch at launch_order={launch_row['launch_order']} for field {field}: "
                    f"{launch_row.get(field, '')!r} != {review_row.get(field, '')!r}"
                )

    package_priorities = [normalize_priority(row["priority"]) for row in sorted_package_rows]
    crosswalk_priorities = [normalize_priority(row["priority"]) for row in sorted_crosswalk_rows]
    manager_priorities = [normalize_priority(row["priority"]) for row in sorted_manager_rows]
    if package_priorities != expected_priorities:
        fail(f"package_rows priorities {package_priorities!r} do not match expected {expected_priorities!r}")
    if crosswalk_priorities != expected_priorities:
        fail(f"priority_crosswalk_rows priorities {crosswalk_priorities!r} do not match expected {expected_priorities!r}")
    if manager_priorities != expected_priorities:
        fail(f"manager_eval_surface_rows priorities {manager_priorities!r} do not match expected {expected_priorities!r}")

    package_by_priority = {normalize_priority(row["priority"]): row for row in sorted_package_rows}
    crosswalk_by_priority = {normalize_priority(row["priority"]): row for row in sorted_crosswalk_rows}
    manager_by_priority = {normalize_priority(row["priority"]): row for row in sorted_manager_rows}
    fuller_comparison = next(
        (row for row in comparison_rows if str(row.get("variant_id", "")) == "FULLER"),
        None,
    )
    if fuller_comparison is None:
        fail("comparison_rows is missing the FULLER variant")
    expected_replay_hedge_snapshot_fields = build_expected_replay_hedge_snapshot_fields(comparison_rows)

    for expected_index, priority in enumerate(expected_priorities):
        launch_group = [row for row in sorted_launch_rows if normalize_priority(row["priority"]) == priority]
        review_group = [row for row in sorted_review_rows if normalize_priority(row["priority"]) == priority]
        package_row = package_by_priority[priority]
        if len(launch_group) != len(expected_stage_contract):
            fail(f"priority {priority} has {len(launch_group)} launch rows instead of 4")
        if len(review_group) != len(expected_stage_contract):
            fail(f"priority {priority} has {len(review_group)} review rows instead of 4")

        expected_row_numbers = [str(expected_index * len(expected_stage_contract) + offset) for offset in range(1, 5)]
        launch_row_numbers = [str(row["launch_order"]) for row in launch_group]
        if launch_row_numbers != expected_row_numbers:
            fail(
                f"priority {priority} launch rows {launch_row_numbers!r} do not match expected block {expected_row_numbers!r}"
            )

        previous_run_id = ""
        for step_index, ((expected_stage, expected_variant, expected_experiment), launch_row, review_row) in enumerate(
            zip(expected_stage_contract, launch_group, review_group),
            start=1,
        ):
            if str(launch_row["stage"]) != expected_stage:
                fail(f"priority {priority} step {step_index} stage is {launch_row['stage']!r}, expected {expected_stage!r}")
            if str(launch_row["variant_id"]) != expected_variant:
                fail(
                    f"priority {priority} step {step_index} variant_id is {launch_row['variant_id']!r}, expected {expected_variant!r}"
                )
            if str(launch_row["experiment_id"]) != expected_experiment:
                fail(
                    f"priority {priority} step {step_index} experiment_id is {launch_row['experiment_id']!r}, expected {expected_experiment!r}"
                )
            expected_dependency = "" if step_index == 1 else previous_run_id
            if str(launch_row["depends_on_run_id"]) != expected_dependency:
                fail(
                    f"priority {priority} step {step_index} depends_on_run_id is {launch_row['depends_on_run_id']!r}, expected {expected_dependency!r}"
                )
            previous_run_id = str(launch_row["run_id"])

            if expected_stage == "proxy":
                if review_row.get("gate_type") != "proxy_pressure_gate":
                    fail(f"priority {priority} proxy gate_type is {review_row.get('gate_type')!r}")
                if review_row.get("flow_admission_stalls_max") != PRESSURE_FREE_FLOW_ADMISSION_STALLS_MAX:
                    fail(f"priority {priority} proxy flow_admission_stalls_max drifted")
                if review_row.get("flow_control_backpressure_max") != PRESSURE_FREE_FLOW_CONTROL_BACKPRESSURE_MAX:
                    fail(f"priority {priority} proxy flow_control_backpressure_max drifted")
                if review_row.get("hidden_latency_target_lt_ms") not in ("", None):
                    fail(f"priority {priority} proxy hidden_latency_target_lt_ms should be empty")
                if review_row.get("energy_target_lt_j") not in ("", None):
                    fail(f"priority {priority} proxy energy_target_lt_j should be empty")
            elif expected_variant in {"FLOW_MESO", "FLOW_PHY"}:
                if review_row.get("gate_type") != "reintegration_pressure_plus_latency_gate":
                    fail(f"priority {priority} {expected_variant} gate_type is {review_row.get('gate_type')!r}")
                if review_row.get("latency_target_lt_ms") in ("", None):
                    fail(f"priority {priority} {expected_variant} latency_target_lt_ms is empty")
                if review_row.get("hidden_latency_target_lt_ms") not in ("", None):
                    fail(f"priority {priority} {expected_variant} hidden_latency_target_lt_ms should be empty")
                if review_row.get("energy_target_lt_j") not in ("", None):
                    fail(f"priority {priority} {expected_variant} energy_target_lt_j should be empty")
            else:
                if review_row.get("gate_type") != "fuller_reintegration_gate":
                    fail(f"priority {priority} FULLER gate_type is {review_row.get('gate_type')!r}")
                if review_row.get("latency_target_lt_ms") in ("", None):
                    fail(f"priority {priority} FULLER latency_target_lt_ms is empty")
                if review_row.get("hidden_latency_target_lt_ms") in ("", None):
                    fail(f"priority {priority} FULLER hidden_latency_target_lt_ms is empty")
                if review_row.get("energy_target_lt_j") in ("", None):
                    fail(f"priority {priority} FULLER energy_target_lt_j is empty")

            expected_review_contract = build_expected_review_contract_fields(
                package_row,
                stage=expected_stage,
                variant_id=expected_variant,
                next_run_id="" if step_index == len(expected_stage_contract) else str(launch_group[step_index]["run_id"]),
            )
            for review_field in ("gate_summary", "baseline_context", "on_pass", "on_fail"):
                expected_value = expected_review_contract[review_field]
                if review_row.get(review_field, "") != expected_value:
                    fail(
                        f"priority {priority} step {step_index} {review_field} drifted: "
                        f"{review_row.get(review_field, '')!r} != {expected_value!r}"
                    )
        for snapshot_field, expected_snapshot in expected_replay_hedge_snapshot_fields.items():
            if package_row.get(snapshot_field, "") != expected_snapshot:
                fail(
                    f"priority {priority} package snapshot drift for {snapshot_field}: "
                    f"{package_row.get(snapshot_field, '')!r} != {expected_snapshot!r}"
                )

        crosswalk_row = crosswalk_by_priority[priority]
        manager_row = manager_by_priority[priority]
        if str(package_row["config_id"]) != str(crosswalk_row["config_id"]) or str(package_row["config_id"]) != str(
            manager_row["config_id"]
        ):
            fail(f"priority {priority} config_id drift across package/crosswalk/manager surfaces")
        if crosswalk_row.get("source_row_span") != f"{expected_row_numbers[0]}-{expected_row_numbers[-1]}":
            fail(f"priority {priority} crosswalk source_row_span is {crosswalk_row.get('source_row_span')!r}")

        shared_branch_fields = [
            ("decision_bucket", "decision_bucket"),
            ("entry_surface_kind", "entry_surface_kind"),
            ("entry_surface_path", "entry_surface_path"),
            ("entry_surface_section", "entry_surface_section"),
            ("recommended_next_action", "recommended_next_action"),
            ("proxy_spec_delta_vs_hedge", "branch_delta_vs_hedge"),
            ("flow_meso_current_hedge_snapshot", "flow_meso_current_hedge_snapshot"),
            ("flow_phy_current_hedge_snapshot", "flow_phy_current_hedge_snapshot"),
            ("fuller_current_hedge_snapshot", "fuller_current_hedge_snapshot"),
        ]
        crosswalk_purpose = crosswalk_row.get("purpose", "")
        crosswalk_branch_purpose = crosswalk_row.get("branch_purpose", "")
        package_purpose = package_row.get("purpose", "")
        if package_purpose != crosswalk_purpose or package_purpose != crosswalk_branch_purpose:
            fail(
                f"priority {priority} branch purpose drift across package/crosswalk surfaces: "
                f"package={package_purpose!r}, purpose={crosswalk_purpose!r}, "
                f"branch_purpose={crosswalk_branch_purpose!r}"
            )
        for package_field, crosswalk_field in shared_branch_fields:
            package_value = package_row.get(package_field, "")
            crosswalk_value = crosswalk_row.get(crosswalk_field, "")
            if package_value != crosswalk_value:
                fail(
                    f"priority {priority} package/crosswalk mismatch for fields "
                    f"{package_field}/{crosswalk_field}: {package_value!r} != {crosswalk_value!r}"
                )

        stage_field_map = {
            "proxy": (launch_group[0], review_group[0]),
            "flow_meso": (launch_group[1], review_group[1]),
            "flow_phy": (launch_group[2], review_group[2]),
            "fuller": (launch_group[3], review_group[3]),
        }
        for prefix, (launch_row, review_row) in stage_field_map.items():
            for source_field, crosswalk_suffix in (
                ("launch_order", "launch_row"),
                ("run_id", "run_id"),
                ("experiment_id", "experiment_id"),
                ("config_path", "config_path"),
                ("launch_command", "launch_command"),
                ("depends_on_run_id", "depends_on_run_id"),
                ("review_artifact_path", "review_artifact_path"),
                ("review_row_selector", "review_row_selector"),
                ("review_artifact_note", "review_artifact_note"),
                ("success_signal", "success_signal"),
            ):
                crosswalk_field = f"{prefix}_{crosswalk_suffix}"
                if launch_row.get(source_field, "") != crosswalk_row.get(crosswalk_field, ""):
                    fail(
                        f"priority {priority} crosswalk mismatch for field {crosswalk_field}: "
                        f"{launch_row.get(source_field, '')!r} != {crosswalk_row.get(crosswalk_field, '')!r}"
                    )
            for source_field, crosswalk_suffix in (
                ("gate_summary", "gate_summary"),
                ("on_pass", "on_pass"),
                ("on_fail", "on_fail"),
            ):
                crosswalk_field = f"{prefix}_{crosswalk_suffix}"
                if review_row.get(source_field, "") != crosswalk_row.get(crosswalk_field, ""):
                    fail(
                        f"priority {priority} crosswalk mismatch for field {crosswalk_field}: "
                        f"{review_row.get(source_field, '')!r} != {crosswalk_row.get(crosswalk_field, '')!r}"
                    )

        for field in manager_projection_fields:
            if package_row.get(field, "") != manager_row.get(field, ""):
                fail(
                    f"priority {priority} package/manager mismatch for field {field}: "
                    f"{package_row.get(field, '')!r} != {manager_row.get(field, '')!r}"
                )
        for manager_field, comparison_field in manager_fuller_comparison_fields:
            if manager_row.get(manager_field, "") != fuller_comparison.get(comparison_field, ""):
                fail(
                    f"priority {priority} manager/comparison mismatch for fields "
                    f"{manager_field}/{comparison_field}: "
                    f"{manager_row.get(manager_field, '')!r} != {fuller_comparison.get(comparison_field, '')!r}"
                )


def render_summary(
    proxy_manifest_path: Path,
    reintegration_manifest_path: Path,
    package_path: Path,
    comparison_path: Path,
    bridge_scorecard_path: Path,
    bridge_plan_path: Path,
    manager_slot_recommendation_path: Path,
    manager_eval_surface_path: Path,
    review_intake_surface_path: Path,
    priority_crosswalk_path: Path,
    priority1_packet_path: Path,
    packet_index_path: Path,
    launch_plan_path: Path,
    proxy_out_dir: Path,
    reintegration_out_dir: Path,
    run_prefix: str,
) -> str:
    lines = [
        "# HOPS v3 Bridge Batch Summary",
        "",
        f"- run prefix: `{run_prefix}`",
        "- objective: probe an 8x8-first reintegration bridge around the safe spill hedge, with 4x4 rescue points held as later fallback work",
        f"- proxy config root: `{proxy_out_dir}`",
        f"- reintegration config root: `{reintegration_out_dir}`",
        f"- proxy manifest: `{proxy_manifest_path}`",
        f"- reintegration manifest: `{reintegration_manifest_path}`",
        f"- reintegration package: `{package_path}`",
        f"- reintegration comparison: `{comparison_path}`",
        f"- bridge scorecard: `{bridge_scorecard_path}`",
        f"- bridge plan: `{bridge_plan_path}`",
        f"- manager slot recommendation: `{manager_slot_recommendation_path}`",
        f"- manager eval surface: `{manager_eval_surface_path}`",
        f"- review intake surface: `{review_intake_surface_path}`",
        f"- priority execution crosswalk: `{priority_crosswalk_path}`",
        f"- priority-1 execution packet: `{priority1_packet_path}`",
        f"- priority packet index: `{packet_index_path}`",
        f"- launch plan: `{launch_plan_path}`",
        "",
        "## Upstream References",
        "",
    ]
    for label, path in REFERENCE_PATHS.items():
        lines.append(f"- `{label}`: `{path}`")

    lines.extend(
        [
            "",
        ]
    )

    measured_paths = {
        "batch2_results": REFERENCE_PATHS["batch2_results"],
        "primary_reintegration_results": REFERENCE_PATHS["primary_reintegration_results"],
        "tile8_reintegration_results": REFERENCE_PATHS["tile8_reintegration_results"],
    }
    missing_measured_paths = [label for label, path in measured_paths.items() if not path.exists()]
    if missing_measured_paths:
        lines.extend(
            [
                "## Measured Baselines",
                "",
                "- measured anchor sections were skipped because these upstream result tables were not available at generation time:",
            ]
        )
        for label in missing_measured_paths:
            lines.append(f"- `{label}`")
        lines.append("")
    else:
        batch2_rows = read_csv_rows(measured_paths["batch2_results"])
        primary_replay_rows = read_csv_rows(measured_paths["primary_reintegration_results"])
        tile8_replay_rows = read_csv_rows(measured_paths["tile8_reintegration_results"])

        strongest_4x4 = require_row(batch2_rows, "config_id", "hopsv3_b5p4e5_4x4_spill")
        tile8_hedge = require_row(batch2_rows, "config_id", "hopsv3_b4p3e4_8x8_spill")
        pressure_boundary_4x4 = require_row(batch2_rows, "config_id", "hopsv3_b4p3e4_4x4_spill")

        lines.extend(
            [
                "## Measured Proxy Anchors",
                "",
                (
                    "- strongest isolated `4x4` anchor: "
                    f"`hopsv3_b5p4e5_4x4_spill` with latency `{fmt_metric(as_float(strongest_4x4, 'latency_ms'), 6)} ms`, "
                    f"bubble `{fmt_metric(as_float(strongest_4x4, 'bubble_cycles'))}`, stalls `{fmt_metric(as_float(strongest_4x4, 'flow_admission_stalls'))}`, "
                    f"backpressure `{fmt_metric(as_float(strongest_4x4, 'flow_control_backpressure'), 6)}`, "
                    f"bubble delta vs `hopsv2_buf4` `{fmt_pct(as_float(strongest_4x4, 'bubble_delta_vs_hopsv2_buf4_pct'))}`, "
                    f"latency delta vs `hopsv2_buf4` `{fmt_pct(as_float(strongest_4x4, 'latency_delta_vs_hopsv2_buf4_pct'))}`"
                ),
                (
                    "- safe reintegration hedge anchor: "
                    f"`hopsv3_b4p3e4_8x8_spill` with latency `{fmt_metric(as_float(tile8_hedge, 'latency_ms'), 6)} ms`, "
                    f"bubble `{fmt_metric(as_float(tile8_hedge, 'bubble_cycles'))}`, stalls `{fmt_metric(as_float(tile8_hedge, 'flow_admission_stalls'))}`, "
                    f"backpressure `{fmt_metric(as_float(tile8_hedge, 'flow_control_backpressure'), 6)}`"
                ),
                (
                    "- known unsafe `4x4` boundary point: "
                    f"`hopsv3_b4p3e4_4x4_spill` already reopens pressure in isolated `E2` with stalls "
                    f"`{fmt_metric(as_float(pressure_boundary_4x4, 'flow_admission_stalls'))}` and backpressure "
                    f"`{fmt_metric(as_float(pressure_boundary_4x4, 'flow_control_backpressure'), 6)}`"
                ),
                (
                    "- spill vs defer note: the stronger `4x4` `spill` and `defer` rows tie exactly in the batch-2 summary, so bridge work stays on `spill` because the reintegration hedge evidence favors explicit pressure relief."
                ),
                (
                    "- bridge implication: stay inside the zero-pressure `8x8` spill neighborhood first, decompose the pacing levers there, and treat the older `4x4` spill point as a later pressure-boundary fallback rather than an early rescue."
                ),
                "",
                "## Measured Reintegration Baselines",
                "",
                "- historical raw values below are reconstructed from the published `*_vs_historical_*` deltas; historical pressure counters are not available in the current CSVs and stay `n/a`.",
                "",
            ]
        )

        for variant in sorted(REINTEGRATION_VARIANTS, key=lambda item: item["variant_order"]):
            primary_row = require_row(primary_replay_rows, "variant_id", variant["variant_id"])
            tile8_row = require_row(tile8_replay_rows, "variant_id", variant["variant_id"])
            historical_latency = baseline_from_pct_delta(
                as_float(tile8_row, "latency_ms"),
                as_float(tile8_row, "latency_delta_vs_historical_pct"),
            )
            historical_hidden_latency = baseline_from_pct_delta(
                as_float(tile8_row, "integrated_hidden_system_latency_ms"),
                as_float(tile8_row, "hidden_latency_delta_vs_historical_pct"),
                suppressed_zero_limit=SUPPRESSED_PCT_LIMIT,
            )
            historical_energy = baseline_from_pct_delta(
                as_float(tile8_row, "energy_j"),
                as_float(tile8_row, "energy_delta_vs_historical_pct"),
            )
            historical_utilization = baseline_from_pp_delta(
                as_float(tile8_row, "utilization_avg"),
                as_float(tile8_row, "util_delta_vs_historical_pp"),
            )
            latency_trim_pct = pct_delta(as_float(tile8_row, "latency_ms"), as_float(primary_row, "latency_ms"))
            hidden_trim_pct = pct_delta(
                as_float(tile8_row, "integrated_hidden_system_latency_ms"),
                as_float(primary_row, "integrated_hidden_system_latency_ms"),
            )
            energy_trim_pct = pct_delta(as_float(tile8_row, "energy_j"), as_float(primary_row, "energy_j"))

            lines.extend(
                [
                    f"### `{variant['variant_id']}`",
                    "",
                    "| Run Type | Reference | Latency (ms) | Hidden (ms) | Energy (J) | Utilization | Stalls | Backpressure | Residency |",
                    "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
                    (
                        f"| Historical baseline | `{tile8_row['historical_run_id']}` | "
                        f"{fmt_baseline_metric(historical_latency, 6)} | "
                        f"{fmt_baseline_metric(historical_hidden_latency, 6)} | "
                        f"{fmt_baseline_metric(historical_energy, 9)} | "
                        f"{fmt_baseline_metric(historical_utilization, 6)} | "
                        "n/a | n/a | n/a |"
                    ),
                    (
                        f"| Stalled 4x4 replay | `{primary_row['run_id']}` | "
                        f"{fmt_metric(as_float(primary_row, 'latency_ms'), 6)} | "
                        f"{fmt_metric(as_float(primary_row, 'integrated_hidden_system_latency_ms'), 6)} | "
                        f"{fmt_metric(as_float(primary_row, 'energy_j'), 9)} | "
                        f"{fmt_metric(as_float(primary_row, 'utilization_avg'), 6)} | "
                        f"{fmt_metric(as_float(primary_row, 'flow_admission_stalls'))} | "
                        f"{fmt_metric(as_float(primary_row, 'flow_control_backpressure'), 6)} | "
                        f"{fmt_metric(as_float(primary_row, 'flow_residency_hit_rate'), 6)} |"
                    ),
                    (
                        f"| Safe 8x8 hedge | `{tile8_row['run_id']}` | "
                        f"{fmt_metric(as_float(tile8_row, 'latency_ms'), 6)} | "
                        f"{fmt_metric(as_float(tile8_row, 'integrated_hidden_system_latency_ms'), 6)} | "
                        f"{fmt_metric(as_float(tile8_row, 'energy_j'), 9)} | "
                        f"{fmt_metric(as_float(tile8_row, 'utilization_avg'), 6)} | "
                        f"{fmt_metric(as_float(tile8_row, 'flow_admission_stalls'))} | "
                        f"{fmt_metric(as_float(tile8_row, 'flow_control_backpressure'), 6)} | "
                        f"{fmt_metric(as_float(tile8_row, 'flow_residency_hit_rate'), 6)} |"
                    ),
                    "",
                    (
                        f"- stalled `4x4` replay baseline: latency `{fmt_metric(as_float(primary_row, 'latency_ms'), 6)} ms`, "
                        f"hidden latency `{fmt_metric(as_float(primary_row, 'integrated_hidden_system_latency_ms'), 6)} ms`, "
                        f"energy `{fmt_metric(as_float(primary_row, 'energy_j'), 9)} J`, stalls `{fmt_metric(as_float(primary_row, 'flow_admission_stalls'))}`, "
                        f"backpressure `{fmt_metric(as_float(primary_row, 'flow_control_backpressure'), 6)}`"
                    ),
                    (
                        f"- current `8x8` hedge replay: latency `{fmt_metric(as_float(tile8_row, 'latency_ms'), 6)} ms` "
                        f"(`{fmt_pct(latency_trim_pct)}` vs stalled `4x4`), hidden latency `{fmt_metric(as_float(tile8_row, 'integrated_hidden_system_latency_ms'), 6)} ms` "
                        f"(`{fmt_pct(hidden_trim_pct)}` vs stalled `4x4`), energy `{fmt_metric(as_float(tile8_row, 'energy_j'), 9)} J` "
                        f"(`{fmt_pct(energy_trim_pct)}` vs stalled `4x4`), stalls `{fmt_metric(as_float(tile8_row, 'flow_admission_stalls'))}`, "
                        f"backpressure `{fmt_metric(as_float(tile8_row, 'flow_control_backpressure'), 6)}`"
                    ),
                    (
                        f"- remaining historical gap on the `8x8` hedge: latency `{fmt_pct(as_float(tile8_row, 'latency_delta_vs_historical_pct'))}`, "
                        f"hidden latency `{fmt_pct_or_suppressed(as_float(tile8_row, 'hidden_latency_delta_vs_historical_pct'))}`, "
                        f"energy `{fmt_pct(as_float(tile8_row, 'energy_delta_vs_historical_pct'))}`"
                    ),
                ]
            )
            if variant["variant_id"] == "FULLER":
                lines.append(
                    "- bridge gate: `FULLER` is still decisive; a bridge point must stay pressure-free and beat the current `8x8` hedge on latency, hidden latency, and energy together."
                )
            else:
                lines.append(
                    "- bridge gate: remain pressure-free and beat the current `8x8` hedge latency without reopening replay backpressure."
                )
            lines.append("")

    lines.extend(
        [
            "## Proxy Decision Guide",
            "",
            "| Priority | Config | Decision Bucket | Recommended Next Action |",
            "| ---: | --- | --- | --- |",
        ]
    )
    for spec in sorted(BRIDGE_SPECS, key=lambda item: item["priority"]):
        lines.append(
            f"| {spec['priority']} | `{spec['config_id']}` | `{spec['decision_bucket']}` | {spec['recommended_next_action']} |"
        )
    lines.append("")

    for spec in sorted(BRIDGE_SPECS, key=lambda item: item["priority"]):
        if spec["priority"] == 1:
            lines.extend(
                [
                    "## Proxy Priority",
                    "",
                ]
            )
        lines.extend(
            [
                f"{spec['priority']}. `{spec['config_id']}`",
                f"   - purpose: {spec['purpose']}",
                f"   - bridge hypothesis: {spec['proxy_hypothesis']}",
                f"   - replay goal: {spec['replay_goal']}",
            ]
        )

    lines.extend(
        [
            "",
            "## Reintegration Replay Order",
            "",
        ]
    )
    for variant in sorted(REINTEGRATION_VARIANTS, key=lambda item: item["variant_order"]):
        lines.append(f"{variant['variant_order']}. `{variant['variant_id']}`: {variant['variant_goal']}")

    lines.extend(
        [
            "",
            "## Manager Notes",
            "",
            "- Start with `reintegration_package.csv` when you want the compact manager-facing one-row-per-bridge-candidate surface; it now carries the proxy acceptance gate, exact proxy/replay `launch_command` strings, matching post-run review artifact paths plus `run_id` selectors and notes for proxy and every replay lane, numeric proxy limits, the current zero-pressure `8x8` proxy hedge context, a concise `proxy_spec_delta_vs_hedge` summary, the exact packet entry surface plus section locator, per-lane bridge gates, hedge-derived replay thresholds, baseline run IDs, and current replay hedge snapshots.",
            "- Treat `manifest.csv` files as the provenance surface: they now carry `bridge_family`, family-local rank, generator/template paths, upstream source references, winner/hedge lineage IDs, and a compact `lineage_hash` so later turns can reconstruct the ladder without inferring geometry from filenames alone.",
            "- Use `bridge_scorecard.csv` when you need the full wide comparison table with explicit historical/stalled/hedge metric columns plus the proxy hedge context, `proxy_spec_delta_vs_hedge`, and hedge-derived numeric replay targets for every replay lane.",
            "- Treat `bridge_plan.csv` as the launch-grain executor surface when you need launch order, measured replay baselines, proxy hedge context, the candidate-vs-hedge spec delta, hedge-derived numeric pass thresholds, and the exact `launch_command` for each runnable row.",
            "- Treat `manager_slot_recommendation_20260421.md` as the shortest lane-facing slot-decision note when you want the current `Package B repair` default, the exact priority-1 HOPS reopen path, and the crosswalk-first navigation contract without rereading the wider bridge package.",
            "- Treat `manager_eval_surface.csv` as the narrowest rerun-judging digest when you want one row per bridge candidate centered on proxy pressure limits, the current zero-pressure `8x8` proxy hedge reference, the compact candidate-vs-hedge spec delta, the exact packet entry surface plus section locator, paste-ready proxy/`FULLER` commands, exact proxy/`FULLER` post-run review artifact paths plus `run_id` selectors and notes, and the `FULLER` replay targets and baseline context that still define the current `NOT_READY` state.",
            "- Treat `review_intake_surface.csv` as the post-launch review surface: every runnable row already includes the exact results artifact path, `run_id` selector, branch purpose, hedge delta, exact packet entry surface plus section locator, gate summary, numeric pass thresholds, and stop/promote guidance so the next intake turn does not need to join multiple CSVs by hand.",
            "- Treat `priority_execution_crosswalk.csv` as the crosswalk-first branch-level navigator when you need the branch delta vs safe hedge, the explicit `branch_purpose` / legacy `purpose` identity field, current `FLOW_MESO` / `FLOW_PHY` / `FULLER` hedge snapshots, exact packet entry surface plus section locator, row span, and every per-stage experiment/command/review-selector/review-note/gate/success-signal field for each priority without reading the markdown packet prose.",
            "- Treat `priority_execution_packet_index.md` as the secondary branch-local narrative surface after the crosswalk has already isolated the branch; it maps every priority to the source review/launch rows, mirrors each branch's hedge delta plus the same file/section locator, groups the decisive proxy/FULLER gates, and now embeds compact per-step run/command/inspect/success/pass-fail guidance for priorities `2`-`7` when a later turn still needs the fuller branch narrative support.",
            "- Treat `priority1_execution_packet.md` as the full standalone handoff surface for the first `8x8` hedge probe; later priorities stay crosswalk-first and only bounce into the index when the fuller branch narrative or compact step digest is still needed.",
            "- Treat `launch_plan.csv` as the thinnest executor-only queue when you only need ordering, dependency edges, emitted config paths, the exact `launch_command` to paste into the terminal, the expected `results_summary.csv` review artifact plus `run_id` selector to open after each launch lands, and the exact packet entry surface plus section locator for the branch being executed.",
            "- Run proxy configs in priority order first; only replay a bridge point after its E2 proxy output is present.",
            "- Prefer the full 8x8-local ladder before any 4x4 rescue point; the current reintegration evidence already shows the 4x4 surface is pressure-fragile.",
            "- Replay each accepted proxy point in `FLOW_MESO`, then `FLOW_PHY`, then `FULLER` order.",
            "- Proxy acceptance gate: keep `flow_admission_stalls <= proxy_flow_admission_stalls_max` and `flow_control_backpressure <= proxy_flow_control_backpressure_max` in isolated `E2`.",
            "- Replay acceptance gate: keep replay pressure at or below the emitted `*_flow_*_max` limits and beat the emitted hedge-derived `*_latency_target_lt_ms`; `FULLER` rows also need `*_hidden_latency_target_lt_ms` and `*_energy_target_lt_j`.",
            "- `FULLER` remains the final gate; do not treat isolated `E2` gains as promotion-ready if `FULLER` hidden latency or energy stay behind the hedge.",
            "",
        ]
    )
    return "\n".join(lines)


def collect_priority_branch_steps(
    priority: int,
    review_intake_surface_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    branch_steps = sorted(
        [row for row in review_intake_surface_rows if int(row["priority"]) == priority],
        key=lambda item: int(item["launch_order"]),
    )
    if not branch_steps:
        raise ValueError(f"Missing review intake rows for priority {priority}")
    return branch_steps


def stage_label_for_step(step: dict[str, object]) -> str:
    return "Proxy E2" if str(step["stage"]) == "proxy" else f"Replay {step['variant_id']}"


def render_step_contract_lines(
    branch_steps: list[dict[str, object]],
    *,
    heading_prefix: str,
) -> list[str]:
    lines: list[str] = []
    for local_step_index, step in enumerate(branch_steps, start=1):
        dependency_line = []
        if step["depends_on_run_id"]:
            dependency_line.append(f"- depends on: `{step['depends_on_run_id']}`")
        review_note_line = []
        if step["review_artifact_note"]:
            review_note_line.append(f"- review note: {step['review_artifact_note']}")
        lines.extend(
            [
                f"{heading_prefix} Step {local_step_index}: {stage_label_for_step(step)}",
                "",
                f"- launch row: `{step['launch_order']}`",
                f"- run id: `{step['run_id']}`",
                f"- experiment: `{step['experiment_id']}`",
                f"- config: `{step['config_path']}`",
                f"- command: `{step['launch_command']}`",
                *dependency_line,
                f"- inspect after run: open `{step['review_artifact_path']}` and match `{step['review_row_selector']}`",
                *review_note_line,
                f"- gate: {step['gate_summary']}",
                f"- success signal: {step['success_signal']}",
                f"- on pass: {step['on_pass']}",
                f"- on fail: {step['on_fail']}",
                "",
            ]
        )
    return lines


def render_priority_branch_packet(
    priority: int,
    package_rows: list[dict[str, object]],
    review_intake_surface_rows: list[dict[str, object]],
    review_intake_surface_path: Path,
    launch_plan_path: Path,
    manager_eval_surface_path: Path,
) -> str:
    branch_row = next((row for row in package_rows if int(row["priority"]) == priority), None)
    if branch_row is None:
        raise ValueError(f"Missing bridge package row for priority {priority}")

    branch_steps = collect_priority_branch_steps(priority, review_intake_surface_rows)
    launch_orders = [int(row["launch_order"]) for row in branch_steps]
    lines = [
        f"# HOPS v3 Priority-{priority} Execution Packet",
        "",
        f"- branch: `{branch_row['config_id']}`",
        f"- decision bucket: `{branch_row['decision_bucket']}`",
        f"- purpose: {branch_row['purpose']}",
        f"- branch delta vs safe hedge: `{branch_row['proxy_spec_delta_vs_hedge']}`",
        f"- compact review rows: `{min(launch_orders)}`-`{max(launch_orders)}` in `{review_intake_surface_path}`",
        f"- source launch surface: `{launch_plan_path}`",
        f"- source manager eval surface: `{manager_eval_surface_path}`",
        "",
        "## Why This Branch",
        "",
        f"- recommended next action: {branch_row['recommended_next_action']}",
        f"- proxy hypothesis: {branch_row['proxy_hypothesis']}",
        f"- replay goal: {branch_row['replay_goal']}",
        f"- current FLOW_MESO hedge snapshot: `{branch_row['flow_meso_current_hedge_snapshot']}`",
        f"- current FLOW_PHY hedge snapshot: `{branch_row['flow_phy_current_hedge_snapshot']}`",
        f"- current decisive FULLER hedge snapshot: `{branch_row['fuller_current_hedge_snapshot']}`",
        "",
        "## Step Contract",
        "",
    ]

    lines.extend(render_step_contract_lines(branch_steps, heading_prefix="###"))

    lines.extend(
        [
            "## Final Promotion Gate",
            "",
            f"- keep `{branch_row['config_id']}` at `NOT_READY` unless the `FULLER` replay row stays pressure-free and beats all three hedge-defined thresholds together:",
            f"- `latency_ms < {branch_row['fuller_latency_target_lt_ms']}`",
            f"- `integrated_hidden_system_latency_ms < {branch_row['fuller_hidden_latency_target_lt_ms']}`",
            f"- `energy_j < {branch_row['fuller_energy_target_lt_j']}`",
            "- if any one of those thresholds fails, stop the branch and keep the default Package B recommendation.",
            "",
        ]
    )
    return "\n".join(lines)


def render_priority_packet_index(
    package_rows: list[dict[str, object]],
    review_intake_surface_rows: list[dict[str, object]],
    priority_crosswalk_path: Path,
    priority1_packet_path: Path,
    packet_index_path: Path,
    review_intake_surface_path: Path,
    launch_plan_path: Path,
) -> str:
    lines = [
        "# HOPS v3 Priority Execution Packet Index",
        "",
        f"- packet family root: `{priority1_packet_path.parent}`",
        f"- priority execution crosswalk: `{priority_crosswalk_path}`",
        f"- review intake surface: `{review_intake_surface_path}`",
        f"- launch plan: `{launch_plan_path}`",
        f"- use `{packet_index_path}` when the crosswalk has already pointed the turn at a later-priority branch and you want the fuller branch-local narrative without re-reading the wide ladder tables",
        f"- if you only need the branch delta vs safe hedge, the explicit `branch_purpose` / legacy `purpose` field, the exact `entry_surface_kind`, current FLOW_MESO/FLOW_PHY/FULLER hedge snapshots, exact packet path plus section locator, row span, and all four stage experiment/command/review-selector/review-note/gate/success fields for each priority, stay in the crosswalk and only open `{packet_index_path}` when you need the grouped decisive proxy/FULLER gate digest or the compact branch-local narrative",
        "- priority `1` still points at the standalone packet; priorities `2`-`7` now carry compact per-step launch/review guidance inline so later turns can stay branch-local without reopening the CSV surfaces",
        "- priorities `1`-`4` are the full `8x8` local ladder; priorities `5`-`7` stay `4x4` rescue-only and should not be opened before the `8x8` ladder is exhausted",
        "",
        "## Packet Map",
        "",
        "| Priority | Config | Decision Bucket | Entry Kind | Entry Surface | Section Locator | Source Rows | First Command |",
        "| ---: | --- | --- | --- | --- | --- | --- | --- |",
    ]

    ordered_package_rows = sorted(package_rows, key=lambda item: int(item["priority"]))
    for branch_row in ordered_package_rows:
        priority = int(branch_row["priority"])
        branch_steps = collect_priority_branch_steps(priority, review_intake_surface_rows)
        first_step = branch_steps[0]
        source_rows = f"{branch_steps[0]['launch_order']}-{branch_steps[-1]['launch_order']}"
        entry_surface_kind = priority_entry_surface_kind(priority)
        entry_surface = (
            f"`{priority1_packet_path}`" if priority == 1 else f"`{packet_index_path}`"
        )
        section_locator = (
            "`standalone packet root`"
            if priority == 1
            else f"`{priority_packet_index_section_heading(priority, str(branch_row['config_id']))}`"
        )
        lines.append(
            "| "
            f"{priority} | "
            f"`{branch_row['config_id']}` | "
            f"`{branch_row['decision_bucket']}` | "
            f"`{entry_surface_kind}` | "
            f"{entry_surface} | "
            f"{section_locator} | "
            f"`{source_rows}` | "
            f"`{first_step['launch_command']}` |"
        )

    lines.extend(
        [
            "",
            "## Use Order",
            "",
        ]
    )
    for branch_row in ordered_package_rows:
        lines.append(
            f"{int(branch_row['priority'])}. `{branch_row['config_id']}`: {branch_row['recommended_next_action']}"
        )

    lines.extend(
        [
            "",
            "## Branch Gate Digest",
            "",
        ]
    )
    for branch_row in ordered_package_rows:
        priority = int(branch_row["priority"])
        branch_steps = collect_priority_branch_steps(priority, review_intake_surface_rows)
        source_rows = f"{branch_steps[0]['launch_order']}-{branch_steps[-1]['launch_order']}"
        lines.extend(
            [
                priority_packet_index_section_heading(priority, str(branch_row["config_id"])),
                "",
                f"- branch delta vs safe hedge: `{branch_row['proxy_spec_delta_vs_hedge']}`",
                f"- source rows: `{source_rows}` from `{review_intake_surface_path}` and `{launch_plan_path}`",
                (
                    "- decisive proxy gate: keep isolated `E2` at "
                    f"`flow_admission_stalls <= {branch_row.get('proxy_flow_admission_stalls_max')}` and "
                    f"`flow_control_backpressure <= {branch_row.get('proxy_flow_control_backpressure_max')}`"
                ),
                (
                    "- decisive FULLER gate: after a pressure-free replay ladder, beat all three current hedge thresholds "
                    f"together: `latency_ms < {branch_row.get('fuller_latency_target_lt_ms')}`, "
                    f"`integrated_hidden_system_latency_ms < {branch_row.get('fuller_hidden_latency_target_lt_ms')}`, "
                    f"`energy_j < {branch_row.get('fuller_energy_target_lt_j')}`"
                ),
                f"- current FLOW_MESO hedge snapshot: `{branch_row['flow_meso_current_hedge_snapshot']}`",
                f"- current FLOW_PHY hedge snapshot: `{branch_row['flow_phy_current_hedge_snapshot']}`",
                f"- current FULLER hedge snapshot: `{branch_row['fuller_current_hedge_snapshot']}`",
                "",
            ]
        )
        if priority == 1:
            lines.extend(
                [
                    f"- standalone packet: `{priority1_packet_path}`",
                    "",
                ]
            )
            continue
        lines.extend(
            [
                "#### Compact Step Digest",
                "",
            ]
        )
        lines.extend(render_step_contract_lines(branch_steps, heading_prefix="#####"))
    return "\n".join(lines)


def render_manager_slot_recommendation(
    recommendation_path: Path,
    summary_path: Path,
    bridge_scorecard_path: Path,
    manager_eval_surface_path: Path,
    package_rows: list[dict[str, object]],
    review_intake_surface_rows: list[dict[str, object]],
    batch2_rows: list[dict[str, str]],
    comparison_rows: list[dict[str, object]],
    priority_crosswalk_path: Path,
    packet_index_path: Path,
    priority1_packet_path: Path,
    review_intake_surface_path: Path,
    launch_plan_path: Path,
) -> str:
    ordered_package_rows = sorted(package_rows, key=lambda item: int(item["priority"]))
    if not ordered_package_rows:
        raise ValueError("Missing bridge package rows for manager slot recommendation rendering")

    priority1_row = ordered_package_rows[0]
    priority1_steps = collect_priority_branch_steps(1, review_intake_surface_rows)
    priority1_row_span = f"{priority1_steps[0]['launch_order']}-{priority1_steps[-1]['launch_order']}"
    reintegration_root = Path(str(priority1_row["flow_meso_config_path"])).parent

    winner_row = require_row(batch2_rows, "config_id", PRIMARY_WINNER_CONFIG_ID) if batch2_rows else {}
    fuller_comparison = next((row for row in comparison_rows if row.get("variant_id") == "FULLER"), {})
    stalled_pressure_row = comparison_rows[0] if comparison_rows else {}
    recommendation_date = format_date_tag_from_path(recommendation_path)

    lines = [
        "# Lane2 HOPS Slot Recommendation",
        "",
        f"Date: `{recommendation_date}`",
        "Lane: `lane2_hops_v3`",
        "Scope: bounded recommendation for the next serialized Phase3 slot using only the owned HOPS v3 bridge surface",
        "",
        "## Decision",
        "",
        "- current readiness verdict: `NO_CANDIDATE_CLEARS_FULLER_NOT_READY`",
        "- default slot recommendation: keep the next serialized slot on Package B repair rather than spending it on broad HOPS continuation",
        f"- conditional HOPS recommendation: if the manager explicitly wants one narrow HOPS hedge probe anyway, use only priority `1` from the bridge package: `{priority1_row['config_id']}`",
        "",
        "## Why",
        "",
        "1. The isolated winner is real but not reintegration-safe.",
        (
            f"   - `{REFERENCE_PATHS['batch2_results']}` shows `{PRIMARY_WINNER_CONFIG_ID}` as the clear local `E2` leader at "
            f"`bubble_cycles={winner_row.get('bubble_cycles', 'n/a')}`, `latency_ms={winner_row.get('latency_ms', 'n/a')}`, "
            f"`utilization_avg={winner_row.get('utilization_avg', 'n/a')}`, with `flow_admission_stalls={winner_row.get('flow_admission_stalls', 'n/a')}` "
            f"and `flow_control_backpressure={winner_row.get('flow_control_backpressure', 'n/a')}`."
        ),
        "   - That lead is narrow and local. The same sweep does not justify reopening a broad HOPS search space.",
        "",
        "2. The stalled `4x4` replay fails the reintegration gate everywhere that matters.",
        (
            f"   - `{REFERENCE_PATHS['primary_reintegration_results']}` still shows `flow_admission_stalls="
            f"{stalled_pressure_row.get('stalled_flow_admission_stalls', 'n/a')}` and `flow_control_backpressure="
            f"{stalled_pressure_row.get('stalled_flow_control_backpressure', 'n/a')}` in `FLOW_MESO`, `FLOW_PHY`, and `FULLER`."
        ),
        "",
        "3. The safe `8x8` hedge fixes pressure, but it still does not clear readiness.",
        (
            f"   - `{REFERENCE_PATHS['tile8_reintegration_results']}` clears pressure in all three replay lanes with "
            f"`flow_admission_stalls={fuller_comparison.get('hedge_flow_admission_stalls', 'n/a')}` and "
            f"`flow_control_backpressure={fuller_comparison.get('hedge_flow_control_backpressure', 'n/a')}`."
        ),
        (
            "   - The decisive blocker remains `FULLER`: "
            f"`latency_delta_vs_historical_pct={fuller_comparison.get('hedge_vs_historical_latency_pct', 'n/a')}`, "
            f"`hidden_latency_delta_vs_historical_pct={fuller_comparison.get('hedge_vs_historical_hidden_latency_pct', 'n/a')}`, "
            f"`energy_delta_vs_historical_pct={fuller_comparison.get('hedge_vs_historical_energy_pct', 'n/a')}`."
        ),
        (
            f"   - `{REFERENCE_PATHS['reintegration_note']}` already classifies the state as `NOT_READY`, "
            "and the current CSVs still support that call."
        ),
        "",
        "4. The existing bridge package already encodes the right narrow continuation path.",
        (
            f"   - `{bridge_scorecard_path}` ranks the 8x8-local ladder first and keeps all 4x4 rescue attempts later."
        ),
        f"   - The first probe is `{priority1_row['proxy_config_path']}`.",
        f"   - Its replay configs are already emitted under `{reintegration_root}/`.",
        "",
        "## Manager Guidance",
        "",
        "- choose `Package B repair` for the next serialized slot if the question is overall Phase3 progress or the highest-probability unblocker",
        (
            f"- choose `{priority1_row['config_id']}` only if the manager explicitly wants one constrained HOPS hedge probe to test whether lighter prefetch pacing trims the current `8x8` FULLER tax without reopening pressure"
        ),
        "- do not schedule any `4x4` bridge/rescue point before the full 8x8-local ladder, because the current evidence still marks the 4x4 surface as pressure-fragile",
        "",
        "## Exact First HOPS Probe If Authorized",
        "",
        f"- crosswalk-first navigator: `{priority_crosswalk_path}`",
        "  It already carries the explicit `branch_purpose` / legacy `purpose` identity field, branch delta vs safe hedge, current `FLOW_MESO` / `FLOW_PHY` / `FULLER` hedge snapshots, exact packet entry surface plus section locator, row span, and every per-stage experiment / command / review-selector / expected-post-run review note / success / pass-fail field for each priority.",
        f"- branch-local narrative index: `{packet_index_path}`",
        f"- branch-local execution packet: `{priority1_packet_path}`",
        f"- proxy config: `{priority1_row['proxy_config_path']}`",
        "- replay configs:",
        f"  - `{priority1_row['flow_meso_config_path']}`",
        f"  - `{priority1_row['flow_phy_config_path']}`",
        f"  - `{priority1_row['fuller_config_path']}`",
        f"- compact review surface: `{review_intake_surface_path}`",
        f"- execution order reference: `{launch_plan_path}`",
        (
            f"- one-row manager judge surface: `{manager_eval_surface_path}` now mirrors the exact proxy/`FULLER` post-run review artifact paths, `run_id` selectors, and expected-post-run notes beside the paste-ready proxy/`FULLER` commands for each candidate."
        ),
        (
            f"- exact post-run review references: use rows `{priority1_row_span}` in `{review_intake_surface_path}` for the priority-1 branch. "
            "Those rows already carry the expected results artifact path, `run_id` selector, branch purpose, hedge delta, exact packet entry surface plus section locator, numeric gate thresholds, and stop/promote guidance without joining multiple CSVs by hand."
        ),
        (
            f"- if priority `1` lands but later HOPS work is explicitly reopened, open `{priority_crosswalk_path}` first for the exact next packet entry surface plus section locator, the explicit `branch_purpose` / legacy `purpose` identity field, branch delta vs safe hedge, current `FLOW_MESO` / `FLOW_PHY` / `FULLER` hedge snapshots, row span, and the full per-stage experiment / dependency / command / review-selector / review-note / success / pass/fail map; only bounce into `{packet_index_path}` when the fuller branch narrative or compact step digest is needed."
        ),
        "",
        "## Priority-1 Execution Contract",
        "",
        "1. Run only the proxy config first.",
        f"   - command: `{priority1_row['proxy_launch_command']}`",
        (
            f"   - promote only if isolated `E2` stays at `flow_admission_stalls={priority1_row['proxy_flow_admission_stalls_max']}` "
            f"and `flow_control_backpressure={priority1_row['proxy_flow_control_backpressure_max']}`"
        ),
        "2. If the proxy stays pressure-free, replay `FLOW_MESO`, then `FLOW_PHY`, then `FULLER`.",
        "   - stop immediately if any replay lane reopens pressure above `flow_admission_stalls=0` or `flow_control_backpressure=0.0`",
        "3. Treat `FULLER` as the final gate.",
        "   - the probe is still `NOT_READY` unless it beats the current `8x8` hedge on all three metrics together:",
        f"   - `latency_ms < {priority1_row['fuller_latency_target_lt_ms']}`",
        f"   - `integrated_hidden_system_latency_ms < {priority1_row['fuller_hidden_latency_target_lt_ms']}`",
        f"   - `energy_j < {priority1_row['fuller_energy_target_lt_j']}`",
        "4. If the probe misses any one of those `FULLER` thresholds while remaining pressure-free, keep the next serialized slot on Package B repair and do not escalate to the `4x4` rescue branch in the same slot.",
        "",
        "## End State For This Turn",
        "",
        f"- refreshed recommendation surface lives at `{recommendation_path}` and is now generated from `experiments/tools/generate_hops_v3_bridge_batch.py` instead of being hand-maintained beside the rest of the bridge packet",
        f"- `{summary_path}` now lists `{recommendation_path.name}` directly in the generated artifact surface so later turns can discover it from the bridge summary",
        f"- `{priority1_packet_path}` remains the branch-local handoff for the first `8x8` hedge probe",
        f"- `{priority_crosswalk_path}` remains the smallest branch-level navigator for the explicit `branch_purpose` / legacy `purpose` identity field, branch delta vs safe hedge, current `FLOW_MESO` / `FLOW_PHY` / `FULLER` hedge snapshots, exact entry surface plus section locator, row span, and the per-stage experiment / dependency / command / review-selector / review-note / success / pass/fail map for every priority",
        f"- `{packet_index_path}` remains the secondary branch-local narrative surface for later priorities once the crosswalk has already pointed the turn at the right branch",
        f"- `{review_intake_surface_path}` remains the compact post-launch review surface with exact review artifact paths, row selectors, branch purpose, hedge delta, exact packet entry surface plus section locator, gate summaries, numeric thresholds, and stop/promote guidance",
        f"- `{launch_plan_path}` remains the thinnest executor queue with exact config paths, commands, dependency edges, expected review selectors, and the exact packet entry surface plus section locator for each runnable row",
        f"- `{manager_eval_surface_path}` now mirrors exact proxy/`FULLER` post-run review artifact paths, selectors, and expected-post-run notes alongside the one-row decision surface",
        "- no rerun was launched",
        "- no shared hotspot was edited",
    ]
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    proxy_template = load_yaml(args.proxy_template)
    reintegration_template = load_yaml(args.reintegration_template)

    proxy_out_dir = args.proxy_out_dir
    reintegration_out_dir = args.reintegration_out_dir
    proxy_manifest_path = proxy_out_dir / "manifest.csv"
    reintegration_manifest_path = reintegration_out_dir / "manifest.csv"
    summary_path = args.summary_path or proxy_out_dir / "bridge_summary.md"
    package_path = args.package_path or proxy_out_dir / "reintegration_package.csv"
    comparison_path = args.comparison_path
    bridge_scorecard_path = args.bridge_scorecard_path
    bridge_plan_path = args.bridge_plan_path
    manager_eval_surface_path = args.manager_eval_surface_path
    review_intake_surface_path = args.review_intake_surface_path
    priority_crosswalk_path = args.priority_crosswalk_path
    priority1_packet_path = args.priority1_packet_path
    packet_index_path = args.packet_index_path
    manager_slot_recommendation_path = args.manager_slot_recommendation_path
    launch_plan_path = args.launch_plan_path or proxy_out_dir / "launch_plan.csv"

    proxy_out_dir.mkdir(parents=True, exist_ok=True)
    reintegration_out_dir.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    package_path.parent.mkdir(parents=True, exist_ok=True)
    comparison_path.parent.mkdir(parents=True, exist_ok=True)
    bridge_scorecard_path.parent.mkdir(parents=True, exist_ok=True)
    bridge_plan_path.parent.mkdir(parents=True, exist_ok=True)
    manager_eval_surface_path.parent.mkdir(parents=True, exist_ok=True)
    review_intake_surface_path.parent.mkdir(parents=True, exist_ok=True)
    priority_crosswalk_path.parent.mkdir(parents=True, exist_ok=True)
    priority1_packet_path.parent.mkdir(parents=True, exist_ok=True)
    packet_index_path.parent.mkdir(parents=True, exist_ok=True)
    manager_slot_recommendation_path.parent.mkdir(parents=True, exist_ok=True)
    launch_plan_path.parent.mkdir(parents=True, exist_ok=True)

    proxy_manifest_rows: list[dict[str, object]] = []
    reintegration_manifest_rows: list[dict[str, object]] = []
    launch_plan_rows: list[dict[str, object]] = []
    launch_order = 1

    for spec in sorted(BRIDGE_SPECS, key=lambda item: item["priority"]):
        branch_routing_fields = build_branch_routing_fields(
            int(spec["priority"]),
            str(spec["config_id"]),
            branch_purpose=str(spec["purpose"]),
            branch_delta_vs_hedge=render_proxy_spec_delta_vs_hedge(spec),
            priority1_packet_path=priority1_packet_path,
            packet_index_path=packet_index_path,
        )
        proxy_cfg = build_proxy_config(proxy_template, spec, args.run_prefix)
        proxy_run_id = proxy_cfg["run"]["run_id"]
        proxy_path = proxy_out_dir / spec["file_name"]
        with proxy_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(proxy_cfg, handle, sort_keys=False, default_flow_style=False, width=120)

        proxy_manifest_rows.append(
            {
                "config_id": spec["config_id"],
                "priority": spec["priority"],
                "bridge_family": classify_bridge_family(spec),
                "family_priority_rank": family_priority_rank(spec),
                "purpose": spec["purpose"],
                "anchor": spec["anchor"],
                "proxy_hypothesis": spec["proxy_hypothesis"],
                "replay_goal": spec["replay_goal"],
                "decision_bucket": spec["decision_bucket"],
                "recommended_next_action": spec["recommended_next_action"],
                "run_id": proxy_run_id,
                "file_path": str(proxy_path),
                "buffer_depth": spec["buffer_depth"],
                "prefetch_credits": spec["prefetch_credits"],
                "execute_credits": spec["execute_credits"],
                "tile_rows": spec["tile_rows"],
                "tile_cols": spec["tile_cols"],
                "control_issue_width": spec["control_issue_width"],
                "prefetch_distance": spec["prefetch_distance"],
                "exception_lane_policy": spec["exception_lane_policy"],
                "generator_script_path": str(GENERATOR_SCRIPT_PATH),
                "template_path": str(args.proxy_template),
                "source_note_path": str(REFERENCE_PATHS["reintegration_note"]),
                "source_batch2_results_path": str(REFERENCE_PATHS["batch2_results"]),
                "source_primary_reintegration_results_path": str(REFERENCE_PATHS["primary_reintegration_results"]),
                "source_tile8_reintegration_results_path": str(REFERENCE_PATHS["tile8_reintegration_results"]),
                "source_winner_config_id": PRIMARY_WINNER_CONFIG_ID,
                "source_hedge_config_id": PROXY_HEDGE_CONFIG_ID,
                "lineage_hash": build_lineage_hash(
                    {
                        "generator_script_path": str(GENERATOR_SCRIPT_PATH),
                        "template_path": str(args.proxy_template),
                        "config_id": spec["config_id"],
                        "priority": spec["priority"],
                        "bridge_family": classify_bridge_family(spec),
                        "family_priority_rank": family_priority_rank(spec),
                        "buffer_depth": spec["buffer_depth"],
                        "prefetch_credits": spec["prefetch_credits"],
                        "execute_credits": spec["execute_credits"],
                        "tile_rows": spec["tile_rows"],
                        "tile_cols": spec["tile_cols"],
                        "control_issue_width": spec["control_issue_width"],
                        "prefetch_distance": spec["prefetch_distance"],
                        "exception_lane_policy": spec["exception_lane_policy"],
                        "source_note_path": str(REFERENCE_PATHS["reintegration_note"]),
                        "source_batch2_results_path": str(REFERENCE_PATHS["batch2_results"]),
                        "source_primary_reintegration_results_path": str(
                            REFERENCE_PATHS["primary_reintegration_results"]
                        ),
                        "source_tile8_reintegration_results_path": str(
                            REFERENCE_PATHS["tile8_reintegration_results"]
                        ),
                        "source_winner_config_id": PRIMARY_WINNER_CONFIG_ID,
                        "source_hedge_config_id": PROXY_HEDGE_CONFIG_ID,
                    }
                ),
            }
        )
        launch_plan_rows.append(
            {
                "launch_order": launch_order,
                "stage": "proxy",
                "priority": spec["priority"],
                "config_id": spec["config_id"],
                **branch_routing_fields,
                "variant_id": "",
                "experiment_id": "E2",
                "run_id": proxy_run_id,
                "depends_on_run_id": "",
                "config_path": str(proxy_path),
                "launch_command": build_launch_command(proxy_path),
                "review_artifact_path": build_review_artifact_path("proxy", proxy_out_dir, reintegration_out_dir),
                "review_row_selector": build_review_row_selector(proxy_run_id),
                "review_artifact_note": REVIEW_ARTIFACT_NOTE,
                "hypothesis": spec["proxy_hypothesis"],
                "success_signal": "E2 remains pressure-safe while staying inside the bounded bridge neighborhood.",
            }
        )
        launch_order += 1

        upstream_run_id = proxy_run_id
        for variant in sorted(REINTEGRATION_VARIANTS, key=lambda item: item["variant_order"]):
            replay_run_id, replay_cfg = build_reintegration_config(
                reintegration_template,
                proxy_cfg,
                spec,
                variant,
                args.run_prefix,
            )
            replay_path = reintegration_out_dir / f"{spec['config_id']}_{variant['file_suffix']}.yaml"
            with replay_path.open("w", encoding="utf-8") as handle:
                yaml.safe_dump(replay_cfg, handle, sort_keys=False, default_flow_style=False, width=120)

            reintegration_manifest_rows.append(
                {
                    "config_id": spec["config_id"],
                    "priority": spec["priority"],
                    "bridge_family": classify_bridge_family(spec),
                    "family_priority_rank": family_priority_rank(spec),
                    "variant_id": variant["variant_id"],
                    "experiment_id": variant["experiment_id"],
                    "proxy_hypothesis": spec["proxy_hypothesis"],
                    "replay_goal": spec["replay_goal"],
                    "variant_goal": variant["variant_goal"],
                    "run_id": replay_run_id,
                    "proxy_config_path": str(proxy_path),
                    "file_path": str(replay_path),
                    "buffer_depth": spec["buffer_depth"],
                    "prefetch_credits": spec["prefetch_credits"],
                    "execute_credits": spec["execute_credits"],
                    "tile_rows": spec["tile_rows"],
                    "tile_cols": spec["tile_cols"],
                    "control_issue_width": spec["control_issue_width"],
                    "prefetch_distance": spec["prefetch_distance"],
                    "exception_lane_policy": spec["exception_lane_policy"],
                    "generator_script_path": str(GENERATOR_SCRIPT_PATH),
                    "template_path": str(args.reintegration_template),
                    "source_proxy_manifest_path": str(proxy_manifest_path),
                    "source_note_path": str(REFERENCE_PATHS["reintegration_note"]),
                    "source_batch2_results_path": str(REFERENCE_PATHS["batch2_results"]),
                    "source_primary_reintegration_results_path": str(
                        REFERENCE_PATHS["primary_reintegration_results"]
                    ),
                    "source_tile8_reintegration_results_path": str(
                        REFERENCE_PATHS["tile8_reintegration_results"]
                    ),
                    "source_winner_config_id": PRIMARY_WINNER_CONFIG_ID,
                    "source_hedge_config_id": PROXY_HEDGE_CONFIG_ID,
                    "lineage_hash": build_lineage_hash(
                        {
                            "generator_script_path": str(GENERATOR_SCRIPT_PATH),
                            "template_path": str(args.reintegration_template),
                            "source_proxy_manifest_path": str(proxy_manifest_path),
                            "config_id": spec["config_id"],
                            "priority": spec["priority"],
                            "bridge_family": classify_bridge_family(spec),
                            "family_priority_rank": family_priority_rank(spec),
                            "variant_id": variant["variant_id"],
                            "experiment_id": variant["experiment_id"],
                            "buffer_depth": spec["buffer_depth"],
                            "prefetch_credits": spec["prefetch_credits"],
                            "execute_credits": spec["execute_credits"],
                            "tile_rows": spec["tile_rows"],
                            "tile_cols": spec["tile_cols"],
                            "control_issue_width": spec["control_issue_width"],
                            "prefetch_distance": spec["prefetch_distance"],
                            "exception_lane_policy": spec["exception_lane_policy"],
                            "source_note_path": str(REFERENCE_PATHS["reintegration_note"]),
                            "source_batch2_results_path": str(REFERENCE_PATHS["batch2_results"]),
                            "source_primary_reintegration_results_path": str(
                                REFERENCE_PATHS["primary_reintegration_results"]
                            ),
                            "source_tile8_reintegration_results_path": str(
                                REFERENCE_PATHS["tile8_reintegration_results"]
                            ),
                            "source_winner_config_id": PRIMARY_WINNER_CONFIG_ID,
                            "source_hedge_config_id": PROXY_HEDGE_CONFIG_ID,
                        }
                    ),
                }
            )
            launch_plan_rows.append(
                {
                    "launch_order": launch_order,
                    "stage": "reintegration",
                    "priority": spec["priority"],
                    "config_id": spec["config_id"],
                    **branch_routing_fields,
                    "variant_id": variant["variant_id"],
                    "experiment_id": variant["experiment_id"],
                    "run_id": replay_run_id,
                    "depends_on_run_id": upstream_run_id,
                    "config_path": str(replay_path),
                    "launch_command": build_launch_command(replay_path),
                    "review_artifact_path": build_review_artifact_path(
                        "reintegration",
                        proxy_out_dir,
                        reintegration_out_dir,
                    ),
                    "review_row_selector": build_review_row_selector(replay_run_id),
                    "review_artifact_note": REVIEW_ARTIFACT_NOTE,
                    "hypothesis": spec["replay_goal"],
                    "success_signal": variant["variant_goal"],
                }
            )
            upstream_run_id = replay_run_id
            launch_order += 1

    comparison_rows: list[dict[str, object]] = []
    batch2_rows: list[dict[str, str]] = []
    if REFERENCE_PATHS["batch2_results"].exists():
        batch2_rows = read_csv_rows(REFERENCE_PATHS["batch2_results"])
    comparison_sources = [
        REFERENCE_PATHS["primary_reintegration_results"],
        REFERENCE_PATHS["tile8_reintegration_results"],
    ]
    if all(path.exists() for path in comparison_sources):
        comparison_rows = build_measured_reintegration_comparison_rows(
            read_csv_rows(REFERENCE_PATHS["primary_reintegration_results"]),
            read_csv_rows(REFERENCE_PATHS["tile8_reintegration_results"]),
        )

    package_rows = build_reintegration_package_rows(
        proxy_manifest_rows,
        reintegration_manifest_rows,
        comparison_rows,
        build_proxy_hedge_context(batch2_rows),
        priority1_packet_path=priority1_packet_path,
        packet_index_path=packet_index_path,
    )
    bridge_scorecard_rows = build_bridge_scorecard_rows(package_rows, comparison_rows)
    bridge_plan_rows = build_bridge_plan_rows(launch_plan_rows, package_rows, comparison_rows)
    manager_eval_surface_rows = build_manager_eval_surface_rows(package_rows, comparison_rows)
    review_intake_surface_rows = build_review_intake_surface_rows(launch_plan_rows, bridge_plan_rows)
    priority_crosswalk_rows = build_priority_execution_crosswalk_rows(
        package_rows,
        review_intake_surface_rows,
        priority1_packet_path=priority1_packet_path,
        packet_index_path=packet_index_path,
    )

    validate_generated_surfaces(
        package_rows,
        comparison_rows,
        launch_plan_rows,
        review_intake_surface_rows,
        priority_crosswalk_rows,
        manager_eval_surface_rows,
    )

    write_csv(
        proxy_manifest_path,
        fieldnames=[
            "config_id",
            "priority",
            "bridge_family",
            "family_priority_rank",
            "purpose",
            "anchor",
            "proxy_hypothesis",
            "replay_goal",
            "decision_bucket",
            "recommended_next_action",
            "run_id",
            "file_path",
            "buffer_depth",
            "prefetch_credits",
            "execute_credits",
            "tile_rows",
            "tile_cols",
            "control_issue_width",
            "prefetch_distance",
            "exception_lane_policy",
            "generator_script_path",
            "template_path",
            "source_note_path",
            "source_batch2_results_path",
            "source_primary_reintegration_results_path",
            "source_tile8_reintegration_results_path",
            "source_winner_config_id",
            "source_hedge_config_id",
            "lineage_hash",
        ],
        rows=proxy_manifest_rows,
    )

    write_csv(
        reintegration_manifest_path,
        fieldnames=[
            "config_id",
            "priority",
            "bridge_family",
            "family_priority_rank",
            "variant_id",
            "experiment_id",
            "proxy_hypothesis",
            "replay_goal",
            "variant_goal",
            "run_id",
            "proxy_config_path",
            "file_path",
            "buffer_depth",
            "prefetch_credits",
            "execute_credits",
            "tile_rows",
            "tile_cols",
            "control_issue_width",
            "prefetch_distance",
            "exception_lane_policy",
            "generator_script_path",
            "template_path",
            "source_proxy_manifest_path",
            "source_note_path",
            "source_batch2_results_path",
            "source_primary_reintegration_results_path",
            "source_tile8_reintegration_results_path",
            "source_winner_config_id",
            "source_hedge_config_id",
            "lineage_hash",
        ],
        rows=reintegration_manifest_rows,
    )

    write_csv(
        package_path,
        fieldnames=REINTEGRATION_PACKAGE_FIELDNAMES,
        rows=package_rows,
    )

    write_csv(
        comparison_path,
        fieldnames=MEASURED_REINTEGRATION_COMPARISON_FIELDNAMES,
        rows=comparison_rows,
    )

    write_csv(
        bridge_scorecard_path,
        fieldnames=BRIDGE_SCORECARD_FIELDNAMES,
        rows=bridge_scorecard_rows,
    )

    write_csv(
        bridge_plan_path,
        fieldnames=BRIDGE_PLAN_FIELDNAMES,
        rows=bridge_plan_rows,
    )

    write_csv(
        manager_eval_surface_path,
        fieldnames=MANAGER_EVAL_SURFACE_FIELDNAMES,
        rows=manager_eval_surface_rows,
    )

    write_csv(
        review_intake_surface_path,
        fieldnames=REVIEW_INTAKE_SURFACE_FIELDNAMES,
        rows=review_intake_surface_rows,
    )

    write_csv(
        priority_crosswalk_path,
        fieldnames=PRIORITY_EXECUTION_CROSSWALK_FIELDNAMES,
        rows=priority_crosswalk_rows,
    )

    write_csv(
        launch_plan_path,
        fieldnames=LAUNCH_PLAN_FIELDNAMES,
        rows=launch_plan_rows,
    )

    summary_path.write_text(
        render_summary(
            proxy_manifest_path=proxy_manifest_path,
            reintegration_manifest_path=reintegration_manifest_path,
            package_path=package_path,
            comparison_path=comparison_path,
            bridge_scorecard_path=bridge_scorecard_path,
            bridge_plan_path=bridge_plan_path,
            manager_slot_recommendation_path=manager_slot_recommendation_path,
            manager_eval_surface_path=manager_eval_surface_path,
            review_intake_surface_path=review_intake_surface_path,
            priority_crosswalk_path=priority_crosswalk_path,
            priority1_packet_path=priority1_packet_path,
            packet_index_path=packet_index_path,
            launch_plan_path=launch_plan_path,
            proxy_out_dir=proxy_out_dir,
            reintegration_out_dir=reintegration_out_dir,
            run_prefix=args.run_prefix,
        )
        + "\n",
        encoding="utf-8",
    )

    manager_slot_recommendation_path.write_text(
        render_manager_slot_recommendation(
            recommendation_path=manager_slot_recommendation_path,
            summary_path=summary_path,
            bridge_scorecard_path=bridge_scorecard_path,
            manager_eval_surface_path=manager_eval_surface_path,
            package_rows=package_rows,
            review_intake_surface_rows=review_intake_surface_rows,
            batch2_rows=batch2_rows,
            comparison_rows=comparison_rows,
            priority_crosswalk_path=priority_crosswalk_path,
            packet_index_path=packet_index_path,
            priority1_packet_path=priority1_packet_path,
            review_intake_surface_path=review_intake_surface_path,
            launch_plan_path=launch_plan_path,
        )
        + "\n",
        encoding="utf-8",
    )

    priority1_packet_path.write_text(
        render_priority_branch_packet(
            priority=1,
            package_rows=package_rows,
            review_intake_surface_rows=review_intake_surface_rows,
            review_intake_surface_path=review_intake_surface_path,
            launch_plan_path=launch_plan_path,
            manager_eval_surface_path=manager_eval_surface_path,
        )
        + "\n",
        encoding="utf-8",
    )

    packet_index_path.write_text(
        render_priority_packet_index(
            package_rows=package_rows,
            review_intake_surface_rows=review_intake_surface_rows,
            priority_crosswalk_path=priority_crosswalk_path,
            priority1_packet_path=priority1_packet_path,
            packet_index_path=packet_index_path,
            review_intake_surface_path=review_intake_surface_path,
            launch_plan_path=launch_plan_path,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
