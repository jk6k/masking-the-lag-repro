#!/usr/bin/env python3
"""Validate the FULLER Phase 4 paper data figure redesign pack."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUN_TAG = "20260425_fuller_phase4_datafig_redesign_freeze"
FINAL_RUN_TAG = "20260428_fuller_final_unreserved_datafig_broad_scaling_flowmeas_promotion"
LEGACY_FINAL_RUN_TAGS = {
    "20260427_fuller_final_unreserved_datafig_rebuild",
    "20260428_fuller_final_unreserved_datafig_repeat3_promotion",
    "20260428_fuller_final_unreserved_datafig_seed5_repeat10_promotion",
}
FINAL_RUN_TAGS = {FINAL_RUN_TAG, *LEGACY_FINAL_RUN_TAGS}
MECHANISM_RUN_TAG = "20260426_fuller_phase4_mechanism_basis_rerun"
DEFAULT_QUICK_DIR = ROOT / "experiments" / "results" / "quick_reports" / FINAL_RUN_TAG
DEFAULT_MECHANISM_QUICK_DIR = ROOT / "experiments" / "results" / "quick_reports" / FINAL_RUN_TAG
DEFAULT_PACK_DIR = ROOT / "figures" / f"paper_figures_{FINAL_RUN_TAG}"
DEFAULT_REVIEW_DIR = ROOT / "experiments" / "results" / "review" / FINAL_RUN_TAG
DEFAULT_FREEZE_JSON = ROOT / "experiments" / "results" / "paper_sync" / "current_freeze.json"

EXPECTED_CSVS = {
    "Fig6": "fig6_phase4_runtime_accuracy_boundary.csv",
    "Fig7": "fig7_runtime_accuracy_pareto.csv",
    "Fig8": "fig8_claim_support_gate_matrix.csv",
    "Fig9": "fig9_noise_robustness_minimal.csv",
    "Fig10": "fig10_scaling_support_minimal.csv",
    "Fig11": "fig11_device_context.csv",
    "Fig12": "fig12_holdout_claim_boundary.csv",
    "AppF1": "appf1_seed_range_variability.csv",
    "AppF2": "appf2_data_figure_compatibility_matrix.csv",
    "AppF3": "appf3_related_work_radar_scores.csv",
    "AppF4": "appf4_mechanism_ablation_context.csv",
    "AppF5": "appf5_mechanism_energy_breakdown.csv",
    "AppF6": "appf6_det_sparse_sweep_phase4_basis.csv",
}
FINAL_EXPECTED_CSVS = {
    **EXPECTED_CSVS,
    "Fig9": "fig9_noise_robustness_current_basis.csv",
    "Fig10": "fig10_scaling_support_current_basis.csv",
}
FINAL_NUMBERED_EXPECTED_CSVS = {
    "Fig3": "fig3_phase4_runtime_accuracy_boundary.csv",
    "Fig4": "fig4_runtime_accuracy_pareto.csv",
    "Fig5": "fig5_bounded_sensitivity_current_basis.csv",
    "Fig6": "fig6_broad_scaling_flow_buffer_current_basis.csv",
    "Fig7": "fig7_device_context.csv",
    "Fig8": "fig8_holdout_claim_boundary.csv",
    "AppF1": "appf1_seed_range_variability.csv",
    "AppF2": "appf2_data_figure_compatibility_matrix.csv",
    "AppF3": "appf3_related_work_radar_scores.csv",
    "AppF4": "appf4_mechanism_ablation_context.csv",
    "AppF5": "appf5_mechanism_energy_breakdown.csv",
    "AppF6": "appf6_det_sparse_sweep_phase4_basis.csv",
}
MECHANISM_FIGURES = {"AppF4", "AppF5", "AppF6"}
EXPECTED_STEMS = {
    "Fig6": "Fig6_Phase4RuntimeAccuracyBoundary",
    "Fig7": "Fig7_RuntimeAccuracyPareto",
    "Fig8": "Fig8_ClaimSupportGateMatrix",
    "Fig9": "Fig9_NoiseRobustness",
    "Fig10": "Fig10_ScalingSupport",
    "Fig11": "Fig11_DeviceContext",
    "Fig12": "Fig12_HoldoutClaimBoundary",
    "AppF1": "AppF1_SeedRangeVariability",
    "AppF2": "AppF2_DataFigureCompatibility",
    "AppF3": "AppF3_RelatedWorkRadar",
    "AppF4": "AppF4_MechanismAblationContext",
    "AppF5": "AppF5_MechanismEnergyBreakdown",
    "AppF6": "AppF6_DETOperatingPointContext",
}
FINAL_NUMBERED_EXPECTED_STEMS = {
    "Fig3": "Fig3_Phase4RuntimeAccuracyBoundary",
    "Fig4": "Fig4_RuntimeAccuracyPareto",
    "Fig5": "Fig5_BoundedSensitivity",
    "Fig6": "Fig6_ScalingSupport",
    "Fig7": "Fig7_DeviceContext",
    "Fig8": "Fig8_HoldoutClaimBoundary",
    "AppF1": "AppF1_SeedRangeVariability",
    "AppF2": "AppF2_DataFigureCompatibility",
    "AppF3": "AppF3_RelatedWorkRadar",
    "AppF4": "AppF4_MechanismAblationContext",
    "AppF5": "AppF5_MechanismEnergyBreakdown",
    "AppF6": "AppF6_DETOperatingPointContext",
}
EXPECTED_ACTIVE = list(EXPECTED_STEMS)
PHASE4_EXPECTED = {
    "SPARSE": {"top1_mean": 27.1985, "speedup_vs_astra": 14.41},
    "FULLER": {"top1_mean": 20.6459, "speedup_vs_astra": 12.11},
}
MECHANISM_ORDER = ["E0", "E1", "E2", "E3", "E4", "E6"]
ALLOWED_FINAL_FLOW_STATUS = {"measured", "derived_from_trace", "not_applicable_for_baseline"}
FIG10_STABILITY_FIELDS = (
    "latency_ms_std",
    "latency_ms_cv_pct",
    "throughput_images_s_std",
    "throughput_images_s_cv_pct",
    "energy_j_std",
    "energy_j_cv_pct",
)
FIG10_BROAD_STABILITY_FIELDS = (
    *FIG10_STABILITY_FIELDS,
    "flow_buffer_peak_frac_std",
    "flow_buffer_peak_frac_cv_pct",
)
FINAL_FORBIDDEN_STRINGS = (
    "repeat_count remains 1",
    "ready_for_mps_queue_not_promoted",
    "blocked_until_current_basis_noise_rebuild",
    "blocked_until_current_basis_scaling_rebuild",
    "minimum_support_completed_from_retained_context",
    "hardware flow-buffer measurement",
    "universal scaling proof",
    "broad scaling proof",
)
FINAL_EVIDENCE_MAP_FORBIDDEN_STRINGS = (
    "candidate_not_promoted",
    "20260428_fuller_fig9_noise_seed5_candidate",
)
TEXT_SUFFIXES = {".csv", ".json", ".md", ".txt", ".svg"}


def _has_current_basis_final_csvs(quick_dir: Path) -> bool:
    return (
        (quick_dir / "fig9_noise_robustness_current_basis.csv").is_file()
        and (quick_dir / "fig10_scaling_support_current_basis.csv").is_file()
    )


def _has_final_numbered_csvs(quick_dir: Path) -> bool:
    return (
        (quick_dir / "fig5_bounded_sensitivity_current_basis.csv").is_file()
        and (quick_dir / "fig6_broad_scaling_flow_buffer_current_basis.csv").is_file()
        and (quick_dir / "final_numbering_mapping.csv").is_file()
    )


def _has_current_basis_quick_tag(run_tag: str) -> bool:
    quick_dir = ROOT / "experiments" / "results" / "quick_reports" / run_tag
    return _has_current_basis_final_csvs(quick_dir) or _has_final_numbered_csvs(quick_dir)


def is_final_unreserved(quick_dir: Path) -> bool:
    return quick_dir.name in FINAL_RUN_TAGS or _has_current_basis_final_csvs(quick_dir) or _has_final_numbered_csvs(quick_dir)


def is_final_unreserved_pack(pack_dir: Path) -> bool:
    if not pack_dir.name.startswith("paper_figures_"):
        return False
    run_tag = pack_dir.name.removeprefix("paper_figures_")
    return run_tag in FINAL_RUN_TAGS or _has_current_basis_quick_tag(run_tag)


def is_final_unreserved_review(review_dir: Path) -> bool:
    return review_dir.name in FINAL_RUN_TAGS or _has_current_basis_quick_tag(review_dir.name)


def final_run_tag_from_quick_dir(quick_dir: Path) -> str:
    return quick_dir.name if is_final_unreserved(quick_dir) else RUN_TAG


def final_run_tag_from_pack_dir(pack_dir: Path) -> str:
    if is_final_unreserved_pack(pack_dir):
        return pack_dir.name.removeprefix("paper_figures_")
    return RUN_TAG


def final_run_tag_from_review_dir(review_dir: Path) -> str:
    return review_dir.name if is_final_unreserved_review(review_dir) else RUN_TAG


def expected_csvs_for(quick_dir: Path) -> dict[str, str]:
    if _has_final_numbered_csvs(quick_dir):
        return FINAL_NUMBERED_EXPECTED_CSVS
    return FINAL_EXPECTED_CSVS if is_final_unreserved(quick_dir) else EXPECTED_CSVS


def expected_stems_for_pack(pack_dir: Path) -> dict[str, str]:
    run_tag = pack_dir.name.removeprefix("paper_figures_") if pack_dir.name.startswith("paper_figures_") else pack_dir.name
    if _has_current_basis_quick_tag(run_tag) and (pack_dir / "Fig5_BoundedSensitivity.svg").is_file():
        return FINAL_NUMBERED_EXPECTED_STEMS
    return EXPECTED_STEMS


def expected_active_for_quick_dir(quick_dir: Path) -> list[str]:
    return list(expected_csvs_for(quick_dir))


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def f(row: dict[str, str], key: str) -> float:
    return float(row[key])


def check_file(path: Path, errors: list[str], min_size: int = 32) -> None:
    if not path.is_file():
        errors.append(f"missing file: {rel(path)}")
    elif path.stat().st_size < min_size:
        errors.append(f"file is unexpectedly small: {rel(path)} size={path.stat().st_size}")


def check_final_fig9_rows(rows: list[dict[str, str]], errors: list[str]) -> None:
    if not rows:
        errors.append("Fig9 final current-basis CSV is empty")
        return
    fuller_profiles = {row.get("profile") for row in rows if row.get("lane") == "FULLER"}
    required_profiles = {"clean", "mild", "medium", "hard"}
    missing_profiles = sorted(required_profiles - fuller_profiles)
    if missing_profiles:
        errors.append(f"Fig9 final current-basis rows missing profiles: {missing_profiles}")
    modeled_rows = [row for row in rows if row.get("lane") == "FULLER" and str(row.get("model") or "").strip()]
    if modeled_rows:
        required_models = {"mobilevit_s", "mobilevit_xs", "mobilevit_xxs"}
        observed_models = {str(row.get("model") or "") for row in modeled_rows}
        missing_models = sorted(required_models - observed_models)
        if missing_models:
            errors.append(f"Fig9 final current-basis envelope missing models: {missing_models}")
        representative_keys = {
            (str(row.get("model") or ""), str(row.get("profile") or ""))
            for row in modeled_rows
            if row.get("profile_class") == "representative"
        }
        missing_reps = sorted(
            (model, profile)
            for model in required_models
            for profile in required_profiles
            if (model, profile) not in representative_keys
        )
        if missing_reps:
            errors.append(f"Fig9 final current-basis envelope missing representative rows: {missing_reps}")
        envelope_keys = {
            (
                str(row.get("model") or ""),
                str(row.get("noise_sigma_lsb") or ""),
                str(row.get("crosstalk_alpha") or ""),
            )
            for row in modeled_rows
        }
        if len(envelope_keys) < 105:
            errors.append(f"Fig9 final current-basis envelope must expose 105 model/noise cells, observed={len(envelope_keys)}")
    for row in rows:
        if row.get("source_status") == "appendix_context_retained":
            errors.append("Fig9 final current-basis rows must not use retained context")
        if row.get("lane") == "FULLER":
            if row.get("source_status") != "regenerated_current":
                errors.append("Fig9 FULLER rows must be regenerated_current")
            representative = row.get("profile") in required_profiles and row.get("profile_class", "representative") == "representative"
            if representative:
                try:
                    seed_count = int(float(str(row.get("seed_count") or 0)))
                except ValueError:
                    seed_count = 0
                if seed_count < 3:
                    errors.append("Fig9 representative FULLER rows must have seed_count>=3")
            if representative and row.get("complete") != "true":
                errors.append("Fig9 representative FULLER rows must be complete=true")
            if row.get("robustness_claim_status") != "bounded_sensitivity":
                errors.append("Fig9 final preflight allows only bounded_sensitivity")
            for field in ("source_run_tag", "source_csv", "evidence_basis", "noise_sigma_lsb", "crosstalk_alpha"):
                if not str(row.get(field) or "").strip():
                    errors.append(f"Fig9 FULLER row missing {field}")


def check_final_fig10_rows(rows: list[dict[str, str]], errors: list[str]) -> None:
    if not rows:
        errors.append("Fig10 final current-basis CSV is empty")
        return
    observed_keys: set[tuple[str, str, str, str]] = set()
    for row in rows:
        if row.get("source_status") == "appendix_context_retained":
            errors.append("Fig10 final current-basis rows must not use retained context")
        if row.get("source_status") != "regenerated_current":
            errors.append("Fig10 rows must be regenerated_current")
        status = str(row.get("flow_buffer_peak_frac_status") or "").strip()
        if status not in ALLOWED_FINAL_FLOW_STATUS:
            errors.append(f"Fig10 invalid flow_buffer_peak_frac_status={status!r}")
        for field in ("model_variant", "method", "batch_size", "sequence_length", "latency_ms", "throughput_images_s", "energy_j", "repeat_count", "source_run_tag", "source_csv"):
            if not str(row.get(field) or "").strip():
                errors.append(f"Fig10 row missing {field}")
        if row.get("complete") != "true":
            errors.append("Fig10 rows must be complete=true")
        repeat_count = int(float(str(row.get("repeat_count") or 1)))
        if repeat_count > 1:
            for field in FIG10_STABILITY_FIELDS:
                value = str(row.get(field) or "").strip()
                if not value:
                    errors.append(f"Fig10 repeat-backed row missing stability field {field}")
                    continue
                try:
                    if float(value) < 0:
                        errors.append(f"Fig10 stability field {field} must be non-negative")
                except ValueError:
                    errors.append(f"Fig10 stability field {field} must be numeric")
        for metric in ("latency_ms", "throughput_images_s", "energy_j"):
            if str(row.get(metric) or "").strip() and f(row, metric) <= 0:
                errors.append(f"Fig10 {metric} must be positive")
        observed_keys.add(
            (
                str(row.get("model_variant") or ""),
                str(row.get("method") or ""),
                str(row.get("batch_size") or ""),
                str(row.get("sequence_length") or ""),
            )
        )
    required_models = {"MobileViT-S", "MobileViT-XS", "MobileViT-XXS"}
    required_methods = {"ASTRA", "DET", "SPARSE", "FULLER"}
    required_batches = {"1", "2", "4"}
    required_sequences = {"128", "197", "256"}
    missing = [
        key
        for key in (
            (model, method, batch, sequence)
            for model in sorted(required_models)
            for method in sorted(required_methods)
            for batch in sorted(required_batches)
            for sequence in sorted(required_sequences)
        )
        if key not in observed_keys
    ]
    if missing:
        errors.append(f"Fig10 final current-basis grid missing {len(missing)} cells")


def _fig10_broad_text(text: str) -> bool:
    lowered = text.lower()
    return (
        "bounded broader scaling" in lowered
        or "flow-buffer trace evidence" in lowered
        or "flow-buffer trace coverage" in lowered
        or "declared-grid timing" in lowered
    )


def _positive_int(row: dict[str, str], field: str) -> bool:
    try:
        return int(float(str(row.get(field) or "0"))) > 0
    except ValueError:
        return False


def check_bounded_broader_fig10_rows(rows: list[dict[str, str]], errors: list[str]) -> None:
    if not rows:
        errors.append("bounded broader Fig10 claim requires Fig10 rows")
        return

    required_models = {"MobileViT-S", "MobileViT-XS", "MobileViT-XXS"}
    required_methods = {"ASTRA", "DET", "SPARSE", "FULLER"}
    base_batches = {"1", "2", "4", "8"}
    base_sequences = {"96", "128", "197", "256", "320", "384"}
    holdout_points = {("3", "160"), ("3", "224"), ("6", "288"), ("6", "352")}
    base_keys = {
        (
            str(row.get("model_variant") or ""),
            str(row.get("method") or ""),
            str(row.get("batch_size") or ""),
            str(row.get("sequence_length") or ""),
        )
        for row in rows
        if row.get("grid_role") == "declared_expanded_grid"
    }
    holdout_keys = {
        (
            str(row.get("model_variant") or ""),
            str(row.get("method") or ""),
            str(row.get("batch_size") or ""),
            str(row.get("sequence_length") or ""),
        )
        for row in rows
        if row.get("grid_role") == "holdout"
    }
    missing_base = [
        key
        for key in (
            (model, method, batch, sequence)
            for model in sorted(required_models)
            for method in sorted(required_methods)
            for batch in sorted(base_batches)
            for sequence in sorted(base_sequences)
        )
        if key not in base_keys
    ]
    missing_holdout = [
        key
        for key in (
            (model, method, batch, sequence)
            for model in sorted(required_models)
            for method in sorted(required_methods)
            for batch, sequence in sorted(holdout_points)
        )
        if key not in holdout_keys
    ]
    if missing_base:
        errors.append(f"bounded broader Fig10 claim missing expanded-grid cells: {missing_base[:5]}...")
    if missing_holdout:
        errors.append(f"bounded broader Fig10 claim missing holdout cells: {missing_holdout[:5]}...")

    for row in rows:
        try:
            repeat_count = int(float(str(row.get("repeat_count") or "0")))
        except ValueError:
            repeat_count = 0
        if repeat_count < 5:
            errors.append("bounded broader Fig10 rows must have repeat_count>=5")
        for field in FIG10_BROAD_STABILITY_FIELDS:
            value = str(row.get(field) or "").strip()
            if not value:
                errors.append(f"bounded broader Fig10 row missing {field}")
                continue
            try:
                if float(value) < 0:
                    errors.append(f"bounded broader Fig10 {field} must be non-negative")
            except ValueError:
                errors.append(f"bounded broader Fig10 {field} must be numeric")
        if str(row.get("method") or "") == "ASTRA":
            if row.get("flow_buffer_peak_frac_status") != "not_applicable_for_baseline":
                errors.append("bounded broader Fig10 ASTRA rows must mark flow status not_applicable_for_baseline")
            continue
        if row.get("flow_buffer_peak_frac_status") != "measured":
            errors.append("bounded broader Fig10 flow-enabled rows must have measured flow-buffer status")
        if row.get("flow_buffer_measurement_truth_class") != "instrumented_runtime_trace":
            errors.append("bounded broader Fig10 flow-enabled rows must use instrumented_runtime_trace truth class")
        if not str(row.get("flow_buffer_trace_path") or "").strip():
            errors.append("bounded broader Fig10 flow-enabled rows must include flow_buffer_trace_path")
        if not _positive_int(row, "flow_buffer_trace_row_count"):
            errors.append("bounded broader Fig10 flow-enabled rows must include positive flow_buffer_trace_row_count")


def check_final_compliance(compliance: dict, errors: list[str]) -> None:
    if compliance.get("build_allowed") is not True:
        errors.append("final compliance_report must set build_allowed=true")
    if compliance.get("ready_for_branch") is not True:
        errors.append("final compliance_report must set ready_for_branch=true")
    if compliance.get("promotion_status") != "promoted":
        errors.append("final compliance_report must set promotion_status=promoted")
    if compliance.get("blockers") != []:
        errors.append("final compliance_report blockers must be empty")
    serialized = json.dumps(compliance, sort_keys=True)
    for forbidden in ("ready_to_launch", "ready_for_mps_queue_not_promoted"):
        if forbidden in serialized:
            errors.append(f"final compliance_report must not retain pre-promotion status {forbidden!r}")
    preflight = compliance.get("preflight_unblock") or {}
    if preflight.get("mps_queue_status") != "completed_no_cpu_fallback":
        errors.append("final compliance_report must record mps_queue_status=completed_no_cpu_fallback")
    boundary = compliance.get("claim_boundary") or {}
    if isinstance(boundary, str):
        boundary_text = boundary
        boundary = {
            "blocked_lanes": ["DET", "FULLER", "SPARSE"],
            "accuracy_preservation_ready": False,
            "summary": boundary_text,
        }
    if isinstance(boundary, str):
        boundary_text = boundary
        boundary = {
            "summary": boundary_text,
            "blocked_lanes": ["DET", "FULLER", "SPARSE"],
            "accuracy_preservation_ready": False,
            "robustness_claim_ready": False,
            "device_superiority_ready": False,
            "blocked_figures": [],
            "bounded_broader_scaling_ready": "bounded broader scaling" in boundary_text,
            "instrumented_flow_buffer_trace_ready": "flow-buffer trace" in boundary_text,
        }
    if boundary.get("blocked_figures") not in ([], None):
        errors.append("final compliance_report must not keep Fig9/Fig10 as blocked figures")
    for field in ("robustness_claim_ready", "device_superiority_ready"):
        if boundary.get(field) is not False:
            errors.append(f"final compliance_report claim_boundary.{field} must remain false")
    if boundary.get("broad_scaling_claim_ready") not in (False, None):
        if boundary.get("bounded_broader_scaling_ready") is not True or boundary.get("instrumented_flow_buffer_trace_ready") is not True:
            errors.append("broad_scaling_claim_ready requires bounded_broader_scaling_ready and instrumented_flow_buffer_trace_ready")


def check_final_fig8_rows(rows: list[dict[str, str]], quick_dir: Path, errors: list[str], *, final_run_tag: str) -> None:
    by_figure = {row.get("figure_id"): row for row in rows}
    expected_sources = {
        "Fig9": rel(quick_dir / "fig9_noise_robustness_current_basis.csv"),
        "Fig10": rel(quick_dir / "fig10_scaling_support_current_basis.csv"),
        "Fig11": rel(quick_dir / "fig11_device_context.csv"),
        "AppF4": rel(quick_dir / "appf4_mechanism_ablation_context.csv"),
        "AppF5": rel(quick_dir / "appf5_mechanism_energy_breakdown.csv"),
        "AppF6": rel(quick_dir / "appf6_det_sparse_sweep_phase4_basis.csv"),
    }
    expected_status = {
        "Fig9": "regenerated_current",
        "Fig10": "regenerated_current",
        "Fig11": "contextual_boundary_carried_forward",
        "AppF4": "regenerated_current",
        "AppF5": "regenerated_current",
        "AppF6": "regenerated_current",
    }
    expected_source_status = {
        "Fig9": "regenerated_current",
        "Fig10": "regenerated_current",
        "Fig11": "carried_forward_contextual_device_boundary",
        "AppF4": "carried_forward_current_basis_mechanism_promoted",
        "AppF5": "carried_forward_current_basis_mechanism_promoted",
        "AppF6": "carried_forward_current_basis_mechanism_promoted",
    }
    expected_source_run_tag = {
        "Fig9": final_run_tag,
        "Fig10": final_run_tag,
        "Fig11": RUN_TAG,
        "AppF4": MECHANISM_RUN_TAG,
        "AppF5": MECHANISM_RUN_TAG,
        "AppF6": MECHANISM_RUN_TAG,
    }
    for figure_id, expected_source in expected_sources.items():
        row = by_figure.get(figure_id)
        if row is None:
            errors.append(f"Fig8 final gate matrix missing {figure_id}")
            continue
        if row.get("status") != expected_status[figure_id]:
            errors.append(f"Fig8 {figure_id} row must be {expected_status[figure_id]}")
        if row.get("source_status") != expected_source_status[figure_id]:
            errors.append(f"Fig8 {figure_id} source_status must be {expected_source_status[figure_id]}")
        for field in ("source", "source_csv", "source_quick_report_csv"):
            if row.get(field) != expected_source:
                errors.append(f"Fig8 {figure_id} {field} must point to {expected_source}")
        if row.get("source_run_tag") != expected_source_run_tag[figure_id] or row.get("final_run_tag") != final_run_tag:
            errors.append(f"Fig8 {figure_id} must use final run_tag provenance")
        stale_tokens = (
            "minimum_support_completed_from_retained_context",
            "appendix_context_retained",
            "20260403_det_sparse_mlx_final_cpu_context_repair_xs_xxs_dense_successor_freeze",
        )
        row_text = ",".join(str(value) for value in row.values())
        for token in stale_tokens:
            if token in row_text:
                errors.append(f"Fig8 {figure_id} row retains stale token {token}")


def _repeat_counts(rows: list[dict[str, str]]) -> set[int]:
    counts: set[int] = set()
    for row in rows:
        try:
            counts.add(int(float(str(row.get("repeat_count") or 1))))
        except ValueError:
            continue
    return counts


def _contract_seed_requirement(text: str) -> int | None:
    matches = [int(match) for match in re.findall(r"seed[- ]count\s*([0-9]+)", text, flags=re.IGNORECASE)]
    return max(matches) if matches else None


def check_final_claim_contract(
    path: Path,
    errors: list[str],
    *,
    final_run_tag: str = FINAL_RUN_TAG,
    fig9_rows: list[dict[str, str]] | None = None,
    fig10_rows: list[dict[str, str]] | None = None,
) -> None:
    check_file(path, errors)
    if not path.is_file():
        return
    by_figure = {row.get("figure_id"): row for row in load_csv(path)}
    if "Fig5" in by_figure or "Fig6" in by_figure:
        expected_basis = {
            "Fig5": {"current_basis_noise", "current_basis_sensitivity"},
            "Fig6": {"current_basis_scaling", "current_basis_scaling_trace"},
        }
    else:
        expected_basis = {
            "Fig9": {"current_basis_noise"},
            "Fig10": {"current_basis_scaling"},
        }
    for figure_id, evidence_bases in expected_basis.items():
        row = by_figure.get(figure_id)
        if row is None:
            errors.append(f"claim contract missing {figure_id}")
            continue
        claim_allowed = str(row.get("claim_allowed") or "")
        if claim_allowed.startswith("blocked_until_current_basis_"):
            errors.append(f"claim contract {figure_id} must not remain blocked_until_current_basis")
        if row.get("source_run_tag") != final_run_tag:
            errors.append(f"claim contract {figure_id} source_run_tag must be final run tag")
        if row.get("evidence_basis") not in evidence_bases:
            errors.append(
                f"claim contract {figure_id} evidence_basis must be one of {sorted(evidence_bases)}"
            )
        contract_text = " ".join(str(value) for value in row.values())
        contract_text_lower = contract_text.lower()
        for forbidden in ("hardware flow-buffer measurement", "universal scaling proof", "broad scaling proof"):
            if forbidden in contract_text_lower:
                errors.append(f"claim contract {figure_id} contains forbidden wording: {forbidden}")
        forbidden_by_figure = {
            "Fig9": ("broad_robustness", "accuracy_preservation"),
            "Fig10": ("broad_scaling", "flow_buffer_measurement", "accuracy_preservation"),
            "Fig5": ("broad_robustness", "accuracy_preservation"),
            "Fig6": ("broad_scaling", "flow_buffer_measurement", "accuracy_preservation"),
        }
        if figure_id in {"Fig6", "Fig10"} and _fig10_broad_text(contract_text):
            forbidden_by_figure[figure_id] = (
                "universal_scaling",
                "hardware_flow_buffer_measurement",
                "accuracy_preservation",
            )
        must_not_imply = row.get("must_not_imply", "")
        for token in forbidden_by_figure[figure_id]:
            if token not in must_not_imply:
                errors.append(f"claim contract {figure_id} must_not_imply lacks {token}")
    fig10_row = by_figure.get("Fig10") or by_figure.get("Fig6")
    if fig10_row is not None and fig10_rows is not None:
        counts = _repeat_counts(fig10_rows)
        contract_text = " ".join(str(value) for value in fig10_row.values())
        repeat_backed_text = "repeat-backed" in contract_text or "repeat_count=3" in contract_text or "repeat_count > 1" in contract_text
        if repeat_backed_text and (not counts or max(counts) <= 1):
            errors.append("claim contract Fig10 mentions repeat-backed support but Fig10 repeat_count is not >1")
        if counts and max(counts) > 1:
            for field in FIG10_STABILITY_FIELDS:
                if any(not str(row.get(field) or "").strip() for row in fig10_rows):
                    errors.append(f"claim contract Fig10 repeat-backed support requires populated {field}")
        if _fig10_broad_text(contract_text):
            check_bounded_broader_fig10_rows(fig10_rows, errors)
    fig9_row = by_figure.get("Fig9") or by_figure.get("Fig5")
    if fig9_row is not None and fig9_rows is not None:
        requirement = _contract_seed_requirement(" ".join(str(value) for value in fig9_row.values()))
        if requirement is not None:
            representative_rows = [
                row
                for row in fig9_rows
                if row.get("lane") == "FULLER"
                and row.get("profile") in {"clean", "mild", "medium", "hard"}
                and row.get("profile_class", "representative") == "representative"
            ]
            low_seed_rows = [
                row for row in representative_rows
                if int(float(str(row.get("seed_count") or 0))) < requirement
            ]
            if low_seed_rows:
                errors.append(f"claim contract Fig9 mentions seed-count {requirement} but representative rows are lower")


def check_quick_reports(quick_dir: Path, mechanism_quick_dir: Path, errors: list[str]) -> None:
    final_mode = is_final_unreserved(quick_dir)
    expected_run_tag = final_run_tag_from_quick_dir(quick_dir) if final_mode else RUN_TAG
    expected_csvs = expected_csvs_for(quick_dir)
    check_file(quick_dir / "compliance_report.json", errors)
    if not (quick_dir / "compliance_report.json").is_file():
        return
    compliance = load_json(quick_dir / "compliance_report.json")
    if compliance.get("run_tag") != expected_run_tag:
        errors.append("compliance_report run_tag mismatch")
    if not final_mode and (compliance.get("build_allowed") is not True or compliance.get("ready_for_branch") is not True):
        errors.append("compliance_report does not allow branch-ready build")
    if final_mode:
        check_final_compliance(compliance, errors)
    boundary = compliance.get("claim_boundary") or {}
    if boundary.get("accuracy_preservation_ready") is not False:
        errors.append("accuracy_preservation_ready must remain false")
    expected_blocked = ["DET", "FULLER", "SPARSE"] if final_mode else ["FULLER", "SPARSE"]
    if sorted(boundary.get("blocked_lanes") or []) != expected_blocked:
        errors.append("claim boundary must block SPARSE and FULLER")

    if not final_mode:
        check_file(mechanism_quick_dir / "compliance_report.json", errors)
    if not final_mode and (mechanism_quick_dir / "compliance_report.json").is_file():
        mechanism_compliance = load_json(mechanism_quick_dir / "compliance_report.json")
        if mechanism_compliance.get("run_tag") != MECHANISM_RUN_TAG:
            errors.append("mechanism compliance_report run_tag mismatch")
        if mechanism_compliance.get("build_allowed") is not True:
            errors.append("mechanism compliance_report must allow build")

    for figure_id, filename in expected_csvs.items():
        base_dir = quick_dir if final_mode else (mechanism_quick_dir if figure_id in MECHANISM_FIGURES else quick_dir)
        path = base_dir / filename
        check_file(path, errors)
        if path.is_file() and not load_csv(path):
            errors.append(f"empty quick-report CSV: {rel(path)}")

    boundary_figure_id = "Fig3" if "Fig3" in expected_csvs else "Fig6"
    boundary_rows = load_csv(quick_dir / expected_csvs[boundary_figure_id])
    by_lane = {row["lane"]: row for row in boundary_rows}
    for lane, expected in PHASE4_EXPECTED.items():
        row = by_lane.get(lane)
        if row is None:
            errors.append(f"missing {lane} row in Fig6 source")
            continue
        if row.get("claim_boundary") != "accuracy_preservation_claim_blocked":
            errors.append(f"{lane} must be marked accuracy_preservation_claim_blocked")
        if row.get("accuracy_preservation_ready") != "false":
            errors.append(f"{lane} accuracy_preservation_ready must be false")
        if abs(f(row, "top1_mean") - expected["top1_mean"]) > 0.002:
            errors.append(f"{lane} top1_mean stale or unexpected: {row['top1_mean']}")
        if abs(f(row, "speedup_vs_astra") - expected["speedup_vs_astra"]) > 0.03:
            errors.append(f"{lane} speedup_vs_astra stale or unexpected: {row['speedup_vs_astra']}")
        if f(row, "top1_mean") <= 10:
            errors.append(f"{lane} top1_mean looks like a stale near-zero value: {row['top1_mean']}")

    if final_mode:
        if "Fig9" in expected_csvs:
            check_final_fig8_rows(load_csv(quick_dir / expected_csvs["Fig8"]), quick_dir, errors, final_run_tag=expected_run_tag)
            check_final_fig9_rows(load_csv(quick_dir / expected_csvs["Fig9"]), errors)
            check_final_fig10_rows(load_csv(quick_dir / expected_csvs["Fig10"]), errors)
        else:
            check_final_fig9_rows(load_csv(quick_dir / expected_csvs["Fig5"]), errors)
            check_final_fig10_rows(load_csv(quick_dir / expected_csvs["Fig6"]), errors)
    else:
        for row in load_csv(quick_dir / expected_csvs["Fig9"]):
            if row.get("lane") == "FULLER":
                if row.get("source_status") != "appendix_context_retained":
                    errors.append("FULLER noise support must be retained-context labeled")
                if row.get("claim_boundary") != "accuracy_preservation_claim_blocked":
                    errors.append("FULLER noise support must be claim-blocked")
        for row in load_csv(quick_dir / expected_csvs["Fig10"]):
            if row.get("source_status") != "appendix_context_retained":
                errors.append("scaling support must be retained-context labeled")
            if "not_accuracy_preservation" not in row.get("claim_boundary", ""):
                errors.append("scaling support must not imply accuracy preservation")
    device_figure_id = "Fig7" if "Fig7" in expected_csvs else "Fig11"
    device_rows = load_csv(quick_dir / expected_csvs[device_figure_id])
    device_labels = {row.get("device_label", "") for row in device_rows}
    required_device_labels = {
        "Apple M5 Pro CPU measured",
        "Apple M5 Pro GPU (MLX MPS) measured",
        "MTL-FULLER modeled",
    }
    missing_device_labels = sorted(required_device_labels - device_labels)
    if missing_device_labels:
        errors.append(f"device context labels lack specific device names: {missing_device_labels}")
    for row in device_rows:
        if row.get("comparison_boundary") != "contextual_comparison_not_benchmark_equivalence":
            errors.append("device comparison boundary must reject benchmark equivalence")
        if row.get("platform_class") == "HPAT" and "MTL-FULLER" not in row.get("device_label", ""):
            errors.append("modeled accelerator row must be labeled MTL-FULLER")
    holdout_figure_id = "Fig8" if "Fig8" in expected_csvs else "Fig12"
    holdout = load_csv(quick_dir / expected_csvs[holdout_figure_id])
    holdout_by_lane = {row["lane"]: row for row in holdout}
    for lane in ("SPARSE", "FULLER"):
        if holdout_by_lane.get(lane, {}).get("holdout_gate") != "blocked":
            errors.append(f"{lane} holdout gate must be blocked")

    matrix = load_csv(quick_dir / expected_csvs["AppF2"])
    observed_actions = {row["compatibility_action"] for row in matrix}
    for required in ("regenerated_current", "retired_incompatible", "appendix_context_retained", "mechanism_context_retained"):
        if required not in observed_actions:
            errors.append(f"compatibility matrix lacks action {required}")

    if final_mode:
        return

    mechanism = load_csv(mechanism_quick_dir / expected_csvs["AppF4"])
    if [row.get("experiment_id") for row in mechanism] != MECHANISM_ORDER:
        errors.append(f"mechanism ablation context must use current order {MECHANISM_ORDER}")
    for row in mechanism:
        if row.get("run_tag") != MECHANISM_RUN_TAG:
            errors.append("mechanism ablation rows must use current-basis run_tag")
        if row.get("source_status") != "current_phase4_basis_phase1_rerun":
            errors.append("mechanism ablation rows must be current-basis labeled")
        if row.get("compatibility_status") != "current_basis_replaces_retained_context":
            errors.append("mechanism ablation rows must replace retained context")
        if "current_phase4_basis" not in row.get("claim_boundary", ""):
            errors.append("mechanism ablation rows must carry current-basis claim boundary")
    energy = load_csv(mechanism_quick_dir / EXPECTED_CSVS["AppF5"])
    if [row.get("experiment_id") for row in energy] != MECHANISM_ORDER:
        errors.append(f"mechanism energy context must use current order {MECHANISM_ORDER}")
    for row in energy:
        if row.get("run_tag") != MECHANISM_RUN_TAG:
            errors.append("mechanism energy rows must use current-basis run_tag")
        if row.get("source_status") != "current_phase4_basis_phase1_rerun":
            errors.append("mechanism energy rows must be current-basis labeled")
        if row.get("compatibility_status") != "current_basis_replaces_retained_context":
            errors.append("mechanism energy rows must replace retained context")
        total = f(row, "total_energy_mj")
        grouped = f(row, "memory_move_mj") + f(row, "conversion_control_mj") + f(row, "optical_static_mj")
        if abs(total - grouped) > max(0.02, total * 0.01):
            errors.append(f"energy breakdown grouped total mismatch for {row.get('experiment_id')}: total={total} grouped={grouped}")
    appf6_rows = load_csv(mechanism_quick_dir / EXPECTED_CSVS["AppF6"])
    det_rows = [row for row in appf6_rows if row.get("row_type") == "det_k_sweep"]
    sparse_rows = [row for row in appf6_rows if row.get("row_type") == "sparse_tau_sweep"]
    if len(det_rows) != 11 or len(sparse_rows) != 9:
        errors.append(f"AppF6 must contain 11 DET rows and 9 SPARSE rows, observed det={len(det_rows)} sparse={len(sparse_rows)}")
    for row in appf6_rows:
        if row.get("run_tag") != MECHANISM_RUN_TAG:
            errors.append("AppF6 rows must use current-basis run_tag")
        if row.get("source_status") != "current_phase4_basis_full_sweep":
            errors.append("AppF6 rows must be current full-sweep labeled")
        if row.get("complete") != "true":
            errors.append("AppF6 displayed rows must be complete=true")
        if row.get("seed_count") != "3":
            errors.append("AppF6 displayed rows must have seed_count=3")
        if "current_phase4_basis_sweep_measured" not in row.get("claim_boundary", ""):
            errors.append("AppF6 rows must carry measured current-basis claim boundary")
    det_by_k = {f(row, "det_k_global"): row for row in det_rows}
    for required_k in (64.0, 80.0, 96.0, 129.0):
        if required_k not in det_by_k:
            errors.append(f"AppF6 missing DET k={required_k:g}")
    if all(k in det_by_k for k in (64.0, 80.0, 96.0)):
        if not (f(det_by_k[80.0], "top1_mean") < f(det_by_k[64.0], "top1_mean") < f(det_by_k[96.0], "top1_mean")):
            errors.append("AppF6 must preserve DET non-monotonicity: k=80 below k=64 and k=96")
    if 129.0 in det_by_k:
        if abs(f(det_by_k[129.0], "paired_delta_vs_e0_quant_pp") - (-18.067407407407412)) > 0.002:
            errors.append("DET k=129 ASTRA gap mismatch")
        best_det = max(det_rows, key=lambda row: f(row, "top1_mean"))
        if f(best_det, "det_k_global") != 129.0:
            errors.append("DET k=129 must remain the best measured DET point")
    sparse_by_tau = {f(row, "sparse_tau_global"): row for row in sparse_rows}
    if 0.5 not in sparse_by_tau:
        errors.append("AppF6 missing SPARSE tau=0.5")
    else:
        if abs(f(sparse_by_tau[0.5], "paired_delta_vs_e0_quant_pp") - (-16.81555555555556)) > 0.002:
            errors.append("SPARSE tau=0.5 ASTRA gap mismatch")
        best_sparse = max(sparse_rows, key=lambda row: f(row, "top1_mean"))
        if f(best_sparse, "sparse_tau_global") != 0.5:
            errors.append("SPARSE tau=0.5 must remain the best measured SPARSE point")

    for path in sorted(list(quick_dir.glob("*.csv")) + list(mechanism_quick_dir.glob("appf*.csv"))):
        text = path.read_text(encoding="utf-8").lower()
        forbidden = [
            "accuracy_preservation_ready,true",
            "accuracy-preservation ready",
            "accuracy preservation ready",
            "sparse/fuller accuracy preserved",
        ]
        for token in forbidden:
            if token in text:
                errors.append(f"forbidden positive accuracy wording in {rel(path)}: {token}")


def check_figure_outputs(pack_dir: Path, review_dir: Path, errors: list[str]) -> None:
    final_mode = is_final_unreserved_pack(pack_dir)
    final_run_tag = final_run_tag_from_pack_dir(pack_dir) if final_mode else RUN_TAG
    expected_stems = expected_stems_for_pack(pack_dir)
    expected_active = list(expected_stems)
    check_file(pack_dir / "figure_traceability.csv", errors)
    check_file(pack_dir / "figure_numbering_registry.csv", errors)
    check_file(pack_dir / "pack_metadata.json", errors)
    for figure_id, stem in expected_stems.items():
        for suffix in ("svg", "pdf", "png"):
            check_file(pack_dir / f"{stem}.{suffix}", errors, min_size=256)
        check_file(review_dir / f"{stem}_grayscale.png", errors, min_size=256)

    trace_path = pack_dir / "figure_traceability.csv"
    if not trace_path.is_file():
        return
    trace_rows = load_csv(trace_path)
    trace_by_id = {row["figure_id"]: row for row in trace_rows}
    missing_trace = [figure_id for figure_id in expected_active if figure_id not in trace_by_id]
    if missing_trace:
        errors.append(f"traceability missing expected data figures: {missing_trace}")
    for row in trace_rows:
        figure_id = row.get("figure_id")
        if figure_id not in expected_stems:
            continue
        expected_run_tag = final_run_tag if final_mode else (MECHANISM_RUN_TAG if figure_id in MECHANISM_FIGURES else RUN_TAG)
        if row.get("run_tag") != expected_run_tag:
            mechanism_source_run_tag = row.get("source_evidence_run_tag") or row.get("source_data_run_tag")
            final_pack_run_tag = row.get("final_pack_run_tag") or row.get("pack_run_tag")
            mechanism_provenance_ok = (
                final_mode
                and figure_id in MECHANISM_FIGURES
                and row.get("run_tag") == MECHANISM_RUN_TAG
                and mechanism_source_run_tag == MECHANISM_RUN_TAG
                and final_pack_run_tag == final_run_tag
            )
            if not mechanism_provenance_ok:
                errors.append(f"traceability run_tag mismatch for {figure_id}")
        if "composition_only" not in row.get("literature_style_anchors", ""):
            errors.append(f"traceability must scope literature anchors as composition-only for {figure_id}")
        figure_file = ROOT / row.get("figure_file", "")
        input_csv = ROOT / row.get("input_csvs", "")
        check_file(figure_file, errors)
        check_file(input_csv, errors)


def check_review_artifacts(review_dir: Path, quick_dir: Path, errors: list[str]) -> None:
    final_mode = is_final_unreserved_review(review_dir)
    final_run_tag = final_run_tag_from_review_dir(review_dir) if final_mode else RUN_TAG
    expected_active = expected_active_for_quick_dir(quick_dir)
    for filename in (
        "data_figure_brief.md",
        "figure_review_report.md",
        "defect_log.csv",
        "review_manifest.json",
        "figure_traceability.csv",
        "current_basis_mechanism_qa_note.md",
        "manuscript_evidence_map.csv",
    ):
        check_file(review_dir / filename, errors)
    report = review_dir / "figure_review_report.md"
    if report.is_file():
        text = report.read_text(encoding="utf-8")
        for gate in ("Gate 0", "Gate 1", "Gate 2", "Gate 3", "Gate 4"):
            if gate not in text:
                errors.append(f"review report missing {gate}")
        if "current-basis" not in text:
            errors.append("review report must record current-basis mechanism evidence")
    qa_note = review_dir / "current_basis_mechanism_qa_note.md"
    if qa_note.is_file():
        text = qa_note.read_text(encoding="utf-8")
        for figure_id in expected_active:
            if f"| {figure_id} |" not in text:
                errors.append(f"QA note missing {figure_id}")
        required_decisions = ("accept",) if final_mode else ("accept", "revise")
        for decision in required_decisions:
            if f"| {decision} |" not in text:
                errors.append(f"QA note missing decision status {decision}")
    if final_mode:
        check_file(review_dir / "data_review_report.md", errors)
        map_path = review_dir / "manuscript_evidence_map.csv"
        if map_path.is_file():
            map_rows = load_csv(map_path)
            map_ids = [row.get("figure_id") for row in map_rows]
            if map_ids != expected_active:
                errors.append(f"manuscript evidence map figure order mismatch: {map_ids}")
            expected_pack_prefix = f"figures/paper_figures_{final_run_tag}/"
            expected_quick_prefix = rel(quick_dir) + "/"
            expected_traceability = f"figures/paper_figures_{final_run_tag}/figure_traceability.csv"
            expected_compliance = rel(quick_dir / "compliance_report.json")
            expected_contract = rel(review_dir / f"claim_contract_final_unreserved_{final_run_tag[:8]}.csv")
            for row in map_rows:
                figure_id = row.get("figure_id", "")
                row_text = ",".join(str(value) for value in row.values())
                for token in FINAL_EVIDENCE_MAP_FORBIDDEN_STRINGS:
                    if token in row_text:
                        errors.append(f"manuscript evidence map {figure_id} retains stale token {token}")
                if row.get("quick_report_run_tag") != final_run_tag:
                    errors.append(f"manuscript evidence map {figure_id} quick_report_run_tag must be final run tag")
                if row.get("promotion_status") != "promoted":
                    errors.append(f"manuscript evidence map {figure_id} promotion_status must be promoted")
                if not row.get("source_csv", "").startswith(expected_quick_prefix):
                    errors.append(f"manuscript evidence map {figure_id} source_csv must point into current quick reports")
                if row.get("traceability_ref") != expected_traceability:
                    errors.append(f"manuscript evidence map {figure_id} traceability_ref must point to current pack")
                if row.get("compliance_ref") != expected_compliance:
                    errors.append(f"manuscript evidence map {figure_id} compliance_ref must point to current quick reports")
                if row.get("claim_contract_ref") != expected_contract:
                    errors.append(f"manuscript evidence map {figure_id} claim_contract_ref must point to current review contract")
                for figure_file in row.get("figure_files", "").split(";"):
                    if not figure_file.startswith(expected_pack_prefix):
                        errors.append(f"manuscript evidence map {figure_id} figure_files must point to current pack")
        contract_paths = [
            review_dir / f"claim_contract_final_unreserved_{final_run_tag[:8]}.csv",
            review_dir / "claim_contract_final_unreserved_20260427.csv",
        ]
        contract_path = next((path for path in contract_paths if path.is_file()), contract_paths[0])
        expected_csvs = expected_csvs_for(quick_dir)
        fig9_key = "Fig9" if "Fig9" in expected_csvs else "Fig5"
        fig10_key = "Fig10" if "Fig10" in expected_csvs else "Fig6"
        fig9_rows = load_csv(quick_dir / expected_csvs[fig9_key]) if (quick_dir / expected_csvs[fig9_key]).is_file() else None
        fig10_rows = load_csv(quick_dir / expected_csvs[fig10_key]) if (quick_dir / expected_csvs[fig10_key]).is_file() else None
        check_final_claim_contract(
            contract_path,
            errors,
            final_run_tag=final_run_tag,
            fig9_rows=fig9_rows,
            fig10_rows=fig10_rows,
        )


def check_freeze_pointer(freeze_json: Path, quick_dir: Path, mechanism_quick_dir: Path, pack_dir: Path, review_dir: Path, errors: list[str]) -> None:
    check_file(freeze_json, errors)
    if not freeze_json.is_file():
        return
    freeze = load_json(freeze_json)
    final_mode = is_final_unreserved(quick_dir)
    final_run_tag = final_run_tag_from_quick_dir(quick_dir) if final_mode else RUN_TAG
    expected = {
        "run_tag": final_run_tag,
        "paper_figures_dir": rel(pack_dir),
        "quick_report_dir": rel(quick_dir),
        "mechanism_quick_report_dir": rel(quick_dir if final_mode else mechanism_quick_dir),
        "review_dir": rel(review_dir),
    }
    for key, value in expected.items():
        observed = freeze.get(key)
        if observed != value:
            errors.append(f"freeze pointer mismatch for {key}: observed={observed!r} expected={value!r}")


def check_forbidden_final_text_artifacts(paths: list[Path], errors: list[str]) -> None:
    for root in paths:
        if not root.exists():
            continue
        candidates = root.rglob("*") if root.is_dir() else [root]
        for path in candidates:
            if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            lowered = text.lower()
            for token in FINAL_FORBIDDEN_STRINGS:
                if token in text or token.lower() in lowered:
                    errors.append(f"forbidden final artifact text in {rel(path)}: {token}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick_dir", type=Path, default=DEFAULT_QUICK_DIR)
    parser.add_argument("--mechanism_quick_dir", type=Path, default=DEFAULT_MECHANISM_QUICK_DIR)
    parser.add_argument("--pack_dir", type=Path, default=DEFAULT_PACK_DIR)
    parser.add_argument("--review_dir", type=Path, default=DEFAULT_REVIEW_DIR)
    parser.add_argument("--freeze_json", type=Path, default=DEFAULT_FREEZE_JSON)
    parser.add_argument("--require_promoted", action="store_true")
    args = parser.parse_args()

    errors: list[str] = []
    check_quick_reports(args.quick_dir.resolve(), args.mechanism_quick_dir.resolve(), errors)
    check_figure_outputs(args.pack_dir.resolve(), args.review_dir.resolve(), errors)
    check_review_artifacts(args.review_dir.resolve(), args.quick_dir.resolve(), errors)
    if is_final_unreserved(args.quick_dir.resolve()) or is_final_unreserved_pack(args.pack_dir.resolve()) or is_final_unreserved_review(args.review_dir.resolve()):
        check_forbidden_final_text_artifacts(
            [args.quick_dir.resolve(), args.pack_dir.resolve(), args.review_dir.resolve()],
            errors,
        )
    if args.require_promoted:
        check_freeze_pointer(
            args.freeze_json.resolve(),
            args.quick_dir.resolve(),
            args.mechanism_quick_dir.resolve(),
            args.pack_dir.resolve(),
            args.review_dir.resolve(),
            errors,
        )

    if errors:
        for error in errors:
            print(f"[fuller-phase4-paper-data-check][error] {error}", file=sys.stderr)
        return 1

    print(
        "[fuller-phase4-paper-data-check] ok "
        f"run_tag={final_run_tag_from_quick_dir(args.quick_dir.resolve()) if is_final_unreserved(args.quick_dir.resolve()) else RUN_TAG} "
        f"figures={len(EXPECTED_STEMS)} quick_reports={len(expected_csvs_for(args.quick_dir.resolve()))} "
        f"require_promoted={args.require_promoted}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
