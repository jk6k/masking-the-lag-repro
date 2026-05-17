#!/usr/bin/env python3
"""Build final-unreserved Fig9/Fig10 quick reports from current-basis run outputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import statistics
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
RUN_TAG = "20260428_fuller_final_unreserved_datafig_broad_scaling_flowmeas_promotion"
SEED_RUN_TAG = "20260428_fuller_fig9_noise_seed5_candidate"
FIG10_SCALING_CANDIDATE_TAG = "20260428_fuller_broad_scaling_flowmeas_candidate"
MECHANISM_RUN_TAG = "20260426_fuller_phase4_mechanism_basis_rerun"
SOURCE_CONTRACT_TAG = "20260403_det_sparse_mlx_final_cpu_context_repair_xs_xxs_dense"
QUICK_DIR = ROOT / "experiments" / "results" / "quick_reports" / RUN_TAG
REPORT_DATA_DIR = ROOT / "experiments" / "results" / "report_data"
REVIEW_DIR = ROOT / "experiments" / "results" / "review" / RUN_TAG
SEED_QUICK_DIR = ROOT / "experiments" / "results" / "quick_reports" / SEED_RUN_TAG
SEED_REVIEW_DIR = ROOT / "experiments" / "results" / "review" / SEED_RUN_TAG
DEFAULT_SCALING_SUMMARY_CSV = (
    REPORT_DATA_DIR
    / FIG10_SCALING_CANDIDATE_TAG
    / f"fuller_scaling_summary_{SOURCE_CONTRACT_TAG}.csv"
)
GENERATED_AT = "2026-04-28T00:00:00Z"

FIG9_PROFILE_COORDS = {
    "clean": (0.0, 0.0),
    "mild": (0.25, 0.01),
    "medium": (0.5, 0.02),
    "hard": (1.0, 0.05),
}
MODEL_LABELS = {
    "mobilevit_s": "MobileViT-S",
    "mobilevit_xs": "MobileViT-XS",
    "mobilevit_xxs": "MobileViT-XXS",
}
ALLOWED_FLOW_STATUS = {"measured", "derived_from_trace", "not_applicable_for_baseline"}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _seed_directory(source_dir: Path | None, dest_dir: Path) -> None:
    if source_dir is None:
        return
    source_dir = source_dir.resolve()
    dest_dir = dest_dir.resolve()
    if source_dir == dest_dir:
        return
    if not source_dir.is_dir():
        raise SystemExit(f"Missing seed directory: {source_dir}")
    dest_dir.mkdir(parents=True, exist_ok=True)
    for source_path in sorted(source_dir.iterdir()):
        if source_path.is_file():
            shutil.copy2(source_path, dest_dir / source_path.name)


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def _float(row: dict[str, str], field: str, default: float = 0.0) -> float:
    try:
        return float(row.get(field, ""))
    except (TypeError, ValueError):
        return default


def _seed_count(row: dict[str, str]) -> int:
    if str(row.get("seed_count") or "").strip():
        return int(float(row["seed_count"]))
    seeds = [item for item in str(row.get("accuracy_seeds") or "").split(";") if item.strip()]
    return len(set(seeds))


def _source_path(value: str) -> Path | None:
    if not str(value or "").strip():
        return None
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _is_true(value: str) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes"}


def _mean(values: list[float]) -> float | str:
    return statistics.fmean(values) if values else ""


def _sample_std(values: list[float]) -> float | str:
    return statistics.stdev(values) if len(values) > 1 else ""


def _accuracy_uncertainty(source: dict[str, str]) -> dict[str, Any]:
    values = {
        "acc_top1_mean": source.get("acc_top1", ""),
        "acc_top1_std": "",
        "acc_drop_pp_mean": source.get("acc_drop_pp", ""),
        "acc_drop_pp_std": "",
        "ci95_pp": "",
    }
    accuracy_path = _source_path(str(source.get("accuracy_results_csv") or ""))
    if accuracy_path is None or not accuracy_path.is_file():
        return values

    raw_rows = _read_csv(accuracy_path)
    baseline_by_seed = {
        str(row.get("seed") or ""): _float(row, "top1")
        for row in raw_rows
        if _is_true(row.get("baseline", ""))
    }
    quantized_rows = [row for row in raw_rows if not _is_true(row.get("baseline", "")) and str(row.get("top1") or "").strip()]
    top1_values = [_float(row, "top1") for row in quantized_rows]
    drop_values: list[float] = []
    for row in quantized_rows:
        if str(row.get("top1_delta") or "").strip():
            drop_values.append(-_float(row, "top1_delta"))
            continue
        seed = str(row.get("seed") or "")
        if seed in baseline_by_seed:
            drop_values.append(baseline_by_seed[seed] - _float(row, "top1"))

    top1_std = _sample_std(top1_values)
    drop_std = _sample_std(drop_values)
    values.update(
        {
            "acc_top1_mean": _mean(top1_values),
            "acc_top1_std": top1_std,
            "acc_drop_pp_mean": _mean(drop_values),
            "acc_drop_pp_std": drop_std,
            "ci95_pp": (1.96 * float(drop_std) / math.sqrt(len(drop_values))) if drop_values and drop_std != "" else "",
        }
    )
    return values


def _complete(row: dict[str, str], *, required_seed_count: int = 3) -> str:
    if str(row.get("complete") or "").strip():
        return str(row["complete"]).strip().lower()
    return "true" if _seed_count(row) >= required_seed_count else "false"


def build_fig9_rows(*, run_tag: str, noise_summary_csv: Path, fig6_csv: Path) -> list[dict[str, Any]]:
    noise_rows = _read_csv(noise_summary_csv)
    fig6_rows = _read_csv(fig6_csv)
    by_lane = {row.get("lane", ""): row for row in fig6_rows}
    by_coord = {
        (_float(row, "gaussian_noise_std"), _float(row, "crosstalk_alpha")): row
        for row in noise_rows
        if row.get("model") == "mobilevit_s"
    }
    rows: list[dict[str, Any]] = []
    astra = by_lane.get("ASTRA")
    if astra is not None:
        rows.append(
            {
                "run_tag": run_tag,
                "lane": "ASTRA",
                "profile": "clean_reference",
                "acc_top1": astra.get("top1_mean", ""),
                "acc_drop_pp": "0",
                "latency_ms": _float(astra, "seconds_per_sample_mean") * 1000.0,
                "energy_j": "",
                "noise_sigma_lsb": "",
                "crosstalk_alpha": "",
                "seed_count": "",
                "complete": "true",
                "source_status": "regenerated_current_reference_only",
                "source_run_tag": astra.get("source_run_tag") or astra.get("run_tag") or run_tag,
                "source_csv": _rel(fig6_csv),
                "evidence_basis": "current_basis_clean_reference",
                "robustness_claim_status": "bounded_sensitivity",
                "claim_boundary": "runtime_materialization_ready",
                "notes": "Current-basis ASTRA clean reference; no robustness claim is inferred.",
            }
        )
    for profile, (gaussian, alpha) in FIG9_PROFILE_COORDS.items():
        source = by_coord.get((gaussian, alpha))
        if source is None:
            raise SystemExit(f"Missing Fig9 current-basis noise row for profile={profile} gaussian={gaussian} alpha={alpha}")
        seed_count = _seed_count(source)
        uncertainty = _accuracy_uncertainty(source)
        rows.append(
            {
                "run_tag": run_tag,
                "lane": "FULLER",
                "profile": profile,
                "acc_top1": source.get("acc_top1", ""),
                "acc_drop_pp": source.get("acc_drop_pp", ""),
                **uncertainty,
                "latency_ms": source.get("latency_ms", ""),
                "energy_j": source.get("energy_j", ""),
                "noise_sigma_lsb": gaussian,
                "crosstalk_alpha": alpha,
                "seed_count": seed_count,
                "complete": _complete(source),
                "source_status": "regenerated_current",
                "source_run_tag": source.get("source_run_tag") or run_tag,
                "source_csv": _rel(noise_summary_csv),
                "evidence_basis": source.get("evidence_basis") or "current_basis_noise",
                "robustness_claim_status": "bounded_sensitivity",
                "claim_boundary": "bounded_sensitivity_not_broad_robustness",
                "notes": "Current-basis Fig9 sensitivity row; do not promote as broad robustness without full claim review.",
            }
        )
    return rows


def _fig9_profile_for(row: dict[str, str]) -> tuple[str, str]:
    gaussian = _float(row, "gaussian_noise_std")
    alpha = _float(row, "crosstalk_alpha")
    for profile, coords in FIG9_PROFILE_COORDS.items():
        if (gaussian, alpha) == coords:
            return profile, "representative"
    return str(row.get("profile") or f"dense_g{gaussian:g}_a{alpha:g}"), "envelope_scan"


def build_fig9_envelope_rows(*, run_tag: str, noise_summary_csvs: list[Path], fig6_csv: Path) -> list[dict[str, Any]]:
    fig6_rows = _read_csv(fig6_csv)
    by_lane = {row.get("lane", ""): row for row in fig6_rows}
    rows: list[dict[str, Any]] = []
    astra = by_lane.get("ASTRA")
    if astra is not None:
        rows.append(
            {
                "run_tag": run_tag,
                "model": "mobilevit_s",
                "model_variant": MODEL_LABELS["mobilevit_s"],
                "lane": "ASTRA",
                "profile": "clean_reference",
                "profile_class": "reference",
                "sweep_resolution": "",
                "acc_top1": astra.get("top1_mean", ""),
                "acc_drop_pp": "0",
                "latency_ms": _float(astra, "seconds_per_sample_mean") * 1000.0,
                "energy_j": "",
                "noise_sigma_lsb": "",
                "crosstalk_alpha": "",
                "seed_count": "",
                "complete": "true",
                "source_status": "regenerated_current_reference_only",
                "source_run_tag": astra.get("source_run_tag") or astra.get("run_tag") or run_tag,
                "source_csv": _rel(fig6_csv),
                "evidence_basis": "current_basis_clean_reference",
                "robustness_claim_status": "bounded_sensitivity",
                "claim_boundary": "runtime_materialization_ready",
                "notes": "Current-basis ASTRA clean reference; no robustness claim is inferred.",
            }
        )
    seen_representatives: set[tuple[str, str]] = set()
    seen_models: set[str] = set()
    for noise_summary_csv in noise_summary_csvs:
        for source in _read_csv(noise_summary_csv):
            model = str(source.get("model") or "mobilevit_s").strip()
            seen_models.add(model)
            profile, profile_class = _fig9_profile_for(source)
            seed_count = _seed_count(source)
            complete = _complete(source)
            if profile_class == "representative":
                seen_representatives.add((model, profile))
            rows.append(
                {
                    "run_tag": run_tag,
                    "model": model,
                    "model_variant": MODEL_LABELS.get(model, model),
                    "lane": "FULLER",
                    "profile": profile,
                    "profile_class": profile_class,
                    "sweep_resolution": source.get("sweep_resolution") or "dense",
                    "acc_top1": source.get("acc_top1", ""),
                    "acc_drop_pp": source.get("acc_drop_pp", ""),
                    **_accuracy_uncertainty(source),
                    "latency_ms": source.get("latency_ms", ""),
                    "energy_j": source.get("energy_j", ""),
                    "noise_sigma_lsb": source.get("gaussian_noise_std", ""),
                    "crosstalk_alpha": source.get("crosstalk_alpha", ""),
                    "seed_count": seed_count,
                    "complete": complete,
                    "source_status": "regenerated_current",
                    "source_run_tag": source.get("source_run_tag") or run_tag,
                    "source_csv": _rel(noise_summary_csv),
                    "evidence_basis": (
                        "current_basis_noise_representative"
                        if profile_class == "representative"
                        else "current_basis_noise_envelope_single_seed"
                    ),
                    "robustness_claim_status": "bounded_sensitivity",
                    "claim_boundary": (
                        f"bounded_sensitivity_representative_seed{seed_count}"
                        if profile_class == "representative"
                        else "bounded_noise_envelope_context_single_seed"
                    ),
                    "notes": (
                        f"Current-basis representative sensitivity row with seed_count={seed_count}."
                        if profile_class == "representative"
                        else "Current-basis dense-envelope context row; use as sensitivity map, not broad robustness proof."
                    ),
                }
            )
    required_models = set(MODEL_LABELS) if len(noise_summary_csvs) > 1 else seen_models
    required = {(model, profile) for model in required_models for profile in FIG9_PROFILE_COORDS}
    missing = sorted(required - seen_representatives)
    if missing:
        raise SystemExit(f"Missing Fig9 representative envelope rows: {missing}")
    rows.sort(
        key=lambda row: (
            0 if row.get("lane") == "ASTRA" else 1,
            str(row.get("model") or ""),
            float(str(row.get("crosstalk_alpha") or 0)),
            float(str(row.get("noise_sigma_lsb") or 0)),
        )
    )
    return rows


def _flow_status(row: dict[str, str]) -> str:
    status = str(row.get("flow_buffer_peak_frac_status") or "").strip()
    if status:
        return status
    if str(row.get("baseline_variant") or "").strip().upper() == "ASTRA":
        return "not_applicable_for_baseline"
    return "derived_from_trace"


def build_fig10_rows(
    *,
    run_tag: str,
    scaling_summary_csv: Path,
    require_flow_buffer_measured: bool = False,
    broad_scaling_claim_status: str | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    source_rows = _read_csv(scaling_summary_csv)
    broad_ready = broad_scaling_claim_status == "bounded_broader_scaling_ready"
    for source in source_rows:
        status = _flow_status(source)
        if status not in ALLOWED_FLOW_STATUS:
            raise SystemExit(f"Invalid flow_buffer_peak_frac_status={status!r}")
        method = str(source.get("baseline_variant") or "").strip().upper()
        if require_flow_buffer_measured and method != "ASTRA" and status != "measured":
            raise SystemExit(f"Fig10 {method} row lacks measured flow-buffer trace status")
        model = str(source.get("model") or "mobilevit_s").strip()
        repeat_count = source.get("repeat_count") or "1"
        repeat_n = int(float(str(repeat_count)))
        claim_status = (
            broad_scaling_claim_status
            if broad_ready
            else (
                "supported_with_repeat_stability_pending_grid_review"
                if repeat_n > 1
                else "supported_pending_grid_review"
            )
        )
        claim_boundary = (
            "bounded_broader_scaling_with_direct_instrumented_flow_buffer_trace"
            if broad_ready
            else "current_basis_scaling_support"
        )
        rows.append(
            {
                "run_tag": run_tag,
                "model": model,
                "model_variant": MODEL_LABELS.get(model, model),
                "method": method,
                "batch_size": int(float(source["batch_size"])),
                "sequence_length": int(float(source["sequence_length"])),
                "grid_role": source.get("grid_role", ""),
                "latency_ms": source.get("latency_ms", ""),
                "latency_ms_std": source.get("latency_ms_std", ""),
                "latency_ms_cv_pct": source.get("latency_ms_cv_pct", ""),
                "throughput_images_s": source.get("throughput_images_s", ""),
                "throughput_images_s_std": source.get("throughput_images_s_std", ""),
                "throughput_images_s_cv_pct": source.get("throughput_images_s_cv_pct", ""),
                "energy_j": source.get("energy_j", ""),
                "energy_j_std": source.get("energy_j_std", ""),
                "energy_j_cv_pct": source.get("energy_j_cv_pct", ""),
                "flow_buffer_peak_frac": source.get("flow_buffer_peak_frac", ""),
                "flow_buffer_peak_frac_std": source.get("flow_buffer_peak_frac_std", ""),
                "flow_buffer_peak_frac_cv_pct": source.get("flow_buffer_peak_frac_cv_pct", ""),
                "flow_buffer_peak_frac_status": status,
                "flow_buffer_measurement_truth_class": source.get("flow_buffer_measurement_truth_class", ""),
                "flow_buffer_trace_path": source.get("flow_buffer_trace_path", ""),
                "flow_buffer_trace_row_count": source.get("flow_buffer_trace_row_count", ""),
                "repeat_count": repeat_count,
                "complete": "true",
                "source_status": "regenerated_current",
                "source_run_tag": run_tag,
                "source_csv": _rel(scaling_summary_csv),
                "evidence_basis": "current_basis_scaling",
                "scaling_claim_status": claim_status,
                "claim_boundary": claim_boundary,
                "notes": (
                    "Bounded broader scaling row with direct instrumented flow-buffer trace evidence; do not state universal scaling or silicon measurement."
                    if broad_ready
                    else "Current-basis Fig10 repeat-backed scaling row; broad wording requires complete-grid review."
                    if repeat_n > 1
                    else "Current-basis Fig10 scaling row; broad wording requires complete-grid review."
                ),
            }
        )
    rows.sort(key=lambda row: (str(row["model"]), str(row["method"]), int(row["batch_size"]), int(row["sequence_length"])))
    return rows


def _repeat_count_summary(rows: list[dict[str, Any]]) -> str:
    counts = sorted({int(float(str(row.get("repeat_count") or 1))) for row in rows})
    if not counts:
        return "repeat_count unavailable"
    if len(counts) == 1:
        return f"repeat_count={counts[0]}"
    return f"repeat_count range {counts[0]}-{counts[-1]}"


def _fig9_seed_count_summary(rows: list[dict[str, Any]]) -> str:
    counts = sorted(
        {
            int(float(str(row.get("seed_count") or 0)))
            for row in rows
            if row.get("lane") == "FULLER"
            and row.get("profile_class") == "representative"
            and str(row.get("seed_count") or "").strip()
        }
    )
    if not counts:
        return "seed_count unavailable"
    if len(counts) == 1:
        return f"seed-count {counts[0]}"
    return f"seed-count range {counts[0]}-{counts[-1]}"


def build_compliance_payload(
    *,
    run_tag: str,
    quick_dir: Path,
    report_data_dir: Path,
    fig9_source_run_tag: str,
    broad_scaling_claim_status: str | None = None,
) -> dict[str, Any]:
    broad_ready = broad_scaling_claim_status == "bounded_broader_scaling_ready"
    return {
        "blockers": [],
        "build_allowed": True,
        "claim_boundary": {
            "accuracy_preservation_ready": False,
            "blocked_figures": [],
            "blocked_lanes": ["DET", "SPARSE", "FULLER"],
            "bounded_broader_scaling_ready": broad_ready,
            "broad_scaling_claim_ready": broad_ready,
            "device_superiority_ready": False,
            "instrumented_flow_buffer_trace_ready": broad_ready,
            "robustness_claim_ready": False,
        },
        "final_outputs": {
            "fig9_current_basis_csv": _rel(quick_dir / "fig9_noise_robustness_current_basis.csv"),
            "fig10_current_basis_csv": _rel(quick_dir / "fig10_scaling_support_current_basis.csv"),
        },
        "fig9_status": "bounded_sensitivity_envelope_current_basis",
        "fig10_status": (
            "bounded_broader_scaling_with_direct_instrumented_flow_buffer_trace"
            if broad_ready
            else "current_basis_scaling_support"
        ),
        "generated_at": GENERATED_AT,
        "preflight_unblock": {
            "closed_blockers": [
                "C1_FIG9_INPUTS_001",
                "C1_FIG10_GRID_002",
                "C1_CONTRACTS_005",
            ],
            "fixed_inputs": {
                "imagenet_val": "experiments/datasets/imagenet/val",
                "weights_npz_manifest": (
                    "experiments/results/accuracy/"
                    "mlx_weights_20260403_det_sparse_mlx_final_cpu_context_repair_xs_xxs_dense/manifest.json"
                ),
            },
            "mps_queue_status": "completed_no_cpu_fallback",
            "noise_manifest_csv": (
                "experiments/results/report_data/"
                f"fuller_noise_execution_manifest_{fig9_source_run_tag}.csv"
            ),
            "scaling_manifest_json": (
                "experiments/results/generated_configs/"
                "20260427_fuller_final_unreserved_datafig_rebuild/fuller_collection_manifest.json"
            ),
        },
        "promotion_status": "promoted",
        "quick_report_dir": _rel(quick_dir),
        "ready_for_branch": True,
        "report_data_dir": _rel(report_data_dir / run_tag),
        "run_tag": run_tag,
    }


def update_fig8_gate_matrix(
    *,
    run_tag: str,
    quick_dir: Path,
    broad_scaling_claim_status: str | None = None,
) -> None:
    path = quick_dir / "fig8_claim_support_gate_matrix.csv"
    rows = _read_csv(path)
    broad_ready = broad_scaling_claim_status == "bounded_broader_scaling_ready"
    updates = {
        "Fig6": {
            "status": "regenerated_current",
            "source": _rel(quick_dir / "fig6_phase4_runtime_accuracy_boundary.csv"),
            "run_tag": run_tag,
            "final_run_tag": run_tag,
            "source_run_tag": "20260425_fuller_phase4_datafig_redesign_freeze",
            "source_quick_report_csv": _rel(quick_dir / "fig6_phase4_runtime_accuracy_boundary.csv"),
            "original_source_status": "carried_forward_final_unreserved_context",
            "evidence_basis": "current_basis_empirical_carried_forward",
            "promotion_action": "carry_forward_current_freeze_row",
            "source_status": "carried_forward_current_freeze_row",
            "source_csv": _rel(quick_dir / "fig6_phase4_runtime_accuracy_boundary.csv"),
        },
        "Fig7": {
            "status": "regenerated_current",
            "source": _rel(quick_dir / "fig7_runtime_accuracy_pareto.csv"),
            "run_tag": run_tag,
            "final_run_tag": run_tag,
            "source_run_tag": "20260425_fuller_phase4_datafig_redesign_freeze",
            "source_quick_report_csv": _rel(quick_dir / "fig7_runtime_accuracy_pareto.csv"),
            "original_source_status": "carried_forward_final_unreserved_context",
            "evidence_basis": "current_basis_empirical_carried_forward",
            "promotion_action": "carry_forward_current_freeze_row",
            "source_status": "carried_forward_current_freeze_row",
            "source_csv": _rel(quick_dir / "fig7_runtime_accuracy_pareto.csv"),
        },
        "Fig8": {
            "status": "regenerated_current",
            "source": _rel(quick_dir / "fig8_claim_support_gate_matrix.csv"),
            "run_tag": run_tag,
            "final_run_tag": run_tag,
            "source_run_tag": run_tag,
            "source_quick_report_csv": _rel(quick_dir / "fig8_claim_support_gate_matrix.csv"),
            "original_source_status": "governance_matrix_rebuilt",
            "evidence_basis": "claim_gate_administrative",
            "promotion_action": "refresh_governance_matrix",
            "source_status": "regenerated_current",
            "source_csv": _rel(quick_dir / "fig8_claim_support_gate_matrix.csv"),
        },
        "Fig9": {
            "status": "regenerated_current",
            "claim_tier": "bounded_sensitivity_not_broad_robustness",
            "accuracy_preservation_status": "not_used_for_positive_claim",
            "gate": "current_basis_noise_completed",
            "source": _rel(quick_dir / "fig9_noise_robustness_current_basis.csv"),
            "run_tag": run_tag,
            "final_run_tag": run_tag,
            "source_run_tag": run_tag,
            "source_quick_report_csv": _rel(quick_dir / "fig9_noise_robustness_current_basis.csv"),
            "original_source_status": "",
            "evidence_basis": "current_basis_noise",
            "promotion_action": "promoted_bounded_sensitivity",
            "source_status": "regenerated_current",
            "source_csv": _rel(quick_dir / "fig9_noise_robustness_current_basis.csv"),
        },
        "Fig10": {
            "status": "regenerated_current",
            "claim_tier": (
                "bounded_broader_scaling_with_direct_instrumented_flow_buffer_trace"
                if broad_ready
                else "current_basis_scaling_support"
            ),
            "accuracy_preservation_status": "not_used_for_positive_claim",
            "gate": (
                "bounded_broader_scaling_flow_trace_completed"
                if broad_ready
                else "current_basis_scaling_grid_completed"
            ),
            "source": _rel(quick_dir / "fig10_scaling_support_current_basis.csv"),
            "run_tag": run_tag,
            "final_run_tag": run_tag,
            "source_run_tag": run_tag,
            "source_quick_report_csv": _rel(quick_dir / "fig10_scaling_support_current_basis.csv"),
            "original_source_status": "",
            "evidence_basis": "current_basis_scaling",
            "promotion_action": (
                "promoted_bounded_broader_scaling_flow_trace_context"
                if broad_ready
                else "promoted_current_basis_scaling_context"
            ),
            "source_status": "regenerated_current",
            "source_csv": _rel(quick_dir / "fig10_scaling_support_current_basis.csv"),
        },
        "Fig11": {
            "status": "contextual_boundary_carried_forward",
            "claim_tier": "device_context_not_benchmark_equivalence",
            "accuracy_preservation_status": "not_used_for_positive_claim",
            "gate": "contextual_device_boundary_explicit",
            "source": _rel(quick_dir / "fig11_device_context.csv"),
            "run_tag": run_tag,
            "final_run_tag": run_tag,
            "source_run_tag": "20260425_fuller_phase4_datafig_redesign_freeze",
            "source_quick_report_csv": _rel(quick_dir / "fig11_device_context.csv"),
            "original_source_status": "contextual_device_boundary_carried_forward",
            "evidence_basis": "contextual_measured_host_plus_modeled_endpoint",
            "promotion_action": "carry_forward_contextual_device_boundary",
            "source_status": "carried_forward_contextual_device_boundary",
            "source_csv": _rel(quick_dir / "fig11_device_context.csv"),
        },
        "Fig12": {
            "status": "claim_blocking_report_generated",
            "source": _rel(quick_dir / "fig12_holdout_claim_boundary.csv"),
            "run_tag": run_tag,
            "final_run_tag": run_tag,
            "source_run_tag": "20260425_fuller_phase4_datafig_redesign_freeze",
            "source_quick_report_csv": _rel(quick_dir / "fig12_holdout_claim_boundary.csv"),
            "original_source_status": "carried_forward_final_unreserved_context",
            "evidence_basis": "current_basis_claim_boundary_carried_forward",
            "promotion_action": "carry_forward_current_freeze_row",
            "source_status": "carried_forward_current_freeze_row",
            "source_csv": _rel(quick_dir / "fig12_holdout_claim_boundary.csv"),
        },
        "AppF1": {
            "status": "regenerated_current",
            "source": _rel(quick_dir / "appf1_seed_range_variability.csv"),
            "run_tag": run_tag,
            "final_run_tag": run_tag,
            "source_run_tag": "20260425_fuller_phase4_datafig_redesign_freeze",
            "source_quick_report_csv": _rel(quick_dir / "appf1_seed_range_variability.csv"),
            "original_source_status": "carried_forward_final_unreserved_context",
            "evidence_basis": "current_basis_variability_context",
            "promotion_action": "carry_forward_current_freeze_row",
            "source_status": "carried_forward_current_freeze_row",
            "source_csv": _rel(quick_dir / "appf1_seed_range_variability.csv"),
        },
        "AppF2": {
            "status": "regenerated_current",
            "source": _rel(quick_dir / "appf2_data_figure_compatibility_matrix.csv"),
            "run_tag": run_tag,
            "final_run_tag": run_tag,
            "source_run_tag": run_tag,
            "source_quick_report_csv": _rel(quick_dir / "appf2_data_figure_compatibility_matrix.csv"),
            "original_source_status": "compatibility_matrix_carried_forward",
            "evidence_basis": "administrative_audit_metadata",
            "promotion_action": "refresh_pack_local_path",
            "source_status": "carried_forward_current_freeze_row",
            "source_csv": _rel(quick_dir / "appf2_data_figure_compatibility_matrix.csv"),
        },
        "AppF3": {
            "status": "appendix_context_retained",
            "source": _rel(quick_dir / "appf3_related_work_radar_scores.csv"),
            "run_tag": run_tag,
            "final_run_tag": run_tag,
            "source_run_tag": "20260403_det_sparse_mlx_final_cpu_context_repair_xs_xxs_dense_successor_freeze",
            "source_quick_report_csv": _rel(quick_dir / "appf3_related_work_radar_scores.csv"),
            "original_source_status": "appendix_context_retained",
            "evidence_basis": "qualitative_literature_context_only",
            "promotion_action": "carry_forward_appendix_context",
            "source_status": "appendix_context_retained",
            "source_csv": _rel(quick_dir / "appf3_related_work_radar_scores.csv"),
        },
        "AppF4": {
            "status": "regenerated_current",
            "claim_tier": "mechanism_current_basis_tradeoff_context",
            "accuracy_preservation_status": "not_used_for_positive_claim",
            "gate": "current_basis_mechanism_promoted",
            "source": _rel(quick_dir / "appf4_mechanism_ablation_context.csv"),
            "run_tag": run_tag,
            "final_run_tag": run_tag,
            "source_run_tag": MECHANISM_RUN_TAG,
            "source_quick_report_csv": _rel(quick_dir / "appf4_mechanism_ablation_context.csv"),
            "original_source_status": "current_phase4_basis_phase1_rerun",
            "evidence_basis": "current_basis_mechanism_empirical",
            "promotion_action": "carry_forward_current_basis_mechanism",
            "source_status": "carried_forward_current_basis_mechanism_promoted",
            "source_csv": _rel(quick_dir / "appf4_mechanism_ablation_context.csv"),
        },
        "AppF5": {
            "status": "regenerated_current",
            "claim_tier": "mechanism_current_basis_tradeoff_context",
            "accuracy_preservation_status": "not_used_for_positive_claim",
            "gate": "current_basis_mechanism_promoted",
            "source": _rel(quick_dir / "appf5_mechanism_energy_breakdown.csv"),
            "run_tag": run_tag,
            "final_run_tag": run_tag,
            "source_run_tag": MECHANISM_RUN_TAG,
            "source_quick_report_csv": _rel(quick_dir / "appf5_mechanism_energy_breakdown.csv"),
            "original_source_status": "current_phase4_basis_phase1_rerun",
            "evidence_basis": "current_basis_mechanism_empirical",
            "promotion_action": "carry_forward_current_basis_mechanism",
            "source_status": "carried_forward_current_basis_mechanism_promoted",
            "source_csv": _rel(quick_dir / "appf5_mechanism_energy_breakdown.csv"),
        },
        "AppF6": {
            "status": "regenerated_current",
            "claim_tier": "mechanism_current_basis_tradeoff_context",
            "accuracy_preservation_status": "not_used_for_positive_claim",
            "gate": "current_basis_mechanism_sweep_promoted",
            "source": _rel(quick_dir / "appf6_det_sparse_sweep_phase4_basis.csv"),
            "run_tag": run_tag,
            "final_run_tag": run_tag,
            "source_run_tag": MECHANISM_RUN_TAG,
            "source_quick_report_csv": _rel(quick_dir / "appf6_det_sparse_sweep_phase4_basis.csv"),
            "original_source_status": "current_phase4_basis_full_sweep",
            "evidence_basis": "current_basis_mechanism_empirical",
            "promotion_action": "carry_forward_current_basis_mechanism",
            "source_status": "carried_forward_current_basis_mechanism_promoted",
            "source_csv": _rel(quick_dir / "appf6_det_sparse_sweep_phase4_basis.csv"),
        },
    }
    seen: set[str] = set()
    for row in rows:
        figure_id = row.get("figure_id", "")
        if figure_id in updates:
            row.update(updates[figure_id])
            seen.add(figure_id)
    required_updates = {"Fig9", "Fig10", "Fig11", "AppF4", "AppF5", "AppF6"}
    missing = sorted(required_updates - seen)
    if missing:
        raise SystemExit(f"Missing Fig8 gate matrix rows: {missing}")
    _write_csv(path, list(rows[0].keys()), rows)


def update_claim_contract(
    *,
    run_tag: str,
    review_dir: Path,
    fig9_seed_count_summary: str = "seed-count 3",
    fig10_repeat_count_summary: str = "repeat_count=1",
    broad_scaling_claim_status: str | None = None,
) -> None:
    path = review_dir / "claim_contract_final_unreserved_20260427.csv"
    rows = _read_csv(path)
    repeat_backed = fig10_repeat_count_summary != "repeat_count=1" and "unavailable" not in fig10_repeat_count_summary
    broad_ready = broad_scaling_claim_status == "bounded_broader_scaling_ready"
    updates = {
        "Fig9": {
            "claim_allowed": "bounded sensitivity only",
            "evidence_basis": "current_basis_noise",
            "source_run_tag": run_tag,
            "must_not_imply": "broad_robustness;accuracy_preservation;noise_tolerance_guarantee",
            "caption_boundary": (
                "Three-model dense current-basis noise envelope plus clean/mild/medium/hard "
                f"{fig9_seed_count_summary} representative rows are regenerated for bounded sensitivity context only; "
                "do not state broad robustness."
            ),
        },
        "Fig10": {
            "claim_allowed": (
                "bounded broader scaling with direct instrumented flow-buffer trace evidence"
                if broad_ready
                else "current-basis scaling support with repeat-backed stability evidence and grid-review caveats"
                if repeat_backed
                else "current-basis scaling support with grid-review caveats"
            ),
            "evidence_basis": "current_basis_scaling",
            "source_run_tag": run_tag,
            "must_not_imply": (
                "universal_scaling;hardware_flow_buffer_measurement;silicon_measurement;accuracy_preservation;device_superiority"
                if broad_ready
                else "broad_scaling_proof;flow_buffer_measurement;accuracy_preservation"
            ),
            "caption_boundary": (
                "Predeclared expanded-grid and holdout cells are present with direct instrumented "
                f"flow-buffer traces for non-baseline rows and {fig10_repeat_count_summary}; "
                "this remains bounded evidence, not a universal law or silicon measurement."
                if broad_ready
                else (
                    "All 108 MobileViT-S/XS/XXS x ASTRA/DET/SPARSE/FULLER x batch/sequence "
                    f"cells are present; flow-buffer status is derived_from_trace and {fig10_repeat_count_summary}"
                    f"{'; repeat-backed stability fields are populated' if repeat_backed else ''}."
                )
            ),
        },
    }
    seen: set[str] = set()
    for row in rows:
        figure_id = row.get("figure_id", "")
        if figure_id in updates:
            row.update(updates[figure_id])
            seen.add(figure_id)
    missing = sorted(set(updates) - seen)
    if missing:
        raise SystemExit(f"Missing claim contract rows: {missing}")
    _write_csv(path, list(rows[0].keys()), rows)
    if run_tag[:8].isdigit() and run_tag[:8] != "20260427":
        successor = review_dir / f"claim_contract_final_unreserved_{run_tag[:8]}.csv"
        _write_csv(successor, list(rows[0].keys()), rows)


def update_manuscript_evidence_map(
    *,
    run_tag: str,
    quick_dir: Path,
    review_dir: Path,
    fig10_rows: list[dict[str, Any]],
    broad_scaling_claim_status: str | None = None,
) -> None:
    if broad_scaling_claim_status != "bounded_broader_scaling_ready":
        return
    path = review_dir / "manuscript_evidence_map.csv"
    if not path.is_file():
        return
    rows = _read_csv(path)
    if not rows:
        return
    source_csv = quick_dir / "fig10_broad_scaling_flow_buffer_current_basis.csv"
    current_pack = f"figures/paper_figures_{run_tag}"
    current_traceability = f"{current_pack}/figure_traceability.csv"
    current_compliance = _rel(quick_dir / "compliance_report.json")
    current_claim_contract = _rel(review_dir / f"claim_contract_final_unreserved_{run_tag[:8]}.csv")
    for row in rows:
        figure_id = row.get("figure_id", "")
        stem = row.get("canonical_stem", "")
        source_name = Path(str(row.get("source_csv") or "")).name
        if figure_id == "Fig10":
            source_name = source_csv.name
        row.update(
            {
                "figure_files": (
                    f"{current_pack}/{stem}.svg;"
                    f"{current_pack}/{stem}.pdf;"
                    f"{current_pack}/{stem}.png"
                ),
                "source_csv": _rel(quick_dir / source_name),
                "quick_report_run_tag": run_tag,
                "traceability_ref": current_traceability,
                "compliance_ref": current_compliance,
                "claim_contract_ref": current_claim_contract,
                "promotion_status": "promoted",
                "claim_upgrade_condition": "Already promoted inside the current freeze; stronger wording requires a new promoted evidence update.",
            }
        )
        if row.get("source_run_tag") == SEED_RUN_TAG or figure_id in {"Fig8", "Fig9", "AppF2"}:
            row["source_run_tag"] = run_tag
        if figure_id != "Fig10":
            continue
        row.update(
            {
                "source_run_tag": run_tag,
                "evidence_basis": "current_basis_scaling",
                "source_status": "regenerated_current",
                "row_count": len(fig10_rows),
                "key_evidence_scope": "Bounded broader scaling evidence over declared expanded-grid and holdout cells.",
                "fig8_status": "regenerated_current",
                "fig8_claim_tier": "bounded_broader_scaling_with_direct_instrumented_flow_buffer_trace",
                "fig8_gate": "bounded_broader_scaling_flow_trace_completed",
                "allowed_claim": "bounded broader scaling with direct instrumented flow-buffer trace evidence",
                "forbidden_claims": "universal_scaling;hardware_flow_buffer_measurement;silicon_measurement;accuracy_preservation;device_superiority",
                "caption_boundary": "Expanded-grid and holdout cells are present; non-baseline flow-buffer rows use direct instrumented runtime traces.",
                "reviewer_caveat": "Bounded evidence only; not a universal law, silicon measurement, or accuracy-preservation claim.",
                "do_not_say": "Do not say universal scaling, silicon measurement, device superiority, or accuracy preservation.",
                "repeat_count_summary": _repeat_count_summary(fig10_rows),
                "stability_fields_present": "true",
                "flow_buffer_peak_frac_status": "measured_nonbaseline;not_applicable_for_baseline",
                "claim_upgrade_condition": "Already promoted for bounded broader scaling and flow-trace evidence; universal, silicon, device-superiority, or accuracy-preservation claims require a new promoted evidence update.",
            }
        )
    _write_csv(path, list(rows[0].keys()), rows)


def build_reports(
    *,
    run_tag: str,
    quick_dir: Path,
    report_data_dir: Path,
    review_dir: Path,
    seed_quick_dir: Path | None = None,
    seed_review_dir: Path | None = None,
    fig9_source_run_tag: str = SEED_RUN_TAG,
    fig10_source_run_tag: str = FIG10_SCALING_CANDIDATE_TAG,
    noise_summary_csv: Path | None = None,
    scaling_summary_csv: Path | None = None,
    broad_scaling_summary_csv: Path | None = None,
    require_flow_buffer_measured: bool = False,
    broad_scaling_claim_status: str | None = None,
) -> None:
    _seed_directory(seed_quick_dir, quick_dir)
    _seed_directory(seed_review_dir, review_dir)
    fig6_csv = quick_dir / "fig6_phase4_runtime_accuracy_boundary.csv"
    fig9_report_dir = report_data_dir / fig9_source_run_tag
    run_report_dir = report_data_dir / run_tag
    if noise_summary_csv is not None:
        noise_csvs = [noise_summary_csv]
    else:
        noise_csvs = sorted(fig9_report_dir.glob(f"fuller_noise_accuracy_summary_*_dense_{SOURCE_CONTRACT_TAG}.csv"))
    if not noise_csvs:
        raise SystemExit(f"Missing Fig9 noise summary CSVs under {fig9_report_dir}")
    scaling_csv = broad_scaling_summary_csv or scaling_summary_csv or (
        DEFAULT_SCALING_SUMMARY_CSV if DEFAULT_SCALING_SUMMARY_CSV.is_file()
        else run_report_dir / f"fuller_scaling_summary_{SOURCE_CONTRACT_TAG}.csv"
    )
    run_report_dir.mkdir(parents=True, exist_ok=True)
    copied_sources: list[dict[str, str]] = []
    for source_path in [*noise_csvs, scaling_csv]:
        dest_path = run_report_dir / source_path.name
        if source_path.resolve() != dest_path.resolve():
            shutil.copy2(source_path, dest_path)
        copied_sources.append({"source": _rel(source_path), "copy": _rel(dest_path)})
    _write_json(
        run_report_dir / "promotion_source_manifest.json",
        {
            "run_tag": run_tag,
            "fig9_source_run_tag": fig9_source_run_tag,
            "fig10_source_run_tag": fig10_source_run_tag,
            "source_contract_tag": SOURCE_CONTRACT_TAG,
            "copied_sources": copied_sources,
        },
    )
    fig9_rows = build_fig9_envelope_rows(run_tag=run_tag, noise_summary_csvs=noise_csvs, fig6_csv=fig6_csv)
    fig10_rows = build_fig10_rows(
        run_tag=run_tag,
        scaling_summary_csv=scaling_csv,
        require_flow_buffer_measured=require_flow_buffer_measured,
        broad_scaling_claim_status=broad_scaling_claim_status,
    )
    fig10_fieldnames = [
        "run_tag",
        "model",
        "model_variant",
        "method",
        "batch_size",
        "sequence_length",
        "grid_role",
        "latency_ms",
        "latency_ms_std",
        "latency_ms_cv_pct",
        "throughput_images_s",
        "throughput_images_s_std",
        "throughput_images_s_cv_pct",
        "energy_j",
        "energy_j_std",
        "energy_j_cv_pct",
        "flow_buffer_peak_frac",
        "flow_buffer_peak_frac_std",
        "flow_buffer_peak_frac_cv_pct",
        "flow_buffer_peak_frac_status",
        "flow_buffer_measurement_truth_class",
        "flow_buffer_trace_path",
        "flow_buffer_trace_row_count",
        "repeat_count",
        "complete",
        "source_status",
        "source_run_tag",
        "source_csv",
        "evidence_basis",
        "scaling_claim_status",
        "claim_boundary",
        "notes",
    ]
    _write_csv(
        quick_dir / "fig9_noise_robustness_current_basis.csv",
        [
            "run_tag",
            "model",
            "model_variant",
            "lane",
            "profile",
            "profile_class",
            "sweep_resolution",
            "acc_top1",
            "acc_drop_pp",
            "acc_top1_mean",
            "acc_top1_std",
            "acc_drop_pp_mean",
            "acc_drop_pp_std",
            "ci95_pp",
            "latency_ms",
            "energy_j",
            "noise_sigma_lsb",
            "crosstalk_alpha",
            "seed_count",
            "complete",
            "source_status",
            "source_run_tag",
            "source_csv",
            "evidence_basis",
            "robustness_claim_status",
            "claim_boundary",
            "notes",
        ],
        fig9_rows,
    )
    _write_csv(
        quick_dir / "fig10_scaling_support_current_basis.csv",
        fig10_fieldnames,
        fig10_rows,
    )
    if broad_scaling_summary_csv is not None:
        _write_csv(
            quick_dir / "fig10_broad_scaling_flow_buffer_current_basis.csv",
            fig10_fieldnames,
            fig10_rows,
        )
    _write_json(
        quick_dir / "compliance_report.json",
        build_compliance_payload(
            run_tag=run_tag,
            quick_dir=quick_dir,
            report_data_dir=report_data_dir,
            fig9_source_run_tag=fig9_source_run_tag,
            broad_scaling_claim_status=broad_scaling_claim_status,
        ),
    )
    update_fig8_gate_matrix(
        run_tag=run_tag,
        quick_dir=quick_dir,
        broad_scaling_claim_status=broad_scaling_claim_status,
    )
    update_claim_contract(
        run_tag=run_tag,
        review_dir=review_dir,
        fig9_seed_count_summary=_fig9_seed_count_summary(fig9_rows),
        fig10_repeat_count_summary=_repeat_count_summary(fig10_rows),
        broad_scaling_claim_status=broad_scaling_claim_status,
    )
    update_manuscript_evidence_map(
        run_tag=run_tag,
        quick_dir=quick_dir,
        review_dir=review_dir,
        fig10_rows=fig10_rows,
        broad_scaling_claim_status=broad_scaling_claim_status,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_tag", default=RUN_TAG)
    parser.add_argument("--quick_dir", type=Path, default=QUICK_DIR)
    parser.add_argument("--report_data_dir", type=Path, default=REPORT_DATA_DIR)
    parser.add_argument("--review_dir", type=Path, default=REVIEW_DIR)
    parser.add_argument("--seed_quick_dir", type=Path, default=SEED_QUICK_DIR)
    parser.add_argument("--seed_review_dir", type=Path, default=SEED_REVIEW_DIR)
    parser.add_argument("--fig9_source_run_tag", default=SEED_RUN_TAG)
    parser.add_argument("--fig10_source_run_tag", default=FIG10_SCALING_CANDIDATE_TAG)
    parser.add_argument("--noise_summary_csv", type=Path, default=None)
    parser.add_argument("--scaling_summary_csv", type=Path, default=DEFAULT_SCALING_SUMMARY_CSV)
    parser.add_argument("--broad_scaling_summary_csv", type=Path, default=None)
    parser.add_argument("--require_flow_buffer_measured", action="store_true")
    parser.add_argument("--broad_scaling_claim_status", default=None)
    args = parser.parse_args()
    build_reports(
        run_tag=args.run_tag,
        quick_dir=args.quick_dir,
        report_data_dir=args.report_data_dir,
        review_dir=args.review_dir,
        seed_quick_dir=args.seed_quick_dir,
        seed_review_dir=args.seed_review_dir,
        fig9_source_run_tag=args.fig9_source_run_tag,
        fig10_source_run_tag=args.fig10_source_run_tag,
        noise_summary_csv=args.noise_summary_csv,
        scaling_summary_csv=args.scaling_summary_csv,
        broad_scaling_summary_csv=args.broad_scaling_summary_csv,
        require_flow_buffer_measured=args.require_flow_buffer_measured,
        broad_scaling_claim_status=args.broad_scaling_claim_status,
    )
    print(f"[fuller-final-unreserved] wrote final quick reports and governance artifacts under {args.quick_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
