#!/usr/bin/env python3
"""Materialize and execute the optimized fuller implementation collection lane."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS_ROOT = ROOT / "experiments"
if str(EXPERIMENTS_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_ROOT))

from accuracy.eval_cvnets_imagenet_noise import RESULT_FIELDNAMES
try:
    from .build_fuller_experiment_program import build_fuller_experiment_program
    from .build_fuller_phase4_intake_contract import build_fuller_phase4_intake_contract
    from .build_fuller_report_pack_contract import build_fuller_report_pack_contract
    from .materialize_fuller_experiment_execution_plan import materialize_fuller_experiment_execution_plan
except ImportError:
    from build_fuller_experiment_program import build_fuller_experiment_program  # type: ignore
    from build_fuller_phase4_intake_contract import build_fuller_phase4_intake_contract  # type: ignore
    from build_fuller_report_pack_contract import build_fuller_report_pack_contract  # type: ignore
    from materialize_fuller_experiment_execution_plan import materialize_fuller_experiment_execution_plan  # type: ignore


DEFAULT_BUNDLE = ROOT / "configs" / "fuller_implementation_execution_bundle_20260319.yaml"
DEFAULT_CONTRACT = ROOT / "configs" / "fuller_implementation_experiment_design_contract_20260319.yaml"
DEFAULT_TEMPLATE = ROOT / "configs" / "fuller_first_implementation_slice_template_20260319.yaml"
DEFAULT_GENERATED_ROOT = ROOT / "experiments" / "results" / "generated_configs"
DEFAULT_OUT_ROOT = ROOT / "experiments" / "results" / "report_data"
DEFAULT_DEVICE_TOOL = ROOT / "experiments" / "tools" / "measure_cvnets_device.py"
DEFAULT_PHASE1_RUNNER = ROOT / "experiments" / "tools" / "phase1_runner.py"
DEFAULT_NOISE_EVAL = ROOT / "experiments" / "accuracy" / "eval_cvnets_imagenet_noise.py"
DEFAULT_MLX_NOISE_EVAL = ROOT / "experiments" / "accuracy" / "eval_mlx_imagenet_noise.py"
DEFAULT_MANIFEST_NAME = "fuller_collection_manifest.json"
DEFAULT_SCRIPT_NAME = "run_fuller_collection.sh"
DEFAULT_PROGRAM_CONTRACT = ROOT / "configs" / "fuller_experiment_program_contract_20260422.yaml"

MANIFEST_JOB_FIELDS = [
    "family_group",
    "step_id",
    "command",
    "config_path",
    "run_id",
    "output_hint",
]
NOISE_SUMMARY_FIELDS = [
    "model",
    "profile",
    "sweep_resolution",
    "crosstalk_alpha",
    "gaussian_noise_std",
    "acc_top1",
    "acc_drop_pp",
    "latency_ms",
    "energy_j",
]
LATENCY_BREAKDOWN_FIELDS = [
    "baseline_variant",
    "stage",
    "latency_ms",
    "cycle_share",
    "module_group",
]
ENERGY_BREAKDOWN_FIELDS = [
    "baseline_variant",
    "component",
    "energy_j",
    "energy_share",
]
ABLATION_SUMMARY_FIELDS = [
    "mechanism_variant",
    "latency_ms",
    "energy_j",
    "avg_power_w",
    "acc_top1",
    "acc_drop_pp",
    "stage_cycles",
    "bubble_cycles",
    "utilization_avg",
    "accuracy_evidence",
    "accuracy_source_csv",
    "accuracy_context_run_id",
    "accuracy_target_notes",
]
SCALING_SUMMARY_FIELDS = [
    "baseline_variant",
    "model",
    "batch_size",
    "sequence_length",
    "grid_role",
    "latency_ms",
    "latency_ms_std",
    "latency_ms_cv_pct",
    "throughput_images_s",
    "throughput_images_s_std",
    "throughput_images_s_cv_pct",
    "throughput_tokens_s",
    "throughput_tokens_s_std",
    "throughput_tokens_s_cv_pct",
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
    "flow_admission_stalls_mean",
    "flow_admission_stalls_std",
    "flow_prefetch_hits_mean",
    "flow_prefetch_drops_mean",
    "flow_residency_hit_rate_mean",
    "flow_control_backpressure_mean",
    "flow_eviction_count_mean",
    "repeat_count",
]
NOISE_PAIRED_FIELDS = [
    "model",
    "profile",
    "sweep_resolution",
    "crosstalk_alpha",
    "gaussian_noise_std",
    "phase1_run_id",
    "latency_ms",
    "energy_j",
]

FLOW_BUFFER_TRACE_REQUIRED_FIELDS = [
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
FLOW_TRACE_TOLERANCE = 1e-6

DEFAULT_ARTIFACT_FILENAMES = {
    "cpu_real_device_metrics": "cpu_real_device_metrics.csv",
    "gpu_real_device_metrics": "gpu_real_device_metrics.csv",
    "astra_substrate_model_summary": "astra_substrate_model_summary.csv",
    "fuller_slice_model_summary": "fuller_slice_model_summary.csv",
    "ablation_summary": "ablation_summary.csv",
    "ablation_appendix_table": "ablation_appendix_table.csv",
    "stage_latency_breakdown": "stage_latency_breakdown.csv",
    "stage_energy_breakdown": "stage_energy_breakdown.csv",
    "scaling_summary": "scaling.csv",
    "noise_accuracy_summary_s_dense": "noise_s_dense.csv",
    "noise_accuracy_summary_s_sparse": "noise_s_sparse.csv",
    "noise_accuracy_summary_xs_dense": "noise_xs_dense.csv",
    "noise_accuracy_summary_xs_sparse": "noise_xs_sparse.csv",
    "noise_accuracy_summary_xxs_dense": "noise_xxs_dense.csv",
    "noise_accuracy_summary_xxs_sparse": "noise_xxs_sparse.csv",
}

NOISE_MODEL_SUFFIXES = {
    "mobilevit_s": "s",
    "mobilevit_xs": "xs",
    "mobilevit_xxs": "xxs",
}


def _noise_model_suffix(model: str) -> str:
    suffix = NOISE_MODEL_SUFFIXES.get(str(model).strip())
    if suffix is None:
        raise SystemExit(f"Unsupported noise model for governed artifact mapping: {model!r}")
    return suffix


def _noise_family_id(*, model: str, sweep_resolution: str) -> str:
    return f"NOISE_IMAGENET_MOBILEVIT_{_noise_model_suffix(model).upper()}_{str(sweep_resolution).upper()}"


def _noise_artifact_id(*, model: str, sweep_resolution: str) -> str:
    return f"noise_accuracy_summary_{_noise_model_suffix(model)}_{str(sweep_resolution).lower()}"


def _noise_group(*, model: str, sweep_resolution: str) -> str:
    return f"{_noise_model_suffix(model)}_{str(sweep_resolution).lower()}"


def _noise_paired_metrics_path(*, out_root: Path, run_tag: str) -> Path:
    return out_root / f"fuller_noise_paired_metrics_{run_tag}.csv"


def _device_repeat_root(*, generated_root: Path, run_tag: str) -> Path:
    return generated_root / run_tag / "device"


def _progress_root(*, generated_root: Path, run_tag: str) -> Path:
    return generated_root / run_tag / "progress"


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise SystemExit(f"Expected YAML mapping in {path}")
    return payload


def _resolve_repo_path(path_value: str | Path | None) -> Path | None:
    if path_value in ("", None):
        return None
    path = Path(str(path_value))
    if path.is_absolute():
        return path
    return ROOT / path


def _load_manifest_realism_profile(manifest: dict[str, Any]) -> dict[str, Any] | None:
    template_path = _resolve_repo_path(manifest.get("template"))
    if template_path is None or not template_path.exists():
        return None
    template = _load_yaml(template_path)
    realism_cfg = dict(template.get("realism") or {})
    profile_path = _resolve_repo_path(
        realism_cfg.get("calibration_profile_yaml") or realism_cfg.get("calibration_profile")
    )
    if profile_path is None or not profile_path.exists():
        return None
    return _load_yaml(profile_path)


def _apply_realism_profile_provenance(row: dict[str, str], profile: dict[str, Any] | None) -> dict[str, str]:
    if not profile:
        return row
    merged = dict(row)
    flow_cfg = dict(profile.get("flow") or {})
    meso_cfg = dict(profile.get("meso") or {})
    phy_cfg = dict(profile.get("phy") or {})
    integrated_cfg = dict(profile.get("integrated_system_costs") or {})
    if flow_cfg.get("calibration_source"):
        merged["flow_timeline_calibration_source"] = str(flow_cfg["calibration_source"])
    if meso_cfg.get("calibration_source"):
        merged["meso_cost_calibration_source"] = str(meso_cfg["calibration_source"])
    if phy_cfg.get("calibration_source"):
        merged["phy_support_calibration_source"] = str(phy_cfg["calibration_source"])
    if integrated_cfg.get("calibration_source"):
        merged["integrated_system_cost_calibration_source"] = str(integrated_cfg["calibration_source"])
    if integrated_cfg.get("uncertainty_method"):
        merged["integrated_system_cost_uncertainty_method"] = str(integrated_cfg["uncertainty_method"])
    return merged


def _bundle_path_value(bundle: dict[str, Any], *keys: str) -> str | None:
    current: Any = bundle
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    if current in ("", None):
        return None
    return str(current)


def _bundle_sequence_length(bundle: dict[str, Any], template: dict[str, Any], *, section: str) -> int:
    section_cfg = bundle.get(section) or {}
    if isinstance(section_cfg, dict):
        value = section_cfg.get("sequence_length")
        if value not in ("", None):
            return int(value)
    data_cfg = template.get("data") or {}
    value = data_cfg.get("sequence_length", data_cfg.get("l", 197))
    return int(value)


def _bundle_max_eval_samples(section_cfg: dict[str, Any]) -> int | None:
    value = section_cfg.get("max_eval_samples")
    if value in ("", None):
        return None
    return int(value)


def _bundle_int_list(value: Any, *, default: list[int]) -> list[int]:
    if value in ("", None):
        return list(default)
    if isinstance(value, int):
        return [value]
    if not isinstance(value, list):
        raise SystemExit(f"Expected integer list, got {value!r}")
    resolved: list[int] = []
    seen: set[int] = set()
    for item in value:
        integer = int(item)
        if integer in seen:
            continue
        seen.add(integer)
        resolved.append(integer)
    if not resolved:
        return list(default)
    return resolved


def _string_list(value: Any, *, default: list[str]) -> list[str]:
    if value in ("", None):
        return list(default)
    if isinstance(value, str):
        resolved = [item.strip() for item in value.split(",") if item.strip()]
    elif isinstance(value, list):
        resolved = [str(item).strip() for item in value if str(item).strip()]
    else:
        raise SystemExit(f"Expected string list, got {value!r}")
    return resolved or list(default)


def _bool_value(value: Any, *, default: bool = False) -> bool:
    if value in ("", None):
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


def _optional_float(value: Any) -> float | None:
    if value in ("", None):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _flow_enabled_for_job(job: dict[str, Any], *, variant_id: str) -> bool:
    if "flow_enabled" in job:
        return _bool_value(job.get("flow_enabled"), default=False)
    if "switches" in job and isinstance(job["switches"], dict):
        return _bool_value(job["switches"].get("flow"), default=False)
    return variant_id.upper() != "ASTRA"


def _flow_buffer_peak_frac_status(
    run_row: dict[str, str],
    *,
    variant_id: str,
    flow_enabled: bool | None = None,
    trace_valid: bool = False,
) -> str:
    if variant_id.upper() == "ASTRA" or flow_enabled is False:
        return "not_applicable_for_baseline"
    if trace_valid:
        return "measured"
    explicit = str(run_row.get("flow_buffer_peak_frac_status") or "").strip()
    if explicit and explicit != "measured":
        return explicit
    return "derived_from_trace"


def _trace_diagnostics(rows: list[dict[str, str]]) -> dict[str, float]:
    layer_diagnostics: dict[str, dict[str, float]] = {}
    for row in rows:
        layer_id = str(row.get("layer_id") or "")
        if not layer_id:
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
        for field in diagnostics:
            diagnostics[field] = max(diagnostics[field], _float(row.get(field)))
    if not layer_diagnostics:
        return {
            "flow_admission_stalls": 0.0,
            "flow_prefetch_hits": 0.0,
            "flow_prefetch_drops": 0.0,
            "flow_residency_hit_rate": 0.0,
            "flow_control_backpressure": 0.0,
            "flow_eviction_count": 0.0,
        }
    layer_rows = list(layer_diagnostics.values())
    return {
        "flow_admission_stalls": float(sum(row["admission_stalls"] for row in layer_rows)),
        "flow_prefetch_hits": float(sum(row["prefetch_hits"] for row in layer_rows)),
        "flow_prefetch_drops": float(sum(row["prefetch_drops"] for row in layer_rows)),
        "flow_residency_hit_rate": sum(row["residency_hit_rate"] for row in layer_rows) / len(layer_rows),
        "flow_control_backpressure": sum(row["control_backpressure"] for row in layer_rows) / len(layer_rows),
        "flow_eviction_count": float(sum(row["eviction_count"] for row in layer_rows)),
    }


def _phase1_metric_sources(run_dir: Path, run_row: dict[str, str]) -> list[tuple[str, dict[str, str]]]:
    sources: list[tuple[str, dict[str, str]]] = [("merged_phase1_outputs", run_row)]
    for filename in ("phase1_summary.csv", "master_metrics.csv"):
        path = run_dir / filename
        if path.is_file():
            sources.append((filename, _single_row(path)))
    return sources


def _metric_matches_any_source(
    *,
    sources: list[tuple[str, dict[str, str]]],
    field: str,
    expected: float,
    required: bool = True,
) -> bool:
    observed_any = False
    for _, source in sources:
        observed = _optional_float(source.get(field))
        if observed is None:
            continue
        observed_any = True
        if math.isclose(observed, expected, rel_tol=FLOW_TRACE_TOLERANCE, abs_tol=FLOW_TRACE_TOLERANCE):
            return True
    return not required and not observed_any


def _flow_buffer_trace_measurement(
    *,
    run_dir: Path,
    run_row: dict[str, str],
    variant_id: str,
    flow_enabled: bool,
) -> dict[str, Any]:
    if variant_id.upper() == "ASTRA" or not flow_enabled:
        trace_path = run_dir / "flow_buffer_trace.csv"
        trace_row_count = len(_read_csv(trace_path)) if trace_path.is_file() else 0
        return {
            "flow_buffer_peak_frac_status": "not_applicable_for_baseline",
            "flow_buffer_measurement_truth_class": "not_applicable_for_baseline",
            "flow_buffer_trace_path": str(trace_path) if trace_path.is_file() else "",
            "flow_buffer_trace_row_count": trace_row_count,
            "flow_admission_stalls": 0.0,
            "flow_prefetch_hits": 0.0,
            "flow_prefetch_drops": 0.0,
            "flow_residency_hit_rate": 0.0,
            "flow_control_backpressure": 0.0,
            "flow_eviction_count": 0.0,
        }

    trace_path = run_dir / "flow_buffer_trace.csv"
    fallback_status = _flow_buffer_peak_frac_status(run_row, variant_id=variant_id, flow_enabled=flow_enabled)
    fallback = {
        "flow_buffer_peak_frac_status": fallback_status,
        "flow_buffer_measurement_truth_class": "",
        "flow_buffer_trace_path": str(trace_path),
        "flow_buffer_trace_row_count": 0,
        "flow_admission_stalls": _float(run_row.get("flow_admission_stalls")),
        "flow_prefetch_hits": _float(run_row.get("flow_prefetch_hits")),
        "flow_prefetch_drops": _float(run_row.get("flow_prefetch_drops")),
        "flow_residency_hit_rate": _float(run_row.get("flow_residency_hit_rate")),
        "flow_control_backpressure": _float(run_row.get("flow_control_backpressure")),
        "flow_eviction_count": _float(run_row.get("flow_eviction_count")),
    }
    if not trace_path.is_file():
        return fallback

    rows = _read_csv(trace_path)
    if not rows:
        return fallback
    missing_fields = sorted(set(FLOW_BUFFER_TRACE_REQUIRED_FIELDS) - set(rows[0].keys()))
    if missing_fields:
        return {**fallback, "flow_buffer_trace_row_count": len(rows)}

    peak_frac = max(_float(row.get("occupancy_frac")) for row in rows)
    peak_cycles = max(_float(row.get("occupancy_cycles")) for row in rows)
    diagnostics = _trace_diagnostics(rows)
    sources = _phase1_metric_sources(run_dir, run_row)
    if not _metric_matches_any_source(sources=sources, field="flow_buffer_peak_frac", expected=peak_frac):
        return {**fallback, "flow_buffer_trace_row_count": len(rows)}
    if not _metric_matches_any_source(sources=sources, field="flow_buffer_peak_cycles", expected=peak_cycles):
        return {**fallback, "flow_buffer_trace_row_count": len(rows)}
    for field, expected in diagnostics.items():
        if not _metric_matches_any_source(sources=sources, field=field, expected=expected):
            return {**fallback, "flow_buffer_trace_row_count": len(rows)}

    return {
        "flow_buffer_peak_frac_status": "measured",
        "flow_buffer_measurement_truth_class": "instrumented_runtime_trace",
        "flow_buffer_trace_path": str(trace_path),
        "flow_buffer_trace_row_count": len(rows),
        **diagnostics,
    }


def _mean(values: list[float]) -> float:
    return sum(values) / float(len(values)) if values else 0.0


def _stddev(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mu = _mean(values)
    return math.sqrt(sum((value - mu) ** 2 for value in values) / float(len(values)))


def _cv_pct(values: list[float]) -> float:
    mean = _mean(values)
    if abs(mean) < 1e-12:
        return 0.0
    return _stddev(values) / abs(mean) * 100.0


def _load_execution_bundle(bundle_path: Path) -> dict[str, Any]:
    bundle = _load_yaml(bundle_path)
    if not bundle:
        raise SystemExit(f"Empty execution bundle: {bundle_path}")
    return bundle


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _json_dump(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _is_blank_number(value: Any) -> bool:
    text = str(value or "").strip().lower()
    return text in {"", "nan", "none", "null"}


def _normalize_bool(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes"}


def _strip_acc_seed(run_id: Any) -> str:
    text = str(run_id or "").strip()
    if not text:
        return ""
    if "_acc_s" not in text:
        return text
    prefix, _, suffix = text.rpartition("_acc_s")
    return prefix if suffix.isdigit() else text


_CLAIM_ACCURACY_CACHE: dict[tuple[str, str, str], dict[str, Any] | None] = {}
_CLAIM_FALLBACK_EXPERIMENT = {
    "E1": "E0",
    "E5": "E0",
}


def _claim_accuracy_summary(
    *,
    accuracy_source_csv: str | Path | None,
    experiment_id: str,
    model: str,
    accuracy_context_run_id: str | None = None,
) -> dict[str, Any] | None:
    source = _resolve_repo_path(accuracy_source_csv)
    if source is None or not source.is_file():
        return None
    claim_experiment = _CLAIM_FALLBACK_EXPERIMENT.get(str(experiment_id or "").strip().upper(), str(experiment_id or "").strip().upper())
    context_key = _strip_acc_seed(accuracy_context_run_id)
    cache_key = (str(source.resolve()), f"{claim_experiment}|{context_key}", str(model or "").strip())
    if cache_key in _CLAIM_ACCURACY_CACHE:
        return _CLAIM_ACCURACY_CACHE[cache_key]

    rows = _read_csv(source)

    def _summarize(filtered: list[dict[str, str]], *, claim_match_mode: str, claim_experiment_id: str) -> dict[str, Any] | None:
        baseline = [
            _float(row.get("top1"), default=math.nan)
            for row in filtered
            if _normalize_bool(row.get("baseline_flag") or row.get("baseline"))
        ]
        target = [
            _float(row.get("top1"), default=math.nan)
            for row in filtered
            if not _normalize_bool(row.get("baseline_flag") or row.get("baseline"))
        ]
        baseline_vals = [value for value in baseline if math.isfinite(value)]
        target_vals = [value for value in target if math.isfinite(value)]
        if not baseline_vals or not target_vals:
            return None
        return {
            "acc_ref_top1": sum(baseline_vals) / len(baseline_vals),
            "acc_top1": sum(target_vals) / len(target_vals),
            "acc_drop_pp": (sum(baseline_vals) / len(baseline_vals)) - (sum(target_vals) / len(target_vals)),
            "claim_experiment_id": claim_experiment_id,
            "claim_match_mode": claim_match_mode,
            "claim_context_run_id": context_key,
        }

    model_key = str(model or "").strip()
    summary: dict[str, Any] | None = None
    if context_key:
        filtered = [
            row
            for row in rows
            if str(row.get("model") or "").strip() == model_key
            and _strip_acc_seed(row.get("source_run_id") or row.get("run_id")) == context_key
        ]
        if filtered:
            matched_experiment = str(filtered[0].get("experiment_id") or "").strip().upper()
            summary = _summarize(
                filtered,
                claim_match_mode="context_run_id_exact",
                claim_experiment_id=matched_experiment,
            )

    if summary is None:
        filtered = [
            row
            for row in rows
            if str(row.get("experiment_id") or "").strip().upper() == claim_experiment
            and str(row.get("model") or "").strip() == model_key
        ]
        match_mode = "experiment_id" if claim_experiment == str(experiment_id or "").strip().upper() else "fallback_experiment"
        summary = _summarize(
            filtered,
            claim_match_mode=match_mode,
            claim_experiment_id=claim_experiment,
        )

    if summary is None:
        _CLAIM_ACCURACY_CACHE[cache_key] = None
        return None
    _CLAIM_ACCURACY_CACHE[cache_key] = summary
    return summary


def _fill_missing_claim_accuracy(row: dict[str, Any]) -> dict[str, Any]:
    needs_fill = any(_is_blank_number(row.get(field)) for field in ("acc_ref_top1", "acc_top1", "acc_drop_pp"))
    if not needs_fill:
        return row
    summary = _claim_accuracy_summary(
        accuracy_source_csv=row.get("accuracy_source_csv"),
        experiment_id=str(row.get("experiment_id") or ""),
        model=str(row.get("model") or ""),
        accuracy_context_run_id=str(row.get("accuracy_context_run_id") or ""),
    )
    if summary is None:
        return row

    filled = dict(row)
    for field in ("acc_ref_top1", "acc_top1", "acc_drop_pp"):
        if _is_blank_number(filled.get(field)):
            filled[field] = summary[field]
    if not str(filled.get("accuracy_context_run_id") or "").strip() and str(summary.get("claim_context_run_id") or "").strip():
        filled["accuracy_context_run_id"] = str(summary["claim_context_run_id"])
    if not str(filled.get("accuracy_evidence") or "").strip():
        mode = str(summary.get("claim_match_mode") or "")
        evidence = {
            "context_run_id_exact": "claim_accuracy_context_run_id",
            "experiment_id": "claim_accuracy_experiment_id",
            "fallback_experiment": "claim_accuracy_fallback_experiment",
        }.get(mode, "claim_accuracy_summary")
        filled["accuracy_evidence"] = evidence
    existing_notes = str(filled.get("accuracy_target_notes") or "").strip()
    note_parts = [part for part in existing_notes.split(";") if part]
    if summary["claim_experiment_id"] != str(row.get("experiment_id") or "").strip().upper():
        fallback_note = f"claim_accuracy_fallback:{row.get('experiment_id')}->{summary['claim_experiment_id']}"
        if fallback_note not in note_parts:
            note_parts.append(fallback_note)
    if str(summary.get("claim_match_mode") or "") == "context_run_id_exact" and str(summary.get("claim_context_run_id") or "").strip():
        context_note = f"claim_accuracy_context:{summary['claim_context_run_id']}"
        if context_note not in note_parts:
            note_parts.append(context_note)
    filled["accuracy_target_notes"] = ";".join(note_parts)
    return filled


def _load_weights_npz_manifest(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    exports = payload.get("exports")
    if isinstance(exports, list):
        mapping: dict[str, str] = {}
        for item in exports:
            if not isinstance(item, dict):
                continue
            model = str(item.get("model") or "").strip()
            weights_npz = str(item.get("weights_npz") or item.get("output_path") or "").strip()
            if model and weights_npz:
                mapping[model] = weights_npz
        return mapping
    model = str(payload.get("model") or "").strip()
    weights_npz = str(payload.get("weights_npz") or payload.get("output_path") or "").strip()
    if model and weights_npz:
        return {model: weights_npz}
    raise SystemExit(f"Unable to parse MLX weights manifest: {path}")


def _device_targets(device_cfg: dict[str, Any]) -> list[tuple[str, str]]:
    raw_devices = device_cfg.get("devices")
    if raw_devices in ("", None):
        requested = ["cpu", "mps"]
    elif isinstance(raw_devices, str):
        requested = [item.strip().lower() for item in raw_devices.split(",") if item.strip()]
    elif isinstance(raw_devices, list):
        requested = [str(item).strip().lower() for item in raw_devices if str(item).strip()]
    else:
        raise SystemExit("device_measurement.devices must be a list or comma-separated string")

    allowed = {
        "cpu": ("cpu", "cpu_real_device_metrics"),
        "mps": ("mps", "gpu_real_device_metrics"),
    }
    targets: list[tuple[str, str]] = []
    seen: set[str] = set()
    for device in requested:
        if device not in allowed:
            raise SystemExit(f"Unsupported device_measurement.devices entry: {device!r}")
        if device in seen:
            continue
        seen.add(device)
        targets.append(allowed[device])
    if not targets:
        raise SystemExit("device_measurement.devices must request at least one device")
    return targets


def _noise_coord_component(value: Any) -> str:
    if value in ("", None):
        return ""
    try:
        return f"{float(value):.12g}"
    except (TypeError, ValueError):
        return str(value).strip()


def _noise_coord_key(
    *,
    model: Any,
    sweep_resolution: Any,
    crosstalk_alpha: Any,
    gaussian_noise_std: Any,
) -> tuple[str, str, str, str]:
    return (
        str(model or "").strip(),
        str(sweep_resolution or "").strip(),
        _noise_coord_component(crosstalk_alpha),
        _noise_coord_component(gaussian_noise_std),
    )


def _artifact_paths(contract_path: Path, *, out_root: Path | None = None) -> dict[str, Path]:
    contract = _load_yaml(contract_path)
    artifact_csv = ROOT / str((contract.get("artifacts") or {}).get("data_contract_csv"))
    if not artifact_csv.is_file():
        if out_root is None:
            raise SystemExit(f"Missing data contract CSV: {artifact_csv}")
        return {artifact_id: out_root / filename for artifact_id, filename in DEFAULT_ARTIFACT_FILENAMES.items()}
    rows = _read_csv(artifact_csv)
    artifact_paths: dict[str, Path] = {}
    for row in rows:
        target = ROOT / str(row["future_output_path"])
        if out_root is not None:
            target = out_root / target.name
        artifact_paths[str(row["artifact_id"])] = target
    return artifact_paths


def _quote_command(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def _with_caffeinate(command: list[str], *, enabled: bool) -> list[str]:
    if enabled:
        return ["caffeinate", "-dimsu", *command]
    return command


def _claim_context_run_id(
    *,
    claim_core_program_tag: str | None,
    experiment_id: str,
) -> str | None:
    if claim_core_program_tag in ("", None):
        return None
    experiment = str(experiment_id or "").strip().upper()
    if experiment not in {"E0", "E1", "E2", "E3", "E4", "E5", "E6"}:
        return None
    return f"{claim_core_program_tag}_core_{experiment.lower()}"


def _noise_pass_count(
    *,
    crosstalk_alpha_values: list[Any],
    gaussian_noise_std_values: list[Any],
) -> int:
    return 1 + (
        len(list(crosstalk_alpha_values))
        * len(list(gaussian_noise_std_values))
    )


def _progress_jsonl_path(
    *,
    generated_root: Path,
    run_tag: str,
    step_id: str,
    seed: int,
) -> Path:
    return _progress_root(generated_root=generated_root, run_tag=run_tag) / f"{step_id}_s{seed}.jsonl"


def _ablation_variant_map(variants: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(item["variant_id"]): item for item in variants}


def _phase1_cfg(
    template: dict[str, Any],
    *,
    run_id: str,
    experiment_id: str,
    model: str,
    batch_size: int,
    sequence_length: int,
    switches: dict[str, bool],
    notes: str,
    accuracy_source_csv: str | None = None,
    accuracy_context_run_id: str | None = None,
) -> dict[str, Any]:
    cfg = copy.deepcopy(template)
    cfg.setdefault("run", {})
    cfg.setdefault("data", {})
    cfg.setdefault("models", {})
    cfg.setdefault("switches", {})
    cfg.setdefault("accuracy", {})
    cfg.setdefault("noise_injection", {})
    cfg.setdefault("outputs", {})
    cfg["run"]["run_id"] = run_id
    cfg["run"]["experiment_id"] = experiment_id
    cfg["run"]["notes"] = notes
    base_batch_size = int((template.get("data") or {}).get("batch_size") or 1)
    base_sequence_length = int(
        ((template.get("data") or {}).get("sequence_length") or (template.get("data") or {}).get("l") or sequence_length)
    )
    cfg["data"]["batch_size"] = int(batch_size)
    cfg["data"]["sequence_length"] = int(sequence_length)
    cfg["data"]["batch_size_ref"] = base_batch_size
    cfg["data"]["sequence_length_ref"] = base_sequence_length
    cfg["models"]["keys"] = [model]
    cfg["switches"] = dict(switches)
    if accuracy_source_csv is not None:
        cfg["accuracy"]["source_csv"] = accuracy_source_csv
        cfg["accuracy"]["context_run_id"] = accuracy_context_run_id
        cfg["accuracy"]["require_context_match"] = False
    cfg["noise_injection"]["enabled"] = False
    cfg["outputs"]["append_master"] = False

    for key in ("meso", "flow", "sparse", "phy"):
        section = cfg.get(key) or {}
        section["enabled"] = bool(switches[key])
        cfg[key] = section

    sc_det = cfg.get("sc_det") or {}
    early_stop = sc_det.get("early_stop") or {}
    early_stop["enabled"] = bool(switches["det"])
    sc_det["early_stop"] = early_stop
    cfg["sc_det"] = sc_det
    return cfg


def _phase1_run_dir(run_id: str) -> Path:
    return ROOT / "experiments" / "results" / "runs" / run_id


def _phase1_command(*, python_bin: str, config_path: Path, use_caffeinate: bool, device: str) -> list[str]:
    if str(device).strip().lower() != "mps":
        raise SystemExit("phase1 local runs in this repository must declare --device mps")
    command = [python_bin, str(DEFAULT_PHASE1_RUNNER), "--config", str(config_path), "--device", "mps"]
    return _with_caffeinate(command, enabled=use_caffeinate)


def _device_command(
    *,
    python_bin: str,
    imagenet_val: str,
    model: str,
    device: str,
    accuracy_backend: str,
    results_csv: Path,
    weights_dir: str | None,
    weights_npz: str | None,
    use_caffeinate: bool,
    batch_size: int,
    sequence_length: int,
    max_eval_samples: int | None,
    quant_bits: int,
    latency_policy: str,
    power_sampler: str,
    profiler_interval_ms: int,
    use_sudo_powermetrics: bool,
    interactive_sudo_powermetrics: bool,
    osascript_admin_powermetrics: bool,
) -> list[str]:
    command = [
        python_bin,
        str(DEFAULT_DEVICE_TOOL),
        "--imagenet_val",
        imagenet_val,
        "--model",
        model,
        "--device",
        device,
        "--accuracy_backend",
        accuracy_backend,
        "--results_csv",
        str(results_csv),
        "--batch_size",
        str(batch_size),
        "--sequence_length",
        str(sequence_length),
        "--quant_bits",
        str(quant_bits),
        "--latency_policy",
        latency_policy,
        "--power_sampler",
        power_sampler,
        "--profiler_interval_ms",
        str(profiler_interval_ms),
    ]
    if max_eval_samples is not None:
        command.extend(["--max_eval_samples", str(max_eval_samples)])
    if use_sudo_powermetrics:
        command.append("--use_sudo_powermetrics")
    if interactive_sudo_powermetrics:
        command.append("--interactive_sudo_powermetrics")
    if osascript_admin_powermetrics:
        command.append("--osascript_admin_powermetrics")
    if weights_npz:
        command.extend(["--weights_npz", weights_npz])
    elif weights_dir:
        command.extend(["--weights_dir", weights_dir])
    return _with_caffeinate(command, enabled=use_caffeinate and device == "mps")


def _noise_eval_command(
    *,
    accuracy_backend: str,
    python_bin: str,
    imagenet_val: str,
    model: str,
    profile: str,
    sweep_resolution: str,
    results_csv: Path,
    run_id: str,
    crosstalk_alpha: str,
    gaussian_noise_std: str,
    eval_batch_size: int,
    max_eval_samples: int | None,
    workers: int,
    quant_bits: int,
    weights_dir: str | None,
    weights_npz: str | None,
    use_caffeinate: bool,
    seed: int,
    append: bool,
    progress_jsonl: Path | None,
    progress_label: str | None,
) -> list[str]:
    eval_script = DEFAULT_NOISE_EVAL if accuracy_backend == "torch" else DEFAULT_MLX_NOISE_EVAL
    command = [
        python_bin,
        str(eval_script),
        "--imagenet_val",
        imagenet_val,
        "--opencv_pipeline",
        "--models",
        model,
        "--device",
        "mps",
        "--workers",
        str(workers),
        "--eval_batch_size",
        str(eval_batch_size),
        "--results_csv",
        str(results_csv),
        "--run_id",
        run_id,
        "--source_run_id",
        run_id,
        "--seed",
        str(seed),
        "--profile",
        profile,
        "--sweep_resolution",
        sweep_resolution,
        "--workload",
        "W0_mobilevit_imagenet",
        "--quant_bits",
        str(quant_bits),
    ]
    if accuracy_backend == "torch":
        command.extend(
            [
                "--workers",
                str(workers),
                "--gaussian_noise_std",
                gaussian_noise_std,
                "--crosstalk_alpha",
                crosstalk_alpha,
                "--enable_attention",
                "--allow_unvalidated_mps",
            ]
        )
    elif accuracy_backend == "mlx":
        command.extend(
            [
                "--noise_sigma_lsb",
                gaussian_noise_std,
                "--crosstalk_alpha",
                crosstalk_alpha,
            ]
        )
        if workers is not None:
            command.extend(["--workers", str(workers)])
    else:
        raise SystemExit(f"Unsupported noise accuracy backend: {accuracy_backend}")
    if append:
        command.append("--append")
    if max_eval_samples is not None:
        command.extend(["--max_eval_samples", str(max_eval_samples)])
    if progress_jsonl is not None:
        command.extend(["--progress_jsonl", str(progress_jsonl)])
    if progress_label:
        command.extend(["--progress_label", progress_label])
    if weights_npz:
        command.extend(["--weights_npz", weights_npz])
    elif weights_dir:
        command.extend(["--weights_dir", weights_dir])
    return _with_caffeinate(command, enabled=use_caffeinate)


def _single_row(path: Path) -> dict[str, str]:
    rows = _read_csv(path)
    if len(rows) != 1:
        raise SystemExit(f"Expected exactly one row in {path}, found {len(rows)}")
    return rows[0]


def _read_phase1_row(run_dir: Path) -> dict[str, str]:
    summary_path = run_dir / "phase1_summary.csv"
    master_path = run_dir / "master_metrics.csv"
    if summary_path.is_file() and master_path.is_file():
        merged = dict(_single_row(summary_path))
        merged.update(_single_row(master_path))
        return merged
    if master_path.is_file():
        return _single_row(master_path)
    if summary_path.is_file():
        return _single_row(summary_path)
    raise SystemExit(f"Missing phase1 outputs under {run_dir}")


def _accuracy_cfg_from_job(job: dict[str, Any]) -> dict[str, Any]:
    config_path = _resolve_repo_path(job.get("config_path"))
    if config_path is None or not config_path.is_file():
        return {}
    cfg = _load_yaml(config_path)
    return dict(cfg.get("accuracy") or {})


def summarize_device_benchmark(
    *,
    benchmark_csv: Path,
    system_json: Path,
    out_csv: Path,
    device_label: str,
    workload_id: str = "W0_mobilevit_imagenet",
    precision_mode: str = "int8_eval",
) -> None:
    rows = _read_csv(benchmark_csv)
    if not rows:
        raise SystemExit(f"No device rows found in {benchmark_csv}")
    payload = json.loads(system_json.read_text(encoding="utf-8"))
    chosen = None
    for row in rows:
        if str(row.get("device", "")).strip().lower() == device_label.lower():
            chosen = row
            break
    if chosen is None:
        raise SystemExit(f"Missing device={device_label} in {benchmark_csv}")
    out_row = {
        "workload_id": workload_id,
        "model": chosen.get("model", ""),
        "latency_ms": _float(chosen.get("latency_ms")),
        "avg_power_w": _float(chosen.get("power_w")),
        "energy_j": _float(chosen.get("energy_mj")) / 1000.0 if chosen.get("energy_mj") not in ("", None) else "",
        "batch_size": chosen.get("batch_size", ""),
        "sequence_length": payload.get("sequence_length", ""),
        "max_eval_samples": chosen.get("max_eval_samples", ""),
        "host_name": payload.get("host_name") or payload.get("os") or "unknown",
        "device_model": payload.get("gpu_name") if device_label == "gpu" else payload.get("cpu_model"),
        "accuracy_backend": chosen.get("accuracy_backend", "mlx" if device_label == "gpu" else "torch"),
        "framework": chosen.get("framework", "mlx" if device_label == "gpu" else "cvnets+pytorch"),
        "precision_mode": precision_mode,
        "power_sampler": chosen.get("power_sampler", ""),
        "profiler_interval_ms": chosen.get("profiler_interval_ms", ""),
    }
    _write_csv(out_csv, list(out_row.keys()), [out_row])


def summarize_noise_accuracy(
    *,
    input_csvs: list[Path],
    out_csv: Path,
    paired_metrics_csv: Path | None = None,
) -> None:
    metric_by_profile: dict[tuple[str, str, str], dict[str, str]] = {}
    metric_by_noise: dict[tuple[str, str, str, str], dict[str, str]] = {}
    if paired_metrics_csv is not None and paired_metrics_csv.is_file():
        for row in _read_csv(paired_metrics_csv):
            profile_key = (
                str(row.get("model", "")).strip(),
                str(row.get("profile", "")).strip(),
                str(row.get("sweep_resolution", "")).strip(),
            )
            metric_by_profile[profile_key] = row
            noise_key = _noise_coord_key(
                model=row.get("model"),
                sweep_resolution=row.get("sweep_resolution"),
                crosstalk_alpha=row.get("crosstalk_alpha"),
                gaussian_noise_std=row.get("gaussian_noise_std"),
            )
            if noise_key[2] and noise_key[3]:
                metric_by_noise[noise_key] = row

    grouped_rows: dict[tuple[str, str, str, str, str], list[dict[str, str]]] = {}
    row_order: list[tuple[str, str, str, str, str]] = []
    for input_csv in input_csvs:
        for row in _read_csv(input_csv):
            if str(row.get("baseline")).strip().lower() == "true":
                continue
            model = str(row.get("model") or "").strip()
            group_key = (
                model,
                str(row.get("profile", "")).strip(),
                str(row.get("sweep_resolution", "")).strip(),
                str(row.get("crosstalk_alpha", "")),
                str(row.get("gaussian_noise_std", "")),
            )
            if group_key not in grouped_rows:
                grouped_rows[group_key] = []
                row_order.append(group_key)
            grouped_rows[group_key].append(row)
    summary_rows: list[dict[str, Any]] = []
    for group_key in row_order:
        model, profile, sweep_resolution, crosstalk_alpha, gaussian_value = group_key
        rows = grouped_rows[group_key]
        top1_values = [_float(row.get("top1")) for row in rows]
        drop_values: list[float] = []
        for row in rows:
            top1_delta = row.get("top1_delta")
            drop_values.append(0.0 if top1_delta in ("", None) else max(0.0, -_float(top1_delta)))
        paired = metric_by_noise.get(
            _noise_coord_key(
                model=model,
                sweep_resolution=sweep_resolution,
                crosstalk_alpha=crosstalk_alpha,
                gaussian_noise_std=gaussian_value,
            ),
            {},
        )
        if not paired:
            paired = metric_by_profile.get((model, profile, sweep_resolution), {})
        summary_rows.append(
            {
                "model": model,
                "profile": profile,
                "sweep_resolution": sweep_resolution,
                "crosstalk_alpha": crosstalk_alpha,
                "gaussian_noise_std": gaussian_value,
                "acc_top1": _mean(top1_values),
                "acc_drop_pp": _mean(drop_values),
                "latency_ms": paired.get("latency_ms", ""),
                "energy_j": paired.get("energy_j", ""),
            }
        )
    _write_csv(out_csv, NOISE_SUMMARY_FIELDS, summary_rows)


def summarize_device_repeats(*, input_csvs: list[Path], out_csv: Path) -> None:
    if not input_csvs:
        raise SystemExit("summarize-device-repeats requires at least one input CSV")
    rows = [_single_row(path) for path in input_csvs]
    first = rows[0]
    for row in rows[1:]:
        for field in (
            "workload_id",
            "model",
            "batch_size",
            "sequence_length",
            "max_eval_samples",
            "accuracy_backend",
            "framework",
            "precision_mode",
            "latency_source",
            "latency_measurement_window",
            "power_measurement_window",
            "energy_derivation",
            "comparison_boundary",
            "comparison_kind",
            "benchmark_equivalence",
            "measurement_evidence_type",
        ):
            if str(row.get(field, "")) != str(first.get(field, "")):
                raise SystemExit(f"Device repeat mismatch for field {field}: {row.get(field)!r} != {first.get(field)!r}")
    latency_values = [_float(row.get("latency_ms")) for row in rows]
    power_values = [_float(row.get("avg_power_w")) for row in rows if row.get("avg_power_w") not in ("", None)]
    energy_values = [_float(row.get("energy_j")) for row in rows if row.get("energy_j") not in ("", None)]
    summary_row = dict(first)
    summary_row["latency_ms"] = _mean(latency_values)
    summary_row["avg_power_w"] = _mean(power_values) if power_values else ""
    summary_row["energy_j"] = _mean(energy_values) if energy_values else ""
    _write_csv(out_csv, list(summary_row.keys()), [summary_row])


def build_breakdown(
    *,
    pairs: list[str],
    out_latency_csv: Path,
    out_energy_csv: Path,
) -> None:
    component_map = {
        "conversion_control": "energy_breakdown_conversion_control_j",
        "memory_move": "energy_breakdown_memory_move_j",
        "oe": "energy_breakdown_oe_j",
        "adc_pca": "energy_breakdown_adc_pca_j",
        "laser_optical": "energy_breakdown_laser_optical_j",
        "other_static": "energy_breakdown_other_static_j",
        "hidden_system_cost": "integrated_hidden_system_cost_j",
    }
    latency_rows: list[dict[str, Any]] = []
    energy_rows: list[dict[str, Any]] = []
    for item in pairs:
        if "=" not in item:
            raise SystemExit(f"Expected VARIANT=/path/to/master.csv, got {item}")
        variant, raw_path = item.split("=", 1)
        source_path = Path(raw_path)
        row = _single_row(source_path)
        stage_cycles = json.loads(str(row.get("stage_cycles") or "{}"))
        total_cycles = sum(_float(value) for value in stage_cycles.values()) or 1.0
        core_latency = _float(row.get("core_latency_ms")) or _float(row.get("latency_ms"))
        total_latency = _float(row.get("latency_ms"))
        hidden_system_latency = _float(row.get("integrated_hidden_system_latency_ms"))
        total_energy = _float(row.get("energy_j")) or 1.0
        for stage, cycles in stage_cycles.items():
            cycle_value = _float(cycles)
            latency_rows.append(
                {
                    "baseline_variant": variant,
                    "stage": stage,
                    "latency_ms": core_latency * (cycle_value / total_cycles),
                    "cycle_share": (core_latency * (cycle_value / total_cycles)) / total_latency if total_latency > 0 else 0.0,
                    "module_group": stage,
                }
            )
        if hidden_system_latency > 0.0:
            latency_rows.append(
                {
                    "baseline_variant": variant,
                    "stage": "hidden_system_latency",
                    "latency_ms": hidden_system_latency,
                    "cycle_share": hidden_system_latency / total_latency if total_latency > 0 else 0.0,
                    "module_group": "hidden_system_latency",
                }
            )
        for component, field in component_map.items():
            energy = _float(row.get(field))
            energy_rows.append(
                {
                    "baseline_variant": variant,
                    "component": component,
                    "energy_j": energy,
                    "energy_share": energy / total_energy if total_energy > 0.0 else 0.0,
                }
            )
    _write_csv(out_latency_csv, LATENCY_BREAKDOWN_FIELDS, latency_rows)
    _write_csv(out_energy_csv, ENERGY_BREAKDOWN_FIELDS, energy_rows)


def _build_stage_latency_rows(variant_to_run: dict[str, Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for variant in ("ASTRA", "FULLER"):
        summary_row = _read_phase1_row(variant_to_run[variant])
        stage_cycles = json.loads(str(summary_row.get("stage_cycles") or "{}"))
        total_cycles = sum(_float(value) for value in stage_cycles.values()) or 1.0
        core_latency_ms = _float(summary_row.get("core_latency_ms")) or _float(summary_row.get("latency_ms"))
        total_latency_ms = _float(summary_row.get("latency_ms"))
        for stage, cycles in stage_cycles.items():
            cycle_value = _float(cycles)
            rows.append(
                {
                    "baseline_variant": variant,
                    "stage": stage,
                    "latency_ms": core_latency_ms * (cycle_value / total_cycles),
                    "cycle_share": (core_latency_ms * (cycle_value / total_cycles)) / total_latency_ms if total_latency_ms > 0 else 0.0,
                    "module_group": stage,
                }
            )
        hidden_system_latency_ms = _float(summary_row.get("integrated_hidden_system_latency_ms"))
        if hidden_system_latency_ms > 0.0:
            rows.append(
                {
                    "baseline_variant": variant,
                    "stage": "hidden_system_latency",
                    "latency_ms": hidden_system_latency_ms,
                    "cycle_share": hidden_system_latency_ms / total_latency_ms if total_latency_ms > 0 else 0.0,
                    "module_group": "hidden_system_latency",
                }
            )
    return rows


def _build_stage_energy_rows(variant_to_run: dict[str, Path]) -> list[dict[str, Any]]:
    component_map = {
        "conversion_control": "energy_breakdown_conversion_control_j",
        "memory_move": "energy_breakdown_memory_move_j",
        "oe": "energy_breakdown_oe_j",
        "adc_pca": "energy_breakdown_adc_pca_j",
        "laser_optical": "energy_breakdown_laser_optical_j",
        "other_static": "energy_breakdown_other_static_j",
        "hidden_system_cost": "integrated_hidden_system_cost_j",
    }
    rows: list[dict[str, Any]] = []
    for variant in ("ASTRA", "FULLER"):
        summary_row = _read_phase1_row(variant_to_run[variant])
        total_energy = _float(summary_row.get("energy_j")) or 1.0
        for component, field in component_map.items():
            energy = _float(summary_row.get(field))
            rows.append(
                {
                    "baseline_variant": variant,
                    "component": component,
                    "energy_j": energy,
                    "energy_share": energy / total_energy if total_energy > 0.0 else 0.0,
                }
            )
    return rows


def _build_ablation_rows(
    variant_to_run: dict[str, Path],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summary_rows: list[dict[str, Any]] = []
    appendix_rows: list[dict[str, Any]] = []
    ordered = ("ASTRA", "HOPS", "FLOW_MESO", "FLOW_PHY", "FULLER")
    astra_row = _read_phase1_row(variant_to_run["ASTRA"])
    astra_latency = _float(astra_row.get("latency_ms")) or 1.0
    astra_energy = _float(astra_row.get("energy_j")) or 1.0
    astra_drop = _float(astra_row.get("acc_drop_pp"))
    for variant in ordered:
        run_row = _read_phase1_row(variant_to_run[variant])
        row = {
            "mechanism_variant": variant,
            "latency_ms": _float(run_row.get("latency_ms")),
            "energy_j": _float(run_row.get("energy_j")),
            "avg_power_w": _float(run_row.get("avg_power_w")),
            "acc_top1": _float(run_row.get("acc_top1")),
            "acc_drop_pp": _float(run_row.get("acc_drop_pp")),
            "stage_cycles": run_row.get("stage_cycles", ""),
            "bubble_cycles": _float(run_row.get("bubble_cycles")),
            "utilization_avg": _float(run_row.get("utilization_avg")),
        }
        summary_rows.append(row)
        appendix_rows.append(
            {
                **row,
                "delta_latency_pct": 100.0 * ((row["latency_ms"] - astra_latency) / astra_latency),
                "delta_energy_pct": 100.0 * ((row["energy_j"] - astra_energy) / astra_energy),
                "delta_acc_drop_pp": row["acc_drop_pp"] - astra_drop,
            }
        )
    return summary_rows, appendix_rows


def _build_scaling_rows(scaling_run_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in scaling_run_rows:
        run_dir = Path(str(item["run_dir"]))
        run_row = _read_phase1_row(run_dir)
        variant_id = str(item["variant"])
        flow_enabled = _flow_enabled_for_job(item, variant_id=variant_id)
        trace = _flow_buffer_trace_measurement(
            run_dir=run_dir,
            run_row=run_row,
            variant_id=variant_id,
            flow_enabled=flow_enabled,
        )
        rows.append(
            {
                "baseline_variant": variant_id,
                "model": str(item.get("model") or ""),
                "batch_size": int(item["batch_size"]),
                "sequence_length": int(item["sequence_length"]),
                "grid_role": str(item.get("grid_role") or "declared_grid"),
                "latency_ms": _float(run_row.get("latency_ms")),
                "throughput_images_s": _float(run_row.get("throughput_images_s")),
                "throughput_tokens_s": _float(run_row.get("throughput_tokens_s")),
                "energy_j": _float(run_row.get("energy_j")),
                "flow_buffer_peak_frac": _float(run_row.get("flow_buffer_peak_frac")),
                "flow_buffer_peak_frac_status": trace["flow_buffer_peak_frac_status"],
                "flow_buffer_measurement_truth_class": trace["flow_buffer_measurement_truth_class"],
                "flow_buffer_trace_path": trace["flow_buffer_trace_path"],
                "flow_buffer_trace_row_count": trace["flow_buffer_trace_row_count"],
                "flow_admission_stalls": trace["flow_admission_stalls"],
                "flow_prefetch_hits": trace["flow_prefetch_hits"],
                "flow_prefetch_drops": trace["flow_prefetch_drops"],
                "flow_residency_hit_rate": trace["flow_residency_hit_rate"],
                "flow_control_backpressure": trace["flow_control_backpressure"],
                "flow_eviction_count": trace["flow_eviction_count"],
                "repeat_count": int(item.get("repeat_count") or 1),
            }
        )
    return rows


def _aggregate_scaling_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, int, int, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            str(row["baseline_variant"]),
            str(row.get("model") or ""),
            int(row["batch_size"]),
            int(row["sequence_length"]),
            str(row.get("grid_role") or "declared_grid"),
        )
        grouped.setdefault(key, []).append(row)
    aggregated: list[dict[str, Any]] = []
    for key, group in grouped.items():
        baseline_variant, model, batch_size, sequence_length, grid_role = key
        statuses = {str(row.get("flow_buffer_peak_frac_status") or "") for row in group}
        status = next(iter(statuses)) if len(statuses) == 1 else "mixed"
        truth_classes = {str(row.get("flow_buffer_measurement_truth_class") or "") for row in group}
        truth_class = next(iter(truth_classes)) if len(truth_classes) == 1 else "mixed"
        trace_paths = [
            str(row.get("flow_buffer_trace_path") or "")
            for row in group
            if str(row.get("flow_buffer_trace_path") or "").strip()
        ]
        repeat_count = max(len(group), max(int(row.get("repeat_count") or 1) for row in group))
        latency_values = [_float(row.get("latency_ms")) for row in group]
        throughput_image_values = [_float(row.get("throughput_images_s")) for row in group]
        throughput_token_values = [_float(row.get("throughput_tokens_s")) for row in group]
        energy_values = [_float(row.get("energy_j")) for row in group]
        flow_values = [_float(row.get("flow_buffer_peak_frac")) for row in group]
        admission_values = [_float(row.get("flow_admission_stalls")) for row in group]
        prefetch_hit_values = [_float(row.get("flow_prefetch_hits")) for row in group]
        prefetch_drop_values = [_float(row.get("flow_prefetch_drops")) for row in group]
        residency_values = [_float(row.get("flow_residency_hit_rate")) for row in group]
        backpressure_values = [_float(row.get("flow_control_backpressure")) for row in group]
        eviction_values = [_float(row.get("flow_eviction_count")) for row in group]
        aggregated.append(
            {
                "baseline_variant": baseline_variant,
                "model": model,
                "batch_size": batch_size,
                "sequence_length": sequence_length,
                "grid_role": grid_role,
                "latency_ms": _mean(latency_values),
                "latency_ms_std": _stddev(latency_values),
                "latency_ms_cv_pct": _cv_pct(latency_values),
                "throughput_images_s": _mean(throughput_image_values),
                "throughput_images_s_std": _stddev(throughput_image_values),
                "throughput_images_s_cv_pct": _cv_pct(throughput_image_values),
                "throughput_tokens_s": _mean(throughput_token_values),
                "throughput_tokens_s_std": _stddev(throughput_token_values),
                "throughput_tokens_s_cv_pct": _cv_pct(throughput_token_values),
                "energy_j": _mean(energy_values),
                "energy_j_std": _stddev(energy_values),
                "energy_j_cv_pct": _cv_pct(energy_values),
                "flow_buffer_peak_frac": _mean(flow_values),
                "flow_buffer_peak_frac_std": _stddev(flow_values),
                "flow_buffer_peak_frac_cv_pct": _cv_pct(flow_values),
                "flow_buffer_peak_frac_status": status,
                "flow_buffer_measurement_truth_class": truth_class,
                "flow_buffer_trace_path": ";".join(trace_paths),
                "flow_buffer_trace_row_count": int(sum(_float(row.get("flow_buffer_trace_row_count")) for row in group)),
                "flow_admission_stalls_mean": _mean(admission_values),
                "flow_admission_stalls_std": _stddev(admission_values),
                "flow_prefetch_hits_mean": _mean(prefetch_hit_values),
                "flow_prefetch_drops_mean": _mean(prefetch_drop_values),
                "flow_residency_hit_rate_mean": _mean(residency_values),
                "flow_control_backpressure_mean": _mean(backpressure_values),
                "flow_eviction_count_mean": _mean(eviction_values),
                "repeat_count": repeat_count,
            }
        )
    aggregated.sort(
        key=lambda row: (
            str(row["model"]),
            str(row["baseline_variant"]),
            str(row.get("grid_role") or ""),
            int(row["batch_size"]),
            int(row["sequence_length"]),
        )
    )
    return aggregated


def _collect_phase1_outputs(
    *,
    manifest_json: Path,
    contract: Path,
    out_root: Path | None,
    families: set[str] | None = None,
    require_flow_buffer_measured: bool = False,
) -> None:
    manifest = json.loads(manifest_json.read_text(encoding="utf-8"))
    artifact_paths = _artifact_paths(contract, out_root=out_root)
    jobs = manifest["jobs"]
    run_tag = str(manifest["run_tag"])
    resolved_out_root = out_root or Path(str(manifest["out_root"]))
    realism_profile = _load_manifest_realism_profile(manifest)

    ablation_rows: list[dict[str, Any]] = []
    ablation_by_variant: dict[str, dict[str, str]] = {}
    scaling_rows: list[dict[str, Any]] = []
    noise_paired_rows: list[dict[str, Any]] = []

    for job in jobs:
        family = str(job.get("family_group") or "")
        if families and "all" not in families and family not in families:
            continue
        run_id = str(job.get("run_id") or "")
        if family not in {"model", "scaling", "noise_model"}:
            continue
        run_row = _read_phase1_row(_phase1_run_dir(run_id))
        accuracy_cfg = _accuracy_cfg_from_job(job)
        if accuracy_cfg:
            if not str(run_row.get("accuracy_source_csv") or "").strip() and str(accuracy_cfg.get("source_csv") or "").strip():
                run_row["accuracy_source_csv"] = str(accuracy_cfg["source_csv"])
            if str(accuracy_cfg.get("context_run_id") or "").strip():
                run_row["accuracy_context_run_id"] = str(accuracy_cfg["context_run_id"])
        run_row = _fill_missing_claim_accuracy(run_row)
        if family == "model":
            variant = str(job.get("variant_id") or "").upper()
            ablation_by_variant[variant] = run_row
            ablation_rows.append(
                {
                    "mechanism_variant": variant,
                    "latency_ms": _float(run_row.get("latency_ms")),
                    "energy_j": _float(run_row.get("energy_j")),
                    "avg_power_w": _float(run_row.get("avg_power_w")),
                    "acc_top1": _float(run_row.get("acc_top1")),
                    "acc_drop_pp": _float(run_row.get("acc_drop_pp")),
                    "stage_cycles": run_row.get("stage_cycles", ""),
                    "bubble_cycles": _float(run_row.get("bubble_cycles")),
                    "utilization_avg": _float(run_row.get("utilization_avg")),
                    "accuracy_evidence": str(run_row.get("accuracy_evidence") or ""),
                    "accuracy_source_csv": str(run_row.get("accuracy_source_csv") or ""),
                    "accuracy_context_run_id": str(run_row.get("accuracy_context_run_id") or ""),
                    "accuracy_target_notes": str(run_row.get("accuracy_target_notes") or ""),
                }
            )
        elif family == "scaling":
            variant_id = str(job.get("variant_id") or "").upper()
            run_dir = _phase1_run_dir(run_id)
            flow_enabled = _flow_enabled_for_job(job, variant_id=variant_id)
            trace = _flow_buffer_trace_measurement(
                run_dir=run_dir,
                run_row=run_row,
                variant_id=variant_id,
                flow_enabled=flow_enabled,
            )
            if require_flow_buffer_measured and flow_enabled and trace["flow_buffer_peak_frac_status"] != "measured":
                raise SystemExit(f"{run_id} lacks validated measured flow-buffer trace evidence")
            scaling_rows.append(
                {
                    "baseline_variant": variant_id,
                    "model": str(job.get("model") or run_row.get("model") or ""),
                    "batch_size": int(job["batch_size"]),
                    "sequence_length": int(job["sequence_length"]),
                    "grid_role": str(job.get("grid_role") or "declared_grid"),
                    "latency_ms": _float(run_row.get("latency_ms")),
                    "throughput_images_s": _float(run_row.get("throughput_images_s")),
                    "throughput_tokens_s": _float(run_row.get("throughput_tokens_s")),
                    "energy_j": _float(run_row.get("energy_j")),
                    "flow_buffer_peak_frac": _float(run_row.get("flow_buffer_peak_frac")),
                    "flow_buffer_peak_frac_status": trace["flow_buffer_peak_frac_status"],
                    "flow_buffer_measurement_truth_class": trace["flow_buffer_measurement_truth_class"],
                    "flow_buffer_trace_path": trace["flow_buffer_trace_path"],
                    "flow_buffer_trace_row_count": trace["flow_buffer_trace_row_count"],
                    "flow_admission_stalls": trace["flow_admission_stalls"],
                    "flow_prefetch_hits": trace["flow_prefetch_hits"],
                    "flow_prefetch_drops": trace["flow_prefetch_drops"],
                    "flow_residency_hit_rate": trace["flow_residency_hit_rate"],
                    "flow_control_backpressure": trace["flow_control_backpressure"],
                    "flow_eviction_count": trace["flow_eviction_count"],
                    "repeat_count": int(job.get("repeat_count") or 1),
                }
            )
        elif family == "noise_model":
            noise_paired_rows.append(
                {
                    "model": str(job.get("model") or run_row.get("model") or ""),
                    "profile": str(job.get("profile") or ""),
                    "sweep_resolution": str(job.get("sweep_resolution") or ""),
                    "crosstalk_alpha": job.get("crosstalk_alpha", ""),
                    "gaussian_noise_std": job.get("gaussian_noise_std", ""),
                    "phase1_run_id": run_id,
                    "latency_ms": _float(run_row.get("latency_ms")),
                    "energy_j": _float(run_row.get("energy_j")),
                }
            )

    has_ablation_pair = "ASTRA" in ablation_by_variant and "FULLER" in ablation_by_variant
    if ablation_rows and not has_ablation_pair:
        raise SystemExit("collect-phase1 requires both ASTRA and FULLER run outputs when ablation rows are present.")

    if has_ablation_pair:
        ordered_ablation = []
        for variant in ("ASTRA", "HOPS", "FLOW_MESO", "FLOW_PHY", "FULLER"):
            row = next((item for item in ablation_rows if item["mechanism_variant"] == variant), None)
            if row is not None:
                ordered_ablation.append(row)
        astra_summary_row = _apply_realism_profile_provenance(ablation_by_variant["ASTRA"], realism_profile)
        fuller_summary_row = _apply_realism_profile_provenance(ablation_by_variant["FULLER"], realism_profile)
        _write_csv(
            artifact_paths["astra_substrate_model_summary"],
            list(astra_summary_row.keys()),
            [astra_summary_row],
        )
        _write_csv(
            artifact_paths["fuller_slice_model_summary"],
            list(fuller_summary_row.keys()),
            [fuller_summary_row],
        )
        _write_csv(artifact_paths["ablation_summary"], ABLATION_SUMMARY_FIELDS, ordered_ablation)
    if scaling_rows:
        _write_csv(artifact_paths["scaling_summary"], SCALING_SUMMARY_FIELDS, _aggregate_scaling_rows(scaling_rows))
    if noise_paired_rows:
        deduped_rows: list[dict[str, Any]] = []
        seen_profiles: set[tuple[str, str, str, str, str]] = set()
        for row in noise_paired_rows:
            profile_key = (
                str(row["model"]),
                str(row["profile"]),
                str(row["sweep_resolution"]),
                _noise_coord_component(row.get("crosstalk_alpha")),
                _noise_coord_component(row.get("gaussian_noise_std")),
            )
            if profile_key in seen_profiles:
                continue
            seen_profiles.add(profile_key)
            deduped_rows.append(row)
        _write_csv(
            _noise_paired_metrics_path(out_root=resolved_out_root, run_tag=run_tag),
            NOISE_PAIRED_FIELDS,
            deduped_rows,
        )


def _build_manifest(
    *,
    bundle: dict[str, Any],
    contract: Path,
    template_path: Path,
    generated_root: Path,
    out_root: Path,
    run_tag: str,
    imagenet_val: str,
    weights_dir: str | None,
    weights_npz_manifest: Path | None,
    python_bin: str,
    device: str,
    workers: int,
    use_caffeinate: bool,
    claim_accuracy_csv: Path | None,
    claim_core_program_tag: str | None,
) -> dict[str, Any]:
    if str(device).strip().lower() != "mps":
        raise SystemExit("local fuller implementation collection requires --device mps")
    template = _load_yaml(template_path)
    artifact_paths = _artifact_paths(contract, out_root=out_root)
    run_root = generated_root / run_tag
    ablation_cfg_root = run_root / "ablation"
    scaling_cfg_root = run_root / "scaling"
    noise_root = run_root / "noise"
    progress_root = _progress_root(generated_root=generated_root, run_tag=run_tag)
    noise_root.mkdir(parents=True, exist_ok=True)

    jobs: list[dict[str, Any]] = []
    device_cfg = dict(bundle.get("device_measurement") or {})
    device_model = str(device_cfg.get("model") or "mobilevit_s")
    device_batch_size = int(device_cfg.get("batch_size") or 1)
    device_sequence_length = _bundle_sequence_length(bundle, template, section="device_measurement")
    device_max_eval_samples = _bundle_max_eval_samples(device_cfg)
    device_quant_bits = int(device_cfg.get("quant_bits") or 8)
    device_mps_accuracy_backend = str(device_cfg.get("mps_accuracy_backend") or "mlx").strip().lower()
    if device_mps_accuracy_backend not in {"torch", "mlx"}:
        raise SystemExit(
            f"device_measurement.mps_accuracy_backend must be torch/mlx, got {device_mps_accuracy_backend!r}"
        )
    device_latency_policy = str(device_cfg.get("latency_policy") or "whole_subprocess").strip().lower()
    if device_latency_policy not in {"whole_subprocess", "quantized_eval_pass"}:
        raise SystemExit(
            "device_measurement.latency_policy must be whole_subprocess/quantized_eval_pass, "
            f"got {device_latency_policy!r}"
        )
    device_power_sampler = str(device_cfg.get("power_sampler") or "powermetrics")
    device_profiler_interval_ms = int(device_cfg.get("profiler_interval_ms") or 200)
    device_use_sudo_powermetrics = bool(device_cfg.get("use_sudo_powermetrics", True))
    device_interactive_sudo_powermetrics = bool(device_cfg.get("interactive_sudo_powermetrics", False))
    device_osascript_admin_powermetrics = bool(device_cfg.get("osascript_admin_powermetrics", False))
    if device_osascript_admin_powermetrics and (
        device_use_sudo_powermetrics or device_interactive_sudo_powermetrics
    ):
        raise SystemExit(
            "device_measurement.osascript_admin_powermetrics cannot be combined with "
            "use_sudo_powermetrics/interactive_sudo_powermetrics"
        )
    device_repeats = max(1, int(device_cfg.get("repeats") or 1))
    device_root = _device_repeat_root(generated_root=generated_root, run_tag=run_tag)
    weights_npz_by_model = _load_weights_npz_manifest(weights_npz_manifest)
    device_targets = _device_targets(device_cfg)
    claim_accuracy_source = _resolve_repo_path(claim_accuracy_csv)

    for device, artifact_id in device_targets:
        run_id = f"{run_tag}_device_{device}"
        device_commands: list[list[str]] = []
        repeat_outputs: list[str] = []
        for repeat_idx in range(device_repeats):
            repeat_csv = device_root / f"{device}_repeat{repeat_idx + 1}.csv"
            repeat_outputs.append(str(repeat_csv))
            resolved_weights_npz = None
            if device == "mps" and device_mps_accuracy_backend == "mlx":
                resolved_weights_npz = weights_npz_by_model.get(device_model)
            device_commands.append(
                _device_command(
                    python_bin=python_bin,
                    imagenet_val=imagenet_val,
                    model=device_model,
                    device=device,
                    accuracy_backend=("torch" if device == "cpu" else device_mps_accuracy_backend),
                    results_csv=repeat_csv,
                    weights_dir=weights_dir,
                    weights_npz=resolved_weights_npz,
                    use_caffeinate=use_caffeinate,
                    batch_size=device_batch_size,
                    sequence_length=device_sequence_length,
                    max_eval_samples=device_max_eval_samples,
                    quant_bits=device_quant_bits,
                    latency_policy=device_latency_policy,
                    power_sampler=device_power_sampler,
                    profiler_interval_ms=device_profiler_interval_ms,
                    use_sudo_powermetrics=device_use_sudo_powermetrics,
                    interactive_sudo_powermetrics=device_interactive_sudo_powermetrics,
                    osascript_admin_powermetrics=device_osascript_admin_powermetrics,
                )
            )
        jobs.append(
            {
                "family_group": "device",
                "step_id": f"device_{device}",
                "command": _quote_command(device_commands[0]),
                "commands": device_commands,
                "config_path": "",
                "run_id": run_id,
                "output_hint": str(artifact_paths[artifact_id]),
                "raw_results_csvs": repeat_outputs,
                "artifact_id": artifact_id,
            }
        )

    ablation_cfg = dict(bundle.get("ablation") or {})
    ablation_model = str(ablation_cfg.get("model") or "mobilevit_s")
    ablation_variants = list(ablation_cfg.get("variants") or [])
    if not ablation_variants:
        raise SystemExit("Execution bundle is missing ablation.variants")
    variant_lookup = _ablation_variant_map(ablation_variants)
    if "FULLER" not in variant_lookup:
        raise SystemExit("Execution bundle ablation.variants must contain FULLER")
    fuller_variant = variant_lookup["FULLER"]

    noise_cfg = dict(bundle.get("noise") or {})
    noise_accuracy_backend = str(noise_cfg.get("accuracy_backend") or "torch").strip().lower()
    if noise_accuracy_backend not in {"torch", "mlx"}:
        raise SystemExit(f"noise.accuracy_backend must be torch/mlx, got {noise_accuracy_backend!r}")
    dense_cfg = dict(noise_cfg.get("dense") or {})
    dense_model = str(dense_cfg.get("model") or "mobilevit_s")
    dense_support = list(noise_cfg.get("dense_support") or [])
    sparse_support = list(noise_cfg.get("sparse_support") or [])
    accuracy_source_by_model: dict[str, Path] = {}
    accuracy_context_run_id_by_model: dict[str, str] = {}
    accuracy_source_by_noise_anchor: dict[tuple[str, str], Path] = {}
    accuracy_context_run_id_by_noise_anchor: dict[tuple[str, str], str] = {}
    seen_noise_variants: set[tuple[str, str]] = set()
    for dense_variant in [{"model": dense_model, **dense_cfg}] + dense_support:
        model = str(dense_variant.get("model") or "")
        if not model:
            raise SystemExit("noise.dense and noise.dense_support entries require model")
        if any(existing_model == model for existing_model, _ in seen_noise_variants):
            raise SystemExit(f"Noise model {model} cannot be configured with mixed sweep resolutions")
        variant_key = (model, "dense")
        if variant_key in seen_noise_variants:
            raise SystemExit(f"Duplicate dense noise variant configured for {model}")
        seen_noise_variants.add(variant_key)
        accuracy_source_by_model[model] = noise_root / f"{model}_dense.csv"
    for support_item in sparse_support:
        model = str(support_item.get("model") or "")
        if not model:
            raise SystemExit("noise.sparse_support entries require model")
        if any(existing_model == model for existing_model, _ in seen_noise_variants):
            raise SystemExit(f"Noise model {model} cannot be configured with mixed sweep resolutions")
        variant_key = (model, "sparse")
        if variant_key in seen_noise_variants:
            raise SystemExit(f"Duplicate sparse noise variant configured for {model}")
        seen_noise_variants.add(variant_key)
        profiles = list(support_item.get("profiles") or [])
        clean_profile = next(
            (profile for profile in profiles if str(profile.get("profile") or "").strip().lower() == "clean"),
            profiles[0] if profiles else None,
        )
        for profile in profiles:
            profile_name = str(profile.get("profile") or "")
            if model and profile_name:
                accuracy_source_by_noise_anchor[(model, profile_name)] = (
                    noise_root / f"{model}_{profile_name}.csv"
                )
        if model and clean_profile is not None:
            accuracy_source_by_model[model] = noise_root / f"{model}_{clean_profile['profile']}.csv"

    noise_eval_batch_size = int(noise_cfg.get("eval_batch_size") or 32)
    noise_max_eval_samples = _bundle_max_eval_samples(noise_cfg)
    noise_quant_bits = int(noise_cfg.get("quant_bits", device_quant_bits) or device_quant_bits)
    noise_workers = int(noise_cfg.get("workers", workers if workers is not None else 0) or 0)
    seed_cfg = dict(noise_cfg.get("seed_policy") or {})
    default_noise_seeds = _bundle_int_list(seed_cfg.get("default_seeds"), default=[0])
    sparse_support_seeds = _bundle_int_list(seed_cfg.get("sparse_support_seeds"), default=default_noise_seeds)
    dense_representative_profiles = list(seed_cfg.get("dense_representative_profiles") or [])
    noise_phase1_profiles: list[dict[str, Any]] = []
    dense_variants = [{"model": dense_model, **dense_cfg}] + dense_support
    for dense_variant in dense_variants:
        model = str(dense_variant.get("model") or "")
        dense_alpha_values = list(dense_variant.get("crosstalk_alpha") or [])
        dense_gaussian_values = list(dense_variant.get("gaussian_noise_std") or [])
        if not dense_alpha_values or not dense_gaussian_values:
            raise SystemExit(
                f"Dense noise variant for {model} must declare crosstalk_alpha and gaussian_noise_std grids"
            )
        dense_pass_count = _noise_pass_count(
            crosstalk_alpha_values=dense_alpha_values,
            gaussian_noise_std_values=dense_gaussian_values,
        )
        dense_csv = accuracy_source_by_model[model]
        dense_run_id = f"{run_tag}_noise_{model}_dense"
        dense_step_id = f"noise_{model}_dense"
        accuracy_context_run_id_by_model[model] = f"{dense_run_id}_s{default_noise_seeds[0]}"
        dense_progress_jsonls = [
            _progress_jsonl_path(
                generated_root=generated_root,
                run_tag=run_tag,
                step_id=dense_step_id,
                seed=seed,
            )
            for seed in default_noise_seeds
        ]
        dense_commands = [
            _noise_eval_command(
                accuracy_backend=noise_accuracy_backend,
                python_bin=python_bin,
                imagenet_val=imagenet_val,
                model=model,
                profile="dense_primary",
                sweep_resolution="dense",
                results_csv=dense_csv,
                run_id=f"{dense_run_id}_s{seed}",
                crosstalk_alpha=",".join(str(value) for value in dense_alpha_values),
                gaussian_noise_std=",".join(str(value) for value in dense_gaussian_values),
                eval_batch_size=noise_eval_batch_size,
                max_eval_samples=noise_max_eval_samples,
                workers=noise_workers,
                quant_bits=noise_quant_bits,
                weights_dir=weights_dir,
                weights_npz=weights_npz_by_model.get(model) if noise_accuracy_backend == "mlx" else None,
                use_caffeinate=use_caffeinate,
                seed=seed,
                append=index > 0,
                progress_jsonl=dense_progress_jsonls[index],
                progress_label=f"{model}:dense_primary:seed{seed}",
            )
            for index, seed in enumerate(default_noise_seeds)
        ]
        jobs.append(
            {
                "family_group": "noise",
                "step_id": dense_step_id,
                "command": _quote_command(dense_commands[0]),
                "commands": dense_commands,
                "config_path": "",
                "run_id": dense_run_id,
                "output_hint": str(dense_csv),
                "noise_group": _noise_group(model=model, sweep_resolution="dense"),
                "noise_artifact_id": _noise_artifact_id(model=model, sweep_resolution="dense"),
                "noise_family_id": _noise_family_id(model=model, sweep_resolution="dense"),
                "model": model,
                "profile": "dense_primary",
                "sweep_resolution": "dense",
                "progress_jsonls": [str(path) for path in dense_progress_jsonls],
                "planned_pass_count_per_command": [dense_pass_count for _ in dense_progress_jsonls],
                "planned_pass_count": dense_pass_count * len(dense_progress_jsonls),
            }
        )
        for item in dense_representative_profiles:
            if not any(float(alpha) == float(item["crosstalk_alpha"]) for alpha in dense_alpha_values):
                continue
            if not any(float(gaussian) == float(item["gaussian_noise_std"]) for gaussian in dense_gaussian_values):
                continue
            extra_seeds = [
                seed
                for seed in _bundle_int_list(item.get("extra_seeds"), default=[])
                if seed not in set(default_noise_seeds)
            ]
            if not extra_seeds:
                continue
            profile_name = str(item.get("profile") or "rep")
            rep_run_id = f"{run_tag}_noise_{model}_{profile_name}_rep"
            rep_step_id = f"noise_{model}_{profile_name}_rep"
            rep_pass_count = _noise_pass_count(
                crosstalk_alpha_values=[item["crosstalk_alpha"]],
                gaussian_noise_std_values=[item["gaussian_noise_std"]],
            )
            rep_progress_jsonls = [
                _progress_jsonl_path(
                    generated_root=generated_root,
                    run_tag=run_tag,
                    step_id=rep_step_id,
                    seed=seed,
                )
                for seed in extra_seeds
            ]
            rep_commands = [
                _noise_eval_command(
                    accuracy_backend=noise_accuracy_backend,
                    python_bin=python_bin,
                    imagenet_val=imagenet_val,
                    model=model,
                    profile=profile_name,
                    sweep_resolution="dense",
                    results_csv=dense_csv,
                    run_id=f"{rep_run_id}_s{seed}",
                    crosstalk_alpha=str(item["crosstalk_alpha"]),
                    gaussian_noise_std=str(item["gaussian_noise_std"]),
                    eval_batch_size=noise_eval_batch_size,
                    max_eval_samples=noise_max_eval_samples,
                    workers=noise_workers,
                    quant_bits=noise_quant_bits,
                    weights_dir=weights_dir,
                    weights_npz=weights_npz_by_model.get(model) if noise_accuracy_backend == "mlx" else None,
                    use_caffeinate=use_caffeinate,
                    seed=seed,
                    append=True,
                    progress_jsonl=rep_progress_jsonls[index],
                    progress_label=f"{model}:{profile_name}:seed{seed}",
                )
                for index, seed in enumerate(extra_seeds)
            ]
            jobs.append(
                {
                    "family_group": "noise",
                    "step_id": rep_step_id,
                    "command": _quote_command(rep_commands[0]),
                    "commands": rep_commands,
                    "config_path": "",
                    "run_id": rep_run_id,
                    "output_hint": str(dense_csv),
                    "noise_group": _noise_group(model=model, sweep_resolution="dense"),
                    "noise_artifact_id": _noise_artifact_id(model=model, sweep_resolution="dense"),
                    "noise_family_id": _noise_family_id(model=model, sweep_resolution="dense"),
                    "model": model,
                    "profile": profile_name,
                    "sweep_resolution": "dense",
                    "progress_jsonls": [str(path) for path in rep_progress_jsonls],
                    "planned_pass_count_per_command": [rep_pass_count for _ in rep_progress_jsonls],
                    "planned_pass_count": rep_pass_count * len(rep_progress_jsonls),
                }
            )
        for alpha in dense_alpha_values:
            for gaussian in dense_gaussian_values:
                noise_phase1_profiles.append(
                    {
                        "model": model,
                        "profile": (
                            f"dense_a{str(alpha).replace('.', 'p').replace('-', 'm')}"
                            f"_g{str(gaussian).replace('.', 'p').replace('-', 'm')}"
                        ),
                        "sweep_resolution": "dense",
                        "crosstalk_alpha": alpha,
                        "gaussian_noise_std": gaussian,
                    }
                )
    for support_item in sparse_support:
        model = str(support_item.get("model") or "")
        for profile in list(support_item.get("profiles") or []):
            noise_phase1_profiles.append(
                {
                    "model": model,
                    "profile": str(profile["profile"]),
                    "sweep_resolution": "sparse",
                    "crosstalk_alpha": profile["crosstalk_alpha"],
                    "gaussian_noise_std": profile["gaussian_noise_std"],
                }
            )

    for support_item in sparse_support:
        model = str(support_item.get("model") or "")
        for profile in list(support_item.get("profiles") or []):
            profile_name = str(profile["profile"])
            accuracy_context_run_id_by_noise_anchor[(model, profile_name)] = (
                f"{run_tag}_noise_{model}_{profile_name}_s{sparse_support_seeds[0]}"
            )
            if profile_name.strip().lower() == "clean":
                accuracy_context_run_id_by_model[model] = accuracy_context_run_id_by_noise_anchor[
                    (model, profile_name)
                ]
            slug = f"{model}_{profile['profile']}"
            out_csv = noise_root / f"{slug}.csv"
            run_id = f"{run_tag}_noise_{slug}"
            step_id = f"noise_{slug}"
            profile_pass_count = _noise_pass_count(
                crosstalk_alpha_values=[profile["crosstalk_alpha"]],
                gaussian_noise_std_values=[profile["gaussian_noise_std"]],
            )
            progress_jsonls = [
                _progress_jsonl_path(
                    generated_root=generated_root,
                    run_tag=run_tag,
                    step_id=step_id,
                    seed=seed,
                )
                for seed in sparse_support_seeds
            ]
            commands = [
                _noise_eval_command(
                    accuracy_backend=noise_accuracy_backend,
                    python_bin=python_bin,
                    imagenet_val=imagenet_val,
                    model=model,
                    profile=str(profile["profile"]),
                    sweep_resolution="sparse",
                    results_csv=out_csv,
                    run_id=f"{run_id}_s{seed}",
                    crosstalk_alpha=str(profile["crosstalk_alpha"]),
                    gaussian_noise_std=str(profile["gaussian_noise_std"]),
                    eval_batch_size=noise_eval_batch_size,
                    max_eval_samples=noise_max_eval_samples,
                    workers=noise_workers,
                    quant_bits=noise_quant_bits,
                    weights_dir=weights_dir,
                    weights_npz=weights_npz_by_model.get(model) if noise_accuracy_backend == "mlx" else None,
                    use_caffeinate=use_caffeinate,
                    seed=seed,
                    append=index > 0,
                    progress_jsonl=progress_jsonls[index],
                    progress_label=f"{model}:{profile['profile']}:seed{seed}",
                )
                for index, seed in enumerate(sparse_support_seeds)
            ]
            jobs.append(
                {
                    "family_group": "noise",
                    "step_id": step_id,
                    "command": _quote_command(commands[0]),
                    "commands": commands,
                    "config_path": "",
                    "run_id": run_id,
                    "output_hint": str(out_csv),
                    "noise_group": _noise_group(model=model, sweep_resolution="sparse"),
                    "noise_artifact_id": _noise_artifact_id(model=model, sweep_resolution="sparse"),
                    "noise_family_id": _noise_family_id(model=model, sweep_resolution="sparse"),
                    "model": model,
                    "profile": str(profile["profile"]),
                    "sweep_resolution": "sparse",
                    "progress_jsonls": [str(path) for path in progress_jsonls],
                    "planned_pass_count_per_command": [profile_pass_count for _ in progress_jsonls],
                    "planned_pass_count": profile_pass_count * len(progress_jsonls),
                }
            )

    for variant in ablation_variants:
        variant_id = str(variant["variant_id"])
        run_id = f"{run_tag}_{variant_id.lower()}"
        variant_experiment_id = str(variant["experiment_id"])
        claim_context_run_id = _claim_context_run_id(
            claim_core_program_tag=claim_core_program_tag,
            experiment_id=variant_experiment_id,
        )
        cfg = _phase1_cfg(
            template,
            run_id=run_id,
            experiment_id=variant_experiment_id,
            model=ablation_model,
            batch_size=1,
            sequence_length=_bundle_sequence_length(bundle, template, section="ablation"),
            switches={key: bool(value) for key, value in variant["switches"].items()},
            notes=f"fuller_ablation_variant:{variant_id}",
            accuracy_source_csv=(
                str(claim_accuracy_source)
                if claim_accuracy_source is not None and claim_context_run_id is not None
                else str(accuracy_source_by_model[ablation_model])
            ),
            accuracy_context_run_id=claim_context_run_id or accuracy_context_run_id_by_model[ablation_model],
        )
        cfg_path = ablation_cfg_root / f"{variant_id.lower()}.yaml"
        _write_yaml(cfg_path, cfg)
        command = _phase1_command(
            python_bin=python_bin,
            config_path=cfg_path,
            use_caffeinate=use_caffeinate,
            device=device,
        )
        jobs.append(
            {
                "family_group": "model",
                "step_id": f"ablation_{variant_id.lower()}",
                "command": _quote_command(command),
                "commands": [command],
                "config_path": str(cfg_path),
                "run_id": run_id,
                "output_hint": str(_phase1_run_dir(run_id)),
                "variant_id": variant_id,
                "flow_enabled": bool(variant["switches"].get("flow")),
                "device": "mps",
            }
        )

    noise_anchor_root = run_root / "noise_model"
    for item in noise_phase1_profiles:
        model = str(item["model"])
        profile = str(item["profile"])
        sweep_resolution = str(item["sweep_resolution"])
        run_id = f"{run_tag}_noise_model_{model}_{profile}"
        cfg = _phase1_cfg(
            template,
            run_id=run_id,
            experiment_id=f"FULLER_NOISE_MODEL_{model.upper()}_{profile.upper()}",
            model=model,
            batch_size=1,
            sequence_length=_bundle_sequence_length(bundle, template, section="noise"),
            switches={key: bool(value) for key, value in fuller_variant["switches"].items()},
            notes=f"fuller_noise_model_anchor:{model}:{profile}",
            accuracy_source_csv=str(
                accuracy_source_by_noise_anchor.get((model, profile), accuracy_source_by_model[model])
            ),
            accuracy_context_run_id=accuracy_context_run_id_by_noise_anchor.get(
                (model, profile),
                accuracy_context_run_id_by_model[model],
            ),
        )
        cfg["noise_injection"]["enabled"] = True
        cfg["noise_injection"]["crosstalk_alpha"] = float(item["crosstalk_alpha"])
        cfg["noise_injection"]["gaussian_noise_std"] = float(item["gaussian_noise_std"])
        cfg["p1_align"]["gaussian_noise_std_ref"] = float(item["gaussian_noise_std"])
        cfg["p1_align"]["crosstalk_alpha_ref"] = float(item["crosstalk_alpha"])
        cfg_path = noise_anchor_root / f"{model}_{profile}.yaml"
        _write_yaml(cfg_path, cfg)
        command = _phase1_command(
            python_bin=python_bin,
            config_path=cfg_path,
            use_caffeinate=use_caffeinate,
            device=device,
        )
        jobs.append(
            {
                "family_group": "noise_model",
                "step_id": f"noise_model_{model}",
                "command": _quote_command(command),
                "commands": [command],
                "config_path": str(cfg_path),
                "run_id": run_id,
                "output_hint": str(_phase1_run_dir(run_id)),
                "model": model,
                "profile": profile,
                "sweep_resolution": sweep_resolution,
                "crosstalk_alpha": float(item["crosstalk_alpha"]),
                "gaussian_noise_std": float(item["gaussian_noise_std"]),
                "flow_enabled": True,
                "device": "mps",
            }
        )

    scaling_cfg = dict(bundle.get("scaling") or {})
    scaling_model = str(scaling_cfg.get("model") or ablation_model)
    scaling_models = _string_list(scaling_cfg.get("models"), default=[scaling_model])
    scaling_variants = _string_list(scaling_cfg.get("variants"), default=["ASTRA", "FULLER"])
    scaling_points = list(scaling_cfg.get("points") or [])
    scaling_repeats = int(scaling_cfg.get("repeats") or 1)
    if scaling_repeats < 1:
        raise SystemExit("scaling.repeats must be >= 1")
    if not scaling_points:
        raise SystemExit("Execution bundle is missing scaling.points")
    multi_model_scaling = len(scaling_models) > 1
    for variant_id in scaling_variants:
        variant_id = str(variant_id).upper()
        if variant_id not in variant_lookup:
            raise SystemExit(f"Execution bundle ablation.variants is missing {variant_id} required for scaling")
        variant = variant_lookup[variant_id]
        for model in scaling_models:
            if model not in accuracy_source_by_model:
                raise SystemExit(f"Scaling model {model!r} lacks a governed accuracy source in noise.dense/dense_support")
            for point in scaling_points:
                batch_size = int(point["batch_size"])
                sequence_length = int(point["sequence_length"])
                point_slug = str(point.get("scale_id") or f"b{batch_size}_s{sequence_length}")
                model_slug = str(model).replace("-", "_")
                claim_context_run_id = _claim_context_run_id(
                    claim_core_program_tag=claim_core_program_tag,
                    experiment_id=str(variant["experiment_id"]),
                )
                base_run_id = (
                    f"{run_tag}_scale_{model_slug}_{variant_id.lower()}_{point_slug}"
                    if multi_model_scaling
                    else f"{run_tag}_scale_{variant_id.lower()}_{point_slug}"
                )
                base_step_id = (
                    f"scaling_{model_slug}_{variant_id.lower()}_{point_slug}"
                    if multi_model_scaling
                    else f"scaling_{variant_id.lower()}_{point_slug}"
                )
                for repeat_index in range(1, scaling_repeats + 1):
                    repeat_suffix = f"_r{repeat_index}" if scaling_repeats > 1 else ""
                    run_id = f"{base_run_id}{repeat_suffix}"
                    cfg = _phase1_cfg(
                        template,
                        run_id=run_id,
                        experiment_id=(
                            f"{variant['experiment_id']}_{model_slug.upper()}_{point_slug.upper()}{repeat_suffix.upper()}"
                            if multi_model_scaling
                            else f"{variant['experiment_id']}_{point_slug.upper()}{repeat_suffix.upper()}"
                        ),
                        model=model,
                        batch_size=batch_size,
                        sequence_length=sequence_length,
                        switches={key: bool(value) for key, value in variant["switches"].items()},
                        notes=f"fuller_scaling:{model}:{variant_id}:{point_slug}:repeat{repeat_index}",
                        accuracy_source_csv=(
                            str(claim_accuracy_source)
                            if claim_accuracy_source is not None and claim_context_run_id is not None
                            else str(accuracy_source_by_model[model])
                        ),
                        accuracy_context_run_id=claim_context_run_id or accuracy_context_run_id_by_model[model],
                    )
                    cfg_path = (
                        scaling_cfg_root / f"{model_slug}_{variant_id.lower()}_{point_slug.lower()}{repeat_suffix}.yaml"
                        if multi_model_scaling
                        else scaling_cfg_root / f"{variant_id.lower()}_{point_slug.lower()}{repeat_suffix}.yaml"
                    )
                    _write_yaml(cfg_path, cfg)
                    command = _phase1_command(
                        python_bin=python_bin,
                        config_path=cfg_path,
                        use_caffeinate=use_caffeinate,
                        device=device,
                    )
                    jobs.append(
                        {
                            "family_group": "scaling",
                            "step_id": f"{base_step_id}{repeat_suffix}",
                            "command": _quote_command(command),
                            "commands": [command],
                            "config_path": str(cfg_path),
                            "run_id": run_id,
                            "output_hint": str(_phase1_run_dir(run_id)),
                            "variant_id": variant_id,
                            "model": model,
                            "batch_size": batch_size,
                            "sequence_length": sequence_length,
                            "grid_role": str(point.get("grid_role") or "declared_grid"),
                            "repeat_count": scaling_repeats,
                            "repeat_index": repeat_index,
                            "flow_enabled": bool(variant["switches"].get("flow")),
                            "device": "mps",
                        }
                    )

    manifest = {
        "run_tag": run_tag,
        "contract": str(contract),
        "template": str(template_path),
        "generated_root": str(run_root),
        "progress_root": str(progress_root),
        "out_root": str(out_root),
        "artifact_paths": {key: str(value) for key, value in artifact_paths.items()},
        "claim_accuracy_csv": str(claim_accuracy_source) if claim_accuracy_source is not None else "",
        "claim_core_program_tag": claim_core_program_tag or "",
        "jobs": jobs,
    }
    return manifest


def _build_postprocess_commands(manifest_json: Path, manifest: dict[str, Any]) -> list[str]:
    artifact_paths = {key: Path(value) for key, value in manifest["artifact_paths"].items()}
    jobs = manifest["jobs"]
    run_tag = str(manifest["run_tag"])
    out_root = Path(str(manifest["out_root"]))
    noise_paired_metrics_csv = _noise_paired_metrics_path(out_root=out_root, run_tag=run_tag)
    astra_master = _phase1_run_dir(f"{run_tag}_astra") / "master_metrics.csv"
    fuller_master = _phase1_run_dir(f"{run_tag}_fuller") / "master_metrics.csv"
    noise_inputs_by_artifact: dict[str, list[Path]] = {}
    has_phase1_jobs = False
    has_model_pair = False
    seen_model_variants: set[str] = set()
    for job in jobs:
        family_group = str(job.get("family_group") or "")
        if family_group == "noise":
            artifact_id = str(job.get("noise_artifact_id") or "")
            if artifact_id:
                artifact_inputs = noise_inputs_by_artifact.setdefault(artifact_id, [])
                candidate = Path(job["output_hint"])
                if candidate not in artifact_inputs:
                    artifact_inputs.append(candidate)
        if family_group in {"model", "scaling", "noise_model"}:
            has_phase1_jobs = True
        if family_group == "model":
            seen_model_variants.add(str(job.get("variant_id") or "").upper())
    has_model_pair = {"ASTRA", "FULLER"}.issubset(seen_model_variants)

    commands = []
    for job in jobs:
        if str(job.get("family_group") or "") != "device":
            continue
        raw_results_csvs = [str(path) for path in list(job.get("raw_results_csvs") or [])]
        if not raw_results_csvs:
            continue
        commands.append(
            _quote_command(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "summarize-device-repeats",
                    "--input_csvs",
                    *raw_results_csvs,
                    "--out_csv",
                    str(artifact_paths[str(job["artifact_id"])]),
                ]
            )
        )

    if has_phase1_jobs:
        commands.append(
            _quote_command(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "collect-phase1",
                    "--manifest_json",
                    str(manifest_json),
                    "--contract",
                    str(manifest["contract"]),
                    "--out_root",
                    str(manifest["out_root"]),
                ]
            )
        )
    for artifact_id, out_csv in artifact_paths.items():
        if not artifact_id.startswith("noise_accuracy_summary_"):
            continue
        input_csvs = noise_inputs_by_artifact.get(artifact_id, [])
        if not input_csvs:
            continue
        commands.append(
            _quote_command(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "summarize-noise",
                    "--out_csv",
                    str(out_csv),
                    "--paired_metrics_csv",
                    str(noise_paired_metrics_csv),
                    "--input_csvs",
                    *[str(path) for path in input_csvs],
                ]
            )
        )
    if has_model_pair:
        commands.extend([
            _quote_command(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "build-breakdown",
                    "--pairs",
                    f"ASTRA={astra_master}",
                    "--pairs",
                    f"FULLER={fuller_master}",
                    "--out_latency_csv",
                    str(artifact_paths["stage_latency_breakdown"]),
                    "--out_energy_csv",
                    str(artifact_paths["stage_energy_breakdown"]),
                ]
            ),
            _quote_command(
                [
                    sys.executable,
                    str(ROOT / "experiments" / "tools" / "build_fuller_ablation_appendix_table.py"),
                    "--input_csv",
                    str(artifact_paths["ablation_summary"]),
                    "--out_csv",
                    str(artifact_paths["ablation_appendix_table"]),
                ]
            ),
        ])
    return commands


def _write_shell_script(script_path: Path, manifest: dict[str, Any], postprocess_commands: list[str]) -> None:
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        "# Collection jobs",
    ]
    for job in manifest["jobs"]:
        lines.append(f"# {job['family_group']} :: {job['step_id']}")
        for command in job["commands"]:
            lines.append(_quote_command(command))
        lines.append("")
    lines.append("# Postprocess jobs")
    for command in postprocess_commands:
        lines.append(command)
    script_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    script_path.chmod(0o755)


def _run_job_commands(manifest_json: Path, families: set[str]) -> None:
    manifest = json.loads(manifest_json.read_text(encoding="utf-8"))
    for job in manifest["jobs"]:
        family = str(job.get("family_group") or "")
        if "all" not in families and family not in families:
            continue
        for command in job["commands"]:
            subprocess.run(command, cwd=str(ROOT), check=True)


def _cmd_legacy_plan(args: argparse.Namespace) -> None:
    bundle_path = args.bundle or DEFAULT_BUNDLE
    bundle = _load_execution_bundle(bundle_path)
    contract_path = args.contract or ROOT / str(_bundle_path_value(bundle, "inputs", "design_contract_yaml") or DEFAULT_CONTRACT)
    template_path = args.template or ROOT / str(_bundle_path_value(bundle, "inputs", "phase1_template_yaml") or DEFAULT_TEMPLATE)
    generated_root = args.generated_root or DEFAULT_GENERATED_ROOT
    out_root = args.out_root or DEFAULT_OUT_ROOT
    manifest = _build_manifest(
        bundle=bundle,
        contract=contract_path,
        template_path=template_path,
        generated_root=generated_root,
        out_root=out_root,
        run_tag=args.run_tag,
        imagenet_val=args.imagenet_val,
        weights_dir=args.weights_dir,
        weights_npz_manifest=args.weights_npz_manifest,
        python_bin=args.python,
        device=args.device,
        workers=args.workers,
        use_caffeinate=args.use_caffeinate,
        claim_accuracy_csv=args.claim_accuracy_csv,
        claim_core_program_tag=args.claim_core_program_tag,
    )
    run_root = generated_root / args.run_tag
    manifest_json = run_root / DEFAULT_MANIFEST_NAME
    script_path = run_root / DEFAULT_SCRIPT_NAME
    _json_dump(manifest_json, manifest)
    postprocess_commands = _build_postprocess_commands(manifest_json, manifest)
    _write_shell_script(script_path, manifest, postprocess_commands)
    print(f"[fuller-plan] wrote {manifest_json}")
    print(f"[fuller-plan] wrote {script_path}")


def _cmd_plan(args: argparse.Namespace) -> None:
    program_contract = args.program_contract or DEFAULT_PROGRAM_CONTRACT
    generated_root = args.generated_root or DEFAULT_GENERATED_ROOT
    run_root = generated_root / args.run_tag
    manifest_json = run_root / DEFAULT_MANIFEST_NAME
    script_path = run_root / DEFAULT_SCRIPT_NAME
    build_fuller_experiment_program(program_contract, root_dir=ROOT)
    execution_payload = materialize_fuller_experiment_execution_plan(program_contract, root_dir=ROOT)
    phase4_payload = build_fuller_phase4_intake_contract(program_contract, root_dir=ROOT)
    report_payload = build_fuller_report_pack_contract(program_contract, root_dir=ROOT)
    payload = {
        "wrapper_mode": "fuller_experiment_program",
        "legacy_entrypoint": str(Path(__file__).resolve()),
        "program_contract": str(Path(program_contract).resolve() if Path(program_contract).is_absolute() else (ROOT / Path(program_contract)).resolve()),
        "selected_families": ["device_compare", "noise_robustness", "scaling_support", "report_pack"],
        "execution_plan_csv": execution_payload["execution_plan_csv"],
        "phase4_intake_contract_csv": phase4_payload["phase4_intake_contract_csv"],
        "report_contract_csv": report_payload["report_contract_csv"],
    }
    run_root.mkdir(parents=True, exist_ok=True)
    _json_dump(manifest_json, payload)
    script_lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        "# Legacy wrapper entrypoint for the unified fuller_experiment_program",
        _quote_command(
            [
                sys.executable,
                str(ROOT / "experiments" / "tools" / "check_fuller_experiment_program.py"),
                "--contract",
                str(payload["program_contract"]),
            ]
        ),
        _quote_command(
            [
                sys.executable,
                str(ROOT / "experiments" / "tools" / "check_fuller_experiment_execution_plan.py"),
                "--contract",
                str(payload["program_contract"]),
            ]
        ),
        _quote_command(
            [
                sys.executable,
                str(ROOT / "experiments" / "tools" / "check_fuller_phase4_intake_contract.py"),
                "--contract",
                str(payload["program_contract"]),
            ]
        ),
        _quote_command(
            [
                sys.executable,
                str(ROOT / "experiments" / "tools" / "check_fuller_report_pack_contract.py"),
                "--contract",
                str(payload["program_contract"]),
            ]
        ),
    ]
    script_path.write_text("\n".join(script_lines) + "\n", encoding="utf-8")
    script_path.chmod(0o755)
    print(f"[fuller-program-wrapper] wrote {manifest_json}")
    print(f"[fuller-program-wrapper] wrote {script_path}")


def _cmd_execute_manifest(args: argparse.Namespace) -> None:
    families = {item.strip() for item in args.families.split(",") if item.strip()}
    _run_job_commands(args.manifest_json, families=families or {"all"})


def _cmd_summarize_device(args: argparse.Namespace) -> None:
    summarize_device_benchmark(
        benchmark_csv=args.benchmark_csv,
        system_json=args.system_json,
        out_csv=args.out_csv,
        device_label=args.device_label,
        workload_id=args.workload_id,
        precision_mode=args.precision_mode,
    )


def _cmd_summarize_device_repeats(args: argparse.Namespace) -> None:
    summarize_device_repeats(
        input_csvs=args.input_csvs,
        out_csv=args.out_csv,
    )


def _cmd_summarize_noise(args: argparse.Namespace) -> None:
    summarize_noise_accuracy(
        input_csvs=args.input_csvs,
        out_csv=args.out_csv,
        paired_metrics_csv=args.paired_metrics_csv,
    )


def _cmd_collect_phase1(args: argparse.Namespace) -> None:
    _collect_phase1_outputs(
        manifest_json=args.manifest_json,
        contract=args.contract,
        out_root=args.out_root,
        families={item.strip() for item in args.families.split(",") if item.strip()} or {"all"},
        require_flow_buffer_measured=args.require_flow_buffer_measured,
    )


def _cmd_build_breakdown(args: argparse.Namespace) -> None:
    build_breakdown(
        pairs=args.pairs,
        out_latency_csv=args.out_latency_csv,
        out_energy_csv=args.out_energy_csv,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fuller implementation collection tool.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan_parser = subparsers.add_parser("plan", help="Legacy wrapper for the active fuller_experiment_program.")
    plan_parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    plan_parser.add_argument("--program_contract", type=Path, default=DEFAULT_PROGRAM_CONTRACT)
    plan_parser.add_argument("--contract", type=Path, default=None)
    plan_parser.add_argument("--template", type=Path, default=None)
    plan_parser.add_argument("--generated_root", type=Path, default=DEFAULT_GENERATED_ROOT)
    plan_parser.add_argument("--out_root", type=Path, default=DEFAULT_OUT_ROOT)
    plan_parser.add_argument("--run_tag", required=True)
    plan_parser.add_argument("--imagenet_val", default=None)
    plan_parser.add_argument("--weights_dir", default=None)
    plan_parser.add_argument("--weights_npz_manifest", type=Path, default=None)
    plan_parser.add_argument("--python", default=sys.executable)
    plan_parser.add_argument("--device", default="mps", choices=["mps"])
    plan_parser.add_argument("--workers", type=int, default=None)
    plan_parser.add_argument("--use_caffeinate", action="store_true")
    plan_parser.add_argument(
        "--claim_accuracy_csv",
        type=Path,
        default=None,
        help="Optional config-conditioned claim accuracy CSV used for ablation/scaling accuracy coupling instead of sparse-noise clean summaries.",
    )
    plan_parser.add_argument(
        "--claim_core_program_tag",
        default=None,
        help="Optional core program tag used to map E0-E6 claim contexts (for example 20260331_det_sparse_mlx_final).",
    )
    plan_parser.set_defaults(func=_cmd_plan)

    legacy_plan_parser = subparsers.add_parser("legacy-plan", help="Archived implementation-collection planner.")
    legacy_plan_parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    legacy_plan_parser.add_argument("--contract", type=Path, default=None)
    legacy_plan_parser.add_argument("--template", type=Path, default=None)
    legacy_plan_parser.add_argument("--generated_root", type=Path, default=DEFAULT_GENERATED_ROOT)
    legacy_plan_parser.add_argument("--out_root", type=Path, default=DEFAULT_OUT_ROOT)
    legacy_plan_parser.add_argument("--run_tag", required=True)
    legacy_plan_parser.add_argument("--imagenet_val", required=True)
    legacy_plan_parser.add_argument("--weights_dir", default=None)
    legacy_plan_parser.add_argument("--weights_npz_manifest", type=Path, default=None)
    legacy_plan_parser.add_argument("--python", default=sys.executable)
    legacy_plan_parser.add_argument("--device", default="mps", choices=["mps"])
    legacy_plan_parser.add_argument("--workers", type=int, default=None)
    legacy_plan_parser.add_argument("--use_caffeinate", action="store_true")
    legacy_plan_parser.add_argument("--claim_accuracy_csv", type=Path, default=None)
    legacy_plan_parser.add_argument("--claim_core_program_tag", default=None)
    legacy_plan_parser.set_defaults(func=_cmd_legacy_plan)

    execute_parser = subparsers.add_parser("execute-manifest", help="Execute a materialized manifest.")
    execute_parser.add_argument("--manifest_json", type=Path, required=True)
    execute_parser.add_argument("--families", default="all")
    execute_parser.set_defaults(func=_cmd_execute_manifest)

    summarize_device_parser = subparsers.add_parser("summarize-device", help="Map raw benchmark output to governed CSV.")
    summarize_device_parser.add_argument("--benchmark_csv", type=Path, required=True)
    summarize_device_parser.add_argument("--system_json", type=Path, required=True)
    summarize_device_parser.add_argument("--out_csv", type=Path, required=True)
    summarize_device_parser.add_argument("--device_label", required=True, choices=["cpu", "gpu"])
    summarize_device_parser.add_argument("--workload_id", default="W0_mobilevit_imagenet")
    summarize_device_parser.add_argument("--precision_mode", default="int8_eval")
    summarize_device_parser.set_defaults(func=_cmd_summarize_device)

    summarize_device_repeats_parser = subparsers.add_parser(
        "summarize-device-repeats",
        help="Average repeated device measurements into the governed CSV.",
    )
    summarize_device_repeats_parser.add_argument("--input_csvs", type=Path, nargs="+", required=True)
    summarize_device_repeats_parser.add_argument("--out_csv", type=Path, required=True)
    summarize_device_repeats_parser.set_defaults(func=_cmd_summarize_device_repeats)

    summarize_noise_parser = subparsers.add_parser("summarize-noise", help="Map raw noise CSVs to governed summary.")
    summarize_noise_parser.add_argument("--input_csvs", type=Path, nargs="+", required=True)
    summarize_noise_parser.add_argument("--out_csv", type=Path, required=True)
    summarize_noise_parser.add_argument("--paired_metrics_csv", type=Path, default=None)
    summarize_noise_parser.set_defaults(func=_cmd_summarize_noise)

    collect_phase1_parser = subparsers.add_parser("collect-phase1", help="Collect phase1 outputs into report-data artifacts.")
    collect_phase1_parser.add_argument("--manifest_json", type=Path, required=True)
    collect_phase1_parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    collect_phase1_parser.add_argument("--out_root", type=Path, default=None)
    collect_phase1_parser.add_argument("--families", default="all")
    collect_phase1_parser.add_argument("--require_flow_buffer_measured", action="store_true")
    collect_phase1_parser.set_defaults(func=_cmd_collect_phase1)

    build_breakdown_parser = subparsers.add_parser("build-breakdown", help="Build latency and energy breakdown CSVs.")
    build_breakdown_parser.add_argument("--pairs", action="append", required=True)
    build_breakdown_parser.add_argument("--out_latency_csv", type=Path, required=True)
    build_breakdown_parser.add_argument("--out_energy_csv", type=Path, required=True)
    build_breakdown_parser.set_defaults(func=_cmd_build_breakdown)
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
