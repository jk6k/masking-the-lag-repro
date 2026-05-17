"""Run config-conditioned accuracy evaluations from saved Phase-1 run configs.

This bridges the gap between Phase-1 architectural configs and the ImageNet
accuracy evaluator by forwarding configuration metadata such as experiment ID,
DET k, and SPARSE tau/active fraction into the accuracy CSV.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import shlex
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT_DIR = Path(__file__).resolve().parents[2]
EXPERIMENTS_ROOT = ROOT_DIR / "experiments"
if str(EXPERIMENTS_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_ROOT))

from exp_common.det_prefix import compute_prefix_error_stats
from exp_common.det_policy import resolve_det_runtime_metadata
from accuracy.bitstream_runtime_safety import (
    BITSTREAM_RUNTIME_WORKING_SET_LIMIT_BYTES,
    build_bitstream_runtime_guardrail,
    default_full_surface_runtime_validation_root,
    governed_full_surface_runtime_validation_ready,
)
from accuracy.bitstream_semantics import (
    BITSTREAM_BRIDGE_MEASUREMENT_TRUTH_CLASS,
    BITSTREAM_MODEL_LEVEL_MEASURED_TRUTH_CLASS,
)
from accuracy.bitstream_truth_authorization import write_truth_class_authorization_note
from exp_common.runtime import current_platform_name
from tools.fuller_host_tuning import (
    PASS_KIND_QUANTIZED_EVAL,
    load_host_tuning_profile,
    resolve_accuracy_policy_bundle,
)
from tools.prepare_bitstream_measured_accuracy_source import (
    prepare_bitstream_measured_accuracy_source,
)

try:
    from tools.phase1_runner import _estimate_sparse_scale_from_tau, _resolve_execution_semantics_cfg
except ImportError:

    def _resolve_execution_semantics_cfg(
        cfg: dict[str, Any],
        *,
        override_semantics: str | None = None,
    ) -> dict[str, Any]:
        raw_cfg = cfg.get("bitstream") or {}
        if raw_cfg is None:
            raw_cfg = {}
        if not isinstance(raw_cfg, dict):
            raise SystemExit("Expected optional 'bitstream' config to be a mapping.")

        requested_semantics = str(
            override_semantics
            if override_semantics is not None
            else raw_cfg.get("execution_semantics") or ""
        ).strip().lower()
        default_semantics = str(raw_cfg.get("default_execution_semantics") or "proxy").strip().lower()
        if default_semantics not in {"proxy", "bitstream"}:
            raise SystemExit(f"Unsupported default execution semantics: {default_semantics!r}")
        raw_bitstream_enabled = str(raw_cfg.get("enabled") or "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
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

        try:
            stream_length = int(raw_cfg.get("stream_length"))
        except (TypeError, ValueError):
            stream_length = 0
        if stream_length <= 0:
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

    def _estimate_sparse_scale_from_tau(sparse_cfg: dict[str, Any]) -> float | None:
        tau_requested = str(sparse_cfg.get("use_tau_for_gating") or "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        raw_tau = (
            sparse_cfg.get("tau_global")
            if sparse_cfg.get("tau_global") not in {None, ""}
            else sparse_cfg.get("tau")
        )
        if not tau_requested and raw_tau in {None, ""}:
            return None
        try:
            tau_reference = float(raw_tau)
        except (TypeError, ValueError):
            return None
        if tau_reference <= 0.0:
            return None

        curve_points: list[tuple[float, float]] = []
        for point in sparse_cfg.get("tau_to_active_curve") or []:
            if isinstance(point, dict):
                tau = point.get("tau")
                active = point.get("active_fraction")
            elif isinstance(point, (list, tuple)) and len(point) >= 2:
                tau, active = point[0], point[1]
            else:
                continue
            try:
                curve_points.append((float(tau), float(active)))
            except (TypeError, ValueError):
                continue
        curve_points.sort(key=lambda item: item[0])
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

        try:
            min_active_fraction = float(sparse_cfg.get("min_active_fraction") or 0.0)
        except (TypeError, ValueError):
            min_active_fraction = 0.0
        return max(min(1.0 - tau_reference, 1.0), min_active_fraction)

try:
    from tools.fuller_v2_runtime_smoke_surface import build_lane_annotation_fields
except ImportError:
    _EXPERIMENT_ID_TO_LANE = {
        "E0": "ASTRA",
        "E1": "MESO",
        "E2": "HOPS",
        "E3": "DET",
        "E4": "SPARSE",
        "E5": "PHY",
        "E6": "FULLER",
    }

    def _fallback_eval_row(
        rows: list[dict[str, str]],
        *,
        eval_run_id: str,
    ) -> dict[str, str] | None:
        for row in rows:
            if str(row.get("run_id") or "").strip() != str(eval_run_id).strip():
                continue
            if str(row.get("baseline") or "").strip().lower() == "true":
                continue
            return row
        return None

    def _fallback_lane_id(cfg: dict[str, Any], row: dict[str, Any]) -> str:
        experiment_id = str(
            row.get("experiment_id") or (cfg.get("run") or {}).get("experiment_id") or ""
        ).strip().upper()
        if experiment_id in _EXPERIMENT_ID_TO_LANE:
            return _EXPERIMENT_ID_TO_LANE[experiment_id]
        for key in ("run_id", "source_run_id"):
            value = str(row.get(key) or "").strip().lower()
            for lane_id in _EXPERIMENT_ID_TO_LANE.values():
                if lane_id.lower() in value:
                    return lane_id
        return ""

    def _fallback_to_float(value: Any) -> float | None:
        if value in {None, ""}:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def build_lane_annotation_fields(
        *,
        cfg_path: Path,
        raw_results_csv: Path,
        eval_run_id: str,
        variant_id: str | None = None,
    ) -> dict[str, Any]:
        """Fallback lane surface fields used when the fuller v2 helper is absent."""
        cfg = _load_yaml(cfg_path)
        with raw_results_csv.open("r", newline="", encoding="utf-8") as handle:
            raw_row = _fallback_eval_row(list(csv.DictReader(handle)), eval_run_id=eval_run_id)
        if raw_row is None:
            return {}

        lane_id = str(variant_id or "").strip().upper() or _fallback_lane_id(cfg, raw_row)
        switches = dict(cfg.get("switches") or {})
        meso_cfg = dict(cfg.get("meso") or {})
        flow_cfg = dict(cfg.get("flow") or {})
        sparse_cfg = dict(cfg.get("sparse") or {})
        sc_det_cfg = dict(cfg.get("sc_det") or {})
        phy_cfg = dict(cfg.get("phy") or {})
        fields: dict[str, Any] = {}

        if lane_id == "ASTRA":
            top1 = _fallback_to_float(raw_row.get("top1"))
            top1_delta = _fallback_to_float(raw_row.get("top1_delta"))
            fields.update(
                {
                    "acc_top1": top1 if top1 is not None else "",
                    "acc_drop_pp": (-top1_delta) if top1_delta is not None else "",
                    "accuracy_measurement_contract_status": (
                        str(raw_row.get("bitstream_truth_class_authorization_status") or "")
                        or "not_required"
                    ),
                    "realism_class": str((cfg.get("realism") or {}).get("target_class") or ""),
                }
            )

        if lane_id in {"MESO", "FULLER"}:
            fields.update(
                {
                    "fanout": meso_cfg.get("fanout", ""),
                    "topology_dimension": meso_cfg.get("topology_dimension", "single_stage"),
                    "meso_cost_model_mode": meso_cfg.get("cost_model_mode", "fallback_static"),
                    "serializers_saved": meso_cfg.get("serializers_saved", ""),
                    "explicit_total_cost_j": meso_cfg.get("explicit_total_cost_j", 0.0),
                    "net_energy_gain_j": meso_cfg.get("net_energy_gain_j", 0.0),
                    "meso_cost_evidence": meso_cfg.get("evidence_type", "config_fallback"),
                }
            )

        if lane_id in {"HOPS", "FULLER"}:
            fields.update(
                {
                    "hops_scheduler_mode": flow_cfg.get("scheduler_mode", "disabled"),
                    "stage_cycles": raw_row.get("stage_cycles", 0),
                    "bubble_cycles": raw_row.get("bubble_cycles", 0),
                    "utilization_avg": raw_row.get("utilization_avg", 0),
                    "flow_timeline_evidence": flow_cfg.get("evidence_type", "config_fallback"),
                    "flow_buffer_peak_cycles": raw_row.get("flow_buffer_peak_cycles", 0),
                    "flow_buffer_peak_frac": raw_row.get("flow_buffer_peak_frac", 0),
                    "flow_residency_hit_rate": raw_row.get("flow_residency_hit_rate", 0),
                    "flow_control_backpressure": raw_row.get("flow_control_backpressure", 0),
                    "flow_eviction_count": raw_row.get("flow_eviction_count", 0),
                    "flow_admission_stalls": raw_row.get("flow_admission_stalls", 0),
                }
            )

        if lane_id in {"DET", "FULLER"}:
            det_metadata = resolve_det_runtime_metadata(sc_det_cfg, switches)
            fields.update(
                {
                    "det_policy": det_metadata.get("det_policy") or "",
                    "det_k_signature": det_metadata.get("det_k_signature") or "",
                    "det_quality_gate_status": det_metadata.get("det_quality_gate_status") or "",
                    "det_quality_gate_policy": (
                        det_metadata.get("det_quality_gate_policy")
                        or (sc_det_cfg.get("quality_gate") or {}).get("policy_label")
                        or ""
                    ),
                    "det_quality_gate_reason": det_metadata.get("det_quality_gate_reason") or "",
                    "det_quality_gate_fallback_policy": (
                        det_metadata.get("det_quality_gate_fallback_policy") or ""
                    ),
                    "det_k_global": raw_row.get("det_k_global")
                    or (sc_det_cfg.get("early_stop") or {}).get("k_global", ""),
                    "det_prefix_error_mean": raw_row.get("det_prefix_error_mean", ""),
                    "det_prefix_error_p95": raw_row.get("det_prefix_error_p95", ""),
                    "det_perturbation": raw_row.get("det_perturbation", ""),
                }
            )

        if lane_id in {"SPARSE", "FULLER"}:
            sparse_active_fraction = (
                raw_row.get("sparse_measured_activity_fraction")
                or sparse_cfg.get("active_fraction")
                or 1.0
            )
            fields.update(
                {
                    "duty_cycle_avg": sparse_active_fraction,
                    "sparse_active_fraction": sparse_active_fraction,
                    "sparse_scale_source": "measured_or_config_fallback",
                    "sparse_measured_activity_fraction": raw_row.get(
                        "sparse_measured_activity_fraction", ""
                    ),
                }
            )

        if lane_id in {"PHY", "FULLER"}:
            fields.update(
                {
                    "phy_link_budget_status": "ready" if switches.get("phy") else "disabled",
                    "N_wdm": phy_cfg.get("wdm_channels_n", ""),
                    "P_laser_mw": phy_cfg.get("p_laser_mw", 0),
                    "PP_crosstalk_db": (phy_cfg.get("crosstalk") or {}).get(
                        "pp_crosstalk_db",
                        "",
                    ),
                    "gaussian_noise_std_ref": (cfg.get("p1_align") or {}).get(
                        "gaussian_noise_sigma_lsb_ref",
                        "",
                    ),
                    "crosstalk_alpha_ref": (cfg.get("p1_align") or {}).get(
                        "crosstalk_alpha_ref",
                        "",
                    ),
                    "phy_support_evidence": phy_cfg.get("evidence_type", "config_fallback"),
                }
            )

        if lane_id == "FULLER":
            fields.update(
                {
                    "integrated_system_cost_mode": (cfg.get("integrated_system_costs") or {}).get(
                        "mode",
                        "",
                    ),
                    "integrated_system_cost_evidence": (
                        cfg.get("integrated_system_costs") or {}
                    ).get("onchip_comm_evidence_type", "config_fallback"),
                    "accuracy_coupling_evidence": (
                        "measured" if raw_row.get("top1") not in {None, ""} else "missing"
                    ),
                    "proxy_promotion_ready": "false",
                    "benchmark_claim_ready": "false",
                }
            )

        return fields

RUNS_DIR = ROOT_DIR / "experiments" / "results" / "runs"
DEFAULT_SPLITS_DIR = ROOT_DIR / "experiments" / "results" / "accuracy" / "splits_20260228_opt_cuda"
TORCH_EVAL_SCRIPT = "experiments/accuracy/eval_cvnets_imagenet_noise.py"
MLX_EVAL_SCRIPT = "experiments/accuracy/eval_mlx_imagenet_noise.py"
DEFAULT_PROGRESS_ROOT = ROOT_DIR / "experiments" / "results" / "accuracy" / "progress"
DEFAULT_PROGRESS_HEARTBEAT_INTERVAL_SECONDS = 15.0
DEFAULT_STALL_TIMEOUT_SECONDS = 180.0
DEFAULT_PRELAUNCH_RUNTIME_SMOKE_SAMPLES = 16
DEFAULT_PRELAUNCH_MIN_SAMPLES_PER_HOUR = 30.0
DEFAULT_PRELAUNCH_MAX_SECONDS_PER_SAMPLE = 120.0
DEFAULT_RUNTIME_MIN_SAMPLES_PER_HOUR = 12.0
DEFAULT_RUNTIME_MAX_SECONDS_PER_SAMPLE = 300.0
DEFAULT_RUNTIME_MAX_ETA_CURRENT_RATE_SECONDS = 86400.0
DEFAULT_RUNTIME_MIN_PROCESSED_SAMPLES = 4
DEFAULT_RUNTIME_MIN_ELAPSED_SECONDS = 300.0
GOVERNED_FULL_SURFACE_RUNTIME_SLOWDOWN_TOLERANCE = 1.5
_ACTIVE_CHILD: subprocess.Popen[str] | None = None
_STOP_REQUESTED = False
RUNTIME_SMOKE_TIER = "runtime_smoke"
ANALYSIS_GRADE_TIER = "analysis_grade"
ANALYSIS_GRADE_REQUIRED_SEEDS = [0, 1, 2]
PASS_MODE_PAIRED = "paired"
PASS_MODE_BASELINE_ONLY = "baseline_only"
PASS_MODE_QUANTIZED_ONLY = "quantized_only"
PASS_MODE_CHOICES = [
    PASS_MODE_PAIRED,
    PASS_MODE_BASELINE_ONLY,
    PASS_MODE_QUANTIZED_ONLY,
]
ANALYSIS_GRADE_ALLOWED_PASS_MODES = {
    PASS_MODE_PAIRED,
    PASS_MODE_QUANTIZED_ONLY,
}
FULL_SURFACE_RUNTIME_VALIDATION_ROOT = default_full_surface_runtime_validation_root(ROOT_DIR)


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise SystemExit(f"Invalid YAML object in {path}")
    return data


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


def _normalize_optional_token(value: str | None) -> str | None:
    text = str(value or "").strip()
    return text or None


def _resolved_quantized_execution_semantics(
    *,
    cfg: dict[str, Any],
    enable_bitstream_pilot: bool,
) -> str:
    if enable_bitstream_pilot:
        return "bitstream"
    execution_semantics_cfg = _resolve_execution_semantics_cfg(cfg)
    return str(execution_semantics_cfg.get("execution_semantics") or "").strip().lower()


def _staged_accuracy_policy_profile_path(
    progress_root: Path,
    *,
    eval_run_id: str,
) -> Path:
    return progress_root / "policy_profiles" / f"{_sanitize_token(eval_run_id)}.json"


def _resolve_existing_manifest_path(
    candidate: str | Path | None,
    *,
    search_root: Path = ROOT_DIR / "experiments" / "results" / "accuracy",
) -> str | None:
    if candidate is None:
        return None
    raw = str(candidate).strip()
    if not raw:
        return None

    path = Path(raw)
    probe_paths = [path]
    if not path.is_absolute():
        probe_paths.append(ROOT_DIR / path)
    for probe in probe_paths:
        if probe.is_file():
            return str(probe)

    filename = path.name.strip()
    if not filename or not search_root.is_dir():
        return raw

    matches = sorted(p for p in search_root.rglob(filename) if p.is_file())
    if len(matches) == 1:
        return str(matches[0])
    return raw


def _resolve_manifest(split: str, data_cfg: dict[str, Any], override: str | None) -> str | None:
    if override:
        return _resolve_existing_manifest_path(override)
    split = split.strip().lower()
    if split == "eval":
        return _resolve_existing_manifest_path(
            data_cfg.get("eval_manifest_csv") or (DEFAULT_SPLITS_DIR / "imagenet_val_eval.csv")
        )
    if split == "calib":
        return _resolve_existing_manifest_path(
            data_cfg.get("calib_manifest_csv") or (DEFAULT_SPLITS_DIR / "imagenet_val_calib.csv")
        )
    if split == "holdout":
        return _resolve_existing_manifest_path(data_cfg.get("holdout_manifest_csv") or "")
    return None


def _sanitize_token(value: str) -> str:
    cleaned = "".join(
        char if char.isalnum() or char in {"-", "_", "."} else "_"
        for char in str(value).strip()
    )
    return cleaned.strip("._") or "unnamed"


def _default_progress_root(results_csv: str) -> Path:
    results_path = Path(results_csv)
    stem = results_path.stem or "accuracy"
    return DEFAULT_PROGRESS_ROOT / stem


def _progress_jsonl_path(
    progress_root: Path,
    *,
    run_id: str,
    model_key: str,
    seed: int,
) -> Path:
    slug = f"{_sanitize_token(run_id)}_{_sanitize_token(model_key)}_s{int(seed)}.jsonl"
    return progress_root / "events" / slug


def _progress_manifest_path(
    progress_root: Path,
    *,
    explicit_manifest: str | None,
) -> Path:
    if explicit_manifest:
        return Path(explicit_manifest)
    return progress_root / "manifest.json"


def _prelaunch_smoke_root(
    progress_root: Path,
    *,
    eval_run_id: str,
) -> Path:
    return progress_root / "prelaunch_smoke" / _sanitize_token(eval_run_id)


def _authorization_note_path(
    root: Path,
    *,
    eval_run_id: str,
) -> Path:
    return root / (
        f"{_sanitize_token(eval_run_id)}_bitstream_model_level_measured_authorization.md"
    )


def _prepared_phase1_config_path(
    root: Path,
    *,
    eval_run_id: str,
) -> Path:
    return root / f"{_sanitize_token(eval_run_id)}_prepared_phase1_config.yaml"


def _prepared_eligibility_report_paths(
    root: Path,
    *,
    eval_run_id: str,
) -> tuple[Path, Path]:
    stem = f"{_sanitize_token(eval_run_id)}_measured_row_eligibility"
    return root / f"{stem}.json", root / f"{stem}.md"


def _eval_config_snapshot_path(*, eval_run_id: str) -> Path:
    return RUNS_DIR / _sanitize_token(eval_run_id) / "config_snapshot.yaml"


def _materialize_eval_snapshot_cfg(
    *,
    cfg: dict[str, Any],
    enable_bitstream_pilot: bool,
    bitstream_generator: str | None,
    bitstream_stream_length: int | None,
    bitstream_encoding_mode: str | None,
    bitstream_multiplier_mode: str | None,
    bitstream_accumulator_mode: str | None,
) -> dict[str, Any]:
    staged_cfg = copy.deepcopy(cfg)
    if not enable_bitstream_pilot:
        return staged_cfg
    raw_bitstream_cfg = staged_cfg.get("bitstream")
    if raw_bitstream_cfg in (None, ""):
        raw_bitstream_cfg = {}
    if not isinstance(raw_bitstream_cfg, dict):
        raise SystemExit("Expected optional 'bitstream' config to be a mapping.")
    bitstream_cfg = dict(raw_bitstream_cfg)
    bitstream_cfg["enabled"] = True
    bitstream_cfg["execution_semantics"] = "bitstream"
    bitstream_cfg["default_execution_semantics"] = "bitstream"
    if bitstream_generator:
        bitstream_cfg["generator"] = str(bitstream_generator)
    if bitstream_stream_length is not None:
        bitstream_cfg["stream_length"] = int(bitstream_stream_length)
    if bitstream_encoding_mode:
        bitstream_cfg["encoding_mode"] = str(bitstream_encoding_mode)
    if bitstream_multiplier_mode:
        bitstream_cfg["multiplier_mode"] = str(bitstream_multiplier_mode)
    if bitstream_accumulator_mode:
        bitstream_cfg["accumulator_mode"] = str(bitstream_accumulator_mode)
    staged_cfg["bitstream"] = bitstream_cfg
    return staged_cfg


def _stage_eval_config_snapshot(
    *,
    cfg: dict[str, Any],
    eval_run_id: str,
    enable_bitstream_pilot: bool,
    bitstream_generator: str | None,
    bitstream_stream_length: int | None,
    bitstream_encoding_mode: str | None,
    bitstream_multiplier_mode: str | None,
    bitstream_accumulator_mode: str | None,
) -> Path:
    staged_cfg = _materialize_eval_snapshot_cfg(
        cfg=cfg,
        enable_bitstream_pilot=enable_bitstream_pilot,
        bitstream_generator=bitstream_generator,
        bitstream_stream_length=bitstream_stream_length,
        bitstream_encoding_mode=bitstream_encoding_mode,
        bitstream_multiplier_mode=bitstream_multiplier_mode,
        bitstream_accumulator_mode=bitstream_accumulator_mode,
    )
    path = _eval_config_snapshot_path(eval_run_id=eval_run_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(staged_cfg, handle, sort_keys=False)
    return path


def _quote_command(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def _latest_progress_event_from_jsonl(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    latest: dict[str, Any] | None = None
    latest_metrics: dict[str, Any] | None = None
    for raw in path.read_text(encoding="utf-8").splitlines()[-100:]:
        if not raw.strip():
            continue
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            continue
        latest = payload
        if payload.get("processed_samples") is not None:
            latest_metrics = payload
    return latest_metrics or latest


def _count_csv_rows(path: Path) -> int | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        line_count = sum(1 for _ in handle)
    return max(0, line_count - 1)


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _to_bool(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _resolve_planned_pass_count(pass_mode: str) -> int:
    return 2 if pass_mode == PASS_MODE_PAIRED else 1


def _write_baseline_reference_summary(
    *,
    raw_results_csv: Path,
    eval_run_id: str,
    output_json: Path,
) -> dict[str, Any]:
    rows = _read_csv_rows(raw_results_csv)
    baseline_rows = [
        row
        for row in rows
        if str(row.get("run_id") or "").strip() == eval_run_id and _to_bool(row.get("baseline"))
    ]
    if not baseline_rows:
        raise SystemExit(
            f"Could not materialize baseline reference summary because no baseline row matched run_id={eval_run_id} in {raw_results_csv}"
        )
    baseline_row = baseline_rows[-1]
    payload = {
        "run_id": eval_run_id,
        "baseline_row_count": len(baseline_rows),
        "baseline_reference_csv": str(raw_results_csv),
        "measurement_window": str(baseline_row.get("measurement_window") or "").strip(),
        "top1": str(baseline_row.get("top1") or "").strip(),
        "top5": str(baseline_row.get("top5") or "").strip(),
        "seed": str(baseline_row.get("seed") or "").strip(),
        "model": str(baseline_row.get("model") or "").strip(),
        "workload": str(baseline_row.get("workload") or "").strip(),
    }
    _write_json(output_json, payload)
    return payload


def _set_single_arg_option(command: list[str], flag: str, value: str) -> list[str]:
    rewritten: list[str] = []
    replaced = False
    index = 0
    while index < len(command):
        token = command[index]
        if token == flag:
            rewritten.extend([flag, value])
            replaced = True
            index += 2
            continue
        rewritten.append(token)
        index += 1
    if not replaced:
        rewritten.extend([flag, value])
    return rewritten


def _drop_flag(command: list[str], flag: str) -> list[str]:
    return [token for token in command if token != flag]


def _build_prelaunch_smoke_command(
    *,
    command: list[str],
    eval_run_id: str,
    progress_root: Path,
    smoke_samples: int,
    min_samples_per_hour: float,
    max_seconds_per_sample: float,
) -> tuple[list[str], dict[str, str]]:
    smoke_root = _prelaunch_smoke_root(progress_root, eval_run_id=eval_run_id)
    smoke_results_csv = smoke_root / "raw_accuracy.csv"
    smoke_progress_jsonl = smoke_root / "progress.jsonl"
    smoke_summary_json = smoke_root / "summary.json"
    smoke_command = list(command)
    smoke_command = _set_single_arg_option(
        smoke_command,
        "--results_csv",
        str(smoke_results_csv),
    )
    smoke_command = _set_single_arg_option(
        smoke_command,
        "--progress_jsonl",
        str(smoke_progress_jsonl),
    )
    smoke_command = _set_single_arg_option(
        smoke_command,
        "--progress_label",
        f"{eval_run_id}:prelaunch_smoke",
    )
    smoke_command = _set_single_arg_option(
        smoke_command,
        "--max_eval_samples",
        str(int(smoke_samples)),
    )
    smoke_command = _set_single_arg_option(
        smoke_command,
        "--pathological_min_processed_samples",
        "1",
    )
    smoke_command = _set_single_arg_option(
        smoke_command,
        "--pathological_min_elapsed_seconds",
        "0",
    )
    smoke_command = _set_single_arg_option(
        smoke_command,
        "--pathological_min_samples_per_hour",
        str(float(min_samples_per_hour)),
    )
    smoke_command = _set_single_arg_option(
        smoke_command,
        "--pathological_max_seconds_per_sample",
        str(float(max_seconds_per_sample)),
    )
    smoke_command = _drop_flag(smoke_command, "--append")
    smoke_command = _drop_flag(smoke_command, "--resume")
    return smoke_command, {
        "smoke_root": str(smoke_root),
        "results_csv": str(smoke_results_csv),
        "progress_jsonl": str(smoke_progress_jsonl),
        "summary_json": str(smoke_summary_json),
    }


def _should_enable_prelaunch_runtime_smoke(
    *,
    accuracy_backend: str,
    resume: bool,
    pass_mode: str = PASS_MODE_PAIRED,
    enable_bitstream_pilot: bool,
    smoke_samples: int,
    bitstream_runtime_guardrail: dict[str, Any] | None = None,
) -> bool:
    if accuracy_backend != "mlx":
        return False
    if int(smoke_samples) <= 0:
        return False
    if pass_mode == PASS_MODE_BASELINE_ONLY:
        return False
    guardrail = dict(bitstream_runtime_guardrail or {})
    if (
        enable_bitstream_pilot
        and bool(guardrail.get("all_target_surface"))
        and bool(guardrail.get("governed_full_surface_runtime_validation_ready"))
    ):
        # Governed validation already proves the full-surface all-target path can
        # take tens of minutes to finish its first quantized sample. The bounded
        # smoke thresholds are intentionally much tighter, so re-running that
        # smoke would only produce a known false-negative gate.
        return False
    if not resume:
        return True
    # Full-surface bitstream resumes have repeatedly re-entered the same
    # pathological-throughput regime before producing a durable quantized row.
    # Re-running the bounded smoke keeps those retries cheap without changing
    # the resumed command itself.
    return bool(enable_bitstream_pilot)


def _load_reusable_prelaunch_smoke_summary(summary_json: Path) -> dict[str, Any] | None:
    if not summary_json.is_file():
        return None
    try:
        payload = json.loads(summary_json.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    returncode = payload.get("returncode")
    if returncode is None or int(returncode) != 0:
        return None
    if str(payload.get("status") or "").strip() != "pass":
        return None
    raw_rows = payload.get("raw_accuracy_rows")
    if raw_rows is not None and int(raw_rows) <= 0:
        return None
    payload["reused"] = True
    return payload


def _effective_pathological_min_processed_samples(
    *,
    accuracy_backend: str,
    resume: bool,
    enable_bitstream_pilot: bool,
    bitstream_surface_scope: str | None,
    pathological_min_processed_samples: int | None,
) -> int | None:
    resolved = (
        None
        if pathological_min_processed_samples is None
        else int(pathological_min_processed_samples)
    )
    if (
        accuracy_backend == "mlx"
        and resume
        and enable_bitstream_pilot
        and str(bitstream_surface_scope or "").strip() == "all"
        and (resolved is None or resolved > 1)
    ):
        # Resumed all-target bitstream retries have repeatedly failed after only
        # one quantized sample. Lower the gate so pathological throughput aborts
        # before another 35m-47m retry burns the serialized slot.
        return 1
    return resolved


def _parse_optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _load_governed_full_surface_runtime_reference(
    *,
    validation_root: str | Path | None,
    model_key: str | None,
) -> dict[str, float] | None:
    if validation_root in (None, ""):
        return None
    raw_csv = Path(validation_root) / "raw_accuracy.csv"
    if not raw_csv.exists():
        return None
    try:
        with raw_csv.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except OSError:
        return None
    candidates: list[dict[str, float]] = []
    for row in rows:
        if str(row.get("baseline") or "").strip().lower() == "true":
            continue
        if str(row.get("execution_semantics") or "").strip() != "bitstream":
            continue
        if model_key and str(row.get("model") or "").strip() != str(model_key):
            continue
        elapsed = _parse_optional_float(row.get("measured_pass_elapsed_s"))
        processed = _parse_optional_float(row.get("measured_processed_samples"))
        if elapsed is None or elapsed <= 0.0 or processed is None or processed <= 0.0:
            continue
        seconds_per_sample = float(elapsed) / float(processed)
        if seconds_per_sample <= 0.0:
            continue
        candidates.append(
            {
                "measured_pass_elapsed_s": float(elapsed),
                "measured_processed_samples": float(processed),
                "seconds_per_sample": seconds_per_sample,
                "samples_per_hour": 3600.0 / seconds_per_sample,
            }
        )
    if not candidates:
        return None
    return min(candidates, key=lambda item: item["seconds_per_sample"])


def _effective_runtime_health_gate(
    *,
    accuracy_backend: str,
    enable_bitstream_pilot: bool,
    bitstream_surface_scope: str | None,
    model_key: str | None,
    pathological_min_samples_per_hour: float | None,
    pathological_max_seconds_per_sample: float | None,
    pathological_max_eta_current_rate_seconds: float | None,
    pathological_min_elapsed_seconds: float | None,
    bitstream_runtime_guardrail: dict[str, Any] | None = None,
) -> dict[str, float | None]:
    gate = {
        "pathological_min_samples_per_hour": (
            None
            if pathological_min_samples_per_hour is None
            else float(pathological_min_samples_per_hour)
        ),
        "pathological_max_seconds_per_sample": (
            None
            if pathological_max_seconds_per_sample is None
            else float(pathological_max_seconds_per_sample)
        ),
        "pathological_max_eta_current_rate_seconds": (
            None
            if pathological_max_eta_current_rate_seconds is None
            else float(pathological_max_eta_current_rate_seconds)
        ),
        "pathological_min_elapsed_seconds": (
            None
            if pathological_min_elapsed_seconds is None
            else float(pathological_min_elapsed_seconds)
        ),
    }
    if accuracy_backend != "mlx" or not enable_bitstream_pilot:
        return gate
    guardrail = dict(bitstream_runtime_guardrail or {})
    if not (
        bool(guardrail.get("all_target_surface"))
        and bool(guardrail.get("governed_full_surface_runtime_validation_ready"))
        and str(bitstream_surface_scope or guardrail.get("surface_scope") or "").strip()
        == "all"
    ):
        return gate
    reference = _load_governed_full_surface_runtime_reference(
        validation_root=guardrail.get("governed_full_surface_runtime_validation_root"),
        model_key=model_key,
    )
    if reference is None:
        return gate
    slowdown = float(GOVERNED_FULL_SURFACE_RUNTIME_SLOWDOWN_TOLERANCE)
    relaxed_min_samples_per_hour = reference["samples_per_hour"] / slowdown
    relaxed_max_seconds_per_sample = reference["seconds_per_sample"] * slowdown
    relaxed_min_elapsed_seconds = reference["seconds_per_sample"]
    current_min_samples_per_hour = gate["pathological_min_samples_per_hour"]
    if current_min_samples_per_hour is None:
        gate["pathological_min_samples_per_hour"] = relaxed_min_samples_per_hour
    else:
        gate["pathological_min_samples_per_hour"] = min(
            float(current_min_samples_per_hour),
            relaxed_min_samples_per_hour,
        )
    current_max_seconds_per_sample = gate["pathological_max_seconds_per_sample"]
    if current_max_seconds_per_sample is None:
        gate["pathological_max_seconds_per_sample"] = relaxed_max_seconds_per_sample
    else:
        gate["pathological_max_seconds_per_sample"] = max(
            float(current_max_seconds_per_sample),
            relaxed_max_seconds_per_sample,
        )
    current_min_elapsed_seconds = gate["pathological_min_elapsed_seconds"]
    if current_min_elapsed_seconds is None:
        gate["pathological_min_elapsed_seconds"] = relaxed_min_elapsed_seconds
    else:
        gate["pathological_min_elapsed_seconds"] = max(
            float(current_min_elapsed_seconds),
            relaxed_min_elapsed_seconds,
        )
    # Current-rate ETA scales with the full 45k-sample evaluation and is not a
    # useful pathology signal once governed validation has already established
    # that a single all-target sample can legitimately take tens of minutes.
    gate["pathological_max_eta_current_rate_seconds"] = 0.0
    return gate


def _run_prelaunch_runtime_smoke(
    *,
    command: list[str],
    eval_run_id: str,
    progress_root: Path,
    smoke_samples: int,
    min_samples_per_hour: float,
    max_seconds_per_sample: float,
) -> dict[str, Any]:
    global _ACTIVE_CHILD
    smoke_command, smoke_paths = _build_prelaunch_smoke_command(
        command=command,
        eval_run_id=eval_run_id,
        progress_root=progress_root,
        smoke_samples=smoke_samples,
        min_samples_per_hour=min_samples_per_hour,
        max_seconds_per_sample=max_seconds_per_sample,
    )
    smoke_root = Path(smoke_paths["smoke_root"])
    smoke_root.mkdir(parents=True, exist_ok=True)
    reusable_summary = _load_reusable_prelaunch_smoke_summary(
        Path(smoke_paths["summary_json"])
    )
    if reusable_summary is not None:
        print(
            "[prelaunch-smoke] reuse passed eval_run_id={eval_run_id} summary={summary}".format(
                eval_run_id=eval_run_id,
                summary=smoke_paths["summary_json"],
            ),
            flush=True,
        )
        return reusable_summary
    print("[prelaunch-smoke]", _quote_command(smoke_command), flush=True)
    try:
        proc = subprocess.Popen(smoke_command, cwd=ROOT_DIR, text=True)
        _ACTIVE_CHILD = proc
        returncode = proc.wait()
        _ACTIVE_CHILD = None
    except KeyboardInterrupt:
        raise SystemExit(130)

    progress_event = _latest_progress_event_from_jsonl(Path(smoke_paths["progress_jsonl"]))
    smoke_summary = {
        "eval_run_id": eval_run_id,
        "command": _quote_command(smoke_command),
        "returncode": int(returncode),
        "results_csv": smoke_paths["results_csv"],
        "progress_jsonl": smoke_paths["progress_jsonl"],
        "samples": smoke_samples,
        "min_samples_per_hour": float(min_samples_per_hour),
        "max_seconds_per_sample": float(max_seconds_per_sample),
        "raw_accuracy_rows": _count_csv_rows(Path(smoke_paths["results_csv"])),
        "latest_progress_event": progress_event,
        "status": "pass" if returncode == 0 else "fail",
    }
    Path(smoke_paths["summary_json"]).write_text(
        json.dumps(smoke_summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    if returncode != 0:
        raise SystemExit(
            "Prelaunch runtime smoke failed for "
            f"{eval_run_id}. See {smoke_paths['summary_json']}."
        )
    if smoke_summary["raw_accuracy_rows"] is not None and int(smoke_summary["raw_accuracy_rows"]) <= 0:
        raise SystemExit(
            "Prelaunch runtime smoke completed without producing accuracy rows for "
            f"{eval_run_id}. See {smoke_paths['summary_json']}."
        )
    print(
        "[prelaunch-smoke] passed eval_run_id={eval_run_id} rows={rows} summary={summary}".format(
            eval_run_id=eval_run_id,
            rows=smoke_summary["raw_accuracy_rows"],
            summary=smoke_paths["summary_json"],
        ),
        flush=True,
    )
    return smoke_summary


def _request_graceful_stop(signum, _frame) -> None:
    global _STOP_REQUESTED, _ACTIVE_CHILD
    if _STOP_REQUESTED:
        return
    _STOP_REQUESTED = True
    signal_name = signal.Signals(signum).name
    print(
        f"[interrupt] received {signal_name}; forwarding to the active eval subprocess so it can stop cleanly for --resume.",
        flush=True,
    )
    child = _ACTIVE_CHILD
    if child is not None and child.poll() is None:
        try:
            child.send_signal(signum)
        except Exception:
            pass


def _install_interrupt_handlers() -> None:
    for signame in ("SIGINT", "SIGTERM"):
        signum = getattr(signal, signame, None)
        if signum is not None:
            signal.signal(signum, _request_graceful_stop)


def _format_float(value: Any) -> str | None:
    if value is None:
        return None
    try:
        return str(float(value))
    except (TypeError, ValueError):
        return None


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _sparse_tau_controls_gating(sparse_cfg: dict[str, Any]) -> bool:
    if _to_float(sparse_cfg.get("tau_global")) is not None:
        return True
    if _estimate_sparse_scale_from_tau(sparse_cfg) is not None:
        return True
    return str(sparse_cfg.get("use_tau_for_gating") or "").strip().lower() in {
        "1",
        "true",
        "yes",
    }


def _resolve_det_mode(sc_det_cfg: dict[str, Any]) -> str:
    mode = str(sc_det_cfg.get("det_mode") or "reorder").strip().lower()
    return mode if mode in {"reorder", "replace"} else "reorder"


def _resolve_sparse_active_fraction(sparse_cfg: dict[str, Any]) -> Any:
    if _sparse_tau_controls_gating(sparse_cfg):
        tau_estimate = _estimate_sparse_scale_from_tau(sparse_cfg)
        if tau_estimate is not None:
            return tau_estimate
        tau_value = _to_float(sparse_cfg.get("tau_global"))
        if tau_value is not None:
            return 1.0 if tau_value <= 0.0 else max(0.0, min(1.0, 1.0 - tau_value))
        return 1.0
    active_fraction = sparse_cfg.get("active_fraction")
    if active_fraction is not None:
        return active_fraction
    if "sparsity" in sparse_cfg:
        try:
            return 1.0 - float(sparse_cfg.get("sparsity") or 0.0)
        except (TypeError, ValueError):
            return None
    return None


def _parse_model_keys(raw: str) -> list[str]:
    return [item.strip() for item in str(raw).split(",") if item.strip()]


def _parse_model_int_overrides(
    raw: str | None,
    *,
    field_name: str,
) -> dict[str, int]:
    if raw in ("", None):
        return {}
    mapping: dict[str, int] = {}
    for item in str(raw).split(","):
        token = item.strip()
        if not token:
            continue
        model_key, sep, raw_value = token.partition("=")
        if not sep:
            raise SystemExit(
                f"{field_name} entries must use model=value syntax; got {token!r}."
            )
        model_key = model_key.strip()
        raw_value = raw_value.strip()
        if not model_key:
            raise SystemExit(f"{field_name} contains an empty model key.")
        try:
            value = int(raw_value)
        except ValueError as exc:
            raise SystemExit(
                f"{field_name} requires integer values; got {token!r}."
            ) from exc
        if value <= 0:
            raise SystemExit(
                f"{field_name} requires positive integer values; got {token!r}."
            )
        mapping[model_key] = value
    return mapping


def _resolve_model_eval_batch_size(
    *,
    model_key: str,
    default_eval_batch_size: int | None,
    model_eval_batch_sizes: dict[str, int],
) -> int | None:
    if model_key in model_eval_batch_sizes:
        return int(model_eval_batch_sizes[model_key])
    return default_eval_batch_size


def _format_gib(value: int) -> str:
    return f"{float(value) / float(1024**3):.2f} GiB"


def _validate_bitstream_runtime_guardrail(
    *,
    model_key: str,
    accuracy_backend: str,
    enable_bitstream_pilot: bool,
    surface_scope: str | None,
    target_module_keys: str | None,
    eval_batch_size: int | None,
    stream_length: int | None,
    allow_unsafe_runtime_shapes: bool,
) -> dict[str, Any] | None:
    if accuracy_backend != "mlx" or not enable_bitstream_pilot:
        return None
    guardrail = build_bitstream_runtime_guardrail(
        model_key=model_key,
        surface_scope=surface_scope,
        target_module_keys_raw=target_module_keys,
        explicit_eval_batch_size=eval_batch_size,
        stream_length=int(stream_length or 64),
        root_dir=ROOT_DIR,
    )
    blockers: list[str] = []
    full_surface_validation_ready = governed_full_surface_runtime_validation_ready(
        FULL_SURFACE_RUNTIME_VALIDATION_ROOT
    )
    guardrail["governed_full_surface_runtime_validation_ready"] = bool(
        full_surface_validation_ready
    )
    guardrail["governed_full_surface_runtime_validation_root"] = str(
        FULL_SURFACE_RUNTIME_VALIDATION_ROOT
    )
    if guardrail["all_target_surface"] and not full_surface_validation_ready:
        blockers.append(
            "all-target bitstream runtime lanes are frozen until the repaired path is validated"
        )
    explicit_batch_size = guardrail["explicit_eval_batch_size"]
    safe_batch_cap = int(guardrail["safe_quantized_batch_cap"])
    if explicit_batch_size is not None and int(explicit_batch_size) > safe_batch_cap:
        blockers.append(
            "explicit eval_batch_size exceeds the safe quantized bitstream batch cap "
            f"({explicit_batch_size} > {safe_batch_cap})"
        )
    estimated_working_set_bytes = int(guardrail["estimated_working_set_bytes"])
    if estimated_working_set_bytes > int(BITSTREAM_RUNTIME_WORKING_SET_LIMIT_BYTES):
        blockers.append(
            "estimated quantized working set exceeds the current launch bound "
            f"({_format_gib(estimated_working_set_bytes)} > "
            f"{_format_gib(int(BITSTREAM_RUNTIME_WORKING_SET_LIMIT_BYTES))})"
        )
    if blockers and not allow_unsafe_runtime_shapes:
        detail = (
            f"model={model_key} "
            f"active_targets={guardrail['active_target_module_count']} "
            f"targetable={guardrail['targetable_module_count']} "
            f"surface_scope={guardrail['surface_scope'] or 'default_all_target'} "
            f"safe_quantized_batch_cap={safe_batch_cap} "
            f"estimated_quantized_batch_size={guardrail['estimated_quantized_batch_size']} "
            f"estimated_working_set={_format_gib(estimated_working_set_bytes)}"
        )
        raise SystemExit(
            "Refusing unsafe bitstream runtime launch shape. "
            + "; ".join(blockers)
            + f". {detail}. Use --allow_unsafe_bitstream_runtime_shapes only for deliberate repair work."
        )
    return guardrail


def _maybe_annotate_bitstream_results(
    *,
    cfg_path: Path,
    eval_run_id: str,
    raw_results_csv: Path,
    annotated_results_csv: Path | None,
    pass_mode: str = PASS_MODE_PAIRED,
    contract_note: str,
    measurement_truth_class: str,
    extra_annotation_fields: dict[str, Any] | None = None,
    output_config: Path | None = None,
    eligibility_report_json: Path | None = None,
    eligibility_report_md: Path | None = None,
) -> dict[str, Any] | None:
    if annotated_results_csv is None:
        return None
    if pass_mode == PASS_MODE_BASELINE_ONLY:
        return None
    cfg = _load_yaml(cfg_path)
    execution_semantics_cfg = _resolve_execution_semantics_cfg(cfg)
    if execution_semantics_cfg["execution_semantics"] != "bitstream":
        return None
    resolved_extra_annotation_fields = dict(extra_annotation_fields or {})
    resolved_extra_annotation_fields.update(
        build_lane_annotation_fields(
            cfg_path=cfg_path,
            raw_results_csv=raw_results_csv,
            eval_run_id=eval_run_id,
        )
    )
    return prepare_bitstream_measured_accuracy_source(
        phase1_config=cfg_path,
        input_csv=raw_results_csv,
        output_csv=annotated_results_csv,
        output_config=output_config,
        raw_match_filters=[f"run_id={eval_run_id}"],
        contract_note=contract_note,
        measurement_truth_class=measurement_truth_class,
        extra_annotation_fields=resolved_extra_annotation_fields,
        drop_explicit_coupling=True,
        eligibility_report_json=eligibility_report_json,
        eligibility_report_md=eligibility_report_md,
    )


def _parse_seeds(raw: str) -> list[int]:
    return [int(item.strip()) for item in str(raw).split(",") if item.strip()]


def _resolve_analysis_grade_status(
    *,
    evidence_tier: str,
    seeds: list[int],
    max_eval_samples: int | None,
    annotated_results_csv: str | None,
    pass_mode: str,
    baseline_reference_csv: str | None = None,
    baseline_reference_run_id: str | None = None,
) -> tuple[bool, list[str]]:
    blockers: list[str] = []
    if evidence_tier != ANALYSIS_GRADE_TIER:
        blockers.append("runtime_smoke_only")
    if sorted(set(seeds)) != ANALYSIS_GRADE_REQUIRED_SEEDS or len(seeds) != len(
        ANALYSIS_GRADE_REQUIRED_SEEDS
    ):
        blockers.append("missing_seeds012")
    if max_eval_samples is not None:
        blockers.append("full_eval_required")
    if not str(annotated_results_csv or "").strip():
        blockers.append("annotated_results_csv_required")
    if pass_mode not in ANALYSIS_GRADE_ALLOWED_PASS_MODES:
        blockers.append("analysis_grade_pass_mode_invalid")
    if pass_mode == PASS_MODE_QUANTIZED_ONLY:
        if not str(baseline_reference_csv or "").strip():
            blockers.append("baseline_reference_csv_required")
        if not str(baseline_reference_run_id or "").strip():
            blockers.append("baseline_reference_run_id_required")
    ready = evidence_tier == ANALYSIS_GRADE_TIER and not blockers
    return ready, blockers


def _validate_evidence_tier_args(
    *,
    evidence_tier: str,
    seeds: list[int],
    max_eval_samples: int | None,
    annotated_results_csv: str | None,
    pass_mode: str,
    baseline_reference_csv: str | None = None,
    baseline_reference_run_id: str | None = None,
) -> None:
    if evidence_tier != ANALYSIS_GRADE_TIER:
        return
    if sorted(set(seeds)) != ANALYSIS_GRADE_REQUIRED_SEEDS or len(seeds) != len(
        ANALYSIS_GRADE_REQUIRED_SEEDS
    ):
        raise SystemExit(
            "--evidence_tier analysis_grade requires --seeds 0,1,2."
        )
    if max_eval_samples is not None:
        raise SystemExit(
            "--evidence_tier analysis_grade requires full eval and does not allow --max_eval_samples."
        )
    if not str(annotated_results_csv or "").strip():
        raise SystemExit(
            "--evidence_tier analysis_grade requires --annotated_results_csv."
        )
    if pass_mode not in ANALYSIS_GRADE_ALLOWED_PASS_MODES:
        raise SystemExit(
            "--evidence_tier analysis_grade requires --pass_mode paired or quantized_only."
        )
    if pass_mode == PASS_MODE_QUANTIZED_ONLY and not str(baseline_reference_csv or "").strip():
        raise SystemExit(
            "--evidence_tier analysis_grade quantized_only requires --baseline_reference_csv."
        )
    if pass_mode == PASS_MODE_QUANTIZED_ONLY and not str(baseline_reference_run_id or "").strip():
        raise SystemExit(
            "--evidence_tier analysis_grade quantized_only requires --baseline_reference_run_id."
        )


def _load_weights_npz_manifest(path: str | Path | None) -> dict[str, str]:
    if path in ("", None):
        return {}
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
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


def _resolve_weights_npz(
    *,
    model_key: str,
    explicit_weights_npz: str | None,
    weights_npz_by_model: dict[str, str],
) -> str | None:
    if explicit_weights_npz:
        return explicit_weights_npz
    return weights_npz_by_model.get(model_key)


def _resolve_det_prefix_stats(
    sc_det_cfg: dict[str, Any],
    switches: dict[str, Any],
) -> tuple[str | None, str | None]:
    if not bool(switches.get("det")):
        return None, None
    early_stop = sc_det_cfg.get("early_stop") or {}
    if not bool(early_stop.get("enabled")):
        return None, None
    k_global = early_stop.get("k_global")
    if k_global is None:
        return None, None

    bsl_max = max(1, int(float(sc_det_cfg.get("bsl_max") or 1)))
    target_k = max(1, min(int(round(float(k_global))), bsl_max))
    raw_grid = early_stop.get("k_grid") or [target_k, bsl_max]
    k_grid = sorted(
        {
            max(1, min(int(round(float(k))), bsl_max))
            for k in raw_grid
            if k is not None
        }
    )
    if target_k not in k_grid:
        k_grid = sorted(set(k_grid + [target_k]))

    prefix_cfg = sc_det_cfg.get("prefix_error") or {}
    rows = compute_prefix_error_stats(
        bsl_max=bsl_max,
        k_grid=k_grid,
        num_prob_points=int(float(prefix_cfg.get("num_prob_points") or 129)),
        p_min=float(prefix_cfg.get("p_min") or 1e-3),
        p_max=float(prefix_cfg.get("p_max") or (1.0 - 1e-3)),
        det_mode=_resolve_det_mode(sc_det_cfg),
        phase_shift=int(float(prefix_cfg.get("phase_shift") or 0)),
        scramble_seed=int(float(prefix_cfg.get("scramble_seed") or 0)),
        # Legacy config snapshots may still carry enforce_monotonic=true, but
        # repaired DET-conditioned accuracy runs must follow the raw path.
        enforce_monotonic=False,
    )
    if not rows:
        return None, None
    selected = min(rows, key=lambda row: abs(float(row.get("k") or 0.0) - target_k))
    return _format_float(selected.get("prefix_error_mean")), _format_float(selected.get("prefix_error_p95"))


def _has_nontrivial_sparse_settings(*, sparse_cfg: dict[str, Any]) -> bool:
    if _sparse_tau_controls_gating(sparse_cfg):
        tau_value = _to_float(sparse_cfg.get("tau_global"))
        if tau_value is not None:
            return tau_value > 0.0
        tau_estimate = _estimate_sparse_scale_from_tau(sparse_cfg)
        return tau_estimate is not None and float(tau_estimate) < 1.0
    active_value = _format_float(sparse_cfg.get("active_fraction"))
    if active_value is not None and float(active_value) < 1.0:
        return True
    if "sparsity" in sparse_cfg:
        try:
            return float(sparse_cfg.get("sparsity") or 0.0) > 0.0
        except (TypeError, ValueError):
            return False
    return False


def _build_command(
    *,
    python_bin: str,
    accuracy_backend: str,
    cfg: dict[str, Any],
    cfg_path: Path,
    run_id: str,
    seed: int,
    results_csv: str,
    imagenet_val: str,
    models: str,
    weights_dir: str,
    workers: int,
    device: str,
    eval_batch_size: int | None,
    max_eval_samples: int | None,
    pass_mode: str = PASS_MODE_PAIRED,
    baseline_reference_csv: str | None = None,
    baseline_reference_run_id: str | None = None,
    split_manifest: str | None,
    append: bool,
    resume: bool,
    enable_sparse_accuracy_proxy: bool,
    weights_npz: str | None,
    mlx_weights_dir: str | None,
    progress_jsonl: str | None = None,
    progress_label: str | None = None,
    progress_heartbeat_interval_seconds: float | None = None,
    stall_timeout_seconds: float | None = None,
    pathological_min_samples_per_hour: float | None = None,
    pathological_max_seconds_per_sample: float | None = None,
    pathological_max_eta_current_rate_seconds: float | None = None,
    pathological_min_processed_samples: int | None = None,
    pathological_min_elapsed_seconds: float | None = None,
    bitstream_runtime_guardrail: dict[str, Any] | None = None,
    enable_bitstream_pilot: bool = False,
    bitstream_surface_scope: str | None = None,
    bitstream_target_module_keys: str | None = None,
    bitstream_generator: str | None = None,
    bitstream_stream_length: int | None = None,
    bitstream_stream_reuse_policy: str | None = None,
    bitstream_encoding_mode: str | None = None,
    bitstream_multiplier_mode: str | None = None,
    bitstream_accumulator_mode: str | None = None,
    bitstream_measurement_truth_class: str | None = None,
    bitstream_truth_class_authorization_note: str | None = None,
    bitstream_contract_note: str | None = None,
    accuracy_policy_profile_json: str | None = None,
) -> list[str]:
    run_cfg = cfg.get("run") or {}
    data_cfg = cfg.get("data") or {}
    sc_det_cfg = cfg.get("sc_det") or {}
    sparse_cfg = cfg.get("sparse") or {}
    noise_cfg = cfg.get("noise_injection") or {}
    switches = cfg.get("switches") or {}
    resolved_pathological_min_processed_samples = (
        _effective_pathological_min_processed_samples(
            accuracy_backend=accuracy_backend,
            resume=resume,
            enable_bitstream_pilot=enable_bitstream_pilot,
            bitstream_surface_scope=bitstream_surface_scope,
            pathological_min_processed_samples=pathological_min_processed_samples,
        )
    )
    resolved_runtime_health_gate = _effective_runtime_health_gate(
        accuracy_backend=accuracy_backend,
        enable_bitstream_pilot=enable_bitstream_pilot,
        bitstream_surface_scope=bitstream_surface_scope,
        model_key=models,
        pathological_min_samples_per_hour=pathological_min_samples_per_hour,
        pathological_max_seconds_per_sample=pathological_max_seconds_per_sample,
        pathological_max_eta_current_rate_seconds=pathological_max_eta_current_rate_seconds,
        pathological_min_elapsed_seconds=pathological_min_elapsed_seconds,
        bitstream_runtime_guardrail=bitstream_runtime_guardrail,
    )

    eval_run_id = f"{run_id}_acc_s{seed}"
    workload = str(data_cfg.get("workload_id") or data_cfg.get("workload") or "")

    cmd = [
        python_bin,
        TORCH_EVAL_SCRIPT if accuracy_backend == "torch" else MLX_EVAL_SCRIPT,
        "--imagenet_val",
        imagenet_val,
        "--opencv_pipeline",
        "--models",
        models,
        "--quant_bits",
        str(int((sc_det_cfg.get("quant_bits") or 8))),
        (
            "--gaussian_noise_sigma_lsb"
            if accuracy_backend == "torch"
            else "--noise_sigma_lsb"
        ),
        _format_float(_cfg_value(noise_cfg, "gaussian_noise_sigma_lsb", "sigma_lsb")) or "0",
        "--crosstalk_alpha",
        _format_float(noise_cfg.get("crosstalk_alpha")) or "0",
        "--drift_lsb",
        _format_float(noise_cfg.get("drift_lsb")) or "0",
        "--noise_correlation",
        _format_float(noise_cfg.get("noise_correlation")) or "0",
        "--burst_error_prob",
        _format_float(noise_cfg.get("burst_error_prob")) or "0",
        "--burst_error_scale_lsb",
        _format_float(noise_cfg.get("burst_error_scale_lsb")) or "0",
        "--burst_span",
        str(int(noise_cfg.get("burst_span") or 1)),
        "--enable_attention",
        "--seed",
        str(seed),
        "--device",
        device,
        "--results_csv",
        results_csv,
        "--run_id",
        eval_run_id,
        "--experiment_id",
        str(run_cfg.get("experiment_id") or ""),
        "--workload",
        workload,
        "--config_snapshot",
        str(cfg_path),
    ]
    if accuracy_backend == "torch":
        cmd.extend(["--workers", str(workers)])
    elif workers is not None:
        cmd.extend(["--workers", str(workers)])
    if weights_npz:
        cmd.extend(["--weights_npz", weights_npz])
    elif accuracy_backend == "mlx" and mlx_weights_dir:
        cmd.extend(["--mlx_weights_dir", mlx_weights_dir])
    elif weights_dir:
        cmd.extend(["--weights_dir", weights_dir])
    if split_manifest:
        cmd.extend(["--imagenet_manifest", split_manifest])
    if eval_batch_size is not None:
        cmd.extend(["--eval_batch_size", str(eval_batch_size)])
    if max_eval_samples is not None:
        cmd.extend(["--max_eval_samples", str(max_eval_samples)])
    cmd.extend(["--pass_mode", pass_mode])
    if baseline_reference_csv:
        cmd.extend(["--baseline_reference_csv", baseline_reference_csv])
    if baseline_reference_run_id:
        cmd.extend(["--baseline_reference_run_id", baseline_reference_run_id])
    if progress_jsonl:
        cmd.extend(["--progress_jsonl", progress_jsonl])
    if progress_label:
        cmd.extend(["--progress_label", progress_label])
    if accuracy_policy_profile_json:
        cmd.extend(["--accuracy_policy_profile_json", accuracy_policy_profile_json])
    if accuracy_backend == "mlx" and progress_heartbeat_interval_seconds is not None:
        cmd.extend(
            [
                "--progress_heartbeat_interval_seconds",
                str(float(progress_heartbeat_interval_seconds)),
            ]
        )
    if accuracy_backend == "mlx" and stall_timeout_seconds is not None:
        cmd.extend(["--stall_timeout_seconds", str(float(stall_timeout_seconds))])
    if (
        accuracy_backend == "mlx"
        and resolved_runtime_health_gate["pathological_min_samples_per_hour"] is not None
    ):
        cmd.extend(
            [
                "--pathological_min_samples_per_hour",
                str(
                    float(
                        resolved_runtime_health_gate[
                            "pathological_min_samples_per_hour"
                        ]
                    )
                ),
            ]
        )
    if (
        accuracy_backend == "mlx"
        and resolved_runtime_health_gate["pathological_max_seconds_per_sample"]
        is not None
    ):
        cmd.extend(
            [
                "--pathological_max_seconds_per_sample",
                str(
                    float(
                        resolved_runtime_health_gate[
                            "pathological_max_seconds_per_sample"
                        ]
                    )
                ),
            ]
        )
    if (
        accuracy_backend == "mlx"
        and resolved_runtime_health_gate[
            "pathological_max_eta_current_rate_seconds"
        ]
        is not None
    ):
        cmd.extend(
            [
                "--pathological_max_eta_current_rate_seconds",
                str(
                    float(
                        resolved_runtime_health_gate[
                            "pathological_max_eta_current_rate_seconds"
                        ]
                    )
                ),
            ]
        )
    if (
        accuracy_backend == "mlx"
        and resolved_pathological_min_processed_samples is not None
    ):
        cmd.extend(
            [
                "--pathological_min_processed_samples",
                str(int(resolved_pathological_min_processed_samples)),
            ]
        )
    if (
        accuracy_backend == "mlx"
        and resolved_runtime_health_gate["pathological_min_elapsed_seconds"]
        is not None
    ):
        cmd.extend(
            [
                "--pathological_min_elapsed_seconds",
                str(
                    float(
                        resolved_runtime_health_gate[
                            "pathological_min_elapsed_seconds"
                        ]
                    )
                ),
            ]
        )
    if accuracy_backend == "mlx" and enable_bitstream_pilot:
        cmd.append("--enable_bitstream_pilot")
        if bitstream_surface_scope:
            cmd.extend(["--bitstream_surface_scope", bitstream_surface_scope])
        if bitstream_target_module_keys:
            cmd.extend(["--bitstream_target_module_keys", bitstream_target_module_keys])
        if bitstream_generator:
            cmd.extend(["--bitstream_generator", bitstream_generator])
        if bitstream_stream_length is not None:
            cmd.extend(["--bitstream_stream_length", str(int(bitstream_stream_length))])
        if bitstream_stream_reuse_policy:
            cmd.extend(
                ["--bitstream_stream_reuse_policy", bitstream_stream_reuse_policy]
            )
        if bitstream_encoding_mode:
            cmd.extend(["--bitstream_encoding_mode", bitstream_encoding_mode])
        if bitstream_multiplier_mode:
            cmd.extend(["--bitstream_multiplier_mode", bitstream_multiplier_mode])
        if bitstream_accumulator_mode:
            cmd.extend(["--bitstream_accumulator_mode", bitstream_accumulator_mode])
        if bitstream_measurement_truth_class:
            cmd.extend(
                ["--bitstream_measurement_truth_class", bitstream_measurement_truth_class]
            )
        if bitstream_truth_class_authorization_note:
            cmd.extend(
                [
                    "--bitstream_truth_class_authorization_note",
                    bitstream_truth_class_authorization_note,
                ]
            )
        if bitstream_contract_note:
            cmd.extend(["--bitstream_contract_note", bitstream_contract_note])
    if append:
        cmd.append("--append")
    if resume:
        cmd.append("--resume")

    det_k = ((sc_det_cfg.get("early_stop") or {}).get("k_global"))
    det_runtime_metadata = resolve_det_runtime_metadata(sc_det_cfg, switches)
    det_runtime_enabled = bool(det_runtime_metadata.get("det_runtime_enabled"))
    det_prefix_mean = det_runtime_metadata.get("det_prefix_error_mean")
    det_prefix_p95 = det_runtime_metadata.get("det_prefix_error_p95")
    if det_runtime_enabled and (det_prefix_mean is None or det_prefix_p95 is None):
        fallback_mean, fallback_p95 = _resolve_det_prefix_stats(sc_det_cfg, switches)
        det_prefix_mean = det_prefix_mean if det_prefix_mean is not None else fallback_mean
        det_prefix_p95 = det_prefix_p95 if det_prefix_p95 is not None else fallback_p95
    if not det_runtime_enabled and bool(switches.get("det")):
        early_stop_cfg = sc_det_cfg.get("early_stop") or {}
        quality_gate_cfg = sc_det_cfg.get("quality_gate") or {}
        fallback_mean, fallback_p95 = _resolve_det_prefix_stats(sc_det_cfg, switches)
        if fallback_mean is not None and fallback_p95 is not None:
            measured_ready = (
                not bool(quality_gate_cfg.get("require_measured_accuracy"))
                or bool(quality_gate_cfg.get("measured_accuracy_ready"))
            )
            max_mean = _to_float(quality_gate_cfg.get("max_prefix_error_mean"))
            max_p95 = _to_float(quality_gate_cfg.get("max_prefix_error_p95"))
            gate_passed = (
                measured_ready
                and (max_mean is None or float(fallback_mean) <= max_mean)
                and (max_p95 is None or float(fallback_p95) <= max_p95)
            )
            det_runtime_enabled = bool(early_stop_cfg.get("enabled")) and gate_passed
            if det_runtime_enabled:
                det_prefix_mean = fallback_mean
                det_prefix_p95 = fallback_p95
    if bool(switches.get("det")) and det_k is not None:
        cmd.extend(["--det_k_global", str(det_k)])
    if det_runtime_enabled and det_prefix_mean is not None:
        cmd.extend(["--det_prefix_error_mean", det_prefix_mean])
    if det_runtime_enabled and det_prefix_p95 is not None:
        cmd.extend(["--det_prefix_error_p95", det_prefix_p95])
    if det_runtime_enabled:
        cmd.append("--apply_det_perturbation")
    tau = sparse_cfg.get("tau_global")
    if bool(switches.get("sparse")) and tau is not None:
        cmd.extend(["--sparse_tau_global", str(tau)])
    active_fraction = _resolve_sparse_active_fraction(sparse_cfg)
    if bool(switches.get("sparse")) and active_fraction is not None:
        cmd.extend(["--sparse_active_fraction", str(active_fraction)])
    if bool(switches.get("sparse")) and _has_nontrivial_sparse_settings(
        sparse_cfg=sparse_cfg,
    ):
        cmd.append("--apply_sparse_perturbation")
    return cmd


def _validate_preflight_args(args: argparse.Namespace) -> None:
    if (
        args.prepared_phase1_config_root or args.prepared_eligibility_report_root
    ) and not args.annotated_results_csv:
        raise SystemExit(
            "Prepared measured-source outputs require --annotated_results_csv."
        )
    if (
        args.bitstream_truth_class_authorization_root
        and args.bitstream_truth_class_authorization_note
    ):
        raise SystemExit(
            "Choose either --bitstream_truth_class_authorization_note or "
            "--bitstream_truth_class_authorization_root, not both."
        )
    if (
        args.bitstream_truth_class_authorization_root
        and args.bitstream_measurement_truth_class
        != BITSTREAM_MODEL_LEVEL_MEASURED_TRUTH_CLASS
    ):
        raise SystemExit(
            "--bitstream_truth_class_authorization_root is only valid when "
            "--bitstream_measurement_truth_class requests bitstream_model_level_measured."
        )
    _validate_evidence_tier_args(
        evidence_tier=str(args.evidence_tier),
        seeds=_parse_seeds(args.seeds),
        max_eval_samples=args.max_eval_samples,
        annotated_results_csv=args.annotated_results_csv,
        pass_mode=str(args.pass_mode),
        baseline_reference_csv=args.baseline_reference_csv,
        baseline_reference_run_id=args.baseline_reference_run_id,
    )
    if str(args.pass_mode) not in PASS_MODE_CHOICES:
        raise SystemExit(f"Unsupported --pass_mode={args.pass_mode!r}")
    if (
        str(args.pass_mode) == PASS_MODE_QUANTIZED_ONLY
        and not str(args.baseline_reference_csv or "").strip()
    ):
        raise SystemExit(
            "--pass_mode quantized_only requires --baseline_reference_csv."
        )
    if (
        str(args.pass_mode) == PASS_MODE_QUANTIZED_ONLY
        and not str(args.baseline_reference_run_id or "").strip()
    ):
        raise SystemExit(
            "--pass_mode quantized_only requires --baseline_reference_run_id."
        )
    normalized_device = str(args.device or "").strip().lower()
    if current_platform_name() == "darwin" and normalized_device != "mps":
        raise SystemExit(
            "Local config-conditioned accuracy runs on this Apple Silicon host must use --device mps."
        )
    if args.accuracy_backend == "mlx" and normalized_device != "mps":
        raise SystemExit("MLX config-conditioned accuracy runs require --device mps.")
    if int(args.prelaunch_runtime_smoke_samples) < 0:
        raise SystemExit("--prelaunch_runtime_smoke_samples must be >= 0.")
    if args.host_tuning_profile_json and not str(args.experiment_family_id or "").strip():
        raise SystemExit(
            "--host_tuning_profile_json requires --experiment_family_id."
        )


def _is_prep_only_manifest(path: Path) -> bool:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return bool(payload.get("prep_only"))


def _is_prep_only_progress_root(progress_root: Path, manifest_path: Path) -> bool:
    if not progress_root.exists() or not progress_root.is_dir():
        return False
    if not manifest_path.exists() or not _is_prep_only_manifest(manifest_path):
        return False
    files = [path for path in progress_root.rglob("*") if path.is_file()]
    return len(files) == 1 and files[0] == manifest_path


def _validate_fresh_output_paths(
    *,
    results_csv: str,
    annotated_results_csv: str | None,
    progress_root: Path,
    manifest_path: Path,
    prepared_phase1_config_root: str | None,
    prepared_eligibility_report_root: str | None,
    bitstream_truth_class_authorization_root: str | None,
    baseline_reference_summary_json: str | None = None,
    dry_run: bool,
    resume: bool,
) -> None:
    if dry_run or resume:
        return

    conflicts: list[str] = []

    def _record_conflict(label: str, path: Path, *, allow_prep_stub: bool = False) -> None:
        if not path.exists():
            return
        if allow_prep_stub and _is_prep_only_progress_root(path, manifest_path):
            return
        conflicts.append(f"{label}={path}")

    def _record_nonempty_dir(label: str, raw_path: str | None) -> None:
        if not str(raw_path or "").strip():
            return
        path = Path(str(raw_path))
        if not path.exists():
            return
        if path.is_file():
            conflicts.append(f"{label}={path}")
            return
        if any(path.iterdir()):
            conflicts.append(f"{label}={path}")

    _record_conflict("results_csv", Path(results_csv))
    if str(annotated_results_csv or "").strip():
        _record_conflict("annotated_results_csv", Path(str(annotated_results_csv)))
    _record_conflict("progress_root", progress_root, allow_prep_stub=True)
    if manifest_path.exists() and not _is_prep_only_manifest(manifest_path):
        conflicts.append(f"progress_manifest_json={manifest_path}")
    _record_nonempty_dir("prepared_phase1_config_root", prepared_phase1_config_root)
    _record_nonempty_dir(
        "prepared_eligibility_report_root",
        prepared_eligibility_report_root,
    )
    _record_nonempty_dir(
        "bitstream_truth_class_authorization_root",
        bitstream_truth_class_authorization_root,
    )
    if str(baseline_reference_summary_json or "").strip():
        _record_conflict(
            "baseline_reference_summary_json",
            Path(str(baseline_reference_summary_json)),
        )

    if conflicts:
        joined = "; ".join(conflicts)
        raise SystemExit(
            "Refusing to start a fresh non-resume run because output paths already exist: "
            f"{joined}. Re-run with --resume or clear/redirect these outputs."
        )


def main() -> None:
    global _ACTIVE_CHILD, _STOP_REQUESTED
    _ACTIVE_CHILD = None
    _STOP_REQUESTED = False
    parser = build_parser()
    args = parser.parse_args()
    _install_interrupt_handlers()
    _validate_preflight_args(args)

    run_ids = [item.strip() for item in args.run_ids.split(",") if item.strip()]
    seeds = _parse_seeds(args.seeds)
    weights_npz_by_model = _load_weights_npz_manifest(args.weights_npz_manifest)
    model_eval_batch_sizes = _parse_model_int_overrides(
        args.model_eval_batch_sizes,
        field_name="--model_eval_batch_sizes",
    )
    if not run_ids:
        raise SystemExit("--run_ids produced an empty list.")
    if not seeds:
        raise SystemExit("--seeds produced an empty list.")
    analysis_grade_ready, analysis_grade_blockers = _resolve_analysis_grade_status(
        evidence_tier=str(args.evidence_tier),
        seeds=seeds,
        max_eval_samples=args.max_eval_samples,
        annotated_results_csv=args.annotated_results_csv,
        pass_mode=str(args.pass_mode),
        baseline_reference_csv=args.baseline_reference_csv,
        baseline_reference_run_id=args.baseline_reference_run_id,
    )
    progress_root = Path(args.progress_root) if args.progress_root else _default_progress_root(args.results_csv)
    manifest_path = _progress_manifest_path(
        progress_root,
        explicit_manifest=args.progress_manifest_json,
    )
    _validate_fresh_output_paths(
        results_csv=args.results_csv,
        annotated_results_csv=args.annotated_results_csv,
        progress_root=progress_root,
        manifest_path=manifest_path,
        prepared_phase1_config_root=args.prepared_phase1_config_root,
        prepared_eligibility_report_root=args.prepared_eligibility_report_root,
        bitstream_truth_class_authorization_root=args.bitstream_truth_class_authorization_root,
        baseline_reference_summary_json=args.baseline_reference_summary_json,
        dry_run=bool(args.dry_run),
        resume=bool(args.resume),
    )
    progress_root.mkdir(parents=True, exist_ok=True)
    host_tuning_payload = (
        load_host_tuning_profile(Path(str(args.host_tuning_profile_json)))
        if str(args.host_tuning_profile_json or "").strip()
        else None
    )

    planned_jobs: list[dict[str, object]] = []
    first_cmd = True
    for run_id in run_ids:
        cfg_path = RUNS_DIR / run_id / "config_snapshot.yaml"
        if not cfg_path.is_file():
            raise SystemExit(f"Missing config snapshot: {cfg_path}")
        cfg = _load_yaml(cfg_path)
        data_cfg = cfg.get("data") or {}
        run_cfg = cfg.get("run") or {}
        split = str(data_cfg.get("split") or "eval")
        manifest = _resolve_manifest(split, data_cfg, args.manifest_override)
        model_keys = _parse_model_keys(args.models)
        if args.accuracy_backend == "mlx":
            if not model_keys:
                raise SystemExit("MLX config-conditioned runs require at least one model key.")
            if args.weights_npz and len(model_keys) > 1:
                raise SystemExit("--weights_npz can only be used with one MLX model at a time.")
        else:
            model_keys = []

        for seed in seeds:
            per_seed_models = model_keys or [args.models]
            for model_key in per_seed_models:
                weights_npz = None
                model_arg = args.models
                resolved_eval_batch_size = _resolve_model_eval_batch_size(
                    model_key=model_key,
                    default_eval_batch_size=args.eval_batch_size,
                    model_eval_batch_sizes=model_eval_batch_sizes,
                )
                if args.accuracy_backend == "mlx":
                    model_arg = model_key
                    weights_npz = _resolve_weights_npz(
                        model_key=model_key,
                        explicit_weights_npz=args.weights_npz,
                        weights_npz_by_model=weights_npz_by_model,
                    )
                quantized_execution_semantics = _resolved_quantized_execution_semantics(
                    cfg=cfg,
                    enable_bitstream_pilot=bool(args.enable_bitstream_pilot),
                )
                accuracy_policy_bundle = (
                    resolve_accuracy_policy_bundle(
                        host_tuning_payload,
                        experiment_family_id=_normalize_optional_token(
                            args.experiment_family_id
                        ),
                        lane_id=_normalize_optional_token(args.lane_id),
                        pass_mode=str(args.pass_mode),
                        quantized_execution_semantics=quantized_execution_semantics,
                        host_id=_normalize_optional_token(args.host_id),
                    )
                    if host_tuning_payload is not None
                    else None
                )
                quantized_pass_policy = (
                    dict(
                        (accuracy_policy_bundle.get("pass_policies") or {}).get(
                            PASS_KIND_QUANTIZED_EVAL
                        )
                        or {}
                    )
                    if accuracy_policy_bundle is not None
                    else {}
                )
                if quantized_pass_policy.get("eval_batch_size") not in (None, ""):
                    resolved_eval_batch_size = int(
                        quantized_pass_policy["eval_batch_size"]
                    )
                bitstream_runtime_guardrail = _validate_bitstream_runtime_guardrail(
                    model_key=model_key,
                    accuracy_backend=args.accuracy_backend,
                    enable_bitstream_pilot=bool(args.enable_bitstream_pilot),
                    surface_scope=args.bitstream_surface_scope,
                    target_module_keys=args.bitstream_target_module_keys,
                    eval_batch_size=resolved_eval_batch_size,
                    stream_length=args.bitstream_stream_length,
                    allow_unsafe_runtime_shapes=bool(
                        args.allow_unsafe_bitstream_runtime_shapes
                    ),
                )
                progress_jsonl = _progress_jsonl_path(
                    progress_root,
                    run_id=run_id,
                    model_key=model_key,
                    seed=seed,
                )
                progress_label = (
                    f"{run_id}:{model_key}:seed{seed}"
                    if args.accuracy_backend == "mlx"
                    else f"{run_id}:{seed}"
                )
                eval_run_id = f"{run_id}_acc_s{seed}"
                accuracy_policy_profile_json = ""
                if accuracy_policy_bundle is not None:
                    accuracy_policy_path = _staged_accuracy_policy_profile_path(
                        progress_root,
                        eval_run_id=eval_run_id,
                    )
                    accuracy_policy_path.parent.mkdir(parents=True, exist_ok=True)
                    accuracy_policy_path.write_text(
                        json.dumps(
                            accuracy_policy_bundle,
                            indent=2,
                            ensure_ascii=False,
                            sort_keys=True,
                        )
                        + "\n",
                        encoding="utf-8",
                    )
                    accuracy_policy_profile_json = str(accuracy_policy_path)
                eval_config_snapshot = _eval_config_snapshot_path(eval_run_id=eval_run_id)
                if not args.dry_run:
                    eval_config_snapshot = _stage_eval_config_snapshot(
                        cfg=cfg,
                        eval_run_id=eval_run_id,
                        enable_bitstream_pilot=bool(args.enable_bitstream_pilot),
                        bitstream_generator=args.bitstream_generator,
                        bitstream_stream_length=args.bitstream_stream_length,
                        bitstream_encoding_mode=args.bitstream_encoding_mode,
                        bitstream_multiplier_mode=args.bitstream_multiplier_mode,
                        bitstream_accumulator_mode=args.bitstream_accumulator_mode,
                    )
                authorization_note = args.bitstream_truth_class_authorization_note
                if (
                    args.bitstream_measurement_truth_class
                    == BITSTREAM_MODEL_LEVEL_MEASURED_TRUTH_CLASS
                    and args.bitstream_truth_class_authorization_root
                ):
                    authorization_note = str(
                        _authorization_note_path(
                            Path(args.bitstream_truth_class_authorization_root),
                            eval_run_id=eval_run_id,
                        )
                    )
                cmd = _build_command(
                    python_bin=args.python_bin,
                    accuracy_backend=args.accuracy_backend,
                    cfg=cfg,
                    cfg_path=eval_config_snapshot,
                    run_id=run_id,
                    seed=seed,
                    results_csv=args.results_csv,
                    imagenet_val=args.imagenet_val,
                    models=model_arg,
                    weights_dir=args.weights_dir,
                    workers=(
                        int(quantized_pass_policy.get("workers") or args.workers)
                        if quantized_pass_policy
                        else args.workers
                    ),
                    device=args.device,
                    eval_batch_size=resolved_eval_batch_size,
                    max_eval_samples=args.max_eval_samples,
                    pass_mode=str(args.pass_mode),
                    baseline_reference_csv=args.baseline_reference_csv,
                    baseline_reference_run_id=args.baseline_reference_run_id,
                    split_manifest=manifest,
                    append=(not first_cmd) or (bool(args.resume) and Path(args.results_csv).exists()),
                    resume=bool(args.resume),
                    enable_sparse_accuracy_proxy=bool(args.enable_sparse_accuracy_proxy),
                    weights_npz=weights_npz,
                    mlx_weights_dir=args.mlx_weights_dir,
                    progress_jsonl=str(progress_jsonl),
                    progress_label=progress_label,
                    progress_heartbeat_interval_seconds=args.progress_heartbeat_interval_seconds,
                    stall_timeout_seconds=args.stall_timeout_seconds,
                    pathological_min_samples_per_hour=args.pathological_min_samples_per_hour,
                    pathological_max_seconds_per_sample=args.pathological_max_seconds_per_sample,
                    pathological_max_eta_current_rate_seconds=args.pathological_max_eta_current_rate_seconds,
                    pathological_min_processed_samples=args.pathological_min_processed_samples,
                    pathological_min_elapsed_seconds=args.pathological_min_elapsed_seconds,
                    bitstream_runtime_guardrail=bitstream_runtime_guardrail,
                    enable_bitstream_pilot=bool(args.enable_bitstream_pilot),
                    bitstream_surface_scope=args.bitstream_surface_scope,
                    bitstream_target_module_keys=args.bitstream_target_module_keys,
                    bitstream_generator=args.bitstream_generator,
                    bitstream_stream_length=args.bitstream_stream_length,
                    bitstream_stream_reuse_policy=args.bitstream_stream_reuse_policy,
                    bitstream_encoding_mode=args.bitstream_encoding_mode,
                    bitstream_multiplier_mode=args.bitstream_multiplier_mode,
                    bitstream_accumulator_mode=args.bitstream_accumulator_mode,
                    bitstream_measurement_truth_class=args.bitstream_measurement_truth_class,
                    bitstream_truth_class_authorization_note=authorization_note,
                    bitstream_contract_note=args.bitstream_contract_note,
                    accuracy_policy_profile_json=accuracy_policy_profile_json,
                )
                first_cmd = False
                prepared_phase1_config = ""
                if args.prepared_phase1_config_root:
                    prepared_phase1_config = str(
                        _prepared_phase1_config_path(
                            Path(args.prepared_phase1_config_root),
                            eval_run_id=eval_run_id,
                        )
                    )
                eligibility_report_json = ""
                eligibility_report_md = ""
                if args.prepared_eligibility_report_root:
                    eligibility_json_path, eligibility_md_path = (
                        _prepared_eligibility_report_paths(
                            Path(args.prepared_eligibility_report_root),
                            eval_run_id=eval_run_id,
                        )
                    )
                    eligibility_report_json = str(eligibility_json_path)
                    eligibility_report_md = str(eligibility_md_path)
                resolved_pathological_min_processed_samples = (
                    _effective_pathological_min_processed_samples(
                        accuracy_backend=str(args.accuracy_backend),
                        resume=bool(args.resume),
                        enable_bitstream_pilot=bool(args.enable_bitstream_pilot),
                        bitstream_surface_scope=args.bitstream_surface_scope,
                        pathological_min_processed_samples=args.pathological_min_processed_samples,
                    )
                )
                resolved_runtime_health_gate = _effective_runtime_health_gate(
                    accuracy_backend=str(args.accuracy_backend),
                    enable_bitstream_pilot=bool(args.enable_bitstream_pilot),
                    bitstream_surface_scope=args.bitstream_surface_scope,
                    model_key=model_arg,
                    pathological_min_samples_per_hour=args.pathological_min_samples_per_hour,
                    pathological_max_seconds_per_sample=args.pathological_max_seconds_per_sample,
                    pathological_max_eta_current_rate_seconds=args.pathological_max_eta_current_rate_seconds,
                    pathological_min_elapsed_seconds=args.pathological_min_elapsed_seconds,
                    bitstream_runtime_guardrail=bitstream_runtime_guardrail,
                )
                pass_kind_profile = (
                    {
                        pass_kind: str(policy.get("profile_id") or "")
                        for pass_kind, policy in (
                            accuracy_policy_bundle.get("pass_policies") or {}
                        ).items()
                    }
                    if accuracy_policy_bundle is not None
                    else {}
                )
                runtime_policy_fingerprint = (
                    {
                        pass_kind: str(
                            policy.get("runtime_policy_fingerprint") or ""
                        )
                        for pass_kind, policy in (
                            accuracy_policy_bundle.get("pass_policies") or {}
                        ).items()
                    }
                    if accuracy_policy_bundle is not None
                    else {}
                )
                semantic_fingerprint = (
                    {
                        pass_kind: str(policy.get("semantic_fingerprint") or "")
                        for pass_kind, policy in (
                            accuracy_policy_bundle.get("pass_policies") or {}
                        ).items()
                    }
                    if accuracy_policy_bundle is not None
                    else {}
                )
                runtime_health_gate_payload: dict[str, Any] = {
                    "command_default": {
                        "progress_heartbeat_interval_seconds": float(
                            args.progress_heartbeat_interval_seconds
                        ),
                        "stall_timeout_seconds": float(args.stall_timeout_seconds),
                        "pathological_min_samples_per_hour": (
                            float(
                                resolved_runtime_health_gate[
                                    "pathological_min_samples_per_hour"
                                ]
                            )
                            if resolved_runtime_health_gate[
                                "pathological_min_samples_per_hour"
                            ]
                            is not None
                            else None
                        ),
                        "pathological_max_seconds_per_sample": (
                            float(
                                resolved_runtime_health_gate[
                                    "pathological_max_seconds_per_sample"
                                ]
                            )
                            if resolved_runtime_health_gate[
                                "pathological_max_seconds_per_sample"
                            ]
                            is not None
                            else None
                        ),
                        "pathological_max_eta_current_rate_seconds": (
                            float(
                                resolved_runtime_health_gate[
                                    "pathological_max_eta_current_rate_seconds"
                                ]
                            )
                            if resolved_runtime_health_gate[
                                "pathological_max_eta_current_rate_seconds"
                            ]
                            is not None
                            else None
                        ),
                        "pathological_min_processed_samples": (
                            int(resolved_pathological_min_processed_samples)
                            if resolved_pathological_min_processed_samples
                            is not None
                            else None
                        ),
                        "pathological_min_elapsed_seconds": (
                            float(
                                resolved_runtime_health_gate[
                                    "pathological_min_elapsed_seconds"
                                ]
                            )
                            if resolved_runtime_health_gate[
                                "pathological_min_elapsed_seconds"
                            ]
                            is not None
                            else None
                        ),
                    }
                }
                if accuracy_policy_bundle is not None:
                    runtime_health_gate_payload["pass_policies"] = {
                        pass_kind: dict(policy.get("runtime_health_gate") or {})
                        for pass_kind, policy in (
                            accuracy_policy_bundle.get("pass_policies") or {}
                        ).items()
                    }
                planned_jobs.append(
                    {
                        "family_group": "config_accuracy",
                        "step_id": f"{run_id}_{model_key}_s{seed}",
                        "run_id": run_id,
                        "eval_run_id": eval_run_id,
                        "config_path": str(cfg_path),
                        "eval_config_snapshot": str(eval_config_snapshot),
                        "experiment_id": str(run_cfg.get("experiment_id") or ""),
                        "experiment_family_id": str(args.experiment_family_id or ""),
                        "lane_id": str(args.lane_id or ""),
                        "model": model_key,
                        "profile": str(run_cfg.get("experiment_id") or ""),
                        "sweep_resolution": "config_conditioned",
                        "eval_batch_size": resolved_eval_batch_size,
                        "command": _quote_command(cmd),
                        "commands": [cmd],
                        "progress_jsonls": [str(progress_jsonl)],
                        "accuracy_policy_profile_json": accuracy_policy_profile_json,
                        "host_profile_id": str(
                            (accuracy_policy_bundle or {}).get("host_profile_id") or ""
                        ),
                        "pass_kind_profile": pass_kind_profile,
                        "runtime_policy_fingerprint": runtime_policy_fingerprint,
                        "semantic_fingerprint": semantic_fingerprint,
                        "calibration_artifact_path": str(
                            (accuracy_policy_bundle or {}).get("calibration_artifact_path")
                            or ""
                        ),
                        "bitstream_truth_class_authorization_note": authorization_note or "",
                        "prepared_phase1_config": prepared_phase1_config,
                        "prepared_eligibility_report_json": eligibility_report_json,
                        "prepared_eligibility_report_md": eligibility_report_md,
                        "accuracy_evidence_tier": str(args.evidence_tier),
                        "analysis_grade_ready": analysis_grade_ready,
                        "analysis_grade_blockers": list(analysis_grade_blockers),
                        "pass_mode": str(args.pass_mode),
                        "baseline_reference_csv": str(args.baseline_reference_csv or ""),
                        "baseline_reference_run_id": str(args.baseline_reference_run_id or ""),
                        "baseline_reference_summary_json": str(args.baseline_reference_summary_json or ""),
                        "planned_pass_count_per_command": [_resolve_planned_pass_count(str(args.pass_mode))],
                        "planned_pass_count": _resolve_planned_pass_count(str(args.pass_mode)),
                        "prelaunch_runtime_smoke": {
                            "enabled": _should_enable_prelaunch_runtime_smoke(
                                accuracy_backend=str(args.accuracy_backend),
                                resume=bool(args.resume),
                                pass_mode=str(args.pass_mode),
                                enable_bitstream_pilot=bool(
                                    args.enable_bitstream_pilot
                                ),
                                smoke_samples=int(
                                    args.prelaunch_runtime_smoke_samples
                                ),
                                bitstream_runtime_guardrail=bitstream_runtime_guardrail,
                            ),
                            "samples": int(args.prelaunch_runtime_smoke_samples),
                            "min_samples_per_hour": float(args.prelaunch_min_samples_per_hour),
                            "max_seconds_per_sample": float(args.prelaunch_max_seconds_per_sample),
                            "summary_json": str(
                                _prelaunch_smoke_root(
                                    progress_root,
                                    eval_run_id=eval_run_id,
                                )
                                / "summary.json"
                            ),
                        },
                        "runtime_health_gate": runtime_health_gate_payload,
                        "bitstream_runtime_guardrail": (
                            dict(bitstream_runtime_guardrail)
                            if bitstream_runtime_guardrail is not None
                            else None
                        ),
                    }
                )

    if (
        args.bitstream_measurement_truth_class == BITSTREAM_MODEL_LEVEL_MEASURED_TRUTH_CLASS
        and not args.bitstream_truth_class_authorization_root
        and not args.bitstream_truth_class_authorization_note
    ):
        raise SystemExit(
            "bitstream_model_level_measured requires either "
            "--bitstream_truth_class_authorization_note or "
            "--bitstream_truth_class_authorization_root."
        )
    if (
        args.bitstream_measurement_truth_class == BITSTREAM_MODEL_LEVEL_MEASURED_TRUTH_CLASS
        and args.bitstream_truth_class_authorization_note
        and len(planned_jobs) > 1
    ):
        raise SystemExit(
            "A single --bitstream_truth_class_authorization_note cannot authorize "
            "multiple eval_run_id values. Use --bitstream_truth_class_authorization_root "
            "for multi-job measured launches."
        )

    manifest_payload = {
        "run_tag": Path(args.results_csv).stem,
        "results_csv": args.results_csv,
        "annotated_results_csv": args.annotated_results_csv,
        "accuracy_evidence_tier": str(args.evidence_tier),
        "host_tuning_profile_json": str(args.host_tuning_profile_json or ""),
        "host_id": str(args.host_id or ""),
        "experiment_family_id": str(args.experiment_family_id or ""),
        "lane_id": str(args.lane_id or ""),
        "analysis_grade_ready": analysis_grade_ready,
        "analysis_grade_blockers": analysis_grade_blockers,
        "annotation_measurement_truth_class": args.annotation_measurement_truth_class,
        "pass_mode": args.pass_mode,
        "baseline_reference_csv": args.baseline_reference_csv,
        "baseline_reference_run_id": args.baseline_reference_run_id,
        "baseline_reference_summary_json": args.baseline_reference_summary_json,
        "prepared_phase1_config_root": args.prepared_phase1_config_root,
        "prepared_eligibility_report_root": args.prepared_eligibility_report_root,
        "bitstream_truth_class_authorization_root": (
            args.bitstream_truth_class_authorization_root
        ),
        "progress_root": str(progress_root),
        "jobs": planned_jobs,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"[progress] manifest={manifest_path}", flush=True)

    for job in planned_jobs:
        if _STOP_REQUESTED:
            raise SystemExit(130)
        command = list(job["commands"][0])
        print("[config-accuracy]", _quote_command(command), flush=True)
        if args.dry_run:
            continue
        authorization_note = str(
            job.get("bitstream_truth_class_authorization_note") or ""
        ).strip()
        if (
            args.bitstream_measurement_truth_class
            == BITSTREAM_MODEL_LEVEL_MEASURED_TRUTH_CLASS
            and args.bitstream_truth_class_authorization_root
            and authorization_note
        ):
            note_path = write_truth_class_authorization_note(
                Path(authorization_note),
                authorized_run_id=str(job["eval_run_id"]),
                extra_fields={
                    "source_phase1_config": str(job["config_path"]),
                    "source_run_id": str(job["run_id"]),
                    "generated_by": "run_config_conditioned_accuracy_matrix",
                    "accuracy_evidence_tier": str(job["accuracy_evidence_tier"]),
                    "analysis_grade_ready": str(bool(job["analysis_grade_ready"])).lower(),
                    "analysis_grade_blockers": ",".join(
                        str(item) for item in (job.get("analysis_grade_blockers") or [])
                    ),
                },
            )
            print(
                f"[config-accuracy] wrote authorization note for {job['eval_run_id']} -> {note_path}",
                flush=True,
            )
        smoke_policy = job.get("prelaunch_runtime_smoke") or {}
        if bool(smoke_policy.get("enabled")):
            _run_prelaunch_runtime_smoke(
                command=command,
                eval_run_id=str(job["eval_run_id"]),
                progress_root=progress_root,
                smoke_samples=int(smoke_policy.get("samples") or 0),
                min_samples_per_hour=float(smoke_policy.get("min_samples_per_hour") or 0.0),
                max_seconds_per_sample=float(smoke_policy.get("max_seconds_per_sample") or 0.0),
            )
        try:
            proc = subprocess.Popen(command, cwd=ROOT_DIR, text=True)
            _ACTIVE_CHILD = proc
            returncode = proc.wait()
            _ACTIVE_CHILD = None
        except KeyboardInterrupt:
            raise SystemExit(130)
        if returncode == 130:
            raise SystemExit(130)
        if returncode != 0:
            raise subprocess.CalledProcessError(returncode, command)
        annotated = _maybe_annotate_bitstream_results(
            cfg_path=Path(str(job.get("eval_config_snapshot") or job["config_path"])),
            eval_run_id=str(job["eval_run_id"]),
            raw_results_csv=Path(args.results_csv),
            annotated_results_csv=(
                Path(args.annotated_results_csv) if args.annotated_results_csv else None
            ),
            pass_mode=str(args.pass_mode),
            contract_note=str(args.annotation_contract_note or ""),
            measurement_truth_class=str(args.annotation_measurement_truth_class),
            extra_annotation_fields={
                "accuracy_evidence_tier": str(job["accuracy_evidence_tier"]),
                "analysis_grade_ready": str(bool(job["analysis_grade_ready"])).lower(),
                "analysis_grade_blockers": json.dumps(
                    list(job.get("analysis_grade_blockers") or []),
                    ensure_ascii=False,
                ),
            },
            output_config=(
                Path(str(job["prepared_phase1_config"]))
                if str(job.get("prepared_phase1_config") or "").strip()
                else None
            ),
            eligibility_report_json=(
                Path(str(job["prepared_eligibility_report_json"]))
                if str(job.get("prepared_eligibility_report_json") or "").strip()
                else None
            ),
            eligibility_report_md=(
                Path(str(job["prepared_eligibility_report_md"]))
                if str(job.get("prepared_eligibility_report_md") or "").strip()
                else None
            ),
        )
        if annotated:
            print(
                f"[config-accuracy] annotated bitstream-measured rows for {job['eval_run_id']} -> {args.annotated_results_csv}",
                flush=True,
            )
            if annotated.get("output_config"):
                print(
                    f"[config-accuracy] prepared phase1 config for {job['eval_run_id']} -> {annotated['output_config']}",
                    flush=True,
                )
            if annotated.get("eligibility_report_json") or annotated.get(
                "eligibility_report_md"
            ):
                print(
                    f"[config-accuracy] wrote measured-row eligibility reports for {job['eval_run_id']}",
                    flush=True,
                )
        if (
            str(args.baseline_reference_summary_json or "").strip()
            and str(args.pass_mode) in {PASS_MODE_PAIRED, PASS_MODE_BASELINE_ONLY}
        ):
            summary_payload = _write_baseline_reference_summary(
                raw_results_csv=Path(args.results_csv),
                eval_run_id=str(job["eval_run_id"]),
                output_json=Path(str(args.baseline_reference_summary_json)),
            )
            print(
                "[config-accuracy] wrote baseline reference summary for {eval_run_id} -> {path}".format(
                    eval_run_id=job["eval_run_id"],
                    path=args.baseline_reference_summary_json,
                ),
                flush=True,
            )
            if summary_payload.get("top1") or summary_payload.get("top5"):
                print(
                    "[config-accuracy] baseline reference top1={top1} top5={top5}".format(
                        top1=summary_payload.get("top1") or "n/a",
                        top5=summary_payload.get("top5") or "n/a",
                    ),
                    flush=True,
                )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run config-conditioned accuracy evaluations from saved run configs.")
    parser.add_argument(
        "--run_ids",
        required=True,
        help="Comma-separated saved Phase-1 run IDs under experiments/results/runs/.",
    )
    parser.add_argument(
        "--results_csv",
        required=True,
        help="Output accuracy CSV. Use a fresh file for a clean batch.",
    )
    parser.add_argument(
        "--annotated_results_csv",
        default=None,
        help="Optional annotated CSV mirror for bitstream-effective run configs. Rows are postprocessed after each successful eval subprocess and default to a non-promotable bridge truth class.",
    )
    parser.add_argument(
        "--annotation_measurement_truth_class",
        default=BITSTREAM_BRIDGE_MEASUREMENT_TRUTH_CLASS,
        help="Truth class stamped onto --annotated_results_csv rows during postprocess. Defaults to the non-promotable bridge class; set this explicitly for governed measured-source preparation.",
    )
    parser.add_argument(
        "--prepared_phase1_config_root",
        default=None,
        help="Optional directory for per-eval prepared phase1 config copies derived from --annotated_results_csv.",
    )
    parser.add_argument(
        "--prepared_eligibility_report_root",
        default=None,
        help="Optional directory for per-eval measured-row eligibility JSON/Markdown reports derived from --annotated_results_csv.",
    )
    parser.add_argument(
        "--imagenet_val",
        default="experiments/datasets/imagenet/val",
        help="ImageNet val root.",
    )
    parser.add_argument(
        "--models",
        default="mobilevit_xxs,mobilevit_xs,mobilevit_s",
        help="Comma-separated model keys.",
    )
    parser.add_argument(
        "--accuracy_backend",
        choices=["torch", "mlx"],
        default="torch",
        help="Accuracy evaluation backend.",
    )
    parser.add_argument(
        "--weights_dir",
        default="weights",
        help="Weights directory.",
    )
    parser.add_argument(
        "--weights_npz",
        default=None,
        help="Explicit MLX weights NPZ. Only valid when exactly one MLX model is requested.",
    )
    parser.add_argument(
        "--weights_npz_manifest",
        default=None,
        help="Optional JSON manifest that maps model keys to explicit MLX NPZ files.",
    )
    parser.add_argument(
        "--mlx_weights_dir",
        default=None,
        help="Optional MLX weights cache dir used only when explicit NPZ paths are not supplied.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda", "mps"],
        default="mps",
        help="Device passed through to the accuracy evaluator. Local Apple Silicon runs must use mps.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Dataloader workers.",
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=None,
        help="Optional eval batch size override forwarded to the accuracy runner.",
    )
    parser.add_argument(
        "--model_eval_batch_sizes",
        default=None,
        help="Optional comma-separated MLX per-model batch overrides like mobilevit_xxs=64,mobilevit_xs=48.",
    )
    parser.add_argument(
        "--max_eval_samples",
        type=int,
        default=None,
        help="Optional smoke-test limit on evaluated samples.",
    )
    parser.add_argument(
        "--pass_mode",
        choices=PASS_MODE_CHOICES,
        default=PASS_MODE_PAIRED,
        help=(
            "Evaluation pass layout. paired runs baseline+quantized, "
            "baseline_only produces the reusable baseline cache, and "
            "quantized_only reuses a supplied baseline reference without re-running baseline."
        ),
    )
    parser.add_argument(
        "--baseline_reference_csv",
        default=None,
        help="Reference accuracy CSV that contains the baseline row used by --pass_mode quantized_only.",
    )
    parser.add_argument(
        "--baseline_reference_run_id",
        default=None,
        help="Baseline run_id to bind against --baseline_reference_csv when --pass_mode quantized_only is used.",
    )
    parser.add_argument(
        "--baseline_reference_summary_json",
        default=None,
        help="Optional JSON path written after paired/baseline_only runs to describe the reusable baseline cache row.",
    )
    parser.add_argument(
        "--host_tuning_profile_json",
        default=None,
        help=(
            "Optional adaptive host-tuning profile artifact used to resolve per-pass "
            "workers, batch size, runtime health gates, and semantic policy."
        ),
    )
    parser.add_argument(
        "--host_id",
        default=None,
        help="Optional host_id override when selecting a host profile from --host_tuning_profile_json.",
    )
    parser.add_argument(
        "--experiment_family_id",
        default=None,
        help="Optional experiment family context used with --host_tuning_profile_json.",
    )
    parser.add_argument(
        "--lane_id",
        default=None,
        help="Optional lane context used with --host_tuning_profile_json.",
    )
    parser.add_argument(
        "--seeds",
        default="0",
        help="Comma-separated seed list.",
    )
    parser.add_argument(
        "--evidence_tier",
        choices=[RUNTIME_SMOKE_TIER, ANALYSIS_GRADE_TIER],
        default=RUNTIME_SMOKE_TIER,
        help="Governance evidence tier for this launch.",
    )
    parser.add_argument(
        "--python_bin",
        default=sys.executable,
        help="Python interpreter for the eval subprocess.",
    )
    parser.add_argument(
        "--manifest_override",
        default=None,
        help="Optional manifest CSV override for every run.",
    )
    parser.add_argument(
        "--progress_root",
        default=None,
        help="Optional directory for per-command progress JSONL files and the ETA manifest.",
    )
    parser.add_argument(
        "--progress_manifest_json",
        default=None,
        help="Optional explicit manifest path consumed by monitor_fuller_eta.py.",
    )
    parser.add_argument(
        "--progress_heartbeat_interval_seconds",
        type=float,
        default=DEFAULT_PROGRESS_HEARTBEAT_INTERVAL_SECONDS,
        help="Progress JSONL heartbeat cadence forwarded to the MLX evaluator.",
    )
    parser.add_argument(
        "--stall_timeout_seconds",
        type=float,
        default=DEFAULT_STALL_TIMEOUT_SECONDS,
        help="Fail a pass if no batch completes within this many seconds.",
    )
    parser.add_argument(
        "--prelaunch_runtime_smoke_samples",
        type=int,
        default=DEFAULT_PRELAUNCH_RUNTIME_SMOKE_SAMPLES,
        help="Bounded MLX smoke samples run before the full experiment command starts. Use 0 to disable.",
    )
    parser.add_argument(
        "--prelaunch_min_samples_per_hour",
        type=float,
        default=DEFAULT_PRELAUNCH_MIN_SAMPLES_PER_HOUR,
        help="Minimum bounded-smoke throughput required before the full MLX run starts.",
    )
    parser.add_argument(
        "--prelaunch_max_seconds_per_sample",
        type=float,
        default=DEFAULT_PRELAUNCH_MAX_SECONDS_PER_SAMPLE,
        help="Maximum bounded-smoke seconds/sample allowed before the full MLX run starts.",
    )
    parser.add_argument(
        "--pathological_min_samples_per_hour",
        type=float,
        default=DEFAULT_RUNTIME_MIN_SAMPLES_PER_HOUR,
        help="Runtime self-stop threshold: abort MLX passes that fall below this throughput.",
    )
    parser.add_argument(
        "--pathological_max_seconds_per_sample",
        type=float,
        default=DEFAULT_RUNTIME_MAX_SECONDS_PER_SAMPLE,
        help="Runtime self-stop threshold: abort MLX passes that exceed this seconds/sample rate.",
    )
    parser.add_argument(
        "--pathological_max_eta_current_rate_seconds",
        type=float,
        default=DEFAULT_RUNTIME_MAX_ETA_CURRENT_RATE_SECONDS,
        help="Runtime self-stop threshold: abort MLX passes whose current-rate ETA exceeds this many seconds.",
    )
    parser.add_argument(
        "--pathological_min_processed_samples",
        type=int,
        default=DEFAULT_RUNTIME_MIN_PROCESSED_SAMPLES,
        help="Minimum processed samples before throughput-based self-stop thresholds activate.",
    )
    parser.add_argument(
        "--pathological_min_elapsed_seconds",
        type=float,
        default=DEFAULT_RUNTIME_MIN_ELAPSED_SECONDS,
        help="Minimum elapsed seconds before throughput-based self-stop thresholds activate.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print commands without executing them.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from an existing results CSV by forwarding --resume to each eval subprocess.",
    )
    parser.add_argument(
        "--enable_sparse_accuracy_proxy",
        action="store_true",
        help="Legacy compatibility flag. Sparse-tagged configs with non-trivial sparse settings now auto-enable sparse perturbation.",
    )
    parser.add_argument(
        "--annotation_contract_note",
        default="",
        help="Optional note stamped onto annotated bitstream-measured bridge rows when --annotated_results_csv is enabled.",
    )
    parser.add_argument("--enable_bitstream_pilot", action="store_true")
    parser.add_argument("--bitstream_surface_scope", default="limited_linear_attention_pilot")
    parser.add_argument("--bitstream_target_module_keys", default=None)
    parser.add_argument("--bitstream_generator", default="low_discrepancy")
    parser.add_argument("--bitstream_stream_length", type=int, default=64)
    parser.add_argument(
        "--allow_unsafe_bitstream_runtime_shapes",
        action="store_true",
        help=(
            "Explicit escape hatch for guarded repair work. Without this flag, "
            "all-target and other non-tractable bitstream runtime shapes fail before launch."
        ),
    )
    parser.add_argument(
        "--bitstream_stream_reuse_policy",
        default="operand_factored_module_call_reuse",
    )
    parser.add_argument("--bitstream_encoding_mode", default="bipolar")
    parser.add_argument("--bitstream_multiplier_mode", default="xnor")
    parser.add_argument("--bitstream_accumulator_mode", default="bitcount")
    parser.add_argument(
        "--bitstream_measurement_truth_class",
        default="bitstream_limited_surface_pilot",
    )
    parser.add_argument("--bitstream_truth_class_authorization_note", default=None)
    parser.add_argument(
        "--bitstream_truth_class_authorization_root",
        default=None,
        help="Optional directory where per-eval model-level measured authorization notes are auto-written and bound to each eval_run_id.",
    )
    parser.add_argument(
        "--bitstream_contract_note",
        default="limited_surface_runtime_pilot_not_full_model_measured",
    )
    return parser


if __name__ == "__main__":
    main()
