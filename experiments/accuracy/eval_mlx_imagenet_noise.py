#!/usr/bin/env python3
"""Evaluate MobileViT accuracy under photonic perturbations with MLX."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import random
import re
import signal
import subprocess
import sys
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    import cv2
except Exception as exc:  # pragma: no cover - runtime dependency
    raise SystemExit(
        "OpenCV (cv2) is required for eval_mlx_imagenet_noise.py."
    ) from exc

import numpy as np

from accuracy.bitstream_semantics import (  # noqa: E402
    BITSTREAM_MODEL_LEVEL_MEASURED_TRUTH_CLASS,
    BITSTREAM_LIMITED_SURFACE_PILOT_TRUTH_CLASS,
)
from accuracy.bitstream_conv_semantics import (  # noqa: E402
    CONV_FOCUSED_CLAIM_SURFACE_STATUS,
    CONV_MEASURED_CLOSURE_STATUS_RUNTIME_MODELED,
    conv_focused_runtime_target_module_keys,
    normalize_conv_target_module_key_for_runtime,
    resolve_conv_evidence_manifest,
)
from accuracy.bitstream_runtime_safety import (  # noqa: E402
    build_bitstream_runtime_guardrail,
    safe_quantized_batch_cap,
)
from accuracy.bitstream_truth_authorization import (  # noqa: E402
    assess_truth_class_authorization,
    resolve_truth_class_authorization_note,
)
from exp_common.io_utils import backup_existing_file  # noqa: E402
from exp_common.model_specs import MODEL_SPECS, parse_model_keys  # noqa: E402
from exp_common.runtime import resolve_data_workers  # noqa: E402
from tools.fuller_host_tuning import (  # noqa: E402
    PASS_KIND_BASELINE_EVAL,
    PASS_KIND_QUANTIZED_EVAL,
    fingerprint_runtime_policy,
    fingerprint_semantic_policy,
    load_accuracy_policy_bundle,
    resolve_pass_policy_from_bundle,
)


DEFAULT_NOISE_SIGMA_LSB = "0,0.25,0.5,1.0,2.0"
DEFAULT_GAUSSIAN_NOISE_STD = DEFAULT_NOISE_SIGMA_LSB
DEFAULT_CROSSTALK_ALPHA = "0,0.01,0.02,0.05"
RESULT_FIELDNAMES = [
    "run_id",
    "source_run_id",
    "experiment_id",
    "baseline",
    "device",
    "accuracy_backend",
    "engine",
    "workload",
    "profile",
    "sweep_resolution",
    "git_hash",
    "imagenet_val",
    "imagenet_manifest",
    "config_snapshot",
    "host_profile_id",
    "pass_kind_profile",
    "runtime_policy_fingerprint",
    "semantic_fingerprint",
    "calibration_artifact_path",
    "execution_semantics",
    "bitstream_surface_scope",
    "bitstream_target_module_keys_json",
    "bitstream_generator",
    "bitstream_stream_length",
    "bitstream_runtime_stream_reuse_policy",
    "bitstream_encoding_mode",
    "bitstream_multiplier_mode",
    "bitstream_accumulator_mode",
    "bitstream_runtime_claim_surface_status",
    "bitstream_runtime_required_operator_families_json",
    "bitstream_runtime_covered_operator_families_json",
    "bitstream_runtime_supported_operator_families_json",
    "bitstream_runtime_missing_operator_families_json",
    "bitstream_runtime_family_policy_source",
    "bitstream_runtime_manifest_path",
    "bitstream_runtime_manifest_sha256",
    "bitstream_conv_evidence_manifest_path",
    "bitstream_conv_evidence_manifest_sha256",
    "bitstream_conv_measured_closure_status",
    "bitstream_conv_measured_package_status",
    "bitstream_conv_target_set_sha256",
    "bitstream_runtime_targetable_module_keys_json",
    "bitstream_runtime_active_target_module_keys_json",
    "bitstream_runtime_targetable_module_count",
    "bitstream_runtime_active_target_module_count",
    "bitstream_measurement_truth_class",
    "bitstream_truth_class_authorization_note",
    "bitstream_truth_class_authorization_status",
    "accuracy_measurement_contract_note",
    "det_policy",
    "det_k_signature",
    "det_k_global",
    "det_prefix_error_mean",
    "det_prefix_error_p95",
    "det_perturbation",
    "sparse_tau_global",
    "sparse_active_fraction",
    "sparse_perturbation",
    "sparse_gate_mode",
    "sparse_measured_activity_fraction",
    "sparse_measured_zero_fraction",
    "sparse_stats_total_elements",
    "sparse_stats_active_elements",
    "sparse_stats_call_count",
    "sparse_stats_module_count",
    "model",
    "input_size",
    "weights_npz",
    "weights_npz_source",
    "calibration_enabled",
    "calibration_scale_source",
    "noise_sigmas",
    "crosstalk_alphas",
    "resolved_batch_size",
    "quant_bits",
    "noise_sigma_lsb",
    "gaussian_noise_std",
    "crosstalk_alpha",
    "drift_lsb",
    "noise_correlation",
    "burst_error_prob",
    "burst_error_scale_lsb",
    "burst_span",
    "top1",
    "top5",
    "top1_delta",
    "top5_delta",
    "measured_pass_elapsed_s",
    "measured_processed_samples",
    "latency_ms_per_sample",
    "measurement_window",
    "seed",
    "notes",
]
ROW_IDENTITY_FIELDS = [
    field
    for field in RESULT_FIELDNAMES
    if field
    not in {
        "source_run_id",
        "profile",
        "sweep_resolution",
        "top1",
        "top5",
        "top1_delta",
        "top5_delta",
        "measured_pass_elapsed_s",
        "measured_processed_samples",
        "latency_ms_per_sample",
        "measurement_window",
        "weights_npz_source",
    }
]
SPARSE_ACTIVITY_LAYER_FIELDNAMES = [
    "run_id",
    "experiment_id",
    "baseline",
    "model",
    "seed",
    "device",
    "gaussian_noise_std",
    "crosstalk_alpha",
    "module_key",
    "sparse_gate_mode",
    "sparse_measured_activity_fraction",
    "sparse_measured_zero_fraction",
    "sparse_stats_total_elements",
    "sparse_stats_active_elements",
    "sparse_stats_call_count",
]
IMG_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".ppm",
    ".pgm",
    ".tif",
    ".tiff",
    ".webp",
}
IMAGENET_SYNSET_DIR_RE = re.compile(r"^n\d{8}$")
MLX_MEMORY_RELEASE_INTERVAL = 8
MAX_PREFETCH_BATCHES = 4
DEFAULT_PROGRESS_HEARTBEAT_INTERVAL_SECONDS = 15.0
DEFAULT_STALL_TIMEOUT_SECONDS = 180.0
DEFAULT_PATHOLOGICAL_MIN_SAMPLES_PER_HOUR = 12.0
DEFAULT_PATHOLOGICAL_MAX_SECONDS_PER_SAMPLE = 300.0
DEFAULT_PATHOLOGICAL_MAX_ETA_CURRENT_RATE_SECONDS = 86400.0
DEFAULT_PATHOLOGICAL_MIN_PROCESSED_SAMPLES = 4
DEFAULT_PATHOLOGICAL_MIN_ELAPSED_SECONDS = 300.0
PASS_RESUME_CHECKPOINT_SCHEMA_VERSION = 2
PASS_MODE_PAIRED = "paired"
PASS_MODE_BASELINE_ONLY = "baseline_only"
PASS_MODE_QUANTIZED_ONLY = "quantized_only"
PASS_MODE_CHOICES = [
    PASS_MODE_PAIRED,
    PASS_MODE_BASELINE_ONLY,
    PASS_MODE_QUANTIZED_ONLY,
]
_STOP_REQUESTED = False


class EvaluationInterrupted(RuntimeError):
    """Raised when a graceful stop was requested during evaluation."""


class EvaluationStalled(RuntimeError):
    """Raised when no batch completes within the configured timeout."""


class EvaluationPathologicallySlow(RuntimeError):
    """Raised when throughput is too poor to justify continuing the run."""


class ProgressRecorder:
    """Append lightweight pass-level progress events to a JSONL file."""

    def __init__(
        self,
        path: Path,
        *,
        label: str | None = None,
        fresh: bool = False,
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if fresh and self.path.exists():
            backup_existing_file(self.path)
        self.label = str(label or "mlx-eval").strip() or "mlx-eval"
        self.command_started_at = time.time()

    def _write_event(self, payload: dict[str, object]) -> None:
        record = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            **payload,
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def emit_command_start(
        self,
        *,
        total_passes: int,
        run_id: str,
        experiment_id: str | None,
        model: str,
        profile: str | None,
        sweep_resolution: str | None,
    ) -> None:
        self.command_started_at = time.time()
        self._write_event(
            {
                "event": "command_start",
                "label": self.label,
                "run_id": run_id,
                "experiment_id": experiment_id,
                "model": model,
                "profile": profile,
                "sweep_resolution": sweep_resolution,
                "total_passes": total_passes,
                "command_fraction": 0.0,
                "command_elapsed_seconds": 0.0,
                "eta_command_seconds": None,
            }
        )

    def emit_pass_event(
        self,
        *,
        event: str,
        total_passes: int,
        pass_index: int,
        pass_kind: str,
        model: str,
        profile: str | None,
        sweep_resolution: str | None,
        gaussian_noise_std: float | None,
        crosstalk_alpha: float | None,
        processed_samples: int,
        total_samples: int | None,
        pass_elapsed_seconds: float,
        milestone_percent: int | None = None,
        batch_index: int | None = None,
        batch_samples: int | None = None,
        failure_kind: str | None = None,
        failure_message: str | None = None,
        runtime_module_key: str | None = None,
        runtime_stage: str | None = None,
        runtime_detail: str | None = None,
        resume_checkpoint_path: str | None = None,
        resume_checkpoint_loaded: bool | None = None,
    ) -> None:
        pass_fraction = None
        if total_samples is not None and total_samples > 0:
            pass_fraction = min(1.0, float(processed_samples) / float(total_samples))
        elif event == "pass_complete":
            pass_fraction = 1.0

        completed_passes = max(0, int(pass_index) - 1)
        current_fraction = 1.0 if event == "pass_complete" else float(pass_fraction or 0.0)
        command_fraction = (
            min(1.0, (float(completed_passes) + current_fraction) / float(total_passes))
            if total_passes > 0
            else 0.0
        )
        command_elapsed_seconds = max(0.0, time.time() - self.command_started_at)
        seconds_per_sample = None
        samples_per_hour = None
        eta_pass_seconds_current_rate = None
        eta_command_seconds_current_rate = None
        if processed_samples > 0 and pass_elapsed_seconds > 0.0:
            seconds_per_sample = float(pass_elapsed_seconds) / float(processed_samples)
            if seconds_per_sample > 0.0:
                samples_per_hour = 3600.0 / seconds_per_sample
            if total_samples is not None and total_samples > processed_samples:
                remaining_samples = int(total_samples) - int(processed_samples)
                eta_pass_seconds_current_rate = remaining_samples * seconds_per_sample
        eta_command_seconds = None
        if command_fraction >= 1.0:
            eta_command_seconds = 0.0
        elif command_fraction > 0.0:
            eta_command_seconds = max(
                0.0,
                (command_elapsed_seconds / command_fraction) - command_elapsed_seconds,
            )
        if eta_pass_seconds_current_rate is not None:
            remaining_passes = max(0, int(total_passes) - int(pass_index))
            eta_command_seconds_current_rate = max(
                0.0,
                float(eta_pass_seconds_current_rate) + (remaining_passes * float(pass_elapsed_seconds)),
            )

        payload: dict[str, object] = {
            "event": event,
            "label": self.label,
            "model": model,
            "profile": profile,
            "sweep_resolution": sweep_resolution,
            "pass_index": int(pass_index),
            "total_passes": int(total_passes),
            "pass_kind": pass_kind,
            "gaussian_noise_std": gaussian_noise_std,
            "crosstalk_alpha": crosstalk_alpha,
            "processed_samples": int(processed_samples),
            "total_samples": total_samples,
            "pass_elapsed_seconds": float(pass_elapsed_seconds),
            "command_fraction": command_fraction,
            "command_elapsed_seconds": command_elapsed_seconds,
            "eta_command_seconds": eta_command_seconds,
        }
        if seconds_per_sample is not None:
            payload["seconds_per_sample"] = float(seconds_per_sample)
        if samples_per_hour is not None:
            payload["samples_per_hour"] = float(samples_per_hour)
        if eta_pass_seconds_current_rate is not None:
            payload["eta_pass_seconds_current_rate"] = float(eta_pass_seconds_current_rate)
        if eta_command_seconds_current_rate is not None:
            payload["eta_command_seconds_current_rate"] = float(eta_command_seconds_current_rate)
        if pass_fraction is not None:
            payload["pass_fraction"] = pass_fraction
        if milestone_percent is not None:
            payload["milestone_percent"] = int(milestone_percent)
        if batch_index is not None:
            payload["batch_index"] = int(batch_index)
        if batch_samples is not None:
            payload["batch_samples"] = int(batch_samples)
        if failure_kind is not None:
            payload["failure_kind"] = str(failure_kind)
        if failure_message is not None:
            payload["failure_message"] = str(failure_message)
        if runtime_module_key is not None:
            payload["runtime_module_key"] = str(runtime_module_key)
        if runtime_stage is not None:
            payload["runtime_stage"] = str(runtime_stage)
        if runtime_detail is not None:
            payload["runtime_detail"] = str(runtime_detail)
        if resume_checkpoint_path is not None:
            payload["resume_checkpoint_path"] = str(resume_checkpoint_path)
        if resume_checkpoint_loaded is not None:
            payload["resume_checkpoint_loaded"] = bool(resume_checkpoint_loaded)
        self._write_event(payload)

    def emit_command_complete(
        self,
        *,
        total_passes: int,
        run_id: str,
        experiment_id: str | None,
        model: str,
        profile: str | None,
        sweep_resolution: str | None,
    ) -> None:
        self._write_event(
            {
                "event": "command_complete",
                "label": self.label,
                "run_id": run_id,
                "experiment_id": experiment_id,
                "model": model,
                "profile": profile,
                "sweep_resolution": sweep_resolution,
                "total_passes": total_passes,
                "command_fraction": 1.0,
                "command_elapsed_seconds": max(0.0, time.time() - self.command_started_at),
                "eta_command_seconds": 0.0,
            }
        )

    def emit_command_failed(
        self,
        *,
        total_passes: int,
        run_id: str,
        experiment_id: str | None,
        model: str,
        profile: str | None,
        sweep_resolution: str | None,
        failure_kind: str,
        failure_message: str,
    ) -> None:
        self._write_event(
            {
                "event": "command_failed",
                "label": self.label,
                "run_id": run_id,
                "experiment_id": experiment_id,
                "model": model,
                "profile": profile,
                "sweep_resolution": sweep_resolution,
                "total_passes": total_passes,
                "command_fraction": None,
                "command_elapsed_seconds": max(0.0, time.time() - self.command_started_at),
                "eta_command_seconds": None,
                "failure_kind": str(failure_kind),
                "failure_message": str(failure_message),
            }
        )


def parse_list(value: str | None, cast_type=float) -> list:
    if value is None:
        return []
    return [cast_type(entry.strip()) for entry in value.split(",") if entry.strip()]


def parse_float_list(raw: str) -> list[float]:
    return [float(x) for x in raw.split(",") if x.strip()]


def parse_csv_tokens(raw: str | None) -> list[str]:
    if raw in {None, ""}:
        return []
    text = str(raw).strip()
    if text.lower() in {"all", "*", "__all__"}:
        return []
    if text.startswith("@"):
        path = Path(text[1:]).expanduser()
        if not path.is_absolute():
            path = ROOT_DIR / path
        payload = path.read_text(encoding="utf-8").strip()
        if not payload:
            return []
        if path.suffix.lower() == ".json":
            parsed = json.loads(payload)
            if isinstance(parsed, dict):
                parsed = parsed.get("target_module_keys") or parsed.get("keys") or []
            if not isinstance(parsed, list):
                raise ValueError(
                    "JSON target module key files must contain a list, or an object "
                    "with target_module_keys/keys."
                )
            return [str(token).strip() for token in parsed if str(token).strip()]
        return [
            token.strip()
            for line in payload.splitlines()
            for token in line.split(",")
            if token.strip()
        ]
    return [token.strip() for token in text.split(",") if token.strip()]


def _serialize_float_list(values: list[float]) -> str:
    return ",".join(str(float(value)) for value in values)


def _blank_bitstream_runtime_coverage_fields() -> dict[str, object]:
    return {
        "bitstream_runtime_claim_surface_status": "",
        "bitstream_runtime_required_operator_families_json": "",
        "bitstream_runtime_covered_operator_families_json": "",
        "bitstream_runtime_supported_operator_families_json": "",
        "bitstream_runtime_missing_operator_families_json": "",
        "bitstream_runtime_family_policy_source": "",
        "bitstream_runtime_manifest_path": "",
        "bitstream_runtime_manifest_sha256": "",
        "bitstream_conv_evidence_manifest_path": "",
        "bitstream_conv_evidence_manifest_sha256": "",
        "bitstream_conv_measured_closure_status": "",
        "bitstream_conv_measured_package_status": "",
        "bitstream_conv_target_set_sha256": "",
        "bitstream_runtime_targetable_module_keys_json": "",
        "bitstream_runtime_active_target_module_keys_json": "",
        "bitstream_runtime_targetable_module_count": None,
        "bitstream_runtime_active_target_module_count": None,
    }


def _load_claim_surface_manifest_metadata(model_key: str) -> dict[str, object]:
    manifest_path = ROOT_DIR / "mtl_model" / "ops" / f"ops_{model_key}.json"
    try:
        raw_text = manifest_path.read_text(encoding="utf-8")
        payload = json.loads(raw_text)
    except (OSError, json.JSONDecodeError):
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
    families = {
        str(row.get("type") or "").strip()
        for row in ops
        if isinstance(row, dict) and str(row.get("type") or "").strip()
    }
    return {
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": hashlib.sha256(raw_text.encode("utf-8")).hexdigest(),
        "required_families": sorted(families),
    }


def _load_runtime_family_policy() -> tuple[dict[str, dict[str, str]], str]:
    from accuracy.mlx_mobilevit import (
        BITSTREAM_RUNTIME_FAMILY_POLICY_SOURCE,
        resolve_bitstream_runtime_family_policy,
    )

    return resolve_bitstream_runtime_family_policy(), BITSTREAM_RUNTIME_FAMILY_POLICY_SOURCE


def _derive_bitstream_runtime_metadata(
    *,
    model_key: str,
    base_metadata: dict[str, object] | None,
    perturb_config: Any,
    expected_run_id: str | None = None,
) -> dict[str, object]:
    metadata = dict(base_metadata or {})
    if metadata.get("execution_semantics") != "bitstream":
        return {
            **metadata,
            **_blank_bitstream_runtime_coverage_fields(),
        }
    runtime_cfg = None if perturb_config is None else perturb_config.bitstream_execution_config
    targetable_map = (
        {}
        if perturb_config is None
        else dict(getattr(perturb_config, "bitstream_runtime_targetable_module_families", {}) or {})
    )
    targetable_keys = sorted(str(key) for key in targetable_map)
    requested_keys = [] if runtime_cfg is None or runtime_cfg.target_module_keys is None else sorted(
        str(key) for key in runtime_cfg.target_module_keys if str(key).strip()
    )
    active_keys = targetable_keys if not requested_keys else [
        key for key in requested_keys if key in targetable_map
    ]
    manifest_metadata = _load_claim_surface_manifest_metadata(model_key)
    conv_manifest_metadata = resolve_conv_evidence_manifest(
        model_key=model_key,
        ops_path=ROOT_DIR / "mtl_model" / "ops" / f"ops_{model_key}.json",
    )
    required_families = list(manifest_metadata["required_families"])
    family_policy, family_policy_source = _load_runtime_family_policy()
    active_native_families = {str(targetable_map[key]) for key in active_keys}
    covered_families: list[str] = []
    for family in required_families:
        policy = family_policy.get(str(family))
        if not isinstance(policy, dict):
            continue
        coverage_mechanism = str(policy.get("coverage_mechanism") or "").strip()
        if coverage_mechanism == "targetable_module_family":
            if family in active_native_families:
                covered_families.append(str(family))
        elif coverage_mechanism == "declared_runtime_support_policy":
            covered_families.append(str(family))
    covered_families = sorted(set(covered_families))
    missing_families = sorted(set(required_families) - set(covered_families))
    requested_truth_class = str(metadata.get("bitstream_measurement_truth_class") or "").strip()
    requested_authorization_note = str(
        metadata.get("bitstream_truth_class_authorization_note") or ""
    ).strip()
    resolved_authorization_note = resolve_truth_class_authorization_note(
        requested_authorization_note,
        search_roots=(ROOT_DIR.parent, ROOT_DIR),
    )
    authorization_assessment = assess_truth_class_authorization(
        resolved_authorization_note,
        expected_run_id=expected_run_id,
    )
    expected_conv_focused_runtime_keys = conv_focused_runtime_target_module_keys(
        conv_manifest_metadata.get("manifest") or {}
    )
    conv_focused_coverage = bool(expected_conv_focused_runtime_keys) and (
        sorted(active_keys) == sorted(expected_conv_focused_runtime_keys)
        and sorted(requested_keys) == sorted(expected_conv_focused_runtime_keys)
    )
    full_runtime_coverage = required_families and not missing_families and (
        not requested_keys or set(targetable_keys).issubset(set(requested_keys))
    )
    authorization_status = "not_required"
    if requested_truth_class == BITSTREAM_MODEL_LEVEL_MEASURED_TRUTH_CLASS:
        if not full_runtime_coverage and not conv_focused_coverage:
            authorization_status = "not_eligible"
        else:
            authorization_status = str(authorization_assessment["status"])
    if (
        full_runtime_coverage
        and requested_truth_class == BITSTREAM_MODEL_LEVEL_MEASURED_TRUTH_CLASS
        and bool(authorization_assessment["authorized"])
    ):
        claim_surface_status = "full_model_claim_surface_runtime"
        truth_class = BITSTREAM_MODEL_LEVEL_MEASURED_TRUTH_CLASS
    elif (
        conv_focused_coverage
        and requested_truth_class == BITSTREAM_MODEL_LEVEL_MEASURED_TRUTH_CLASS
    ):
        claim_surface_status = CONV_FOCUSED_CLAIM_SURFACE_STATUS
        truth_class = BITSTREAM_MODEL_LEVEL_MEASURED_TRUTH_CLASS
    else:
        claim_surface_status = (
            "full_model_claim_surface_runtime_nonpromotable"
            if full_runtime_coverage
            else "limited_model_claim_surface_runtime"
        )
        truth_class = BITSTREAM_LIMITED_SURFACE_PILOT_TRUTH_CLASS
    contract_note = str(metadata.get("accuracy_measurement_contract_note") or "").strip()
    if not contract_note:
        contract_note = (
            "full_model_claim_surface_runtime_measured"
            if truth_class == BITSTREAM_MODEL_LEVEL_MEASURED_TRUTH_CLASS
            and claim_surface_status == "full_model_claim_surface_runtime"
            else (
                "conv_focused_claim_surface_runtime_measured"
                if truth_class == BITSTREAM_MODEL_LEVEL_MEASURED_TRUTH_CLASS
                and claim_surface_status == CONV_FOCUSED_CLAIM_SURFACE_STATUS
                else (
                    "full_model_claim_surface_runtime_missing_authorization_note"
                    if full_runtime_coverage
                    and requested_truth_class == BITSTREAM_MODEL_LEVEL_MEASURED_TRUTH_CLASS
                    and authorization_status == "missing"
                    else (
                        "conv_focused_claim_surface_runtime_missing_authorization_note"
                        if conv_focused_coverage
                        and requested_truth_class
                        == BITSTREAM_MODEL_LEVEL_MEASURED_TRUTH_CLASS
                        and authorization_status == "missing"
                        else (
                            "full_model_claim_surface_runtime_authorization_unsatisfied"
                            if full_runtime_coverage
                            and requested_truth_class
                            == BITSTREAM_MODEL_LEVEL_MEASURED_TRUTH_CLASS
                            and authorization_status not in {"authorized", "missing"}
                            else (
                                "conv_focused_claim_surface_runtime_authorization_unsatisfied"
                                if conv_focused_coverage
                                and requested_truth_class
                                == BITSTREAM_MODEL_LEVEL_MEASURED_TRUTH_CLASS
                                and authorization_status not in {"authorized", "missing"}
                                else "coverage_derived_limited_surface_runtime_pilot_not_full_model_measured"
                            )
                        )
                    )
                )
            )
        )
    return {
        **metadata,
        "bitstream_runtime_claim_surface_status": claim_surface_status,
        "bitstream_runtime_required_operator_families_json": json.dumps(
            required_families,
            ensure_ascii=False,
        ),
        "bitstream_runtime_covered_operator_families_json": json.dumps(
            covered_families,
            ensure_ascii=False,
        ),
        "bitstream_runtime_supported_operator_families_json": json.dumps(
            covered_families,
            ensure_ascii=False,
        ),
        "bitstream_runtime_missing_operator_families_json": json.dumps(
            missing_families,
            ensure_ascii=False,
        ),
        "bitstream_runtime_family_policy_source": family_policy_source,
        "bitstream_runtime_manifest_path": str(manifest_metadata["manifest_path"] or ""),
        "bitstream_runtime_manifest_sha256": str(
            manifest_metadata["manifest_sha256"] or ""
        ),
        "bitstream_conv_evidence_manifest_path": str(
            conv_manifest_metadata.get("manifest_path") or ""
        ),
        "bitstream_conv_evidence_manifest_sha256": str(
            conv_manifest_metadata.get("manifest_sha256") or ""
        ),
        "bitstream_conv_measured_closure_status": CONV_MEASURED_CLOSURE_STATUS_RUNTIME_MODELED,
        "bitstream_conv_measured_package_status": "",
        "bitstream_conv_target_set_sha256": str(
            (
                (
                    (conv_manifest_metadata.get("manifest") or {}).get(
                        "conv_focused_target_set"
                    )
                    or {}
                ).get("target_set_sha256")
            )
            or ""
        ),
        "bitstream_runtime_targetable_module_keys_json": json.dumps(
            targetable_keys,
            ensure_ascii=False,
        ),
        "bitstream_runtime_active_target_module_keys_json": json.dumps(
            active_keys,
            ensure_ascii=False,
        ),
        "bitstream_runtime_targetable_module_count": len(targetable_keys),
        "bitstream_runtime_active_target_module_count": len(active_keys),
        "bitstream_measurement_truth_class": truth_class,
        "bitstream_truth_class_authorization_note": (
            str(authorization_assessment["resolved_path"] or "")
        ),
        "bitstream_truth_class_authorization_status": authorization_status,
        "accuracy_measurement_contract_note": contract_note,
    }


def _build_bitstream_runtime_metadata(args: argparse.Namespace) -> dict[str, object]:
    raw_target_module_keys = parse_csv_tokens(args.bitstream_target_module_keys)
    target_module_keys: list[str] = []
    seen_target_module_keys: set[str] = set()
    for raw_key in raw_target_module_keys:
        normalized_key = normalize_conv_target_module_key_for_runtime(str(raw_key))
        if not normalized_key or normalized_key in seen_target_module_keys:
            continue
        seen_target_module_keys.add(normalized_key)
        target_module_keys.append(normalized_key)
    if not bool(args.enable_bitstream_pilot):
        return {
            "execution_semantics": "",
            "bitstream_surface_scope": "",
            "bitstream_target_module_keys_json": "",
            "bitstream_generator": "",
            "bitstream_stream_length": None,
            "bitstream_runtime_stream_reuse_policy": "",
            "bitstream_encoding_mode": "",
            "bitstream_multiplier_mode": "",
            "bitstream_accumulator_mode": "",
            **_blank_bitstream_runtime_coverage_fields(),
            "bitstream_measurement_truth_class": "",
            "bitstream_truth_class_authorization_note": "",
            "bitstream_truth_class_authorization_status": "",
            "accuracy_measurement_contract_note": "",
            "target_module_keys": [],
        }
    contract_note = str(
        args.bitstream_contract_note or "limited_surface_runtime_pilot_not_full_model_measured"
    ).strip()
    return {
        "execution_semantics": "bitstream",
        "bitstream_surface_scope": str(args.bitstream_surface_scope or "").strip(),
        "bitstream_target_module_keys_json": json.dumps(
            target_module_keys,
            ensure_ascii=False,
        ),
        "bitstream_generator": str(args.bitstream_generator or "").strip(),
        "bitstream_stream_length": int(args.bitstream_stream_length),
        "bitstream_runtime_stream_reuse_policy": str(
            args.bitstream_stream_reuse_policy or ""
        ).strip(),
        "bitstream_encoding_mode": str(args.bitstream_encoding_mode or "").strip(),
        "bitstream_multiplier_mode": str(args.bitstream_multiplier_mode or "").strip(),
        "bitstream_accumulator_mode": str(args.bitstream_accumulator_mode or "").strip(),
        **_blank_bitstream_runtime_coverage_fields(),
        "bitstream_measurement_truth_class": str(
            args.bitstream_measurement_truth_class or ""
        ).strip(),
        "bitstream_truth_class_authorization_note": str(
            args.bitstream_truth_class_authorization_note or ""
        ).strip(),
        "bitstream_truth_class_authorization_status": "",
        "accuracy_measurement_contract_note": contract_note,
        "target_module_keys": target_module_keys,
    }


def _validate_requested_model_level_measured_authorization(
    args: argparse.Namespace,
) -> None:
    requested_truth_class = str(args.bitstream_measurement_truth_class or "").strip()
    if requested_truth_class != BITSTREAM_MODEL_LEVEL_MEASURED_TRUTH_CLASS:
        return
    resolved_authorization_note = resolve_truth_class_authorization_note(
        args.bitstream_truth_class_authorization_note,
        search_roots=(ROOT_DIR.parent, ROOT_DIR),
    )
    assessment = assess_truth_class_authorization(
        resolved_authorization_note,
        expected_run_id=str(args.run_id or "").strip() or None,
    )
    if bool(assessment["authorized"]):
        return
    raise SystemExit(
        "Refusing to launch bitstream_model_level_measured without a valid bound "
        "authorization note. "
        f"status={assessment['status']} "
        f"expected_run_id={assessment['expected_run_id'] or ''} "
        f"note={assessment['resolved_path'] or ''}"
    )


def _runtime_policy_provenance_fields(
    pass_policy: dict[str, object] | None,
) -> dict[str, object]:
    policy = dict(pass_policy or {})
    return {
        "host_profile_id": str(policy.get("host_profile_id") or ""),
        "pass_kind_profile": str(policy.get("profile_id") or ""),
        "runtime_policy_fingerprint": str(policy.get("runtime_policy_fingerprint") or ""),
        "semantic_fingerprint": str(policy.get("semantic_fingerprint") or ""),
        "calibration_artifact_path": str(policy.get("calibration_artifact_path") or ""),
    }


def _row_runtime_metadata(
    runtime_metadata: dict[str, object] | None,
    *,
    enabled: bool,
    pass_policy: dict[str, object] | None = None,
) -> dict[str, object]:
    metadata = dict(runtime_metadata or {})
    metadata.pop("target_module_keys", None)
    provenance = _runtime_policy_provenance_fields(pass_policy)
    if enabled:
        return {
            **provenance,
            **metadata,
        }
    return {
        **provenance,
        "execution_semantics": "",
        "bitstream_surface_scope": "",
        "bitstream_target_module_keys_json": "",
        "bitstream_generator": "",
        "bitstream_stream_length": None,
        "bitstream_runtime_stream_reuse_policy": "",
        "bitstream_encoding_mode": "",
        "bitstream_multiplier_mode": "",
        "bitstream_accumulator_mode": "",
        **_blank_bitstream_runtime_coverage_fields(),
        "bitstream_measurement_truth_class": "",
        "bitstream_truth_class_authorization_note": "",
        "bitstream_truth_class_authorization_status": "",
        "accuracy_measurement_contract_note": "",
    }


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _normalize_row_identity_value(field: str, value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text:
        return ""
    if field in {"baseline", "det_perturbation", "sparse_perturbation"}:
        return "1" if text.lower() == "true" else "0"
    if field in {
        "input_size",
        "quant_bits",
        "noise_sigma_lsb",
        "crosstalk_alpha",
        "drift_lsb",
        "noise_correlation",
        "burst_error_prob",
        "burst_error_scale_lsb",
        "burst_span",
        "seed",
        "det_k_global",
        "det_prefix_error_mean",
        "det_prefix_error_p95",
        "sparse_tau_global",
        "sparse_active_fraction",
    }:
        try:
            return f"{float(text):.12g}"
        except ValueError:
            return text
    return text


def result_row_identity(row: dict[str, object]) -> tuple[str, ...]:
    return tuple(
        _normalize_row_identity_value(field, row.get(field))
        for field in ROW_IDENTITY_FIELDS
    )


def _load_accuracy_policy_profile(raw: str | None) -> dict[str, Any] | None:
    if raw in (None, ""):
        return None
    payload = load_accuracy_policy_bundle(str(raw))
    return dict(payload or {}) if payload is not None else None


def _resolve_pass_kind_runtime_health_gate(
    policy_bundle: dict[str, Any] | None,
    *,
    pass_kind: str,
    fallback: dict[str, Any],
) -> dict[str, Any]:
    resolved = resolve_pass_policy_from_bundle(policy_bundle, pass_kind=pass_kind) or {}
    return {
        **dict(fallback),
        **dict(resolved.get("runtime_health_gate") or {}),
    }


def _resolve_pass_kind_semantic_profile(
    policy_bundle: dict[str, Any] | None,
    *,
    pass_kind: str,
) -> dict[str, Any]:
    resolved = resolve_pass_policy_from_bundle(policy_bundle, pass_kind=pass_kind) or {}
    return dict(resolved.get("semantic_policy") or {})


def _apply_quantized_semantic_profile(
    perturb_config: Any,
    semantic_policy: dict[str, Any],
) -> None:
    if perturb_config is None:
        runtime_cfg = None
    elif hasattr(perturb_config, "bitstream_execution_config"):
        runtime_cfg = perturb_config.bitstream_execution_config
    else:
        runtime_cfg = perturb_config
    if runtime_cfg is None:
        return
    attr_map = {
        "bitstream_generator": "generator",
        "bitstream_stream_length": "stream_length",
        "bitstream_stream_reuse_policy": "stream_reuse_policy",
        "bitstream_encoding_mode": "encoding_mode",
        "bitstream_multiplier_mode": "multiplier_mode",
        "bitstream_accumulator_mode": "accumulator_mode",
    }
    for source_key, attr_name in attr_map.items():
        value = semantic_policy.get(source_key)
        if value in ("", None):
            continue
        setattr(runtime_cfg, attr_name, value)


def _resolve_pass_runtime_tuning(
    *,
    policy_bundle: dict[str, Any] | None,
    pass_kind: str,
    fallback_workers: int,
    fallback_eval_batch_size: int,
    fallback_runtime_health_gate: dict[str, Any],
    runtime_metadata: dict[str, object],
    model_key: str,
    target_module_keys_raw: str | None,
    max_eval_samples: int | None,
) -> dict[str, Any]:
    policy = resolve_pass_policy_from_bundle(policy_bundle, pass_kind=pass_kind) or {}
    workers = int(policy.get("workers") or fallback_workers)
    prefetch_batches = int(
        policy.get("prefetch_batches") or _resolve_prefetch_batches(workers)
    )
    requested_batch_size = int(
        policy.get("eval_batch_size") or fallback_eval_batch_size
    )
    runtime_health_gate = _resolve_pass_kind_runtime_health_gate(
        policy_bundle,
        pass_kind=pass_kind,
        fallback=fallback_runtime_health_gate,
    )
    safe_batch_cap = None
    quantized_batch_policy: dict[str, Any] = {}
    resolved_batch_size = requested_batch_size
    if pass_kind == PASS_KIND_QUANTIZED_EVAL:
        quantized_batch_policy = _resolve_quantized_bitstream_batch_policy(
            requested_batch_size=requested_batch_size,
            runtime_metadata=runtime_metadata,
            model_key=model_key,
            target_module_keys_raw=target_module_keys_raw,
            max_eval_samples=max_eval_samples,
        )
        safe_batch_cap = quantized_batch_policy.get("safe_batch_cap")
        resolved_batch_size = int(quantized_batch_policy["quantized_batch_size"])
    effective_policy = {
        "pass_kind": pass_kind,
        "execution_semantics": str(policy.get("execution_semantics") or ""),
        "workers": workers,
        "prefetch_batches": prefetch_batches,
        "eval_batch_size": resolved_batch_size,
        "safe_batch_cap": safe_batch_cap,
        "runtime_health_gate": runtime_health_gate,
    }
    return {
        "host_profile_id": str((policy_bundle or {}).get("host_profile_id") or ""),
        "profile_id": str(policy.get("profile_id") or ""),
        "workers": workers,
        "prefetch_batches": prefetch_batches,
        "requested_eval_batch_size": requested_batch_size,
        "eval_batch_size": resolved_batch_size,
        "runtime_health_gate": runtime_health_gate,
        "semantic_policy": _resolve_pass_kind_semantic_profile(
            policy_bundle,
            pass_kind=pass_kind,
        ),
        "runtime_policy_fingerprint": fingerprint_runtime_policy(effective_policy),
        "semantic_fingerprint": fingerprint_semantic_policy(
            _resolve_pass_kind_semantic_profile(policy_bundle, pass_kind=pass_kind),
            execution_semantics=str(policy.get("execution_semantics") or ""),
        ),
        "calibration_artifact_path": str(
            policy.get("calibration_artifact_path")
            or (policy_bundle or {}).get("calibration_artifact_path")
            or ""
        ),
        "quantized_batch_policy": quantized_batch_policy,
    }


def _sparse_metadata_fields(args: argparse.Namespace) -> dict[str, object]:
    return {
        "sparse_tau_global": args.sparse_tau_global,
        "sparse_active_fraction": args.sparse_active_fraction,
        "sparse_perturbation": bool(args.apply_sparse_perturbation),
    }


def load_existing_result_identities(path: Path) -> set[tuple[str, ...]]:
    if not path.is_file():
        return set()
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return {result_row_identity(row) for row in reader}


def load_existing_result_rows_by_identity(path: Path) -> dict[tuple[str, ...], dict[str, str]]:
    if not path.is_file():
        return {}
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return {result_row_identity(row): row for row in reader}


def _atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    tmp_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    tmp_path.replace(path)


def _resume_artifact_root(
    *,
    progress_jsonl: str | None,
    results_path: Path,
) -> Path:
    if progress_jsonl:
        return Path(progress_jsonl).resolve().parent / "resume_checkpoints"
    return results_path.resolve().parent / f"{results_path.stem}.resume_checkpoints"


def _identity_digest(row_identity: tuple[str, ...]) -> str:
    return hashlib.sha256(
        json.dumps(list(row_identity), ensure_ascii=False, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()[:24]


def _pass_resume_checkpoint_path(
    *,
    root: Path,
    row_identity: tuple[str, ...],
    pass_kind: str,
) -> Path:
    digest = _identity_digest(row_identity)
    safe_pass_kind = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(pass_kind)).strip("_") or "pass"
    return root / f"{safe_pass_kind}_{digest}.json"


def _calibration_scale_cache_path(
    *,
    root: Path,
    baseline_identity: tuple[str, ...],
) -> Path:
    return root / f"calibration_scale_cache_{_identity_digest(baseline_identity)}.npz"


def _coerce_optional_int(value: object) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def _snapshot_mlx_random_state(mx_module: Any) -> list[list[int]] | None:
    random_module = getattr(mx_module, "random", None)
    state = getattr(random_module, "state", None)
    if not isinstance(state, list):
        return None
    snapshot: list[list[int]] = []
    for entry in state:
        snapshot.append(np.array(entry, dtype=np.uint32).reshape(-1).astype(int).tolist())
    return snapshot


def _restore_mlx_random_state(mx_module: Any, snapshot: object) -> None:
    if not isinstance(snapshot, list):
        return
    random_module = getattr(mx_module, "random", None)
    state = getattr(random_module, "state", None)
    if not isinstance(state, list):
        return
    for index, entry in enumerate(snapshot):
        if index >= len(state):
            break
        state[index] = mx_module.array(np.array(entry, dtype=np.uint32))


def _load_pass_resume_checkpoint(
    path: Path,
    *,
    row_identity: tuple[str, ...],
    expected_total_samples: int,
    batch_size: int,
    max_samples: int | None,
    runtime_policy_fingerprint: str | None = None,
    semantic_fingerprint: str | None = None,
) -> dict[str, object] | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if int(payload.get("schema_version") or 0) != PASS_RESUME_CHECKPOINT_SCHEMA_VERSION:
        raise RuntimeError(f"Unsupported resume checkpoint schema: {path}")
    if tuple(str(value) for value in payload.get("row_identity") or []) != row_identity:
        raise RuntimeError(f"Resume checkpoint row identity mismatch: {path}")
    if int(payload.get("expected_total_samples") or 0) != int(expected_total_samples):
        raise RuntimeError(f"Resume checkpoint sample-count mismatch: {path}")
    if int(payload.get("batch_size") or 0) != int(batch_size):
        raise RuntimeError(
            f"Resume checkpoint batch-size mismatch: {path}; rerun with the same command."
        )
    if _coerce_optional_int(payload.get("max_samples")) != _coerce_optional_int(max_samples):
        raise RuntimeError(
            f"Resume checkpoint max-samples mismatch: {path}; rerun with the same command."
        )
    if runtime_policy_fingerprint is not None and str(
        payload.get("runtime_policy_fingerprint") or ""
    ) != str(runtime_policy_fingerprint):
        raise RuntimeError(
            f"Resume checkpoint runtime-policy mismatch: {path}; rerun with the same command."
        )
    if semantic_fingerprint is not None and str(
        payload.get("semantic_fingerprint") or ""
    ) != str(semantic_fingerprint):
        raise RuntimeError(
            f"Resume checkpoint semantic mismatch: {path}; rerun with the same command."
        )
    processed_samples = int(payload.get("processed_samples") or 0)
    if processed_samples < 0 or processed_samples > int(expected_total_samples):
        raise RuntimeError(f"Resume checkpoint has invalid processed_samples: {path}")
    if processed_samples % int(batch_size) != 0 and processed_samples != int(expected_total_samples):
        raise RuntimeError(
            f"Resume checkpoint stops mid-batch: {path}; refusing partial-batch resume."
        )
    return payload


def _write_pass_resume_checkpoint(
    path: Path,
    *,
    row_identity: tuple[str, ...],
    pass_kind: str,
    model_key: str,
    seed: int,
    batch_size: int,
    max_samples: int | None,
    expected_total_samples: int,
    processed_samples: int,
    top1_correct: int,
    top5_correct: int,
    batch_index: int | None,
    pass_elapsed_seconds: float,
    mlx_random_state: list[list[int]] | None,
    completed: bool,
    runtime_policy_fingerprint: str | None = None,
    semantic_fingerprint: str | None = None,
) -> None:
    _atomic_write_json(
        path,
        {
            "schema_version": PASS_RESUME_CHECKPOINT_SCHEMA_VERSION,
            "row_identity": list(row_identity),
            "pass_kind": pass_kind,
            "model": model_key,
            "seed": int(seed),
            "batch_size": int(batch_size),
            "max_samples": max_samples,
            "expected_total_samples": int(expected_total_samples),
            "processed_samples": int(processed_samples),
            "top1_correct": int(top1_correct),
            "top5_correct": int(top5_correct),
            "batch_index": batch_index,
            "pass_elapsed_seconds": float(pass_elapsed_seconds),
            "mlx_random_state": mlx_random_state,
            "completed": bool(completed),
            "runtime_policy_fingerprint": str(runtime_policy_fingerprint or ""),
            "semantic_fingerprint": str(semantic_fingerprint or ""),
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        },
    )


def _save_calibration_scale_cache(path: Path, perturb_config: Any) -> bool:
    scale_cache = dict(getattr(perturb_config, "scale_cache", {}) or {})
    if not scale_cache:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        key: np.array(value, dtype=np.float32)
        for key, value in scale_cache.items()
    }
    tmp_path = path.with_suffix(f"{path.suffix}.tmp.npz")
    np.savez_compressed(tmp_path, **payload)
    tmp_path.replace(path)
    return True


def _load_calibration_scale_cache(path: Path, perturb_config: Any) -> bool:
    if not path.is_file():
        return False
    import mlx.core as mx

    getattr(perturb_config, "scale_cache").clear()
    with np.load(path, allow_pickle=False) as loaded:
        for key in loaded.files:
            perturb_config.scale_cache[str(key)] = mx.array(loaded[key])
    return True


def _request_graceful_stop(signum, _frame) -> None:
    global _STOP_REQUESTED
    if _STOP_REQUESTED:
        return
    _STOP_REQUESTED = True
    signal_name = signal.Signals(signum).name
    print(
        f"[interrupt] received {signal_name}; will stop at the next safe checkpoint so resume can continue cleanly.",
        flush=True,
    )


def _install_interrupt_handlers() -> None:
    for signame in ("SIGINT", "SIGTERM"):
        signum = getattr(signal, signame, None)
        if signum is not None:
            signal.signal(signum, _request_graceful_stop)


def _raise_if_stop_requested(where: str) -> None:
    if _STOP_REQUESTED:
        raise EvaluationInterrupted(where)


def _pathological_slow_reason(
    *,
    processed_samples: int,
    total_samples: int | None,
    elapsed_seconds: float,
    min_samples_per_hour: float | None,
    max_seconds_per_sample: float | None,
    max_eta_current_rate_seconds: float | None,
    min_processed_samples: int | None,
    min_elapsed_seconds: float | None,
) -> str | None:
    if processed_samples <= 0 or elapsed_seconds <= 0.0:
        return None
    if min_processed_samples is not None and int(min_processed_samples) > 0:
        if int(processed_samples) < int(min_processed_samples):
            return None
    if min_elapsed_seconds is not None and float(min_elapsed_seconds) > 0.0:
        if float(elapsed_seconds) < float(min_elapsed_seconds):
            return None

    seconds_per_sample = float(elapsed_seconds) / float(processed_samples)
    samples_per_hour = 3600.0 / seconds_per_sample if seconds_per_sample > 0.0 else None
    reasons: list[str] = []
    if (
        max_seconds_per_sample is not None
        and float(max_seconds_per_sample) > 0.0
        and seconds_per_sample > float(max_seconds_per_sample)
    ):
        reasons.append(
            f"seconds_per_sample={seconds_per_sample:.2f} > {float(max_seconds_per_sample):.2f}"
        )
    if (
        min_samples_per_hour is not None
        and float(min_samples_per_hour) > 0.0
        and samples_per_hour is not None
        and samples_per_hour < float(min_samples_per_hour)
    ):
        reasons.append(
            f"samples_per_hour={samples_per_hour:.2f} < {float(min_samples_per_hour):.2f}"
        )
    if (
        max_eta_current_rate_seconds is not None
        and float(max_eta_current_rate_seconds) > 0.0
        and total_samples is not None
        and int(total_samples) > int(processed_samples)
    ):
        eta_current_rate = (int(total_samples) - int(processed_samples)) * seconds_per_sample
        if eta_current_rate > float(max_eta_current_rate_seconds):
            reasons.append(
                f"eta_current_rate={eta_current_rate:.2f}s > {float(max_eta_current_rate_seconds):.2f}s"
            )
    return "; ".join(reasons) or None


def _resolve_model_bitstream_runtime_config(model: Any) -> Any | None:
    perturb_config = getattr(model, "perturb_config", None)
    if perturb_config is None:
        return None
    runtime_cfg = getattr(perturb_config, "bitstream_execution_config", None)
    if runtime_cfg is None or not bool(getattr(runtime_cfg, "enabled", False)):
        return None
    return runtime_cfg


def _attach_runtime_progress_callback(
    *,
    model: Any,
    progress_recorder: ProgressRecorder | None,
    total_passes: int | None,
    pass_index: int | None,
    pass_kind: str,
    expected_total_samples: int | None,
    profile: str | None,
    sweep_resolution: str | None,
    gaussian_noise_std: float | None,
    crosstalk_alpha: float | None,
    stall_timeout_value: float | None,
    heartbeat_interval_value: float | None,
    pathological_min_samples_per_hour: float | None,
    pathological_max_seconds_per_sample: float | None,
    pathological_max_eta_current_rate_seconds: float | None,
    pathological_min_processed_samples: int | None,
    pathological_min_elapsed_seconds: float | None,
    pathological_inflight_sample_floor: int,
    started_at: float,
    get_processed_samples,
    get_batch_index,
    elapsed_offset_seconds: float = 0.0,
) -> tuple[Any, Any] | None:
    runtime_cfg = _resolve_model_bitstream_runtime_config(model)
    if runtime_cfg is None:
        return None
    previous_callback = getattr(runtime_cfg, "progress_callback", None)
    last_alarm_refresh_at = started_at
    last_emit_at = started_at

    def _runtime_progress_callback(
        *,
        module_key: str,
        stage: str,
        call_index: int,
        detail: str,
        timestamp: float,
    ) -> None:
        nonlocal last_alarm_refresh_at, last_emit_at
        now = float(timestamp)
        if (
            stall_timeout_value is not None
            and hasattr(signal, "setitimer")
            and (now - last_alarm_refresh_at) >= 1.0
        ):
            signal.setitimer(signal.ITIMER_REAL, stall_timeout_value)
            last_alarm_refresh_at = now
        processed_samples = int(get_processed_samples())
        pathology_processed_samples = processed_samples
        if get_batch_index():
            inflight_sample_floor = max(1, int(pathological_inflight_sample_floor))
            if pathology_processed_samples <= 0:
                pathology_processed_samples = inflight_sample_floor
            else:
                pathology_processed_samples = pathology_processed_samples + inflight_sample_floor
                if expected_total_samples is not None and int(expected_total_samples) > 0:
                    pathology_processed_samples = min(
                        pathology_processed_samples,
                        int(expected_total_samples),
                    )
        pathological_reason = _pathological_slow_reason(
            processed_samples=pathology_processed_samples,
            total_samples=expected_total_samples,
            elapsed_seconds=float(elapsed_offset_seconds) + max(0.0, now - started_at),
            min_samples_per_hour=pathological_min_samples_per_hour,
            max_seconds_per_sample=pathological_max_seconds_per_sample,
            max_eta_current_rate_seconds=pathological_max_eta_current_rate_seconds,
            min_processed_samples=pathological_min_processed_samples,
            min_elapsed_seconds=pathological_min_elapsed_seconds,
        )
        if pathological_reason:
            raise EvaluationPathologicallySlow(pathological_reason)
        if (
            progress_recorder is None
            or total_passes is None
            or pass_index is None
        ):
            return
        if (
            heartbeat_interval_value is not None
            and (now - last_emit_at) < heartbeat_interval_value
        ):
            return
        progress_recorder.emit_pass_event(
            event="pass_runtime_heartbeat",
            total_passes=total_passes,
            pass_index=pass_index,
            pass_kind=pass_kind,
            model=getattr(model, "model_key", "unknown"),
            profile=profile,
            sweep_resolution=sweep_resolution,
            gaussian_noise_std=gaussian_noise_std,
            crosstalk_alpha=crosstalk_alpha,
            processed_samples=processed_samples,
            total_samples=expected_total_samples,
            pass_elapsed_seconds=float(elapsed_offset_seconds) + max(0.0, now - started_at),
            batch_index=get_batch_index(),
            runtime_module_key=module_key,
            runtime_stage=stage,
            runtime_detail=f"call_index={call_index} {detail}".strip(),
        )
        last_emit_at = now

    runtime_cfg.progress_callback = _runtime_progress_callback
    return runtime_cfg, previous_callback


def _is_image_file(path: Path) -> bool:
    return path.suffix.lower() in IMG_EXTENSIONS


def _find_imagenet_samples(root_dir: str) -> tuple[list[tuple[str, int]], dict[str, int]]:
    root = Path(root_dir)
    if not root.is_dir():
        raise SystemExit(f"ImageNet val directory not found: {root}")
    class_dirs = [d for d in root.iterdir() if d.is_dir()]
    class_dirs.sort(key=lambda p: p.name)
    synset_dirs = [d for d in class_dirs if IMAGENET_SYNSET_DIR_RE.fullmatch(d.name)]
    if len(synset_dirs) != len(class_dirs):
        nested_val = root / "val"
        nested_val_synsets = []
        if nested_val.is_dir():
            nested_val_synsets = sorted(
                [
                    d
                    for d in nested_val.iterdir()
                    if d.is_dir() and IMAGENET_SYNSET_DIR_RE.fullmatch(d.name)
                ],
                key=lambda p: p.name,
            )
        if nested_val_synsets:
            raise SystemExit(
                "ImageNet val root must point at the synset directory layer. "
                f"Received {root}, pass {nested_val} instead."
            )
        raise SystemExit(
            "ImageNet val root must contain synset-named directories like n01440764."
        )
    class_dirs = synset_dirs
    class_to_idx = {cls.name: idx for idx, cls in enumerate(class_dirs)}
    samples: list[tuple[str, int]] = []
    for cls_dir in class_dirs:
        for path in sorted(cls_dir.rglob("*")):
            if path.is_file() and _is_image_file(path):
                samples.append((str(path), class_to_idx[cls_dir.name]))
    if not samples:
        raise SystemExit(f"No images found under {root}")
    return samples, class_to_idx


def _load_imagenet_manifest(manifest_path: str) -> list[tuple[str, int]]:
    path = Path(manifest_path)
    if not path.is_file():
        raise SystemExit(f"ImageNet manifest not found: {path}")
    samples: list[tuple[str, int]] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "path" not in reader.fieldnames or "label" not in reader.fieldnames:
            raise SystemExit(f"Manifest must have columns path,label: {path}")
        for row in reader:
            image_path = (row.get("path") or "").strip()
            label_str = (row.get("label") or "").strip()
            if not image_path:
                continue
            samples.append((image_path, int(label_str)))
    if not samples:
        raise SystemExit(f"No samples in manifest: {path}")
    return samples


class OpenCVImageNetDataset:
    """Minimal OpenCV ImageNet dataset for MLX inference."""

    def __init__(
        self,
        root_dir: str,
        *,
        manifest_path: str | None = None,
        resize_size: int,
        center_crop_size: int,
        percentage: float,
        seed: int,
        enable_mean_std: bool,
        mean_std_mean: list[float],
        mean_std_std: list[float],
        input_color_order: str,
        model_color_order: str,
        input_scale: float,
    ) -> None:
        if manifest_path:
            samples = _load_imagenet_manifest(manifest_path)
        else:
            samples, _ = _find_imagenet_samples(root_dir)
        if percentage < 100.0:
            rng = random.Random(seed)
            indices = list(range(len(samples)))
            rng.shuffle(indices)
            keep = max(1, int(len(indices) * (percentage / 100.0)))
            samples = [samples[i] for i in indices[:keep]]
        self.samples = samples
        self.resize_size = resize_size
        self.center_crop_size = center_crop_size
        self.enable_mean_std = enable_mean_std
        self.input_color_order = input_color_order
        self.model_color_order = model_color_order
        self.input_scale = input_scale
        self.manifest_root = (
            Path(manifest_path).resolve().parent if manifest_path else None
        )
        self._mean = None
        self._std = None
        if enable_mean_std:
            self._mean = np.array(mean_std_mean, dtype="float32").reshape(1, 1, 3)
            self._std = np.array(mean_std_std, dtype="float32").reshape(1, 1, 3)

    def __len__(self) -> int:
        return len(self.samples)

    def _resize_shorter_side(self, image: np.ndarray, size: int) -> np.ndarray:
        h, w = image.shape[:2]
        if h < w:
            new_h = size
            new_w = int(round(w * size / h))
        else:
            new_w = size
            new_h = int(round(h * size / w))
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    def _center_crop(self, image: np.ndarray, crop_size: int) -> np.ndarray:
        h, w = image.shape[:2]
        top = max(0, (h - crop_size) // 2)
        left = max(0, (w - crop_size) // 2)
        return image[top : top + crop_size, left : left + crop_size]

    def _candidate_image_paths(self, image_path: str) -> list[Path]:
        raw_path = Path(image_path)
        candidates: list[Path] = []
        if raw_path.is_absolute():
            candidates.append(raw_path)
        else:
            candidates.append(raw_path)
            candidates.append(ROOT_DIR / raw_path)
            if self.manifest_root is not None:
                candidates.append(self.manifest_root / raw_path)
        deduped: list[Path] = []
        seen: set[str] = set()
        for candidate in candidates:
            key = str(candidate)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(candidate)
        return deduped

    def _read_image(self, image_path: str) -> np.ndarray | None:
        candidates = self._candidate_image_paths(image_path)
        for attempt in range(2):
            for candidate in candidates:
                image = cv2.imread(str(candidate), cv2.IMREAD_COLOR)
                if image is not None:
                    return image
            if attempt == 0:
                time.sleep(0.02)
        return None

    def __getitem__(self, idx: int) -> tuple[np.ndarray, int]:
        image_path, target = self.samples[idx]
        image = self._read_image(image_path)
        if image is None:
            candidates = ", ".join(str(path) for path in self._candidate_image_paths(image_path))
            raise RuntimeError(f"Failed to read image: {image_path} (candidates: {candidates})")
        image = self._resize_shorter_side(image, self.resize_size)
        image = self._center_crop(image, self.center_crop_size)
        image = image.astype("float32") / 255.0
        if self.enable_mean_std:
            image = (image - self._mean) / self._std
        if self.input_color_order != self.model_color_order:
            image = image[:, :, ::-1]
        if self.input_scale != 1.0:
            image = image * float(self.input_scale)
        return image, int(target)


def _resolve_total_samples(
    dataset: OpenCVImageNetDataset,
    *,
    max_samples: int | None,
) -> int:
    return len(dataset) if max_samples is None else min(len(dataset), int(max_samples))


def _load_batch_slice(
    dataset: OpenCVImageNetDataset,
    start: int,
    end: int,
) -> tuple[np.ndarray, np.ndarray]:
    images: list[np.ndarray] = []
    targets: list[int] = []
    for idx in range(start, end):
        image, target = dataset[idx]
        images.append(image)
        targets.append(target)
    return np.stack(images, axis=0), np.array(targets, dtype=np.int64)


def _batch_iter(
    dataset: OpenCVImageNetDataset,
    *,
    batch_size: int,
    max_samples: int | None = None,
    workers: int = 0,
    prefetch_batches: int = 1,
    start_sample: int = 0,
):
    total = _resolve_total_samples(dataset, max_samples=max_samples)
    start_offset = max(0, min(int(start_sample), int(total)))
    batch_ranges = [
        (start, min(total, start + batch_size))
        for start in range(start_offset, total, batch_size)
    ]
    if workers <= 0 or prefetch_batches <= 1:
        for start, end in batch_ranges:
            yield _load_batch_slice(dataset, start, end)
        return

    inflight_limit = max(1, min(len(batch_ranges), int(prefetch_batches)))
    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as executor:
        pending = deque()
        range_iter = iter(batch_ranges)

        def submit_next() -> bool:
            try:
                start, end = next(range_iter)
            except StopIteration:
                return False
            pending.append(
                executor.submit(_load_batch_slice, dataset, start, end)
            )
            return True

        for _ in range(inflight_limit):
            if not submit_next():
                break

        while pending:
            future = pending.popleft()
            yield future.result()
            submit_next()


def _maybe_print_percent_progress(
    *,
    prefix: str,
    processed_samples: int,
    total_samples: int | None,
    next_percent_marker: int,
    progress_callback=None,
) -> int:
    if total_samples is None or total_samples <= 0:
        return next_percent_marker
    percent = (100.0 * float(processed_samples)) / float(total_samples)
    while percent >= next_percent_marker:
        marker = min(next_percent_marker, 100)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(
            f"{timestamp} [progress] {prefix} {marker}% ({processed_samples}/{total_samples})",
            flush=True,
        )
        if progress_callback is not None:
            progress_callback(marker)
        next_percent_marker += 5
    return next_percent_marker


def _compute_topk_correct_counts_mlx(
    logits,
    targets,
    *,
    max_k: int = 5,
):
    import mlx.core as mx

    resolved_k = max(1, min(int(max_k), int(logits.shape[1])))
    top1_indices = mx.argmax(logits, axis=1)
    top1_correct = mx.sum(top1_indices == targets)
    topk_indices = mx.argsort(logits, axis=1)[:, -resolved_k:]
    topk_correct = mx.sum(mx.any(topk_indices == targets[:, None], axis=1))
    return top1_correct, topk_correct


def _mlx_scalar_to_int(value: Any) -> int:
    item = getattr(value, "item", None)
    if callable(item):
        return int(item())
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        payload = tolist()
        if isinstance(payload, list):
            if not payload:
                return 0
            return int(payload[0])
        return int(payload)
    return int(value)


def _release_mlx_memory(mx_module: Any | None = None) -> None:
    gc.collect()
    if mx_module is None:
        return
    clear_cache = getattr(mx_module, "clear_cache", None)
    if callable(clear_cache):
        try:
            clear_cache()
            return
        except Exception:
            pass
    metal = getattr(mx_module, "metal", None)
    if metal is None:
        return
    metal_clear_cache = getattr(metal, "clear_cache", None)
    if callable(metal_clear_cache):
        try:
            metal_clear_cache()
        except Exception:
            pass


def evaluate_model(
    model,
    dataset: OpenCVImageNetDataset,
    *,
    batch_size: int,
    max_samples: int | None = None,
    workers: int = 0,
    prefetch_batches: int = 1,
    log_progress: bool = True,
    progress_prefix: str | None = None,
    progress_recorder: ProgressRecorder | None = None,
    total_passes: int | None = None,
    pass_index: int | None = None,
    pass_kind: str = "eval",
    profile: str | None = None,
    sweep_resolution: str | None = None,
    gaussian_noise_std: float | None = None,
    crosstalk_alpha: float | None = None,
    progress_heartbeat_interval_seconds: float | None = DEFAULT_PROGRESS_HEARTBEAT_INTERVAL_SECONDS,
    stall_timeout_seconds: float | None = DEFAULT_STALL_TIMEOUT_SECONDS,
    pathological_min_samples_per_hour: float | None = DEFAULT_PATHOLOGICAL_MIN_SAMPLES_PER_HOUR,
    pathological_max_seconds_per_sample: float | None = DEFAULT_PATHOLOGICAL_MAX_SECONDS_PER_SAMPLE,
    pathological_max_eta_current_rate_seconds: float | None = DEFAULT_PATHOLOGICAL_MAX_ETA_CURRENT_RATE_SECONDS,
    pathological_min_processed_samples: int | None = DEFAULT_PATHOLOGICAL_MIN_PROCESSED_SAMPLES,
    pathological_min_elapsed_seconds: float | None = DEFAULT_PATHOLOGICAL_MIN_ELAPSED_SECONDS,
    seed: int = 0,
    resume_checkpoint_path: Path | None = None,
    resume_row_identity: tuple[str, ...] | None = None,
    resume_enabled: bool = False,
    runtime_policy_fingerprint: str | None = None,
    semantic_fingerprint: str | None = None,
) -> tuple[float, float, dict[str, float | int]]:
    import mlx.core as mx

    started_at = time.perf_counter()
    expected_total_samples = _resolve_total_samples(
        dataset,
        max_samples=max_samples,
    )
    loaded_resume_checkpoint = None
    if resume_enabled and resume_checkpoint_path is not None and resume_row_identity is not None:
        loaded_resume_checkpoint = _load_pass_resume_checkpoint(
            Path(resume_checkpoint_path),
            row_identity=resume_row_identity,
            expected_total_samples=expected_total_samples,
            batch_size=batch_size,
            max_samples=max_samples,
            runtime_policy_fingerprint=runtime_policy_fingerprint,
            semantic_fingerprint=semantic_fingerprint,
        )
    total_samples = (
        int(loaded_resume_checkpoint.get("processed_samples") or 0)
        if loaded_resume_checkpoint is not None
        else 0
    )
    top1_correct = (
        int(loaded_resume_checkpoint.get("top1_correct") or 0)
        if loaded_resume_checkpoint is not None
        else 0
    )
    top5_correct = (
        int(loaded_resume_checkpoint.get("top5_correct") or 0)
        if loaded_resume_checkpoint is not None
        else 0
    )
    elapsed_offset_seconds = (
        float(loaded_resume_checkpoint.get("pass_elapsed_seconds") or 0.0)
        if loaded_resume_checkpoint is not None
        else 0.0
    )
    if loaded_resume_checkpoint is not None:
        _restore_mlx_random_state(
            mx,
            loaded_resume_checkpoint.get("mlx_random_state"),
        )
        print(
            "[resume] loaded pass checkpoint "
            f"path={resume_checkpoint_path} processed={total_samples}/{expected_total_samples} "
            f"pass_kind={pass_kind}",
            flush=True,
        )
    if progress_recorder is not None and total_passes is not None and pass_index is not None:
        progress_recorder.emit_pass_event(
            event="pass_start",
            total_passes=total_passes,
            pass_index=pass_index,
            pass_kind=pass_kind,
            model=getattr(model, "model_key", "unknown"),
            profile=profile,
            sweep_resolution=sweep_resolution,
            gaussian_noise_std=gaussian_noise_std,
            crosstalk_alpha=crosstalk_alpha,
            processed_samples=total_samples,
            total_samples=expected_total_samples,
            pass_elapsed_seconds=elapsed_offset_seconds,
            resume_checkpoint_path=(
                str(resume_checkpoint_path)
                if resume_checkpoint_path is not None
                else None
            ),
            resume_checkpoint_loaded=loaded_resume_checkpoint is not None,
        )
    stall_timeout_value = (
        None
        if stall_timeout_seconds is None or float(stall_timeout_seconds) <= 0.0
        else float(stall_timeout_seconds)
    )
    heartbeat_interval_value = (
        None
        if progress_heartbeat_interval_seconds is None
        or float(progress_heartbeat_interval_seconds) <= 0.0
        else float(progress_heartbeat_interval_seconds)
    )
    previous_alarm_handler = None
    current_batch_index = (
        int(loaded_resume_checkpoint.get("batch_index") or 0)
        if loaded_resume_checkpoint is not None
        else 0
    )
    last_heartbeat_at = time.perf_counter()
    if stall_timeout_value is not None and hasattr(signal, "SIGALRM") and hasattr(signal, "setitimer"):
        def _stall_alarm_handler(_signum, _frame) -> None:
            raise EvaluationStalled(
                f"no batch completed for {stall_timeout_value:.1f}s during {pass_kind}"
            )

        previous_alarm_handler = signal.getsignal(signal.SIGALRM)
        signal.signal(signal.SIGALRM, _stall_alarm_handler)
        signal.setitimer(signal.ITIMER_REAL, stall_timeout_value)
    runtime_progress_attachment = _attach_runtime_progress_callback(
        model=model,
        progress_recorder=progress_recorder,
        total_passes=total_passes,
        pass_index=pass_index,
        pass_kind=pass_kind,
        expected_total_samples=expected_total_samples,
        profile=profile,
        sweep_resolution=sweep_resolution,
        gaussian_noise_std=gaussian_noise_std,
        crosstalk_alpha=crosstalk_alpha,
        stall_timeout_value=stall_timeout_value,
        heartbeat_interval_value=heartbeat_interval_value,
        pathological_min_samples_per_hour=pathological_min_samples_per_hour,
        pathological_max_seconds_per_sample=pathological_max_seconds_per_sample,
        pathological_max_eta_current_rate_seconds=pathological_max_eta_current_rate_seconds,
        pathological_min_processed_samples=pathological_min_processed_samples,
        pathological_min_elapsed_seconds=pathological_min_elapsed_seconds,
        pathological_inflight_sample_floor=max(1, int(batch_size)),
        started_at=started_at,
        get_processed_samples=lambda: total_samples,
        get_batch_index=lambda: current_batch_index or None,
        elapsed_offset_seconds=elapsed_offset_seconds,
    )
    next_percent_marker = 5
    if expected_total_samples > 0 and total_samples > 0:
        completed_percent = (100.0 * float(total_samples)) / float(expected_total_samples)
        while completed_percent >= next_percent_marker:
            next_percent_marker += 5
    try:
        for batch_idx, (images_np, targets_np) in enumerate(
            _batch_iter(
                dataset,
                batch_size=batch_size,
                max_samples=max_samples,
                workers=workers,
                prefetch_batches=prefetch_batches,
                start_sample=total_samples,
            ),
            start=current_batch_index + 1,
        ):
            current_batch_index = int(batch_idx)
            _raise_if_stop_requested("batch_start")
            images_mx = mx.array(images_np)
            targets_mx = mx.array(targets_np)
            logits = model(images_mx)
            top1_batch, top5_batch = _compute_topk_correct_counts_mlx(
                logits,
                targets_mx,
                max_k=5,
            )
            mx.eval(top1_batch)
            mx.eval(top5_batch)
            top1_correct += _mlx_scalar_to_int(top1_batch)
            top5_correct += _mlx_scalar_to_int(top5_batch)
            batch_samples = int(targets_np.shape[0])
            total_samples += batch_samples
            if stall_timeout_value is not None and hasattr(signal, "setitimer"):
                signal.setitimer(signal.ITIMER_REAL, stall_timeout_value)
            progress_label = progress_prefix or "eval"
            elapsed_now = elapsed_offset_seconds + max(0.0, time.perf_counter() - started_at)
            if resume_checkpoint_path is not None and resume_row_identity is not None:
                _write_pass_resume_checkpoint(
                    Path(resume_checkpoint_path),
                    row_identity=resume_row_identity,
                    pass_kind=pass_kind,
                    model_key=getattr(model, "model_key", "unknown"),
                    seed=int(seed),
                    batch_size=int(batch_size),
                    max_samples=max_samples,
                    expected_total_samples=expected_total_samples,
                    processed_samples=total_samples,
                    top1_correct=top1_correct,
                    top5_correct=top5_correct,
                    batch_index=batch_idx,
                    pass_elapsed_seconds=elapsed_now,
                    mlx_random_state=_snapshot_mlx_random_state(mx),
                    completed=total_samples >= expected_total_samples,
                    runtime_policy_fingerprint=runtime_policy_fingerprint,
                    semantic_fingerprint=semantic_fingerprint,
                )
            if (
                progress_recorder is not None
                and total_passes is not None
                and pass_index is not None
                and (
                    heartbeat_interval_value is None
                    or (time.perf_counter() - last_heartbeat_at) >= heartbeat_interval_value
                    or batch_idx == 1
                )
            ):
                progress_recorder.emit_pass_event(
                    event="pass_batch_complete",
                    total_passes=total_passes,
                    pass_index=pass_index,
                    pass_kind=pass_kind,
                    model=getattr(model, "model_key", "unknown"),
                    profile=profile,
                    sweep_resolution=sweep_resolution,
                    gaussian_noise_std=gaussian_noise_std,
                    crosstalk_alpha=crosstalk_alpha,
                    processed_samples=total_samples,
                    total_samples=expected_total_samples,
                    pass_elapsed_seconds=elapsed_now,
                    batch_index=batch_idx,
                    batch_samples=batch_samples,
                )
                last_heartbeat_at = time.perf_counter()
            pathological_reason = _pathological_slow_reason(
                processed_samples=total_samples,
                total_samples=expected_total_samples,
                elapsed_seconds=elapsed_now,
                min_samples_per_hour=pathological_min_samples_per_hour,
                max_seconds_per_sample=pathological_max_seconds_per_sample,
                max_eta_current_rate_seconds=pathological_max_eta_current_rate_seconds,
                min_processed_samples=pathological_min_processed_samples,
                min_elapsed_seconds=pathological_min_elapsed_seconds,
            )
            if pathological_reason:
                raise EvaluationPathologicallySlow(pathological_reason)
            next_percent_marker = _maybe_print_percent_progress(
                prefix=progress_label,
                processed_samples=total_samples,
                total_samples=expected_total_samples,
                next_percent_marker=next_percent_marker,
                progress_callback=(
                    None
                    if progress_recorder is None or total_passes is None or pass_index is None
                    else lambda marker: progress_recorder.emit_pass_event(
                        event="pass_progress",
                        total_passes=total_passes,
                        pass_index=pass_index,
                        pass_kind=pass_kind,
                        model=getattr(model, "model_key", "unknown"),
                        profile=profile,
                        sweep_resolution=sweep_resolution,
                        gaussian_noise_std=gaussian_noise_std,
                        crosstalk_alpha=crosstalk_alpha,
                        processed_samples=total_samples,
                        total_samples=expected_total_samples,
                        pass_elapsed_seconds=elapsed_offset_seconds + max(0.0, time.perf_counter() - started_at),
                        milestone_percent=marker,
                        batch_index=batch_idx,
                        batch_samples=batch_samples,
                    )
                ),
            )
            if log_progress and batch_idx % 32 == 0:
                elapsed = elapsed_offset_seconds + max(0.0, time.perf_counter() - started_at)
                print(
                    f"[eval-status] {progress_label} processed={total_samples}/{expected_total_samples} elapsed={elapsed:5.2f}s",
                    flush=True,
                )
            del logits
            del top1_batch
            del top5_batch
            del images_mx
            del targets_mx
            del images_np
            del targets_np
            if batch_idx % MLX_MEMORY_RELEASE_INTERVAL == 0:
                _release_mlx_memory(mx)
            _raise_if_stop_requested("batch_end")
    except (EvaluationStalled, EvaluationPathologicallySlow) as exc:
        if progress_recorder is not None and total_passes is not None and pass_index is not None:
            progress_recorder.emit_pass_event(
                event="pass_failed",
                total_passes=total_passes,
                pass_index=pass_index,
                pass_kind=pass_kind,
                model=getattr(model, "model_key", "unknown"),
                profile=profile,
                sweep_resolution=sweep_resolution,
                gaussian_noise_std=gaussian_noise_std,
                crosstalk_alpha=crosstalk_alpha,
                processed_samples=total_samples,
                total_samples=expected_total_samples,
                pass_elapsed_seconds=elapsed_offset_seconds + max(0.0, time.perf_counter() - started_at),
                batch_index=current_batch_index or None,
                failure_kind=(
                    "stall_timeout"
                    if isinstance(exc, EvaluationStalled)
                    else "pathological_slow"
                ),
                failure_message=str(exc),
            )
        raise
    finally:
        if runtime_progress_attachment is not None:
            runtime_cfg, previous_callback = runtime_progress_attachment
            runtime_cfg.progress_callback = previous_callback
        if stall_timeout_value is not None and hasattr(signal, "setitimer"):
            signal.setitimer(signal.ITIMER_REAL, 0.0)
        if previous_alarm_handler is not None and hasattr(signal, "SIGALRM"):
            signal.signal(signal.SIGALRM, previous_alarm_handler)
    _release_mlx_memory(mx)
    elapsed_s = elapsed_offset_seconds + max(0.0, time.perf_counter() - started_at)
    if total_samples <= 0:
        raise RuntimeError("No samples were processed during evaluation.")
    if expected_total_samples and total_samples >= expected_total_samples and next_percent_marker <= 100:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(
            f"{timestamp} [progress] {(progress_prefix or 'eval')} 100% ({total_samples}/{expected_total_samples})",
            flush=True,
        )
    top1 = 100.0 * float(top1_correct) / float(total_samples)
    top5 = 100.0 * float(top5_correct) / float(total_samples)
    if resume_checkpoint_path is not None and resume_row_identity is not None:
        _write_pass_resume_checkpoint(
            Path(resume_checkpoint_path),
            row_identity=resume_row_identity,
            pass_kind=pass_kind,
            model_key=getattr(model, "model_key", "unknown"),
            seed=int(seed),
            batch_size=int(batch_size),
            max_samples=max_samples,
            expected_total_samples=expected_total_samples,
            processed_samples=total_samples,
            top1_correct=top1_correct,
            top5_correct=top5_correct,
            batch_index=current_batch_index or None,
            pass_elapsed_seconds=elapsed_s,
            mlx_random_state=_snapshot_mlx_random_state(mx),
            completed=total_samples >= expected_total_samples,
            runtime_policy_fingerprint=runtime_policy_fingerprint,
            semantic_fingerprint=semantic_fingerprint,
        )
    if progress_recorder is not None and total_passes is not None and pass_index is not None:
        progress_recorder.emit_pass_event(
            event="pass_complete",
            total_passes=total_passes,
            pass_index=pass_index,
            pass_kind=pass_kind,
            model=getattr(model, "model_key", "unknown"),
            profile=profile,
            sweep_resolution=sweep_resolution,
            gaussian_noise_std=gaussian_noise_std,
            crosstalk_alpha=crosstalk_alpha,
            processed_samples=total_samples,
            total_samples=expected_total_samples,
            pass_elapsed_seconds=elapsed_s,
            milestone_percent=100 if expected_total_samples else None,
        )
    stats = {
        "processed_samples": total_samples,
        "elapsed_s": elapsed_s,
    }
    return top1, top5, stats


def _get_git_hash(repo_root: Path) -> str | None:
    try:
        output = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(repo_root),
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return None
    return output.decode("utf-8", errors="ignore").strip() or None


def _build_result_row(
    *,
    base_row: dict[str, object],
    baseline: bool,
    model_key: str,
    input_size: int,
    quant_bits: int | None,
    noise_sigma_lsb: float,
    crosstalk_alpha: float,
    drift_lsb: float,
    noise_correlation: float,
    burst_error_prob: float,
    burst_error_scale_lsb: float,
    burst_span: int,
    top1: float | None,
    top5: float | None,
    top1_delta: float | None,
    top5_delta: float | None,
    measured_pass_elapsed_s: float | None,
    measured_processed_samples: int | None,
    latency_ms_per_sample: float | None,
    measurement_window: str,
    seed: int,
    notes: str | None,
) -> dict[str, object]:
    row = dict(base_row)
    row.update(
        {
            "baseline": baseline,
            "model": model_key,
            "input_size": input_size,
            "quant_bits": quant_bits,
            "noise_sigma_lsb": noise_sigma_lsb,
            "gaussian_noise_std": noise_sigma_lsb,
            "crosstalk_alpha": crosstalk_alpha,
            "drift_lsb": drift_lsb,
            "noise_correlation": noise_correlation,
            "burst_error_prob": burst_error_prob,
            "burst_error_scale_lsb": burst_error_scale_lsb,
            "burst_span": burst_span,
            "top1": top1,
            "top5": top5,
            "top1_delta": top1_delta,
            "top5_delta": top5_delta,
            "measured_pass_elapsed_s": measured_pass_elapsed_s,
            "measured_processed_samples": measured_processed_samples,
            "latency_ms_per_sample": latency_ms_per_sample,
            "measurement_window": measurement_window,
            "seed": seed,
            "notes": notes,
        }
    )
    return row


def write_results(path: Path, rows: list[dict[str, object]], *, append: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists()
    mode = "a" if append and file_exists else "w"
    if file_exists and mode == "w":
        backup = backup_existing_file(path)
        if backup:
            print(f"Existing results moved to {backup}", flush=True)
    with path.open(mode, newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDNAMES)
        if mode == "w":
            writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in RESULT_FIELDNAMES})


def write_sparse_layer_rows(path: Path, rows: list[dict[str, object]], *, append: bool) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append and path.exists() else "w"
    with path.open(mode, newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SPARSE_ACTIVITY_LAYER_FIELDNAMES)
        if mode == "w":
            writer.writeheader()
        for row in rows:
            writer.writerow(
                {field: row.get(field) for field in SPARSE_ACTIVITY_LAYER_FIELDNAMES}
            )


def prepare_sparse_layer_rows_path(
    path: Path | None,
    *,
    append: bool,
    resume: bool,
) -> None:
    if path is None or append or resume or not path.exists():
        return
    backup = backup_existing_file(path)
    if backup:
        print(f"Existing sparse activity layer rows moved to {backup}", flush=True)


def _parse_optional_json_map(raw: str | None) -> dict[str, float] | None:
    if not raw:
        return None
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise SystemExit("Expected a JSON object for per-layer overrides.")
    return {str(key): float(value) for key, value in payload.items()}


def _resolve_batch_size(requested: int | None) -> int:
    if requested is not None and requested > 0:
        return int(requested)
    return 64


def _resolve_quantized_bitstream_batch_policy(
    *,
    requested_batch_size: int,
    runtime_metadata: dict[str, object] | None,
    model_key: str,
    target_module_keys_raw: str | None,
    max_eval_samples: int | None,
) -> dict[str, int | bool | str]:
    metadata = dict(runtime_metadata or {})
    if str(metadata.get("execution_semantics") or "").strip() != "bitstream":
        return {
            "enabled": False,
            "requested_batch_size": int(requested_batch_size),
            "safe_batch_cap": int(requested_batch_size),
            "quantized_batch_size": int(requested_batch_size),
            "estimated_working_set_bytes": 0,
            "target_count_source": "",
        }
    active_target_module_count = int(
        metadata.get("bitstream_runtime_active_target_module_count") or 0
    )
    if active_target_module_count <= 0:
        guardrail = build_bitstream_runtime_guardrail(
            model_key=model_key,
            surface_scope=str(metadata.get("bitstream_surface_scope") or ""),
            target_module_keys_raw=target_module_keys_raw,
            explicit_eval_batch_size=int(requested_batch_size),
            stream_length=int(metadata.get("bitstream_stream_length") or 64),
            root_dir=ROOT_DIR.parent,
        )
        active_target_module_count = int(guardrail["active_target_module_count"])
        target_count_source = str(guardrail["target_count_source"] or "")
    else:
        target_count_source = "runtime_metadata"
        guardrail = build_bitstream_runtime_guardrail(
            model_key=model_key,
            surface_scope=str(metadata.get("bitstream_surface_scope") or ""),
            target_module_keys_raw=target_module_keys_raw,
            explicit_eval_batch_size=int(requested_batch_size),
            stream_length=int(metadata.get("bitstream_stream_length") or 64),
            root_dir=ROOT_DIR.parent,
        )
    safe_batch_cap = safe_quantized_batch_cap(
        active_target_module_count=active_target_module_count,
        surface_scope=str(metadata.get("bitstream_surface_scope") or ""),
    )
    quantized_batch_size = min(int(requested_batch_size), int(safe_batch_cap))
    if max_eval_samples is not None:
        quantized_batch_size = min(int(quantized_batch_size), max(1, int(max_eval_samples)))
    return {
        "enabled": True,
        "requested_batch_size": int(requested_batch_size),
        "safe_batch_cap": int(safe_batch_cap),
        "quantized_batch_size": max(1, int(quantized_batch_size)),
        "estimated_working_set_bytes": int(guardrail["estimated_working_set_bytes"]),
        "target_count_source": target_count_source,
    }


def _weights_npz_source(args: argparse.Namespace) -> str:
    if args.weights_npz:
        return "explicit_weights_npz_arg"
    if args.force_reexport_mlx_weights:
        return "exported_from_torch_checkpoint_forced"
    if args.weights_dir or args.weights_override or args.mlx_weights_dir:
        return "exported_from_torch_checkpoint_cached"
    return "default_mlx_weights_cache"


def _calibration_scale_source(*, calibration_enabled: bool) -> str:
    if not calibration_enabled:
        return "disabled_no_calibration_scale"
    return "baseline_eval_pass_calibrated_scale"


def _resolve_prefetch_batches(worker_count: int) -> int:
    if worker_count <= 0:
        return 1
    return max(1, min(MAX_PREFETCH_BATCHES, int(worker_count)))


def _planned_pass_count(
    *,
    model_count: int,
    noise_sigmas: list[float],
    crosstalk_alphas: list[float],
    pass_mode: str,
) -> int:
    quantized_pass_count = len(noise_sigmas) * len(crosstalk_alphas)
    if pass_mode == PASS_MODE_BASELINE_ONLY:
        per_model = 1
    elif pass_mode == PASS_MODE_QUANTIZED_ONLY:
        per_model = quantized_pass_count
    else:
        per_model = 1 + quantized_pass_count
    return int(model_count) * int(per_model)


def _load_baseline_reference_row(
    *,
    baseline_reference_csv: str,
    baseline_reference_run_id: str,
    model_key: str,
) -> dict[str, str]:
    path = Path(baseline_reference_csv)
    if not path.is_file():
        raise SystemExit(f"Missing --baseline_reference_csv: {path}")
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    matches = [
        row
        for row in rows
        if str(row.get("run_id") or "").strip() == str(baseline_reference_run_id).strip()
        and str(row.get("model") or "").strip() == str(model_key).strip()
        and str(row.get("baseline") or "").strip().lower() == "true"
    ]
    if not matches:
        raise SystemExit(
            "Could not find a baseline reference row for "
            f"run_id={baseline_reference_run_id!r}, model={model_key!r} in {path}."
        )
    return matches[-1]


def _baseline_metrics_from_reference(row: dict[str, str]) -> tuple[float, float]:
    try:
        top1 = float(row["top1"])
        top5 = float(row["top5"])
    except (KeyError, TypeError, ValueError) as exc:
        raise SystemExit(
            "Baseline reference row is missing numeric top1/top5 values."
        ) from exc
    return top1, top5


def _build_metadata_payload(
    *,
    run_id: str,
    model_keys: list[str],
    results_path: Path,
    noise_sigmas: list[float],
    crosstalk_alphas: list[float],
    resolved_batch_size: int,
    calibration_enabled: bool,
    calibration_scale_source: str,
    model_contexts: list[dict[str, object]],
    runtime_metadata: dict[str, object] | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "run_id": run_id,
        "device": "mps",
        "accuracy_backend": "mlx",
        "engine": "mlx",
        "models": model_keys,
        "results_csv": str(results_path),
        "noise_sigmas": [float(value) for value in noise_sigmas],
        "crosstalk_alphas": [float(value) for value in crosstalk_alphas],
        "resolved_batch_size": int(resolved_batch_size),
        "calibration_enabled": bool(calibration_enabled),
        "calibration_scale_source": calibration_scale_source,
        "model_contexts": model_contexts,
    }
    for key, value in (runtime_metadata or {}).items():
        if key == "target_module_keys":
            continue
        payload[key] = value
    if len(model_contexts) == 1:
        payload["weights_npz"] = model_contexts[0]["weights_npz"]
        payload["weights_npz_source"] = model_contexts[0]["weights_npz_source"]
    return payload


def _resolve_noise_sigmas(args: argparse.Namespace) -> list[float]:
    noise_sigmas = parse_list(args.gaussian_noise_std, float) or [0.0]
    legacy_noise_sigmas = parse_list(args.noise_sigma_lsb, float) or [0.0]
    if legacy_noise_sigmas != noise_sigmas:
        default_noise_sigmas = parse_list(DEFAULT_NOISE_SIGMA_LSB, float) or [0.0]
        default_gaussian_noise_stds = parse_list(DEFAULT_GAUSSIAN_NOISE_STD, float) or [0.0]
        legacy_overridden = legacy_noise_sigmas != default_noise_sigmas
        gaussian_overridden = noise_sigmas != default_gaussian_noise_stds
        if legacy_overridden and gaussian_overridden:
            raise SystemExit(
                "--noise_sigma_lsb and --gaussian_noise_std encode the same active "
                "noise axis in the MLX harness and must match when both are set."
            )
        if legacy_overridden:
            noise_sigmas = legacy_noise_sigmas
    return noise_sigmas


def _mlx_runtime_imports() -> tuple[Any, Any, Any, Any]:
    from accuracy.mlx_mobilevit import (
        BitstreamExecutionConfig,
        MLXPerturbationConfig,
        SparseActivityRecorder,
        build_mlx_mobilevit,
    )

    return BitstreamExecutionConfig, MLXPerturbationConfig, SparseActivityRecorder, build_mlx_mobilevit


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate MobileViT accuracy with MLX on Apple Silicon.",
    )
    parser.add_argument("--imagenet_val", required=True)
    parser.add_argument("--imagenet_manifest", default=None)
    parser.add_argument("--models", default="mobilevit_xxs")
    parser.add_argument("--results_csv", required=True)
    parser.add_argument("--append", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from an existing results CSV by skipping already written rows.",
    )
    parser.add_argument("--run_id", default="mlx_eval")
    parser.add_argument("--source_run_id", default=None)
    parser.add_argument("--experiment_id", default=None)
    parser.add_argument("--profile", default=None)
    parser.add_argument("--sweep_resolution", default=None)
    parser.add_argument("--workload", default="W0_mobilevit_imagenet")
    parser.add_argument("--git_hash", default=None)
    parser.add_argument("--config_snapshot", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--percentage", type=float, default=100.0)
    parser.add_argument("--eval_batch_size", type=int, default=None)
    parser.add_argument("--max_eval_samples", type=int, default=None)
    parser.add_argument(
        "--pass_mode",
        choices=PASS_MODE_CHOICES,
        default=PASS_MODE_PAIRED,
    )
    parser.add_argument("--baseline_reference_csv", default=None)
    parser.add_argument("--baseline_reference_run_id", default=None)
    parser.add_argument("--workers", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--resize_size", type=int, default=None)
    parser.add_argument("--center_crop_size", type=int, default=None)
    parser.add_argument("--input_color_order", choices=["rgb", "bgr"], default="rgb")
    parser.add_argument("--data_color_order", choices=["rgb", "bgr"], default="bgr")
    parser.add_argument("--input_scale", type=float, default=1.0)
    parser.add_argument("--enable_mean_std", action="store_true")
    parser.add_argument("--mean_std_mean", default="0.485,0.456,0.406")
    parser.add_argument("--mean_std_std", default="0.229,0.224,0.225")
    parser.add_argument("--quant_bits", type=int, default=8)
    parser.add_argument(
        "--noise_sigma_lsb",
        "--gaussian_noise_sigma_lsb",
        dest="noise_sigma_lsb",
        default=DEFAULT_NOISE_SIGMA_LSB,
    )
    parser.add_argument(
        "--gaussian_noise_std",
        default=DEFAULT_GAUSSIAN_NOISE_STD,
        help="Alias of --noise_sigma_lsb for shared runner compatibility.",
    )
    parser.add_argument("--crosstalk_alpha", default=DEFAULT_CROSSTALK_ALPHA)
    parser.add_argument("--drift_lsb", type=float, default=0.0)
    parser.add_argument("--noise_correlation", type=float, default=0.0)
    parser.add_argument("--burst_error_prob", type=float, default=0.0)
    parser.add_argument("--burst_error_scale_lsb", type=float, default=0.0)
    parser.add_argument("--burst_span", type=int, default=1)
    parser.add_argument("--apply_det_perturbation", action="store_true")
    parser.add_argument("--det_mode", default="reorder")
    parser.add_argument("--det_bsl_max", type=float, default=None)
    parser.add_argument("--det_policy", default=None)
    parser.add_argument("--det_k_signature", default=None)
    parser.add_argument("--det_k_global", type=float, default=None)
    parser.add_argument("--det_k_by_layer_json", default=None)
    parser.add_argument("--det_prefix_error_mean", type=float, default=0.0)
    parser.add_argument("--det_prefix_error_p95", type=float, default=None)
    parser.add_argument("--det_prefix_error_by_layer_json", default=None)
    parser.add_argument("--apply_sparse_perturbation", action="store_true")
    parser.add_argument("--sparse_tau_global", type=float, default=None)
    parser.add_argument("--sparse_active_fraction", type=float, default=None)
    parser.add_argument("--sparse_target_module_keys", default=None)
    parser.add_argument("--disable_calibration", action="store_true")
    parser.add_argument("--weights_npz", default=None)
    parser.add_argument("--weights_dir", default=None)
    parser.add_argument("--weights_override", default=None)
    parser.add_argument("--mlx_weights_dir", default=None)
    parser.add_argument("--force_reexport_mlx_weights", action="store_true")
    parser.add_argument("--metadata_json", default=None)
    parser.add_argument("--progress_jsonl", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--progress_label", default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--progress_heartbeat_interval_seconds",
        type=float,
        default=DEFAULT_PROGRESS_HEARTBEAT_INTERVAL_SECONDS,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--stall_timeout_seconds",
        type=float,
        default=DEFAULT_STALL_TIMEOUT_SECONDS,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--pathological_min_samples_per_hour",
        type=float,
        default=DEFAULT_PATHOLOGICAL_MIN_SAMPLES_PER_HOUR,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--pathological_max_seconds_per_sample",
        type=float,
        default=DEFAULT_PATHOLOGICAL_MAX_SECONDS_PER_SAMPLE,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--pathological_max_eta_current_rate_seconds",
        type=float,
        default=DEFAULT_PATHOLOGICAL_MAX_ETA_CURRENT_RATE_SECONDS,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--pathological_min_processed_samples",
        type=int,
        default=DEFAULT_PATHOLOGICAL_MIN_PROCESSED_SAMPLES,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--pathological_min_elapsed_seconds",
        type=float,
        default=DEFAULT_PATHOLOGICAL_MIN_ELAPSED_SECONDS,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--sparse_activity_layers_csv", default=None)
    parser.add_argument("--device", choices=["auto", "mps"], default="mps")
    parser.add_argument("--allow_unvalidated_mps", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--enable_attention", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--enable_bitstream_pilot", action="store_true")
    parser.add_argument(
        "--bitstream_surface_scope",
        default="limited_linear_attention_pilot",
    )
    parser.add_argument(
        "--bitstream_target_module_keys",
        default=None,
        help=(
            "Optional comma-separated module keys, '@file' key list, or 'all' for "
            "all-target bitstream runtime execution."
        ),
    )
    parser.add_argument("--bitstream_generator", default="low_discrepancy")
    parser.add_argument("--bitstream_stream_length", type=int, default=64)
    parser.add_argument(
        "--bitstream_stream_reuse_policy",
        default="operand_factored_module_call_reuse",
    )
    parser.add_argument("--bitstream_encoding_mode", default="bipolar")
    parser.add_argument("--bitstream_multiplier_mode", default="xnor")
    parser.add_argument("--bitstream_accumulator_mode", default="bitcount")
    parser.add_argument(
        "--bitstream_measurement_truth_class",
        default=BITSTREAM_LIMITED_SURFACE_PILOT_TRUTH_CLASS,
    )
    parser.add_argument(
        "--bitstream_truth_class_authorization_note",
        default=None,
        help=(
            "Optional local governance note path required before a full-target bitstream "
            "runtime row may emit bitstream_model_level_measured."
        ),
    )
    parser.add_argument(
        "--bitstream_contract_note",
        default="limited_surface_runtime_pilot_not_full_model_measured",
    )
    parser.add_argument(
        "--accuracy_policy_profile_json",
        default=None,
        help=(
            "Optional JSON path or JSON object that resolves per-pass workers, batch, "
            "runtime health gates, and semantic policy."
        ),
    )
    parser.add_argument("--opencv_pipeline", action="store_true")
    return parser


def main() -> int:
    global _STOP_REQUESTED
    _STOP_REQUESTED = False
    args = _build_arg_parser().parse_args()
    _validate_requested_model_level_measured_authorization(args)
    _install_interrupt_handlers()
    if args.pass_mode == PASS_MODE_QUANTIZED_ONLY:
        if not str(args.baseline_reference_csv or "").strip():
            raise SystemExit(
                "--pass_mode quantized_only requires --baseline_reference_csv."
            )
        if not str(args.baseline_reference_run_id or "").strip():
            raise SystemExit(
                "--pass_mode quantized_only requires --baseline_reference_run_id."
            )
    if args.device not in {"auto", "mps"}:
        raise SystemExit("MLX evaluation only supports --device mps/auto on this host.")
    set_seeds(args.seed)
    model_keys = parse_model_keys(args.models)
    if not model_keys:
        raise SystemExit("No models were requested.")
    results_path = Path(args.results_csv)
    existing_row_identities = (
        load_existing_result_identities(results_path)
        if args.resume
        else set()
    )
    existing_result_rows_by_identity = (
        load_existing_result_rows_by_identity(results_path)
        if args.resume
        else {}
    )
    resume_artifact_root = _resume_artifact_root(
        progress_jsonl=args.progress_jsonl,
        results_path=results_path,
    )
    append_results = bool(args.append or (args.resume and results_path.exists()))
    noise_sigmas = _resolve_noise_sigmas(args)
    crosstalk_alphas = parse_list(args.crosstalk_alpha, float) or [0.0]
    mean_std_mean = parse_float_list(args.mean_std_mean)
    mean_std_std = parse_float_list(args.mean_std_std)
    git_hash = args.git_hash or _get_git_hash(ROOT_DIR) or "nogit"
    resolved_batch_size = _resolve_batch_size(args.eval_batch_size)
    resolved_worker_count = resolve_data_workers(args.workers, "mps")
    resolved_prefetch_batches = _resolve_prefetch_batches(resolved_worker_count)
    accuracy_policy_bundle = _load_accuracy_policy_profile(
        args.accuracy_policy_profile_json
    )
    calibration_enabled = (
        not bool(args.disable_calibration)
        and args.pass_mode != PASS_MODE_QUANTIZED_ONLY
    )
    calibration_scale_source = _calibration_scale_source(
        calibration_enabled=calibration_enabled
    )
    weights_npz_source = _weights_npz_source(args)
    bitstream_runtime_metadata = _build_bitstream_runtime_metadata(args)
    sparse_layers_path = (
        Path(args.sparse_activity_layers_csv)
        if args.sparse_activity_layers_csv
        else None
    )
    prepare_sparse_layer_rows_path(
        sparse_layers_path,
        append=bool(args.append),
        resume=bool(args.resume),
    )
    model_contexts: list[dict[str, object]] = []
    runtime_metadata_by_model: dict[str, dict[str, object]] = {}
    runtime_tuning: dict[str, dict[str, object]] = {}
    planned_passes_by_model: dict[str, int] = {}
    (
        BitstreamExecutionConfig,
        MLXPerturbationConfig,
        SparseActivityRecorder,
        build_mlx_mobilevit,
    ) = _mlx_runtime_imports()
    total_planned_passes = _planned_pass_count(
        model_count=len(model_keys),
        noise_sigmas=noise_sigmas,
        crosstalk_alphas=crosstalk_alphas,
        pass_mode=str(args.pass_mode),
    )
    progress_recorder = (
        ProgressRecorder(
            Path(args.progress_jsonl),
            label=args.progress_label,
            fresh=not args.resume,
        )
        if args.progress_jsonl
        else None
    )
    if progress_recorder is not None:
        progress_recorder.emit_command_start(
            total_passes=total_planned_passes,
            run_id=args.run_id,
            experiment_id=args.experiment_id,
            model=",".join(model_keys),
            profile=args.profile,
            sweep_resolution=args.sweep_resolution,
        )
    pass_index = 0
    command_completed = False
    command_failure_payload: dict[str, str] | None = None

    try:
        for model_key in model_keys:
            if model_key not in MODEL_SPECS:
                raise SystemExit(f"Unsupported model: {model_key}")
            spec = MODEL_SPECS[model_key]
            input_size = int(spec["input_size"])
            crop_size = args.center_crop_size if args.center_crop_size is not None else input_size
            resize_size = args.resize_size if args.resize_size is not None else input_size + 32
            if resize_size < crop_size:
                resize_size = crop_size
            dataset = OpenCVImageNetDataset(
                args.imagenet_val,
                manifest_path=args.imagenet_manifest,
                resize_size=resize_size,
                center_crop_size=crop_size,
                percentage=float(args.percentage),
                seed=int(args.seed),
                enable_mean_std=bool(args.enable_mean_std),
                mean_std_mean=mean_std_mean,
                mean_std_std=mean_std_std,
                input_color_order=args.data_color_order,
                model_color_order=args.input_color_order,
                input_scale=float(args.input_scale),
            )
            sparse_recorder = SparseActivityRecorder(enabled=bool(args.apply_sparse_perturbation))
            bitstream_execution_config = None
            if args.enable_bitstream_pilot:
                raw_target_module_keys = bitstream_runtime_metadata.get("target_module_keys") or []
                target_module_keys = (
                    set(str(key).strip() for key in raw_target_module_keys if str(key).strip())
                    or None
                )
                bitstream_execution_config = BitstreamExecutionConfig(
                    enabled=True,
                    target_module_keys=target_module_keys,
                    encoding_mode=str(args.bitstream_encoding_mode),
                    multiplier_mode=str(args.bitstream_multiplier_mode),
                    accumulator_mode=str(args.bitstream_accumulator_mode),
                    stream_length=int(args.bitstream_stream_length),
                    generator=str(args.bitstream_generator),
                    stream_reuse_policy=str(args.bitstream_stream_reuse_policy),
                    seed=int(args.seed),
                    surface_scope=str(args.bitstream_surface_scope),
                    measurement_truth_class=str(args.bitstream_measurement_truth_class),
                    contract_note=str(args.bitstream_contract_note),
                )
                _apply_quantized_semantic_profile(
                    bitstream_execution_config,
                    _resolve_pass_kind_semantic_profile(
                        accuracy_policy_bundle,
                        pass_kind=PASS_KIND_QUANTIZED_EVAL,
                    ),
                )
            sparse_target_module_keys = None
            if args.sparse_target_module_keys is not None:
                sparse_target_module_keys = set(
                    str(key).strip()
                    for key in parse_csv_tokens(args.sparse_target_module_keys)
                    if str(key).strip()
                )
            elif (
                bool(args.apply_sparse_perturbation)
                and bitstream_execution_config is not None
                and bitstream_execution_config.target_module_keys
            ):
                sparse_target_module_keys = set(bitstream_execution_config.target_module_keys)

            perturb_config = MLXPerturbationConfig(
                bits=int(args.quant_bits),
                gaussian_noise_std=0.0,
                crosstalk_alpha=0.0,
                drift_lsb=float(args.drift_lsb),
                noise_correlation=float(args.noise_correlation),
                burst_error_prob=float(args.burst_error_prob),
                burst_error_scale_lsb=float(args.burst_error_scale_lsb),
                burst_span=max(1, int(args.burst_span)),
                det_enabled=bool(args.apply_det_perturbation),
                det_mode=str(args.det_mode or "reorder"),
                det_bsl_max=args.det_bsl_max,
                det_k_global=args.det_k_global,
                det_k_by_layer=_parse_optional_json_map(args.det_k_by_layer_json),
                det_prefix_error_mean=float(args.det_prefix_error_mean or 0.0),
                det_prefix_error_by_layer=_parse_optional_json_map(args.det_prefix_error_by_layer_json),
                sparse_enabled=bool(args.apply_sparse_perturbation),
                sparse_active_fraction=args.sparse_active_fraction,
                sparse_tau_global=args.sparse_tau_global,
                sparse_target_module_keys=sparse_target_module_keys,
                sparse_activity_recorder=sparse_recorder,
                bitstream_execution_config=bitstream_execution_config,
            )
            model, weights_path = build_mlx_mobilevit(
                model_key,
                perturb_config=perturb_config,
                weights_npz=args.weights_npz,
                weights_dir=args.weights_dir,
                weights_override=args.weights_override,
                mlx_weights_dir=args.mlx_weights_dir,
                force_reexport=args.force_reexport_mlx_weights,
            )
            setattr(model, "model_key", model_key)
            resolved_runtime_metadata = _derive_bitstream_runtime_metadata(
                model_key=model_key,
                base_metadata=bitstream_runtime_metadata,
                perturb_config=perturb_config,
                expected_run_id=args.run_id,
            )
            runtime_metadata_by_model[model_key] = {
                key: value
                for key, value in resolved_runtime_metadata.items()
                if key != "target_module_keys"
            }
            command_runtime_health_gate = {
                "progress_heartbeat_interval_seconds": float(
                    args.progress_heartbeat_interval_seconds
                ),
                "stall_timeout_seconds": float(args.stall_timeout_seconds),
                "pathological_min_samples_per_hour": float(
                    args.pathological_min_samples_per_hour
                ),
                "pathological_max_seconds_per_sample": float(
                    args.pathological_max_seconds_per_sample
                ),
                "pathological_max_eta_current_rate_seconds": float(
                    args.pathological_max_eta_current_rate_seconds
                ),
                "pathological_min_processed_samples": int(
                    args.pathological_min_processed_samples
                ),
                "pathological_min_elapsed_seconds": float(
                    args.pathological_min_elapsed_seconds
                ),
            }
            baseline_pass_tuning = _resolve_pass_runtime_tuning(
                policy_bundle=accuracy_policy_bundle,
                pass_kind=PASS_KIND_BASELINE_EVAL,
                fallback_workers=resolved_worker_count,
                fallback_eval_batch_size=resolved_batch_size,
                fallback_runtime_health_gate=command_runtime_health_gate,
                runtime_metadata=resolved_runtime_metadata,
                model_key=model_key,
                target_module_keys_raw=args.bitstream_target_module_keys,
                max_eval_samples=args.max_eval_samples,
            )
            quantized_pass_tuning = _resolve_pass_runtime_tuning(
                policy_bundle=accuracy_policy_bundle,
                pass_kind=PASS_KIND_QUANTIZED_EVAL,
                fallback_workers=resolved_worker_count,
                fallback_eval_batch_size=resolved_batch_size,
                fallback_runtime_health_gate=command_runtime_health_gate,
                runtime_metadata=resolved_runtime_metadata,
                model_key=model_key,
                target_module_keys_raw=args.bitstream_target_module_keys,
                max_eval_samples=args.max_eval_samples,
            )
            runtime_tuning[model_key] = {
                "device": "mps",
                "workers": resolved_worker_count,
                "prefetch_batches": resolved_prefetch_batches,
                "baseline_eval_batch_size": int(
                    baseline_pass_tuning["eval_batch_size"]
                ),
                "quantized_eval_batch_size": int(
                    quantized_pass_tuning["eval_batch_size"]
                ),
                "pass_policies": {
                    PASS_KIND_BASELINE_EVAL: dict(baseline_pass_tuning),
                    PASS_KIND_QUANTIZED_EVAL: dict(quantized_pass_tuning),
                },
                "bitstream_quantized_batch_policy": dict(
                    quantized_pass_tuning["quantized_batch_policy"]
                ),
            }
            planned_passes_by_model[model_key] = _planned_pass_count(
                model_count=1,
                noise_sigmas=noise_sigmas,
                crosstalk_alphas=crosstalk_alphas,
                pass_mode=str(args.pass_mode),
            )
            model_contexts.append(
                {
                    "model": model_key,
                    "input_size": input_size,
                    "weights_npz": str(weights_path),
                    "weights_npz_source": weights_npz_source,
                }
            )
            common_base_row = {
                "run_id": args.run_id,
                "source_run_id": args.source_run_id or args.run_id,
                "experiment_id": args.experiment_id,
                "device": "mps",
                "accuracy_backend": "mlx",
                "engine": "mlx",
                "workload": args.workload,
                "profile": args.profile,
                "sweep_resolution": args.sweep_resolution,
                "git_hash": git_hash,
                "imagenet_val": args.imagenet_val,
                "imagenet_manifest": args.imagenet_manifest,
                "config_snapshot": args.config_snapshot,
                "det_policy": args.det_policy,
                "det_k_signature": args.det_k_signature,
                "det_k_global": args.det_k_global,
                "det_prefix_error_mean": args.det_prefix_error_mean if args.apply_det_perturbation else None,
                "det_prefix_error_p95": args.det_prefix_error_p95 if args.apply_det_perturbation else None,
                "det_perturbation": bool(args.apply_det_perturbation),
                **_sparse_metadata_fields(args),
                "sparse_gate_mode": None,
                "sparse_measured_activity_fraction": None,
                "sparse_measured_zero_fraction": None,
                "sparse_stats_total_elements": None,
                "sparse_stats_active_elements": None,
                "sparse_stats_call_count": None,
                "sparse_stats_module_count": None,
                "weights_npz": str(weights_path),
                "weights_npz_source": weights_npz_source,
                "calibration_enabled": calibration_enabled,
                "calibration_scale_source": calibration_scale_source,
                "noise_sigmas": _serialize_float_list(noise_sigmas),
                "crosstalk_alphas": _serialize_float_list(crosstalk_alphas),
                "resolved_batch_size": resolved_batch_size,
            }
            baseline_note = f"baseline_mlx weights={weights_path}"
            baseline_row_template = _build_result_row(
                base_row={
                    **common_base_row,
                    **_row_runtime_metadata(
                        resolved_runtime_metadata,
                        enabled=False,
                        pass_policy=baseline_pass_tuning,
                    ),
                },
                baseline=True,
                model_key=model_key,
                input_size=input_size,
                quant_bits=None,
                noise_sigma_lsb=0.0,
                crosstalk_alpha=0.0,
                drift_lsb=float(args.drift_lsb),
                noise_correlation=float(args.noise_correlation),
                burst_error_prob=float(args.burst_error_prob),
                burst_error_scale_lsb=float(args.burst_error_scale_lsb),
                burst_span=max(1, int(args.burst_span)),
                top1=None,
                top5=None,
                top1_delta=None,
                top5_delta=None,
                measured_pass_elapsed_s=None,
                measured_processed_samples=None,
                latency_ms_per_sample=None,
                measurement_window="baseline_eval_pass",
                seed=int(args.seed),
                notes=baseline_note,
            )
            sweep_row_templates = [
                _build_result_row(
                    base_row={
                        **common_base_row,
                        **_row_runtime_metadata(
                            resolved_runtime_metadata,
                            enabled=True,
                            pass_policy=quantized_pass_tuning,
                        ),
                    },
                    baseline=False,
                    model_key=model_key,
                    input_size=input_size,
                    quant_bits=int(args.quant_bits),
                    noise_sigma_lsb=float(sigma_value),
                    crosstalk_alpha=float(alpha_value),
                    drift_lsb=float(args.drift_lsb),
                    noise_correlation=float(args.noise_correlation),
                    burst_error_prob=float(args.burst_error_prob),
                    burst_error_scale_lsb=float(args.burst_error_scale_lsb),
                    burst_span=max(1, int(args.burst_span)),
                    top1=None,
                    top5=None,
                    top1_delta=None,
                    top5_delta=None,
                    measured_pass_elapsed_s=None,
                    measured_processed_samples=None,
                    latency_ms_per_sample=None,
                    measurement_window="quantized_eval_pass",
                    seed=int(args.seed),
                    notes=f"mlx photonic_perturb sigma={float(sigma_value)} alpha={float(alpha_value)}",
                )
                for sigma_value in noise_sigmas
                for alpha_value in crosstalk_alphas
            ]
            expected_row_identities = set()
            baseline_identity_template = result_row_identity(baseline_row_template)
            calibration_scale_cache_path = _calibration_scale_cache_path(
                root=resume_artifact_root,
                baseline_identity=baseline_identity_template,
            )
            if args.pass_mode in {PASS_MODE_PAIRED, PASS_MODE_BASELINE_ONLY}:
                expected_row_identities.add(baseline_identity_template)
            if args.pass_mode in {PASS_MODE_PAIRED, PASS_MODE_QUANTIZED_ONLY}:
                expected_row_identities.update(
                    result_row_identity(row) for row in sweep_row_templates
                )
            if args.resume and all(
                row_identity in existing_row_identities
                for row_identity in expected_row_identities
            ):
                print(
                    f"[resume] skip completed model={model_key} run_id={args.run_id}",
                    flush=True,
                )
                del model
                del perturb_config
                continue

            print(
                "[mlx-acc] "
                f"model={model_key} device=mps "
                f"baseline_workers={int(baseline_pass_tuning['workers'])} "
                f"baseline_prefetch_batches={int(baseline_pass_tuning['prefetch_batches'])} "
                f"baseline_eval_batch_size={int(baseline_pass_tuning['eval_batch_size'])} "
                f"quantized_workers={int(quantized_pass_tuning['workers'])} "
                f"quantized_prefetch_batches={int(quantized_pass_tuning['prefetch_batches'])} "
                f"quantized_eval_batch_size={int(quantized_pass_tuning['eval_batch_size'])}",
                flush=True,
            )

            baseline_top1: float | None = None
            baseline_top5: float | None = None
            if args.pass_mode == PASS_MODE_QUANTIZED_ONLY:
                baseline_reference_row = _load_baseline_reference_row(
                    baseline_reference_csv=str(args.baseline_reference_csv),
                    baseline_reference_run_id=str(args.baseline_reference_run_id),
                    model_key=model_key,
                )
                baseline_top1, baseline_top5 = _baseline_metrics_from_reference(
                    baseline_reference_row
                )
            else:
                baseline_existing_row = existing_result_rows_by_identity.get(
                    baseline_identity_template
                )
                baseline_cache_loaded = False
                if calibration_enabled and baseline_existing_row is not None:
                    baseline_cache_loaded = _load_calibration_scale_cache(
                        calibration_scale_cache_path,
                        perturb_config,
                    )
                can_skip_baseline = baseline_existing_row is not None and (
                    not calibration_enabled or baseline_cache_loaded
                )
                if can_skip_baseline:
                    pass_index += 1
                    baseline_top1, baseline_top5 = _baseline_metrics_from_reference(
                        baseline_existing_row
                    )
                    if calibration_enabled:
                        perturb_config.calibration_mode = False
                        perturb_config.use_calibrated_scale = True
                    print(
                        "[resume] skip completed baseline "
                        f"model={model_key} run_id={args.run_id} "
                        f"calibration_cache={calibration_scale_cache_path if calibration_enabled else 'not_required'}",
                        flush=True,
                    )
                else:
                    if baseline_existing_row is not None and calibration_enabled:
                        print(
                            "[resume] completed baseline row found, but calibration scale "
                            f"cache is missing; rebuilding cache at {calibration_scale_cache_path}",
                            flush=True,
                        )
                    if calibration_enabled:
                        perturb_config.calibration_mode = True
                    pass_index += 1
                    _raise_if_stop_requested("before_baseline_eval")
                    set_seeds(args.seed)
                    baseline_checkpoint_path = _pass_resume_checkpoint_path(
                        root=resume_artifact_root,
                        row_identity=baseline_identity_template,
                        pass_kind="baseline_eval_pass",
                    )
                    baseline_top1, baseline_top5, baseline_stats = evaluate_model(
                        model,
                        dataset,
                        batch_size=int(baseline_pass_tuning["eval_batch_size"]),
                        max_samples=args.max_eval_samples,
                        workers=int(baseline_pass_tuning["workers"]),
                        prefetch_batches=int(baseline_pass_tuning["prefetch_batches"]),
                        progress_prefix=f"{args.progress_label or model_key}:baseline",
                        progress_recorder=progress_recorder,
                        total_passes=total_planned_passes,
                        pass_index=pass_index,
                        pass_kind="baseline_eval_pass",
                        profile=args.profile,
                        sweep_resolution=args.sweep_resolution,
                        gaussian_noise_std=0.0,
                        crosstalk_alpha=0.0,
                        progress_heartbeat_interval_seconds=float(
                            baseline_pass_tuning["runtime_health_gate"][
                                "progress_heartbeat_interval_seconds"
                            ]
                        ),
                        stall_timeout_seconds=float(
                            baseline_pass_tuning["runtime_health_gate"][
                                "stall_timeout_seconds"
                            ]
                        ),
                        pathological_min_samples_per_hour=float(
                            baseline_pass_tuning["runtime_health_gate"][
                                "pathological_min_samples_per_hour"
                            ]
                        ),
                        pathological_max_seconds_per_sample=float(
                            baseline_pass_tuning["runtime_health_gate"][
                                "pathological_max_seconds_per_sample"
                            ]
                        ),
                        pathological_max_eta_current_rate_seconds=float(
                            baseline_pass_tuning["runtime_health_gate"][
                                "pathological_max_eta_current_rate_seconds"
                            ]
                        ),
                        pathological_min_processed_samples=int(
                            baseline_pass_tuning["runtime_health_gate"][
                                "pathological_min_processed_samples"
                            ]
                        ),
                        pathological_min_elapsed_seconds=float(
                            baseline_pass_tuning["runtime_health_gate"][
                                "pathological_min_elapsed_seconds"
                            ]
                        ),
                        seed=int(args.seed),
                        resume_checkpoint_path=baseline_checkpoint_path,
                        resume_row_identity=baseline_identity_template,
                        resume_enabled=bool(args.resume),
                        runtime_policy_fingerprint=str(
                            baseline_pass_tuning["runtime_policy_fingerprint"]
                        ),
                        semantic_fingerprint=str(
                            baseline_pass_tuning["semantic_fingerprint"]
                        ),
                    )
                    if calibration_enabled:
                        _save_calibration_scale_cache(
                            calibration_scale_cache_path,
                            perturb_config,
                        )
                        perturb_config.calibration_mode = False
                        perturb_config.use_calibrated_scale = True

                    baseline_latency_ms = (
                        (float(baseline_stats["elapsed_s"]) / float(baseline_stats["processed_samples"])) * 1000.0
                    )
                    baseline_row = _build_result_row(
                        base_row={
                            **common_base_row,
                            **_row_runtime_metadata(
                                resolved_runtime_metadata,
                                enabled=False,
                                pass_policy=baseline_pass_tuning,
                            ),
                        },
                        baseline=True,
                        model_key=model_key,
                        input_size=input_size,
                        quant_bits=None,
                        noise_sigma_lsb=0.0,
                        crosstalk_alpha=0.0,
                        drift_lsb=float(args.drift_lsb),
                        noise_correlation=float(args.noise_correlation),
                        burst_error_prob=float(args.burst_error_prob),
                        burst_error_scale_lsb=float(args.burst_error_scale_lsb),
                        burst_span=max(1, int(args.burst_span)),
                        top1=baseline_top1,
                        top5=baseline_top5,
                        top1_delta=None,
                        top5_delta=None,
                        measured_pass_elapsed_s=float(baseline_stats["elapsed_s"]),
                        measured_processed_samples=int(baseline_stats["processed_samples"]),
                        latency_ms_per_sample=baseline_latency_ms,
                        measurement_window="baseline_eval_pass",
                        seed=int(args.seed),
                        notes=baseline_note,
                    )
                    baseline_identity = result_row_identity(baseline_row)
                    if baseline_identity not in existing_row_identities:
                        write_results(results_path, [baseline_row], append=append_results)
                        append_results = True
                        existing_row_identities.add(baseline_identity)
                        existing_result_rows_by_identity[baseline_identity] = {
                            field: str(baseline_row.get(field) or "")
                            for field in RESULT_FIELDNAMES
                        }

            if args.pass_mode == PASS_MODE_BASELINE_ONLY:
                perturb_config.enabled = False
                del model
                del perturb_config
                _release_mlx_memory()
                continue

            perturb_config.enabled = True
            perturb_config.bits = int(args.quant_bits)
            for sigma_value in noise_sigmas:
                for alpha_value in crosstalk_alphas:
                    row_note = (
                        f"mlx photonic_perturb sigma={float(sigma_value)} alpha={float(alpha_value)}"
                    )
                    pending_row = _build_result_row(
                        base_row={
                            **common_base_row,
                            **_row_runtime_metadata(resolved_runtime_metadata, enabled=True),
                        },
                        baseline=False,
                        model_key=model_key,
                        input_size=input_size,
                        quant_bits=int(args.quant_bits),
                        noise_sigma_lsb=float(sigma_value),
                        crosstalk_alpha=float(alpha_value),
                        drift_lsb=float(args.drift_lsb),
                        noise_correlation=float(args.noise_correlation),
                        burst_error_prob=float(args.burst_error_prob),
                        burst_error_scale_lsb=float(args.burst_error_scale_lsb),
                        burst_span=max(1, int(args.burst_span)),
                        top1=None,
                        top5=None,
                        top1_delta=None,
                        top5_delta=None,
                        measured_pass_elapsed_s=None,
                        measured_processed_samples=None,
                        latency_ms_per_sample=None,
                        measurement_window="quantized_eval_pass",
                        seed=int(args.seed),
                        notes=row_note,
                    )
                    pending_identity = result_row_identity(pending_row)
                    if args.resume and pending_identity in existing_row_identities:
                        print(
                            f"[resume] skip completed row model={model_key} sigma={float(sigma_value):.12g} alpha={float(alpha_value):.12g} run_id={args.run_id}",
                            flush=True,
                        )
                        continue

                    pass_index += 1
                    perturb_config.gaussian_noise_std = float(sigma_value)
                    perturb_config.crosstalk_alpha = float(alpha_value)
                    if sparse_recorder.enabled:
                        sparse_recorder.reset()
                    _raise_if_stop_requested("before_sweep_eval")
                    set_seeds(args.seed)
                    quantized_checkpoint_path = _pass_resume_checkpoint_path(
                        root=resume_artifact_root,
                        row_identity=pending_identity,
                        pass_kind="quantized_eval_pass",
                    )
                    top1, top5, stats = evaluate_model(
                        model,
                        dataset,
                        batch_size=int(quantized_pass_tuning["eval_batch_size"]),
                        max_samples=args.max_eval_samples,
                        workers=int(quantized_pass_tuning["workers"]),
                        prefetch_batches=int(quantized_pass_tuning["prefetch_batches"]),
                        progress_prefix=(
                            f"{args.progress_label or model_key}:sigma={float(sigma_value):.4g}:alpha={float(alpha_value):.4g}"
                        ),
                        progress_recorder=progress_recorder,
                        total_passes=total_planned_passes,
                        pass_index=pass_index,
                        pass_kind="quantized_eval_pass",
                        profile=args.profile,
                        sweep_resolution=args.sweep_resolution,
                        gaussian_noise_std=float(sigma_value),
                        crosstalk_alpha=float(alpha_value),
                        progress_heartbeat_interval_seconds=float(
                            quantized_pass_tuning["runtime_health_gate"][
                                "progress_heartbeat_interval_seconds"
                            ]
                        ),
                        stall_timeout_seconds=float(
                            quantized_pass_tuning["runtime_health_gate"][
                                "stall_timeout_seconds"
                            ]
                        ),
                        pathological_min_samples_per_hour=float(
                            quantized_pass_tuning["runtime_health_gate"][
                                "pathological_min_samples_per_hour"
                            ]
                        ),
                        pathological_max_seconds_per_sample=float(
                            quantized_pass_tuning["runtime_health_gate"][
                                "pathological_max_seconds_per_sample"
                            ]
                        ),
                        pathological_max_eta_current_rate_seconds=float(
                            quantized_pass_tuning["runtime_health_gate"][
                                "pathological_max_eta_current_rate_seconds"
                            ]
                        ),
                        pathological_min_processed_samples=int(
                            quantized_pass_tuning["runtime_health_gate"][
                                "pathological_min_processed_samples"
                            ]
                        ),
                        pathological_min_elapsed_seconds=float(
                            quantized_pass_tuning["runtime_health_gate"][
                                "pathological_min_elapsed_seconds"
                            ]
                        ),
                        seed=int(args.seed),
                        resume_checkpoint_path=quantized_checkpoint_path,
                        resume_row_identity=pending_identity,
                        resume_enabled=bool(args.resume),
                        runtime_policy_fingerprint=str(
                            quantized_pass_tuning["runtime_policy_fingerprint"]
                        ),
                        semantic_fingerprint=str(
                            quantized_pass_tuning["semantic_fingerprint"]
                        ),
                    )
                    perturb_summary = sparse_recorder.summary() if sparse_recorder.enabled else {}
                    latency_ms = (float(stats["elapsed_s"]) / float(stats["processed_samples"])) * 1000.0
                    row = _build_result_row(
                        base_row={
                            **common_base_row,
                            **_row_runtime_metadata(
                                resolved_runtime_metadata,
                                enabled=True,
                                pass_policy=quantized_pass_tuning,
                            ),
                            **perturb_summary,
                        },
                        baseline=False,
                        model_key=model_key,
                        input_size=input_size,
                        quant_bits=int(args.quant_bits),
                        noise_sigma_lsb=float(sigma_value),
                        crosstalk_alpha=float(alpha_value),
                        drift_lsb=float(args.drift_lsb),
                        noise_correlation=float(args.noise_correlation),
                        burst_error_prob=float(args.burst_error_prob),
                        burst_error_scale_lsb=float(args.burst_error_scale_lsb),
                        burst_span=max(1, int(args.burst_span)),
                        top1=top1,
                        top5=top5,
                        top1_delta=top1 - baseline_top1,
                        top5_delta=top5 - baseline_top5,
                        measured_pass_elapsed_s=float(stats["elapsed_s"]),
                        measured_processed_samples=int(stats["processed_samples"]),
                        latency_ms_per_sample=latency_ms,
                        measurement_window="quantized_eval_pass",
                        seed=int(args.seed),
                        notes=row_note,
                    )
                    write_results(results_path, [row], append=append_results)
                    append_results = True
                    existing_row_identities.add(pending_identity)
                    existing_result_rows_by_identity[pending_identity] = {
                        field: str(row.get(field) or "")
                        for field in RESULT_FIELDNAMES
                    }
                    if sparse_layers_path is not None and sparse_recorder.enabled:
                        write_sparse_layer_rows(
                            sparse_layers_path,
                            sparse_recorder.layer_rows(
                                row_context={
                                    "run_id": args.run_id,
                                    "experiment_id": args.experiment_id,
                                    "baseline": False,
                                    "model": model_key,
                                    "seed": int(args.seed),
                                    "device": "mps",
                                    "gaussian_noise_std": float(sigma_value),
                                    "crosstalk_alpha": float(alpha_value),
                                }
                            ),
                            append=True,
                        )
            perturb_config.enabled = False
            del model
            del perturb_config
            _release_mlx_memory()
        command_completed = True
    except EvaluationInterrupted as exc:
        command_failure_payload = {
            "failure_kind": "interrupted",
            "failure_message": str(exc),
        }
        print(
            f"[resume] graceful stop recorded at {exc}; rerun the same command with --resume to continue from the existing CSV.",
            flush=True,
        )
        return 130
    except EvaluationStalled as exc:
        command_failure_payload = {
            "failure_kind": "stall_timeout",
            "failure_message": str(exc),
        }
        print(
            f"[stall] {exc}. Progress JSONL was updated with a terminal failure event.",
            flush=True,
        )
        return 124
    except EvaluationPathologicallySlow as exc:
        command_failure_payload = {
            "failure_kind": "pathological_slow",
            "failure_message": str(exc),
        }
        print(
            f"[health] {exc}. Progress JSONL was updated with a terminal failure event.",
            flush=True,
        )
        return 125
    except Exception as exc:
        command_failure_payload = {
            "failure_kind": type(exc).__name__,
            "failure_message": str(exc),
        }
        raise
    finally:
        if progress_recorder is not None and command_completed:
            progress_recorder.emit_command_complete(
                total_passes=total_planned_passes,
                run_id=args.run_id,
                experiment_id=args.experiment_id,
                model=",".join(model_keys),
                profile=args.profile,
                sweep_resolution=args.sweep_resolution,
            )
        elif progress_recorder is not None and command_failure_payload is not None:
            progress_recorder.emit_command_failed(
                total_passes=total_planned_passes,
                run_id=args.run_id,
                experiment_id=args.experiment_id,
                model=",".join(model_keys),
                profile=args.profile,
                sweep_resolution=args.sweep_resolution,
                failure_kind=command_failure_payload["failure_kind"],
                failure_message=command_failure_payload["failure_message"],
            )

    if args.metadata_json:
        metadata = _build_metadata_payload(
            run_id=args.run_id,
            model_keys=model_keys,
            results_path=results_path,
            noise_sigmas=noise_sigmas,
            crosstalk_alphas=crosstalk_alphas,
            resolved_batch_size=resolved_batch_size,
            calibration_enabled=calibration_enabled,
            calibration_scale_source=calibration_scale_source,
            model_contexts=model_contexts,
            runtime_metadata={
                key: value
                for key, value in bitstream_runtime_metadata.items()
                if key != "target_module_keys"
            },
        )
        metadata["runtime_tuning"] = runtime_tuning
        metadata["bitstream_runtime_by_model"] = runtime_metadata_by_model
        metadata["planned_passes_by_model"] = planned_passes_by_model
        metadata["progress_jsonl"] = args.progress_jsonl
        metadata["resume_artifact_root"] = str(resume_artifact_root)
        metadata["pass_mode"] = args.pass_mode
        metadata["baseline_reference_csv"] = args.baseline_reference_csv
        metadata["baseline_reference_run_id"] = args.baseline_reference_run_id
        Path(args.metadata_json).write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    print(f"Wrote results to {results_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
