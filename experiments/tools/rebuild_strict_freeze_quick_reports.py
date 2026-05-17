from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "experiments" / "results" / "runs"
REPORT_DATA = ROOT / "experiments" / "results" / "report_data"
PHASE1_RUNNER = ROOT / "experiments" / "tools" / "phase1_runner.py"
OVERLAY_ROOT = ROOT / "experiments" / "results" / "dev" / "headline_runs_overlay_20260328"
DEFAULT_RUN_TAG = "20260328_mps_full_eval_freeze"
DEFAULT_CANDIDATE_TAG = "20260328_mps_full_eval"
DEFAULT_QUICK_DIR = ROOT / "experiments" / "results" / "quick_reports" / DEFAULT_RUN_TAG
DEFAULT_CANDIDATE_DIR = DEFAULT_QUICK_DIR / "candidate_data"
CPU_DEVICE_METRICS: Path | None = REPORT_DATA / "fuller_cpu_device_metrics_20260319_fullerexp_v1.csv"
GPU_DEVICE_METRICS = REPORT_DATA / "fuller_gpu_device_metrics_20260319_fullerexp_v1.csv"
FULLER_MODEL_SUMMARY = REPORT_DATA / "fuller_slice_model_summary_20260319_fullerexp_v1.csv"
FULLER_ABLATION_SUMMARY = REPORT_DATA / "fuller_ablation_summary_20260319_fullerexp_v1.csv"
CANDIDATE_TAG = DEFAULT_CANDIDATE_TAG
DEFAULT_SPARSE_BACKUP = (
    ROOT
    / "experiments"
    / "results"
    / "accuracy"
    / "accuracy_config_conditioned_cuda_sparse_tau_sweep4096_seeds12_20260307.csv.20260307_233604.bak"
)
ACCELERATOR_DEVICE_NAME = "MTL-FULLER accelerator"
ACCELERATOR_SHORT_NAME = "MTL-FULLER"

CORE_RUNS = {
    "E0": "20260228_opt_sync_core_e0",
    "E1": "20260228_opt_sync_core_e1",
    "E2": "20260228_opt_sync_core_e2",
    "E3": "20260228_opt_sync_core_e3",
    "E4": "20260228_opt_sync_core_e4",
    "E5": "20260228_opt_sync_core_e5",
    "E6": "20260228_opt_sync_core_e6",
}

SCALING_RUNS = [
    "20260228_opt_sync_scan_e0_batch2",
    "20260228_opt_sync_scan_e0_batch4",
    "20260228_opt_sync_scan_e0_seq128",
    "20260228_opt_sync_scan_e0_seq256",
]

SPARSE_TAU_RUNS = [
    "20260228_opt_sync_scan_e4_t00",
    "20260228_opt_sync_scan_e4_t05",
    "20260228_opt_sync_scan_e4_t10",
    "20260228_opt_sync_scan_e4_t15",
    "20260228_opt_sync_scan_e4_t20",
    "20260228_opt_sync_scan_e4_t25",
    "20260228_opt_sync_scan_e4_t30",
    "20260228_opt_sync_scan_e4_t40",
    "20260228_opt_sync_scan_e4_t50",
]

RERUN_TARGETS = [
    CORE_RUNS["E4"],
    CORE_RUNS["E6"],
    *SCALING_RUNS,
    *SPARSE_TAU_RUNS,
]

CORE_SOURCE_DIRS = {
    "E0": OVERLAY_ROOT / CORE_RUNS["E0"],
    "E1": RUNS / CORE_RUNS["E1"],
    "E2": OVERLAY_ROOT / CORE_RUNS["E2"],
    "E3": OVERLAY_ROOT / CORE_RUNS["E3"],
    "E4": OVERLAY_ROOT / CORE_RUNS["E4"],
    "E5": RUNS / CORE_RUNS["E5"],
    "E6": OVERLAY_ROOT / CORE_RUNS["E6"],
}


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _to_num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _write_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[strict-freeze-rebuild] wrote {path}")


def _load_device_metrics_row(path: Path) -> dict[str, Any]:
    df = pd.read_csv(path)
    if df.empty:
        raise SystemExit(f"Empty device metrics csv: {path}")
    return df.iloc[0].to_dict()


def _candidate_file(candidate_dir: Path, stem: str, suffix: str = "csv") -> Path:
    tagged = candidate_dir / f"{stem}_{CANDIDATE_TAG}.{suffix}"
    if tagged.exists():
        return tagged
    matches = sorted(candidate_dir.glob(f"{stem}_*.{suffix}"))
    if len(matches) == 1:
        return matches[0]
    raise SystemExit(
        f"Unable to resolve candidate file for stem={stem!r} tag={CANDIDATE_TAG!r} under {candidate_dir}"
    )


def _load_single_report_row(path: Path) -> dict[str, Any]:
    df = pd.read_csv(path)
    if df.empty:
        raise SystemExit(f"Empty report-data csv: {path}")
    return df.iloc[0].to_dict()


def _num(value: Any, default: float = math.nan) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _gpu_backend_label(row: dict[str, Any]) -> str:
    accuracy_backend = str(row.get("accuracy_backend") or "").strip().lower()
    framework = str(row.get("framework") or "").strip().lower()
    if accuracy_backend == "mlx" or framework == "mlx":
        return "mlx"
    return "torch"


def _display_device_name(
    platform_class: str,
    device_model: str,
    *,
    gpu_backend: str = "torch",
) -> str:
    text = str(device_model).strip()
    if platform_class == "CPU":
        return f"{text} CPU" if text else "CPU"
    if platform_class == "GPU":
        pretty = text.split(":", 1)[1] if text.startswith("Apple-MPS:") else text
        if gpu_backend == "mlx":
            return f"{pretty} GPU (MLX MPS)" if pretty else "GPU (MLX MPS)"
        return f"{pretty} GPU (PyTorch MPS)" if pretty else "GPU (PyTorch MPS)"
    return text or ACCELERATOR_DEVICE_NAME


def _platform_device_metadata() -> dict[str, dict[str, Any]]:
    gpu = _load_device_metrics_row(GPU_DEVICE_METRICS)
    gpu_host = str(gpu.get("host_name") or "").strip()
    gpu_model = str(gpu.get("device_model") or "").strip()
    gpu_backend = _gpu_backend_label(gpu)
    meta = {
        "GPU": {
            "host_name": gpu_host,
            "device_model": gpu_model,
            "device_display_name": _display_device_name("GPU", gpu_model, gpu_backend=gpu_backend),
            "measurement_surface": (
                "local_gpu_measured_mlx_mps"
                if gpu_backend == "mlx"
                else "local_gpu_measured_mps"
            ),
            "platform_display_name": (
                "GPU measured (MLX MPS)"
                if gpu_backend == "mlx"
                else "GPU measured (PyTorch MPS)"
            ),
            "device_metadata_source": str(GPU_DEVICE_METRICS),
            "evidence_tier": "measured",
            "device_name": _display_device_name("GPU", gpu_model, gpu_backend=gpu_backend),
        },
        "HPAT": {
            "host_name": "",
            "device_model": ACCELERATOR_DEVICE_NAME,
            "device_display_name": ACCELERATOR_DEVICE_NAME,
            "measurement_surface": "accelerator_model",
            "platform_display_name": ACCELERATOR_SHORT_NAME,
            "device_metadata_source": "modeled_from_freeze_quick_reports",
            "evidence_tier": "modeled",
            "device_name": ACCELERATOR_DEVICE_NAME,
        },
    }
    if CPU_DEVICE_METRICS is not None and CPU_DEVICE_METRICS.exists():
        cpu = _load_device_metrics_row(CPU_DEVICE_METRICS)
        cpu_host = str(cpu.get("host_name") or "").strip()
        cpu_model = str(cpu.get("device_model") or "").strip()
        meta["CPU"] = {
            "host_name": cpu_host,
            "device_model": cpu_model,
            "device_display_name": _display_device_name("CPU", cpu_model),
            "measurement_surface": "local_cpu_measured",
            "platform_display_name": "CPU measured",
            "device_metadata_source": str(CPU_DEVICE_METRICS),
            "evidence_tier": "measured",
            "device_name": _display_device_name("CPU", cpu_model),
        }
    return meta


def build_contextual_platform_compare(run_tag: str) -> pd.DataFrame:
    cpu = _load_device_metrics_row(CPU_DEVICE_METRICS) if CPU_DEVICE_METRICS is not None and CPU_DEVICE_METRICS.exists() else None
    gpu = _load_device_metrics_row(GPU_DEVICE_METRICS)
    fuller = _load_single_report_row(FULLER_MODEL_SUMMARY)
    meta = _platform_device_metadata()

    workload_id = str(fuller.get("workload_id") or (cpu or {}).get("workload_id") or gpu.get("workload_id") or "").strip()
    model = str(fuller.get("model") or (cpu or {}).get("model") or gpu.get("model") or "").strip()
    input_size = int(_num(fuller.get("input_size"), 0.0))
    batch_size = int(_num(fuller.get("batch_size"), _num((cpu or {}).get("batch_size"), _num(gpu.get("batch_size"), 1.0))))
    sequence_length = int(_num((cpu or {}).get("sequence_length"), _num(gpu.get("sequence_length"), 0.0)))
    if sequence_length <= 0:
        sequence_length = int(_num(fuller.get("sequence_length"), 0.0))
    acc_top1 = _num(fuller.get("acc_top1"))
    acc_drop_pp = _num(fuller.get("acc_drop_pp"))
    fuller_latency_ms = _num(fuller.get("latency_ms"), 0.0)
    fuller_energy_j = _num(fuller.get("energy_j"), 0.0)
    fuller_power_w = _num(fuller.get("avg_power_w"), 0.0)
    fuller_tops_w = _num(fuller.get("tops_w"), 0.0)
    fuller_peak_tops = fuller_tops_w * fuller_power_w if fuller_tops_w > 0.0 and fuller_power_w > 0.0 else math.nan

    def _throughput(latency_ms: float, *, tokens: bool = False) -> float:
        if not math.isfinite(latency_ms) or latency_ms <= 0.0:
            return math.nan
        count = batch_size * sequence_length if tokens else batch_size
        return count / (latency_ms / 1e3)

    rows: list[dict[str, Any]] = []
    if cpu is not None:
        rows.append(
            {
                "run_id": str(cpu.get("host_name") or "cpu_contextual_device"),
                "run_tag": run_tag,
                "experiment_id": "CPU_REAL_DEVICE_LOCAL",
                "model": model,
                "model_family": "mobilevit",
                "task_id": "imagenet_cls",
                "workload_id": workload_id,
                "latency_ms": _num(cpu.get("latency_ms")),
                "energy_j": _num(cpu.get("energy_j")),
                "avg_power_w": _num(cpu.get("avg_power_w")),
                "tops_w": 0.0,
                "throughput_images_s": _throughput(_num(cpu.get("latency_ms"))),
                "throughput_tokens_s": _throughput(_num(cpu.get("latency_ms")), tokens=True),
                "primary_metric_name": "Top1",
                "primary_metric_value": math.nan,
                "primary_metric_drop": math.nan,
                "acc_top1": math.nan,
                "acc_drop_pp": math.nan,
                "area_mm2": 0.0,
                "peak_tops": 0.0,
                "platform_class": "CPU",
                "process_node_nm": "unknown",
                "source_type": "measured",
                "batch_size": batch_size,
                "sequence_length": sequence_length,
                "input_size": input_size,
                "N_wdm": 0.0,
                "mrr_tile_k": 0.0,
                "quant_bits": 0.0,
                "fanout": 0.0,
                "broadcast_mode": "one_to_one",
                **meta["CPU"],
                "device_metadata_source": str(CPU_DEVICE_METRICS),
            }
        )
    rows.extend([
        {
            "run_id": str(gpu.get("host_name") or "gpu_contextual_device"),
            "run_tag": run_tag,
            "experiment_id": "GPU_REAL_DEVICE_LOCAL_MPS",
            "model": model,
            "model_family": "mobilevit",
            "task_id": "imagenet_cls",
            "workload_id": workload_id,
            "latency_ms": _num(gpu.get("latency_ms")),
            "energy_j": _num(gpu.get("energy_j")),
            "avg_power_w": _num(gpu.get("avg_power_w")),
            "tops_w": 0.0,
            "throughput_images_s": _throughput(_num(gpu.get("latency_ms"))),
            "throughput_tokens_s": _throughput(_num(gpu.get("latency_ms")), tokens=True),
            "primary_metric_name": "Top1",
            "primary_metric_value": math.nan,
            "primary_metric_drop": math.nan,
            "acc_top1": math.nan,
            "acc_drop_pp": math.nan,
            "area_mm2": 0.0,
            "peak_tops": 0.0,
            "platform_class": "GPU",
            "process_node_nm": "unknown",
            "source_type": "measured",
            "batch_size": batch_size,
            "sequence_length": sequence_length,
            "input_size": input_size,
            "N_wdm": 0.0,
            "mrr_tile_k": 0.0,
            "quant_bits": 0.0,
            "fanout": 0.0,
            "broadcast_mode": "one_to_one",
            **meta["GPU"],
            "device_metadata_source": str(GPU_DEVICE_METRICS),
        },
        {
            "run_id": str(fuller.get("run_id") or ""),
            "run_tag": run_tag,
            "experiment_id": str(fuller.get("experiment_id") or "FULLER_SLICE_V1"),
            "model": model,
            "model_family": "mobilevit",
            "task_id": "imagenet_cls",
            "workload_id": workload_id,
            "latency_ms": fuller_latency_ms,
            "energy_j": fuller_energy_j,
            "avg_power_w": fuller_power_w,
            "tops_w": fuller_tops_w,
            "throughput_images_s": _throughput(fuller_latency_ms),
            "throughput_tokens_s": _throughput(fuller_latency_ms, tokens=True),
            "primary_metric_name": "Top1",
            "primary_metric_value": acc_top1,
            "primary_metric_drop": acc_drop_pp,
            "acc_top1": acc_top1,
            "acc_drop_pp": acc_drop_pp,
            "area_mm2": 0.0,
            "peak_tops": fuller_peak_tops,
            "platform_class": "HPAT",
            "process_node_nm": "unknown",
            "source_type": "simulated_hpat",
            "batch_size": batch_size,
            "sequence_length": sequence_length,
            "input_size": input_size,
            "N_wdm": _num(fuller.get("N_wdm"), 0.0),
            "mrr_tile_k": 16.0,
            "quant_bits": 8.0,
            "fanout": _num(fuller.get("fanout"), 0.0),
            "broadcast_mode": "broadcast",
            "device_name": ACCELERATOR_DEVICE_NAME,
            "host_name": "",
            "device_model": ACCELERATOR_DEVICE_NAME,
            "device_display_name": ACCELERATOR_DEVICE_NAME,
            "measurement_surface": "accelerator_model",
            "platform_display_name": ACCELERATOR_SHORT_NAME,
            "device_metadata_source": str(FULLER_MODEL_SUMMARY),
            "evidence_tier": "modeled",
        },
    ])
    return pd.DataFrame(rows)


def _clean_platform_compare_rows(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    if df.empty:
        return df.copy(), []

    notes: list[str] = []
    work = df.drop_duplicates().copy()
    removed_exact = len(df) - len(work)
    if removed_exact:
        notes.append(f"removed_exact_duplicates={removed_exact}")

    src_series = work.get("source_type", pd.Series("", index=work.index)).astype(str).str.lower()
    latency_series = _to_num(work.get("latency_ms", pd.Series(math.nan, index=work.index)))
    energy_series = _to_num(work.get("energy_j", pd.Series(math.nan, index=work.index)))
    power_series = _to_num(work.get("avg_power_w", pd.Series(math.nan, index=work.index)))

    incomplete_mask = (
        src_series.eq("measured")
        & latency_series.notna()
        & energy_series.notna()
        & power_series.notna()
        & (energy_series <= 0.0)
        & (power_series <= 0.0)
    )
    removable_idx: list[int] = []
    for idx, row in work[incomplete_mask].iterrows():
        model = str(row.get("model", ""))
        platform = str(row.get("platform_class", "")).upper()
        src = str(row.get("source_type", "")).lower()
        latency = float(row.get("latency_ms", math.nan))
        if not math.isfinite(latency):
            continue
        lat_close = pd.Series(
            np.isclose(latency_series.to_numpy(dtype=float), latency, rtol=1e-9, atol=1e-9),
            index=work.index,
        )
        candidate_mask = (
            (work.index != idx)
            & work.get("model", pd.Series("", index=work.index)).astype(str).eq(model)
            & work.get("platform_class", pd.Series("", index=work.index)).astype(str).str.upper().eq(platform)
            & src_series.eq(src)
            & latency_series.notna()
            & lat_close
            & (energy_series > 0.0)
            & (power_series > 0.0)
        )
        if candidate_mask.any():
            removable_idx.append(idx)

    if removable_idx:
        work = work.drop(index=removable_idx).copy()
        notes.append(f"removed_incomplete_measured_rows={len(removable_idx)}")

    return work.reset_index(drop=True), notes


def enrich_platform_compare_table(path: Path) -> None:
    if not path.exists():
        return
    df = pd.read_csv(path)
    if df.empty or "platform_class" not in df.columns:
        return
    meta = _platform_device_metadata()
    work, clean_notes = _clean_platform_compare_rows(df)
    for col in [
        "host_name",
        "device_model",
        "device_display_name",
        "measurement_surface",
        "platform_display_name",
        "device_metadata_source",
        "evidence_tier",
    ]:
        if col not in work.columns:
            work[col] = ""
    platform_series = work["platform_class"].astype(str).str.upper()
    for platform, payload in meta.items():
        mask = platform_series.eq(platform)
        if not mask.any():
            continue
        for col, value in payload.items():
            work.loc[mask, col] = value
    if clean_notes:
        print(f"[strict-freeze-rebuild] cleaned {path.name}: {'; '.join(clean_notes)}")
    _write_csv(path, work)


def _group_master_by_model_from_dir(run_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(run_dir / "master_metrics.csv")
    if df.empty:
        raise SystemExit(f"Empty master_metrics.csv for {run_dir}")
    work = df.copy()
    for col in work.columns:
        if col != "model":
            work[col] = _to_num(work[col])
    numeric_cols = [col for col in work.columns if col != "model" and work[col].notna().any()]
    grouped = work.groupby("model", as_index=False)[numeric_cols].mean()
    grouped["model"] = grouped["model"].astype(str)
    return grouped


def _group_master_by_model(run_id: str) -> pd.DataFrame:
    return _group_master_by_model_from_dir(RUNS / run_id)


def _phase1_summary(run_id: str) -> pd.DataFrame:
    return pd.read_csv(RUNS / run_id / "phase1_summary.csv")


def _config_snapshot(run_id: str) -> dict[str, Any]:
    return _read_yaml(RUNS / run_id / "config_snapshot.yaml")


def _mean_of_models(grouped: pd.DataFrame, col: str) -> float | None:
    if col not in grouped.columns:
        return None
    vals = _to_num(grouped[col]).dropna()
    if vals.empty:
        return None
    return float(vals.mean())


def _aggregate_core_row(
    exp: str,
    run_id: str,
    source_dir: Path,
    baseline_latency_by_model: dict[str, float],
) -> dict[str, Any]:
    grouped = _group_master_by_model_from_dir(source_dir)
    raw = pd.read_csv(source_dir / "master_metrics.csv")
    experiment_id = str(raw["experiment_id"].dropna().astype(str).iloc[0])
    row: dict[str, Any] = {
        "experiment_id": experiment_id,
        "run_id": run_id,
        "n_models": int(grouped["model"].nunique()),
    }
    copy_cols = [
        "latency_ms",
        "energy_j",
        "avg_power_w",
        "tops_w",
        "throughput_images_s",
        "throughput_tokens_s",
        "avg_effective_bsl",
        "bubble_cycles",
        "utilization_avg",
        "fanout",
        "serializers_saved",
        "broadcast_driver_energy_j",
        "net_energy_gain_j",
        "det_net_gain_j",
        "N_wdm",
        "PP_crosstalk_db",
        "P_laser_dbm",
        "P_laser_mw",
        "duty_cycle_avg",
    ]
    for col in copy_cols:
        row[col] = _mean_of_models(grouped, col)
    speedups: list[float] = []
    for _, item in grouped.iterrows():
        model = str(item["model"])
        base = baseline_latency_by_model.get(model)
        lat = float(item["latency_ms"]) if pd.notna(item["latency_ms"]) else math.nan
        if base and math.isfinite(lat) and lat > 0:
            speedups.append(base / lat)
    row["speedup_vs_E0"] = float(np.mean(speedups)) if speedups else (1.0 if experiment_id == "E0" else math.nan)
    if "pass_det_net_gain" in raw.columns:
        row["pass_det_net_gain_true"] = int(_to_num(raw["pass_det_net_gain"]).fillna(0).sum())
    else:
        row["pass_det_net_gain_true"] = 0
    return row


def _load_accuracy_headline(candidate_dir: Path) -> pd.DataFrame:
    return pd.read_csv(_candidate_file(candidate_dir, "headline_statistics_summary"))


def _load_accuracy_per_model(candidate_dir: Path) -> pd.DataFrame:
    return pd.read_csv(_candidate_file(candidate_dir, "config_conditioned_accuracy_per_model"))


def _accuracy_row_map(candidate_dir: Path) -> dict[str, dict[str, Any]]:
    df = _load_accuracy_headline(candidate_dir)
    out: dict[str, dict[str, Any]] = {}
    for _, row in df.iterrows():
        out[str(row["experiment_id"])] = row.to_dict()
    return out


def _e0_model_accuracy(candidate_dir: Path) -> dict[str, dict[str, float]]:
    df = _load_accuracy_per_model(candidate_dir)
    df = df[df["experiment_id"].astype(str) == "E0"].copy()
    out: dict[str, dict[str, float]] = {}
    for _, row in df.iterrows():
        out[str(row["model"])] = {
            "top1": float(row["nonbaseline_top1_mean"]),
            "drop_vs_fp32": float(-row["delta_vs_fp32_pp"]),
        }
    return out


def _merge_accuracy_fields(quick_df: pd.DataFrame, candidate_dir: Path) -> pd.DataFrame:
    headline = _accuracy_row_map(candidate_dir)
    e0 = headline["E0"]
    inherited = {
        "measured_acc_top1": float(e0["top1_measured_mean"]),
        "measured_acc_drop_pp_vs_fp32_mean": float(e0["acc_drop_pp_vs_fp32_mean"]),
        "measured_acc_drop_pp_vs_int8_noapprox_mean": 0.0,
        "measured_acc_drop_pp_vs_E0_mean": 0.0,
        "accuracy_evidence": "shared_e0_full_eval_reference",
        "accuracy_note": "shares E0 quantized full-eval reference",
        "acc_drop_pp_nonempty": int(e0["n_rows"]),
    }
    rows: list[dict[str, Any]] = []
    for _, row in quick_df.iterrows():
        exp = str(row["experiment_id"])
        merged = dict(row)
        if exp in {"E1", "E5"}:
            stats = inherited
            merged.update(stats)
            merged["measured_acc_drop_pp_mean"] = stats["measured_acc_drop_pp_vs_fp32_mean"]
            merged["modeled_acc_top1"] = math.nan
            merged["modeled_acc_drop_pp_mean"] = math.nan
            merged["modeled_acc_drop_pp_vs_fp32_mean"] = math.nan
            merged["modeled_acc_drop_pp_vs_int8_noapprox_mean"] = math.nan
            merged["modeled_acc_drop_pp_vs_E0_mean"] = math.nan
            merged["acc_top1_raw"] = stats["measured_acc_top1"]
            merged["acc_top1"] = stats["measured_acc_top1"]
            merged["acc_drop_pp_mean"] = stats["measured_acc_drop_pp_vs_fp32_mean"]
        else:
            stats = headline.get(exp)
            if not stats:
                raise SystemExit(f"Missing headline accuracy row for {exp}")
            evidence = str(stats.get("accuracy_evidence") or "")
            modeled = "modeled" in evidence
            if modeled:
                merged["measured_acc_top1"] = math.nan
                merged["measured_acc_drop_pp_mean"] = math.nan
                merged["measured_acc_drop_pp_vs_fp32_mean"] = math.nan
                merged["measured_acc_drop_pp_vs_int8_noapprox_mean"] = math.nan
                merged["measured_acc_drop_pp_vs_E0_mean"] = math.nan
                merged["modeled_acc_top1"] = float(stats["top1_measured_mean"])
                merged["modeled_acc_drop_pp_mean"] = float(stats["acc_drop_pp_vs_fp32_mean"])
                merged["modeled_acc_drop_pp_vs_fp32_mean"] = float(stats["acc_drop_pp_vs_fp32_mean"])
                merged["modeled_acc_drop_pp_vs_E0_mean"] = float(stats["acc_drop_pp_vs_fp32_mean"]) - float(e0["acc_drop_pp_vs_fp32_mean"])
                merged["modeled_acc_drop_pp_vs_int8_noapprox_mean"] = merged["modeled_acc_drop_pp_vs_E0_mean"]
                merged["acc_top1_raw"] = merged["modeled_acc_top1"]
                merged["acc_top1"] = merged["modeled_acc_top1"]
                merged["acc_drop_pp_mean"] = merged["modeled_acc_drop_pp_vs_fp32_mean"]
            else:
                merged["measured_acc_top1"] = float(stats["top1_measured_mean"])
                merged["measured_acc_drop_pp_mean"] = float(stats["acc_drop_pp_vs_fp32_mean"])
                merged["measured_acc_drop_pp_vs_fp32_mean"] = float(stats["acc_drop_pp_vs_fp32_mean"])
                merged["measured_acc_drop_pp_vs_E0_mean"] = float(stats["acc_drop_pp_vs_fp32_mean"]) - float(e0["acc_drop_pp_vs_fp32_mean"])
                merged["measured_acc_drop_pp_vs_int8_noapprox_mean"] = merged["measured_acc_drop_pp_vs_E0_mean"]
                merged["modeled_acc_top1"] = math.nan
                merged["modeled_acc_drop_pp_mean"] = math.nan
                merged["modeled_acc_drop_pp_vs_fp32_mean"] = math.nan
                merged["modeled_acc_drop_pp_vs_int8_noapprox_mean"] = math.nan
                merged["modeled_acc_drop_pp_vs_E0_mean"] = math.nan
                merged["acc_top1_raw"] = merged["measured_acc_top1"]
                merged["acc_top1"] = merged["measured_acc_top1"]
                merged["acc_drop_pp_mean"] = merged["measured_acc_drop_pp_vs_fp32_mean"]
            merged["accuracy_evidence"] = evidence
            merged["accuracy_note"] = str(stats.get("accuracy_note") or "")
            merged["acc_drop_pp_nonempty"] = int(stats["n_rows"])
        merged["acc_modeled_add_pp"] = 0.0
        rows.append(merged)
    return pd.DataFrame(rows)


def build_quickpack(quick_dir: Path, candidate_dir: Path) -> pd.DataFrame:
    baseline_grouped = _group_master_by_model_from_dir(CORE_SOURCE_DIRS["E0"])
    baseline_latency_by_model = {
        str(row["model"]): float(row["latency_ms"])
        for _, row in baseline_grouped.iterrows()
        if pd.notna(row["latency_ms"])
    }
    rows = [
        _aggregate_core_row(exp, CORE_RUNS[exp], CORE_SOURCE_DIRS[exp], baseline_latency_by_model)
        for exp in CORE_RUNS
    ]
    df = pd.DataFrame(rows)
    df = _merge_accuracy_fields(df, candidate_dir)
    ordered = [
        "experiment_id",
        "run_id",
        "n_models",
        "latency_ms",
        "energy_j",
        "tops_w",
        "speedup_vs_E0",
        "duty_cycle_avg",
        "avg_effective_bsl",
        "det_net_gain_j",
        "PP_crosstalk_db",
        "P_laser_dbm",
        "P_laser_mw",
        "utilization_avg",
        "bubble_cycles",
        "fanout",
        "serializers_saved",
        "broadcast_driver_energy_j",
        "net_energy_gain_j",
        "N_wdm",
        "measured_acc_top1",
        "measured_acc_drop_pp_mean",
        "measured_acc_drop_pp_vs_fp32_mean",
        "measured_acc_drop_pp_vs_int8_noapprox_mean",
        "measured_acc_drop_pp_vs_E0_mean",
        "modeled_acc_top1",
        "modeled_acc_drop_pp_mean",
        "modeled_acc_drop_pp_vs_fp32_mean",
        "modeled_acc_drop_pp_vs_int8_noapprox_mean",
        "modeled_acc_drop_pp_vs_E0_mean",
        "acc_top1_raw",
        "acc_modeled_add_pp",
        "acc_top1",
        "acc_drop_pp_mean",
        "throughput_images_s",
        "acc_drop_pp_nonempty",
        "pass_det_net_gain_true",
        "accuracy_evidence",
        "accuracy_note",
    ]
    return df[ordered].sort_values("experiment_id").reset_index(drop=True)


def build_ablation_summary(quickpack: pd.DataFrame) -> pd.DataFrame:
    e0 = quickpack[quickpack["experiment_id"] == "E0"].iloc[0]
    rows: list[dict[str, Any]] = []
    for _, row in quickpack.iterrows():
        measured = pd.notna(row["measured_acc_drop_pp_vs_fp32_mean"])
        rows.append(
            {
                "experiment_id": row["experiment_id"],
                "run_id": row["run_id"],
                "n_models": row["n_models"],
                "energy_j": row["energy_j"],
                "latency_ms": row["latency_ms"],
                "tops_w": row["tops_w"],
                "speedup_vs_E0": row["speedup_vs_E0"],
                "measured_acc_drop_pp_mean": row["measured_acc_drop_pp_vs_fp32_mean"] if measured else math.nan,
                "modeled_acc_drop_pp_mean": row["modeled_acc_drop_pp_vs_fp32_mean"] if not measured else math.nan,
                "measured_acc_drop_pp_vs_fp32_mean": row["measured_acc_drop_pp_vs_fp32_mean"],
                "measured_acc_drop_pp_vs_int8_noapprox_mean": row["measured_acc_drop_pp_vs_int8_noapprox_mean"],
                "measured_acc_drop_pp_vs_E0_mean": row["measured_acc_drop_pp_vs_E0_mean"],
                "modeled_acc_drop_pp_vs_fp32_mean": row["modeled_acc_drop_pp_vs_fp32_mean"],
                "modeled_acc_drop_pp_vs_int8_noapprox_mean": row["modeled_acc_drop_pp_vs_int8_noapprox_mean"],
                "modeled_acc_drop_pp_vs_E0_mean": row["modeled_acc_drop_pp_vs_E0_mean"],
                "acc_modeled_add_pp": 0.0,
                "acc_drop_pp_mean": row["acc_drop_pp_mean"],
                "utilization_avg": row["utilization_avg"],
                "energy_ratio_vs_E0": float(row["energy_j"]) / float(e0["energy_j"]),
                "latency_ratio_vs_E0": float(row["latency_ms"]) / float(e0["latency_ms"]),
                "accuracy_evidence": row["accuracy_evidence"],
                "accuracy_note": row["accuracy_note"],
            }
        )
    return pd.DataFrame(rows).sort_values("experiment_id").reset_index(drop=True)


def build_energy_breakdown_summary() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for exp, run_id in CORE_RUNS.items():
        grouped = _group_master_by_model_from_dir(CORE_SOURCE_DIRS[exp])
        item = {
            "experiment_id": exp,
            "run_id": run_id,
            "n_models": int(grouped["model"].nunique()),
            "energy_j": _mean_of_models(grouped, "energy_j"),
            "energy_breakdown_conversion_control_j": _mean_of_models(grouped, "energy_breakdown_conversion_control_j"),
            "energy_breakdown_memory_move_j": _mean_of_models(grouped, "energy_breakdown_memory_move_j"),
            "energy_breakdown_oe_j": _mean_of_models(grouped, "energy_breakdown_oe_j"),
            "energy_breakdown_adc_pca_j": _mean_of_models(grouped, "energy_breakdown_adc_pca_j"),
            "energy_breakdown_laser_optical_j": _mean_of_models(grouped, "energy_breakdown_laser_optical_j"),
            "energy_breakdown_other_static_j": _mean_of_models(grouped, "energy_breakdown_other_static_j"),
        }
        rows.append(item)
    return pd.DataFrame(rows).sort_values("experiment_id").reset_index(drop=True)


def build_det_net_gain_waterfall() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for exp in ["E3", "E6"]:
        run_id = CORE_RUNS[exp]
        raw = pd.read_csv(CORE_SOURCE_DIRS[exp] / "master_metrics.csv")
        if raw.empty:
            continue
        for col in [
            "det_overhead_j",
            "det_saved_j",
            "det_net_gain_j",
            "energy_j",
            "avg_effective_bsl",
        ]:
            if col in raw.columns:
                raw[col] = _to_num(raw[col])
        rows.append(
            {
                "experiment_id": exp,
                "run_id": run_id,
                "n_models": int(len(raw)),
                "det_overhead_j": float(raw["det_overhead_j"].mean()) if "det_overhead_j" in raw else math.nan,
                "det_saved_j": float(raw["det_saved_j"].mean()) if "det_saved_j" in raw else math.nan,
                "det_net_gain_j": float(raw["det_net_gain_j"].mean()) if "det_net_gain_j" in raw else math.nan,
                "pass_det_net_gain_true": int(_to_num(raw.get("pass_det_net_gain", pd.Series(0, index=raw.index))).fillna(0).sum()),
                "energy_j": float(raw["energy_j"].mean()) if "energy_j" in raw else math.nan,
                "avg_effective_bsl": float(raw["avg_effective_bsl"].mean()) if "avg_effective_bsl" in raw else math.nan,
            }
        )
    return pd.DataFrame(rows).sort_values("experiment_id").reset_index(drop=True)


def _module_group(name: str) -> str:
    lower = name.lower()
    if "pre_norm_mha" in lower or "qkv_proj" in lower or "out_proj" in lower:
        return "attention"
    if "pre_norm_ffn" in lower:
        return "ffn"
    if ".conv" in lower:
        return "conv"
    return "other"


def build_module_breakdown(run_tag: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for exp, run_id in CORE_RUNS.items():
        ops_dir = CORE_SOURCE_DIRS[exp] / "mtl_ops"
        model_rows: list[dict[str, Any]] = []
        for ops_path in sorted(ops_dir.glob("*_ops.csv")):
            df = pd.read_csv(ops_path)
            if df.empty:
                continue
            df["module_group"] = df["name"].astype(str).map(_module_group)
            df["latency_ms"] = _to_num(df["latency_ms"])
            df["energy_j"] = _to_num(df["energy_mj"]) / 1e3
            agg = df.groupby("module_group", as_index=False)[["latency_ms", "energy_j"]].sum()
            total_latency = float(agg["latency_ms"].sum())
            total_energy = float(agg["energy_j"].sum())
            for _, item in agg.iterrows():
                model_rows.append(
                    {
                        "module_group": item["module_group"],
                        "module_latency_ms": float(item["latency_ms"]),
                        "module_energy_j": float(item["energy_j"]),
                        "latency_ms": total_latency,
                        "energy_j": total_energy,
                    }
                )
        if not model_rows:
            continue
        work = pd.DataFrame(model_rows)
        for mod, sub in work.groupby("module_group"):
            module_latency = float(sub["module_latency_ms"].mean())
            module_energy = float(sub["module_energy_j"].mean())
            total_latency = float(sub["latency_ms"].mean())
            total_energy = float(sub["energy_j"].mean())
            rows.append(
                {
                    "run_id": run_id,
                    "run_tag": run_tag,
                    "experiment_id": exp,
                    "model": "all_models",
                    "module_group": mod,
                    "module_latency_ms": module_latency,
                    "module_energy_j": module_energy,
                    "module_latency_ratio": module_latency / total_latency if total_latency > 0 else math.nan,
                    "module_energy_ratio": module_energy / total_energy if total_energy > 0 else math.nan,
                    "latency_ms": total_latency,
                    "energy_j": total_energy,
                }
            )
    return pd.DataFrame(rows).sort_values(["experiment_id", "module_group"]).reset_index(drop=True)


def build_batch_seq_scaling(run_tag: str, candidate_dir: Path) -> pd.DataFrame:
    e0_accuracy = _e0_model_accuracy(candidate_dir)
    run_ids = [CORE_RUNS["E0"], *SCALING_RUNS]
    rows: list[dict[str, Any]] = []
    for run_id in run_ids:
        grouped = _group_master_by_model(run_id)
        cfg = _config_snapshot(run_id)
        data_cfg = cfg.get("data") or {}
        batch_size = int(float(data_cfg.get("batch_size") or 1))
        sequence_length = int(float(data_cfg.get("sequence_length") or 197))
        workload_id = str(data_cfg.get("workload_id") or "W0_mobilevit_imagenet")
        for _, item in grouped.iterrows():
            model = str(item["model"])
            latency_ms = float(item["latency_ms"])
            throughput_images = (batch_size / (latency_ms / 1e3)) if latency_ms > 0 else math.nan
            throughput_tokens = (batch_size * sequence_length / (latency_ms / 1e3)) if latency_ms > 0 else math.nan
            acc = e0_accuracy[model]
            rows.append(
                {
                    "run_id": run_id,
                    "run_tag": run_tag,
                    "experiment_id": "E0",
                    "model": model,
                    "model_family": "mobilevit",
                    "task_id": "imagenet_cls",
                    "workload_id": workload_id,
                    "latency_ms": latency_ms,
                    "energy_j": float(item["energy_j"]),
                    "avg_power_w": float(item["avg_power_w"]),
                    "tops_w": float(item["tops_w"]),
                    "throughput_images_s": throughput_images,
                    "throughput_tokens_s": throughput_tokens,
                    "primary_metric_name": "Top1",
                    "primary_metric_value": acc["top1"],
                    "primary_metric_drop": acc["drop_vs_fp32"],
                    "acc_top1": acc["top1"],
                    "acc_drop_pp": acc["drop_vs_fp32"],
                    "platform_class": "HPAT",
                    "source_type": "simulated_run",
                    "batch_size": batch_size,
                    "sequence_length": sequence_length,
                    "input_size": int(float(item["input_size"])) if "input_size" in item and pd.notna(item["input_size"]) else math.nan,
                    "N_wdm": 16,
                    "mrr_tile_k": 16,
                    "quant_bits": 8,
                    "fanout": 0,
                    "broadcast_mode": "one_to_one",
                    "n_runs": 1,
                    "scaling_profile_id": f"batch={batch_size}|seq={sequence_length}|model={model}",
                }
            )
    return pd.DataFrame(rows).sort_values(["model", "sequence_length", "batch_size"]).reset_index(drop=True)


def _sparse_subset_measurements(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    base = df[(df["experiment_id"].astype(str) == "E0") & (~df["baseline"].astype(bool))].copy()
    base["seed"] = _to_num(base["seed"])
    base_map = base.groupby(["model", "seed"], as_index=False)["top1"].mean().rename(columns={"top1": "e0_top1"})
    sub = df[(df["experiment_id"].astype(str) == "E4") & (~df["baseline"].astype(bool))].copy()
    sub["seed"] = _to_num(sub["seed"])
    merged = sub.merge(base_map, on=["model", "seed"], how="left")
    merged["drop_vs_e0_quant_pp"] = merged["e0_top1"] - merged["top1"]
    out = (
        merged.groupby("sparse_tau_global", as_index=False)["drop_vs_e0_quant_pp"]
        .mean()
        .rename(columns={"sparse_tau_global": "tau", "drop_vs_e0_quant_pp": "measured_subset_acc_drop_pp_vs_e0_mean"})
    )
    return out.sort_values("tau").reset_index(drop=True)


def _project_sparse_drop_vs_e0(tau: float, duty: float, measured_subset: pd.DataFrame, quickpack: pd.DataFrame) -> float:
    inactive = 1.0 - duty
    subset = measured_subset.copy()
    subset["duty_cycle_avg"] = np.exp(-subset["tau"].astype(float))
    subset["inactive_fraction"] = 1.0 - subset["duty_cycle_avg"]
    anchor_rows = subset[["inactive_fraction", "measured_subset_acc_drop_pp_vs_e0_mean"]].to_dict("records")
    e4 = quickpack[quickpack["experiment_id"] == "E4"].iloc[0]
    full_anchor = {
        "inactive_fraction": 1.0 - float(e4["duty_cycle_avg"]),
        "measured_subset_acc_drop_pp_vs_e0_mean": float(e4["measured_acc_drop_pp_vs_E0_mean"]),
    }
    anchors = sorted(anchor_rows + [full_anchor], key=lambda item: item["inactive_fraction"])
    xs = np.array([float(item["inactive_fraction"]) for item in anchors], dtype=float)
    ys = np.array([float(item["measured_subset_acc_drop_pp_vs_e0_mean"]) for item in anchors], dtype=float)
    if inactive <= xs[0]:
        return float(ys[0])
    if inactive >= xs[-1]:
        slope = (ys[-1] - ys[-2]) / max(xs[-1] - xs[-2], 1e-9)
        return float(ys[-1] + slope * (inactive - xs[-1]))
    return float(np.interp(inactive, xs, ys))


def build_sparse_tau_pareto(quick_dir: Path, sparse_backup: Path, quickpack: pd.DataFrame) -> pd.DataFrame:
    subset = _sparse_subset_measurements(sparse_backup)
    e0_drop = float(quickpack[quickpack["experiment_id"] == "E0"]["measured_acc_drop_pp_vs_fp32_mean"].iloc[0])
    perf_rows: list[dict[str, Any]] = []
    baseline_energy = None
    for run_id in SPARSE_TAU_RUNS:
        grouped = _group_master_by_model(run_id)
        tau = _mean_of_models(grouped, "tau_i")
        duty = _mean_of_models(grouped, "duty_cycle_avg")
        energy = _mean_of_models(grouped, "energy_j")
        if tau is None or duty is None or energy is None:
            continue
        if baseline_energy is None and abs(tau) < 1e-9:
            baseline_energy = energy
        projected_vs_e0 = _project_sparse_drop_vs_e0(float(tau), float(duty), subset, quickpack)
        measured_subset_vs_e0 = math.nan
        hit = subset[np.isclose(subset["tau"], float(tau))]
        if not hit.empty:
            measured_subset_vs_e0 = float(hit["measured_subset_acc_drop_pp_vs_e0_mean"].iloc[0])
        perf_rows.append(
            {
                "run_id": run_id,
                "tau": float(tau),
                "duty_cycle_avg": float(duty),
                "energy_j": float(energy),
                "projected_acc_drop_pp_vs_e0_mean": projected_vs_e0,
                "measured_subset_acc_drop_pp_vs_e0_mean": measured_subset_vs_e0,
                "modeled_acc_drop_pp_mean": e0_drop + projected_vs_e0,
                "measured_acc_drop_pp_mean": (e0_drop + measured_subset_vs_e0) if pd.notna(measured_subset_vs_e0) else math.nan,
                "acc_drop_pp_mean": projected_vs_e0,
                "accuracy_evidence": (
                    "subset_measured_anchor" if pd.notna(measured_subset_vs_e0) else "activity_calibrated_projection"
                ),
            }
        )
    out = pd.DataFrame(perf_rows).sort_values("tau").reset_index(drop=True)
    if baseline_energy is None:
        raise SystemExit("Sparse tau baseline energy is missing.")
    out["energy_saved_pct"] = (1.0 - (out["energy_j"] / float(baseline_energy))) * 100.0
    return out


def build_alignment_summary(quickpack: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for exp in ["E5", "E6"]:
        run_id = CORE_RUNS[exp]
        cfg = _config_snapshot(run_id)
        p1 = cfg.get("p1_align") or {}
        points = p1.get("p1_alignment_points_db") or []
        sigma_step = float(p1.get("sigma_lsb_per_3db") or 0.1)
        alpha_step = float(p1.get("crosstalk_alpha_per_3db") or 0.02)
        quick_row = quickpack[quickpack["experiment_id"] == exp].iloc[0]
        p_ref = quick_row["P_laser_dbm"]
        for raw in points:
            delta = float(raw)
            severity = max(0.0, -delta) / 3.0
            rows.append(
                {
                    "experiment_id": exp,
                    "run_id": run_id,
                    "delta_p_db": delta,
                    "p_laser_dbm_eff": float(p_ref) + delta if pd.notna(p_ref) else math.nan,
                    "sigma_lsb_pred": sigma_step * severity,
                    "crosstalk_alpha_pred": alpha_step * severity,
                    "acc_drop_pp_mean": float(quick_row["acc_drop_pp_mean"]),
                    "alignment_evidence": "config_fixed_rebuilt",
                }
            )
    return pd.DataFrame(rows).sort_values(["experiment_id", "delta_p_db"]).reset_index(drop=True)


def build_det_projection(quick_dir: Path, candidate_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    det = pd.read_csv(quick_dir / "quickscan_e3_k_sweep.csv").copy()
    det["k_global"] = _to_num(det["k_global"])
    det["prefix_error_mean"] = _to_num(det["prefix_error_mean"])
    det["speedup_vs_E0"] = _to_num(det["speedup_vs_E0"])
    det = det.sort_values("k_global").reset_index(drop=True)
    anchor = float(
        pd.read_csv(_candidate_file(candidate_dir, "config_conditioned_accuracy_summary"))
        .set_index("experiment_id")
        .loc["E3", "paired_model_mean_delta_vs_e0_quant_pp"]
    )
    det_cap = float(
        pd.read_csv(_candidate_file(candidate_dir, "headline_statistics_pair_ci"))
        .set_index("experiment_id")
        .loc["E3", "acc_drop_pp_vs_e0_ci95_high"]
    )
    e0_drop = float(
        pd.read_csv(_candidate_file(candidate_dir, "headline_statistics_summary"))
        .set_index("experiment_id")
        .loc["E0", "acc_drop_pp_vs_fp32_mean"]
    )
    prefix_anchor = float(det.loc[np.isclose(det["k_global"], 64.0), "prefix_error_mean"].iloc[0])
    projected_delta_vs_e0 = []
    evidence = []
    for _, row in det.iterrows():
        k = float(row["k_global"])
        if abs(k - 64.0) < 1e-9:
            projected_delta_vs_e0.append(anchor)
            evidence.append("measured_full_eval_anchor")
        elif abs(k - 129.0) < 1e-9:
            projected_delta_vs_e0.append(0.0)
            evidence.append("equivalence_anchor")
        else:
            ratio = float(row["prefix_error_mean"]) / max(prefix_anchor, 1e-12)
            projected = -min(det_cap, abs(anchor) * math.sqrt(max(ratio, 0.0)))
            projected_delta_vs_e0.append(projected)
            evidence.append("prefix_error_projected")
    det["projected_delta_vs_e0_quant_pp"] = projected_delta_vs_e0
    det["measured_acc_drop_pp_vs_E0_mean"] = np.where(np.isclose(det["k_global"], 64.0), -anchor, np.nan)
    det["modeled_acc_drop_pp_vs_E0_mean"] = -det["projected_delta_vs_e0_quant_pp"]
    det["measured_acc_drop_pp_mean"] = np.where(
        np.isclose(det["k_global"], 64.0),
        e0_drop - anchor,
        np.nan,
    )
    det["modeled_acc_drop_pp_mean"] = e0_drop - det["projected_delta_vs_e0_quant_pp"]
    det["acc_drop_pp_mean"] = det["modeled_acc_drop_pp_mean"]
    det["acc_evidence_tier"] = evidence

    det_summary = pd.DataFrame(
        {
            "det_k_global": det["k_global"],
            "avg_effective_bsl": det.get("avg_effective_bsl"),
            "speedup_vs_E0": det["speedup_vs_E0"],
            "prefix_error_mean": det["prefix_error_mean"],
            "paired_model_mean_delta_vs_e0_quant_pp": det["projected_delta_vs_e0_quant_pp"],
            "paired_model_mean_delta_vs_fp32_pp": -det["modeled_acc_drop_pp_mean"],
            "evidence_tier": evidence,
        }
    )
    return det, det_summary


def build_appf1_internal_model_scale_context(quick_dir: Path) -> pd.DataFrame:
    path = quick_dir / "task_generalization_summary.csv"
    df = pd.read_csv(path)
    if df.empty:
        raise SystemExit(f"Empty task generalization summary: {path}")

    src = df.get("source_type", pd.Series("", index=df.index)).astype(str).str.lower()
    work = df.loc[src.ne("external_anchor")].copy()
    work = work[work["experiment_id"].astype(str).isin(list(CORE_RUNS.keys()))].copy()
    if work.empty:
        raise SystemExit("No internal AppF1 rows remain after the external-anchor gate.")

    numeric_cols = ["latency_ms", "energy_j", "tops_w", "primary_metric_value", "peak_tops"]
    for col in numeric_cols:
        if col in work.columns:
            work[col] = _to_num(work[col])

    group_cols = [
        "experiment_id",
        "model",
        "model_family",
        "task_id",
        "workload_id",
        "primary_metric_name",
    ]
    agg = (
        work.groupby(group_cols, as_index=False)
        .agg(
            latency_ms=("latency_ms", "mean"),
            energy_j=("energy_j", "mean"),
            tops_w=("tops_w", "mean"),
            primary_metric_value=("primary_metric_value", "mean"),
            peak_tops=("peak_tops", "mean"),
            n_source_rows=("experiment_id", "size"),
        )
    )
    order = list(CORE_RUNS.keys())
    agg["experiment_id"] = pd.Categorical(agg["experiment_id"], categories=order, ordered=True)
    agg = agg.sort_values(["experiment_id", "model"]).reset_index(drop=True)
    agg["source_scope"] = "internal_single_workload_context"
    agg["evidence_tier"] = "simulated_internal_context"
    return agg[
        [
            "experiment_id",
            "model",
            "model_family",
            "task_id",
            "workload_id",
            "primary_metric_name",
            "primary_metric_value",
            "latency_ms",
            "energy_j",
            "tops_w",
            "peak_tops",
            "n_source_rows",
            "source_scope",
            "evidence_tier",
        ]
    ]


def build_fig13_heatmap_aggregated(quick_dir: Path) -> pd.DataFrame:
    path = quick_dir / "fig_h_accuracy_heatmap_points.csv"
    df = pd.read_csv(path)
    if df.empty:
        raise SystemExit(f"Empty heatmap source: {path}")

    for col in ["quant_bits", "sigma_lsb", "crosstalk_alpha", "acc_drop_pp", "top1", "top1_ref"]:
        df[col] = _to_num(df[col])
    df = df.dropna(subset=["quant_bits", "sigma_lsb", "crosstalk_alpha", "acc_drop_pp"])
    df = df.drop_duplicates().reset_index(drop=True)
    if df.empty:
        raise SystemExit("No valid Fig13 source rows after numeric filtering.")

    per_model = (
        df.groupby(["model", "quant_bits", "sigma_lsb", "crosstalk_alpha"], as_index=False)
        .agg(
            acc_drop_pp_model_mean=("acc_drop_pp", "mean"),
            top1_model_mean=("top1", "mean"),
            top1_ref_mean=("top1_ref", "mean"),
            replicate_count=("acc_drop_pp", "size"),
        )
    )
    agg = (
        per_model.groupby(["quant_bits", "sigma_lsb", "crosstalk_alpha"], as_index=False)
        .agg(
            acc_drop_pp_cell_mean=("acc_drop_pp_model_mean", "mean"),
            top1_cell_mean=("top1_model_mean", "mean"),
            top1_ref_cell_mean=("top1_ref_mean", "mean"),
            model_count=("model", "nunique"),
            replicate_count_total=("replicate_count", "sum"),
            replicate_count_per_model_min=("replicate_count", "min"),
            replicate_count_per_model_max=("replicate_count", "max"),
            source_models=("model", lambda s: "|".join(sorted(set(str(v) for v in s)))),
        )
    )
    agg["aggregation_rule"] = "mean_over_replicates_per_model_then_mean_over_models"
    agg = agg.sort_values(["quant_bits", "crosstalk_alpha", "sigma_lsb"]).reset_index(drop=True)
    return agg[
        [
            "quant_bits",
            "sigma_lsb",
            "crosstalk_alpha",
            "acc_drop_pp_cell_mean",
            "top1_cell_mean",
            "top1_ref_cell_mean",
            "model_count",
            "replicate_count_total",
            "replicate_count_per_model_min",
            "replicate_count_per_model_max",
            "source_models",
            "aggregation_rule",
        ]
    ]


def build_fig19_contract_ablation() -> pd.DataFrame:
    path = FULLER_ABLATION_SUMMARY
    df = pd.read_csv(path)
    if df.empty:
        raise SystemExit(f"Empty contract ablation source: {path}")

    rename = {
        "ASTRA": "ASTRA",
        "HOPS": "HOPS",
        "FLOW_MESO": "HOPS+MESO",
        "FLOW_PHY": "HOPS+PHY",
        "FULLER": "FULLER",
    }
    work = df[df["mechanism_variant"].astype(str).isin(rename)].copy()
    if work.empty:
        raise SystemExit("No recognized mechanism rows in fuller ablation summary.")
    work["variant_label"] = work["mechanism_variant"].astype(str).map(rename)
    order = ["ASTRA", "HOPS", "HOPS+MESO", "HOPS+PHY", "FULLER"]
    work["variant_label"] = pd.Categorical(work["variant_label"], categories=order, ordered=True)
    work = work.sort_values("variant_label").reset_index(drop=True)

    astra = work[work["variant_label"] == "ASTRA"]
    if astra.empty:
        raise SystemExit("ASTRA baseline missing in fuller ablation summary.")
    astra_latency = float(astra["latency_ms"].iloc[0])
    astra_energy = float(astra["energy_j"].iloc[0])

    work["speedup_vs_astra"] = astra_latency / work["latency_ms"].astype(float)
    work["energy_ratio_vs_astra"] = work["energy_j"].astype(float) / astra_energy
    work["energy_reduction_pct_vs_astra"] = (1.0 - work["energy_ratio_vs_astra"]) * 100.0
    if "accuracy_evidence" not in work.columns:
        work["accuracy_evidence"] = "report_data_ablation_row"
    if "accuracy_source_csv" not in work.columns:
        work["accuracy_source_csv"] = ""
    if "accuracy_context_run_id" not in work.columns:
        work["accuracy_context_run_id"] = ""
    if "accuracy_target_notes" not in work.columns:
        work["accuracy_target_notes"] = ""
    work["source_table"] = str(path)
    return work[
        [
            "variant_label",
            "mechanism_variant",
            "latency_ms",
            "energy_j",
            "avg_power_w",
            "acc_top1",
            "acc_drop_pp",
            "speedup_vs_astra",
            "energy_ratio_vs_astra",
            "energy_reduction_pct_vs_astra",
            "bubble_cycles",
            "utilization_avg",
            "accuracy_evidence",
            "accuracy_source_csv",
            "accuracy_context_run_id",
            "accuracy_target_notes",
            "source_table",
        ]
    ]


def rerun_targets(run_ids: list[str], python_bin: str) -> None:
    tmp_dir = Path("/tmp/fyp_strict_repair_cfgs")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    for index, run_id in enumerate(run_ids, start=1):
        cfg = _config_snapshot(run_id)
        outputs = dict(cfg.get("outputs") or {})
        outputs["out_dir"] = str(RUNS)
        outputs["append_master"] = False
        outputs["save_config_snapshot"] = False
        cfg["outputs"] = outputs
        cfg_path = tmp_dir / f"{run_id}.yaml"
        with cfg_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(cfg, handle, sort_keys=False)
        print(f"[strict-freeze-rebuild] ({index}/{len(run_ids)}) rerun {run_id}")
        subprocess.run(
            [python_bin, str(PHASE1_RUNNER), "--config", str(cfg_path)],
            check=True,
            cwd=str(ROOT),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebuild strict-content quick reports for the frozen paper pack.")
    parser.add_argument("--quick_dir", type=Path, default=DEFAULT_QUICK_DIR)
    parser.add_argument("--candidate_dir", type=Path, default=DEFAULT_CANDIDATE_DIR)
    parser.add_argument("--candidate_tag", default=DEFAULT_CANDIDATE_TAG)
    parser.add_argument("--sparse_backup_csv", type=Path, default=DEFAULT_SPARSE_BACKUP)
    parser.add_argument("--cpu_device_metrics_csv", type=Path, default=None)
    parser.add_argument("--gpu_device_metrics_csv", type=Path, default=GPU_DEVICE_METRICS)
    parser.add_argument("--fuller_model_summary_csv", type=Path, default=FULLER_MODEL_SUMMARY)
    parser.add_argument("--fuller_ablation_summary_csv", type=Path, default=FULLER_ABLATION_SUMMARY)
    parser.add_argument("--core_runs_json", default="")
    parser.add_argument("--scaling_runs_json", default="")
    parser.add_argument("--sparse_tau_runs_json", default="")
    parser.add_argument("--core_source_root", type=Path, default=None)
    parser.add_argument("--python_bin", default=str(ROOT / ".venv311-mps" / "bin" / "python"))
    parser.add_argument("--rerun", action="store_true", help="Recompute the bounded stale performance runs before rebuilding quick reports.")
    return parser.parse_args()


def _json_payload(raw: str) -> Any:
    text = str(raw or "").strip()
    if not text:
        return None
    if not text.startswith(("{", "[")):
        candidate = Path(text)
        try:
            if candidate.exists():
                text = candidate.read_text(encoding="utf-8")
        except OSError:
            pass
    return json.loads(text)


def main() -> None:
    global CANDIDATE_TAG, CPU_DEVICE_METRICS, GPU_DEVICE_METRICS, FULLER_MODEL_SUMMARY, FULLER_ABLATION_SUMMARY
    global CORE_RUNS, SCALING_RUNS, SPARSE_TAU_RUNS, CORE_SOURCE_DIRS, RERUN_TARGETS
    args = parse_args()
    quick_dir = args.quick_dir.resolve()
    quick_dir.mkdir(parents=True, exist_ok=True)
    CANDIDATE_TAG = str(args.candidate_tag).strip()
    CPU_DEVICE_METRICS = args.cpu_device_metrics_csv.resolve() if args.cpu_device_metrics_csv else None
    GPU_DEVICE_METRICS = args.gpu_device_metrics_csv.resolve()
    FULLER_MODEL_SUMMARY = args.fuller_model_summary_csv.resolve()
    FULLER_ABLATION_SUMMARY = args.fuller_ablation_summary_csv.resolve()
    core_runs_payload = _json_payload(args.core_runs_json)
    if core_runs_payload:
        CORE_RUNS = {str(key): str(value) for key, value in dict(core_runs_payload).items()}
    scaling_runs_payload = _json_payload(args.scaling_runs_json)
    if scaling_runs_payload:
        SCALING_RUNS = [str(item) for item in list(scaling_runs_payload)]
    sparse_tau_payload = _json_payload(args.sparse_tau_runs_json)
    if sparse_tau_payload:
        SPARSE_TAU_RUNS = [str(item) for item in list(sparse_tau_payload)]
    core_source_root = args.core_source_root.resolve() if args.core_source_root else None
    if core_source_root is not None:
        CORE_SOURCE_DIRS = {
            exp: core_source_root / run_id
            for exp, run_id in CORE_RUNS.items()
        }
    elif core_runs_payload:
        CORE_SOURCE_DIRS = {exp: RUNS / run_id for exp, run_id in CORE_RUNS.items()}
    RERUN_TARGETS = [
        CORE_RUNS["E4"],
        CORE_RUNS["E6"],
        *SCALING_RUNS,
        *SPARSE_TAU_RUNS,
    ]
    if args.rerun:
        rerun_targets(RERUN_TARGETS, args.python_bin)

    quickpack = build_quickpack(quick_dir, args.candidate_dir.resolve())
    _write_csv(quick_dir / "quickpack_e0_e6_overview.csv", quickpack)
    _write_csv(quick_dir / "ablation_summary.csv", build_ablation_summary(quickpack))
    _write_csv(quick_dir / "energy_breakdown_summary.csv", build_energy_breakdown_summary())
    _write_csv(quick_dir / "det_net_gain_waterfall.csv", build_det_net_gain_waterfall())
    _write_csv(quick_dir / "module_breakdown_by_block.csv", build_module_breakdown(quick_dir.name))
    _write_csv(quick_dir / "quickscan_batch_seq_scaling.csv", build_batch_seq_scaling(quick_dir.name, args.candidate_dir.resolve()))
    _write_csv(quick_dir / "fig_j_sparse_tau_pareto.csv", build_sparse_tau_pareto(quick_dir, args.sparse_backup_csv.resolve(), quickpack))
    _write_csv(quick_dir / "p0_p1_alignment_summary.csv", build_alignment_summary(quickpack))
    det, det_summary = build_det_projection(quick_dir, args.candidate_dir.resolve())
    _write_csv(quick_dir / "quickscan_e3_k_sweep.csv", det)
    _write_csv(quick_dir / "fig8_det_k_summary.csv", det_summary)
    _write_csv(quick_dir / "appf1_internal_model_scale_context.csv", build_appf1_internal_model_scale_context(quick_dir))
    _write_csv(quick_dir / "fig13_accuracy_heatmap_aggregated.csv", build_fig13_heatmap_aggregated(quick_dir))
    _write_csv(quick_dir / "fig19_ablation_contract_summary.csv", build_fig19_contract_ablation())
    contextual_compare = build_contextual_platform_compare(quick_dir.name)
    _write_csv(quick_dir / "hpat_cpu_gpu_compare.csv", contextual_compare)
    _write_csv(quick_dir / "accelerator_compare_summary.csv", contextual_compare.copy())


if __name__ == "__main__":
    main()
