#!/usr/bin/env python3
"""Rebuild the MLX-final quick-report freeze candidate from fresh MLX artifacts."""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import subprocess
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "experiments" / "results" / "runs"
ACCURACY = ROOT / "experiments" / "results" / "accuracy"
REPORT_DATA = ROOT / "experiments" / "results" / "report_data"
QUICK_REPORTS = ROOT / "experiments" / "results" / "quick_reports"

DEFAULT_PROGRAM_TAG = "20260331_det_sparse_mlx_final"
DEFAULT_FREEZE_TAG = "20260402_det_sparse_mlx_final_cpu_context_freeze"
DEFAULT_SOURCE_QUICK_DIR = QUICK_REPORTS / "20260331_det_sparse_reentry_freeze"
DEFAULT_CLAIM_ACCURACY_CSV = ACCURACY / "accuracy_config_conditioned_20260331_det_sparse_mlx_final_mlx_seeds012.csv"
DEFAULT_SUPPORT_ACCURACY_CSV = ACCURACY / "accuracy_support_det_sparse_sweeps_20260331_det_sparse_mlx_final_mlx.csv"
DEFAULT_QUICK_DIR = QUICK_REPORTS / DEFAULT_FREEZE_TAG
DEFAULT_CPU_DEVICE_METRICS = REPORT_DATA / "fuller_cpu_device_metrics_20260402_det_sparse_mlx_final_cpu_context.csv"
DEFAULT_GPU_DEVICE_METRICS = REPORT_DATA / "fuller_gpu_device_metrics_20260402_det_sparse_mlx_final_cpu_context.csv"
DEFAULT_FULLER_MODEL_SUMMARY = REPORT_DATA / "fuller_reentry_model_summary_20260402_det_sparse_mlx_final_cpu_context.csv"
DEFAULT_FULLER_ABLATION_SUMMARY = REPORT_DATA / "fuller_ablation_summary_20260402_det_sparse_mlx_final_cpu_context.csv"
DEFAULT_SCALING_SUMMARY = REPORT_DATA / "fuller_scaling_summary_20260402_det_sparse_mlx_final_cpu_context.csv"
DEFAULT_NOISE_DENSE = REPORT_DATA / "fuller_noise_accuracy_summary_s_dense_20260402_det_sparse_mlx_final_cpu_context.csv"
DEFAULT_NOISE_XS = REPORT_DATA / "fuller_noise_accuracy_summary_xs_sparse_20260402_det_sparse_mlx_final_cpu_context.csv"
DEFAULT_NOISE_XXS = REPORT_DATA / "fuller_noise_accuracy_summary_xxs_sparse_20260402_det_sparse_mlx_final_cpu_context.csv"
DEFAULT_PYTHON_BIN = ROOT / ".venv311-mps" / "bin" / "python"
DEFAULT_REPORT_NOTE = "docs/reports/20260402_det_sparse_mlx_final_cpu_context_freeze_promotion_note.md"

DET_K_VALUES = [4, 8, 16, 24, 32, 48, 64, 80, 96, 112, 129]
SPARSE_TAU_VALUES = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
FANOUT_VALUES = [2, 4, 6, 8, 12, 16, 24, 32]
PHY_N_VALUES = [4, 8, 12, 16, 20, 24, 32, 48, 64]


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except Exception:
        return str(path.resolve())


def _run(cmd: list[str]) -> None:
    print("[mlx-final-rebuild]", " ".join(str(part) for part in cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=str(ROOT))


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _write_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[mlx-final-rebuild] wrote {_rel(path)}", flush=True)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[mlx-final-rebuild] wrote {_rel(path)}", flush=True)


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _clamp_unit_interval(value: float) -> float:
    return max(0.0, min(1.0, value))


def _resolve_sparse_compat_active_fraction(sparse_cfg: dict[str, Any]) -> float | None:
    use_tau_for_gating = sparse_cfg.get("use_tau_for_gating")
    tau = sparse_cfg.get("tau_global")
    active_fraction = sparse_cfg.get("active_fraction")
    if use_tau_for_gating is False and active_fraction is not None:
        try:
            return _clamp_unit_interval(float(active_fraction))
        except (TypeError, ValueError):
            return None
    if tau is not None:
        try:
            tau_value = float(tau)
            min_active_fraction = _clamp_unit_interval(float(sparse_cfg.get("min_active_fraction") or 0.0))
            if tau_value <= 0.0:
                return 1.0
            return _clamp_unit_interval(max(min_active_fraction, 1.0 - tau_value))
        except (TypeError, ValueError):
            pass
    if active_fraction is not None:
        try:
            return _clamp_unit_interval(float(active_fraction))
        except (TypeError, ValueError):
            return None
    return None


def _format_sparse_operating_point_label(model: str, compat_active_fraction: float | None) -> str:
    if compat_active_fraction is None or not math.isfinite(float(compat_active_fraction)):
        return f"sparse_active_fraction_unknown_{model}_v1"
    compat_pct = int(round(_clamp_unit_interval(float(compat_active_fraction)) * 100.0))
    return f"sparse_active_fraction_{compat_pct:03d}_{model}_v1"


def _artifact_future_output_path(data_contract_csv: Path, artifact_id: str) -> Path | None:
    df = _read_csv(data_contract_csv)
    matched = df[df["artifact_id"].astype(str) == artifact_id]
    if matched.empty:
        return None
    rel = str(matched.iloc[0]["future_output_path"]).strip()
    if not rel:
        return None
    return ROOT / rel


def _resolve_report_input(
    *,
    provided_path: Path | None,
    default_path: Path | None,
    data_contract_csv: Path | None,
    artifact_ids: list[str],
) -> Path | None:
    if provided_path is not None:
        return provided_path
    if data_contract_csv is not None:
        for artifact_id in artifact_ids:
            resolved = _artifact_future_output_path(data_contract_csv, artifact_id)
            if resolved is not None:
                return resolved
    return default_path


def _normalize_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin({"1", "true", "yes"})


def _to_num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _first_finite_value(*series_list: pd.Series) -> float:
    for series in series_list:
        values = _to_num(series).dropna()
        if not values.empty:
            return float(values.mean())
    return math.nan


def _strip_acc_seed(run_id: str) -> str:
    return re.sub(r"_acc_s\d+$", "", str(run_id).strip())


def _string_replacements(args: argparse.Namespace) -> dict[str, str]:
    return {
        "20260331_det_sparse_reentry_freeze": args.freeze_tag,
        "20260328_mps_full_eval_freeze": args.freeze_tag,
        "20260328_mps_full_eval": args.candidate_tag,
    }


def _sanitize_object_columns(df: pd.DataFrame, replacements: dict[str, str]) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if out[col].dtype != object:
            continue
        series = out[col].astype(str)
        for src, dst in replacements.items():
            series = series.str.replace(src, dst, regex=False)
        out[col] = series
    return out


def _copy_csv_with_replacements(src: Path, dst: Path, replacements: dict[str, str]) -> None:
    df = _sanitize_object_columns(_read_csv(src), replacements)
    _write_csv(dst, df)


def _copy_static_support_files(args: argparse.Namespace) -> None:
    replacements = _string_replacements(args)
    task_df = _sanitize_object_columns(
        _read_csv(args.source_quick_dir / "task_generalization_summary.csv"),
        replacements,
    )
    if "run_tag" in task_df.columns:
        task_df["run_tag"] = args.freeze_tag
    task_df = task_df.drop_duplicates().reset_index(drop=True)
    _write_csv(args.quick_dir / "task_generalization_summary.csv", task_df)
    _copy_csv_with_replacements(
        args.source_quick_dir / "fig_a_related_work_radar_scores.csv",
        args.quick_dir / "fig_a_related_work_radar_scores.csv",
        replacements,
    )


def _core_runs(program_tag: str) -> dict[str, str]:
    return {
        "E0": f"{program_tag}_core_e0",
        "E1": f"{program_tag}_core_e1",
        "E2": f"{program_tag}_core_e2",
        "E3": f"{program_tag}_core_e3",
        "E4": f"{program_tag}_core_e4",
        "E5": f"{program_tag}_core_e5",
        "E6": f"{program_tag}_core_e6",
    }


def _headline_runs_json(program_tag: str) -> str:
    core = _core_runs(program_tag)
    payload = {key: core[key] for key in ["E0", "E2", "E3", "E4", "E6"]}
    return json.dumps(payload, ensure_ascii=False)


def _scaling_runs(program_tag: str) -> list[str]:
    return [
        f"{program_tag}_scan_e0_batch2",
        f"{program_tag}_scan_e0_batch4",
        f"{program_tag}_scan_e0_seq128",
        f"{program_tag}_scan_e0_seq256",
    ]


def _sparse_runs(program_tag: str) -> list[str]:
    return [f"{program_tag}_scan_e4_t{int(round(tau * 100)):02d}" for tau in SPARSE_TAU_VALUES]


def _fanout_runs(program_tag: str) -> list[str]:
    return [f"{program_tag}_scan_e1_f{value}" for value in FANOUT_VALUES]


def _phy_runs(program_tag: str) -> list[str]:
    return [f"{program_tag}_scan_e5_n{value}" for value in PHY_N_VALUES]


def _det_runs(program_tag: str) -> list[str]:
    return [f"{program_tag}_scan_e3_k{value}" for value in DET_K_VALUES]


def _candidate_paths(args: argparse.Namespace) -> dict[str, Path]:
    base = args.quick_dir / "candidate_data"
    tag = args.candidate_tag
    return {
        "per_seed": base / f"headline_statistics_per_seed_{tag}.csv",
        "per_model": base / f"headline_statistics_per_model_{tag}.csv",
        "summary": base / f"headline_statistics_summary_{tag}.csv",
        "pair": base / f"headline_statistics_pair_ci_{tag}.csv",
        "report": base / f"headline_statistics_report_{tag}.md",
        "appendix": base / f"headline_statistics_appendix_table_{tag}.tex",
        "config_per_model": base / f"config_conditioned_accuracy_per_model_{tag}.csv",
        "config_summary": base / f"config_conditioned_accuracy_summary_{tag}.csv",
        "config_report": base / f"config_conditioned_accuracy_report_{tag}.md",
    }


def _main_claim_paths(args: argparse.Namespace) -> dict[str, Path]:
    tag = args.candidate_tag
    return {
        "csv": args.quick_dir / "main_text_data" / f"main_claim_unified_summary_{tag}.csv",
        "md": args.quick_dir / "main_text_data" / f"main_claim_unified_summary_{tag}.md",
        "table_csv": args.quick_dir / "main_text_tables" / f"Table1_MainResults_E0_E6_{tag}.csv",
        "table_tex": args.quick_dir / "main_text_tables" / f"Table1_MainResults_E0_E6_{tag}.tex",
    }


def _ensure_weights_npz_provenance(path: Path) -> None:
    df = _read_csv(path)
    if df.empty:
        return
    needs_column = "weights_npz_source" not in df.columns
    weights_npz = df.get("weights_npz", pd.Series("", index=df.index)).fillna("").astype(str).str.strip()
    if needs_column:
        insert_at = df.columns.get_loc("weights_npz") + 1 if "weights_npz" in df.columns else len(df.columns)
        df.insert(insert_at, "weights_npz_source", "")
    source = df["weights_npz_source"].fillna("").astype(str).str.strip()
    fill_mask = source.eq("") & weights_npz.ne("")
    if fill_mask.any():
        df.loc[fill_mask, "weights_npz_source"] = "explicit_weights_npz_arg"
    if needs_column or fill_mask.any():
        _write_csv(path, df)


def _build_candidate_assets(args: argparse.Namespace) -> None:
    candidate = _candidate_paths(args)
    main_claim = _main_claim_paths(args)
    core_runs = _core_runs(args.program_tag)

    headline_cmd = [
        str(args.python_bin),
        "experiments/tools/build_headline_statistics_report.py",
        "--runs_dir",
        _rel(RUNS),
        "--accuracy_csv",
        _rel(args.claim_accuracy_csv),
        "--out_data_dir",
        _rel(args.quick_dir / "candidate_data"),
        "--tag",
        args.candidate_tag,
        "--experiment_run_ids_json",
        _headline_runs_json(args.program_tag),
    ]
    _run(headline_cmd)

    config_cmd = [
        str(args.python_bin),
        "experiments/tools/build_config_conditioned_accuracy_appendix.py",
        "--full_csv",
        _rel(args.claim_accuracy_csv),
        "--out_data_dir",
        _rel(args.quick_dir / "candidate_data"),
        "--out_fig_dir",
        _rel(args.quick_dir / "candidate_figures" / "data"),
        "--tag",
        args.candidate_tag,
    ]
    _run(config_cmd)

    main_claim_cmd = [
        str(args.python_bin),
        "experiments/tools/build_main_claim_unified_summary.py",
        "--headline_summary_csv",
        _rel(candidate["summary"]),
        "--headline_pair_ci_csv",
        _rel(candidate["pair"]),
        "--headline_per_seed_csv",
        _rel(candidate["per_seed"]),
        "--headline_per_model_csv",
        _rel(candidate["per_model"]),
        "--out_dir",
        _rel(args.quick_dir / "main_text_data"),
        "--tag",
        args.candidate_tag,
    ]
    _run(main_claim_cmd)

    table_cmd = [
        str(args.python_bin),
        "experiments/tools/build_main_results_table.py",
        "--main_claim_csv",
        _rel(main_claim["csv"]),
        "--out_csv",
        _rel(main_claim["table_csv"]),
        "--out_tex",
        _rel(main_claim["table_tex"]),
    ]
    _run(table_cmd)

    shutil.copy2(candidate["pair"], args.quick_dir / "headline_claim_pair_ci.csv")
    shutil.copy2(candidate["summary"], args.quick_dir / "headline_claim_run_ci.csv")
    print(f"[mlx-final-rebuild] wrote {_rel(args.quick_dir / 'headline_claim_pair_ci.csv')}", flush=True)
    print(f"[mlx-final-rebuild] wrote {_rel(args.quick_dir / 'headline_claim_run_ci.csv')}", flush=True)

    # Explicitly touch the main claim outputs so callers can audit them directly.
    for path in [
        candidate["summary"],
        candidate["pair"],
        candidate["config_summary"],
        main_claim["csv"],
        main_claim["table_csv"],
    ]:
        if not path.is_file():
            raise FileNotFoundError(path)


def _group_master_by_model(run_id: str) -> pd.DataFrame:
    df = _read_csv(RUNS / run_id / "master_metrics.csv")
    if df.empty:
        raise RuntimeError(f"Empty master_metrics for {run_id}")
    work = df.copy()
    for col in work.columns:
        if col == "model":
            continue
        work[col] = _to_num(work[col])
    numeric_cols = [col for col in work.columns if col != "model" and work[col].notna().any()]
    grouped = work.groupby("model", as_index=False)[numeric_cols].mean()
    grouped["model"] = grouped["model"].astype(str)
    return grouped


def _row_for_model(run_id: str, model: str) -> pd.Series:
    df = _read_csv(RUNS / run_id / "master_metrics.csv")
    sub = df[df["model"].astype(str) == model].copy()
    if sub.empty:
        raise RuntimeError(f"Missing model={model} in {run_id}")
    return sub.iloc[0]


def _build_heatmap_points(args: argparse.Namespace) -> None:
    frames = []
    for path in [args.noise_dense_csv, args.noise_xs_csv, args.noise_xxs_csv]:
        df = _read_csv(path)
        if df.empty:
            continue
        df["gaussian_noise_std"] = _to_num(df["gaussian_noise_std"])
        df["crosstalk_alpha"] = _to_num(df["crosstalk_alpha"])
        df["acc_top1"] = _to_num(df["acc_top1"])
        df["acc_drop_pp"] = _to_num(df["acc_drop_pp"])
        df = df.dropna(subset=["model", "gaussian_noise_std", "crosstalk_alpha", "acc_top1", "acc_drop_pp"])
        if df.empty:
            continue
        out = pd.DataFrame(
            {
                "model": df["model"].astype(str),
                "quant_bits": 8,
                "sigma_lsb": df["gaussian_noise_std"],
                "crosstalk_alpha": df["crosstalk_alpha"],
                "acc_drop_pp": df["acc_drop_pp"],
                "top1": df["acc_top1"],
                "top1_ref": df["acc_top1"] + df["acc_drop_pp"],
            }
        )
        frames.append(out)
    if not frames:
        raise RuntimeError("No noise summary rows were available to build Fig13 points.")
    merged = pd.concat(frames, ignore_index=True)
    merged = merged.drop_duplicates().reset_index(drop=True)
    merged = merged.sort_values(["model", "crosstalk_alpha", "sigma_lsb"]).reset_index(drop=True)
    _write_csv(args.quick_dir / "fig_h_accuracy_heatmap_points.csv", merged)


def _claim_accuracy_rows(path: Path) -> pd.DataFrame:
    df = _read_csv(path)
    df["baseline_flag"] = _normalize_bool(df["baseline"])
    df["seed"] = _to_num(df["seed"]).astype("Int64")
    for col in [
        "top1",
        "top1_delta",
        "det_k_global",
        "det_prefix_error_mean",
        "det_prefix_error_p95",
        "sparse_tau_global",
        "sparse_active_fraction",
        "sparse_measured_activity_fraction",
    ]:
        if col in df.columns:
            df[col] = _to_num(df[col])
    return df


def _support_nonbaseline(args: argparse.Namespace) -> pd.DataFrame:
    df = _claim_accuracy_rows(args.support_accuracy_csv)
    support = df[~df["baseline_flag"]].copy()
    support["source_run_id"] = support["run_id"].astype(str).map(_strip_acc_seed)
    return support


def _e0_quant_map(claim_df: pd.DataFrame) -> pd.DataFrame:
    e0 = claim_df[(claim_df["experiment_id"].astype(str) == "E0") & (~claim_df["baseline_flag"])].copy()
    return e0[["model", "seed", "top1"]].rename(columns={"top1": "e0_quant_top1"})


def _claim_drop_stats(claim_df: pd.DataFrame, experiment_id: str) -> dict[str, float]:
    sub = claim_df[(claim_df["experiment_id"].astype(str) == experiment_id) & (~claim_df["baseline_flag"])].copy()
    if sub.empty:
        raise RuntimeError(f"Missing nonbaseline claim rows for {experiment_id}")
    base = claim_df[(claim_df["experiment_id"].astype(str) == experiment_id) & (claim_df["baseline_flag"])][["model", "seed", "top1"]].copy()
    base = base.rename(columns={"top1": "fp32_top1"})
    merged = sub.merge(base, on=["model", "seed"], how="left", validate="many_to_one")
    merged["acc_drop_pp_vs_fp32"] = merged["fp32_top1"] - merged["top1"]
    if experiment_id == "E0":
        merged["acc_drop_pp_vs_e0"] = 0.0
    else:
        merged = merged.merge(_e0_quant_map(claim_df), on=["model", "seed"], how="left", validate="many_to_one")
        merged["acc_drop_pp_vs_e0"] = merged["e0_quant_top1"] - merged["top1"]
    return {
        "measured_acc_drop_pp_vs_E0_mean": float(_to_num(merged["acc_drop_pp_vs_e0"]).mean()),
        "measured_acc_drop_pp_vs_E0_std": float(_to_num(merged["acc_drop_pp_vs_e0"]).std(ddof=1)) if len(merged) > 1 else 0.0,
        "measured_acc_drop_pp_vs_fp32_mean": float(_to_num(merged["acc_drop_pp_vs_fp32"]).mean()),
        "acc_drop_pp_mean": float(_to_num(merged["acc_drop_pp_vs_fp32"]).mean()),
        "acc_drop_pp_nonempty": int(len(merged)),
    }


def _expected_det_sources(args: argparse.Namespace) -> set[str]:
    return {f"{args.program_tag}_scan_e3_k{value}" for value in DET_K_VALUES}


def _expected_sparse_sources(args: argparse.Namespace) -> set[str]:
    return {f"{args.program_tag}_scan_e4_t{int(round(tau * 100)):02d}" for tau in SPARSE_TAU_VALUES}


def _validate_support_sources(actual: set[str], expected: set[str], label: str) -> None:
    missing = sorted(expected - actual)
    if missing:
        raise RuntimeError(f"Missing {label} support rows for: {', '.join(missing)}")


def _det_support_table(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    claim_df = _claim_accuracy_rows(args.claim_accuracy_csv)
    support = _support_nonbaseline(args)
    support = support[support["experiment_id"].astype(str) == "E3"].copy()
    support["det_k_global"] = _to_num(support["det_k_global"])
    support["top1"] = _to_num(support["top1"])
    support["top1_delta"] = _to_num(support["top1_delta"])
    support = support.dropna(subset=["source_run_id", "model", "seed", "det_k_global", "top1"])
    if support.empty:
        raise RuntimeError("No E3 support rows were found in the support accuracy CSV.")
    _validate_support_sources(set(support["source_run_id"].astype(str)), _expected_det_sources(args), "DET")
    support = support.merge(_e0_quant_map(claim_df), on=["model", "seed"], how="left", validate="many_to_one")
    support["acc_drop_pp_vs_e0"] = support["e0_quant_top1"] - support["top1"]
    support["acc_drop_pp_vs_fp32"] = -support["top1_delta"]

    rows: list[dict[str, Any]] = []
    for run_id in sorted(support["source_run_id"].astype(str).unique(), key=lambda item: int(item.rsplit("k", 1)[1])):
        grouped = _group_master_by_model(run_id)
        sub = support[support["source_run_id"].astype(str) == run_id].copy()
        k_value = float(sub["det_k_global"].dropna().iloc[0])
        rows.append(
            {
                "k_global": k_value,
                "run_id": run_id,
                "n_models": int(grouped["model"].nunique()),
                "latency_ms": float(_to_num(grouped["latency_ms"]).mean()),
                "energy_j": float(_to_num(grouped["energy_j"]).mean()),
                "tops_w": float(_to_num(grouped["tops_w"]).mean()),
                "speedup_vs_E0": float(_to_num(grouped["speedup_vs_E0"]).mean()),
                "det_net_gain_j": float(_to_num(grouped.get("det_net_gain_j", pd.Series(dtype=float))).mean()),
                "avg_effective_bsl": float(_to_num(grouped.get("avg_effective_bsl", pd.Series(dtype=float))).mean()),
                "bubble_cycles": float(_to_num(grouped.get("bubble_cycles", pd.Series(dtype=float))).mean()),
                "utilization_avg": float(_to_num(grouped.get("utilization_avg", pd.Series(dtype=float))).mean()),
                "measured_acc_drop_pp_vs_E0_mean": float(_to_num(sub["acc_drop_pp_vs_e0"]).mean()),
                "measured_acc_drop_pp_vs_E0_std": float(_to_num(sub["acc_drop_pp_vs_e0"]).std(ddof=1)) if len(sub) > 1 else math.nan,
                "measured_acc_drop_pp_vs_fp32_mean": float(_to_num(sub["acc_drop_pp_vs_fp32"]).mean()),
                "acc_drop_pp_mean": float(_to_num(sub["acc_drop_pp_vs_fp32"]).mean()),
                "acc_drop_pp_nonempty": int(len(sub)),
                "pass_det_net_gain_true": int(_to_num(grouped.get("pass_det_net_gain", pd.Series(dtype=float))).fillna(0).sum()),
                "prefix_error_mean": float(_to_num(sub["det_prefix_error_mean"]).mean()),
                "prefix_error_p95": float(_to_num(sub["det_prefix_error_p95"]).mean()),
                "acc_evidence_tier": "measured_mlx_support",
            }
        )
    quickscan = pd.DataFrame(rows).sort_values("k_global").reset_index(drop=True)
    override_by_k = {
        64.0: _claim_drop_stats(claim_df, "E3"),
        129.0: _claim_drop_stats(claim_df, "E0"),
    }
    for k_value, stats in override_by_k.items():
        mask = quickscan["k_global"].astype(float).eq(k_value)
        if not mask.any():
            continue
        for key, value in stats.items():
            quickscan.loc[mask, key] = value
    det_summary = pd.DataFrame(
        {
            "det_k_global": quickscan["k_global"],
            "avg_effective_bsl": quickscan["avg_effective_bsl"],
            "speedup_vs_E0": quickscan["speedup_vs_E0"],
            "prefix_error_mean": quickscan["prefix_error_mean"],
            "paired_model_mean_delta_vs_e0_quant_pp": -quickscan["measured_acc_drop_pp_vs_E0_mean"],
            "paired_model_std_delta_vs_e0_quant_pp": quickscan["measured_acc_drop_pp_vs_E0_std"],
            "paired_model_mean_delta_vs_fp32_pp": -quickscan["measured_acc_drop_pp_vs_fp32_mean"],
            "evidence_tier": "measured_mlx_support",
        }
    )
    prefix = pd.DataFrame(
        {
            "k": quickscan["k_global"],
            "prefix_error_mean": quickscan["prefix_error_mean"],
            "prefix_error_p95": quickscan["prefix_error_p95"],
            "acc_drop_pp_mean": quickscan["measured_acc_drop_pp_vs_fp32_mean"],
            "measured_acc_drop_pp_mean": quickscan["measured_acc_drop_pp_vs_fp32_mean"],
        }
    )
    return quickscan, det_summary, prefix


def _sparse_support_table(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    claim_df = _claim_accuracy_rows(args.claim_accuracy_csv)
    support = _support_nonbaseline(args)
    support = support[support["experiment_id"].astype(str) == "E4"].copy()
    support["sparse_tau_global"] = _to_num(support["sparse_tau_global"])
    support["top1"] = _to_num(support["top1"])
    support["top1_delta"] = _to_num(support["top1_delta"])
    support["sparse_active_fraction"] = _to_num(support.get("sparse_active_fraction", pd.Series(dtype=float)))
    support = support.dropna(subset=["source_run_id", "model", "seed", "sparse_tau_global", "top1"])
    if support.empty:
        raise RuntimeError("No E4 support rows were found in the support accuracy CSV.")
    _validate_support_sources(set(support["source_run_id"].astype(str)), _expected_sparse_sources(args), "SPARSE")
    support = support.merge(_e0_quant_map(claim_df), on=["model", "seed"], how="left", validate="many_to_one")
    support["acc_drop_pp_vs_e0"] = support["e0_quant_top1"] - support["top1"]
    support["acc_drop_pp_vs_fp32"] = -support["top1_delta"]

    rows: list[dict[str, Any]] = []
    for run_id in sorted(support["source_run_id"].astype(str).unique(), key=lambda item: float(item.rsplit("t", 1)[1]) if item.rsplit("t", 1)[1] else 0.0):
        grouped = _group_master_by_model(run_id)
        sub = support[support["source_run_id"].astype(str) == run_id].copy()
        tau = float(_to_num(sub["sparse_tau_global"]).dropna().iloc[0])
        duty = float(_to_num(grouped.get("duty_cycle_avg", pd.Series(dtype=float))).mean())
        measured_activity = _to_num(sub.get("sparse_measured_activity_fraction", pd.Series(dtype=float))).dropna()
        active_fraction = float(measured_activity.mean()) if not measured_activity.empty else duty
        rows.append(
            {
                "run_id": run_id,
                "tau": tau,
                "sparse_active_fraction": active_fraction,
                "duty_cycle_avg": duty,
                "energy_j": float(_to_num(grouped["energy_j"]).mean()),
                "latency_ms": float(_to_num(grouped["latency_ms"]).mean()),
                "speedup_vs_E0": float(_to_num(grouped["speedup_vs_E0"]).mean()),
                "measured_acc_drop_pp_vs_E0_mean": float(_to_num(sub["acc_drop_pp_vs_e0"]).mean()),
                "measured_acc_drop_pp_vs_E0_std": float(_to_num(sub["acc_drop_pp_vs_e0"]).std(ddof=1)) if len(sub) > 1 else math.nan,
                "measured_acc_drop_pp_vs_fp32_mean": float(_to_num(sub["acc_drop_pp_vs_fp32"]).mean()),
                "measured_acc_drop_pp_mean": float(_to_num(sub["acc_drop_pp_vs_fp32"]).mean()),
                "accuracy_evidence": "measured_mlx_support",
            }
        )
    quickscan = pd.DataFrame(rows).sort_values("tau").reset_index(drop=True)
    if quickscan.empty:
        raise RuntimeError("No sparse support rows were aggregated.")
    baseline = quickscan.loc[quickscan["tau"].abs() < 1e-12, "energy_j"]
    if baseline.empty:
        raise RuntimeError("Sparse support sweep is missing tau=0.0 energy baseline.")
    baseline_energy = float(baseline.iloc[0])
    quickscan["energy_saved_pct"] = (1.0 - (quickscan["energy_j"] / baseline_energy)) * 100.0
    quickscan["support_acc_drop_pp_vs_E0_mean"] = quickscan["measured_acc_drop_pp_vs_E0_mean"]
    quickscan["support_acc_drop_pp_vs_fp32_mean"] = quickscan["measured_acc_drop_pp_vs_fp32_mean"]
    quickscan["accuracy_note"] = "measured_sparse_support_sweep"

    e4_claim = claim_df[
        (claim_df["experiment_id"].astype(str) == "E4")
        & (~claim_df["baseline_flag"])
    ].copy()
    if not e4_claim.empty:
        anchor_duty_cycle = _first_finite_value(
            e4_claim.get("duty_cycle_avg", pd.Series(dtype=float)),
            e4_claim.get("sparse_measured_activity_fraction", pd.Series(dtype=float)),
            e4_claim.get("sparse_active_fraction", pd.Series(dtype=float)),
        )
        if not math.isfinite(anchor_duty_cycle):
            cfg = _read_yaml(RUNS / _core_runs(args.program_tag)["E4"] / "config_snapshot.yaml")
            resolved_anchor = _resolve_sparse_compat_active_fraction(cfg.get("sparse") or {})
            if resolved_anchor is not None:
                anchor_duty_cycle = resolved_anchor
        if not math.isfinite(anchor_duty_cycle):
            grouped = _group_master_by_model(_core_runs(args.program_tag)["E4"])
            anchor_duty_cycle = _first_finite_value(grouped.get("duty_cycle_avg", pd.Series(dtype=float)))
        if math.isfinite(anchor_duty_cycle):
            anchor_mask = quickscan["duty_cycle_avg"].astype(float).sub(anchor_duty_cycle).abs() < 1e-9
            if anchor_mask.any():
                claim_stats = _claim_drop_stats(claim_df, "E4")
                for key in [
                    "measured_acc_drop_pp_vs_E0_mean",
                    "measured_acc_drop_pp_vs_E0_std",
                    "measured_acc_drop_pp_vs_fp32_mean",
                    "acc_drop_pp_mean",
                ]:
                    quickscan.loc[anchor_mask, key] = claim_stats[key]
                quickscan.loc[anchor_mask, "accuracy_evidence"] = "measured_full_eval_anchor"
                quickscan.loc[anchor_mask, "accuracy_note"] = (
                    "repaired_e4_full_eval_anchor_on_governed_sparse_operating_point"
                )
    pareto = quickscan[[
        "run_id",
        "tau",
        "sparse_active_fraction",
        "duty_cycle_avg",
        "energy_j",
        "measured_acc_drop_pp_vs_E0_mean",
        "measured_acc_drop_pp_vs_fp32_mean",
        "support_acc_drop_pp_vs_E0_mean",
        "support_acc_drop_pp_vs_fp32_mean",
        "accuracy_evidence",
        "accuracy_note",
        "energy_saved_pct",
    ]].copy()
    pareto["acc_drop_pp_mean"] = pareto["measured_acc_drop_pp_vs_E0_mean"]
    ordered = [
        "run_id",
        "tau",
        "sparse_active_fraction",
        "duty_cycle_avg",
        "energy_j",
        "measured_acc_drop_pp_vs_E0_mean",
        "measured_acc_drop_pp_vs_fp32_mean",
        "support_acc_drop_pp_vs_E0_mean",
        "support_acc_drop_pp_vs_fp32_mean",
        "acc_drop_pp_mean",
        "accuracy_evidence",
        "accuracy_note",
        "energy_saved_pct",
    ]
    pareto = pareto[ordered]
    return quickscan, pareto


def _fanout_support_table(args: argparse.Namespace) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for run_id in _fanout_runs(args.program_tag):
        grouped = _group_master_by_model(run_id)
        fanout = float(_to_num(grouped.get("fanout", pd.Series(dtype=float))).mean())
        rows.append(
            {
                "fanout_cfg": int(round(fanout)),
                "run_id": run_id,
                "n_models": int(grouped["model"].nunique()),
                "latency_ms": float(_to_num(grouped["latency_ms"]).mean()),
                "energy_j": float(_to_num(grouped["energy_j"]).mean()),
                "tops_w": float(_to_num(grouped["tops_w"]).mean()),
                "speedup_vs_E0": float(_to_num(grouped["speedup_vs_E0"]).mean()),
                "fanout": fanout,
                "serializers_saved": float(_to_num(grouped.get("serializers_saved", pd.Series(dtype=float))).mean()),
                "broadcast_driver_energy_j": float(_to_num(grouped.get("broadcast_driver_energy_j", pd.Series(dtype=float))).mean()),
                "net_energy_gain_j": float(_to_num(grouped.get("net_energy_gain_j", pd.Series(dtype=float))).mean()),
                "acc_drop_pp_nonempty": int(grouped["model"].nunique()),
            }
        )
    return pd.DataFrame(rows).sort_values("fanout").reset_index(drop=True)


def _phy_support_table(args: argparse.Namespace) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for run_id in _phy_runs(args.program_tag):
        grouped = _group_master_by_model(run_id)
        rows.append(
            {
                "N_wdm": float(_to_num(grouped.get("N_wdm", pd.Series(dtype=float))).mean()),
                "run_id": run_id,
                "n_models": int(grouped["model"].nunique()),
                "P_laser_dbm": float(_to_num(grouped.get("P_laser_dbm", pd.Series(dtype=float))).mean()),
                "P_laser_mw": float(_to_num(grouped.get("P_laser_mw", pd.Series(dtype=float))).mean()),
                "PP_crosstalk_db": float(_to_num(grouped.get("PP_crosstalk_db", pd.Series(dtype=float))).mean()),
                "Loss_path_db": float(_to_num(grouped.get("Loss_path_db", pd.Series(dtype=float))).mean()),
                "energy_j": float(_to_num(grouped["energy_j"]).mean()),
                "tops_w": float(_to_num(grouped["tops_w"]).mean()),
                "latency_ms": float(_to_num(grouped["latency_ms"]).mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("N_wdm").reset_index(drop=True)


def _wdm_quant_table(args: argparse.Namespace) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for run_id in _phy_runs(args.program_tag):
        cfg = _read_yaml(RUNS / run_id / "config_snapshot.yaml")
        quant_bits = float((cfg.get("sc_det") or {}).get("quant_bits") or 8)
        mrr_tile_k = 16.0
        df = _read_csv(RUNS / run_id / "master_metrics.csv")
        if df.empty:
            continue
        for col in ["latency_ms", "energy_j", "avg_power_w", "tops_w", "throughput_images_s", "throughput_tokens_s", "N_wdm"]:
            if col in df.columns:
                df[col] = _to_num(df[col])
        for row in df.itertuples(index=False):
            rows.append(
                {
                    "run_id": run_id,
                    "run_tag": args.freeze_tag,
                    "experiment_id": "E5",
                    "model": str(row.model),
                    "model_family": "mobilevit",
                    "task_id": "imagenet_cls",
                    "workload_id": str(getattr(row, "workload_id", "W0_mobilevit_imagenet")),
                    "latency_ms": float(getattr(row, "latency_ms", math.nan)),
                    "energy_j": float(getattr(row, "energy_j", math.nan)),
                    "avg_power_w": float(getattr(row, "avg_power_w", math.nan)),
                    "tops_w": float(getattr(row, "tops_w", math.nan)),
                    "throughput_images_s": float(getattr(row, "throughput_images_s", math.nan)),
                    "throughput_tokens_s": float(getattr(row, "throughput_tokens_s", math.nan)),
                    "primary_metric_name": "Top1",
                    "primary_metric_value": math.nan,
                    "primary_metric_drop": math.nan,
                    "acc_top1": math.nan,
                    "acc_drop_pp": math.nan,
                    "area_mm2": 0.0,
                    "peak_tops": 0.0,
                    "platform_class": "HPAT",
                    "process_node_nm": "unknown",
                    "source_type": "simulated_run",
                    "batch_size": 1.0,
                    "sequence_length": 197.0,
                    "input_size": float(getattr(row, "input_size", math.nan)),
                    "N_wdm": float(getattr(row, "N_wdm", math.nan)),
                    "mrr_tile_k": mrr_tile_k,
                    "quant_bits": quant_bits,
                    "fanout": float(getattr(row, "fanout", math.nan)),
                    "broadcast_mode": "one_to_one",
                }
            )
    return pd.DataFrame(rows).sort_values(["mrr_tile_k", "quant_bits", "N_wdm", "model"]).reset_index(drop=True)


def _noise_surface_table(args: argparse.Namespace) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in [args.noise_dense_csv, args.noise_xs_csv, args.noise_xxs_csv]:
        df = _read_csv(path)
        if df.empty:
            continue
        for row in df.itertuples(index=False):
            model = str(row.model)
            profile = str(row.profile)
            rows.append(
                {
                    "run_id": f"{args.program_tag}_noise_{model}_{profile}",
                    "run_tag": args.freeze_tag,
                    "experiment_id": "E6" if model == "mobilevit_s" else "E4",
                    "model": model,
                    "model_family": "mobilevit",
                    "task_id": "imagenet_cls",
                    "workload_id": "W0_mobilevit_imagenet_eval",
                    "latency_ms": float(row.latency_ms),
                    "energy_j": float(row.energy_j),
                    "avg_power_w": math.nan,
                    "tops_w": math.nan,
                    "throughput_images_s": math.nan,
                    "throughput_tokens_s": math.nan,
                    "primary_metric_name": "Top1",
                    "primary_metric_value": float(row.acc_top1),
                    "primary_metric_drop": float(row.acc_drop_pp),
                    "acc_top1": float(row.acc_top1),
                    "acc_drop_pp": float(row.acc_drop_pp),
                    "area_mm2": 0.0,
                    "peak_tops": 0.0,
                    "platform_class": "HPAT",
                    "process_node_nm": "unknown",
                    "source_type": "simulated_accuracy",
                    "batch_size": 1.0,
                    "sequence_length": 197.0,
                    "input_size": 256.0 if model == "mobilevit_s" else (224.0 if model == "mobilevit_xs" else 192.0),
                    "N_wdm": 16.0,
                    "mrr_tile_k": 16.0,
                    "quant_bits": 8.0,
                    "fanout": 4.0,
                    "broadcast_mode": "one_to_one",
                    "noise_sigma_lsb": float(row.gaussian_noise_std),
                    "crosstalk_alpha": float(row.crosstalk_alpha),
                    "drift_lsb": 0.0,
                    "noise_correlation": 0.0,
                    "burst_error_prob": 0.0,
                    "burst_error_scale_lsb": 0.0,
                    "burst_span": 1.0,
                    "notes": profile,
                }
            )
    return pd.DataFrame(rows).sort_values(["model", "crosstalk_alpha", "noise_sigma_lsb"]).reset_index(drop=True)


def _build_support_surfaces(args: argparse.Namespace) -> None:
    det_quickscan, det_summary, prefix = _det_support_table(args)
    _write_csv(args.quick_dir / "quickscan_e3_k_sweep.csv", det_quickscan)
    _write_csv(args.quick_dir / "fig8_det_k_summary.csv", det_summary)
    _write_csv(args.quick_dir / "fig_d_prefix_error_vs_k.csv", prefix)

    sparse_quickscan, sparse_pareto = _sparse_support_table(args)
    _write_csv(args.quick_dir / "quickscan_e4_active_fraction_sweep.csv", sparse_quickscan)
    _write_csv(args.quick_dir / "fig_j_sparse_tau_pareto.csv", sparse_pareto)

    _write_csv(args.quick_dir / "quickscan_e1_fanout_sweep.csv", _fanout_support_table(args))
    _write_csv(args.quick_dir / "quickscan_e5_phy_n_sweep.csv", _phy_support_table(args))
    _write_csv(args.quick_dir / "quickscan_mrr_wdm_quant.csv", _wdm_quant_table(args))
    _write_csv(args.quick_dir / "noise_robustness_surface.csv", _noise_surface_table(args))


def _claim_rows_for_model(claim_df: pd.DataFrame, experiment_id: str, model: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    sub = claim_df[(claim_df["experiment_id"].astype(str) == experiment_id) & (claim_df["model"].astype(str) == model)].copy()
    baseline = sub[sub["baseline_flag"]].copy()
    nonbaseline = sub[~sub["baseline_flag"]].copy()
    if baseline.empty or nonbaseline.empty:
        raise RuntimeError(f"Missing baseline/nonbaseline claim rows for {experiment_id} {model}")
    return baseline, nonbaseline


def _model_summary_row(args: argparse.Namespace, experiment_id: str, model: str) -> dict[str, float]:
    claim_df = _claim_accuracy_rows(args.claim_accuracy_csv)
    baseline, nonbaseline = _claim_rows_for_model(claim_df, experiment_id, model)
    return {
        "acc_ref_top1": float(_to_num(baseline["top1"]).mean()),
        "acc_top1": float(_to_num(nonbaseline["top1"]).mean()),
        "acc_drop_pp": float(_to_num(baseline["top1"]).mean() - _to_num(nonbaseline["top1"]).mean()),
    }


def _write_report_data_summaries(args: argparse.Namespace) -> None:
    claim_df = _claim_accuracy_rows(args.claim_accuracy_csv)
    core = _core_runs(args.program_tag)
    e0_row = _row_for_model(core["E0"], args.model_anchor)
    e3_row = _row_for_model(core["E3"], args.model_anchor)
    e4_row = _row_for_model(core["E4"], args.model_anchor)

    e3_acc = _model_summary_row(args, "E3", args.model_anchor)
    e4_acc = _model_summary_row(args, "E4", args.model_anchor)

    e3_cfg = _read_yaml(RUNS / core["E3"] / "config_snapshot.yaml")
    e4_cfg = _read_yaml(RUNS / core["E4"] / "config_snapshot.yaml")
    e3_budget = float(((e3_cfg.get("accuracy") or {}).get("delta_pp_budget") or 0.0))
    e4_budget = float(((e4_cfg.get("accuracy") or {}).get("delta_pp_budget") or 0.0))
    e3_k = float((((e3_cfg.get("sc_det") or {}).get("early_stop") or {}).get("k_global") or 0.0))
    e4_tau = float(((e4_cfg.get("sparse") or {}).get("tau_global") or 0.0))
    e4_duty_cycle = float(e4_row.get("duty_cycle_avg", math.nan))
    e4_compat_active = _resolve_sparse_compat_active_fraction(e4_cfg.get("sparse") or {})
    if e4_compat_active is None and math.isfinite(e4_duty_cycle):
        e4_compat_active = e4_duty_cycle
    if e4_compat_active is None:
        e4_compat_active = 0.0

    det_summary = pd.DataFrame(
        [
            {
                "candidate_label": f"det_mlx_final_k{int(round(e3_k))}_{args.model_anchor}",
                "run_id": core["E3"],
                "experiment_id": "DET_REENTRY",
                "model": args.model_anchor,
                "det_k_global": e3_k,
                "latency_ms": float(e3_row["latency_ms"]),
                "speedup_vs_e0": float(e3_row["speedup_vs_E0"]),
                "energy_j": float(e3_row["energy_j"]),
                "energy_reduction_pct_vs_e0": 100.0 * (1.0 - (float(e3_row["energy_j"]) / float(e0_row["energy_j"]))),
                "det_net_gain_j": float(e3_row.get("det_net_gain_j", 0.0) or 0.0),
                **e3_acc,
                "pass_delta": bool(e3_acc["acc_drop_pp"] <= e3_budget),
                "pass_det_net_gain": bool(float(e3_row.get("det_net_gain_j", 0.0) or 0.0) > 0.0),
                "accuracy_evidence": "measured_mlx_full_eval",
                "performance_run_path": str((RUNS / core["E3"] / "master_metrics.csv").resolve()),
                "phase1_summary_csv": str((RUNS / core["E3"] / "phase1_summary.csv").resolve()),
                "accuracy_source_csv": str(args.claim_accuracy_csv.resolve()),
                "accuracy_baseline_row_id": f"{core['E3']}_acc_s0",
                "accuracy_target_row_id": f"{core['E3']}_acc_s0",
                "accuracy_context_run_id": core["E3"],
                "legacy_evidence_role": "mlx_final_active",
                "definition_note_md": str(args.report_note),
            }
        ]
    )
    _write_csv(REPORT_DATA / f"det_reentry_summary_{args.program_tag}.csv", det_summary)

    sparse_summary = pd.DataFrame(
        [
            {
                "operating_point_label": _format_sparse_operating_point_label(
                    args.model_anchor,
                    e4_compat_active,
                ),
                "run_id": core["E4"],
                "experiment_id": "SPARSE_REENTRY_V1",
                "model": args.model_anchor,
                "control_mode": "tau_threshold_primary",
                "sparse_tau_global": e4_tau,
                "sparse_active_fraction": e4_compat_active,
                "duty_cycle_avg": e4_duty_cycle,
                "latency_ms": float(e4_row["latency_ms"]),
                "speedup_vs_e0": float(e4_row["speedup_vs_E0"]),
                "energy_j": float(e4_row["energy_j"]),
                "energy_reduction_pct_vs_e0": 100.0 * (1.0 - (float(e4_row["energy_j"]) / float(e0_row["energy_j"]))),
                **e4_acc,
                "pass_delta": bool(e4_acc["acc_drop_pp"] <= e4_budget),
                "accuracy_evidence": "measured_mlx_full_eval",
                "projection_only": False,
                "promotion_safe": bool(e4_acc["acc_drop_pp"] <= e4_budget),
                "performance_run_path": str((RUNS / core["E4"] / "master_metrics.csv").resolve()),
                "phase1_summary_csv": str((RUNS / core["E4"] / "phase1_summary.csv").resolve()),
                "per_layer_phy_csv": str((RUNS / core["E4"] / "per_layer_phy.csv").resolve()),
                "accuracy_source_csv": str(args.claim_accuracy_csv.resolve()),
                "accuracy_baseline_row_id": f"{core['E4']}_acc_s0",
                "accuracy_target_row_id": f"{core['E4']}_acc_s0",
                "definition_note_md": str(args.report_note),
            }
        ]
    )
    _write_csv(REPORT_DATA / f"sparse_reentry_operating_point_{args.program_tag}.csv", sparse_summary)


def _write_compliance_report(args: argparse.Namespace) -> None:
    payload = {
        "generated_at": str(date.today()),
        "run_tag": args.freeze_tag,
        "out_dir": _rel(args.quick_dir),
        "program_tag": args.program_tag,
        "candidate_tag": args.candidate_tag,
        "claim_accuracy_csv": _rel(args.claim_accuracy_csv),
        "support_accuracy_csv": _rel(args.support_accuracy_csv),
        "cpu_device_metrics_csv": _rel(args.cpu_device_metrics_csv) if args.cpu_device_metrics_csv else "",
        "gpu_device_metrics_csv": _rel(args.gpu_device_metrics_csv),
        "fuller_model_summary_csv": _rel(args.fuller_model_summary_csv),
        "fuller_ablation_summary_csv": _rel(args.fuller_ablation_summary_csv),
        "static_support_reemitted_locally": True,
        "core_runs": _core_runs(args.program_tag),
        "scaling_runs": _scaling_runs(args.program_tag),
        "sparse_runs": _sparse_runs(args.program_tag),
        "det_runs": _det_runs(args.program_tag),
        # A successful governed rebuild produces a branch-ready freeze-local pack.
        "ready_for_branch": True,
        "build_allowed": True,
    }
    _write_json(args.quick_dir / "compliance_report.json", payload)


def _run_strict_rebuild(args: argparse.Namespace) -> None:
    cmd = [
        str(args.python_bin),
        "experiments/tools/rebuild_strict_freeze_quick_reports.py",
        "--quick_dir",
        _rel(args.quick_dir),
        "--candidate_dir",
        _rel(args.quick_dir / "candidate_data"),
        "--candidate_tag",
        args.candidate_tag,
    ]
    if args.cpu_device_metrics_csv is not None:
        cmd.extend(["--cpu_device_metrics_csv", _rel(args.cpu_device_metrics_csv)])
    cmd.extend(
        [
            "--gpu_device_metrics_csv",
            _rel(args.gpu_device_metrics_csv),
            "--fuller_model_summary_csv",
            _rel(args.fuller_model_summary_csv),
            "--fuller_ablation_summary_csv",
            _rel(args.fuller_ablation_summary_csv),
            "--core_runs_json",
            json.dumps(_core_runs(args.program_tag), ensure_ascii=False),
            "--scaling_runs_json",
            json.dumps(_scaling_runs(args.program_tag), ensure_ascii=False),
            "--sparse_tau_runs_json",
            json.dumps(_sparse_runs(args.program_tag), ensure_ascii=False),
            "--python_bin",
            str(args.python_bin),
        ]
    )
    _run(cmd)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebuild the MLX-final quick-report freeze candidate.")
    parser.add_argument("--program_tag", default=DEFAULT_PROGRAM_TAG)
    parser.add_argument("--freeze_tag", default=DEFAULT_FREEZE_TAG)
    parser.add_argument("--candidate_tag", default=DEFAULT_FREEZE_TAG)
    parser.add_argument("--quick_dir", type=Path, default=DEFAULT_QUICK_DIR)
    parser.add_argument("--source_quick_dir", type=Path, default=DEFAULT_SOURCE_QUICK_DIR)
    parser.add_argument("--design_contract_yaml", type=Path, default=None)
    parser.add_argument("--data_contract_csv", type=Path, default=None)
    parser.add_argument("--claim_accuracy_csv", type=Path, default=DEFAULT_CLAIM_ACCURACY_CSV)
    parser.add_argument("--support_accuracy_csv", type=Path, default=DEFAULT_SUPPORT_ACCURACY_CSV)
    parser.add_argument("--cpu_device_metrics_csv", type=Path, default=None)
    parser.add_argument("--gpu_device_metrics_csv", type=Path, default=None)
    parser.add_argument("--fuller_model_summary_csv", type=Path, default=None)
    parser.add_argument("--fuller_ablation_summary_csv", type=Path, default=None)
    parser.add_argument("--scaling_summary_csv", type=Path, default=None)
    parser.add_argument("--noise_dense_csv", type=Path, default=None)
    parser.add_argument("--noise_xs_csv", type=Path, default=None)
    parser.add_argument("--noise_xxs_csv", type=Path, default=None)
    parser.add_argument("--python_bin", type=Path, default=DEFAULT_PYTHON_BIN)
    parser.add_argument("--model_anchor", default="mobilevit_s")
    parser.add_argument("--report_note", type=Path, default=Path(DEFAULT_REPORT_NOTE))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.quick_dir = args.quick_dir.resolve()
    args.source_quick_dir = args.source_quick_dir.resolve()
    args.design_contract_yaml = args.design_contract_yaml.resolve() if args.design_contract_yaml else None
    args.data_contract_csv = args.data_contract_csv.resolve() if args.data_contract_csv else None
    if args.data_contract_csv is None and args.design_contract_yaml is not None:
        contract = _read_yaml(args.design_contract_yaml)
        data_contract_rel = str(((contract.get("artifacts") or {}).get("data_contract_csv")) or "").strip()
        if data_contract_rel:
            args.data_contract_csv = (ROOT / data_contract_rel).resolve()
    args.claim_accuracy_csv = args.claim_accuracy_csv.resolve()
    args.support_accuracy_csv = args.support_accuracy_csv.resolve()
    args.cpu_device_metrics_csv = _resolve_report_input(
        provided_path=args.cpu_device_metrics_csv,
        default_path=DEFAULT_CPU_DEVICE_METRICS,
        data_contract_csv=args.data_contract_csv,
        artifact_ids=["cpu_real_device_metrics"],
    )
    args.gpu_device_metrics_csv = _resolve_report_input(
        provided_path=args.gpu_device_metrics_csv,
        default_path=DEFAULT_GPU_DEVICE_METRICS,
        data_contract_csv=args.data_contract_csv,
        artifact_ids=["gpu_real_device_metrics"],
    )
    args.fuller_model_summary_csv = _resolve_report_input(
        provided_path=args.fuller_model_summary_csv,
        default_path=DEFAULT_FULLER_MODEL_SUMMARY,
        data_contract_csv=args.data_contract_csv,
        artifact_ids=["fuller_slice_model_summary"],
    )
    args.fuller_ablation_summary_csv = _resolve_report_input(
        provided_path=args.fuller_ablation_summary_csv,
        default_path=DEFAULT_FULLER_ABLATION_SUMMARY,
        data_contract_csv=args.data_contract_csv,
        artifact_ids=["ablation_summary"],
    )
    args.scaling_summary_csv = _resolve_report_input(
        provided_path=args.scaling_summary_csv,
        default_path=DEFAULT_SCALING_SUMMARY,
        data_contract_csv=args.data_contract_csv,
        artifact_ids=["scaling_summary"],
    )
    args.noise_dense_csv = _resolve_report_input(
        provided_path=args.noise_dense_csv,
        default_path=DEFAULT_NOISE_DENSE,
        data_contract_csv=args.data_contract_csv,
        artifact_ids=["noise_accuracy_summary_s_dense"],
    )
    args.noise_xs_csv = _resolve_report_input(
        provided_path=args.noise_xs_csv,
        default_path=DEFAULT_NOISE_XS,
        data_contract_csv=args.data_contract_csv,
        artifact_ids=["noise_accuracy_summary_xs_dense", "noise_accuracy_summary_xs_sparse"],
    )
    args.noise_xxs_csv = _resolve_report_input(
        provided_path=args.noise_xxs_csv,
        default_path=DEFAULT_NOISE_XXS,
        data_contract_csv=args.data_contract_csv,
        artifact_ids=["noise_accuracy_summary_xxs_dense", "noise_accuracy_summary_xxs_sparse"],
    )
    args.cpu_device_metrics_csv = args.cpu_device_metrics_csv.resolve() if args.cpu_device_metrics_csv else None
    args.gpu_device_metrics_csv = args.gpu_device_metrics_csv.resolve() if args.gpu_device_metrics_csv else None
    args.fuller_model_summary_csv = args.fuller_model_summary_csv.resolve() if args.fuller_model_summary_csv else None
    args.fuller_ablation_summary_csv = args.fuller_ablation_summary_csv.resolve() if args.fuller_ablation_summary_csv else None
    args.scaling_summary_csv = args.scaling_summary_csv.resolve() if args.scaling_summary_csv else None
    args.noise_dense_csv = args.noise_dense_csv.resolve() if args.noise_dense_csv else None
    args.noise_xs_csv = args.noise_xs_csv.resolve() if args.noise_xs_csv else None
    args.noise_xxs_csv = args.noise_xxs_csv.resolve() if args.noise_xxs_csv else None
    args.report_note = args.report_note.resolve()
    # Keep the venv launcher path intact instead of resolving through its symlink
    # to the host interpreter, otherwise subprocesses lose the repository env.
    args.python_bin = args.python_bin.absolute()

    args.quick_dir.mkdir(parents=True, exist_ok=True)
    (args.quick_dir / "candidate_data").mkdir(parents=True, exist_ok=True)
    (args.quick_dir / "candidate_figures" / "data").mkdir(parents=True, exist_ok=True)
    (args.quick_dir / "main_text_data").mkdir(parents=True, exist_ok=True)
    (args.quick_dir / "main_text_tables").mkdir(parents=True, exist_ok=True)

    _ensure_weights_npz_provenance(args.claim_accuracy_csv)
    _ensure_weights_npz_provenance(args.support_accuracy_csv)
    _copy_static_support_files(args)
    _build_heatmap_points(args)
    _build_candidate_assets(args)
    # Seed strict rebuild with measured support tables, then overwrite again after
    # it finishes so the active freeze retains the MLX-measured surfaces.
    _build_support_surfaces(args)
    _run_strict_rebuild(args)
    _build_support_surfaces(args)
    _write_report_data_summaries(args)
    _write_compliance_report(args)

    print(_rel(args.quick_dir))


if __name__ == "__main__":
    main()
