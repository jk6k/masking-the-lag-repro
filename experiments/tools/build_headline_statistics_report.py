#!/usr/bin/env python3
"""Build seed-level headline statistics assets for the E0/E2/E3/E4/E6 configuration set."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from experiments.tools.path_policy import MAIN_PROJECT_REPORT_DATA_DIR
except ModuleNotFoundError:  # direct script execution
    from path_policy import MAIN_PROJECT_REPORT_DATA_DIR


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUNS_DIR = ROOT / "experiments" / "results" / "runs"
DEFAULT_ACCURACY_CSV = (
    ROOT / "experiments" / "results" / "accuracy" / "accuracy_config_conditioned_cuda_fulleval_seeds012_20260307.csv"
)
DEFAULT_OUT_DATA_DIR = MAIN_PROJECT_REPORT_DATA_DIR
DEFAULT_TAG = "20260310"
DEFAULT_BOOTSTRAP_SAMPLES = 10000
DEFAULT_BOOTSTRAP_SEED = 20260310

EXPERIMENT_RUN_IDS = {
    "E0": "20260228_opt_sync_core_e0",
    "E2": "20260228_opt_sync_core_e2",
    "E3": "20260228_opt_sync_core_e3",
    "E4": "20260228_opt_sync_core_e4",
    "E6": "20260228_opt_sync_core_e6",
}
EXPERIMENT_ORDER = ["E0", "E2", "E3", "E4", "E6"]
MODEL_ORDER = ["mobilevit_xxs", "mobilevit_xs", "mobilevit_s"]


def _is_config_conditioned_proxy(note: str) -> bool:
    note_norm = str(note or "").strip().lower()
    return note_norm in {"config_conditioned_sim", "mtl_sim"}


def _is_modeled_accuracy_evidence(value: str) -> bool:
    return str(value or "").strip().lower() in {"modeled_full_eval", "proxy_config_conditioned"}


def _accuracy_evidence_label(note: str) -> str:
    note_norm = str(note or "").strip().lower()
    if note_norm == "shared_e0_quant_full_eval_reference":
        return "shared_e0_full_eval_reference"
    if _is_config_conditioned_proxy(note_norm):
        return "modeled_full_eval"
    return "measured_full_eval"


def _reference_basis_label(experiment_id: str, note: str) -> str:
    if _is_config_conditioned_proxy(note):
        return "modeled config-conditioned full-eval; local E0 quant delta also reported"
    if str(note or "").strip().lower() == "shared_e0_quant_full_eval_reference":
        return "local E0 quant (shared full-eval reference)"
    return "local E0 quant"


def _bootstrap_mean(values: np.ndarray, rng: np.random.Generator, n_bootstrap: int) -> dict[str, float]:
    vals = np.asarray(values, dtype=float)
    if vals.size == 0:
        raise ValueError("bootstrap requires at least one value")
    samples = rng.choice(vals, size=(n_bootstrap, vals.size), replace=True)
    boot_means = samples.mean(axis=1)
    return {
        "mean": float(vals.mean()),
        "ci95_low": float(np.percentile(boot_means, 2.5)),
        "ci95_high": float(np.percentile(boot_means, 97.5)),
    }


def _format_interval(mean: float | int, low: float | int, high: float | int, digits: int = 2, suffix: str = "") -> str:
    return f"{float(mean):.{digits}f}{suffix} [{float(low):.{digits}f}, {float(high):.{digits}f}]"


def _format_experiment_list(experiments: list[str]) -> str:
    return ", ".join(f"`{exp}`" for exp in experiments)


def _experiment_run_ids(raw: str | None) -> dict[str, str]:
    mapping = dict(EXPERIMENT_RUN_IDS)
    if not raw:
        return mapping
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise SystemExit("--experiment_run_ids_json must decode to an object.")
    for experiment_id, run_id in payload.items():
        exp = str(experiment_id).strip()
        rid = str(run_id).strip()
        if exp in mapping and rid:
            mapping[exp] = rid
    return mapping


def _read_run_metrics(path: Path, experiment_id: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Missing run metrics: {path}")
    df = pd.read_csv(path)
    df = df[df["experiment_id"] == experiment_id].copy()
    keep = ["run_id", "experiment_id", "model", "seed", "latency_ms", "energy_j", "tops_w"]
    df = df[keep].copy()
    for column in ["seed", "latency_ms", "energy_j", "tops_w"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df["source_run_path"] = str(path)
    return df


def _load_performance_rows(runs_dir: Path, experiment_run_ids: dict[str, str]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for experiment_id in EXPERIMENT_ORDER:
        run_id = experiment_run_ids[experiment_id]
        frames.append(_read_run_metrics(runs_dir / run_id / "master_metrics.csv", experiment_id))
    out = pd.concat(frames, ignore_index=True)
    out["model_order"] = out["model"].map({model: idx for idx, model in enumerate(MODEL_ORDER)})
    out = out.sort_values(["experiment_id", "model_order", "seed"]).drop(columns="model_order").reset_index(drop=True)
    return out


def _load_accuracy_rows(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Missing accuracy CSV: {path}")
    df = pd.read_csv(path)
    df["baseline_flag"] = df["baseline"].astype(str).str.strip().str.lower().isin({"1", "true", "yes"})
    df = df[df["experiment_id"].isin({"E0", "E2", "E3", "E4", "E6"})].copy()
    keep = ["run_id", "experiment_id", "model", "seed", "top1", "top1_delta", "notes", "baseline_flag"]
    df = df[keep].copy()
    for column in ["seed", "top1", "top1_delta"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    fp32 = (
        df[df["baseline_flag"]]
        .groupby(["model", "seed"], as_index=False)
        .agg(fp32_top1=("top1", "mean"))
    )
    quant = (
        df[~df["baseline_flag"]]
        .rename(columns={"top1": "top1_measured"})
        [["experiment_id", "model", "seed", "top1_measured", "top1_delta", "notes"]]
        .copy()
    )
    if "E2" not in set(quant["experiment_id"].astype(str)):
        e2 = quant[quant["experiment_id"] == "E0"].copy()
        e2["experiment_id"] = "E2"
        e2["notes"] = "shared_e0_quant_full_eval_reference"
        quant = pd.concat([quant, e2], ignore_index=True)
    quant["accuracy_source_csv"] = str(path)
    quant = quant.merge(fp32, on=["model", "seed"], how="left", validate="many_to_one")
    quant["acc_drop_pp_vs_fp32"] = quant["fp32_top1"] - quant["top1_measured"]
    quant["accuracy_evidence"] = quant["notes"].map(_accuracy_evidence_label)
    quant["reference_basis"] = quant.apply(
        lambda row: _reference_basis_label(
            str(row["experiment_id"]),
            str(row["notes"]),
        ),
        axis=1,
    )
    return quant


def _expand_seed_invariant_performance_rows(perf_df: pd.DataFrame, acc_df: pd.DataFrame) -> pd.DataFrame:
    acc_seed_keys = acc_df[["experiment_id", "model", "seed"]].drop_duplicates().copy()
    perf_seed_keys = perf_df[["experiment_id", "model", "seed"]].drop_duplicates().copy()
    missing_seed_rows = acc_seed_keys.merge(
        perf_seed_keys,
        on=["experiment_id", "model", "seed"],
        how="left",
        indicator=True,
    )
    if missing_seed_rows["_merge"].eq("both").all():
        return perf_df

    expanded_frames: list[pd.DataFrame] = []
    for (experiment_id, model), acc_subset in acc_seed_keys.groupby(["experiment_id", "model"], sort=False):
        perf_subset = perf_df[
            (perf_df["experiment_id"] == experiment_id) & (perf_df["model"] == model)
        ].copy()
        if perf_subset.empty:
            raise SystemExit(f"Missing performance rows for {experiment_id}/{model}.")

        required_seeds = sorted(acc_subset["seed"].dropna().astype(int).unique().tolist())
        available_seeds = sorted(perf_subset["seed"].dropna().astype(int).unique().tolist())
        if set(required_seeds).issubset(set(available_seeds)):
            expanded_frames.append(perf_subset[perf_subset["seed"].isin(required_seeds)].copy())
            continue

        if len(available_seeds) != 1:
            missing = ",".join(str(seed) for seed in sorted(set(required_seeds) - set(available_seeds)))
            raise SystemExit(
                f"Performance rows for {experiment_id}/{model} do not cover seeds [{missing}] "
                "and cannot be expanded from a single seed-invariant row."
            )

        seed_invariant_perf = perf_subset.drop(columns="seed").drop_duplicates().copy()
        if len(seed_invariant_perf) != 1:
            raise SystemExit(
                f"Expected one seed-invariant performance row for {experiment_id}/{model}, "
                f"found {len(seed_invariant_perf)}."
            )

        seed_invariant_perf["__join"] = 1
        governed_seeds = acc_subset[["seed"]].copy()
        governed_seeds["__join"] = 1
        expanded_subset = seed_invariant_perf.merge(governed_seeds, on="__join", how="inner").drop(columns="__join")
        expanded_frames.append(expanded_subset[perf_df.columns].copy())

    out = pd.concat(expanded_frames, ignore_index=True)
    out["model_order"] = out["model"].map({model: idx for idx, model in enumerate(MODEL_ORDER)})
    out = out.sort_values(["experiment_id", "model_order", "seed"]).drop(columns="model_order").reset_index(drop=True)
    return out


def _build_per_seed(perf_df: pd.DataFrame, acc_df: pd.DataFrame) -> pd.DataFrame:
    perf_df = _expand_seed_invariant_performance_rows(perf_df, acc_df)
    merged = perf_df.merge(
        acc_df,
        on=["experiment_id", "model", "seed"],
        how="left",
        validate="one_to_one",
    )
    e0 = (
        merged[merged["experiment_id"] == "E0"][["model", "seed", "top1_measured"]]
        .rename(columns={"top1_measured": "e0_top1_measured"})
        .drop_duplicates()
    )
    merged = merged.merge(e0, on=["model", "seed"], how="left", validate="many_to_one")
    merged["acc_drop_pp_vs_e0"] = merged["e0_top1_measured"] - merged["top1_measured"]
    return merged.sort_values(["experiment_id", "model", "seed"]).reset_index(drop=True)


def _build_per_model_summary(per_seed_df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        per_seed_df.groupby(["experiment_id", "model"], as_index=False)
        .agg(
            n_seeds=("seed", "nunique"),
            latency_ms_mean=("latency_ms", "mean"),
            latency_ms_std=("latency_ms", "std"),
            energy_j_mean=("energy_j", "mean"),
            energy_j_std=("energy_j", "std"),
            tops_w_mean=("tops_w", "mean"),
            tops_w_std=("tops_w", "std"),
            top1_mean=("top1_measured", "mean"),
            top1_std=("top1_measured", "std"),
            acc_drop_pp_vs_fp32_mean=("acc_drop_pp_vs_fp32", "mean"),
            acc_drop_pp_vs_e0_mean=("acc_drop_pp_vs_e0", "mean"),
            accuracy_note=("notes", "first"),
            accuracy_evidence=("accuracy_evidence", "first"),
            reference_basis=("reference_basis", "first"),
        )
    )
    grouped["model_order"] = grouped["model"].map({model: idx for idx, model in enumerate(MODEL_ORDER)})
    return grouped.sort_values(["experiment_id", "model_order"]).drop(columns="model_order").reset_index(drop=True)


def _build_config_summary(
    per_model_df: pd.DataFrame,
    runs_dir: Path,
    experiment_run_ids: dict[str, str],
    rng: np.random.Generator,
    n_bootstrap: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for experiment_id in EXPERIMENT_ORDER:
        subset = per_model_df[per_model_df["experiment_id"] == experiment_id].copy()
        if subset.empty:
            continue
        metrics = {
            "latency_ms": subset["latency_ms_mean"].to_numpy(dtype=float),
            "energy_j": subset["energy_j_mean"].to_numpy(dtype=float),
            "tops_w": subset["tops_w_mean"].to_numpy(dtype=float),
            "top1_measured": subset["top1_mean"].to_numpy(dtype=float),
            "acc_drop_pp_vs_fp32": subset["acc_drop_pp_vs_fp32_mean"].to_numpy(dtype=float),
        }
        row: dict[str, object] = {
            "experiment_id": experiment_id,
            "pair_basis": "model_mean_over_seeds",
            "n_rows": int(len(subset)),
            "n_models": int(len(subset)),
            "n_seeds": int(subset["n_seeds"].max()),
            "performance_run_id": experiment_run_ids[experiment_id],
            "performance_run_path": str(runs_dir / experiment_run_ids[experiment_id] / "master_metrics.csv"),
            "accuracy_note": str(subset["accuracy_note"].iloc[0]),
            "accuracy_evidence": str(subset["accuracy_evidence"].iloc[0]),
            "reference_basis": str(subset["reference_basis"].iloc[0]),
        }
        for metric_name, values in metrics.items():
            stats = _bootstrap_mean(values, rng, n_bootstrap)
            row[f"{metric_name}_mean"] = stats["mean"]
            row[f"{metric_name}_ci95_low"] = stats["ci95_low"]
            row[f"{metric_name}_ci95_high"] = stats["ci95_high"]
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary["experiment_order"] = summary["experiment_id"].map({exp: idx for idx, exp in enumerate(EXPERIMENT_ORDER)})
    return summary.sort_values("experiment_order").drop(columns="experiment_order").reset_index(drop=True)


def _build_pair_summary(
    per_model_df: pd.DataFrame,
    rng: np.random.Generator,
    n_bootstrap: int,
) -> pd.DataFrame:
    e0 = (
        per_model_df[per_model_df["experiment_id"] == "E0"][
            ["model", "latency_ms_mean", "energy_j_mean", "tops_w_mean", "top1_mean"]
        ]
        .rename(
            columns={
                "latency_ms_mean": "latency_ms_e0",
                "energy_j_mean": "energy_j_e0",
                "tops_w_mean": "tops_w_e0",
                "top1_mean": "top1_measured_e0",
            }
        )
        .copy()
    )
    rows: list[dict[str, object]] = []
    for experiment_id in [exp for exp in EXPERIMENT_ORDER if exp != "E0"]:
        subset = per_model_df[per_model_df["experiment_id"] == experiment_id].copy()
        paired = subset.merge(e0, on=["model"], how="inner", validate="one_to_one")
        paired["speedup_vs_e0"] = paired["latency_ms_e0"] / paired["latency_ms_mean"]
        paired["energy_reduction_pct_vs_e0"] = 100.0 * (1.0 - paired["energy_j_mean"] / paired["energy_j_e0"])
        paired["tops_w_gain_vs_e0"] = paired["tops_w_mean"] / paired["tops_w_e0"]
        paired["acc_drop_pp_vs_e0"] = paired["top1_measured_e0"] - paired["top1_mean"]
        row: dict[str, object] = {
            "experiment_id": experiment_id,
            "pair_basis": "model_mean_over_seeds",
            "n_pairs": int(len(paired)),
        }
        for metric_name in [
            "speedup_vs_e0",
            "energy_reduction_pct_vs_e0",
            "tops_w_gain_vs_e0",
            "acc_drop_pp_vs_e0",
        ]:
            stats = _bootstrap_mean(paired[metric_name].to_numpy(dtype=float), rng, n_bootstrap)
            row[f"{metric_name}_mean"] = stats["mean"]
            row[f"{metric_name}_ci95_low"] = stats["ci95_low"]
            row[f"{metric_name}_ci95_high"] = stats["ci95_high"]
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary["experiment_order"] = summary["experiment_id"].map({exp: idx for idx, exp in enumerate(EXPERIMENT_ORDER)})
    return summary.sort_values("experiment_order").drop(columns="experiment_order").reset_index(drop=True)


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, quoting=csv.QUOTE_MINIMAL)


def _write_appendix_table(path: Path, config_summary: pd.DataFrame, pair_summary: pd.DataFrame) -> None:
    pair_by_exp = {row["experiment_id"]: row for _, row in pair_summary.iterrows()}
    lines = [
        "\\begin{tabular}{lcccccc}",
        "\\toprule",
        "Config & Top-1 (95\\% CI) & Latency ms (95\\% CI) & Energy J (95\\% CI) & Speedup vs E0 & Energy Red. vs E0 & Acc. Drop vs E0 \\\\",
        "\\midrule",
    ]
    for _, row in config_summary.iterrows():
        exp = row["experiment_id"]
        pair = pair_by_exp.get(exp)
        top1 = _format_interval(row["top1_measured_mean"], row["top1_measured_ci95_low"], row["top1_measured_ci95_high"], digits=2)
        latency = _format_interval(row["latency_ms_mean"], row["latency_ms_ci95_low"], row["latency_ms_ci95_high"], digits=3)
        energy = _format_interval(row["energy_j_mean"], row["energy_j_ci95_low"], row["energy_j_ci95_high"], digits=4)
        if pair is None:
            speedup = "baseline"
            energy_red = "baseline"
            acc_drop = "baseline"
        else:
            speedup = _format_interval(pair["speedup_vs_e0_mean"], pair["speedup_vs_e0_ci95_low"], pair["speedup_vs_e0_ci95_high"], digits=2, suffix="x")
            energy_red = _format_interval(
                pair["energy_reduction_pct_vs_e0_mean"],
                pair["energy_reduction_pct_vs_e0_ci95_low"],
                pair["energy_reduction_pct_vs_e0_ci95_high"],
                digits=2,
                suffix="\\%",
            )
            acc_drop = _format_interval(
                pair["acc_drop_pp_vs_e0_mean"],
                pair["acc_drop_pp_vs_e0_ci95_low"],
                pair["acc_drop_pp_vs_e0_ci95_high"],
                digits=2,
                suffix=" pp",
            )
        lines.append(f"{exp} & {top1} & {latency} & {energy} & {speedup} & {energy_red} & {acc_drop} \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_report(
    path: Path,
    per_seed_path: Path,
    per_model_path: Path,
    config_summary_path: Path,
    pair_summary_path: Path,
    table_path: Path,
    accuracy_csv: Path,
    runs_dir: Path,
    config_summary: pd.DataFrame,
    pair_summary: pd.DataFrame,
    experiment_run_ids: dict[str, str],
    n_bootstrap: int,
    bootstrap_seed: int,
    tag: str,
) -> None:
    pair_by_exp = {row["experiment_id"]: row for _, row in pair_summary.iterrows()}
    e2_row = config_summary.loc[config_summary["experiment_id"] == "E2"].iloc[0]
    e2_uses_shared_e0_reference = str(e2_row["accuracy_evidence"]) == "shared_e0_full_eval_reference"
    config_conditioned_summary = config_summary.loc[
        config_summary["experiment_id"].isin({"E3", "E4", "E6"})
    ].copy()
    modeled_experiments = config_conditioned_summary.loc[
        config_conditioned_summary["accuracy_evidence"].map(_is_modeled_accuracy_evidence),
        "experiment_id",
    ].astype(str).tolist()
    measured_experiments = config_conditioned_summary.loc[
        config_conditioned_summary["accuracy_evidence"] == "measured_full_eval",
        "experiment_id",
    ].astype(str).tolist()
    if e2_uses_shared_e0_reference:
        e2_method_line = "- `E2` has no dedicated full-eval accuracy rows in the repaired chain; its Top-1 rows are explicitly tagged as `shared_e0_full_eval_reference` because `E2` changes `HOPS` only and reuses the same local E0 quantized full-eval reference."
        e2_interpretation_line = "- `E2` remains a latency-positive HOPS-only point, and its Top-1 is carried by the shared local `E0` quantized full-eval reference rather than a separate E2 rerun."
    else:
        e2_method_line = "- `E2` now has dedicated full-eval accuracy rows under the same repaired protocol used for the retained chain, so its Top-1 is tagged as `measured_full_eval`."
        e2_interpretation_line = "- `E2` now has both a measured latency gain and dedicated full-eval Top-1 evidence under the repaired protocol."
    modeled_method_line = None
    if modeled_experiments:
        modeled_method_line = (
            f"- Modeled config-conditioned Top-1 rows in this chain ({_format_experiment_list(modeled_experiments)}) "
            "are tagged as `modeled_full_eval` because they come from config-conditioned simulation under the repaired full-eval protocol rather than dedicated measured full-eval reruns."
        )
    measured_method_line = None
    if measured_experiments:
        measured_method_line = (
            f"- Measured config-conditioned rows in this chain ({_format_experiment_list(measured_experiments)}) "
            "now carry dedicated `measured_full_eval` evidence under the repaired protocol."
        )
    modeled_interpretation_line = None
    if modeled_experiments:
        modeled_interpretation_line = (
            f"- Modeled config-conditioned configs in this chain ({_format_experiment_list(modeled_experiments)}) "
            "remain modeled tradeoff indicators rather than direct measured acceptance evidence."
        )
    measured_interpretation_line = None
    if measured_experiments:
        measured_interpretation_line = (
            f"- Measured config-conditioned configs in this chain ({_format_experiment_list(measured_experiments)}) "
            "should be interpreted from their own measured rows and not conflated with the modeled configurations."
        )
    lines = [
        f"# Headline Statistics Report ({tag})",
        "",
        "Scope",
        f"- Performance runs: `{runs_dir / experiment_run_ids['E0'] / 'master_metrics.csv'}`, `{runs_dir / experiment_run_ids['E2'] / 'master_metrics.csv'}`, `{runs_dir / experiment_run_ids['E3'] / 'master_metrics.csv'}`, `{runs_dir / experiment_run_ids['E4'] / 'master_metrics.csv'}`, `{runs_dir / experiment_run_ids['E6'] / 'master_metrics.csv'}`",
        f"- Accuracy source CSV: `{accuracy_csv}`",
        f"- Per-model per-seed export: `{per_seed_path}`",
        f"- Per-model summary: `{per_model_path}`",
        f"- Config summary with bootstrap CI: `{config_summary_path}`",
        f"- Paired bootstrap CI vs E0: `{pair_summary_path}`",
        f"- Appendix table: `{table_path}`",
        "",
        "Method",
        "- Headline CI uses `model_mean_over_seeds` as the estimand, giving `3` matched model-level observations per non-baseline configuration after averaging over the released seeds.",
        f"- All confidence intervals are percentile bootstrap intervals over the mean with `{n_bootstrap}` resamples and RNG seed `{bootstrap_seed}`.",
        "- The current repaired chain releases all governed accuracy seeds into the per-seed exports; it no longer silently collapses the retained configs to `seed=0`.",
        "- Per-model per-seed rows are still exported separately, but the headline CI avoids pseudo-replication from seed-invariant simulator performance metrics.",
        "- When a retained `master_metrics.csv` publishes only one seed, its latency/energy/TOPS/W row is broadcast across the governed accuracy seeds for the same `(experiment, model)` because those simulator metrics are seed-invariant in the released chain.",
        "- Paired deltas are reported against `E0` for speedup, energy reduction, TOPS/W gain, and Top-1 drop using model-level means.",
        e2_method_line,
        "",
        "Config Means",
    ]
    if modeled_method_line is not None:
        lines.insert(-2, modeled_method_line)
    if measured_method_line is not None:
        lines.insert(-2, measured_method_line)
    for _, row in config_summary.iterrows():
        lines.append(
            f"- `{row['experiment_id']}`: Top-1 {_format_interval(row['top1_measured_mean'], row['top1_measured_ci95_low'], row['top1_measured_ci95_high'], digits=2)}, "
            f"latency {_format_interval(row['latency_ms_mean'], row['latency_ms_ci95_low'], row['latency_ms_ci95_high'], digits=3)} ms, "
            f"energy {_format_interval(row['energy_j_mean'], row['energy_j_ci95_low'], row['energy_j_ci95_high'], digits=4)} J, "
            f"acc. drop vs FP32 {_format_interval(row['acc_drop_pp_vs_fp32_mean'], row['acc_drop_pp_vs_fp32_ci95_low'], row['acc_drop_pp_vs_fp32_ci95_high'], digits=2)} pp, "
            f"accuracy evidence `{row['accuracy_evidence']}`."
        )
    lines.extend(["", "Paired Delta CI vs E0"])
    for exp in ["E2", "E3", "E4", "E6"]:
        row = pair_by_exp[exp]
        lines.append(
            f"- `{exp}`: speedup {_format_interval(row['speedup_vs_e0_mean'], row['speedup_vs_e0_ci95_low'], row['speedup_vs_e0_ci95_high'], digits=2, suffix='x')}, "
            f"energy reduction {_format_interval(row['energy_reduction_pct_vs_e0_mean'], row['energy_reduction_pct_vs_e0_ci95_low'], row['energy_reduction_pct_vs_e0_ci95_high'], digits=2)}%, "
            f"TOPS/W gain {_format_interval(row['tops_w_gain_vs_e0_mean'], row['tops_w_gain_vs_e0_ci95_low'], row['tops_w_gain_vs_e0_ci95_high'], digits=2, suffix='x')}, "
            f"Top-1 drop {_format_interval(row['acc_drop_pp_vs_e0_mean'], row['acc_drop_pp_vs_e0_ci95_low'], row['acc_drop_pp_vs_e0_ci95_high'], digits=2)} pp."
        )
    lines.extend(
        [
            "",
            "Interpretation",
            "- This closes the main-config uncertainty gap for `E0/E2/E3/E4/E6` with seed-level exports plus conservative model-level paired bootstrap CI instead of relying on descriptive cross-model spread.",
            e2_interpretation_line,
            "",
            "P1-1 Completion Check",
            "- `per-model per-seed latency`: satisfied in the per-seed CSV.",
            "- `per-model per-seed energy`: satisfied in the per-seed CSV.",
            "- `per-model per-seed Top-1`: satisfied in the per-seed CSV.",
            "- `paired delta CI`: satisfied in the paired CI CSV and appendix table.",
            "- `new statistics note`: satisfied by this markdown report.",
        ]
    )
    if modeled_interpretation_line is not None:
        lines.insert(-7, modeled_interpretation_line)
    if measured_interpretation_line is not None:
        lines.insert(-7, measured_interpretation_line)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the P1-1 headline statistics report and supporting tables.")
    parser.add_argument("--runs_dir", type=Path, default=DEFAULT_RUNS_DIR)
    parser.add_argument("--accuracy_csv", type=Path, default=DEFAULT_ACCURACY_CSV)
    parser.add_argument("--out_data_dir", type=Path, default=DEFAULT_OUT_DATA_DIR)
    parser.add_argument("--tag", default=DEFAULT_TAG)
    parser.add_argument(
        "--experiment_run_ids_json",
        default=None,
        help='Optional JSON object overriding headline experiment->run_id mapping, e.g. {"E0":"run_e0"}.',
    )
    parser.add_argument("--bootstrap_samples", type=int, default=DEFAULT_BOOTSTRAP_SAMPLES)
    parser.add_argument("--bootstrap_seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    args = parser.parse_args()

    args.out_data_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.bootstrap_seed)
    experiment_run_ids = _experiment_run_ids(args.experiment_run_ids_json)

    perf_df = _load_performance_rows(args.runs_dir, experiment_run_ids)
    acc_df = _load_accuracy_rows(args.accuracy_csv)
    per_seed_df = _build_per_seed(perf_df, acc_df)
    per_model_df = _build_per_model_summary(per_seed_df)
    config_summary_df = _build_config_summary(per_model_df, args.runs_dir, experiment_run_ids, rng, args.bootstrap_samples)
    pair_summary_df = _build_pair_summary(per_model_df, rng, args.bootstrap_samples)

    per_seed_path = args.out_data_dir / f"headline_statistics_per_seed_{args.tag}.csv"
    per_model_path = args.out_data_dir / f"headline_statistics_per_model_{args.tag}.csv"
    config_summary_path = args.out_data_dir / f"headline_statistics_summary_{args.tag}.csv"
    pair_summary_path = args.out_data_dir / f"headline_statistics_pair_ci_{args.tag}.csv"
    report_path = args.out_data_dir / f"headline_statistics_report_{args.tag}.md"
    table_path = args.out_data_dir / f"headline_statistics_appendix_table_{args.tag}.tex"

    _write_csv(per_seed_path, per_seed_df)
    _write_csv(per_model_path, per_model_df)
    _write_csv(config_summary_path, config_summary_df)
    _write_csv(pair_summary_path, pair_summary_df)
    _write_appendix_table(table_path, config_summary_df, pair_summary_df)
    _write_report(
        report_path,
        per_seed_path,
        per_model_path,
        config_summary_path,
        pair_summary_path,
        table_path,
        args.accuracy_csv,
        args.runs_dir,
        config_summary_df,
        pair_summary_df,
        experiment_run_ids,
        args.bootstrap_samples,
        args.bootstrap_seed,
        args.tag,
    )

    print(per_seed_path)
    print(per_model_path)
    print(config_summary_path)
    print(pair_summary_path)
    print(table_path)
    print(report_path)


if __name__ == "__main__":
    main()
