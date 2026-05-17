#!/usr/bin/env python3
"""Build a conservative SPARSE interaction note from released full-eval and trace artifacts."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PAIR_CI_CSV = ROOT / "AICAS" / "assets" / "candidate_data" / "headline_statistics_pair_ci_20260310.csv"
DEFAULT_FULL_EVAL_PER_MODEL_CSV = (
    ROOT / "AICAS" / "assets" / "candidate_data" / "config_conditioned_accuracy_per_model_20260307_fulleval_seeds012.csv"
)
DEFAULT_SUBSET_TAU_SUMMARY_CSV = (
    ROOT / "AICAS" / "assets" / "candidate_data" / "sparse_tau_accuracy_summary_20260307_sparsetau4096_seeds012.csv"
)
DEFAULT_PASCAL_SUMMARY_CSV = (
    ROOT / "AICAS" / "assets" / "candidate_data" / "pascalvoc_seg_config_conditioned_summary_20260308_pascalvocconfig.csv"
)
DEFAULT_ADE_SUMMARY_CSV = (
    ROOT / "AICAS" / "assets" / "candidate_data" / "ade20k_seg_config_conditioned_summary_20260308_ade20kconfig.csv"
)
DEFAULT_RUNS_ROOT = ROOT / "experiments" / "results" / "runs"
DEFAULT_OUT_DATA_DIR = ROOT / "AICAS" / "assets" / "candidate_data"
DEFAULT_TAG = "20260310"

MAINCHAIN_RUNS = {
    "E4": ["20260228_opt_sync_core_e4", "20260228_opt_sync_core_e4_s1", "20260228_opt_sync_core_e4_s2"],
    "E6": ["20260228_opt_sync_core_e6", "20260228_opt_sync_core_e6_s1", "20260228_opt_sync_core_e6_s2"],
}
REFERENCE_TRACE_RUN = "20260222_cuda_v31_core_e4_t20"
EXPERIMENT_ORDER = ["E3", "E4", "E6"]
MODEL_ORDER = ["mobilevit_xxs", "mobilevit_xs", "mobilevit_s"]


def _format_num(value: float | None, digits: int = 2, suffix: str = "") -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value):.{digits}f}{suffix}"


def _clamp_unit_interval(value: float) -> float:
    return max(0.0, min(1.0, value))


def _resolve_sparse_compat_active_fraction(sparse_cfg: dict[str, object]) -> float | None:
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
            min_active = sparse_cfg.get("min_active_fraction")
            min_active_value = float(min_active) if min_active is not None else 0.0
            min_active_value = _clamp_unit_interval(min_active_value)
            if tau_value <= 0.0:
                return 1.0
            return _clamp_unit_interval(max(min_active_value, 1.0 - tau_value))
        except (TypeError, ValueError):
            pass
    if active_fraction is None:
        return None
    try:
        return _clamp_unit_interval(float(active_fraction))
    except (TypeError, ValueError):
        return None


def _validate_unit_interval_series(frame: pd.DataFrame, column: str, source: Path) -> pd.Series:
    values = pd.to_numeric(frame[column], errors="coerce")
    invalid_mask = values.isna() | ~values.map(math.isfinite) | (values < 0.0) | (values > 1.0)
    if invalid_mask.any():
        invalid_rows = (invalid_mask[invalid_mask].index + 2).tolist()
        bad_values = ", ".join(str(frame.loc[idx, column]) for idx in invalid_mask[invalid_mask].index[:3])
        raise ValueError(
            f"{source} has invalid {column} values outside [0, 1] at CSV rows {invalid_rows}: {bad_values}"
        )
    return values


def _extract_sparse_cfg(config_path: Path) -> dict[str, float | bool | None]:
    if not config_path.is_file():
        return {
            "configured_tau_global": None,
            "configured_active_fraction": None,
            "use_tau_for_gating": None,
        }
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    sparse_cfg = payload.get("sparse") or {}
    configured_tau_global = sparse_cfg.get("tau_global")
    try:
        configured_tau_global = (
            None if configured_tau_global is None else float(configured_tau_global)
        )
    except (TypeError, ValueError):
        configured_tau_global = None
    configured_active_fraction = _resolve_sparse_compat_active_fraction(sparse_cfg)
    return {
        "configured_tau_global": configured_tau_global,
        "configured_active_fraction": configured_active_fraction,
        "use_tau_for_gating": sparse_cfg.get("use_tau_for_gating"),
    }


def _load_headline_summary(pair_ci_csv: Path) -> pd.DataFrame:
    pair = pd.read_csv(pair_ci_csv)
    pair = pair[pair["experiment_id"].isin(EXPERIMENT_ORDER)].copy()
    pair["speedup_vs_e0_mean"] = pd.to_numeric(pair["speedup_vs_e0_mean"], errors="coerce")
    pair["energy_reduction_pct_vs_e0_mean"] = pd.to_numeric(pair["energy_reduction_pct_vs_e0_mean"], errors="coerce")
    pair["acc_drop_pp_vs_e0_mean"] = pd.to_numeric(pair["acc_drop_pp_vs_e0_mean"], errors="coerce")
    return pair.sort_values("experiment_id").reset_index(drop=True)


def _load_model_interaction(full_eval_per_model_csv: Path) -> tuple[pd.DataFrame, float]:
    frame = pd.read_csv(full_eval_per_model_csv)
    frame = frame[frame["experiment_id"].isin(EXPERIMENT_ORDER)].copy()
    frame["delta_vs_e0_quant_pp"] = pd.to_numeric(frame["delta_vs_e0_quant_pp"], errors="coerce")
    rows: list[dict[str, object]] = []
    residuals: list[float] = []
    for model in MODEL_ORDER:
        subset = frame[frame["model"] == model].set_index("experiment_id")
        if any(exp not in subset.index for exp in EXPERIMENT_ORDER):
            continue
        e3 = float(subset.loc["E3", "delta_vs_e0_quant_pp"])
        e4 = float(subset.loc["E4", "delta_vs_e0_quant_pp"])
        e6 = float(subset.loc["E6", "delta_vs_e0_quant_pp"])
        residual = e6 - (e3 + e4)
        residuals.append(residual)
        rows.append(
            {
                "model": model,
                "e3_delta_vs_e0_quant_pp": e3,
                "e4_delta_vs_e0_quant_pp": e4,
                "e6_delta_vs_e0_quant_pp": e6,
                "residual_vs_additive_pp": residual,
            }
        )
    return pd.DataFrame(rows), (sum(residuals) / len(residuals) if residuals else float("nan"))


def _load_subset_tradeoff(subset_tau_summary_csv: Path) -> pd.DataFrame:
    subset = pd.read_csv(subset_tau_summary_csv)
    keep = [
        "run_id",
        "sparse_tau_global",
        "sparse_active_fraction",
        "paired_model_mean_delta_vs_e0_quant_pp",
        "speedup_vs_E0",
        "duty_cycle_avg",
    ]
    subset = subset[keep].copy()
    for column in keep[1:]:
        subset[column] = pd.to_numeric(subset[column], errors="coerce")
    subset["sparse_active_fraction"] = _validate_unit_interval_series(
        subset,
        "sparse_active_fraction",
        subset_tau_summary_csv,
    )
    return subset.sort_values("sparse_tau_global").reset_index(drop=True)


def _subset_interpretation(subset_df: pd.DataFrame) -> list[str]:
    notes: list[str] = []
    tau15 = subset_df[subset_df["sparse_tau_global"].round(6) == 0.15]
    tau20 = subset_df[subset_df["sparse_tau_global"].round(6) == 0.20]
    if not tau15.empty and not tau20.empty:
        tau15_drop = float(tau15.iloc[0]["paired_model_mean_delta_vs_e0_quant_pp"])
        tau20_drop = float(tau20.iloc[0]["paired_model_mean_delta_vs_e0_quant_pp"])
        if tau15_drop <= -5.0 and tau20_drop <= -5.0:
            notes.append(
                f"- On the repaired direct-semantics subset sweep, both `tau=0.15` and `tau=0.20` already collapse (`{_format_num(tau15_drop, 2, ' pp')}` and `{_format_num(tau20_drop, 2, ' pp')}` vs E0 quant), so these points should not be framed as moderate sparse presets."
            )
        else:
            notes.append(
                f"- On the repaired direct-semantics subset sweep, `tau=0.15` and `tau=0.20` land at `{_format_num(tau15_drop, 2, ' pp')}` and `{_format_num(tau20_drop, 2, ' pp')}` vs E0 quant, so their usability must be read from the measured curve rather than assumed."
            )
    best_nonzero = subset_df[subset_df["sparse_tau_global"] > 0.0].copy()
    if not best_nonzero.empty:
        best_row = best_nonzero.sort_values("paired_model_mean_delta_vs_e0_quant_pp", ascending=False).iloc[0]
        notes.append(
            f"- The least-negative non-zero sparse row in this subset summary is `tau={float(best_row['sparse_tau_global']):.2f}` with `{_format_num(best_row['paired_model_mean_delta_vs_e0_quant_pp'], 2, ' pp')}` vs E0 quant."
        )
    return notes


def _load_segmentation_interaction(path: Path, dataset: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame = frame[frame["experiment_id"].isin(EXPERIMENT_ORDER)].copy()
    keep_metrics = ["finite_mean_iou_delta_pp", "global_correct_delta_pp"]
    for column in keep_metrics:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    by_exp = frame.set_index("experiment_id")
    rows: list[dict[str, object]] = []
    for metric in keep_metrics:
        e3 = float(by_exp.loc["E3", metric])
        e4 = float(by_exp.loc["E4", metric])
        e6 = float(by_exp.loc["E6", metric])
        rows.append(
            {
                "dataset": dataset,
                "metric": metric,
                "e3_delta": e3,
                "e4_delta": e4,
                "e6_delta": e6,
                "residual_vs_additive": e6 - (e3 + e4),
            }
        )
    return pd.DataFrame(rows)


def _audit_trace_run(run_root: Path, experiment_id: str, run_id: str, run_group: str) -> dict[str, object]:
    run_dir = run_root / run_id
    config_path = run_dir / "config_snapshot.yaml"
    phy_path = run_dir / "per_layer_phy.csv"
    sparse_cfg = _extract_sparse_cfg(config_path)
    configured_active_fraction = sparse_cfg["configured_active_fraction"]
    if not phy_path.is_file():
        return {
            "experiment_id": experiment_id,
            "run_group": run_group,
            "run_id": run_id,
            "configured_tau_global": sparse_cfg["configured_tau_global"],
            "configured_active_fraction": configured_active_fraction,
            "use_tau_for_gating": sparse_cfg["use_tau_for_gating"],
            "nonzero_layer_count": 0,
            "unique_nonzero_active_channels": "",
            "unique_nonzero_duty_cycle": "",
            "mean_nonzero_duty_cycle": None,
            "trace_sparse_visible": False,
            "trace_status": "missing_per_layer_phy",
        }

    frame = pd.read_csv(phy_path)
    frame["active_channels"] = pd.to_numeric(frame["active_channels"], errors="coerce")
    frame["duty_cycle"] = pd.to_numeric(frame["duty_cycle"], errors="coerce")
    nonzero = frame[frame["active_channels"] > 0].copy()
    unique_channels = sorted({int(value) for value in nonzero["active_channels"].dropna().tolist()})
    unique_duty = sorted({round(float(value), 6) for value in nonzero["duty_cycle"].dropna().tolist()})
    mean_duty = float(nonzero["duty_cycle"].mean()) if not nonzero.empty else None
    trace_sparse_visible = mean_duty is not None and mean_duty < 0.999
    if not nonzero.empty and len(unique_channels) == 1 and len(unique_duty) == 1:
        trace_status = "uniform_nonzero_layers"
    elif nonzero.empty:
        trace_status = "no_nonzero_layers"
    else:
        trace_status = "heterogeneous_nonzero_layers"
    return {
        "experiment_id": experiment_id,
        "run_group": run_group,
        "run_id": run_id,
        "configured_tau_global": sparse_cfg["configured_tau_global"],
        "configured_active_fraction": configured_active_fraction,
        "use_tau_for_gating": sparse_cfg["use_tau_for_gating"],
        "nonzero_layer_count": int(len(nonzero)),
        "unique_nonzero_active_channels": ";".join(str(value) for value in unique_channels),
        "unique_nonzero_duty_cycle": ";".join(f"{value:.6f}" for value in unique_duty),
        "mean_nonzero_duty_cycle": mean_duty,
        "trace_sparse_visible": bool(trace_sparse_visible),
        "trace_status": trace_status,
    }


def _build_trace_audit(run_root: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for experiment_id, run_ids in MAINCHAIN_RUNS.items():
        for run_id in run_ids:
            rows.append(_audit_trace_run(run_root, experiment_id, run_id, "main_chain"))
    rows.append(_audit_trace_run(run_root, "E4", REFERENCE_TRACE_RUN, "subset_tau020_reference"))
    out = pd.DataFrame(rows)
    return out.sort_values(["run_group", "experiment_id", "run_id"]).reset_index(drop=True)


def _build_summary(
    headline_df: pd.DataFrame,
    model_df: pd.DataFrame,
    subset_df: pd.DataFrame,
    segmentation_df: pd.DataFrame,
    trace_df: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for row in headline_df.itertuples(index=False):
        rows.append(
            {
                "section": "imagenet_headline",
                "row_key": row.experiment_id,
                "metric": "top1_delta_vs_e0_quant_pp",
                "speedup_vs_e0": row.speedup_vs_e0_mean,
                "energy_reduction_pct_vs_e0": row.energy_reduction_pct_vs_e0_mean,
                "delta_value": -row.acc_drop_pp_vs_e0_mean,
                "notes": "paired_mean_over_models",
            }
        )
    for row in model_df.itertuples(index=False):
        rows.append(
            {
                "section": "imagenet_model_interaction",
                "row_key": row.model,
                "metric": "residual_vs_additive_pp",
                "speedup_vs_e0": None,
                "energy_reduction_pct_vs_e0": None,
                "delta_value": row.residual_vs_additive_pp,
                "notes": "negative means E6 is worse than E3+E4 additive expectation",
            }
        )
    for row in subset_df.itertuples(index=False):
        rows.append(
            {
                "section": "subset_tau_tradeoff",
                "row_key": row.run_id,
                "metric": "delta_vs_e0_quant_pp",
                "speedup_vs_e0": row.speedup_vs_E0,
                "energy_reduction_pct_vs_e0": None,
                "delta_value": row.paired_model_mean_delta_vs_e0_quant_pp,
                "notes": (
                    f"tau={row.sparse_tau_global:.2f}; "
                    f"compat_active_fraction={row.sparse_active_fraction:.2f}; "
                    f"duty={row.duty_cycle_avg:.3f}"
                ),
            }
        )
    for row in segmentation_df.itertuples(index=False):
        rows.append(
            {
                "section": f"segmentation_{row.dataset}",
                "row_key": row.metric,
                "metric": "residual_vs_additive",
                "speedup_vs_e0": None,
                "energy_reduction_pct_vs_e0": None,
                "delta_value": row.residual_vs_additive,
                "notes": "negative means E6 closes more negatively than additive expectation",
            }
        )
    for row in trace_df.itertuples(index=False):
        rows.append(
            {
                "section": "per_layer_trace_audit",
                "row_key": row.run_id,
                "metric": "mean_nonzero_duty_cycle",
                "speedup_vs_e0": None,
                "energy_reduction_pct_vs_e0": None,
                "delta_value": row.mean_nonzero_duty_cycle,
                "notes": (
                    f"configured_tau={_format_num(row.configured_tau_global, 2)}; "
                    f"configured_active_fraction={_format_num(row.configured_active_fraction, 2)}; "
                    f"channels={row.unique_nonzero_active_channels or 'n/a'}; "
                    f"trace_status={row.trace_status}"
                ),
            }
        )
    return pd.DataFrame(rows)


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _write_report(
    report_path: Path,
    summary_csv: Path,
    trace_csv: Path,
    pair_ci_csv: Path,
    full_eval_per_model_csv: Path,
    subset_tau_summary_csv: Path,
    pascal_summary_csv: Path,
    ade_summary_csv: Path,
    tag: str,
    headline_df: pd.DataFrame,
    model_df: pd.DataFrame,
    subset_df: pd.DataFrame,
    segmentation_df: pd.DataFrame,
    trace_df: pd.DataFrame,
    mean_model_residual: float,
) -> None:
    headline = {row["experiment_id"]: row for _, row in headline_df.iterrows()}
    main_e4 = trace_df[(trace_df["experiment_id"] == "E4") & (trace_df["run_group"] == "main_chain")].copy()
    main_e6 = trace_df[(trace_df["experiment_id"] == "E6") & (trace_df["run_group"] == "main_chain")].copy()
    subset_trace = trace_df[trace_df["run_group"] == "subset_tau020_reference"].iloc[0]
    pascal = segmentation_df[segmentation_df["dataset"] == "pascalvoc"].set_index("metric")
    ade = segmentation_df[segmentation_df["dataset"] == "ade20k"].set_index("metric")
    lines = [
        f"# SPARSE Interaction Note ({tag})",
        "",
        "Scope",
        f"- Summary CSV: `{summary_csv}`",
        f"- Trace audit CSV: `{trace_csv}`",
        f"- ImageNet headline source: `{pair_ci_csv}`",
        f"- Full-eval per-model source: `{full_eval_per_model_csv}`",
        f"- Subset sparse-tau source: `{subset_tau_summary_csv}`",
        f"- Broader-task sources: `{pascal_summary_csv}` and `{ade_summary_csv}`",
        "",
        "Full-eval separation",
        (
            f"- `E3 DET-only`: speedup `{_format_num(headline['E3']['speedup_vs_e0_mean'], 2, 'x')}`, "
            f"energy reduction `{_format_num(headline['E3']['energy_reduction_pct_vs_e0_mean'], 2, '%')}`, "
            f"Top-1 delta vs E0 quant `-{_format_num(headline['E3']['acc_drop_pp_vs_e0_mean'], 2, ' pp')}`."
        ),
        (
            f"- `E4 SPARSE-only`: speedup `{_format_num(headline['E4']['speedup_vs_e0_mean'], 2, 'x')}`, "
            f"energy reduction `{_format_num(headline['E4']['energy_reduction_pct_vs_e0_mean'], 2, '%')}`, "
            f"Top-1 delta vs E0 quant `-{_format_num(headline['E4']['acc_drop_pp_vs_e0_mean'], 2, ' pp')}`."
        ),
        (
            f"- `E6 DET+SPARSE`: speedup `{_format_num(headline['E6']['speedup_vs_e0_mean'], 2, 'x')}`, "
            f"energy reduction `{_format_num(headline['E6']['energy_reduction_pct_vs_e0_mean'], 2, '%')}`, "
            f"Top-1 delta vs E0 quant `-{_format_num(headline['E6']['acc_drop_pp_vs_e0_mean'], 2, ' pp')}`."
        ),
        (
            f"- Model residual vs additive (`E6 - (E3 + E4)`): mean `{_format_num(mean_model_residual, 2, ' pp')}`; "
            + ", ".join(
                f"{row.model} `{_format_num(row.residual_vs_additive_pp, 2, ' pp')}`"
                for row in model_df.itertuples(index=False)
            )
            + ". Negative values mean the combined point closes worse than additive."
        ),
        "",
        "Subset tradeoff support",
    ]
    for row in subset_df.itertuples(index=False):
        lines.append(
            f"- `tau={row.sparse_tau_global:.2f}` with compatibility active fraction `{row.sparse_active_fraction:.2f}`: "
            f"subset delta vs E0 quant `{_format_num(row.paired_model_mean_delta_vs_e0_quant_pp, 2, ' pp')}`, "
            f"speedup `{_format_num(row.speedup_vs_E0, 2, 'x')}`, duty `{_format_num(row.duty_cycle_avg, 3)}`."
        )
    lines.extend(
        [
            "",
            "Per-layer trace audit",
            (
                f"- Main-chain `E4` seed runs all configure tau `{_format_num(main_e4['configured_tau_global'].dropna().iloc[0], 2)}` "
                f"with compatibility active fraction `{_format_num(main_e4['configured_active_fraction'].dropna().iloc[0], 2)}`, "
                f"but every available `per_layer_phy.csv` stays at nonzero duty `{main_e4['unique_nonzero_duty_cycle'].iloc[0]}` "
                f"with nonzero channels `{main_e4['unique_nonzero_active_channels'].iloc[0]}`."
            ),
            (
                f"- Main-chain `E6` seed runs behave the same way: tau `{_format_num(main_e6['configured_tau_global'].dropna().iloc[0], 2)}` "
                f"with compatibility active fraction `{_format_num(main_e6['configured_active_fraction'].dropna().iloc[0], 2)}`, "
                f"trace duty `{main_e6['unique_nonzero_duty_cycle'].iloc[0]}`, nonzero channels `{main_e6['unique_nonzero_active_channels'].iloc[0]}`."
            ),
            (
                f"- The older subset reference run `{subset_trace['run_id']}` does show sparse gating in trace "
                f"(`mean duty {_format_num(subset_trace['mean_nonzero_duty_cycle'], 3)}`, channels `{subset_trace['unique_nonzero_active_channels']}`), "
                "but that gating is uniform across all nonzero photonic layers rather than layer-conditioned."
            ),
            "- The repository therefore still lacks layer-conditioned tau-governed sparse-duty evidence; the available trace is either uniform across nonzero layers or inconsistent with the configured full-eval main-chain gating.",
            "",
            "Broader-task interaction screen",
            (
                f"- `Pascal VOC` finite mIoU residual `{_format_num(pascal.loc['finite_mean_iou_delta_pp', 'residual_vs_additive'], 2, ' pp')}`; "
                f"global-correct residual `{_format_num(pascal.loc['global_correct_delta_pp', 'residual_vs_additive'], 2, ' pp')}`."
            ),
            (
                f"- `ADE20K` finite mIoU residual `{_format_num(ade.loc['finite_mean_iou_delta_pp', 'residual_vs_additive'], 2, ' pp')}`; "
                f"global-correct residual `{_format_num(ade.loc['global_correct_delta_pp', 'residual_vs_additive'], 2, ' pp')}`."
            ),
            "",
            "Interpretation",
            "- SPARSE independent value is not established on the released full-eval main chain: `E4` keeps the accuracy cost but shows no standalone headline efficiency gain.",
            "- The 4096-image/model sparse-tau sweep provides subset-level measured tradeoff evidence; whether SPARSE retains any usable region or collapses under direct semantics must be read from that measured curve rather than assumed.",
            *_subset_interpretation(subset_df),
            "- Classification and Pascal VOC both close more negatively than additive under `DET+SPARSE`, but ADE20K does not; `SPARSE amplifies DET error` is therefore a plausible but unclosed mechanism rather than a settled claim.",
            "- The manuscript-safe position is `SPARSE as a secondary policy knob`, not `SPARSE as a standalone main contribution`.",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a conservative SPARSE interaction note from released artifacts.")
    parser.add_argument("--pair_ci_csv", type=Path, default=DEFAULT_PAIR_CI_CSV)
    parser.add_argument("--full_eval_per_model_csv", type=Path, default=DEFAULT_FULL_EVAL_PER_MODEL_CSV)
    parser.add_argument("--subset_tau_summary_csv", type=Path, default=DEFAULT_SUBSET_TAU_SUMMARY_CSV)
    parser.add_argument("--pascal_summary_csv", type=Path, default=DEFAULT_PASCAL_SUMMARY_CSV)
    parser.add_argument("--ade_summary_csv", type=Path, default=DEFAULT_ADE_SUMMARY_CSV)
    parser.add_argument("--runs_root", type=Path, default=DEFAULT_RUNS_ROOT)
    parser.add_argument("--out_data_dir", type=Path, default=DEFAULT_OUT_DATA_DIR)
    parser.add_argument("--tag", default=DEFAULT_TAG)
    args = parser.parse_args()

    headline_df = _load_headline_summary(args.pair_ci_csv)
    model_df, mean_model_residual = _load_model_interaction(args.full_eval_per_model_csv)
    subset_df = _load_subset_tradeoff(args.subset_tau_summary_csv)
    segmentation_df = pd.concat(
        [
            _load_segmentation_interaction(args.pascal_summary_csv, "pascalvoc"),
            _load_segmentation_interaction(args.ade_summary_csv, "ade20k"),
        ],
        ignore_index=True,
    )
    trace_df = _build_trace_audit(args.runs_root)
    summary_df = _build_summary(headline_df, model_df, subset_df, segmentation_df, trace_df)

    args.out_data_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = args.out_data_dir / f"sparse_interaction_summary_{args.tag}.csv"
    trace_csv = args.out_data_dir / f"sparse_per_layer_trace_audit_{args.tag}.csv"
    report_md = args.out_data_dir / f"sparse_interaction_report_{args.tag}.md"

    _write_csv(summary_csv, summary_df)
    _write_csv(trace_csv, trace_df)
    _write_report(
        report_md,
        summary_csv,
        trace_csv,
        args.pair_ci_csv,
        args.full_eval_per_model_csv,
        args.subset_tau_summary_csv,
        args.pascal_summary_csv,
        args.ade_summary_csv,
        args.tag,
        headline_df,
        model_df,
        subset_df,
        segmentation_df,
        trace_df,
        mean_model_residual,
    )

    print(f"Wrote summary CSV: {summary_csv}")
    print(f"Wrote trace audit CSV: {trace_csv}")
    print(f"Wrote report: {report_md}")


if __name__ == "__main__":
    main()
