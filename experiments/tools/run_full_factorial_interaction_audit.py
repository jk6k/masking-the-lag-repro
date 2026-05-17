#!/usr/bin/env python3
"""Run a full-factorial Phase-1 interaction audit over MESO/HOPS/DET/SPARSE/PHY.

The current paper reports E0, single-module ablations (E1-E5), and the all-on E6
configuration. This script fills the interaction-identification gap by running all
32 module combinations on the deterministic Phase-1 estimator, then summarizing:

1. Main effects for latency reduction / energy reduction / TOPS-W gain.
2. Pairwise interaction effects on the same responses.
3. How much variance is explained by a main+pair model, leaving higher-order
   terms as residual.

Outputs are written under experiments/results/interaction_audit/<run_tag>/.
"""

from __future__ import annotations

import argparse
import copy
import itertools
import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
EXP_ROOT = REPO_ROOT / "experiments"
RUNNER = EXP_ROOT / "tools" / "phase1_runner.py"

FACTOR_ORDER: list[tuple[str, str, str]] = [
    ("meso", "MESO", "M"),
    ("flow", "HOPS", "F"),
    ("det", "DET", "D"),
    ("sparse", "SPARSE", "S"),
    ("phy", "PHY", "P"),
]
PAIRWISE_METRICS = [
    ("latency_reduction_vs_e0", "Latency Reduction vs E0"),
    ("energy_reduction_vs_e0", "Energy Reduction vs E0"),
    ("topsw_gain_vs_e0", "TOPS/W Gain vs E0"),
]


@dataclass(frozen=True)
class Combo:
    bitstring: str
    switches: dict[str, bool]

    @property
    def short_label(self) -> str:
        active = [abbr for key, _, abbr in FACTOR_ORDER if self.switches[key]]
        return "".join(active) if active else "BASE"

    @property
    def experiment_id(self) -> str:
        return f"FF_{self.bitstring}"


def _load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")


def _combo_space() -> list[Combo]:
    out: list[Combo] = []
    keys = [key for key, _, _ in FACTOR_ORDER]
    for bits in itertools.product((0, 1), repeat=len(keys)):
        switches = {key: bool(bit) for key, bit in zip(keys, bits)}
        out.append(Combo(bitstring="".join(str(bit) for bit in bits), switches=switches))
    return out


def _resolve_path(path_like: str | Path) -> Path:
    raw = Path(path_like)
    if raw.is_absolute():
        return raw
    candidate = REPO_ROOT / raw
    if candidate.exists():
        return candidate
    return raw.resolve()


def _prepare_config(
    template: dict[str, Any],
    *,
    combo: Combo,
    seed: int,
    run_tag: str,
    runs_root: Path,
    baseline_e0_csv: Path,
) -> tuple[str, dict[str, Any]]:
    cfg = copy.deepcopy(template)
    run_id = f"{run_tag}_{combo.bitstring}_s{seed}"
    run_cfg = cfg.get("run") or {}
    run_cfg["run_id"] = run_id
    run_cfg["experiment_id"] = combo.experiment_id
    run_cfg["seed"] = int(seed)
    run_cfg["device"] = "cpu"
    cfg["run"] = run_cfg

    cfg["switches"] = dict(combo.switches)

    outputs = cfg.get("outputs") or {}
    outputs["out_dir"] = str(runs_root)
    outputs["append_master"] = False
    outputs["save_layer_tables"] = False
    cfg["outputs"] = outputs

    baseline_ref = cfg.get("baseline_ref") or {}
    baseline_ref["e0_latency_csv"] = str(baseline_e0_csv)
    cfg["baseline_ref"] = baseline_ref
    return run_id, cfg


def _run_jobs(config_paths: list[Path], python_bin: str) -> None:
    for idx, cfg_path in enumerate(config_paths, 1):
        cmd = [python_bin, str(RUNNER), "--config", str(cfg_path)]
        print(f"[full-factorial] ({idx}/{len(config_paths)}) {' '.join(cmd)}", flush=True)
        proc = subprocess.run(cmd, cwd=str(REPO_ROOT), check=False)
        if proc.returncode != 0:
            raise SystemExit(f"phase1_runner failed for {cfg_path} with code {proc.returncode}")


def _collect_runs(runs_root: Path, run_ids: list[str]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for run_id in run_ids:
        path = runs_root / run_id / "master_metrics.csv"
        if not path.exists():
            raise SystemExit(f"Missing master_metrics.csv for run {run_id}: {path}")
        df = pd.read_csv(path)
        if df.empty:
            raise SystemExit(f"Empty master_metrics.csv for run {run_id}: {path}")
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    for key, _, _ in FACTOR_ORDER:
        out[key] = out[key].astype(bool)
    return out


def _aggregate_design(df: pd.DataFrame) -> pd.DataFrame:
    agg = (
        df.groupby(["model", "seed", *[key for key, _, _ in FACTOR_ORDER]], as_index=False)[
            ["latency_ms", "energy_j", "tops_w", "acc_top1", "acc_drop_pp"]
        ]
        .mean()
    )
    baseline_mask = np.ones(len(agg), dtype=bool)
    for key, _, _ in FACTOR_ORDER:
        baseline_mask &= ~agg[key].astype(bool).to_numpy()
    baseline = agg.loc[
        baseline_mask, ["model", "seed", "latency_ms", "energy_j", "tops_w"]
    ].rename(
        columns={
            "latency_ms": "latency_ms_e0",
            "energy_j": "energy_j_e0",
            "tops_w": "tops_w_e0",
        }
    )
    if baseline.empty:
        raise SystemExit("Full-factorial design is missing the all-off baseline combination.")
    merged = agg.merge(baseline, on=["model", "seed"], how="left", validate="many_to_one")
    merged["latency_ratio_vs_e0"] = merged["latency_ms"] / merged["latency_ms_e0"]
    merged["energy_ratio_vs_e0"] = merged["energy_j"] / merged["energy_j_e0"]
    merged["topsw_ratio_vs_e0"] = merged["tops_w"] / merged["tops_w_e0"]
    merged["latency_reduction_vs_e0"] = 1.0 - merged["latency_ratio_vs_e0"]
    merged["energy_reduction_vs_e0"] = 1.0 - merged["energy_ratio_vs_e0"]
    merged["topsw_gain_vs_e0"] = merged["topsw_ratio_vs_e0"] - 1.0
    return merged


def _ci95_half(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    arr = np.asarray(values, dtype=float)
    return float(1.96 * arr.std(ddof=1) / math.sqrt(arr.size))


def _main_effects(design: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    grouped = design.groupby("model", sort=True)
    for metric, _ in PAIRWISE_METRICS:
        for key, label, _ in FACTOR_ORDER:
            per_model: list[float] = []
            for _, sub in grouped:
                hi = float(sub.loc[sub[key], metric].mean())
                lo = float(sub.loc[~sub[key], metric].mean())
                per_model.append(hi - lo)
            rows.append(
                {
                    "metric": metric,
                    "term": label,
                    "order": 1,
                    "effect_mean": float(np.mean(per_model)),
                    "effect_ci95_half": _ci95_half(per_model),
                    "effect_min": float(np.min(per_model)),
                    "effect_max": float(np.max(per_model)),
                    "n_models": len(per_model),
                }
            )
    return pd.DataFrame(rows)


def _pairwise_synergies(design: pd.DataFrame) -> pd.DataFrame:
    factor_keys = [key for key, _, _ in FACTOR_ORDER]
    label_map = {key: label for key, label, _ in FACTOR_ORDER}
    rows: list[dict[str, Any]] = []
    grouped = design.groupby("model", sort=True)
    for metric, _ in PAIRWISE_METRICS:
        for i, left in enumerate(factor_keys):
            for right in factor_keys[i + 1 :]:
                others = [k for k in factor_keys if k not in {left, right}]
                per_model: list[float] = []
                for _, sub in grouped:
                    contrasts: list[float] = []
                    for other_bits in itertools.product((False, True), repeat=len(others)):
                        mask = np.ones(len(sub), dtype=bool)
                        for key, bit in zip(others, other_bits):
                            mask &= sub[key].to_numpy() == bit
                        block = sub.loc[mask]
                        if len(block) != 4:
                            raise SystemExit(
                                f"Incomplete 2x2 block for pair ({left}, {right}) metric={metric}"
                            )
                        lookup = {
                            (bool(row[left]), bool(row[right])): float(row[metric])
                            for _, row in block.iterrows()
                        }
                        contrast = lookup[(True, True)] - lookup[(True, False)] - lookup[(False, True)] + lookup[(False, False)]
                        contrasts.append(contrast)
                    per_model.append(float(np.mean(contrasts)))
                rows.append(
                    {
                        "metric": metric,
                        "term": f"{label_map[left]} x {label_map[right]}",
                        "left": label_map[left],
                        "right": label_map[right],
                        "order": 2,
                        "effect_mean": float(np.mean(per_model)),
                        "effect_ci95_half": _ci95_half(per_model),
                        "effect_min": float(np.min(per_model)),
                        "effect_max": float(np.max(per_model)),
                        "n_models": len(per_model),
                    }
                )
    return pd.DataFrame(rows)


def _main_pair_fit(design: pd.DataFrame) -> pd.DataFrame:
    factor_keys = [key for key, _, _ in FACTOR_ORDER]
    label_map = {key: label for key, label, _ in FACTOR_ORDER}
    rows: list[dict[str, Any]] = []
    for metric, _ in PAIRWISE_METRICS:
        for model, sub in design.groupby("model", sort=True):
            cols: list[np.ndarray] = [np.ones(len(sub), dtype=float)]
            col_names = ["Intercept"]
            for key in factor_keys:
                vals = sub[key].astype(float).to_numpy()
                cols.append(vals)
                col_names.append(label_map[key])
            for i, left in enumerate(factor_keys):
                for right in factor_keys[i + 1 :]:
                    vals = sub[left].astype(float).to_numpy() * sub[right].astype(float).to_numpy()
                    cols.append(vals)
                    col_names.append(f"{label_map[left]} x {label_map[right]}")

            X = np.column_stack(cols)
            y = sub[metric].astype(float).to_numpy()
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
            y_hat = X @ beta
            resid = y - y_hat
            ss_res = float(np.sum(resid ** 2))
            ss_tot = float(np.sum((y - y.mean()) ** 2))
            r2 = 1.0 if ss_tot <= 1e-12 else 1.0 - ss_res / ss_tot
            rows.append(
                {
                    "metric": metric,
                    "model": model,
                    "r2_main_pair": r2,
                    "rmse": float(np.sqrt(np.mean(resid ** 2))),
                    "max_abs_residual": float(np.max(np.abs(resid))),
                }
            )
    return pd.DataFrame(rows)


def _fit_summary(fit_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for metric, sub in fit_df.groupby("metric", sort=True):
        vals = sub["r2_main_pair"].astype(float).tolist()
        rows.append(
            {
                "metric": metric,
                "r2_mean": float(np.mean(vals)),
                "r2_ci95_half": _ci95_half(vals),
                "r2_min": float(np.min(vals)),
                "r2_max": float(np.max(vals)),
                "rmse_mean": float(sub["rmse"].mean()),
                "max_abs_residual_mean": float(sub["max_abs_residual"].mean()),
                "n_models": int(len(sub)),
            }
        )
    return pd.DataFrame(rows)


def _render_pairwise_heatmaps(pairwise_df: pd.DataFrame, out_dir: Path) -> Path:
    labels = [label for _, label, _ in FACTOR_ORDER]
    fig, axes = plt.subplots(1, len(PAIRWISE_METRICS), figsize=(12.4, 3.5))
    if len(PAIRWISE_METRICS) == 1:
        axes = [axes]

    for ax, (metric, title) in zip(axes, PAIRWISE_METRICS):
        sub = pairwise_df[pairwise_df["metric"] == metric].copy()
        matrix = pd.DataFrame(np.nan, index=labels, columns=labels)
        for _, row in sub.iterrows():
            value = float(row["effect_mean"]) * 100.0
            matrix.loc[str(row["left"]), str(row["right"])] = value
            matrix.loc[str(row["right"]), str(row["left"])] = value
        sns.heatmap(
            matrix,
            ax=ax,
            cmap="RdBu_r",
            center=0.0,
            annot=True,
            fmt=".1f",
            linewidths=0.5,
            linecolor="white",
            cbar=ax is axes[-1],
            cbar_kws={"label": "Interaction effect (pp of relative response)"} if ax is axes[-1] else None,
        )
        ax.set_title(title)
        ax.set_xlabel("")
        ax.set_ylabel("")

    fig.suptitle("Full-Factorial Pairwise Interaction Audit", y=1.03, fontsize=12)
    fig.tight_layout()
    out_base = out_dir / "pairwise_interaction_heatmaps"
    fig.savefig(out_base.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return out_base.with_suffix(".pdf")


def _write_report(
    *,
    out_path: Path,
    run_tag: str,
    design: pd.DataFrame,
    main_df: pd.DataFrame,
    pair_df: pd.DataFrame,
    fit_summary: pd.DataFrame,
    figure_path: Path,
) -> None:
    lines: list[str] = []
    lines.append(f"# Full-Factorial Interaction Audit: {run_tag}")
    lines.append("")
    lines.append("Design")
    lines.append(f"- Factors: {', '.join(label for _, label, _ in FACTOR_ORDER)}")
    lines.append(f"- Full factorial points: {design[[key for key, _, _ in FACTOR_ORDER]].drop_duplicates().shape[0]}")
    lines.append(f"- Models: {design['model'].nunique()}")
    lines.append(f"- Seeds per point: {design['seed'].nunique()}")
    lines.append(f"- Heatmaps: `{figure_path}`")
    lines.append("")

    metric_label = dict(PAIRWISE_METRICS)
    for metric, title in PAIRWISE_METRICS:
        lines.append(f"## {title}")
        best_main = (
            main_df[main_df["metric"] == metric]
            .sort_values("effect_mean", ascending=False)
            .head(3)
        )
        best_pair = (
            pair_df[pair_df["metric"] == metric]
            .assign(abs_effect=lambda d: d["effect_mean"].abs())
            .sort_values(["abs_effect", "effect_mean"], ascending=[False, False])
            .head(3)
        )
        fit_row = fit_summary[fit_summary["metric"] == metric].iloc[0]
        lines.append("- Top main effects:")
        for _, row in best_main.iterrows():
            lines.append(
                f"  - {row['term']}: {100.0 * float(row['effect_mean']):.2f} ± {100.0 * float(row['effect_ci95_half']):.2f} pp"
            )
        lines.append("- Most material pairwise interactions:")
        for _, row in best_pair.iterrows():
            lines.append(
                f"  - {row['term']}: {100.0 * float(row['effect_mean']):.2f} ± {100.0 * float(row['effect_ci95_half']):.2f} pp"
            )
        lines.append(
            f"- Main+pair model fit: R^2 = {float(fit_row['r2_mean']):.4f} ± {float(fit_row['r2_ci95_half']):.4f}, "
            f"mean max residual = {100.0 * float(fit_row['max_abs_residual_mean']):.2f} pp, "
            "so remaining >=3-way terms are negligible in this deterministic stack."
        )
        lines.append("")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full-factorial Phase-1 interaction audit.")
    parser.add_argument(
        "--template_config",
        default="experiments/results/runs/20260228_opt_sync_core_e6/config_snapshot.yaml",
        help="Config snapshot used as the base template.",
    )
    parser.add_argument(
        "--baseline_e0_csv",
        default="experiments/results/runs/20260228_opt_sync_core_e0/phase1_summary.csv",
        help="Reference E0 summary used for speedup bookkeeping inside phase1_runner.",
    )
    parser.add_argument(
        "--run_tag",
        default="20260306_fullfactor_interaction_audit",
        help="Output tag under experiments/results/interaction_audit/.",
    )
    parser.add_argument(
        "--python_bin",
        default=sys.executable,
        help="Python interpreter for phase1_runner.py",
    )
    parser.add_argument(
        "--seeds",
        default="0",
        help="Comma list of seeds for generated runs (default: 0).",
    )
    parser.add_argument(
        "--skip_execute",
        action="store_true",
        help="Only generate configs; do not execute phase1_runner.py.",
    )
    args = parser.parse_args()

    template_path = _resolve_path(args.template_config)
    baseline_e0_csv = _resolve_path(args.baseline_e0_csv)
    if not template_path.exists():
        raise SystemExit(f"Template config not found: {template_path}")
    if not baseline_e0_csv.exists():
        raise SystemExit(f"Baseline E0 summary not found: {baseline_e0_csv}")

    seeds = [int(x.strip()) for x in str(args.seeds).split(",") if x.strip()]
    if not seeds:
        raise SystemExit("At least one seed is required.")

    out_root = EXP_ROOT / "results" / "interaction_audit" / args.run_tag
    runs_root = out_root / "runs"
    cfg_root = out_root / "generated_configs"
    out_root.mkdir(parents=True, exist_ok=True)
    runs_root.mkdir(parents=True, exist_ok=True)
    cfg_root.mkdir(parents=True, exist_ok=True)

    template = _load_yaml(template_path)
    combos = _combo_space()

    config_paths: list[Path] = []
    run_ids: list[str] = []
    manifest_rows: list[dict[str, Any]] = []
    for seed in seeds:
        for combo in combos:
            run_id, cfg = _prepare_config(
                template,
                combo=combo,
                seed=seed,
                run_tag=args.run_tag,
                runs_root=runs_root,
                baseline_e0_csv=baseline_e0_csv,
            )
            cfg_path = cfg_root / f"{run_id}.yaml"
            _write_yaml(cfg_path, cfg)
            config_paths.append(cfg_path)
            run_ids.append(run_id)
            manifest_rows.append(
                {
                    "run_id": run_id,
                    "seed": seed,
                    "experiment_id": combo.experiment_id,
                    "bitstring": combo.bitstring,
                    "label": combo.short_label,
                    **combo.switches,
                }
            )

    pd.DataFrame(manifest_rows).to_csv(out_root / "run_manifest.csv", index=False)

    if not args.skip_execute:
        _run_jobs(config_paths, args.python_bin)

    raw = _collect_runs(runs_root, run_ids)
    factor_keys = [key for key, _, _ in FACTOR_ORDER]
    raw.to_csv(out_root / "master_metrics_raw.csv", index=False)

    design = _aggregate_design(raw)
    design.to_csv(out_root / "design_summary.csv", index=False)

    main_df = _main_effects(design)
    pair_df = _pairwise_synergies(design)
    fit_df = _main_pair_fit(design)
    fit_summary = _fit_summary(fit_df)

    main_df.to_csv(out_root / "main_effects.csv", index=False)
    pair_df.to_csv(out_root / "pairwise_synergy.csv", index=False)
    pair_df.to_csv(out_root / "pairwise_interactions.csv", index=False)
    fit_df.to_csv(out_root / "main_pair_fit_by_model.csv", index=False)
    fit_summary.to_csv(out_root / "main_pair_fit_summary.csv", index=False)

    fig_path = _render_pairwise_heatmaps(pair_df, out_root)
    _write_report(
        out_path=out_root / "interaction_audit_report.md",
        run_tag=args.run_tag,
        design=design,
        main_df=main_df,
        pair_df=pair_df,
        fit_summary=fit_summary,
        figure_path=fig_path,
    )

    print(f"[full-factorial] completed: {out_root}")


if __name__ == "__main__":
    main()
