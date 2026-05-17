#!/usr/bin/env python3
"""Run a seed-driven stochastic uncertainty audit for the E0/E6 headline claim.

This script preserves the main deterministic artifact and generates a separate
simulation-only sensitivity analysis over stochastic Top-1 uncertainty scales.
It answers: if we inject seed-dependent accuracy jitter into the simulator, how
stable is the E0 -> E6 accuracy delta?
"""

from __future__ import annotations

import argparse
import copy
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
EXP_ROOT = REPO_ROOT / "experiments"
RUNNER = EXP_ROOT / "tools" / "phase1_runner.py"


def _load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")


def _resolve_path(path_like: str | Path) -> Path:
    raw = Path(path_like)
    if raw.is_absolute():
        return raw
    candidate = REPO_ROOT / raw
    if candidate.exists():
        return candidate
    return raw.resolve()


def _ci95_half(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    arr = np.asarray(values, dtype=float)
    return float(1.96 * arr.std(ddof=1) / math.sqrt(arr.size))


def _prepare_config(
    base_cfg: dict[str, Any],
    *,
    run_id: str,
    seed: int,
    runs_root: Path,
    accuracy_csv: Path,
    std_pp: float,
    noise_scale: float,
    min_std_pp: float,
    max_std_pp: float | None,
    data_split: str,
    eval_manifest: Path | None,
    holdout_manifest: Path | None,
) -> dict[str, Any]:
    cfg = copy.deepcopy(base_cfg)
    run_cfg = cfg.get("run") or {}
    run_cfg["run_id"] = run_id
    run_cfg["seed"] = int(seed)
    run_cfg["device"] = "cpu"
    cfg["run"] = run_cfg

    outputs = cfg.get("outputs") or {}
    outputs["out_dir"] = str(runs_root)
    outputs["append_master"] = False
    outputs["save_layer_tables"] = False
    cfg["outputs"] = outputs

    accuracy = cfg.get("accuracy") or {}
    accuracy["source_csv"] = str(accuracy_csv)
    stoch = accuracy.get("stochastic_uncertainty") or {}
    stoch["enabled"] = True
    stoch["std_pp"] = float(std_pp)
    stoch["noise_scale"] = float(noise_scale)
    stoch["min_std_pp"] = float(min_std_pp)
    stoch["max_std_pp"] = float(max_std_pp) if max_std_pp is not None else None
    accuracy["stochastic_uncertainty"] = stoch
    cfg["accuracy"] = accuracy

    data_cfg = cfg.get("data") or {}
    data_cfg["split"] = data_split
    if eval_manifest is not None:
        data_cfg["eval_manifest_csv"] = str(eval_manifest)
    if holdout_manifest is not None:
        data_cfg["holdout_manifest_csv"] = str(holdout_manifest)
    if data_split == "holdout":
        workload = str(data_cfg.get("workload_id") or "").strip() or "W0_mobilevit_imagenet"
        if not workload.endswith("_holdout"):
            if workload.endswith("_eval"):
                workload = workload[:-5] + "_holdout"
            else:
                workload = workload + "_holdout"
        data_cfg["workload_id"] = workload
    cfg["data"] = data_cfg
    return cfg


def _run_jobs(config_paths: list[Path], python_bin: str) -> None:
    for idx, cfg_path in enumerate(config_paths, 1):
        cmd = [python_bin, str(RUNNER), "--config", str(cfg_path)]
        print(f"[stochastic-audit] ({idx}/{len(config_paths)}) {' '.join(cmd)}", flush=True)
        proc = subprocess.run(cmd, cwd=str(REPO_ROOT), check=False)
        if proc.returncode != 0:
            raise SystemExit(f"phase1_runner failed for {cfg_path} with code {proc.returncode}")


def _read_master(runs_root: Path, run_id: str) -> pd.DataFrame:
    path = runs_root / run_id / "master_metrics.csv"
    if not path.exists():
        raise SystemExit(f"Missing master_metrics.csv: {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise SystemExit(f"Empty master_metrics.csv: {path}")
    return df


def _collect_summary(
    *,
    runs_root: Path,
    run_tag: str,
    std_values: list[float],
    seeds: list[int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pair_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for std_pp in std_values:
        e0_frames: list[pd.DataFrame] = []
        e6_frames: list[pd.DataFrame] = []
        for seed in seeds:
            std_token = f"{std_pp:.3f}".replace(".", "p")
            e0_id = f"{run_tag}_std{std_token}_e0_s{seed}"
            e6_id = f"{run_tag}_std{std_token}_e6_s{seed}"
            e0_frames.append(_read_master(runs_root, e0_id))
            e6_frames.append(_read_master(runs_root, e6_id))

        e0 = pd.concat(e0_frames, ignore_index=True)
        e6 = pd.concat(e6_frames, ignore_index=True)

        e0_model = (
            e0.groupby("model", as_index=False)[["acc_top1", "acc_drop_pp"]]
            .mean()
            .rename(columns={"acc_top1": "e0_acc_top1", "acc_drop_pp": "e0_acc_drop_pp"})
        )
        e6_model = (
            e6.groupby("model", as_index=False)[["acc_top1", "acc_drop_pp"]]
            .mean()
            .rename(columns={"acc_top1": "e6_acc_top1", "acc_drop_pp": "e6_acc_drop_pp"})
        )
        pair = e0_model.merge(e6_model, on="model", how="inner", validate="one_to_one")
        pair["delta_acc_top1_pp"] = pair["e6_acc_top1"] - pair["e0_acc_top1"]
        pair["delta_drop_pp"] = pair["e6_acc_drop_pp"] - pair["e0_acc_drop_pp"]
        pair["std_pp"] = std_pp
        pair_rows.extend(pair.to_dict("records"))

        delta_vals = pair["delta_acc_top1_pp"].astype(float).tolist()
        e0_vals = pair["e0_acc_top1"].astype(float).tolist()
        e6_vals = pair["e6_acc_top1"].astype(float).tolist()
        summary_rows.append(
            {
                "std_pp": std_pp,
                "n_models": int(len(pair)),
                "n_seeds": int(len(seeds)),
                "pair_basis": "model_mean_over_seeds",
                "e0_acc_top1_mean": float(np.mean(e0_vals)),
                "e0_acc_top1_ci95_half": _ci95_half(e0_vals),
                "e6_acc_top1_mean": float(np.mean(e6_vals)),
                "e6_acc_top1_ci95_half": _ci95_half(e6_vals),
                "delta_acc_top1_mean": float(np.mean(delta_vals)),
                "delta_acc_top1_ci95_half": _ci95_half(delta_vals),
                "delta_acc_top1_min": float(np.min(delta_vals)),
                "delta_acc_top1_max": float(np.max(delta_vals)),
            }
        )

    return pd.DataFrame(summary_rows), pd.DataFrame(pair_rows)


def _render_plot(summary_df: pd.DataFrame, out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    x = summary_df["std_pp"].astype(float).to_numpy()
    y = summary_df["delta_acc_top1_mean"].astype(float).to_numpy()
    ci = summary_df["delta_acc_top1_ci95_half"].astype(float).to_numpy()

    ax.axhline(0.0, color="#555555", linestyle="--", linewidth=1.0)
    ax.plot(x, y, marker="o", color="#1f77b4", linewidth=1.8, label="E6 - E0 mean delta")
    ax.fill_between(x, y - ci, y + ci, color="#1f77b4", alpha=0.18, label="95% CI over model pairs")
    ax.set_xlabel("stochastic std_pp")
    ax.set_ylabel("Top-1 delta (pp)")
    ax.set_title("Uncalibrated Stochastic Sensitivity: E0 -> E6")
    ax.legend(frameon=False, loc="best")
    ax.grid(alpha=0.25, linewidth=0.6)
    fig.tight_layout()

    out_base = out_dir / "stochastic_claim_sensitivity"
    fig.savefig(out_base.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return out_base.with_suffix(".pdf")


def _write_report(
    *,
    out_path: Path,
    data_split: str,
    accuracy_csv: Path,
    summary_df: pd.DataFrame,
    pair_df: pd.DataFrame,
    plot_path: Path,
    std_values: list[float],
    seeds: list[int],
) -> None:
    lines: list[str] = []
    lines.append(f"# Stochastic Claim Audit ({data_split})")
    lines.append("")
    lines.append("Setup")
    lines.append(f"- accuracy_csv: `{accuracy_csv}`")
    lines.append(f"- seeds: `{','.join(str(s) for s in seeds)}`")
    lines.append(f"- std_pp grid: `{','.join(f'{v:.3f}' for v in std_values)}`")
    lines.append(f"- plot: `{plot_path}`")
    lines.append("- primary quantity: paired E6-E0 Top-1 delta after averaging each model over seeds")
    lines.append("- note: absolute E0/E6 CI values below reflect cross-model spread, not a hardware-calibrated uncertainty model")
    lines.append("")

    for _, row in summary_df.iterrows():
        lines.append(
            f"- std_pp={float(row['std_pp']):.3f}: "
            f"E0={float(row['e0_acc_top1_mean']):.3f}±{float(row['e0_acc_top1_ci95_half']):.3f}, "
            f"E6={float(row['e6_acc_top1_mean']):.3f}±{float(row['e6_acc_top1_ci95_half']):.3f}, "
            f"E6-E0={float(row['delta_acc_top1_mean']):+.3f}±{float(row['delta_acc_top1_ci95_half']):.3f} pp"
        )
    lines.append("")

    worst = pair_df.assign(abs_delta=lambda d: d["delta_acc_top1_pp"].abs()).sort_values("abs_delta", ascending=False).head(6)
    lines.append("Largest per-model deltas")
    for _, row in worst.iterrows():
        lines.append(
            f"- std_pp={float(row['std_pp']):.3f}, model={row['model']}: "
            f"E6-E0={float(row['delta_acc_top1_pp']):+.3f} pp"
        )

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run stochastic sensitivity audit for E0/E6 headline claim.")
    parser.add_argument(
        "--base_e0_config",
        default="experiments/results/runs/20260228_opt_sync_core_e0/config_snapshot.yaml",
        help="Base E0 config snapshot.",
    )
    parser.add_argument(
        "--base_e6_config",
        default="experiments/results/runs/20260228_opt_sync_core_e6/config_snapshot.yaml",
        help="Base E6 config snapshot.",
    )
    parser.add_argument(
        "--accuracy_csv",
        default="experiments/results/accuracy/accuracy_noise_20260228_opt_cuda.csv",
        help="Accuracy CSV used as the deterministic baseline source.",
    )
    parser.add_argument(
        "--run_tag",
        default="20260306_stochastic_claim_audit_eval",
        help="Output tag under experiments/results/stochastic_audit/.",
    )
    parser.add_argument(
        "--std_grid",
        default="0.05,0.10,0.15,0.20,0.30",
        help="Comma-separated std_pp grid.",
    )
    parser.add_argument(
        "--seeds",
        default="0,1,2",
        help="Comma-separated seed list.",
    )
    parser.add_argument(
        "--noise_scale",
        type=float,
        default=0.0,
        help="Additional std scaling proportional to noise severity.",
    )
    parser.add_argument(
        "--min_std_pp",
        type=float,
        default=0.0,
        help="Minimum stochastic std (pp).",
    )
    parser.add_argument(
        "--max_std_pp",
        type=float,
        default=None,
        help="Optional max stochastic std (pp).",
    )
    parser.add_argument(
        "--data_split",
        default="eval",
        choices=["eval", "holdout"],
        help="Which split label to bind into the configs.",
    )
    parser.add_argument(
        "--eval_manifest",
        default="",
        help="Optional eval manifest override.",
    )
    parser.add_argument(
        "--holdout_manifest",
        default="",
        help="Optional holdout manifest override.",
    )
    parser.add_argument(
        "--python_bin",
        default=sys.executable,
        help="Python interpreter for phase1_runner.py",
    )
    parser.add_argument(
        "--skip_execute",
        action="store_true",
        help="Reuse existing run outputs; do not execute phase1_runner.",
    )
    args = parser.parse_args()

    base_e0_path = _resolve_path(args.base_e0_config)
    base_e6_path = _resolve_path(args.base_e6_config)
    accuracy_csv = _resolve_path(args.accuracy_csv)
    eval_manifest = _resolve_path(args.eval_manifest) if args.eval_manifest else None
    holdout_manifest = _resolve_path(args.holdout_manifest) if args.holdout_manifest else None

    std_values = [float(x.strip()) for x in str(args.std_grid).split(",") if x.strip()]
    seeds = [int(x.strip()) for x in str(args.seeds).split(",") if x.strip()]
    if not std_values:
        raise SystemExit("std_grid must contain at least one value.")
    if not seeds:
        raise SystemExit("seeds must contain at least one value.")

    out_root = EXP_ROOT / "results" / "stochastic_audit" / args.run_tag
    runs_root = out_root / "runs"
    cfg_root = out_root / "generated_configs"
    out_root.mkdir(parents=True, exist_ok=True)
    runs_root.mkdir(parents=True, exist_ok=True)
    cfg_root.mkdir(parents=True, exist_ok=True)

    base_e0 = _load_yaml(base_e0_path)
    base_e6 = _load_yaml(base_e6_path)

    config_paths: list[Path] = []
    for std_pp in std_values:
        std_token = f"{std_pp:.3f}".replace(".", "p")
        for seed in seeds:
            e0_id = f"{args.run_tag}_std{std_token}_e0_s{seed}"
            e6_id = f"{args.run_tag}_std{std_token}_e6_s{seed}"
            e0_cfg = _prepare_config(
                base_e0,
                run_id=e0_id,
                seed=seed,
                runs_root=runs_root,
                accuracy_csv=accuracy_csv,
                std_pp=std_pp,
                noise_scale=args.noise_scale,
                min_std_pp=args.min_std_pp,
                max_std_pp=args.max_std_pp,
                data_split=args.data_split,
                eval_manifest=eval_manifest,
                holdout_manifest=holdout_manifest,
            )
            e6_cfg = _prepare_config(
                base_e6,
                run_id=e6_id,
                seed=seed,
                runs_root=runs_root,
                accuracy_csv=accuracy_csv,
                std_pp=std_pp,
                noise_scale=args.noise_scale,
                min_std_pp=args.min_std_pp,
                max_std_pp=args.max_std_pp,
                data_split=args.data_split,
                eval_manifest=eval_manifest,
                holdout_manifest=holdout_manifest,
            )
            e0_path = cfg_root / f"{e0_id}.yaml"
            e6_path = cfg_root / f"{e6_id}.yaml"
            _write_yaml(e0_path, e0_cfg)
            _write_yaml(e6_path, e6_cfg)
            config_paths.extend([e0_path, e6_path])

    if not args.skip_execute:
        _run_jobs(config_paths, args.python_bin)

    summary_df, pair_df = _collect_summary(
        runs_root=runs_root,
        run_tag=args.run_tag,
        std_values=std_values,
        seeds=seeds,
    )
    summary_df.to_csv(out_root / "stochastic_claim_summary.csv", index=False)
    pair_df.to_csv(out_root / "stochastic_claim_pairs.csv", index=False)

    plot_path = _render_plot(summary_df, out_root)
    _write_report(
        out_path=out_root / "stochastic_claim_report.md",
        data_split=args.data_split,
        accuracy_csv=accuracy_csv,
        summary_df=summary_df,
        pair_df=pair_df,
        plot_path=plot_path,
        std_values=std_values,
        seeds=seeds,
    )
    print(f"[stochastic-audit] completed: {out_root}")


if __name__ == "__main__":
    main()
