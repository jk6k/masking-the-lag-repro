#!/usr/bin/env python3
"""Run a split-aware E0/E6 audit on eval and holdout manifests."""

from __future__ import annotations

import argparse
import copy
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

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


def _infer_split(workload: str) -> str:
    value = str(workload or "").strip().lower()
    if value.endswith("_holdout"):
        return "holdout"
    if value.endswith("_eval"):
        return "eval"
    return "unknown"


def _inspect_accuracy_source(path: Path) -> dict[str, Any]:
    df = pd.read_csv(path)
    if "workload" not in df.columns:
        raise SystemExit(f"Missing workload column in accuracy source: {path}")
    split_series = df["workload"].map(_infer_split)
    df = df.assign(split=split_series)
    experiment_ids = sorted({str(item) for item in df.get("experiment_id", []) if str(item).strip()})
    split_experiment_ids: dict[str, list[str]] = {}
    split_note_counts: list[dict[str, Any]] = []
    for split, split_df in df.groupby("split", sort=True):
        split_experiment_ids[str(split)] = sorted(
            {str(item) for item in split_df.get("experiment_id", []) if str(item).strip()}
        )
        if "notes" in split_df.columns:
            grouped = (
                split_df.groupby("notes", dropna=False)
                .size()
                .reset_index(name="row_count")
            )
            for _, row in grouped.iterrows():
                split_note_counts.append(
                    {
                        "split": str(split),
                        "notes": str(row["notes"]),
                        "row_count": int(row["row_count"]),
                    }
                )
    return {
        "experiment_ids": experiment_ids,
        "has_independent_e6_rows": "E6" in experiment_ids,
        "split_experiment_ids": split_experiment_ids,
        "split_note_counts": split_note_counts,
    }


def _missing_required_experiments(
    source_info: dict[str, Any],
    *,
    splits: list[str],
    required_experiments: set[str],
) -> dict[str, list[str]]:
    split_ids = source_info.get("split_experiment_ids") or {}
    missing: dict[str, list[str]] = {}
    for split in splits:
        observed = {str(item) for item in split_ids.get(split, [])}
        missing_items = sorted(required_experiments - observed)
        if missing_items:
            missing[split] = missing_items
    return missing


def _write_blocker_report(
    *,
    out_path: Path,
    accuracy_csv: Path,
    eval_manifest: Path,
    holdout_manifest: Path,
    source_info: dict[str, Any],
    missing_by_split: dict[str, list[str]],
) -> None:
    lines = [
        "# Holdout Claim Audit Blocker",
        "",
        f"- accuracy_csv: `{accuracy_csv}`",
        f"- eval_manifest: `{eval_manifest}`",
        f"- holdout_manifest: `{holdout_manifest}`",
        "",
        "Missing required experiments by split",
    ]
    for split, missing in sorted(missing_by_split.items()):
        lines.append(f"- split={split}: missing {', '.join(missing)}")
    lines.append("")
    lines.append("Observed source rows")
    for row in source_info.get("split_note_counts") or []:
        lines.append(
            f"- split={row['split']}, notes={row['notes']}, row_count={row['row_count']}"
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _build_workload_id(base_cfg: dict[str, Any], split: str) -> str:
    data_cfg = base_cfg.get("data") or {}
    workload = str(data_cfg.get("workload_id") or data_cfg.get("workload") or "").strip()
    if not workload:
        return f"W0_mobilevit_imagenet_{split}"
    if workload.endswith("_eval") or workload.endswith("_holdout"):
        head = workload.rsplit("_", 1)[0]
        return f"{head}_{split}"
    return f"{workload}_{split}"


def _prepare_config(
    base_cfg: dict[str, Any],
    *,
    run_id: str,
    seed: int,
    split: str,
    runs_root: Path,
    accuracy_csv: Path,
    eval_manifest: Path,
    holdout_manifest: Path,
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
    cfg["accuracy"] = accuracy

    data_cfg = cfg.get("data") or {}
    data_cfg["split"] = split
    data_cfg["eval_manifest_csv"] = str(eval_manifest)
    data_cfg["holdout_manifest_csv"] = str(holdout_manifest)
    data_cfg["workload_id"] = _build_workload_id(cfg, split)
    cfg["data"] = data_cfg
    return cfg


def _run_jobs(config_paths: list[Path], python_bin: str) -> None:
    for idx, cfg_path in enumerate(config_paths, 1):
        cmd = [python_bin, str(RUNNER), "--config", str(cfg_path)]
        print(f"[holdout-claim] ({idx}/{len(config_paths)}) {' '.join(cmd)}", flush=True)
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
    splits: list[str],
    seeds: list[int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pair_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for split in splits:
        e0_frames: list[pd.DataFrame] = []
        e6_frames: list[pd.DataFrame] = []
        for seed in seeds:
            e0_id = f"{run_tag}_{split}_e0_s{seed}"
            e6_id = f"{run_tag}_{split}_e6_s{seed}"
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
        pair["split"] = split
        pair_rows.extend(pair.to_dict("records"))

        delta_vals = pair["delta_acc_top1_pp"].astype(float).tolist()
        e0_vals = pair["e0_acc_top1"].astype(float).tolist()
        e6_vals = pair["e6_acc_top1"].astype(float).tolist()
        summary_rows.append(
            {
                "split": split,
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


def _write_report(
    *,
    out_path: Path,
    accuracy_csv: Path,
    eval_manifest: Path,
    holdout_manifest: Path,
    summary_df: pd.DataFrame,
    pair_df: pd.DataFrame,
    seeds: list[int],
) -> None:
    lines: list[str] = []
    lines.append("# Holdout Claim Audit")
    lines.append("")
    lines.append("Setup")
    lines.append(f"- accuracy_csv: `{accuracy_csv}`")
    lines.append(f"- eval_manifest: `{eval_manifest}`")
    lines.append(f"- holdout_manifest: `{holdout_manifest}`")
    lines.append(f"- seeds: `{','.join(str(s) for s in seeds)}`")
    lines.append("- primary quantity: paired E6-E0 Top-1 delta after averaging each model over seeds")
    lines.append("- note: E0 and E6 share the same split-matched measured source rows here, so zero delta verifies split routing rather than hardware-grounded generalization")
    lines.append("")

    for _, row in summary_df.iterrows():
        lines.append(
            f"- split={row['split']}: "
            f"E0={float(row['e0_acc_top1_mean']):.3f}, "
            f"E6={float(row['e6_acc_top1_mean']):.3f}, "
            f"E6-E0={float(row['delta_acc_top1_mean']):+.3f}±{float(row['delta_acc_top1_ci95_half']):.3f} pp"
        )
    lines.append("")

    worst = pair_df.assign(abs_delta=lambda d: d["delta_acc_top1_pp"].abs()).sort_values("abs_delta", ascending=False).head(8)
    lines.append("Largest per-model deltas")
    for _, row in worst.iterrows():
        lines.append(
            f"- split={row['split']}, model={row['model']}: "
            f"E6-E0={float(row['delta_acc_top1_pp']):+.3f} pp"
        )

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run eval/holdout E0/E6 claim audit from existing accuracy CSVs.")
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
        default="experiments/results/accuracy/accuracy_noise_20260305_stage2_holdout_uncert_cuda.csv",
        help="Accuracy CSV containing eval/holdout rows.",
    )
    parser.add_argument(
        "--eval_manifest",
        default="experiments/results/accuracy/splits_smoke_holdout_uncert_20260305/imagenet_val_eval.csv",
        help="Eval manifest used for split matching.",
    )
    parser.add_argument(
        "--holdout_manifest",
        default="experiments/results/accuracy/splits_smoke_holdout_uncert_20260305/imagenet_val_holdout.csv",
        help="Holdout manifest used for split matching.",
    )
    parser.add_argument(
        "--splits",
        default="eval,holdout",
        help="Comma-separated split list.",
    )
    parser.add_argument(
        "--run_tag",
        default="20260306_holdout_claim_audit",
        help="Output tag under experiments/results/holdout_audit/.",
    )
    parser.add_argument(
        "--seeds",
        default="0,1,2",
        help="Comma-separated seed list.",
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
    eval_manifest = _resolve_path(args.eval_manifest)
    holdout_manifest = _resolve_path(args.holdout_manifest)

    splits = [part.strip() for part in str(args.splits).split(",") if part.strip()]
    seeds = [int(part.strip()) for part in str(args.seeds).split(",") if part.strip()]
    if not splits:
        raise SystemExit("At least one split is required.")
    if not seeds:
        raise SystemExit("At least one seed is required.")

    out_root = EXP_ROOT / "results" / "holdout_audit" / args.run_tag
    runs_root = out_root / "runs"
    cfg_root = out_root / "generated_configs"
    out_root.mkdir(parents=True, exist_ok=True)
    runs_root.mkdir(parents=True, exist_ok=True)
    cfg_root.mkdir(parents=True, exist_ok=True)

    base_e0 = _load_yaml(base_e0_path)
    base_e6 = _load_yaml(base_e6_path)

    config_paths: list[Path] = []
    for split in splits:
        for seed in seeds:
            e0_id = f"{args.run_tag}_{split}_e0_s{seed}"
            e6_id = f"{args.run_tag}_{split}_e6_s{seed}"
            e0_cfg = _prepare_config(
                base_e0,
                run_id=e0_id,
                seed=seed,
                split=split,
                runs_root=runs_root,
                accuracy_csv=accuracy_csv,
                eval_manifest=eval_manifest,
                holdout_manifest=holdout_manifest,
            )
            e6_cfg = _prepare_config(
                base_e6,
                run_id=e6_id,
                seed=seed,
                split=split,
                runs_root=runs_root,
                accuracy_csv=accuracy_csv,
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
        splits=splits,
        seeds=seeds,
    )
    summary_df.to_csv(out_root / "holdout_claim_summary.csv", index=False)
    pair_df.to_csv(out_root / "holdout_claim_pairs.csv", index=False)
    _write_report(
        out_path=out_root / "holdout_claim_report.md",
        accuracy_csv=accuracy_csv,
        eval_manifest=eval_manifest,
        holdout_manifest=holdout_manifest,
        summary_df=summary_df,
        pair_df=pair_df,
        seeds=seeds,
    )
    print(f"[holdout-claim] completed: {out_root}")


if __name__ == "__main__":
    main()
