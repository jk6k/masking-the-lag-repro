#!/usr/bin/env python3
"""Build a held-out PHY validation report from nominal calibration and robust runs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

try:
    from experiments.tools.path_policy import MAIN_PROJECT_REPORT_DATA_DIR
except ModuleNotFoundError:  # direct script execution
    from path_policy import MAIN_PROJECT_REPORT_DATA_DIR


ROOT = Path(__file__).resolve().parents[2]
RUNS_ROOT = ROOT / "experiments" / "results" / "runs"
DEFAULT_ACCURACY_CSV = ROOT / "experiments" / "results" / "accuracy" / "accuracy_noise_20260228_opt_cuda.csv"
DEFAULT_ALIGNMENT_SUMMARY_CSV = (
    ROOT / "experiments" / "results" / "calibration_audit" / "20260306_calibration_traceability" / "calibration_alignment_summary.csv"
)
DEFAULT_OUT_DIR = MAIN_PROJECT_REPORT_DATA_DIR
DEFAULT_TAG = "20260310"
DEFAULT_VALIDATION_RUN_IDS = [
    "20260228_opt_noncuda_robust_e5_light",
    "20260228_opt_noncuda_robust_e5_moderate",
    "20260228_opt_noncuda_robust_e5_stress",
]
DEFAULT_REPLICATION_RUN_IDS = [
    "20260228_opt_noncuda_robust_e6_light",
    "20260228_opt_noncuda_robust_e6_moderate",
    "20260228_opt_noncuda_robust_e6_stress",
]


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        text = str(value).strip()
        return float(text) if text else None
    except (TypeError, ValueError):
        return None


def _fmt(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def _profile_from_run_id(run_id: str) -> str:
    return run_id.rsplit("_", 1)[-1]


def _build_nominal_grid(accuracy_csv: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(accuracy_csv)
    calib_mask = df["workload"].astype(str).str.contains("calib", case=False, na=False)
    quant_mask = df["quant_bits"].astype(float) == 8.0
    calib = df.loc[calib_mask & quant_mask].copy()
    calib["acc_drop_pp_nominal"] = -calib["top1_delta"].astype(float)

    model_points = (
        calib.groupby(["model", "noise_sigma_lsb", "crosstalk_alpha"], as_index=False)["acc_drop_pp_nominal"]
        .mean()
        .sort_values(["model", "noise_sigma_lsb", "crosstalk_alpha"])
    )
    profile_points = (
        model_points.groupby(["noise_sigma_lsb", "crosstalk_alpha"], as_index=False)["acc_drop_pp_nominal"]
        .mean()
        .sort_values(["noise_sigma_lsb", "crosstalk_alpha"])
    )
    return model_points, profile_points


def _build_validation_rows(
    *,
    run_ids: list[str],
    model_grid: pd.DataFrame,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for run_id in run_ids:
        run_dir = RUNS_ROOT / run_id
        cfg = _load_yaml(run_dir / "config_snapshot.yaml")
        injected = cfg.get("noise_injection") or {}
        sigma = float(injected["sigma_lsb"])
        alpha = float(injected["crosstalk_alpha"])
        drift = float(injected.get("drift_lsb", 0.0))
        corr = float(injected.get("noise_correlation", 0.0))
        burst = float(injected.get("burst_error_prob", 0.0))

        master = pd.read_csv(run_dir / "master_metrics.csv")
        for _, metric in master.iterrows():
            model = str(metric["model"])
            nominal = model_grid[
                (model_grid["model"] == model)
                & (model_grid["noise_sigma_lsb"].astype(float) == sigma)
                & (model_grid["crosstalk_alpha"].astype(float) == alpha)
            ]
            if nominal.empty:
                raise SystemExit(
                    f"Missing nominal calibration point for model={model}, sigma={sigma}, alpha={alpha} used by {run_id}"
                )
            nominal_drop = float(nominal.iloc[0]["acc_drop_pp_nominal"])
            actual_drop = float(metric["acc_drop_pp"])
            residual = actual_drop - nominal_drop
            rows.append(
                {
                    "run_id": run_id,
                    "profile": _profile_from_run_id(run_id),
                    "experiment_id": str(metric.get("experiment_id", "")),
                    "model": model,
                    "sigma_lsb": sigma,
                    "crosstalk_alpha": alpha,
                    "drift_lsb": drift,
                    "noise_correlation": corr,
                    "burst_error_prob": burst,
                    "nominal_grid_acc_drop_pp": nominal_drop,
                    "heldout_acc_drop_pp": actual_drop,
                    "extra_drop_vs_nominal_pp": residual,
                    "abs_extra_drop_pp": abs(residual),
                    "p1_align_fit_error_pp": _to_float(metric.get("p1_align_fit_error")),
                    "p1_align_method": str(metric.get("p1_align_method", "")),
                }
            )
    return rows


def _build_profile_summary(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    df = pd.DataFrame(rows)
    grouped = df.groupby("profile", as_index=False).agg(
        sigma_lsb=("sigma_lsb", "first"),
        crosstalk_alpha=("crosstalk_alpha", "first"),
        drift_lsb=("drift_lsb", "first"),
        noise_correlation=("noise_correlation", "first"),
        burst_error_prob=("burst_error_prob", "first"),
        n_models=("model", "nunique"),
        nominal_grid_acc_drop_pp_mean=("nominal_grid_acc_drop_pp", "mean"),
        heldout_acc_drop_pp_mean=("heldout_acc_drop_pp", "mean"),
        extra_drop_vs_nominal_pp_mean=("extra_drop_vs_nominal_pp", "mean"),
        abs_extra_drop_pp_mean=("abs_extra_drop_pp", "mean"),
        max_abs_extra_drop_pp=("abs_extra_drop_pp", "max"),
    )
    return grouped.sort_values("sigma_lsb").to_dict("records")


def _build_replication_rows(primary_run_ids: list[str], replication_run_ids: list[str]) -> list[dict[str, object]]:
    if not replication_run_ids:
        return []
    primary_map = {_profile_from_run_id(run_id): run_id for run_id in primary_run_ids}
    replication_map = {_profile_from_run_id(run_id): run_id for run_id in replication_run_ids}
    rows: list[dict[str, object]] = []
    for profile, primary_run_id in sorted(primary_map.items()):
        secondary_run_id = replication_map.get(profile)
        if secondary_run_id is None:
            continue
        primary = pd.read_csv(RUNS_ROOT / primary_run_id / "master_metrics.csv")
        secondary = pd.read_csv(RUNS_ROOT / secondary_run_id / "master_metrics.csv")
        merged = primary[["model", "acc_drop_pp"]].merge(
            secondary[["model", "acc_drop_pp"]],
            on="model",
            suffixes=("_primary", "_replica"),
        )
        merged["delta_pp"] = merged["acc_drop_pp_replica"] - merged["acc_drop_pp_primary"]
        rows.append(
            {
                "profile": profile,
                "primary_run_id": primary_run_id,
                "replication_run_id": secondary_run_id,
                "mean_acc_drop_pp_primary": float(merged["acc_drop_pp_primary"].mean()),
                "mean_acc_drop_pp_replica": float(merged["acc_drop_pp_replica"].mean()),
                "max_model_delta_pp": float(merged["delta_pp"].abs().max()),
            }
        )
    return rows


def _write_report(
    *,
    out_path: Path,
    accuracy_csv: Path,
    alignment_summary_csv: Path,
    validation_rows: list[dict[str, object]],
    profile_summary: list[dict[str, object]],
    replication_rows: list[dict[str, object]],
) -> None:
    alignment = pd.read_csv(alignment_summary_csv)
    fit_error_min = float(alignment["fit_error_pp_min"].min())
    fit_error_max = float(alignment["fit_error_pp_max"].max())
    target_budget = float(alignment["target_acc_drop_pp"].dropna().iloc[0])

    worst_row = max(validation_rows, key=lambda row: float(row["abs_extra_drop_pp"]))
    lines = [
        "# PHY Validation Report (20260310)",
        "",
        "Scope",
        f"- nominal calibration grid source: `{accuracy_csv}`",
        f"- released alignment summary: `{alignment_summary_csv}`",
        (
            "- held-out validation runs: "
            + ", ".join(f"`{row['run_id']}`" for row in validation_rows[:: max(1, len(validation_rows) // 3)])
        ),
        "",
        "Released calibration anchor",
        (
            f"- The released P0->P1 alignment still relies on a 16-point grid and selects the nominal "
            f"`(gaussian_noise_sigma_lsb_ref, crosstalk_alpha_ref) = ({_fmt(float(alignment['sigma_lsb_ref'].iloc[0]), 3)}, "
            f"{_fmt(float(alignment['crosstalk_alpha_ref'].iloc[0]), 3)})` reference point."
        ),
        (
            f"- Reported run-level fit error remains `{_fmt(fit_error_min, 3)}-{_fmt(fit_error_max, 3)} pp`, "
            f"which is already above the manuscript's `{_fmt(target_budget, 1)} pp` target budget."
        ),
        "",
        "Held-out simulator validation",
        (
            "- The validation set uses separate robust-profile runs that keep the same Gaussian-noise/crosstalk anchors as the nominal "
            "grid but additionally turn on drift, correlation, and burst perturbations."
        ),
        (
            f"- Worst single-model miss in the held-out set is `{_fmt(float(worst_row['extra_drop_vs_nominal_pp']))} pp` "
            f"for `{worst_row['model']}` under `{worst_row['profile']}`."
        ),
    ]
    for row in profile_summary:
        lines.insert(
            -1,
            (
                f"- `{row['profile']}`: nominal `{_fmt(float(row['nominal_grid_acc_drop_pp_mean']))} pp` -> "
                f"held-out `{_fmt(float(row['heldout_acc_drop_pp_mean']))} pp` "
                f"(`+{_fmt(float(row['extra_drop_vs_nominal_pp_mean']))} pp`)."
            ),
        )

    if replication_rows:
        max_replica_delta = max(float(row["max_model_delta_pp"]) for row in replication_rows)
        lines.extend(
            [
                "",
                "Replication check",
                (
                    f"- Matching `E6` robust counterparts reproduce the canonical `E5` profile means with "
                    f"`max_model_delta_pp = {_fmt(max_replica_delta, 4)}`, so they do not add independent evidence."
                ),
            ]
        )

    lines.extend(
        [
            "",
            "Interpretation",
            (
                f"- The smallest profile-mean held-out residual is "
                f"`{_fmt(min(float(row['extra_drop_vs_nominal_pp_mean']) for row in profile_summary))} pp`, "
                f"so the omitted terms alone still exceed the `{_fmt(target_budget, 1)} pp` budget."
            ),
            (
                "- This supports only a local, simulation-calibrated sensitivity boundary around the nominal point; "
                "it does not support a strong main-text feasibility region or any hardware-validated wording."
            ),
            (
                "- Safe manuscript position: keep PHY as appendix sensitivity/context, and avoid defining a "
                "supportable region from the coarse calibration surface alone."
            ),
        ]
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a held-out PHY validation report.")
    parser.add_argument("--accuracy_csv", type=Path, default=DEFAULT_ACCURACY_CSV)
    parser.add_argument("--alignment_summary_csv", type=Path, default=DEFAULT_ALIGNMENT_SUMMARY_CSV)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--tag", default=DEFAULT_TAG)
    parser.add_argument("--validation_run_id", dest="validation_run_ids", action="append")
    parser.add_argument("--replication_run_id", dest="replication_run_ids", action="append")
    args = parser.parse_args()

    validation_run_ids = args.validation_run_ids or list(DEFAULT_VALIDATION_RUN_IDS)
    replication_run_ids = args.replication_run_ids or list(DEFAULT_REPLICATION_RUN_IDS)

    model_grid, _ = _build_nominal_grid(args.accuracy_csv)
    validation_rows = _build_validation_rows(run_ids=validation_run_ids, model_grid=model_grid)
    profile_summary = _build_profile_summary(validation_rows)
    replication_rows = _build_replication_rows(validation_run_ids, replication_run_ids)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    points_csv = args.out_dir / f"phy_validation_independent_points_{args.tag}.csv"
    profile_csv = args.out_dir / f"phy_validation_profile_summary_{args.tag}.csv"
    replication_csv = args.out_dir / f"phy_validation_replication_check_{args.tag}.csv"
    report_md = args.out_dir / f"phy_validation_report_{args.tag}.md"

    _write_csv(
        points_csv,
        validation_rows,
        [
            "run_id",
            "profile",
            "experiment_id",
            "model",
            "sigma_lsb",
            "crosstalk_alpha",
            "drift_lsb",
            "noise_correlation",
            "burst_error_prob",
            "nominal_grid_acc_drop_pp",
            "heldout_acc_drop_pp",
            "extra_drop_vs_nominal_pp",
            "abs_extra_drop_pp",
            "p1_align_fit_error_pp",
            "p1_align_method",
        ],
    )
    _write_csv(
        profile_csv,
        profile_summary,
        [
            "profile",
            "sigma_lsb",
            "crosstalk_alpha",
            "drift_lsb",
            "noise_correlation",
            "burst_error_prob",
            "n_models",
            "nominal_grid_acc_drop_pp_mean",
            "heldout_acc_drop_pp_mean",
            "extra_drop_vs_nominal_pp_mean",
            "abs_extra_drop_pp_mean",
            "max_abs_extra_drop_pp",
        ],
    )
    _write_csv(
        replication_csv,
        replication_rows,
        [
            "profile",
            "primary_run_id",
            "replication_run_id",
            "mean_acc_drop_pp_primary",
            "mean_acc_drop_pp_replica",
            "max_model_delta_pp",
        ],
    )
    _write_report(
        out_path=report_md,
        accuracy_csv=args.accuracy_csv,
        alignment_summary_csv=args.alignment_summary_csv,
        validation_rows=validation_rows,
        profile_summary=profile_summary,
        replication_rows=replication_rows,
    )


if __name__ == "__main__":
    main()
