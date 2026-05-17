#!/usr/bin/env python3
"""Audit whether DET prefix-error is empirically validated against measured Top-1."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


def _resolve_path(path_like: str | Path) -> Path:
    raw = Path(path_like)
    if raw.is_absolute():
        return raw
    candidate = REPO_ROOT / raw
    if candidate.exists():
        return candidate
    return raw.resolve()


def _load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _pearson(x: pd.Series, y: pd.Series) -> float | None:
    if x.nunique(dropna=True) <= 1 or y.nunique(dropna=True) <= 1:
        return None
    return float(x.corr(y, method="pearson"))


def _spearman(x: pd.Series, y: pd.Series) -> float | None:
    if x.nunique(dropna=True) <= 1 or y.nunique(dropna=True) <= 1:
        return None
    return float(x.corr(y, method="spearman"))


def _render_plot(df: pd.DataFrame, out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(5.8, 3.8))
    x = df["k_global"].astype(float).to_numpy()
    y_meas = df["measured_vs_e0_pp"].astype(float).to_numpy()
    y_pred = df["proxy_implied_delta_pp"].astype(float).to_numpy()

    ax.plot(x, y_meas, marker="o", linewidth=1.9, color="#1f77b4", label="Measured Top-1 delta vs E0")
    ax.plot(x, y_pred, marker="s", linewidth=1.6, color="#d62728", label="Proxy-implied delta (prefix_error_mean x scale)")
    ax.axhline(0.0, color="#555555", linestyle="--", linewidth=0.9)
    ax.set_xlabel("DET truncated prefix length k")
    ax.set_ylabel("Relative Top-1 delta (pp)")
    ax.set_title("DET Proxy Audit: Measured vs Proxy-Implied Relative Delta")
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()

    out_base = out_dir / "prefix_proxy_validation"
    fig.savefig(out_base.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return out_base.with_suffix(".pdf")


def _write_report(
    *,
    out_path: Path,
    quickscan_path: Path,
    config_path: Path,
    measured_col: str,
    scale: float,
    delta_budget_pp: float | None,
    selected_k: float | None,
    summary_row: dict[str, Any],
    per_k_df: pd.DataFrame,
    plot_path: Path,
) -> None:
    lines: list[str] = []
    lines.append("# Prefix Proxy Validation Audit")
    lines.append("")
    lines.append("Setup")
    lines.append(f"- quickscan_csv: `{quickscan_path}`")
    lines.append(f"- config_snapshot: `{config_path}`")
    lines.append(f"- measured_relative_metric: `{measured_col}`")
    lines.append(f"- internal proxy scale: `{scale:.3f} pp per unit prefix_error_mean`")
    lines.append(f"- proxy decision budget: `{delta_budget_pp if delta_budget_pp is not None else 'n/a'}` pp")
    lines.append(f"- selected k in config: `{selected_k if selected_k is not None else 'n/a'}`")
    lines.append(f"- plot: `{plot_path}`")
    lines.append("")

    lines.append("Summary")
    lines.append(
        "- "
        f"measured span={float(summary_row['measured_span_pp']):.6f} pp, "
        f"proxy-implied span={float(summary_row['proxy_implied_span_pp']):.6f} pp"
    )
    lines.append(
        "- "
        f"pearson(prefix_error, measured)={summary_row['pearson_prefix_vs_measured']}, "
        f"spearman(prefix_error, measured)={summary_row['spearman_prefix_vs_measured']}"
    )
    lines.append(
        "- "
        f"status: {summary_row['validation_status']}"
    )
    lines.append("")

    lines.append("Interpretation")
    lines.append("- In the released measured artifact, the DET sweep has zero measured E0-relative Top-1 separation across k.")
    lines.append("- The proxy still induces a large internal ranking because `proxy_implied_delta_pp = prefix_error_mean x scale`.")
    lines.append("- Therefore prefix-error is currently a heuristic screening signal, not an empirically validated surrogate for measured Top-1.")
    lines.append("")

    lines.append("Per-k check")
    focus = per_k_df[["k_global", "measured_vs_e0_pp", "proxy_implied_delta_pp", "pass_proxy_budget"]]
    for _, row in focus.sort_values("k_global").iterrows():
        lines.append(
            "- "
            f"k={int(row['k_global'])}: "
            f"measured_vs_E0={float(row['measured_vs_e0_pp']):.4f} pp, "
            f"proxy_implied={float(row['proxy_implied_delta_pp']):.4f} pp, "
            f"pass_proxy_budget={bool(row['pass_proxy_budget'])}"
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit DET prefix-error proxy against measured Top-1 deltas.")
    parser.add_argument(
        "--quickscan_csv",
        default="experiments/results/quick_reports/20260305_stage2_seedtrue_fullgrid/quickscan_e3_k_sweep.csv",
        help="Current quickscan_e3_k_sweep.csv to audit.",
    )
    parser.add_argument(
        "--config_snapshot",
        default="experiments/results/runs/20260228_opt_sync_scan_e3_k64/config_snapshot.yaml",
        help="Config snapshot used to read internal proxy scale and selected k.",
    )
    parser.add_argument(
        "--out_tag",
        default="20260306_prefix_proxy_validation",
        help="Output tag under experiments/results/proxy_audit/.",
    )
    args = parser.parse_args()

    quickscan_path = _resolve_path(args.quickscan_csv)
    config_path = _resolve_path(args.config_snapshot)

    out_dir = REPO_ROOT / "experiments" / "results" / "proxy_audit" / args.out_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(quickscan_path)
    cfg = _load_yaml(config_path)

    measured_col = "measured_acc_drop_pp_vs_E0_mean"
    if measured_col not in df.columns:
        raise SystemExit(f"Expected measured relative column missing: {measured_col}")

    scale = float(
        (((cfg.get("sc_det") or {}).get("prefix_error") or {}).get("error_to_acc_pp_scale"))
        or ((cfg.get("sc_det") or {}).get("prefix_error_to_acc_pp_scale"))
        or 100.0
    )
    delta_budget_pp = (
        ((cfg.get("sc_det") or {}).get("prefix_error") or {}).get("delta_pp_budget")
    )
    if delta_budget_pp is None:
        delta_budget_pp = (cfg.get("accuracy") or {}).get("delta_pp_budget")
    if delta_budget_pp is None:
        delta_budget_pp = (cfg.get("sc_det") or {}).get("delta_pp")
    delta_budget_pp = float(delta_budget_pp) if delta_budget_pp is not None else None

    selected_k = _load_yaml(config_path).get("sc_det", {}).get("early_stop", {}).get("k_global")
    selected_k = float(selected_k) if selected_k is not None else None

    per_k_df = df.copy()
    per_k_df["measured_vs_e0_pp"] = per_k_df[measured_col].astype(float)
    per_k_df["proxy_implied_delta_pp"] = per_k_df["prefix_error_mean"].astype(float) * scale
    if delta_budget_pp is not None:
        per_k_df["pass_proxy_budget"] = per_k_df["proxy_implied_delta_pp"] <= float(delta_budget_pp)
    else:
        per_k_df["pass_proxy_budget"] = False
    per_k_df = per_k_df.sort_values("k_global")

    measured = per_k_df["measured_vs_e0_pp"].astype(float)
    prefix = per_k_df["prefix_error_mean"].astype(float)
    proxy_implied = per_k_df["proxy_implied_delta_pp"].astype(float)

    pearson = _pearson(prefix, measured)
    spearman = _spearman(prefix, measured)
    validation_status = (
        "not_empirically_identifiable_zero_measured_variance"
        if measured.nunique(dropna=True) <= 1
        else "empirically_correlatable"
    )

    summary_row = {
        "quickscan_csv": str(quickscan_path),
        "config_snapshot": str(config_path),
        "n_k": int(len(per_k_df)),
        "measured_metric": measured_col,
        "measured_span_pp": float(measured.max() - measured.min()) if not measured.empty else math.nan,
        "proxy_implied_span_pp": float(proxy_implied.max() - proxy_implied.min()) if not proxy_implied.empty else math.nan,
        "pearson_prefix_vs_measured": pearson,
        "spearman_prefix_vs_measured": spearman,
        "internal_proxy_scale_pp": scale,
        "delta_budget_pp": delta_budget_pp,
        "selected_k": selected_k,
        "validation_status": validation_status,
    }

    summary_df = pd.DataFrame([summary_row])
    summary_df.to_csv(out_dir / "prefix_proxy_validation_summary.csv", index=False)
    per_k_df[
        [
            "k_global",
            "run_id",
            "measured_vs_e0_pp",
            "prefix_error_mean",
            "prefix_error_p95",
            "proxy_implied_delta_pp",
            "pass_proxy_budget",
        ]
    ].to_csv(out_dir / "prefix_proxy_validation_per_k.csv", index=False)

    plot_path = _render_plot(per_k_df, out_dir)
    _write_report(
        out_path=out_dir / "prefix_proxy_validation_report.md",
        quickscan_path=quickscan_path,
        config_path=config_path,
        measured_col=measured_col,
        scale=scale,
        delta_budget_pp=delta_budget_pp,
        selected_k=selected_k,
        summary_row=summary_row,
        per_k_df=per_k_df,
        plot_path=plot_path,
    )
    print(f"[prefix-proxy-audit] completed: {out_dir}")


if __name__ == "__main__":
    main()
