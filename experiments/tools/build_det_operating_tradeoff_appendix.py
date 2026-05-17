#!/usr/bin/env python3
"""Build a conservative DET operating-region note and replace Table 2."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SUBSET_SUMMARY = ROOT / "AICAS" / "assets" / "candidate_data" / "det_k_accuracy_summary_20260307_detk4096_seeds012.csv"
DEFAULT_QUICKSCAN = ROOT / "experiments" / "results" / "quick_reports" / "20260305_stage2_seedtrue_fullgrid" / "quickscan_e3_k_sweep.csv"
DEFAULT_FULL_EVAL = ROOT / "AICAS" / "assets" / "candidate_data" / "config_conditioned_accuracy_summary_20260307_fulleval_seeds012.csv"
DEFAULT_OUT_DATA = ROOT / "AICAS" / "assets" / "candidate_data"
DEFAULT_OUT_FIG = ROOT / "AICAS" / "assets" / "candidate_figures" / "data"
DEFAULT_TABLE_CSV = ROOT / "AICAS" / "assets" / "main_text_tables" / "Table2_DET_OperatingPoints.csv"
DEFAULT_TABLE_TEX = ROOT / "AICAS" / "assets" / "main_text_tables" / "Table2_DET_OperatingPoints.tex"
DEFAULT_TAG = "20260310"


def _format_num(value: float | None, digits: int = 2, suffix: str = "") -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value):.{digits}f}{suffix}"


def _load_tradeoff_rows(subset_summary: Path, quickscan_csv: Path, full_eval_csv: Path) -> pd.DataFrame:
    subset = pd.read_csv(subset_summary)
    quickscan = pd.read_csv(quickscan_csv)
    full_eval = pd.read_csv(full_eval_csv)

    subset["det_k_global"] = pd.to_numeric(subset["det_k_global"], errors="coerce")
    quickscan["k_global"] = pd.to_numeric(quickscan["k_global"], errors="coerce")

    subset_by_k = subset.set_index("det_k_global")
    quickscan_by_k = quickscan.set_index("k_global")
    full_eval_e3 = full_eval[full_eval["experiment_id"] == "E3"].iloc[0]

    rows: list[dict[str, object]] = []
    for k, evidence_tier in [(64.0, "full_eval"), (96.0, "subset_only"), (112.0, "efficiency_only"), (129.0, "reference")]:
        quick = quickscan_by_k.loc[k]
        row: dict[str, object] = {
            "policy": "fixed-k",
            "k": int(k),
            "accuracy_evidence": "",
            "speedup_vs_E0": float(quick["speedup_vs_E0"]),
            "acc_delta_vs_E0_quant_pp": None,
            "det_net_gain_j": float(quick["det_net_gain_j"]),
            "prefix_error_mean": float(quick["prefix_error_mean"]),
            "evidence_tier": evidence_tier,
            "support_status": "",
        }
        if k == 64.0:
            row["accuracy_evidence"] = "full-eval seeds012"
            row["acc_delta_vs_E0_quant_pp"] = float(full_eval_e3["paired_model_mean_delta_vs_e0_quant_pp"])
            row["support_status"] = "full-eval available but accuracy-negative"
        elif k == 96.0:
            row["accuracy_evidence"] = "4096-image/model subset"
            row["acc_delta_vs_E0_quant_pp"] = float(subset_by_k.loc[k, "paired_model_mean_delta_vs_e0_quant_pp"])
            row["support_status"] = "subset-only screen, not promotable to default"
        elif k == 112.0:
            row["accuracy_evidence"] = "none in repaired measured path"
            row["support_status"] = "dense-grid efficiency point lacks measured accuracy"
        else:
            row["accuracy_evidence"] = "E0 full-eval reference"
            row["acc_delta_vs_E0_quant_pp"] = 0.0
            row["support_status"] = "zero-approximation reference only"
        rows.append(row)

    rows.extend(
        [
            {
                "policy": "layer-wise k",
                "k": None,
                "accuracy_evidence": "not run",
                "speedup_vs_E0": None,
                "acc_delta_vs_E0_quant_pp": None,
                "det_net_gain_j": None,
                "prefix_error_mean": None,
                "evidence_tier": "missing_policy",
                "support_status": "required policy control missing",
            },
            {
                "policy": "confidence-conditioned k",
                "k": None,
                "accuracy_evidence": "not run",
                "speedup_vs_E0": None,
                "acc_delta_vs_E0_quant_pp": None,
                "det_net_gain_j": None,
                "prefix_error_mean": None,
                "evidence_tier": "missing_policy",
                "support_status": "required policy control missing",
            },
        ]
    )
    return pd.DataFrame(rows)


def _write_report(report_path: Path, summary_csv: Path, figure_pdf: Path, rows: pd.DataFrame) -> None:
    lines = [
        "# DET Operating-Region Tradeoff Note (20260310)",
        "",
        "Scope",
        f"- Summary CSV: `{summary_csv}`",
        f"- Figure: `{figure_pdf}`",
        f"- Subset DET sweep source: `{DEFAULT_SUBSET_SUMMARY}`",
        f"- Efficiency source: `{DEFAULT_QUICKSCAN}`",
        f"- Full-eval config-conditioned source: `{DEFAULT_FULL_EVAL}`",
        "",
        "Evidence status",
    ]
    for row in rows.itertuples(index=False):
        k_label = "n/a" if pd.isna(row.k) else str(int(row.k))
        lines.append(
            f"- `{row.policy}` / `k={k_label}`: "
            f"accuracy evidence `{row.accuracy_evidence}`, "
            f"speedup `{_format_num(row.speedup_vs_E0, 2, 'x')}`, "
            f"delta vs E0 quant `{_format_num(row.acc_delta_vs_E0_quant_pp, 2, ' pp')}`, "
            f"status `{row.support_status}`."
        )
    lines.extend(
        [
            "",
            "Interpretation",
            "- The only non-trivial fixed-k point with repaired full-eval accuracy is `k=64`, and it remains clearly accuracy-negative at `-5.21 pp` versus the local E0 quantized reference.",
            "- `k=96` is still accuracy-negative and only backed by the 4096-image/model subset sweep, so it cannot be promoted as a reviewer-safe operating point.",
            "- `k=112` improves the dense grid on the efficiency axis, but the repository does not contain repaired measured accuracy rows for that point as of March 10, 2026.",
            "- No `layer-wise k` or `confidence-conditioned k` control rows are present in the repository, so the current DET evidence does not satisfy the planned policy-comparison requirement.",
            "- The defensible manuscript position is therefore `DET as an explicit tradeoff knob`, not `DET as a validated dominant beneficial module`.",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _render_figure(rows: pd.DataFrame, out_base: Path) -> tuple[Path, Path, Path]:
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6), gridspec_kw={"width_ratios": [1.45, 1.0]})

    fixed = rows[rows["policy"] == "fixed-k"].copy()
    style = {
        "full_eval": dict(marker="o", facecolor="#4e79a7", edgecolor="#1f1f1f", label="Full-eval"),
        "subset_only": dict(marker="s", facecolor="white", edgecolor="#f28e2b", label="Subset only"),
        "efficiency_only": dict(marker="D", facecolor="#d0d0d0", edgecolor="#555555", label="No measured accuracy"),
        "reference": dict(marker="^", facecolor="#59a14f", edgecolor="#1f1f1f", label="Reference"),
    }

    ax = axes[0]
    for tier, group in fixed.groupby("evidence_tier"):
        plot = group.dropna(subset=["acc_delta_vs_E0_quant_pp"])
        if plot.empty:
            continue
        st = style[tier]
        ax.scatter(
            plot["speedup_vs_E0"],
            plot["acc_delta_vs_E0_quant_pp"],
            s=70,
            marker=st["marker"],
            facecolors=st["facecolor"],
            edgecolors=st["edgecolor"],
            linewidths=1.2,
            label=st["label"],
        )
        for row in plot.itertuples(index=False):
            ax.text(float(row.speedup_vs_E0) + 0.01, float(row.acc_delta_vs_E0_quant_pp) + 0.12, f"k={int(row.k)}", fontsize=8)
    ax.axhline(-2.0, color="#aa0000", linestyle="--", linewidth=0.8)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel("Speedup vs E0")
    ax.set_ylabel("Top-1 delta vs E0 quant (pp)")
    ax.set_title("Panel A: DET fixed-k evidence tiers")
    ax.grid(alpha=0.25, linewidth=0.5)
    ax.legend(frameon=False, fontsize=8, loc="lower left")

    ax2 = axes[1]
    ax2.axis("off")
    text_lines = [
        "Panel B: uncovered evidence gaps",
        "",
        "k=64: only non-trivial full-eval point",
        "  still -5.21 pp vs E0 quant",
        "",
        "k=96: subset-only screen",
        "  still -4.23 pp on 4096-image/model sweep",
        "",
        "k=112: efficiency point only",
        "  no repaired measured accuracy row",
        "",
        "layer-wise k: not run",
        "confidence-conditioned k: not run",
        "",
        "Conclusion:",
        "DET is supported as a tradeoff knob,",
        "not as a recommended default.",
    ]
    ax2.text(0.0, 1.0, "\n".join(text_lines), va="top", ha="left", fontsize=8.5, family="monospace")

    fig.tight_layout()
    svg_path = out_base.with_suffix(".svg")
    pdf_path = out_base.with_suffix(".pdf")
    png_path = out_base.with_suffix(".png")
    fig.savefig(svg_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight", dpi=600)
    plt.close(fig)
    return svg_path, pdf_path, png_path


def _write_table_csv(path: Path, rows: pd.DataFrame) -> None:
    table = rows.copy()
    table["k"] = table["k"].apply(lambda value: "" if pd.isna(value) else int(value))
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)


def _write_table_tex(path: Path, rows: pd.DataFrame) -> None:
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{2.5pt}",
        "\\renewcommand{\\arraystretch}{0.94}",
        "\\caption{Conservative DET operating evidence. Only fixed-$k{=}64$ has repaired full-eval accuracy among non-trivial DET points; $k{=}96$ is subset-only, $k{=}112$ lacks measured accuracy, and layer-wise/confidence-conditioned controls remain unrun. DET is therefore supported as an explicit tradeoff knob rather than a recommended default.}",
        "\\label{tab:det_operating}",
        "\\resizebox{\\columnwidth}{!}{%",
        "\\begin{tabular}{lllrrl}",
        "\\toprule",
        "Policy & $k$ & Accuracy Evidence & Speedup & Top-1 $\\Delta$ vs E0 (pp) & Status \\\\",
        "\\midrule",
    ]
    for row in rows.itertuples(index=False):
        k_label = "--" if pd.isna(row.k) else str(int(row.k))
        lines.append(
            f"{row.policy} & {k_label} & {row.accuracy_evidence} & "
            f"{_format_num(row.speedup_vs_E0, 3)} & {_format_num(row.acc_delta_vs_E0_quant_pp, 2)} & {row.support_status} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "}", "\\end{table}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a conservative DET operating-region appendix note and replace Table 2.")
    parser.add_argument("--subset_summary_csv", type=Path, default=DEFAULT_SUBSET_SUMMARY)
    parser.add_argument("--quickscan_csv", type=Path, default=DEFAULT_QUICKSCAN)
    parser.add_argument("--full_eval_csv", type=Path, default=DEFAULT_FULL_EVAL)
    parser.add_argument("--out_data_dir", type=Path, default=DEFAULT_OUT_DATA)
    parser.add_argument("--out_fig_dir", type=Path, default=DEFAULT_OUT_FIG)
    parser.add_argument("--table_csv", type=Path, default=DEFAULT_TABLE_CSV)
    parser.add_argument("--table_tex", type=Path, default=DEFAULT_TABLE_TEX)
    parser.add_argument("--tag", default=DEFAULT_TAG)
    args = parser.parse_args()

    rows = _load_tradeoff_rows(args.subset_summary_csv, args.quickscan_csv, args.full_eval_csv)

    args.out_data_dir.mkdir(parents=True, exist_ok=True)
    args.out_fig_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = args.out_data_dir / f"det_operating_tradeoff_summary_{args.tag}.csv"
    report_md = args.out_data_dir / f"det_operating_tradeoff_report_{args.tag}.md"
    fig_base = args.out_fig_dir / f"DET_Operating_Tradeoff_{args.tag}"

    rows.to_csv(summary_csv, index=False)
    _, fig_pdf, _ = _render_figure(rows, fig_base)
    _write_report(report_md, summary_csv, fig_pdf, rows)
    _write_table_csv(args.table_csv, rows)
    _write_table_tex(args.table_tex, rows)

    print(summary_csv)
    print(report_md)
    print(fig_pdf)
    print(args.table_csv)
    print(args.table_tex)


if __name__ == "__main__":
    main()
