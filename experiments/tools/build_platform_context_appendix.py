#!/usr/bin/env python3
"""Build a reviewer-facing platform-context appendix pack.

This packages same-model CPU/GPU measured baselines, the simulated HPAT point,
and external accelerator anchors into a deduplicated appendix-ready figure,
summary CSV, and note. It is explicitly a platform-context asset, not a
photonic hardware validation result.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from statistics import geometric_mean

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_QUICK_DIR = ROOT / "experiments" / "results" / "quick_reports" / "20260306_stage2_seedtrue_fullgrid_repairedacc"
DEFAULT_OUT_DATA_DIR = ROOT / "AICAS" / "assets" / "candidate_data"
DEFAULT_OUT_FIG_DIR = ROOT / "AICAS" / "assets" / "candidate_figures" / "data"
TAG = "20260307_platformcontext"

PLATFORM_ORDER = ["CPU", "GPU", "HPAT"]
PLATFORM_COLORS = {
    "CPU": "#4c78a8",
    "GPU": "#f58518",
    "HPAT": "#54a24b",
    "ASIC": "#e45756",
    "FPGA": "#72b7b2",
    "PHOTONIC": "#b279a2",
}
SOURCE_MARKERS = {
    "measured": "o",
    "simulated_hpat": "s",
    "external_anchor": "^",
}


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Missing CSV: {path}")
    return pd.read_csv(path)


def _to_num(df: pd.DataFrame, cols: list[str]) -> None:
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")


def _pick_measured_row(group: pd.DataFrame) -> pd.Series:
    ranked = group.copy()
    ranked["energy_pos"] = ranked["energy_j"].fillna(0.0) > 0.0
    ranked["power_pos"] = ranked["avg_power_w"].fillna(0.0) > 0.0
    ranked = ranked.sort_values(
        ["energy_pos", "power_pos", "throughput_images_s", "latency_ms"],
        ascending=[False, False, False, True],
        kind="mergesort",
    )
    return ranked.iloc[0]


def dedupe_platform_rows(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["source_type"] = df["source_type"].astype(str).str.lower()
    df["platform_class"] = df["platform_class"].astype(str)
    df["model"] = df["model"].astype(str)
    df["device_name"] = df.get("device_name", "").fillna("").astype(str)
    _to_num(
        df,
        [
            "latency_ms",
            "energy_j",
            "avg_power_w",
            "tops_w",
            "peak_tops",
            "throughput_images_s",
            "area_mm2",
            "batch_size",
            "input_size",
            "sequence_length",
        ],
    )

    external_mask = df["source_type"] == "external_anchor"
    if external_mask.any():
        if {"provenance_status", "source_citation"}.issubset(df.columns):
            verified_mask = (
                df["provenance_status"].fillna("").astype(str).str.lower().eq("verified")
                & df["source_citation"].fillna("").astype(str).str.strip().ne("")
            )
        else:
            verified_mask = pd.Series(False, index=df.index)
        removed = int((external_mask & ~verified_mask).sum())
        if removed:
            print(f"[platform-context][warn] removed {removed} unverified external anchors from the appendix pack.")
            df = df.loc[~(external_mask & ~verified_mask)].copy()

    picked: list[dict[str, object]] = []

    measured = df[df["source_type"] == "measured"].copy()
    for _, group in measured.groupby(["model", "platform_class"], sort=True, dropna=False):
        picked.append(_pick_measured_row(group).to_dict())

    simulated = df[df["source_type"] == "simulated_hpat"].copy()
    if not simulated.empty:
        simulated = simulated.drop_duplicates(
            subset=[
                "model",
                "platform_class",
                "source_type",
                "latency_ms",
                "energy_j",
                "avg_power_w",
                "throughput_images_s",
                "batch_size",
                "input_size",
                "sequence_length",
            ]
        )
        picked.extend(simulated.to_dict(orient="records"))

    external = df[df["source_type"] == "external_anchor"].copy()
    if not external.empty:
        external = external.drop_duplicates(
            subset=[
                "device_name",
                "platform_class",
                "source_type",
                "latency_ms",
                "energy_j",
                "avg_power_w",
                "tops_w",
                "peak_tops",
                "input_size",
                "sequence_length",
            ]
        )
        picked.extend(external.to_dict(orient="records"))

    if not picked:
        return pd.DataFrame(columns=df.columns)

    out = pd.DataFrame(picked)

    out = out.sort_values(
        ["source_type", "platform_class", "model", "device_name"],
        kind="mergesort",
    ).reset_index(drop=True)
    return out


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary = df.copy()
    summary["latency_ratio_vs_hpat"] = math.nan
    summary["energy_ratio_vs_hpat"] = math.nan

    internal = summary[summary["platform_class"].isin(PLATFORM_ORDER)].copy()
    for model, group in internal.groupby("model", sort=True):
        hpat = group[group["platform_class"] == "HPAT"]
        if hpat.empty:
            continue
        hpat_latency = float(hpat.iloc[0]["latency_ms"])
        hpat_energy = float(hpat.iloc[0]["energy_j"])
        mask = summary["model"] == model
        summary.loc[mask, "latency_ratio_vs_hpat"] = summary.loc[mask, "latency_ms"] / hpat_latency
        summary.loc[mask, "energy_ratio_vs_hpat"] = summary.loc[mask, "energy_j"] / hpat_energy
    return summary


def _bar_panel(ax: plt.Axes, df: pd.DataFrame, metric: str, ylabel: str, title: str) -> None:
    models = ["mobilevit_xxs", "mobilevit_xs", "mobilevit_s"]
    width = 0.22
    x = list(range(len(models)))
    for idx, platform in enumerate(PLATFORM_ORDER):
        sub = df[df["platform_class"] == platform].copy()
        sub = sub.set_index("model")
        vals = [float(sub.loc[m, metric]) if m in sub.index else math.nan for m in models]
        offsets = [v + (idx - 1) * width for v in x]
        ax.bar(
            offsets,
            vals,
            width=width,
            color=PLATFORM_COLORS[platform],
            edgecolor="#222222",
            linewidth=0.5,
            label=platform,
        )
    ax.set_xticks(x, ["xxs", "xs", "s"])
    ax.set_yscale("log")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", linestyle=":", alpha=0.35)


def _scatter_panel(ax: plt.Axes, df: pd.DataFrame) -> None:
    for _, row in df.iterrows():
        lat = float(row["latency_ms"])
        ene = float(row["energy_j"])
        if not math.isfinite(lat) or not math.isfinite(ene) or ene <= 0.0:
            continue
        platform = str(row["platform_class"])
        src = str(row["source_type"])
        marker = SOURCE_MARKERS.get(src, "o")
        color = PLATFORM_COLORS.get(platform, "#777777")
        face = "none" if src == "external_anchor" else color
        ax.scatter(
            lat,
            ene,
            s=64 if src != "external_anchor" else 78,
            marker=marker,
            facecolors=face,
            edgecolors=color,
            linewidths=1.0,
            alpha=0.9,
        )
        if src == "external_anchor":
            label = str(row["device_name"]).replace("-like", "")
            ax.annotate(label, (lat, ene), xytext=(4, 3), textcoords="offset points", fontsize=6.5)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Latency (ms)")
    ax.set_ylabel("Energy (J)")
    ax.set_title("Panel C: same-model baselines + external anchors")
    ax.grid(True, linestyle=":", alpha=0.35)

    legend_rows = [
        ("CPU/GPU/HPAT", "o", "#4c78a8", "#4c78a8"),
        ("simulated HPAT", "s", "#54a24b", "#54a24b"),
        ("external anchor", "^", "#777777", "none"),
    ]
    handles = []
    labels = []
    for label, marker, edge, face in legend_rows:
        handles.append(
            plt.Line2D(
                [0],
                [0],
                marker=marker,
                linestyle="none",
                markeredgecolor=edge,
                markerfacecolor=face,
                markersize=6,
            )
        )
        labels.append(label)
    ax.legend(handles, labels, frameon=False, fontsize=7, loc="best")


def render_figure(summary: pd.DataFrame, out_path: Path) -> None:
    internal = summary[summary["platform_class"].isin(PLATFORM_ORDER)].copy()

    fig, axes = plt.subplots(1, 3, figsize=(9.2, 3.2))
    _bar_panel(axes[0], internal, "latency_ms", "Latency (ms)", "Panel A: measured CPU/GPU vs HPAT")
    _bar_panel(axes[1], internal, "energy_j", "Energy (J)", "Panel B: energy at batch=1")
    _scatter_panel(axes[2], summary)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(0.34, 1.03))
    fig.suptitle("Platform Context: measured digital baselines and external anchors", y=1.08, fontsize=11)
    fig.text(
        0.5,
        0.01,
        "Measured CPU/GPU rows are same-model digital baselines; external anchors are contextual only and do not constitute photonic hardware validation.",
        ha="center",
        fontsize=7,
        color="#666666",
    )
    fig.tight_layout(rect=[0, 0.06, 1, 0.98])
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _fmt_ratio(value: float) -> str:
    if not math.isfinite(value):
        return "NA"
    return f"{value:.2f}x"


def build_note(
    summary: pd.DataFrame,
    out_path: Path,
    figure_path: Path,
    summary_csv: Path,
    source_paths: list[Path],
) -> None:
    internal = summary[summary["platform_class"].isin(PLATFORM_ORDER)].copy()
    external = summary[summary["source_type"] == "external_anchor"].copy()

    cpu_lat = []
    gpu_lat = []
    cpu_energy = []
    gpu_energy = []
    for model, group in internal.groupby("model", sort=True):
        row_map = {str(r["platform_class"]): r for _, r in group.iterrows()}
        if {"CPU", "GPU", "HPAT"}.issubset(row_map):
            cpu_lat.append(float(row_map["CPU"]["latency_ratio_vs_hpat"]))
            gpu_lat.append(float(row_map["GPU"]["latency_ratio_vs_hpat"]))
            cpu_energy.append(float(row_map["CPU"]["energy_ratio_vs_hpat"]))
            gpu_energy.append(float(row_map["GPU"]["energy_ratio_vs_hpat"]))

    lines = [
        "# Platform Context Note",
        "",
        "Scope",
        f"- Source CSVs: `{source_paths[0]}` and `{source_paths[1]}`",
        f"- Summary CSV: `{summary_csv}`",
        f"- Figure: `{figure_path}`",
        "",
        "What is actually measured in-house",
        f"- Same-model digital baselines: `{len(internal[internal['source_type'] == 'measured'])}` deduplicated CPU/GPU rows",
        f"- Same-model simulated HPAT rows: `{len(internal[internal['source_type'] == 'simulated_hpat'])}` deduplicated rows",
        "- Coverage: MobileViT `xxs/xs/s`, `batch=1`, native input size per model",
        "",
        "Matched same-model comparison",
        f"- HPAT vs CPU latency geomean: `{_fmt_ratio(geometric_mean(cpu_lat) if cpu_lat else math.nan)}` faster",
        f"- HPAT vs GPU latency geomean: `{_fmt_ratio(geometric_mean(gpu_lat) if gpu_lat else math.nan)}` faster",
        f"- CPU-to-HPAT energy ratio geomean: `{_fmt_ratio(geometric_mean(cpu_energy) if cpu_energy else math.nan)}`",
        f"- GPU-to-HPAT energy ratio geomean: `{_fmt_ratio(geometric_mean(gpu_energy) if gpu_energy else math.nan)}`",
        "",
        "External anchor boundary",
        f"- External-anchor rows: `{len(external)}`",
        f"- External platforms: `{';'.join(sorted(external['platform_class'].astype(str).unique().tolist())) or 'NA'}`",
        "- External points differ in process, device assumptions, and in some cases sequence/model details; they are contextual only.",
        "",
        "Interpretation",
        "- This asset materially improves the paper's platform context because it places the simulated HPAT point against measured CPU/GPU baselines on the same MobileViT models.",
        "- It still does not close the hardware-validation gap for photonic hardware. There is no silicon, FPGA implementation of HOPS, or board-level photonic prototype here.",
        "- Reviewer use: cite this when clarifying that the artifact includes measured digital baselines and broader cross-platform context, while keeping the formal claim simulation-based.",
        "",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build platform-context appendix assets.")
    parser.add_argument("--quick_dir", type=Path, default=DEFAULT_QUICK_DIR)
    parser.add_argument("--out_data_dir", type=Path, default=DEFAULT_OUT_DATA_DIR)
    parser.add_argument("--out_fig_dir", type=Path, default=DEFAULT_OUT_FIG_DIR)
    parser.add_argument("--tag", type=str, default=TAG)
    args = parser.parse_args()

    args.out_data_dir.mkdir(parents=True, exist_ok=True)
    args.out_fig_dir.mkdir(parents=True, exist_ok=True)

    p_cmp = args.quick_dir / "hpat_cpu_gpu_compare.csv"
    p_acc = args.quick_dir / "accelerator_compare_summary.csv"
    df = pd.concat([_read_csv(p_cmp), _read_csv(p_acc)], ignore_index=True)

    deduped = dedupe_platform_rows(df)
    summary = build_summary(deduped)

    summary_csv = args.out_data_dir / f"platform_context_summary_{args.tag}.csv"
    note_md = args.out_data_dir / f"platform_context_note_{args.tag}.md"
    fig_pdf = args.out_fig_dir / f"Platform_Context_{args.tag}.pdf"
    fig_png = args.out_fig_dir / f"Platform_Context_{args.tag}.png"

    summary.to_csv(summary_csv, index=False)
    render_figure(summary, fig_pdf)
    render_figure(summary, fig_png)
    build_note(summary, note_md, fig_pdf, summary_csv, [p_cmp, p_acc])

    print(f"[ok] wrote {summary_csv}")
    print(f"[ok] wrote {note_md}")
    print(f"[ok] wrote {fig_pdf}")
    print(f"[ok] wrote {fig_png}")


if __name__ == "__main__":
    main()
