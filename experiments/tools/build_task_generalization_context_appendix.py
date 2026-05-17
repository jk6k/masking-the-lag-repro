"""Build appendix-ready task/family context assets from task_generalization_summary."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

try:
    from experiments.tools.path_policy import MAIN_PROJECT_REPORT_DATA_DIR, MAIN_PROJECT_REPORT_FIG_DIR
except ModuleNotFoundError:  # direct script execution
    from path_policy import MAIN_PROJECT_REPORT_DATA_DIR, MAIN_PROJECT_REPORT_FIG_DIR


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SUMMARY_CSV = ROOT / "experiments/results/quick_reports/20260305_stage2_seedtrue_fullgrid/task_generalization_summary.csv"
DEFAULT_OUT_DATA = MAIN_PROJECT_REPORT_DATA_DIR
DEFAULT_OUT_FIG = MAIN_PROJECT_REPORT_FIG_DIR


TASK_COLORS = {
    "imagenet_cls": "#59a14f",
    "coco_det": "#4e79a7",
    "coco_seg": "#f28e2b",
    "video_keypoint": "#e15759",
}
FAMILY_MARKERS = {
    "mobilevit": "o",
    "mobile-former": "s",
    "edgevit": "^",
    "efficientvit": "D",
}


def _load_rows(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in ["primary_metric_value", "tops_w", "latency_ms"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["task_id"] = df["task_id"].astype(str)
    df["model_family"] = df["model_family"].astype(str)
    df["source_type"] = df["source_type"].astype(str)
    keep = ["run_id", "model", "model_family", "task_id", "source_type", "primary_metric_value", "tops_w", "latency_ms"]
    df = df[keep].dropna(subset=["primary_metric_value", "tops_w", "latency_ms"])
    return df.drop_duplicates()


def _build_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for source_type, sub in df.groupby("source_type"):
        rows.append(
            {
                "source_type": source_type,
                "row_count": int(len(sub)),
                "task_count": int(sub["task_id"].nunique()),
                "tasks": ";".join(sorted(sub["task_id"].unique().tolist())),
                "family_count": int(sub["model_family"].nunique()),
                "families": ";".join(sorted(sub["model_family"].unique().tolist())),
            }
        )
    return pd.DataFrame(rows)


def _write_note(note_path: Path, summary_path: Path, fig_path: Path, df: pd.DataFrame, summary: pd.DataFrame) -> None:
    internal = df[df["source_type"] == "simulated_run"]
    external = df[df["source_type"] == "external_anchor"]
    lines = [
        "# Task/Family Context Note",
        "",
        "Scope",
        f"- Source CSV: `{DEFAULT_SUMMARY_CSV}`",
        f"- Summary CSV: `{summary_path}`",
        f"- Figure: `{fig_path}`",
        "",
        "Key point",
        "- This asset is contextual only. It does not convert the paper into a multi-workload evaluation.",
        "",
        "What is actually covered by HOPS runs",
        f"- Internal simulated rows: `{len(internal)}`",
        f"- Internal tasks: `{';'.join(sorted(internal['task_id'].unique().tolist()))}`",
        f"- Internal model families: `{';'.join(sorted(internal['model_family'].unique().tolist()))}`",
        "",
        "What comes from external anchors",
        f"- External-anchor rows: `{len(external)}`",
        f"- External tasks: `{';'.join(sorted(external['task_id'].unique().tolist()))}`",
        f"- External model families: `{';'.join(sorted(external['model_family'].unique().tolist()))}`",
        "",
        "Interpretation",
        "- The figure places the in-house MobileViT/ImageNet points in a broader task/family backdrop.",
        "- External anchors broaden context, but they are not evidence that the current HOPS artifact has been run and validated on those workloads.",
        "- Reviewer use: cite this when clarifying that broader context exists, while keeping the paper's formal empirical scope restricted to MobileViT/ImageNet.",
    ]
    note_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _render_figure(df: pd.DataFrame, out_base: Path) -> tuple[Path, Path, Path]:
    fig, ax = plt.subplots(figsize=(4.2, 3.1))
    internal = df[df["source_type"] == "simulated_run"].copy()
    external = df[df["source_type"] == "external_anchor"].copy()

    for _, row in external.iterrows():
        ax.scatter(
            row["primary_metric_value"],
            row["tops_w"],
            s=54,
            marker=FAMILY_MARKERS.get(row["model_family"], "X"),
            facecolors="none",
            edgecolors=TASK_COLORS.get(row["task_id"], "#777777"),
            linewidths=1.2,
            alpha=0.95,
        )

    for _, row in internal.iterrows():
        ax.scatter(
            row["primary_metric_value"],
            row["tops_w"],
            s=28,
            marker="o",
            color="#b07aa1",
            edgecolors="#222222",
            linewidths=0.4,
            alpha=0.75,
        )

    ax.set_xlabel("Primary Metric Value")
    ax.set_ylabel("TOPS/W")
    ax.set_title("Task/Family Context (Internal vs External)")
    ax.grid(alpha=0.18, linewidth=0.5)

    task_handles = [
        ax.scatter([], [], marker="o", facecolors="none", edgecolors=color, linewidths=1.2, label=f"anchor task={task}")
        for task, color in TASK_COLORS.items()
    ]
    source_handles = [
        ax.scatter([], [], marker="o", color="#b07aa1", edgecolors="#222222", linewidths=0.4, label="HOPS internal"),
        ax.scatter([], [], marker="o", facecolors="none", edgecolors="#555555", linewidths=1.2, label="external anchor"),
    ]
    ax.legend(handles=source_handles + task_handles, frameon=False, fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1.0))

    fig.text(
        0.5,
        0.02,
        "internal HOPS rows remain MobileViT/ImageNet only; external points are contextual anchors",
        ha="center",
        fontsize=7,
        color="#666666",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    svg_path = out_base.with_suffix(".svg")
    pdf_path = out_base.with_suffix(".pdf")
    png_path = out_base.with_suffix(".png")
    fig.savefig(svg_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight", dpi=600)
    plt.close(fig)
    return svg_path, pdf_path, png_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build appendix-ready task/family context assets.")
    parser.add_argument("--summary_csv", type=Path, default=DEFAULT_SUMMARY_CSV)
    parser.add_argument("--out_data_dir", type=Path, default=DEFAULT_OUT_DATA)
    parser.add_argument("--out_fig_dir", type=Path, default=DEFAULT_OUT_FIG)
    parser.add_argument("--tag", default="20260307_taskcontext")
    args = parser.parse_args()

    args.out_data_dir.mkdir(parents=True, exist_ok=True)
    args.out_fig_dir.mkdir(parents=True, exist_ok=True)

    df = _load_rows(args.summary_csv)
    summary = _build_summary(df)
    summary_csv = args.out_data_dir / f"task_generalization_context_summary_{args.tag}.csv"
    note_md = args.out_data_dir / f"task_generalization_context_note_{args.tag}.md"
    fig_base = args.out_fig_dir / f"Task_Generalization_Context_{args.tag}"

    summary.to_csv(summary_csv, index=False)
    _, fig_pdf, _ = _render_figure(df, fig_base)
    _write_note(note_md, summary_csv, fig_pdf, df, summary)

    print(f"[build-task-context] summary={summary_csv}")
    print(f"[build-task-context] note={note_md}")
    print(f"[build-task-context] figure={fig_pdf}")


if __name__ == "__main__":
    main()
