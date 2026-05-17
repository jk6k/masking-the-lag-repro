"""Render guidance-document experimental data figures as SVG.

This script draws data-driven figures (Fig-C/D/E/F/G/H/I/J/K/L/M) directly
from quick-report CSVs and run-level outputs without third-party plotting libs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from exp_common.io_utils import resolve_workspace_path  # noqa: E402

RESULTS = ROOT / "results"
RUNS = RESULTS / "runs"

W = 980
H = 620
ML = 90
MR = 60
MT = 70
MB = 90

PALETTE = [
    "#4e79a7",
    "#f28e2b",
    "#59a14f",
    "#e15759",
    "#76b7b2",
    "#edc948",
    "#b07aa1",
    "#ff9da7",
]


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _f(v: Any, default: float = 0.0) -> float:
    try:
        if v is None:
            return default
        s = str(v).strip()
        if not s:
            return default
        return float(s)
    except Exception:
        return default


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    return sum(_f(r.get(key), 0.0) for r in rows) / len(rows)


def _safe_denorm(v: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 0.5
    return (v - lo) / (hi - lo)


def _svg_header(title: str, width: int = W, height: int = H) -> list[str]:
    return [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}'>",
        "<rect width='100%' height='100%' fill='white'/>",
        f"<text x='{width/2}' y='36' text-anchor='middle' font-size='20' font-family='Arial'>{title}</text>",
    ]


def _svg_axes(svg: list[str], x_label: str, y_label: str, width: int = W, height: int = H) -> tuple[float, float, float, float]:
    x0 = ML
    y0 = height - MB
    cw = width - ML - MR
    ch = height - MT - MB
    svg.append(f"<line x1='{x0}' y1='{y0}' x2='{x0+cw}' y2='{y0}' stroke='#333'/>")
    svg.append(f"<line x1='{x0}' y1='{y0}' x2='{x0}' y2='{y0-ch}' stroke='#333'/>")
    svg.append(
        f"<text x='{x0 + cw/2}' y='{height - 30}' text-anchor='middle' font-size='13' font-family='Arial'>{x_label}</text>"
    )
    svg.append(
        f"<text x='22' y='{y0 - ch/2}' text-anchor='middle' font-size='13' font-family='Arial' transform='rotate(-90 22 {y0 - ch/2})'>{y_label}</text>"
    )
    return x0, y0, cw, ch


def _add_y_ticks(svg: list[str], x0: float, y0: float, ch: float, y_min: float, y_max: float, n: int = 5) -> None:
    for i in range(n + 1):
        t = i / n
        y = y0 - ch * t
        val = y_min + (y_max - y_min) * t
        svg.append(f"<line x1='{x0-5}' y1='{y}' x2='{x0}' y2='{y}' stroke='#333'/>")
        svg.append(
            f"<text x='{x0-8}' y='{y+4}' text-anchor='end' font-size='11' font-family='Arial'>{val:.3g}</text>"
        )


def _save_svg(path: Path, svg: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    svg.append("</svg>")
    path.write_text("\n".join(svg), encoding="utf-8")


def render_line_chart(
    *,
    title: str,
    x_label: str,
    y_label: str,
    xs: list[float],
    series: list[tuple[str, list[float], str]],
    out_path: Path,
) -> None:
    if not xs or not series:
        return
    x_min, x_max = min(xs), max(xs)
    vals = [v for _, ys, _ in series for v in ys]
    y_min, y_max = min(vals), max(vals)
    if math.isclose(y_min, y_max):
        y_max = y_min + 1.0
    pad = 0.08 * (y_max - y_min)
    y_min -= pad
    y_max += pad

    svg = _svg_header(title)
    x0, y0, cw, ch = _svg_axes(svg, x_label, y_label)
    _add_y_ticks(svg, x0, y0, ch, y_min, y_max)

    def xp(v: float) -> float:
        return x0 + cw * _safe_denorm(v, x_min, x_max)

    def yp(v: float) -> float:
        return y0 - ch * _safe_denorm(v, y_min, y_max)

    # x ticks at data points
    for xv in xs:
        x = xp(xv)
        svg.append(f"<line x1='{x}' y1='{y0}' x2='{x}' y2='{y0+5}' stroke='#333'/>")
        svg.append(f"<text x='{x}' y='{y0+20}' text-anchor='middle' font-size='11' font-family='Arial'>{xv:g}</text>")

    # series
    for idx, (label, ys, color) in enumerate(series):
        pts = [(xp(x), yp(y)) for x, y in zip(xs, ys, strict=True)]
        if not pts:
            continue
        d = " ".join(f"{x:.2f},{y:.2f}" for x, y in pts)
        svg.append(f"<polyline points='{d}' fill='none' stroke='{color}' stroke-width='2'/>")
        for x, y in pts:
            svg.append(f"<circle cx='{x:.2f}' cy='{y:.2f}' r='3' fill='{color}'/>")
        lx = ML + idx * 220
        ly = 52
        svg.append(f"<rect x='{lx}' y='{ly-10}' width='14' height='14' fill='{color}'/>")
        svg.append(f"<text x='{lx+20}' y='{ly+2}' font-size='12' font-family='Arial'>{label}</text>")

    _save_svg(out_path, svg)


def render_grouped_bars(
    *,
    title: str,
    x_label: str,
    y_label: str,
    categories: list[str],
    series: list[tuple[str, list[float], str]],
    out_path: Path,
) -> None:
    if not categories or not series:
        return
    vals = [v for _, ys, _ in series for v in ys]
    y_min = min(0.0, min(vals))
    y_max = max(vals)
    if math.isclose(y_min, y_max):
        y_max = y_min + 1.0
    pad = 0.1 * (y_max - y_min)
    y_min -= pad
    y_max += pad

    svg = _svg_header(title)
    x0, y0, cw, ch = _svg_axes(svg, x_label, y_label)
    _add_y_ticks(svg, x0, y0, ch, y_min, y_max)

    n_cat = len(categories)
    n_ser = len(series)
    group_w = cw / n_cat
    bar_w = group_w / (n_ser + 1)

    def yp(v: float) -> float:
        return y0 - ch * _safe_denorm(v, y_min, y_max)

    y_zero = yp(0.0)
    svg.append(f"<line x1='{x0}' y1='{y_zero}' x2='{x0+cw}' y2='{y_zero}' stroke='#777' stroke-dasharray='4,3'/>")

    for ci, cat in enumerate(categories):
        gx = x0 + ci * group_w
        svg.append(
            f"<text x='{gx + group_w/2}' y='{y0+20}' text-anchor='middle' font-size='11' font-family='Arial'>{cat}</text>"
        )
        for si, (label, ys, color) in enumerate(series):
            v = ys[ci]
            x = gx + (si + 0.5) * bar_w
            y = yp(v)
            h = abs(y_zero - y)
            y_top = min(y, y_zero)
            svg.append(f"<rect x='{x:.2f}' y='{y_top:.2f}' width='{bar_w*0.75:.2f}' height='{h:.2f}' fill='{color}'/>")
            svg.append(
                f"<text x='{x + bar_w*0.375:.2f}' y='{y_top-4:.2f}' text-anchor='middle' font-size='10' font-family='Arial'>{v:.3g}</text>"
            )

    for i, (label, _, color) in enumerate(series):
        lx = ML + i * 220
        ly = 52
        svg.append(f"<rect x='{lx}' y='{ly-10}' width='14' height='14' fill='{color}'/>")
        svg.append(f"<text x='{lx+20}' y='{ly+2}' font-size='12' font-family='Arial'>{label}</text>")

    _save_svg(out_path, svg)


def render_stacked_bars(
    *,
    title: str,
    x_label: str,
    y_label: str,
    categories: list[str],
    stack_labels: list[str],
    values: dict[str, list[float]],
    colors: list[str],
    out_path: Path,
) -> None:
    if not categories:
        return
    totals = [sum(values[c]) for c in categories]
    y_min = 0.0
    y_max = max(totals) if totals else 1.0
    if math.isclose(y_min, y_max):
        y_max = 1.0
    y_max *= 1.15

    svg = _svg_header(title)
    x0, y0, cw, ch = _svg_axes(svg, x_label, y_label)
    _add_y_ticks(svg, x0, y0, ch, y_min, y_max)

    bar_w = cw / max(1, len(categories)) * 0.45

    def yp(v: float) -> float:
        return y0 - ch * _safe_denorm(v, y_min, y_max)

    for ci, cat in enumerate(categories):
        cx = x0 + (ci + 0.5) * (cw / len(categories))
        svg.append(
            f"<text x='{cx}' y='{y0+20}' text-anchor='middle' font-size='11' font-family='Arial'>{cat}</text>"
        )
        cur = 0.0
        for si, comp in enumerate(values[cat]):
            y1 = yp(cur)
            cur += comp
            y2 = yp(cur)
            y_top = min(y1, y2)
            h = abs(y1 - y2)
            svg.append(
                f"<rect x='{cx - bar_w/2:.2f}' y='{y_top:.2f}' width='{bar_w:.2f}' height='{h:.2f}' fill='{colors[si % len(colors)]}'/>"
            )
        svg.append(
            f"<text x='{cx}' y='{yp(cur)-5:.2f}' text-anchor='middle' font-size='10' font-family='Arial'>{cur:.3g}</text>"
        )

    for i, label in enumerate(stack_labels):
        lx = ML + (i % 3) * 260
        ly = 52 + (i // 3) * 18
        col = colors[i % len(colors)]
        svg.append(f"<rect x='{lx}' y='{ly-10}' width='14' height='14' fill='{col}'/>")
        svg.append(f"<text x='{lx+20}' y='{ly+2}' font-size='12' font-family='Arial'>{label}</text>")

    _save_svg(out_path, svg)


def render_heatmap(
    *,
    title: str,
    x_label: str,
    y_label: str,
    xs: list[float],
    ys: list[float],
    values: dict[tuple[float, float], float],
    out_path: Path,
) -> None:
    if not xs or not ys:
        return
    val_list = list(values.values()) or [0.0]
    vmin, vmax = min(val_list), max(val_list)
    if math.isclose(vmin, vmax):
        vmax = vmin + 1.0

    svg = _svg_header(title)
    x0, y0, cw, ch = _svg_axes(svg, x_label, y_label)
    # no y ticks for heatmap, draw custom labels
    cell_w = cw / len(xs)
    cell_h = ch / len(ys)

    def color(v: float) -> str:
        t = _safe_denorm(v, vmin, vmax)
        # blue -> white -> red
        if t < 0.5:
            tt = t / 0.5
            r = int(70 + (255 - 70) * tt)
            g = int(110 + (255 - 110) * tt)
            b = int(190 + (255 - 190) * tt)
        else:
            tt = (t - 0.5) / 0.5
            r = int(255 - (255 - 210) * tt)
            g = int(255 - (255 - 60) * tt)
            b = int(255 - (255 - 60) * tt)
        return f"#{r:02x}{g:02x}{b:02x}"

    y_order = sorted(ys, reverse=True)
    for yi, yv in enumerate(y_order):
        for xi, xv in enumerate(xs):
            v = values.get((xv, yv))
            if v is None:
                continue
            x = x0 + xi * cell_w
            y = y0 - ch + yi * cell_h
            col = color(v)
            svg.append(f"<rect x='{x:.2f}' y='{y:.2f}' width='{cell_w:.2f}' height='{cell_h:.2f}' fill='{col}' stroke='white'/>")
            svg.append(
                f"<text x='{x+cell_w/2:.2f}' y='{y+cell_h/2+4:.2f}' text-anchor='middle' font-size='10' font-family='Arial'>{v:.2f}</text>"
            )

    for xi, xv in enumerate(xs):
        x = x0 + xi * cell_w + cell_w / 2
        svg.append(f"<text x='{x:.2f}' y='{y0+20}' text-anchor='middle' font-size='11' font-family='Arial'>{xv:g}</text>")
    for yi, yv in enumerate(y_order):
        y = y0 - ch + yi * cell_h + cell_h / 2 + 4
        svg.append(f"<text x='{x0-10}' y='{y:.2f}' text-anchor='end' font-size='11' font-family='Arial'>{yv:g}</text>")

    # colorbar
    cbx = W - MR + 15
    cby = y0 - ch
    cbw = 16
    cbh = ch
    steps = 60
    for i in range(steps):
        t = i / (steps - 1)
        v = vmax - (vmax - vmin) * t
        y = cby + i * cbh / steps
        svg.append(f"<rect x='{cbx}' y='{y:.2f}' width='{cbw}' height='{cbh/steps:.2f}' fill='{color(v)}'/>")
    svg.append(f"<text x='{cbx+cbw+6}' y='{cby+10}' font-size='10' font-family='Arial'>{vmax:.2f}</text>")
    svg.append(f"<text x='{cbx+cbw+6}' y='{cby+cbh}' font-size='10' font-family='Arial'>{vmin:.2f}</text>")

    _save_svg(out_path, svg)


def render_scatter(
    *,
    title: str,
    x_label: str,
    y_label: str,
    points: list[tuple[float, float, str, str]],
    out_path: Path,
) -> None:
    if not points:
        return
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    if math.isclose(x_min, x_max):
        x_max = x_min + 1.0
    if math.isclose(y_min, y_max):
        y_max = y_min + 1.0
    x_pad = 0.1 * (x_max - x_min)
    y_pad = 0.1 * (y_max - y_min)
    x_min -= x_pad
    x_max += x_pad
    y_min -= y_pad
    y_max += y_pad

    svg = _svg_header(title)
    x0, y0, cw, ch = _svg_axes(svg, x_label, y_label)
    _add_y_ticks(svg, x0, y0, ch, y_min, y_max)

    def xp(v: float) -> float:
        return x0 + cw * _safe_denorm(v, x_min, x_max)

    def yp(v: float) -> float:
        return y0 - ch * _safe_denorm(v, y_min, y_max)

    # x ticks
    for i in range(6):
        t = i / 5
        xv = x_min + (x_max - x_min) * t
        x = xp(xv)
        svg.append(f"<line x1='{x}' y1='{y0}' x2='{x}' y2='{y0+5}' stroke='#333'/>")
        svg.append(f"<text x='{x}' y='{y0+20}' text-anchor='middle' font-size='11' font-family='Arial'>{xv:.3g}</text>")

    placed_labels: list[tuple[float, float, float, float]] = []

    def _overlap(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> bool:
        return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])

    for x, y, lbl, col in points:
        xx, yy = xp(x), yp(y)
        svg.append(f"<circle cx='{xx:.2f}' cy='{yy:.2f}' r='5' fill='{col}'/>")
        if lbl:
            # Greedy offset search to reduce label overlap in dense regions.
            label_w = max(38.0, 6.4 * len(lbl))
            label_h = 12.0
            candidates = [
                (8.0, -6.0),
                (10.0, 12.0),
                (10.0, -16.0),
                (-label_w - 8.0, -6.0),
                (-label_w - 8.0, 10.0),
                (0.0, 20.0),
                (0.0, -20.0),
            ]
            chosen = candidates[0]
            chosen_box = (
                xx + chosen[0],
                yy + chosen[1] - label_h,
                xx + chosen[0] + label_w,
                yy + chosen[1] + 2.0,
            )
            for off in candidates:
                cand_box = (
                    xx + off[0],
                    yy + off[1] - label_h,
                    xx + off[0] + label_w,
                    yy + off[1] + 2.0,
                )
                if any(_overlap(cand_box, p) for p in placed_labels):
                    continue
                chosen = off
                chosen_box = cand_box
                break
            placed_labels.append(chosen_box)
            svg.append(
                f"<text x='{xx+chosen[0]:.2f}' y='{yy+chosen[1]:.2f}' font-size='11' font-family='Arial'>{lbl}</text>"
            )

    _save_svg(out_path, svg)


def _load_master(run_id: str) -> list[dict[str, Any]]:
    return _read_csv(RUNS / run_id / "master_metrics.csv")


def _mean_breakdown(run_id: str) -> list[float]:
    rows = _load_master(run_id)
    return [
        _mean(rows, "energy_breakdown_conversion_control_j"),
        _mean(rows, "energy_breakdown_memory_move_j"),
        _mean(rows, "energy_breakdown_oe_j"),
        _mean(rows, "energy_breakdown_adc_pca_j"),
        _mean(rows, "energy_breakdown_laser_optical_j"),
        _mean(rows, "energy_breakdown_other_static_j"),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Render guidance data plots (SVG).")
    parser.add_argument("--run_prefix", default="quickpack_accfix_20260212")
    parser.add_argument(
        "--quick_reports_dir",
        default="experiments/results/quick_reports/final_paper_v2",
    )
    parser.add_argument(
        "--out_dir",
        default="experiments/results/plots/guidance_final_paper_v2",
    )
    args = parser.parse_args()

    quick_dir = resolve_workspace_path(args.quick_reports_dir, anchor=ROOT)
    out_dir = resolve_workspace_path(args.out_dir, anchor=ROOT)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Fig-C / Fig-D / Fig-I / Fig-J from quick reports.
    e3 = _read_csv(quick_dir / "quickscan_e3_k_sweep.csv")
    fig_d = _read_csv(quick_dir / "fig_d_prefix_error_vs_k.csv")
    e1 = _read_csv(quick_dir / "quickscan_e1_fanout_sweep.csv")
    fig_j = _read_csv(quick_dir / "fig_j_sparse_tau_pareto.csv")
    fig_h_points = _read_csv(quick_dir / "fig_h_accuracy_heatmap_points.csv")
    overview = _read_csv(quick_dir / "quickpack_e0_e6_overview.csv")

    # Fig-C: Accuracy drop vs Avg Effective BSL (E3 sweep)
    if e3:
        e3s = sorted(e3, key=lambda r: _f(r.get("avg_effective_bsl")))
        xs = [_f(r.get("avg_effective_bsl")) for r in e3s]
        ys = [_f(r.get("acc_drop_pp_mean")) for r in e3s]
        render_line_chart(
            title="Fig-C: Accuracy Drop vs Avg Effective BSL",
            x_label="Avg Effective BSL",
            y_label="Accuracy Drop (pp)",
            xs=xs,
            series=[("E3 mean drop", ys, PALETTE[0])],
            out_path=out_dir / "fig_c_accuracy_vs_effective_bsl.svg",
        )

    # Fig-D: Prefix error vs k
    if fig_d:
        d = sorted(fig_d, key=lambda r: _f(r.get("k")))
        xs = [_f(r.get("k")) for r in d]
        m = [_f(r.get("prefix_error_mean")) for r in d]
        p95 = [_f(r.get("prefix_error_p95")) for r in d]
        render_line_chart(
            title="Fig-D: Prefix Error vs k",
            x_label="k",
            y_label="Prefix Error",
            xs=xs,
            series=[
                ("mean", m, PALETTE[0]),
                ("p95", p95, PALETTE[3]),
            ],
            out_path=out_dir / "fig_d_prefix_error_vs_k.svg",
        )

    # Fig-E: Energy breakdown (E0/E3/E4/E5/E6)
    exps = ["E0", "E3", "E4", "E5", "E6"]
    cat = []
    vals: dict[str, list[float]] = {}
    for e in exps:
        rid = f"{args.run_prefix}_{e.lower()}"
        rows = _load_master(rid)
        if not rows:
            continue
        cat.append(e)
        vals[e] = _mean_breakdown(rid)
    if cat:
        render_stacked_bars(
            title="Fig-E: Energy Breakdown (J/inference)",
            x_label="Experiment",
            y_label="Energy (J)",
            categories=cat,
            stack_labels=[
                "conversion/control",
                "memory/move",
                "oe",
                "adc/pca",
                "laser/optical",
                "other/static",
            ],
            values=vals,
            colors=PALETTE,
            out_path=out_dir / "fig_e_energy_breakdown.svg",
        )

    # Fig-F: HOPS timeline proxy (bubble + utilization) using E0 vs E2
    flow_rows = []
    for e in ["E0", "E2"]:
        rid = f"{args.run_prefix}_{e.lower()}"
        rows = _load_master(rid)
        if rows:
            flow_rows.append((e, _mean(rows, "bubble_cycles"), _mean(rows, "utilization_avg")))
    if flow_rows:
        cats = [r[0] for r in flow_rows]
        bubble = [r[1] for r in flow_rows]
        util = [r[2] for r in flow_rows]
        render_grouped_bars(
            title="Fig-F: HOPS Timeline Proxy (Bubble / Utilization)",
            x_label="Experiment",
            y_label="Value (bubble cycles or utilization)",
            categories=cats,
            series=[
                ("bubble_cycles", bubble, PALETTE[3]),
                ("utilization_avg", util, PALETTE[2]),
            ],
            out_path=out_dir / "fig_f_flow_timeline_proxy.svg",
        )

    # Fig-G: PHY sweep (P_laser vs N), annotate PP crosstalk in labels.
    phy_sweep = _read_csv(RUNS / f"{args.run_prefix}_e5" / "phy_sweep.csv")
    if phy_sweep:
        s = sorted(phy_sweep, key=lambda r: _f(r.get("wdm_channels_n")))
        xs = [_f(r.get("wdm_channels_n")) for r in s]
        laser = [_f(r.get("p_laser_dbm")) for r in s]
        render_line_chart(
            title="Fig-G: P_laser vs N_wdm (E5 PHY sweep)",
            x_label="N_wdm",
            y_label="P_laser (dBm)",
            xs=xs,
            series=[("P_laser_dbm", laser, PALETTE[1])],
            out_path=out_dir / "fig_g_p_laser_vs_n.svg",
        )
        # Extra crosstalk chart for completeness.
        cros = [_f(r.get("pp_crosstalk_db")) for r in s]
        render_line_chart(
            title="Fig-G2: PP Crosstalk vs N_wdm (E5 PHY sweep)",
            x_label="N_wdm",
            y_label="PP crosstalk (dB)",
            xs=xs,
            series=[("PP_crosstalk_db", cros, PALETTE[0])],
            out_path=out_dir / "fig_g2_pp_crosstalk_vs_n.svg",
        )

    # Fig-H: sigma x alpha heatmap (mean acc_drop over models)
    if fig_h_points:
        sigmas = sorted({_f(r.get("sigma_lsb")) for r in fig_h_points})
        alphas = sorted({_f(r.get("crosstalk_alpha")) for r in fig_h_points})
        buckets: dict[tuple[float, float], list[float]] = {}
        for r in fig_h_points:
            key = (_f(r.get("sigma_lsb")), _f(r.get("crosstalk_alpha")))
            buckets.setdefault(key, []).append(_f(r.get("acc_drop_pp")))
        heat = {k: (sum(v) / len(v) if v else 0.0) for k, v in buckets.items()}
        render_heatmap(
            title="Fig-H: Accuracy Drop Heatmap (mean over models)",
            x_label="Gaussian noise sigma [LSB]",
            y_label="crosstalk_alpha",
            xs=sigmas,
            ys=alphas,
            values=heat,
            out_path=out_dir / "fig_h_acc_drop_heatmap.svg",
        )

    # Fig-I: MESO break-even sweep.
    if e1:
        e1s = sorted(e1, key=lambda r: _f(r.get("fanout_cfg")))
        cats = [str(int(_f(r.get("fanout_cfg")))) for r in e1s]
        net_gain = [_f(r.get("net_energy_gain_j")) for r in e1s]
        saved = [_f(r.get("serializers_saved")) for r in e1s]
        render_grouped_bars(
            title="Fig-I: MESO Break-even Sweep",
            x_label="fanout",
            y_label="net gain / serializers_saved",
            categories=cats,
            series=[
                ("net_energy_gain_j", net_gain, PALETTE[2]),
                ("serializers_saved", saved, PALETTE[0]),
            ],
            out_path=out_dir / "fig_i_meso_break_even.svg",
        )

    # Fig-J: sparse tau Pareto.
    if fig_j:
        points = [
            (
                _f(r.get("acc_drop_pp_mean")),
                _f(r.get("energy_saved_pct")),
                f"tau={_f(r.get('tau')):.2g}",
                PALETTE[idx % len(PALETTE)],
            )
            for idx, r in enumerate(sorted(fig_j, key=lambda rr: _f(rr.get("tau"))))
        ]
        render_scatter(
            title="Fig-J: Sparse Tau Pareto",
            x_label="Accuracy Drop (pp)",
            y_label="Energy Saved (%)",
            points=points,
            out_path=out_dir / "fig_j_sparse_tau_pareto.svg",
        )

    # Fig-K: P0 vs P1 consistency proxy (E5 vs E6).
    k_rows = []
    for e in ["E5", "E6"]:
        rid = f"{args.run_prefix}_{e.lower()}"
        rows = _load_master(rid)
        if rows:
            k_rows.append(
                (
                    e,
                    _mean(rows, "P_laser_dbm"),
                    _mean(rows, "sigma_lsb_ref"),
                    _mean(rows, "crosstalk_alpha_ref"),
                    _mean(rows, "acc_drop_pp"),
                )
            )
    if k_rows:
        cats = ["P_laser_dbm", "Gaussian noise sigma ref", "crosstalk_alpha_ref", "acc_drop_pp"]
        e5 = next((r for r in k_rows if r[0] == "E5"), None)
        e6 = next((r for r in k_rows if r[0] == "E6"), None)
        if e5 and e6:
            render_grouped_bars(
                title="Fig-K: P0 vs P1 Consistency Proxy (E5 vs E6)",
                x_label="Metric",
                y_label="Value",
                categories=cats,
                series=[
                    ("E5", [e5[1], e5[2], e5[3], e5[4]], PALETTE[0]),
                    ("E6", [e6[1], e6[2], e6[3], e6[4]], PALETTE[1]),
                ],
                out_path=out_dir / "fig_k_p0_p1_consistency_proxy.svg",
            )

    # Fig-L: overall Pareto (acc_top1 vs tops_w; label throughput).
    pareto_points: list[tuple[float, float, str, str]] = []
    for e in ["E0", "E1", "E2", "E3", "E4", "E5", "E6"]:
        rid = f"{args.run_prefix}_{e.lower()}"
        rows = _load_master(rid)
        if not rows:
            continue
        acc = _mean(rows, "acc_top1")
        tw = _mean(rows, "tops_w")
        th = _mean(rows, "throughput_images_s")
        label = f"{e} ({th:.1f} img/s)" if e in {"E2", "E3", "E6"} else ""
        pareto_points.append((acc, tw, label, PALETTE[int(e[1]) % len(PALETTE)]))
    if pareto_points:
        render_scatter(
            title="Fig-L: Overall Pareto (Accuracy vs TOPS/W)",
            x_label="Top-1 Accuracy (%)",
            y_label="TOPS/W",
            points=pareto_points,
            out_path=out_dir / "fig_l_overall_pareto.svg",
        )

    # Fig-M: DET net gain closure.
    m_rows = [r for r in overview if str(r.get("experiment_id")) in {"E3", "E6"}] if overview else []
    if m_rows:
        m_rows = sorted(m_rows, key=lambda r: r.get("experiment_id"))
        cats = [r["experiment_id"] for r in m_rows]
        net = [_f(r.get("det_net_gain_j")) for r in m_rows]
        passed = [_f(r.get("pass_det_net_gain_true")) for r in m_rows]
        render_grouped_bars(
            title="Fig-M: DET Net Gain Closure",
            x_label="Experiment",
            y_label="det_net_gain_j / pass count",
            categories=cats,
            series=[
                ("det_net_gain_j", net, PALETTE[2]),
                ("pass_det_net_gain_true", passed, PALETTE[3]),
            ],
            out_path=out_dir / "fig_m_det_net_gain_closure.svg",
        )

    manifest = {
        "run_prefix": args.run_prefix,
        "quick_reports_dir": str(quick_dir),
        "out_dir": str(out_dir),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[guidance-plots] saved SVG figures to {out_dir}")


if __name__ == "__main__":
    main()
