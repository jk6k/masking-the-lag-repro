"""Offline fallback renderer for Fig7/Fig13/Fig17/Fig18 (no pandas/matplotlib required).

Use this when plotting dependencies are unavailable. It renders:
- Fig7_RelatedWork (radar, normalized 0-1 scale)
- Fig13_AccHeatmap (explicit mean-over-models aggregation)
- Fig17_OverallPareto (overlap-safe markers + labels)
- Fig18_DET_Waterfall (DET saved/overhead/net waterfall)
to `figures/paper_figures_v2`, then mirrors files to
`experiments/results/paper_figures_v2`.
"""

from __future__ import annotations

import argparse
import csv
import html
import math
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_QUICK_DIR = ROOT / "experiments" / "results" / "quick_reports" / "final_paper_v2"
DEFAULT_OUT_DIR = ROOT / "figures" / "paper_figures_v2"
DEFAULT_MIRROR_DIR = ROOT / "experiments" / "results" / "paper_figures_v2"
CAIROSVG_BIN = ROOT / ".venv" / "bin" / "cairosvg"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _mean(values: Iterable[float]) -> float:
    items = list(values)
    return sum(items) / len(items) if items else 0.0


def _hex_interp(c1: tuple[int, int, int], c2: tuple[int, int, int], t: float) -> str:
    t = 0.0 if t < 0 else 1.0 if t > 1 else t
    r = round(c1[0] + (c2[0] - c1[0]) * t)
    g = round(c1[1] + (c2[1] - c1[1]) * t)
    b = round(c1[2] + (c2[2] - c1[2]) * t)
    return f"#{r:02x}{g:02x}{b:02x}"


def _color_scale(v: float, vmin: float, vmax: float) -> str:
    # Blue -> Yellow -> Red
    blue = (44, 123, 182)
    yellow = (255, 255, 191)
    red = (215, 25, 28)
    if vmax <= vmin:
        return _hex_interp(yellow, yellow, 0.0)
    t = (v - vmin) / (vmax - vmin)
    if t <= 0.5:
        return _hex_interp(blue, yellow, t / 0.5)
    return _hex_interp(yellow, red, (t - 0.5) / 0.5)


def _to_pdf(svg_path: Path) -> None:
    if not CAIROSVG_BIN.exists():
        print(f"[warn] cairosvg not found at {CAIROSVG_BIN}, skip PDF for {svg_path.name}")
        return
    pdf_path = svg_path.with_suffix(".pdf")
    subprocess.run([str(CAIROSVG_BIN), str(svg_path), "-o", str(pdf_path)], check=True)


def _mirror_outputs(names: list[str], out_dir: Path, mirror_dir: Path) -> None:
    mirror_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        for ext in (".svg", ".pdf"):
            src = out_dir / f"{name}{ext}"
            if src.exists():
                shutil.copy2(src, mirror_dir / src.name)


def _polar_point(cx: float, cy: float, radius: float, angle: float) -> tuple[float, float]:
    return (cx + radius * math.cos(angle), cy + radius * math.sin(angle))


def _marker_svg(
    x: float,
    y: float,
    marker: str,
    size: float,
    fill: str,
    stroke: str = "#111111",
    alpha: float = 0.82,
) -> str:
    s = size
    if marker == "square":
        return (
            f'<rect x="{x - s:.2f}" y="{y - s:.2f}" width="{2*s:.2f}" height="{2*s:.2f}" '
            f'fill="{fill}" fill-opacity="{alpha:.2f}" stroke="{stroke}" stroke-width="1"/>'
        )
    if marker == "diamond":
        pts = f"{x:.2f},{y - s:.2f} {x + s:.2f},{y:.2f} {x:.2f},{y + s:.2f} {x - s:.2f},{y:.2f}"
        return (
            f'<polygon points="{pts}" fill="{fill}" fill-opacity="{alpha:.2f}" '
            f'stroke="{stroke}" stroke-width="1"/>'
        )
    if marker == "tri_up":
        pts = f"{x:.2f},{y - s:.2f} {x + s:.2f},{y + s:.2f} {x - s:.2f},{y + s:.2f}"
        return (
            f'<polygon points="{pts}" fill="{fill}" fill-opacity="{alpha:.2f}" '
            f'stroke="{stroke}" stroke-width="1"/>'
        )
    if marker == "tri_down":
        pts = f"{x - s:.2f},{y - s:.2f} {x + s:.2f},{y - s:.2f} {x:.2f},{y + s:.2f}"
        return (
            f'<polygon points="{pts}" fill="{fill}" fill-opacity="{alpha:.2f}" '
            f'stroke="{stroke}" stroke-width="1"/>'
        )
    if marker == "plus":
        a = alpha
        w = s * 0.65
        h = s * 1.8
        return (
            f'<rect x="{x - w/2:.2f}" y="{y - h/2:.2f}" width="{w:.2f}" height="{h:.2f}" '
            f'fill="{fill}" fill-opacity="{a:.2f}" stroke="{stroke}" stroke-width="1"/>'
            f'<rect x="{x - h/2:.2f}" y="{y - w/2:.2f}" width="{h:.2f}" height="{w:.2f}" '
            f'fill="{fill}" fill-opacity="{a:.2f}" stroke="{stroke}" stroke-width="1"/>'
        )
    if marker == "x":
        return (
            f'<line x1="{x - s:.2f}" y1="{y - s:.2f}" x2="{x + s:.2f}" y2="{y + s:.2f}" '
            f'stroke="{fill}" stroke-width="2"/>'
            f'<line x1="{x + s:.2f}" y1="{y - s:.2f}" x2="{x - s:.2f}" y2="{y + s:.2f}" '
            f'stroke="{fill}" stroke-width="2"/>'
        )
    # circle
    return (
        f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{s:.2f}" fill="{fill}" fill-opacity="{alpha:.2f}" '
        f'stroke="{stroke}" stroke-width="1"/>'
    )


def render_fig7_radar(csv_path: Path, out_svg: Path) -> None:
    rows = _read_csv(csv_path)
    if not rows:
        raise RuntimeError("fig_a_related_work_radar_scores.csv is empty")

    metrics = [k for k in rows[0].keys() if str(k).strip().lower() not in {"work", "method", "name"}]
    if not metrics:
        raise RuntimeError("Radar CSV has no metric columns")

    # Render in semantic score space [1,5] to avoid "minimum looks like zero".
    raw_vals: list[float] = []
    for row in rows:
        for m in metrics:
            raw_vals.append(float(row[m]))
    raw_min = min(raw_vals)
    raw_max = max(raw_vals)

    def to_score_1_5(v: float) -> float:
        if raw_min >= -1e-9 and raw_max <= 1.05:
            # Normalized input -> convert back to semantic score.
            return 1.0 + 4.0 * max(0.0, min(1.0, v))
        if raw_min >= 0.95 and raw_max <= 5.05:
            return max(1.0, min(5.0, v))
        if raw_max <= raw_min:
            return 3.0
        return 1.0 + 4.0 * ((v - raw_min) / (raw_max - raw_min))

    width, height = 660, 480
    cx, cy, radius = 240.0, 240.0, 150.0
    n = len(metrics)
    angles = [(-math.pi / 2.0) + 2.0 * math.pi * i / n for i in range(n)]

    color_cycle = ["#0072b2", "#009e73", "#cc79a7", "#56b4e9", "#f0e442"]
    body: list[str] = []
    body.append(f'<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff"/>')

    # Grid rings at semantic score levels (1..5).
    for score in (1, 2, 3, 4, 5):
        ring = score / 5.0
        pts = []
        for a in angles:
            x, y = _polar_point(cx, cy, ring * radius, a)
            pts.append(f"{x:.2f},{y:.2f}")
        body.append(
            '<polygon points="{}" fill="none" stroke="#d0d0d0" stroke-width="1"/>'.format(
                " ".join(pts)
            )
        )
        lx, ly = _polar_point(cx, cy, ring * radius, -math.pi / 2.0)
        body.append(
            f'<text x="{lx + 8:.2f}" y="{ly + 4:.2f}" fill="#666666" font-size="10">{score:d}</text>'
        )

    # Axis spokes + metric labels.
    for i, m in enumerate(metrics):
        ax, ay = _polar_point(cx, cy, radius, angles[i])
        body.append(
            f'<line x1="{cx:.2f}" y1="{cy:.2f}" x2="{ax:.2f}" y2="{ay:.2f}" '
            'stroke="#b8b8b8" stroke-width="1"/>'
        )
        tx, ty = _polar_point(cx, cy, radius + 22.0, angles[i])
        body.append(
            f'<text x="{tx:.2f}" y="{ty:.2f}" text-anchor="middle" '
            'font-size="12" fill="#333333">{}</text>'.format(html.escape(m))
        )

    # Polygons.
    legend_x = 430
    legend_y = 95
    legend_step = 26
    legend_idx = 0
    for row in rows:
        label = row.get("Work") or row.get("work") or f"Method {legend_idx + 1}"
        this_work = ("This Work" in label) or ("Masking" in label)
        color = "#d55e00" if this_work else color_cycle[legend_idx % len(color_cycle)]
        legend_idx += 1
        stroke_w = 2.8 if this_work else 1.8
        fill_opacity = 0.20 if this_work else 0.05

        pts = []
        for i, m in enumerate(metrics):
            score = to_score_1_5(float(row[m]))
            rv = score / 5.0
            x, y = _polar_point(cx, cy, rv * radius, angles[i])
            pts.append((x, y))
        pt_s = " ".join(f"{x:.2f},{y:.2f}" for x, y in pts)
        body.append(
            f'<polygon points="{pt_s}" fill="{color}" fill-opacity="{fill_opacity:.2f}" '
            f'stroke="{color}" stroke-width="{stroke_w:.1f}" />'
        )

        ly = legend_y + (legend_idx - 1) * legend_step
        body.append(
            f'<line x1="{legend_x}" y1="{ly}" x2="{legend_x + 22}" y2="{ly}" '
            f'stroke="{color}" stroke-width="{stroke_w:.1f}" />'
        )
        body.append(
            f'<text x="{legend_x + 30}" y="{ly + 4}" font-size="12" fill="#333333">'
            f'{html.escape(label)}</text>'
        )

    body.append(
        '<text x="330" y="34" text-anchor="middle" font-size="16" '
        'font-weight="bold" fill="#111111">Related Work Comparison (Score 1-5)</text>'
    )

    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">' + "".join(body) + "</svg>"
    )
    out_svg.write_text(svg, encoding="utf-8")


def render_fig13_heatmap(csv_path: Path, out_svg: Path) -> None:
    rows = _read_csv(csv_path)
    if not rows:
        raise RuntimeError("fig_h_accuracy_heatmap_points.csv is empty")

    # Stage 1: average repeated samples per (model, sigma, alpha).
    by_model: dict[tuple[str, float, float], list[float]] = defaultdict(list)
    for r in rows:
        key = (r["model"], float(r["sigma_lsb"]), float(r["crosstalk_alpha"]))
        by_model[key].append(float(r["acc_drop_pp"]))
    stage1: dict[tuple[str, float, float], float] = {
        k: _mean(v) for k, v in by_model.items()
    }

    # Stage 2: average across models per (sigma, alpha).
    by_pair: dict[tuple[float, float], list[float]] = defaultdict(list)
    for (model, sigma, alpha), val in stage1.items():
        _ = model
        by_pair[(sigma, alpha)].append(val)
    agg: dict[tuple[float, float], float] = {k: _mean(v) for k, v in by_pair.items()}

    model_order = ["mobilevit_s", "mobilevit_xs", "mobilevit_xxs"]
    model_labels = {
        "mobilevit_s": "MobileViT-S",
        "mobilevit_xs": "MobileViT-XS",
        "mobilevit_xxs": "MobileViT-XXS",
    }
    sigma_vals = sorted({s for _, s, _ in stage1.keys()})
    alpha_vals = sorted({a for _, _, a in stage1.keys()}, reverse=True)
    values = list(stage1.values()) + list(agg.values())
    vmin = min(values)
    vmax = max(values)

    left, top = 90, 82
    cell_w, cell_h = 54, 44
    cols, rows_n = len(sigma_vals), len(alpha_vals)
    heat_w, heat_h = cols * cell_w, rows_n * cell_h
    panel_gap_x, panel_gap_y = 66, 74
    width = left + (2 * heat_w) + panel_gap_x + 170
    height = top + (2 * heat_h) + panel_gap_y + 120
    cbar_x, cbar_y, cbar_w, cbar_h = left + (2 * heat_w) + panel_gap_x + 34, top + 16, 24, (2 * heat_h) + panel_gap_y - 18

    def draw_panel(origin_x: float, origin_y: float, title: str, values_by_cell: dict[tuple[float, float], float], *, show_ylabel: bool, show_xlabel: bool) -> list[str]:
        body: list[str] = []
        body.append(
            f'<text x="{origin_x + heat_w / 2:.1f}" y="{origin_y - 18:.1f}" text-anchor="middle" '
            'font-size="16" font-weight="bold" fill="#111111">'
            f"{html.escape(title)}</text>"
        )

        for r_idx, alpha in enumerate(alpha_vals):
            for c_idx, sigma in enumerate(sigma_vals):
                x = origin_x + c_idx * cell_w
                y = origin_y + r_idx * cell_h
                key = (sigma, alpha)
                val = values_by_cell.get(key)
                if val is None:
                    body.append(
                        f'<rect x="{x}" y="{y}" width="{cell_w}" height="{cell_h}" '
                        'fill="#f3f3f3" stroke="#d0d0d0" stroke-width="1"/>'
                    )
                    continue
                color = _color_scale(val, vmin, vmax)
                body.append(
                    f'<rect x="{x}" y="{y}" width="{cell_w}" height="{cell_h}" '
                    f'fill="{color}" stroke="#d0d0d0" stroke-width="1"/>'
                )
                if val <= 1.0:
                    body.append(
                        f'<rect x="{x + 1.2:.1f}" y="{y + 1.2:.1f}" width="{cell_w - 2.4:.1f}" height="{cell_h - 2.4:.1f}" '
                        'fill="none" stroke="#111111" stroke-width="1.4"/>'
                    )
                body.append(
                    f'<text x="{x + cell_w / 2:.1f}" y="{y + cell_h / 2 + 4:.1f}" '
                    'text-anchor="middle" font-size="12" font-weight="bold" fill="#111111">{:.1f}</text>'.format(val)
                )

        for c_idx, sigma in enumerate(sigma_vals):
            x = origin_x + c_idx * cell_w + cell_w / 2
            y = origin_y + heat_h + 22
            body.append(
                f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="middle" font-size="12" fill="#222222">{sigma:.2f}</text>'
            )
        for r_idx, alpha in enumerate(alpha_vals):
            x = origin_x - 12
            y = origin_y + r_idx * cell_h + cell_h / 2 + 4
            anchor = "end" if show_ylabel else "start"
            tx = x if show_ylabel else (origin_x + heat_w + 8)
            fill = "#222222" if show_ylabel else "#ffffff"
            body.append(
                f'<text x="{tx:.1f}" y="{y:.1f}" text-anchor="{anchor}" font-size="12" fill="{fill}">{alpha:.2f}</text>'
            )

        if show_xlabel:
            body.append(
                f'<text x="{origin_x + heat_w / 2:.1f}" y="{origin_y + heat_h + 48:.1f}" text-anchor="middle" '
                'font-size="13" fill="#111111">Gaussian Noise σ (LSB)</text>'
            )
        if show_ylabel:
            body.append(
                f'<text x="{origin_x - 52:.1f}" y="{origin_y + heat_h / 2:.1f}" text-anchor="middle" font-size="13" '
                f'fill="#111111" transform="rotate(-90, {origin_x - 52:.1f}, {origin_y + heat_h / 2:.1f})">Crosstalk α</text>'
            )
        return body

    body: list[str] = []
    body.append(f'<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff"/>')

    panel_specs = [
        (model_labels["mobilevit_s"], {(sigma, alpha): val for (model, sigma, alpha), val in stage1.items() if model == "mobilevit_s"}),
        (model_labels["mobilevit_xs"], {(sigma, alpha): val for (model, sigma, alpha), val in stage1.items() if model == "mobilevit_xs"}),
        (model_labels["mobilevit_xxs"], {(sigma, alpha): val for (model, sigma, alpha), val in stage1.items() if model == "mobilevit_xxs"}),
        ("Mean (3 variants)", agg),
    ]

    for idx, (title, panel_vals) in enumerate(panel_specs):
        panel_col = idx % 2
        panel_row = idx // 2
        origin_x = left + panel_col * (heat_w + panel_gap_x)
        origin_y = top + panel_row * (heat_h + panel_gap_y)
        body.extend(
            draw_panel(
                origin_x,
                origin_y,
                title,
                panel_vals,
                show_ylabel=(panel_col == 0),
                show_xlabel=(panel_row == 1),
            )
        )

    # Colorbar.
    steps = 80
    for i in range(steps):
        t0 = i / steps
        val = vmin + (vmax - vmin) * (1.0 - t0)
        y = cbar_y + i * (cbar_h / steps)
        body.append(
            f'<rect x="{cbar_x}" y="{y:.2f}" width="{cbar_w}" height="{(cbar_h / steps) + 0.2:.2f}" '
            f'fill="{_color_scale(val, vmin, vmax)}" stroke="none"/>'
        )
    body.append(
        f'<rect x="{cbar_x}" y="{cbar_y}" width="{cbar_w}" height="{cbar_h}" '
        'fill="none" stroke="#999999" stroke-width="1"/>'
    )
    for t, label in ((0.0, f"{vmax:.1f}"), (0.5, f"{(vmin + vmax) / 2.0:.1f}"), (1.0, f"{vmin:.1f}")):
        y = cbar_y + t * cbar_h + 4
        body.append(
            f'<text x="{cbar_x + cbar_w + 8}" y="{y:.1f}" font-size="11" fill="#222222">{label}</text>'
        )
    body.append(
        f'<text x="{cbar_x + cbar_w / 2:.1f}" y="{cbar_y - 12}" text-anchor="middle" '
        'font-size="11" fill="#222222">Acc Drop (pp)</text>'
    )

    # Main title and note.
    body.append(
        '<text x="{}" y="34" text-anchor="middle" font-size="16" font-weight="bold" fill="#111111">'
        "Accuracy Sensitivity to Noise &amp; Crosstalk"
        "</text>".format(left + heat_w / 2)
    )
    body.append(
        f'<text x="{width / 2:.1f}" y="{height - 22:.1f}" text-anchor="middle" font-size="11" fill="#666666">'
        "Model panels = mean over 3 replicate rows per cell; mean panel = mean over 3 MobileViT variants."
        "</text>"
    )

    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">' + "".join(body) + "</svg>"
    )
    out_svg.write_text(svg, encoding="utf-8")


def render_fig17_overall_pareto(csv_path: Path, out_svg: Path) -> None:
    rows = _read_csv(csv_path)
    if not rows:
        raise RuntimeError("quickpack_e0_e6_overview.csv is empty")

    exp_order = ["E0", "E1", "E2", "E3", "E4", "E5", "E6"]
    data: list[dict[str, float | str]] = []
    for r in rows:
        exp = r.get("experiment_id", "")
        if exp not in exp_order:
            continue
        data.append(
            {
                "exp": exp,
                "energy": float(r.get("energy_j", "0") or 0.0),
                "speedup": float(r.get("speedup_vs_E0", "0") or 0.0),
                "thr": float(r.get("throughput_images_s", "0") or 0.0),
            }
        )
    if not data:
        raise RuntimeError("No E0-E6 rows found for Fig17")

    x_vals = [float(d["energy"]) for d in data]
    y_vals = [float(d["speedup"]) for d in data]
    x_min, x_max = min(x_vals), max(x_vals)
    y_min, y_max = min(y_vals), max(y_vals)
    x_pad = (x_max - x_min) * 0.08 if x_max > x_min else 1.0
    y_pad = (y_max - y_min) * 0.12 if y_max > y_min else 1.0
    x_min -= x_pad
    x_max += x_pad
    y_min -= y_pad
    y_max += y_pad

    width, height = 780, 500
    left, right, top, bottom = 86, 215, 68, 78
    plot_w = width - left - right
    plot_h = height - top - bottom

    def map_x(v: float) -> float:
        return left + (v - x_min) / (x_max - x_min) * plot_w

    def map_y(v: float) -> float:
        return top + (y_max - v) / (y_max - y_min) * plot_h

    # Overlap-safe offsets for duplicated coordinates.
    groups: dict[tuple[float, float], list[int]] = defaultdict(list)
    for i, d in enumerate(data):
        groups[(round(float(d["energy"]), 6), round(float(d["speedup"]), 6))].append(i)
    point_off: dict[int, tuple[float, float]] = {}
    label_off: dict[int, tuple[float, float]] = {}
    for idxs in groups.values():
        n = len(idxs)
        if n == 1:
            point_off[idxs[0]] = (0.0, 0.0)
            label_off[idxs[0]] = (8.0, -8.0)
            continue
        for j, idx in enumerate(idxs):
            ang = 2.0 * math.pi * j / n
            point_off[idx] = (7.5 * math.cos(ang), 7.5 * math.sin(ang))
            label_off[idx] = (18.0 * math.cos(ang), 18.0 * math.sin(ang))

    colors = {
        "E0": "#333333",
        "E1": "#4c72b0",
        "E2": "#55a868",
        "E3": "#c44e52",
        "E4": "#8172b3",
        "E5": "#dd8452",
        "E6": "#937860",
    }
    markers = {
        "E0": "circle",
        "E1": "square",
        "E2": "diamond",
        "E3": "tri_up",
        "E4": "tri_down",
        "E5": "plus",
        "E6": "x",
    }

    body: list[str] = []
    body.append(f'<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff"/>')
    body.append(
        f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="#ffffff" stroke="#d8d8d8" stroke-width="1"/>'
    )

    # Grid + ticks.
    xticks = 6
    yticks = 6
    for i in range(xticks):
        xv = x_min + i * (x_max - x_min) / (xticks - 1)
        px = map_x(xv)
        body.append(
            f'<line x1="{px:.2f}" y1="{top}" x2="{px:.2f}" y2="{top + plot_h}" '
            'stroke="#d9d9d9" stroke-dasharray="3,3" stroke-width="1"/>'
        )
        body.append(
            f'<text x="{px:.2f}" y="{top + plot_h + 24}" text-anchor="middle" font-size="11" fill="#444444">{xv:.3f}</text>'
        )
    for i in range(yticks):
        yv = y_min + i * (y_max - y_min) / (yticks - 1)
        py = map_y(yv)
        body.append(
            f'<line x1="{left}" y1="{py:.2f}" x2="{left + plot_w}" y2="{py:.2f}" '
            'stroke="#d9d9d9" stroke-dasharray="3,3" stroke-width="1"/>'
        )
        body.append(
            f'<text x="{left - 10}" y="{py + 4:.2f}" text-anchor="end" font-size="11" fill="#444444">{yv:.2f}</text>'
        )

    # Points and labels.
    # Keep callouts sparse (key Pareto points only) to avoid dense overlaps.
    annotate_exps = {"E2", "E3", "E6"}
    manual_label_offsets: dict[str, tuple[float, float]] = {
        "E2": (18.0, 14.0),
        "E3": (18.0, -20.0),
        "E6": (18.0, 10.0),
    }
    for i, d in enumerate(data):
        px = map_x(float(d["energy"]))
        py = map_y(float(d["speedup"]))
        dx, dy = point_off.get(i, (0.0, 0.0))
        px += dx
        py += dy
        exp = str(d["exp"])
        body.append(_marker_svg(px, py, markers[exp], 5.0, colors[exp], alpha=0.80))

        if exp in annotate_exps:
            lx, ly = manual_label_offsets.get(exp, label_off.get(i, (10.0, -10.0)))
            if py <= top + 0.06 * plot_h:
                ly = max(ly, 12.0)
            if py >= top + 0.94 * plot_h:
                ly = min(ly, -12.0)
            if px <= left + 0.06 * plot_w and lx < 0:
                lx = abs(lx) + 8.0
            if px >= left + 0.94 * plot_w and lx > 0:
                lx = -abs(lx) - 8.0

            tx = px + lx
            ty = py + ly
            body.append(
                f'<line x1="{px:.2f}" y1="{py:.2f}" x2="{tx - 3.0:.2f}" y2="{ty - 3.0:.2f}" '
                'stroke="#7a7a7a" stroke-width="0.8"/>'
            )
            body.append(
                f'<text x="{tx:.2f}" y="{ty:.2f}" font-size="10" fill="#222222">'
                f'{exp} {float(d["speedup"]):.2f}x</text>'
            )

    # Titles and axes labels.
    body.append(
        f'<text x="{left + plot_w/2:.2f}" y="34" text-anchor="middle" font-size="16" font-weight="bold" fill="#111111">'
        "Overall Pareto: Energy vs Speedup"
        "</text>"
    )
    body.append(
        f'<text x="{left + plot_w/2:.2f}" y="{height - 20}" text-anchor="middle" font-size="13" fill="#111111">'
        "Energy per Inference (J)"
        "</text>"
    )
    body.append(
        f'<text x="24" y="{top + plot_h/2:.2f}" text-anchor="middle" font-size="13" fill="#111111" '
        f'transform="rotate(-90, 24, {top + plot_h/2:.2f})">Speedup vs. E0 (x)</text>'
    )

    # Legend
    lx0 = width - right + 18
    ly0 = top + 36
    body.append(
        f'<text x="{lx0}" y="{ly0 - 16}" font-size="12" font-weight="bold" fill="#111111">Experiment</text>'
    )
    for i, exp in enumerate(exp_order):
        y = ly0 + i * 24
        body.append(_marker_svg(lx0 + 8, y - 3, markers[exp], 4.4, colors[exp], alpha=0.82))
        body.append(f'<text x="{lx0 + 20}" y="{y + 1}" font-size="11" fill="#222222">{exp}</text>')

    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">' + "".join(body) + "</svg>"
    )
    out_svg.write_text(svg, encoding="utf-8")


def render_fig18_waterfall(csv_path: Path, out_svg: Path) -> None:
    rows = _read_csv(csv_path)
    if not rows:
        raise RuntimeError("det_net_gain_waterfall.csv is empty")

    row = None
    for r in rows:
        if r.get("experiment_id") == "E6":
            row = r
            break
    if row is None:
        row = rows[-1]

    saved = float(row.get("det_saved_j", "0") or 0.0)
    overhead = float(row.get("det_overhead_j", "0") or 0.0)
    net = float(row.get("det_net_gain_j", "0") or 0.0)

    # Waterfall bars: saved (positive), overhead (negative), net result.
    b_vals = [saved, -overhead, net]
    labels = ["Gross Saved", "Overhead", "Net Gain"]
    colors = ["#55a868", "#c44e52", "#4c72b0"]

    ymax = max(saved, net, saved - overhead, 1e-9) * 1.35
    ymin = min(0.0, -overhead * 1.35)
    if ymax - ymin < 1e-12:
        ymax = ymin + 1e-9

    width, height = 700, 430
    left, right, top, bottom = 84, 70, 58, 82
    plot_w = width - left - right
    plot_h = height - top - bottom

    def map_y(v: float) -> float:
        return top + (ymax - v) / (ymax - ymin) * plot_h

    x_centers = [left + plot_w * (0.18 + i * 0.30) for i in range(3)]
    bar_w = 82.0

    body: list[str] = []
    body.append(f'<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff"/>')
    body.append(
        f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="#ffffff" stroke="#d8d8d8" stroke-width="1"/>'
    )

    # Y grid/ticks.
    yticks = 6
    for i in range(yticks):
        yv = ymin + i * (ymax - ymin) / (yticks - 1)
        py = map_y(yv)
        body.append(
            f'<line x1="{left}" y1="{py:.2f}" x2="{left + plot_w}" y2="{py:.2f}" '
            'stroke="#d9d9d9" stroke-dasharray="3,3" stroke-width="1"/>'
        )
        body.append(
            f'<text x="{left - 10}" y="{py + 4:.2f}" text-anchor="end" font-size="11" fill="#444444">{yv:.4f}</text>'
        )

    running = 0.0
    for i, (xc, val, color) in enumerate(zip(x_centers, b_vals, colors)):
        if i == 0:
            y0 = map_y(0.0)
            y1 = map_y(val)
            y = min(y0, y1)
            h = abs(y1 - y0)
            running = val
        elif i == 1:
            y0 = map_y(running)
            y1 = map_y(running + val)
            y = min(y0, y1)
            h = abs(y1 - y0)
            running = running + val
        else:
            y0 = map_y(0.0)
            y1 = map_y(running)
            y = min(y0, y1)
            h = abs(y1 - y0)

        body.append(
            f'<rect x="{xc - bar_w/2:.2f}" y="{y:.2f}" width="{bar_w:.2f}" height="{max(h,1.2):.2f}" '
            f'fill="{color}" fill-opacity="0.86" stroke="#333333" stroke-width="1"/>'
        )

        disp = [saved, overhead, net][i]
        body.append(
            f'<text x="{xc:.2f}" y="{y - 7:.2f}" text-anchor="middle" font-size="11" fill="#222222">{disp:.6f}</text>'
        )

    # Connector guides.
    y_saved = map_y(saved)
    y_after_over = map_y(saved - overhead)
    body.append(
        f'<line x1="{x_centers[0] + bar_w/2:.2f}" y1="{y_saved:.2f}" x2="{x_centers[1] - bar_w/2:.2f}" y2="{y_saved:.2f}" '
        'stroke="#777777" stroke-dasharray="4,3" stroke-width="1"/>'
    )
    body.append(
        f'<line x1="{x_centers[1] + bar_w/2:.2f}" y1="{y_after_over:.2f}" x2="{x_centers[2] - bar_w/2:.2f}" y2="{y_after_over:.2f}" '
        'stroke="#777777" stroke-dasharray="4,3" stroke-width="1"/>'
    )

    # X labels.
    for xc, lab in zip(x_centers, labels):
        body.append(
            f'<text x="{xc:.2f}" y="{top + plot_h + 28}" text-anchor="middle" font-size="12" fill="#222222">{lab}</text>'
        )

    # Axes and title.
    body.append(
        f'<text x="{left + plot_w/2:.2f}" y="34" text-anchor="middle" font-size="16" font-weight="bold" fill="#111111">'
        "DET Mechanism Efficiency Analysis (E6)"
        "</text>"
    )
    body.append(
        f'<text x="24" y="{top + plot_h/2:.2f}" text-anchor="middle" font-size="13" fill="#111111" '
        f'transform="rotate(-90, 24, {top + plot_h/2:.2f})">Energy (J)</text>'
    )

    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">' + "".join(body) + "</svg>"
    )
    out_svg.write_text(svg, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick_dir", type=Path, default=DEFAULT_QUICK_DIR)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--mirror_dir", type=Path, default=DEFAULT_MIRROR_DIR)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    fig7_svg = args.out_dir / "Fig7_RelatedWork.svg"
    fig13_svg = args.out_dir / "Fig13_AccHeatmap.svg"
    fig17_svg = args.out_dir / "Fig17_OverallPareto.svg"
    fig18_svg = args.out_dir / "Fig18_DET_Waterfall.svg"

    render_fig7_radar(args.quick_dir / "fig_a_related_work_radar_scores.csv", fig7_svg)
    render_fig13_heatmap(args.quick_dir / "fig_h_accuracy_heatmap_points.csv", fig13_svg)
    render_fig17_overall_pareto(args.quick_dir / "quickpack_e0_e6_overview.csv", fig17_svg)
    render_fig18_waterfall(args.quick_dir / "det_net_gain_waterfall.csv", fig18_svg)
    _to_pdf(fig7_svg)
    _to_pdf(fig13_svg)
    _to_pdf(fig17_svg)
    _to_pdf(fig18_svg)

    _mirror_outputs(["Fig7_RelatedWork", "Fig13_AccHeatmap", "Fig17_OverallPareto", "Fig18_DET_Waterfall"], args.out_dir, args.mirror_dir)
    print("[ok] Rendered fallback figures: Fig7_RelatedWork, Fig13_AccHeatmap, Fig17_OverallPareto, Fig18_DET_Waterfall")


if __name__ == "__main__":
    main()
