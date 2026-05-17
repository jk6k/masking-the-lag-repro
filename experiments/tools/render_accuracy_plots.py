"""Render SVG plots for noise/crosstalk accuracy results."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


SVG_WIDTH = 980
SVG_HEIGHT = 620
MARGIN_LEFT = 90
MARGIN_RIGHT = 90
MARGIN_TOP = 70
MARGIN_BOTTOM = 90

OVERVIEW_WIDTH = 1500
OVERVIEW_HEIGHT = 1000
OVERVIEW_MARGIN = 30
OVERVIEW_GAP = 30


MODEL_COLORS = {
    "mobilevit_xxs": "#4e79a7",
    "mobilevit_xs": "#59a14f",
    "mobilevit_s": "#f28e2b",
}

TOP5_COLORS = {
    "mobilevit_xxs": "#9cc3e5",
    "mobilevit_xs": "#9fd9b0",
    "mobilevit_s": "#f6c27a",
}


def to_float(value: str) -> float:
    return float(value)


def load_noise_data(csv_path: Path, metric: str) -> dict:
    with csv_path.open(newline="") as f:
        rows = list(csv.DictReader(f))

    noise_rows = [r for r in rows if r.get("notes") != "baseline"]
    baseline_rows = [r for r in rows if r.get("notes") == "baseline"]

    models = sorted({r["model"] for r in rows})
    sigmas = sorted({float(r["noise_sigma_lsb"]) for r in noise_rows})
    alphas = sorted({float(r["crosstalk_alpha"]) for r in noise_rows})

    data = {m: {} for m in models}
    for r in noise_rows:
        model = r["model"]
        sigma = float(r["noise_sigma_lsb"])
        alpha = float(r["crosstalk_alpha"])
        data.setdefault(model, {}).setdefault(sigma, {})[alpha] = to_float(r[metric])

    baselines = {}
    for r in baseline_rows:
        baselines[r["model"]] = {
            "top1": to_float(r["top1"]),
            "top5": to_float(r["top5"]),
        }

    return {
        "models": models,
        "sigmas": sigmas,
        "alphas": alphas,
        "data": data,
        "baselines": baselines,
    }


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _color_to_hex(r: int, g: int, b: int) -> str:
    return f"#{r:02x}{g:02x}{b:02x}"


def _color_scale(value: float, vmin: float, vmax: float) -> str:
    if vmax <= vmin:
        return "#cccccc"
    t = (value - vmin) / (vmax - vmin)
    t = _clamp(t, 0.0, 1.0)
    # blue -> white -> red
    if t < 0.5:
        t2 = t / 0.5
        r = int(_lerp(69, 255, t2))
        g = int(_lerp(117, 255, t2))
        b = int(_lerp(180, 255, t2))
    else:
        t2 = (t - 0.5) / 0.5
        r = int(_lerp(255, 215, t2))
        g = int(_lerp(255, 48, t2))
        b = int(_lerp(255, 39, t2))
    return _color_to_hex(r, g, b)


def render_heatmap(
    *,
    model: str,
    sigmas: list[float],
    alphas: list[float],
    data: dict,
    baselines: dict,
    vmin: float,
    vmax: float,
    metric_label: str,
    out_path: Path,
    sigma_label: str,
    alpha_label: str,
) -> None:
    width = SVG_WIDTH
    height = SVG_HEIGHT
    grid_left = MARGIN_LEFT
    grid_top = MARGIN_TOP
    grid_w = width - MARGIN_LEFT - MARGIN_RIGHT
    grid_h = height - MARGIN_TOP - MARGIN_BOTTOM
    cell_w = grid_w / len(sigmas)
    cell_h = grid_h / len(alphas)

    svg = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}'>",
        "<rect width='100%' height='100%' fill='white'/>",
        f"<text x='{width/2}' y='36' text-anchor='middle' font-size='20' font-family='Arial'>{model} {metric_label} heatmap</text>",
    ]

    # Axes labels
    svg.append(
        f"<text x='{width/2}' y='{height-30}' text-anchor='middle' font-size='13' font-family='Arial'>{sigma_label}</text>"
    )
    svg.append(
        f"<text x='20' y='{height/2}' text-anchor='middle' font-size='13' font-family='Arial' transform='rotate(-90 20 {height/2})'>{alpha_label}</text>"
    )

    # Cells
    for row, alpha in enumerate(sorted(alphas, reverse=True)):
        for col, sigma in enumerate(sigmas):
            val = data.get(model, {}).get(sigma, {}).get(alpha)
            if val is None:
                continue
            x = grid_left + col * cell_w
            y = grid_top + row * cell_h
            color = _color_scale(val, vmin, vmax)
            svg.append(
                f"<rect x='{x}' y='{y}' width='{cell_w}' height='{cell_h}' fill='{color}' stroke='#ffffff'/>"
            )
            svg.append(
                f"<text x='{x + cell_w/2}' y='{y + cell_h/2 + 4}' text-anchor='middle' font-size='11' font-family='Arial' fill='#111'>{val:.1f}</text>"
            )

    # Axis ticks
    for col, sigma in enumerate(sigmas):
        x = grid_left + col * cell_w + cell_w / 2
        svg.append(
            f"<text x='{x}' y='{grid_top + grid_h + 20}' text-anchor='middle' font-size='11' font-family='Arial'>{sigma:g}</text>"
        )
    for row, alpha in enumerate(sorted(alphas, reverse=True)):
        y = grid_top + row * cell_h + cell_h / 2 + 4
        svg.append(
            f"<text x='{grid_left - 10}' y='{y}' text-anchor='end' font-size='11' font-family='Arial'>{alpha:g}</text>"
        )

    # Baseline annotation
    baseline = baselines.get(model, {}).get("top1")
    if baseline is not None and metric_label.startswith("Top-1"):
        svg.append(
            f"<text x='{grid_left}' y='{grid_top - 12}' font-size='11' font-family='Arial'>baseline top1: {baseline:.2f}</text>"
        )
    baseline_top5 = baselines.get(model, {}).get("top5")
    if baseline_top5 is not None and metric_label.startswith("Top-5"):
        svg.append(
            f"<text x='{grid_left}' y='{grid_top - 12}' font-size='11' font-family='Arial'>baseline top5: {baseline_top5:.2f}</text>"
        )

    # Colorbar (right)
    bar_x = width - MARGIN_RIGHT + 20
    bar_y = grid_top
    bar_w = 16
    bar_h = grid_h
    steps = 60
    for i in range(steps):
        t = i / (steps - 1)
        v = vmin + (vmax - vmin) * (1 - t)
        color = _color_scale(v, vmin, vmax)
        y = bar_y + i * (bar_h / steps)
        svg.append(f"<rect x='{bar_x}' y='{y}' width='{bar_w}' height='{bar_h/steps}' fill='{color}'/>")
    svg.append(
        f"<text x='{bar_x + bar_w + 6}' y='{bar_y + 10}' font-size='11' font-family='Arial'>{vmax:.1f}</text>"
    )
    svg.append(
        f"<text x='{bar_x + bar_w + 6}' y='{bar_y + bar_h}' font-size='11' font-family='Arial'>{vmin:.1f}</text>"
    )

    svg.append("</svg>")
    out_path.write_text("\n".join(svg), encoding="utf-8")


def _avg_by_key(rows: list[dict], key: str) -> dict[float, float]:
    buckets: dict[float, list[float]] = {}
    for r in rows:
        buckets.setdefault(float(r[key]), []).append(float(r["top1"]))
    return {k: sum(v) / len(v) for k, v in buckets.items()}


def render_line_plot(
    *,
    title: str,
    subtitle: str | None,
    x_label: str,
    y_label: str,
    x_values: list[float],
    series: dict[str, dict[float, float]],
    out_path: Path,
) -> None:
    width = SVG_WIDTH
    height = SVG_HEIGHT
    chart_left = MARGIN_LEFT
    chart_top = MARGIN_TOP
    chart_w = width - MARGIN_LEFT - MARGIN_RIGHT
    chart_h = height - MARGIN_TOP - MARGIN_BOTTOM

    all_vals = [v for model in series.values() for v in model.values()]
    y_min = 0.0
    y_max = max(all_vals) if all_vals else 1.0
    y_max *= 1.15

    def x_pos(x: float) -> float:
        if len(x_values) == 1:
            return chart_left + chart_w / 2
        x_min = min(x_values)
        x_max = max(x_values)
        if x_max == x_min:
            return chart_left + chart_w / 2
        return chart_left + (x - x_min) / (x_max - x_min) * chart_w

    def y_pos(y: float) -> float:
        return chart_top + chart_h - (y - y_min) / (y_max - y_min) * chart_h

    svg = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}'>",
        "<rect width='100%' height='100%' fill='white'/>",
        f"<text x='{width/2}' y='36' text-anchor='middle' font-size='20' font-family='Arial'>{title}</text>",
    ]
    if subtitle:
        svg.append(
            f"<text x='{width/2}' y='58' text-anchor='middle' font-size='12' font-family='Arial' fill='#444'>{subtitle}</text>"
        )

    # Axes
    svg.append(
        f"<line x1='{chart_left}' y1='{chart_top + chart_h}' x2='{chart_left + chart_w}' y2='{chart_top + chart_h}' stroke='#333'/>"
    )
    svg.append(
        f"<line x1='{chart_left}' y1='{chart_top}' x2='{chart_left}' y2='{chart_top + chart_h}' stroke='#333'/>"
    )

    # Y ticks
    ticks = 5
    for i in range(ticks + 1):
        y = chart_top + chart_h * (1 - i / ticks)
        val = y_min + (y_max - y_min) * i / ticks
        svg.append(f"<line x1='{chart_left-5}' y1='{y}' x2='{chart_left}' y2='{y}' stroke='#333'/>")
        svg.append(
            f"<text x='{chart_left-10}' y='{y+4}' text-anchor='end' font-size='11' font-family='Arial'>{val:.1f}</text>"
        )

    # X ticks
    for x in x_values:
        xp = x_pos(x)
        svg.append(f"<line x1='{xp}' y1='{chart_top + chart_h}' x2='{xp}' y2='{chart_top + chart_h + 5}' stroke='#333'/>")
        svg.append(
            f"<text x='{xp}' y='{chart_top + chart_h + 20}' text-anchor='middle' font-size='11' font-family='Arial'>{x:g}</text>"
        )

    # Lines
    for model, points in series.items():
        color = MODEL_COLORS.get(model, "#222222")
        sorted_points = sorted(points.items())
        coords = [
            (x_pos(x), y_pos(y)) for x, y in sorted_points
        ]
        if not coords:
            continue
        path = " ".join(
            ["M"] + [f"{coords[0][0]:.2f},{coords[0][1]:.2f}"]
            + [f"L {x:.2f},{y:.2f}" for x, y in coords[1:]]
        )
        svg.append(f"<path d='{path}' fill='none' stroke='{color}' stroke-width='2'/>")
        for x, y in coords:
            svg.append(f"<circle cx='{x:.2f}' cy='{y:.2f}' r='3' fill='{color}'/>")

    # Labels
    svg.append(
        f"<text x='{width/2}' y='{height-30}' text-anchor='middle' font-size='13' font-family='Arial'>{x_label}</text>"
    )
    svg.append(
        f"<text x='20' y='{height/2}' text-anchor='middle' font-size='13' font-family='Arial' transform='rotate(-90 20 {height/2})'>{y_label}</text>"
    )

    # Legend
    legend_x = chart_left
    legend_y = 50
    gap = 180
    for i, model in enumerate(series.keys()):
        color = MODEL_COLORS.get(model, "#222222")
        lx = legend_x + i * gap
        svg.append(f"<rect x='{lx}' y='{legend_y}' width='14' height='14' fill='{color}'/>")
        svg.append(f"<text x='{lx+20}' y='{legend_y+12}' font-size='12' font-family='Arial'>{model}</text>")

    svg.append("</svg>")
    out_path.write_text("\n".join(svg), encoding="utf-8")


def render_dual_metric_plot(
    *,
    title: str,
    x_label: str,
    x_values: list[float],
    series_top1: dict[str, dict[float, float]],
    series_top5: dict[str, dict[float, float]],
    out_path: Path,
) -> None:
    width = SVG_WIDTH
    height = SVG_HEIGHT
    chart_left = MARGIN_LEFT
    chart_top = MARGIN_TOP
    chart_w = width - MARGIN_LEFT - MARGIN_RIGHT
    chart_h = height - MARGIN_TOP - MARGIN_BOTTOM

    all_vals = []
    for model in series_top1.values():
        all_vals.extend(model.values())
    for model in series_top5.values():
        all_vals.extend(model.values())
    y_min = 0.0
    y_max = max(all_vals) if all_vals else 1.0
    y_max *= 1.15

    def x_pos(x: float) -> float:
        if len(x_values) == 1:
            return chart_left + chart_w / 2
        x_min = min(x_values)
        x_max = max(x_values)
        if x_max == x_min:
            return chart_left + chart_w / 2
        return chart_left + (x - x_min) / (x_max - x_min) * chart_w

    def y_pos(y: float) -> float:
        return chart_top + chart_h - (y - y_min) / (y_max - y_min) * chart_h

    svg = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}'>",
        "<rect width='100%' height='100%' fill='white'/>",
        f"<text x='{width/2}' y='36' text-anchor='middle' font-size='20' font-family='Arial'>{title}</text>",
    ]

    # Axes
    svg.append(
        f"<line x1='{chart_left}' y1='{chart_top + chart_h}' x2='{chart_left + chart_w}' y2='{chart_top + chart_h}' stroke='#333'/>"
    )
    svg.append(
        f"<line x1='{chart_left}' y1='{chart_top}' x2='{chart_left}' y2='{chart_top + chart_h}' stroke='#333'/>"
    )

    # Y ticks
    ticks = 5
    for i in range(ticks + 1):
        y = chart_top + chart_h * (1 - i / ticks)
        val = y_min + (y_max - y_min) * i / ticks
        svg.append(f"<line x1='{chart_left-5}' y1='{y}' x2='{chart_left}' y2='{y}' stroke='#333'/>")
        svg.append(
            f"<text x='{chart_left-10}' y='{y+4}' text-anchor='end' font-size='11' font-family='Arial'>{val:.1f}</text>"
        )

    # X ticks
    for x in x_values:
        xp = x_pos(x)
        svg.append(f"<line x1='{xp}' y1='{chart_top + chart_h}' x2='{xp}' y2='{chart_top + chart_h + 5}' stroke='#333'/>")
        svg.append(
            f"<text x='{xp}' y='{chart_top + chart_h + 20}' text-anchor='middle' font-size='11' font-family='Arial'>{x:g}</text>"
        )

    # Lines: Top-1 (solid)
    for model, points in series_top1.items():
        color = MODEL_COLORS.get(model, "#222222")
        sorted_points = sorted(points.items())
        coords = [(x_pos(x), y_pos(y)) for x, y in sorted_points]
        if not coords:
            continue
        path = " ".join(
            ["M"] + [f"{coords[0][0]:.2f},{coords[0][1]:.2f}"]
            + [f"L {x:.2f},{y:.2f}" for x, y in coords[1:]]
        )
        svg.append(f"<path d='{path}' fill='none' stroke='{color}' stroke-width='2'/>")
        for x, y in coords:
            svg.append(f"<circle cx='{x:.2f}' cy='{y:.2f}' r='3' fill='{color}'/>")

    # Lines: Top-5 (dashed + lighter color)
    for model, points in series_top5.items():
        color = TOP5_COLORS.get(model, "#999999")
        sorted_points = sorted(points.items())
        coords = [(x_pos(x), y_pos(y)) for x, y in sorted_points]
        if not coords:
            continue
        path = " ".join(
            ["M"] + [f"{coords[0][0]:.2f},{coords[0][1]:.2f}"]
            + [f"L {x:.2f},{y:.2f}" for x, y in coords[1:]]
        )
        svg.append(
            f"<path d='{path}' fill='none' stroke='{color}' stroke-width='2' stroke-dasharray='6,4'/>"
        )
        for x, y in coords:
            svg.append(f"<circle cx='{x:.2f}' cy='{y:.2f}' r='3' fill='{color}'/>")

    # Labels
    svg.append(
        f"<text x='{width/2}' y='{height-30}' text-anchor='middle' font-size='13' font-family='Arial'>{x_label}</text>"
    )
    svg.append(
        f"<text x='20' y='{height/2}' text-anchor='middle' font-size='13' font-family='Arial' transform='rotate(-90 20 {height/2})'>Accuracy (%)</text>"
    )

    # Legend: model + metric
    legend_x = chart_left
    legend_y = 50
    gap = 190
    idx = 0
    for model in series_top1.keys():
        color = MODEL_COLORS.get(model, "#222222")
        lx = legend_x + (idx % 3) * gap
        ly = legend_y + (idx // 3) * 18
        svg.append(f"<rect x='{lx}' y='{ly}' width='14' height='14' fill='{color}'/>")
        svg.append(f"<text x='{lx+20}' y='{ly+12}' font-size='12' font-family='Arial'>{model} (Top-1)</text>")
        idx += 1
    for model in series_top5.keys():
        color = TOP5_COLORS.get(model, "#999999")
        lx = legend_x + (idx % 3) * gap
        ly = legend_y + (idx // 3) * 18
        svg.append(f"<rect x='{lx}' y='{ly}' width='14' height='14' fill='{color}'/>")
        svg.append(f"<text x='{lx+20}' y='{ly+12}' font-size='12' font-family='Arial'>{model} (Top-5)</text>")
        idx += 1

    svg.append("</svg>")
    out_path.write_text("\n".join(svg), encoding="utf-8")

def _extract_svg_body(svg_text: str) -> str:
    start = svg_text.find(">")
    end = svg_text.rfind("</svg>")
    if start == -1 or end == -1:
        return svg_text
    return svg_text[start + 1 : end].strip()


def render_top5_overview(out_dir: Path) -> Path:
    # Compose three heatmaps + two line plots into a single overview canvas.
    heatmaps = [
        out_dir / "accuracy_heatmap_mobilevit_xxs_top5.svg",
        out_dir / "accuracy_heatmap_mobilevit_xs_top5.svg",
        out_dir / "accuracy_heatmap_mobilevit_s_top5.svg",
    ]
    line_sigma = out_dir / "accuracy_top5_vs_sigma.svg"
    line_alpha = out_dir / "accuracy_top5_vs_alpha.svg"

    scale_heatmap = 0.45
    scale_line = 0.45
    item_w = SVG_WIDTH * scale_heatmap
    item_h = SVG_HEIGHT * scale_heatmap

    top_y = OVERVIEW_MARGIN + 30
    bottom_y = top_y + item_h + OVERVIEW_GAP

    x0 = OVERVIEW_MARGIN
    x1 = x0 + item_w + OVERVIEW_GAP
    x2 = x1 + item_w + OVERVIEW_GAP

    svg = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{OVERVIEW_WIDTH}' height='{OVERVIEW_HEIGHT}' viewBox='0 0 {OVERVIEW_WIDTH} {OVERVIEW_HEIGHT}'>",
        "<rect width='100%' height='100%' fill='white'/>",
        f"<text x='{OVERVIEW_WIDTH/2}' y='36' text-anchor='middle' font-size='22' font-family='Arial'>Top-5 accuracy overview</text>",
    ]

    for idx, path in enumerate(heatmaps):
        if not path.is_file():
            continue
        body = _extract_svg_body(path.read_text(encoding='utf-8'))
        x = [x0, x1, x2][idx]
        svg.append(f"<g transform='translate({x:.1f},{top_y:.1f}) scale({scale_heatmap})'>")
        svg.append(body)
        svg.append("</g>")

    # Line plots
    if line_sigma.is_file():
        body = _extract_svg_body(line_sigma.read_text(encoding='utf-8'))
        svg.append(f"<g transform='translate({x0:.1f},{bottom_y:.1f}) scale({scale_line})'>")
        svg.append(body)
        svg.append("</g>")

    if line_alpha.is_file():
        body = _extract_svg_body(line_alpha.read_text(encoding='utf-8'))
        svg.append(f"<g transform='translate({x1:.1f},{bottom_y:.1f}) scale({scale_line})'>")
        svg.append(body)
        svg.append("</g>")

    svg.append("</svg>")

    out_path = out_dir / "accuracy_top5_overview.svg"
    out_path.write_text("\n".join(svg), encoding="utf-8")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Render SVG plots from accuracy CSV.")
    parser.add_argument(
        "--csv",
        default="results/accuracy_noise_gpu.csv",
        help="Path to accuracy_noise CSV.",
    )
    parser.add_argument(
        "--out_dir",
        default="results/plots",
        help="Output directory for SVG plots.",
    )
    parser.add_argument(
        "--metric",
        default="top1",
        choices=["top1", "top5"],
        help="Metric for heatmaps.",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = load_noise_data(csv_path, metric=args.metric)
    models = payload["models"]
    sigmas = payload["sigmas"]
    alphas = payload["alphas"]
    data = payload["data"]
    baselines = payload["baselines"]

    sigma_label = "Gaussian noise sigma [LSB]"
    alpha_label = "Crosstalk level"

    # Heatmaps per model (Top-1 or Top-5)
    all_vals = []
    for m in models:
        for sigma in sigmas:
            for alpha in alphas:
                val = data.get(m, {}).get(sigma, {}).get(alpha)
                if val is not None:
                    all_vals.append(val)
    vmin = min(all_vals) if all_vals else 0.0
    vmax = max(all_vals) if all_vals else 1.0
    metric_label = "Top-1 (%)" if args.metric == "top1" else "Top-5 (%)"

    for m in models:
        out_path = out_dir / f"accuracy_heatmap_{m}_{args.metric}.svg"
        render_heatmap(
            model=m,
            sigmas=sigmas,
            alphas=alphas,
            data=data,
            baselines=baselines,
            vmin=vmin,
            vmax=vmax,
            metric_label=metric_label,
            out_path=out_path,
            sigma_label=sigma_label,
            alpha_label=alpha_label,
        )

    # Line plots
    if args.metric in {"top1", "top5"}:
        # Re-load raw rows for averaging
        with csv_path.open(newline="") as f:
            rows = list(csv.DictReader(f))
        noise_rows = [r for r in rows if r.get("notes") != "baseline"]

        by_model = {}
        for m in models:
            rows_m = [r for r in noise_rows if r["model"] == m]
            by_model[m] = rows_m

        metric_key = "top1" if args.metric == "top1" else "top5"

        def avg_by_key(rows_m, key):
            buckets: dict[float, list[float]] = {}
            for r in rows_m:
                buckets.setdefault(float(r[key]), []).append(float(r[metric_key]))
            return {k: sum(v) / len(v) for k, v in buckets.items()}

        series_sigma = {m: avg_by_key(rows_m, "noise_sigma_lsb") for m, rows_m in by_model.items()}
        series_alpha = {m: avg_by_key(rows_m, "crosstalk_alpha") for m, rows_m in by_model.items()}

        y_label = "Top-1 (%)" if args.metric == "top1" else "Top-5 (%)"
        title_prefix = "Top-1" if args.metric == "top1" else "Top-5"

        render_line_plot(
            title=f"{title_prefix} vs Gaussian noise sigma",
            subtitle=None,
            x_label="Gaussian noise sigma [LSB]",
            y_label=y_label,
            x_values=sigmas,
            series=series_sigma,
            out_path=out_dir / f"accuracy_{metric_key}_vs_sigma.svg",
        )
        render_line_plot(
            title=f"{title_prefix} vs Crosstalk",
            subtitle=None,
            x_label="Crosstalk level",
            y_label=y_label,
            x_values=alphas,
            series=series_alpha,
            out_path=out_dir / f"accuracy_{metric_key}_vs_alpha.svg",
        )

        # Combined Top-1/Top-5 plots for the same x-axis.
        if args.metric == "top1":
            # Load Top-5 series once from the same CSV.
            metric_key = "top5"
            series_sigma_top5 = {m: avg_by_key(rows_m, "noise_sigma_lsb") for m, rows_m in by_model.items()}
            series_alpha_top5 = {m: avg_by_key(rows_m, "crosstalk_alpha") for m, rows_m in by_model.items()}

            render_dual_metric_plot(
                title="Top-1 / Top-5 vs Gaussian noise sigma",
                x_label="Gaussian noise sigma [LSB]",
                x_values=sigmas,
                series_top1=series_sigma,
                series_top5=series_sigma_top5,
                out_path=out_dir / "accuracy_top1_top5_vs_sigma.svg",
            )
            render_dual_metric_plot(
                title="Top-1 / Top-5 vs Crosstalk",
                x_label="Crosstalk level",
                x_values=alphas,
                series_top1=series_alpha,
                series_top5=series_alpha_top5,
                out_path=out_dir / "accuracy_top1_top5_vs_alpha.svg",
            )

    print(f"Saved accuracy plots to {out_dir}")
    if args.metric == "top5":
        overview = render_top5_overview(out_dir)
        print(f"Saved top5 overview to {overview}")


if __name__ == "__main__":
    main()
