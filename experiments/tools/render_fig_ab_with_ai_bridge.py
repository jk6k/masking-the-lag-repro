"""Render Fig-A/Fig-B with an AI-architecture composition bridge.

Fig-A:
- Script-generated related-work radar chart from an explicit score table.

Fig-B:
- Architecture + data composite figure.
- Architecture area can be either:
  1) AI-generated image passed by --architecture_image
  2) Built-in fallback vector schematic (no external dependency)
- Data area is always script-generated from quick-report CSVs.
"""

from __future__ import annotations

import argparse
import base64
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

RADAR_AXES = [
    "dynamic_operand_support",
    "broadcast_cost_modeling",
    "early_stop",
    "sparse_power_reallocation",
    "phy_closure",
    "reproducibility",
]

DEFAULT_RADAR_ROWS = [
    {
        "work": "ASTRA",
        "dynamic_operand_support": 3,
        "broadcast_cost_modeling": 2,
        "early_stop": 1,
        "sparse_power_reallocation": 1,
        "phy_closure": 1,
        "reproducibility": 3,
    },
    {
        "work": "Lightening",
        "dynamic_operand_support": 5,
        "broadcast_cost_modeling": 2,
        "early_stop": 2,
        "sparse_power_reallocation": 1,
        "phy_closure": 1,
        "reproducibility": 3,
    },
    {
        "work": "Opto-ViT",
        "dynamic_operand_support": 4,
        "broadcast_cost_modeling": 2,
        "early_stop": 1,
        "sparse_power_reallocation": 4,
        "phy_closure": 2,
        "reproducibility": 2,
    },
    {
        "work": "SCATTER",
        "dynamic_operand_support": 3,
        "broadcast_cost_modeling": 3,
        "early_stop": 1,
        "sparse_power_reallocation": 5,
        "phy_closure": 1,
        "reproducibility": 3,
    },
    {
        "work": "ASCEND",
        "dynamic_operand_support": 2,
        "broadcast_cost_modeling": 4,
        "early_stop": 1,
        "sparse_power_reallocation": 1,
        "phy_closure": 4,
        "reproducibility": 4,
    },
    {
        "work": "Masking-the-Lag",
        "dynamic_operand_support": 5,
        "broadcast_cost_modeling": 5,
        "early_stop": 5,
        "sparse_power_reallocation": 5,
        "phy_closure": 5,
        "reproducibility": 5,
    },
]

COLORS = [
    "#4e79a7",
    "#f28e2b",
    "#59a14f",
    "#e15759",
    "#76b7b2",
    "#b07aa1",
    "#edc948",
]


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fieldnames})


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


def _safe_norm(v: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 0.5
    return (v - lo) / (hi - lo)


def _save_svg(path: Path, body: list[str], width: int, height: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}'>",
        "<rect width='100%' height='100%' fill='white'/>",
        *body,
        "</svg>",
    ]
    path.write_text("\n".join(text), encoding="utf-8")


def _ensure_radar_csv(path: Path) -> None:
    if path.exists():
        return
    _write_csv(path, ["work", *RADAR_AXES], DEFAULT_RADAR_ROWS)


def render_fig_a_radar(radar_csv: Path, out_svg: Path) -> None:
    _ensure_radar_csv(radar_csv)
    rows = _read_csv(radar_csv)
    if not rows:
        return

    w, h = 1240, 860
    cx, cy = 430, 450
    radius = 280
    levels = 5

    body: list[str] = []
    body.append(
        "<text x='620' y='42' text-anchor='middle' font-size='24' font-family='Arial'>Fig-A: Related-Work Radar (Script-Generated)</text>"
    )

    # radar grid
    n = len(RADAR_AXES)
    angles = [(-math.pi / 2) + i * 2 * math.pi / n for i in range(n)]
    for lv in range(1, levels + 1):
        r = radius * lv / levels
        pts = []
        for a in angles:
            x = cx + r * math.cos(a)
            y = cy + r * math.sin(a)
            pts.append(f"{x:.2f},{y:.2f}")
        body.append(
            f"<polygon points='{' '.join(pts)}' fill='none' stroke='#d0d0d0' stroke-width='1'/>"
        )
        body.append(
            f"<text x='{cx + 4}' y='{cy - r + 14:.2f}' font-size='10' font-family='Arial' fill='#666'>{lv}</text>"
        )

    for idx, axis in enumerate(RADAR_AXES):
        a = angles[idx]
        x = cx + radius * math.cos(a)
        y = cy + radius * math.sin(a)
        body.append(f"<line x1='{cx}' y1='{cy}' x2='{x:.2f}' y2='{y:.2f}' stroke='#b0b0b0' stroke-width='1'/>")
        lx = cx + (radius + 48) * math.cos(a)
        ly = cy + (radius + 48) * math.sin(a)
        label = axis.replace("_", " ")
        body.append(
            f"<text x='{lx:.2f}' y='{ly:.2f}' text-anchor='middle' font-size='13' font-family='Arial'>{label}</text>"
        )

    # series
    legend_x = 820
    legend_y = 120
    for i, row in enumerate(rows):
        color = COLORS[i % len(COLORS)]
        pts = []
        for j, axis in enumerate(RADAR_AXES):
            v = max(0.0, min(5.0, _f(row.get(axis), 0.0)))
            r = radius * (v / 5.0)
            a = angles[j]
            x = cx + r * math.cos(a)
            y = cy + r * math.sin(a)
            pts.append(f"{x:.2f},{y:.2f}")
        body.append(
            f"<polygon points='{' '.join(pts)}' fill='{color}' fill-opacity='0.12' stroke='{color}' stroke-width='2'/>"
        )
        ly = legend_y + i * 28
        body.append(f"<rect x='{legend_x}' y='{ly-12}' width='14' height='14' fill='{color}'/>")
        body.append(
            f"<text x='{legend_x+22}' y='{ly}' font-size='14' font-family='Arial'>{row.get('work','')}</text>"
        )

    body.append(
        "<text x='820' y='330' font-size='12' font-family='Arial' fill='#555'>Scale: 0 (absent) to 5 (strong support).</text>"
    )
    body.append(
        "<text x='820' y='350' font-size='12' font-family='Arial' fill='#555'>Scores are explicit and editable in fig_a_related_work_radar_scores.csv.</text>"
    )

    _save_svg(out_svg, body, w, h)


def _card_frame(body: list[str], x: float, y: float, w: float, h: float, title: str) -> None:
    body.append(f"<rect x='{x}' y='{y}' width='{w}' height='{h}' fill='#fafafa' stroke='#d8d8d8'/>")
    body.append(f"<text x='{x+12}' y='{y+22}' font-size='13' font-family='Arial'>{title}</text>")


def _draw_fallback_architecture(body: list[str], x: float, y: float, w: float, h: float) -> None:
    body.append(f"<rect x='{x}' y='{y}' width='{w}' height='{h}' fill='#ffffff' stroke='#cfcfcf'/>")
    body.append(
        f"<text x='{x + w/2}' y='{y+28}' text-anchor='middle' font-size='18' font-family='Arial'>Masking-the-Lag: Overall Architecture</text>"
    )

    gx = x + 20
    gy = y + 52
    gw = w - 40
    gh = h - 80

    # group backgrounds
    body.append(f"<rect x='{gx}' y='{gy}' width='{gw*0.28:.2f}' height='{gh}' fill='#f7f7f7' stroke='#d0d0d0'/>")
    body.append(
        f"<rect x='{gx + gw*0.28:.2f}' y='{gy}' width='{gw*0.45:.2f}' height='{gh}' fill='#eef6ff' stroke='#d0d0d0'/>"
    )
    body.append(
        f"<rect x='{gx + gw*0.73:.2f}' y='{gy}' width='{gw*0.27:.2f}' height='{gh}' fill='#f7f7f7' stroke='#d0d0d0'/>"
    )
    body.append(f"<text x='{gx+8}' y='{gy+18}' font-size='12' font-family='Arial'>Electronic Frontend</text>")
    body.append(
        f"<text x='{gx + gw*0.28 + 8:.2f}' y='{gy+18}' font-size='12' font-family='Arial'>Optical Core</text>"
    )
    body.append(
        f"<text x='{gx + gw*0.73 + 8:.2f}' y='{gy+18}' font-size='12' font-family='Arial'>Electronic Backend</text>"
    )

    # module boxes
    mods = [
        ("1 Workload Mapper", gx + 20, gy + 52, 170, 76),
        ("2 DET-Aware BtoS", gx + 20, gy + 170, 170, 76),
        ("3 MESO Broadcast", gx + gw * 0.28 + 20, gy + 52, 190, 76),
        ("4 HOPS Scheduler", gx + gw * 0.28 + 20, gy + 170, 190, 76),
        ("5 Photonic Compute", gx + gw * 0.28 + 20, gy + 288, 190, 76),
        ("6 O/E + ADC/PCA", gx + gw * 0.73 + 24, gy + 170, 170, 76),
    ]
    for name, mx, my, mw, mh in mods:
        body.append(f"<rect x='{mx}' y='{my}' width='{mw}' height='{mh}' fill='white' stroke='#555'/>")
        body.append(
            f"<text x='{mx+mw/2}' y='{my+43}' text-anchor='middle' font-size='12' font-family='Arial'>{name}</text>"
        )

    # data-plane arrows
    arrows = [
        (mods[0][1] + 170, mods[0][2] + 38, mods[2][1], mods[2][2] + 38),
        (mods[2][1] + 190, mods[2][2] + 38, mods[5][1], mods[5][2] + 38),
        (mods[1][1] + 170, mods[1][2] + 38, mods[3][1], mods[3][2] + 38),
        (mods[3][1] + 190, mods[3][2] + 38, mods[5][1], mods[5][2] + 38),
        (mods[4][1] + 190, mods[4][2] + 38, mods[5][1], mods[5][2] + 60),
    ]
    for x1, y1, x2, y2 in arrows:
        body.append(f"<line x1='{x1}' y1='{y1}' x2='{x2}' y2='{y2}' stroke='#222' stroke-width='2' marker-end='url(#arrow)'/>")

    # loops
    body.append(
        f"<path d='M {mods[5][1]+170} {mods[5][2]+15} C {gx+gw-10} {gy+40}, {gx+gw-10} {gy+gh-20}, {mods[1][1]+80} {mods[1][2]+76}' fill='none' stroke='#1f77b4' stroke-width='2' stroke-dasharray='6,4' marker-end='url(#arrow_blue)'/>"
    )
    body.append(
        f"<path d='M {mods[5][1]+170} {mods[5][2]+60} C {gx+gw-30} {gy+gh+10}, {gx+gw*0.55} {gy+gh+10}, {mods[2][1]+100} {mods[2][2]+76}' fill='none' stroke='#ff7f0e' stroke-width='2' stroke-dasharray='6,4' marker-end='url(#arrow_orange)'/>"
    )
    body.append(
        f"<text x='{gx+gw-170}' y='{gy+90}' font-size='11' font-family='Arial' fill='#1f77b4'>Accuracy loop</text>"
    )
    body.append(
        f"<text x='{gx+gw-170}' y='{gy+gh-8}' font-size='11' font-family='Arial' fill='#ff7f0e'>Power loop</text>"
    )


def _data_uri_for_image(path: Path) -> str | None:
    if not path.exists():
        return None
    suffix = path.suffix.lower()
    if suffix in {".png"}:
        mime = "image/png"
    elif suffix in {".jpg", ".jpeg"}:
        mime = "image/jpeg"
    elif suffix == ".svg":
        mime = "image/svg+xml"
    else:
        return None
    raw = path.read_bytes()
    b64 = base64.b64encode(raw).decode("ascii")
    return f"data:{mime};base64,{b64}"


def _draw_line_inset(
    body: list[str],
    *,
    x: float,
    y: float,
    w: float,
    h: float,
    xs: list[float],
    ys: list[float],
    color: str,
    x_label: str,
    y_label: str,
) -> None:
    if not xs or not ys:
        return
    px, py = x + 42, y + 30
    pw, ph = w - 60, h - 56
    body.append(f"<line x1='{px}' y1='{py+ph}' x2='{px+pw}' y2='{py+ph}' stroke='#333'/>")
    body.append(f"<line x1='{px}' y1='{py}' x2='{px}' y2='{py+ph}' stroke='#333'/>")
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    if math.isclose(ymin, ymax):
        ymax = ymin + 1.0

    def xp(v: float) -> float:
        return px + pw * _safe_norm(v, xmin, xmax)

    def yp(v: float) -> float:
        return py + ph * (1 - _safe_norm(v, ymin, ymax))

    pts = [(xp(a), yp(b)) for a, b in zip(xs, ys, strict=True)]
    path = " ".join(f"{xx:.2f},{yy:.2f}" for xx, yy in pts)
    body.append(f"<polyline points='{path}' fill='none' stroke='{color}' stroke-width='2'/>")
    for xx, yy in pts:
        body.append(f"<circle cx='{xx:.2f}' cy='{yy:.2f}' r='2.8' fill='{color}'/>")
    body.append(f"<text x='{x+w/2}' y='{y+h-8}' text-anchor='middle' font-size='10' font-family='Arial'>{x_label}</text>")
    body.append(
        f"<text x='{x+10}' y='{y+h/2}' text-anchor='middle' font-size='10' font-family='Arial' transform='rotate(-90 {x+10} {y+h/2})'>{y_label}</text>"
    )


def _draw_scatter_inset(
    body: list[str],
    *,
    x: float,
    y: float,
    w: float,
    h: float,
    points: list[tuple[float, float, str]],
) -> None:
    if not points:
        return
    px, py = x + 42, y + 30
    pw, ph = w - 60, h - 56
    body.append(f"<line x1='{px}' y1='{py+ph}' x2='{px+pw}' y2='{py+ph}' stroke='#333'/>")
    body.append(f"<line x1='{px}' y1='{py}' x2='{px}' y2='{py+ph}' stroke='#333'/>")
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    if math.isclose(xmin, xmax):
        xmax = xmin + 1.0
    if math.isclose(ymin, ymax):
        ymax = ymin + 1.0

    def xp(v: float) -> float:
        return px + pw * _safe_norm(v, xmin, xmax)

    def yp(v: float) -> float:
        return py + ph * (1 - _safe_norm(v, ymin, ymax))

    for idx, (xx, yy, label) in enumerate(points):
        cx, cy = xp(xx), yp(yy)
        col = COLORS[idx % len(COLORS)]
        body.append(f"<circle cx='{cx:.2f}' cy='{cy:.2f}' r='4' fill='{col}'/>")
        body.append(f"<text x='{cx+8:.2f}' y='{cy-4:.2f}' font-size='10' font-family='Arial'>{label}</text>")

    body.append(
        f"<text x='{x+w/2}' y='{y+h-8}' text-anchor='middle' font-size='10' font-family='Arial'>acc_drop_pp_mean</text>"
    )
    body.append(
        f"<text x='{x+10}' y='{y+h/2}' text-anchor='middle' font-size='10' font-family='Arial' transform='rotate(-90 {x+10} {y+h/2})'>energy_saved_pct</text>"
    )


def render_fig_b_composite(
    *,
    quick_reports_dir: Path,
    run_prefix: str,
    out_svg: Path,
    architecture_image: Path | None,
) -> None:
    overview = _read_csv(quick_reports_dir / "quickpack_e0_e6_overview.csv")
    fig_d = _read_csv(quick_reports_dir / "fig_d_prefix_error_vs_k.csv")
    fig_j = _read_csv(quick_reports_dir / "fig_j_sparse_tau_pareto.csv")

    w, h = 1600, 900
    body: list[str] = []

    # marker defs
    body.append("<defs>")
    body.append("<marker id='arrow' markerWidth='8' markerHeight='8' refX='7' refY='3' orient='auto'><path d='M0,0 L8,3 L0,6 z' fill='#222'/></marker>")
    body.append("<marker id='arrow_blue' markerWidth='8' markerHeight='8' refX='7' refY='3' orient='auto'><path d='M0,0 L8,3 L0,6 z' fill='#1f77b4'/></marker>")
    body.append("<marker id='arrow_orange' markerWidth='8' markerHeight='8' refX='7' refY='3' orient='auto'><path d='M0,0 L8,3 L0,6 z' fill='#ff7f0e'/></marker>")
    body.append("</defs>")

    body.append(
        "<text x='800' y='40' text-anchor='middle' font-size='24' font-family='Arial'>Fig-B: Overall Architecture + Data Insets</text>"
    )

    arch_x, arch_y, arch_w, arch_h = 30, 70, 1000, 800
    in_x, in_y, in_w, in_h = 1050, 70, 520, 800

    # architecture panel
    if architecture_image and architecture_image.exists():
        uri = _data_uri_for_image(architecture_image)
        if uri:
            body.append(f"<rect x='{arch_x}' y='{arch_y}' width='{arch_w}' height='{arch_h}' fill='white' stroke='#c8c8c8'/>")
            body.append(
                f"<image href='{uri}' x='{arch_x+8}' y='{arch_y+8}' width='{arch_w-16}' height='{arch_h-16}' preserveAspectRatio='xMidYMid meet'/>"
            )
            body.append(
                f"<text x='{arch_x+12}' y='{arch_y+24}' font-size='11' font-family='Arial' fill='#666'>Architecture source: {architecture_image.name}</text>"
            )
        else:
            _draw_fallback_architecture(body, arch_x, arch_y, arch_w, arch_h)
    else:
        _draw_fallback_architecture(body, arch_x, arch_y, arch_w, arch_h)

    # data inset panel
    body.append(f"<rect x='{in_x}' y='{in_y}' width='{in_w}' height='{in_h}' fill='#ffffff' stroke='#c8c8c8'/>")
    body.append(f"<text x='{in_x+12}' y='{in_y+24}' font-size='14' font-family='Arial'>Data Insets (from quick reports)</text>")

    # inset1: speedup over experiments
    _card_frame(body, in_x + 12, in_y + 38, in_w - 24, 230, "Inset-1: E0-E6 speedup_vs_E0")
    if overview:
        o = sorted(overview, key=lambda r: str(r.get("experiment_id")))
        xs = [i for i in range(len(o))]
        ys = [_f(r.get("speedup_vs_E0")) for r in o]
        _draw_line_inset(
            body,
            x=in_x + 12,
            y=in_y + 38,
            w=in_w - 24,
            h=230,
            xs=xs,
            ys=ys,
            color=COLORS[0],
            x_label="E0..E6 index",
            y_label="speedup_vs_E0",
        )
        for idx, r in enumerate(o):
            tx = in_x + 54 + (in_w - 84) * _safe_norm(float(idx), 0.0, float(max(1, len(o) - 1)))
            body.append(
                f"<text x='{tx:.2f}' y='{in_y+258}' text-anchor='middle' font-size='9' font-family='Arial'>{r.get('experiment_id')}</text>"
            )

    # inset2: prefix error
    _card_frame(body, in_x + 12, in_y + 282, in_w - 24, 220, "Inset-2: DET prefix error (Fig-D data)")
    if fig_d:
        d = sorted(fig_d, key=lambda r: _f(r.get("k")))
        xs = [_f(r.get("k")) for r in d]
        ys = [_f(r.get("prefix_error_mean")) for r in d]
        _draw_line_inset(
            body,
            x=in_x + 12,
            y=in_y + 282,
            w=in_w - 24,
            h=220,
            xs=xs,
            ys=ys,
            color=COLORS[3],
            x_label="k",
            y_label="prefix_error_mean",
        )

    # inset3: sparse tau pareto
    _card_frame(body, in_x + 12, in_y + 516, in_w - 24, 340, "Inset-3: SPARSE tau Pareto (Fig-J data)")
    if fig_j:
        pts = []
        for r in sorted(fig_j, key=lambda rr: _f(rr.get("tau"))):
            pts.append((_f(r.get("acc_drop_pp_mean")), _f(r.get("energy_saved_pct")), f"tau={_f(r.get('tau')):.2g}"))
        _draw_scatter_inset(
            body,
            x=in_x + 12,
            y=in_y + 516,
            w=in_w - 24,
            h=340,
            points=pts,
        )

    _save_svg(out_svg, body, w, h)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render Fig-A/Fig-B with AI bridge.")
    parser.add_argument("--run_prefix", default="quickpack_accfix_20260212")
    parser.add_argument(
        "--quick_reports_dir",
        default="experiments/results/quick_reports/final_paper_v2",
    )
    parser.add_argument(
        "--out_dir",
        default="experiments/results/plots/guidance_final_paper_v2",
    )
    parser.add_argument(
        "--architecture_image",
        default="",
        help="Optional AI-generated architecture image path (png/jpg/svg).",
    )
    args = parser.parse_args()

    out_dir = resolve_workspace_path(args.out_dir, anchor=ROOT)
    out_dir.mkdir(parents=True, exist_ok=True)
    quick_dir = resolve_workspace_path(args.quick_reports_dir, anchor=ROOT)

    radar_csv = quick_dir / "fig_a_related_work_radar_scores.csv"
    fig_a_svg = out_dir / "fig_a_related_work_radar.svg"
    render_fig_a_radar(radar_csv, fig_a_svg)

    arch_img = Path(args.architecture_image).resolve() if str(args.architecture_image).strip() else None
    fig_b_svg = out_dir / "fig_b_architecture_data_composite.svg"
    render_fig_b_composite(
        quick_reports_dir=quick_dir,
        run_prefix=args.run_prefix,
        out_svg=fig_b_svg,
        architecture_image=arch_img,
    )

    manifest = {
        "run_prefix": args.run_prefix,
        "quick_reports_dir": str(quick_dir),
        "fig_a_svg": str(fig_a_svg),
        "fig_b_svg": str(fig_b_svg),
        "radar_score_csv": str(radar_csv),
        "architecture_image": str(arch_img) if arch_img else None,
    }
    (out_dir / "fig_ab_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"[fig-ab] saved {fig_a_svg}")
    print(f"[fig-ab] saved {fig_b_svg}")
    print(f"[fig-ab] saved radar score table {radar_csv}")


if __name__ == "__main__":
    main()
