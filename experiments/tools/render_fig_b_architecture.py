#!/usr/bin/env python3
"""Render Fig-B (paper Fig.1): architecture overview.

This renderer enforces the design constraints from §3.6:
- Data plane must be: 1 -> 2 -> 3 -> 4 -> 5 -> 6 (solid arrows)
- Module 7 is sidecar control plane (not in serial data chain)
- Three domains are preserved:
  Electronic Frontend | Optical Core | Electronic Backend
- Internal glyph semantics:
  module-3 fanout/WDM, module-5 array, module-6 PD->TIA->ADC->PCA
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle


COLORS = {
    "bg_front": "#eef3fa",
    "bg_optical": "#edf8ee",
    "bg_back": "#fcf2e8",
    "domain_border": "#b8c0cc",
    "data": "#2f3437",
    "acc": "#4e79a7",
    "power": "#b07aa1",
    "wm": "#d6e4f0",
    "det": "#f8d7da",
    "meso": "#d9efdc",
    "flow": "#fde8cd",
    "compute": "#d8efec",
    "backend": "#fef3e2",
    "phy": "#efe5f7",
    "det_red": "#e15759",
    "meso_green": "#59a14f",
    "flow_orange": "#f28e2b",
    "sparse_cyan": "#76b7b2",
    "phy_purple": "#b07aa1",
}


MODULES = {
    "m1": {
        "num": 1,
        "title": "Workload Mapper",
        "subtitle": "layer mapping + tile allocation",
        "x": 0.07,
        "y": 0.63,
        "w": 0.17,
        "h": 0.17,
        "face": COLORS["wm"],
        "accent": "#4e79a7",
    },
    "m2": {
        "num": 2,
        "title": "DET-Aware BtoS Frontend",
        "subtitle": "baseline / LD switchable",
        "x": 0.07,
        "y": 0.37,
        "w": 0.17,
        "h": 0.17,
        "face": COLORS["det"],
        "accent": COLORS["det_red"],
    },
    "m3": {
        "num": 3,
        "title": "MESO Broadcast Fabric",
        "subtitle": "hierarchical fanout + WDM",
        "x": 0.34,
        "y": 0.63,
        "w": 0.20,
        "h": 0.17,
        "face": COLORS["meso"],
        "accent": COLORS["meso_green"],
    },
    "m4": {
        "num": 4,
        "title": "HOPS Scheduler",
        "subtitle": "Q/K/V and K^T overlap",
        "x": 0.34,
        "y": 0.37,
        "w": 0.20,
        "h": 0.17,
        "face": COLORS["flow"],
        "accent": COLORS["flow_orange"],
    },
    "m5": {
        "num": 5,
        "title": "Photonic Compute Cluster",
        "subtitle": "OAG / OSSM / VDP array",
        "x": 0.59,
        "y": 0.50,
        "w": 0.20,
        "h": 0.22,
        "face": COLORS["compute"],
        "accent": COLORS["sparse_cyan"],
    },
    "m6": {
        "num": 6,
        "title": "O/E + ADC/PCA Backend",
        "subtitle": "detect and domain conversion",
        "x": 0.81,
        "y": 0.50,
        "w": 0.15,
        "h": 0.22,
        "face": COLORS["backend"],
        "accent": "#c4a570",
    },
    "m7": {
        "num": 7,
        "title": "PHY Closure Manager",
        "subtitle": "Loss + PP_xtalk + BER -> P_laser",
        "x": 0.80,
        "y": 0.21,
        "w": 0.16,
        "h": 0.18,
        "face": COLORS["phy"],
        "accent": COLORS["phy_purple"],
    },
}


def _center(box: Dict[str, float]) -> Tuple[float, float]:
    return box["x"] + box["w"] * 0.5, box["y"] + box["h"] * 0.5


def _edge(box: Dict[str, float], side: str) -> Tuple[float, float]:
    cx, cy = _center(box)
    if side == "top":
        return cx, box["y"] + box["h"]
    if side == "bottom":
        return cx, box["y"]
    if side == "left":
        return box["x"], cy
    if side == "right":
        return box["x"] + box["w"], cy
    raise ValueError(f"Unknown side: {side}")


def _add_domain(ax, x, y, w, h, label, face, label_color):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.01,rounding_size=0.02",
        linewidth=1.2,
        edgecolor=COLORS["domain_border"],
        facecolor=face,
        zorder=0,
    )
    ax.add_patch(patch)
    ax.text(
        x + w * 0.5,
        y + h - 0.025,
        label,
        ha="center",
        va="top",
        fontsize=10,
        fontweight="bold",
        color=label_color,
    )


def _add_module(ax, module: Dict[str, float]) -> None:
    x, y, w, h = module["x"], module["y"], module["w"], module["h"]
    accent = Rectangle((x, y), 0.012, h, facecolor=module["accent"], edgecolor="none", zorder=3)
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.006,rounding_size=0.012",
        linewidth=1.1,
        edgecolor="#6d7480",
        facecolor=module["face"],
        zorder=2,
    )
    ax.add_patch(box)
    ax.add_patch(accent)

    badge = Circle((x + 0.026, y + h - 0.027), 0.0145, facecolor="white", edgecolor=module["accent"], linewidth=1.4, zorder=4)
    ax.add_patch(badge)
    ax.text(x + 0.026, y + h - 0.027, str(module["num"]), ha="center", va="center", fontsize=9.0, fontweight="bold", color=module["accent"], zorder=5)

    ax.text(x + 0.048, y + h - 0.020, module["title"], ha="left", va="top", fontsize=9.3, fontweight="bold", color="#1f2530", zorder=5)
    ax.text(x + 0.048, y + h - 0.052, module["subtitle"], ha="left", va="top", fontsize=8.1, color="#374151", zorder=5)


def _add_arrow(
    ax,
    p0: Tuple[float, float],
    p1: Tuple[float, float],
    color: str,
    lw: float = 2.0,
    linestyle: str = "solid",
    rad: float = 0.0,
    zorder: int = 4,
) -> None:
    arrow = FancyArrowPatch(
        p0,
        p1,
        arrowstyle="-|>",
        mutation_scale=12,
        linewidth=lw,
        color=color,
        linestyle=linestyle,
        connectionstyle=f"arc3,rad={rad}",
        shrinkA=2,
        shrinkB=2,
        zorder=zorder,
    )
    ax.add_patch(arrow)


def _draw_det_glyph(ax, box: Dict[str, float]) -> None:
    x, y, w, h = box["x"], box["y"], box["w"], box["h"]
    y0 = y + 0.035
    ax.plot(
        [x + 0.022, x + 0.038, x + 0.038, x + 0.054, x + 0.054, x + 0.070, x + 0.070, x + 0.086],
        [y0, y0, y0 + 0.020, y0 + 0.020, y0 + 0.040, y0 + 0.040, y0 + 0.060, y0 + 0.060],
        color="#374151",
        lw=1.0,
        zorder=5,
    )
    for idx, bar_h in enumerate([0.018, 0.034, 0.050]):
        bx = x + 0.105 + idx * 0.013
        ax.add_patch(Rectangle((bx, y0), 0.0075, bar_h, facecolor=COLORS["det_red"], edgecolor="none", zorder=5))
    ax.text(x + 0.145, y0 + 0.055, "early-stop", fontsize=6.6, color=COLORS["det_red"], ha="center", va="center", zorder=5)


def _draw_meso_glyph(ax, box: Dict[str, float]) -> None:
    x, y, w, h = box["x"], box["y"], box["w"], box["h"]
    root = (x + 0.055, y + 0.060)
    mids = [(x + 0.095, y + 0.095), (x + 0.095, y + 0.040)]
    leaves = [
        (x + 0.135, y + 0.112),
        (x + 0.135, y + 0.078),
        (x + 0.135, y + 0.056),
        (x + 0.135, y + 0.026),
    ]
    for m in mids:
        ax.plot([root[0], m[0]], [root[1], m[1]], color=COLORS["meso_green"], lw=1.2, zorder=5)
    for m, l in zip([mids[0], mids[0], mids[1], mids[1]], leaves):
        ax.plot([m[0], l[0]], [m[1], l[1]], color=COLORS["meso_green"], lw=1.2, zorder=5)
        ax.add_patch(Circle(l, 0.0038, facecolor=COLORS["meso_green"], edgecolor="none", zorder=5))
    ax.text(x + 0.162, y + 0.038, "lambda1..N", fontsize=6.5, color=COLORS["meso_green"], ha="left", va="center", zorder=5)


def _draw_flow_glyph(ax, box: Dict[str, float]) -> None:
    x, y, w, h = box["x"], box["y"], box["w"], box["h"]
    lane_y = y + 0.043
    ax.add_patch(Rectangle((x + 0.025, lane_y + 0.026), 0.070, 0.016, facecolor="#8bb7df", edgecolor="#5d88b3", linewidth=0.7, zorder=5))
    ax.add_patch(Rectangle((x + 0.090, lane_y), 0.082, 0.016, facecolor="#f5c58a", edgecolor="#d59549", linewidth=0.7, zorder=5))
    ax.add_patch(Rectangle((x + 0.126, lane_y + 0.026), 0.046, 0.016, facecolor="#8bb7df", edgecolor="#5d88b3", linewidth=0.7, zorder=5))
    ax.text(x + 0.058, lane_y + 0.034, "Q", fontsize=6.2, ha="center", va="center", color="#1f2937", zorder=6)
    ax.text(x + 0.131, lane_y + 0.008, "K^T", fontsize=6.2, ha="center", va="center", color="#1f2937", zorder=6)
    ax.text(x + 0.149, lane_y + 0.034, "V", fontsize=6.2, ha="center", va="center", color="#1f2937", zorder=6)


def _draw_compute_glyph(ax, box: Dict[str, float]) -> None:
    x, y, w, h = box["x"], box["y"], box["w"], box["h"]
    x0 = x + 0.030
    y0 = y + 0.040
    dx, dy = 0.026, 0.024
    for row in range(4):
        for col in range(5):
            cx = x0 + col * dx
            cy = y0 + row * dy
            ax.add_patch(Circle((cx, cy), 0.0055, facecolor="#ffffff", edgecolor="#4f8f88", linewidth=0.8, zorder=5))
    ax.text(x + 0.162, y + 0.042, "4x5 core array", fontsize=6.4, color="#2f6f68", ha="left", va="center", zorder=5)


def _draw_backend_glyph(ax, box: Dict[str, float]) -> None:
    x, y, w, h = box["x"], box["y"], box["w"], box["h"]
    labels = ["PD", "TIA", "ADC", "PCA"]
    bx = x + 0.020
    by = y + 0.050
    bw = 0.024
    bh = 0.026
    gap = 0.008
    for i, label in enumerate(labels):
        px = bx + i * (bw + gap)
        ax.add_patch(Rectangle((px, by), bw, bh, facecolor="#ffffff", edgecolor="#7a6b56", linewidth=0.8, zorder=5))
        ax.text(px + bw * 0.5, by + bh * 0.5, label, fontsize=6.1, ha="center", va="center", color="#4b5563", zorder=6)
        if i < len(labels) - 1:
            _add_arrow(ax, (px + bw, by + bh * 0.5), (px + bw + gap, by + bh * 0.5), color="#8a6d3b", lw=0.9, zorder=5)


def _draw_phy_glyph(ax, box: Dict[str, float]) -> None:
    x, y, w, h = box["x"], box["y"], box["w"], box["h"]
    txt = ["Loss", "PP_xtalk", "BER", "P_laser"]
    for i, token in enumerate(txt):
        ax.text(x + 0.028, y + 0.038 + i * 0.025, token, fontsize=6.7, ha="left", va="center", color="#5b4f73", zorder=5)
    ax.text(x + w - 0.028, y + 0.073, "=>", fontsize=8.4, ha="right", va="center", color="#5b4f73", zorder=5)


def build_figure():
    fig = plt.figure(figsize=(14.8, 8.2))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    _add_domain(ax, 0.04, 0.12, 0.26, 0.76, "Electronic Frontend", COLORS["bg_front"], "#2b5d8a")
    _add_domain(ax, 0.32, 0.12, 0.44, 0.76, "Optical Core", COLORS["bg_optical"], "#2d6e2d")
    _add_domain(ax, 0.78, 0.12, 0.18, 0.76, "Electronic Backend", COLORS["bg_back"], "#8a6523")

    # Horizontal die split keeps "system architecture" semantics and avoids flowchart look.
    die_y = 0.47
    ax.plot([0.04, 0.96], [die_y, die_y], color="#96a0ad", lw=1.4, zorder=1)
    ax.text(0.045, 0.81, "Electronic Die (CMOS)", ha="left", va="center", fontsize=9.3, color="#4b5563")
    ax.text(0.045, 0.205, "Photonic Die (PIC)", ha="left", va="center", fontsize=9.3, color="#4b5563")

    for module in MODULES.values():
        _add_module(ax, module)

    _draw_det_glyph(ax, MODULES["m2"])
    _draw_meso_glyph(ax, MODULES["m3"])
    _draw_flow_glyph(ax, MODULES["m4"])
    _draw_compute_glyph(ax, MODULES["m5"])
    _draw_backend_glyph(ax, MODULES["m6"])
    _draw_phy_glyph(ax, MODULES["m7"])

    # Data plane: fixed 1 -> 2 -> 3 -> 4 -> 5 -> 6
    _add_arrow(ax, _edge(MODULES["m1"], "bottom"), _edge(MODULES["m2"], "top"), color=COLORS["data"], lw=2.8)
    _add_arrow(ax, _edge(MODULES["m2"], "right"), _edge(MODULES["m3"], "left"), color=COLORS["data"], lw=2.8, rad=0.06)
    _add_arrow(ax, _edge(MODULES["m3"], "bottom"), _edge(MODULES["m4"], "top"), color=COLORS["data"], lw=2.8)
    _add_arrow(ax, _edge(MODULES["m4"], "right"), _edge(MODULES["m5"], "left"), color=COLORS["data"], lw=2.8, rad=-0.06)
    _add_arrow(ax, _edge(MODULES["m5"], "right"), _edge(MODULES["m6"], "left"), color=COLORS["data"], lw=2.8)

    # Accuracy control loop: 6 -> 7 -> (2,4,5)
    dashed = (0, (6, 4))
    _add_arrow(ax, _edge(MODULES["m6"], "bottom"), _edge(MODULES["m7"], "top"), color=COLORS["acc"], lw=1.8, linestyle=dashed, rad=0.0)
    _add_arrow(ax, _edge(MODULES["m7"], "left"), _edge(MODULES["m2"], "bottom"), color=COLORS["acc"], lw=1.6, linestyle=dashed, rad=-0.38)
    _add_arrow(ax, _edge(MODULES["m7"], "left"), _edge(MODULES["m4"], "bottom"), color=COLORS["acc"], lw=1.6, linestyle=dashed, rad=-0.24)
    _add_arrow(ax, _edge(MODULES["m7"], "top"), _edge(MODULES["m5"], "bottom"), color=COLORS["acc"], lw=1.6, linestyle=dashed, rad=-0.14)
    ax.text(0.76, 0.455, "Eval Accuracy(6) -> calibrate@7", fontsize=8.1, color=COLORS["acc"], ha="right", va="center")
    ax.text(0.66, 0.128, "writeback to 2 / 4 / 5", fontsize=8.0, color=COLORS["acc"], ha="center", va="center")

    # Power control loop: 5 -> 7 -> (3,5)
    _add_arrow(ax, _edge(MODULES["m5"], "bottom"), _edge(MODULES["m7"], "top"), color=COLORS["power"], lw=1.8, linestyle=dashed, rad=0.26)
    _add_arrow(ax, _edge(MODULES["m7"], "left"), _edge(MODULES["m3"], "bottom"), color=COLORS["power"], lw=1.6, linestyle=dashed, rad=0.34)
    _add_arrow(ax, _edge(MODULES["m7"], "top"), _edge(MODULES["m5"], "bottom"), color=COLORS["power"], lw=1.6, linestyle=dashed, rad=0.14)
    ax.text(0.77, 0.305, "duty_cycle / active_channels(5)", fontsize=8.0, color=COLORS["power"], ha="right", va="center")
    ax.text(0.60, 0.233, "update P_laser / power allocation to 3 / 5", fontsize=8.0, color=COLORS["power"], ha="center", va="center")

    # Optical-core ambient glyph
    ax.add_patch(Circle((0.74, 0.82), 0.010, facecolor="none", edgecolor=COLORS["meso_green"], linewidth=1.1, zorder=2))
    ax.add_patch(Circle((0.72, 0.79), 0.010, facecolor="none", edgecolor=COLORS["meso_green"], linewidth=1.1, zorder=2))
    ax.plot([0.68, 0.76], [0.77, 0.77], color=COLORS["meso_green"], lw=1.1, zorder=2)
    handles = [
        Line2D([0], [0], color=COLORS["data"], lw=2.8, label="Data plane"),
        Line2D([0], [0], color=COLORS["acc"], lw=1.8, linestyle=dashed, label="Accuracy loop"),
        Line2D([0], [0], color=COLORS["power"], lw=1.8, linestyle=dashed, label="Power loop"),
    ]
    leg = ax.legend(handles=handles, loc="lower right", bbox_to_anchor=(0.965, 0.125), fontsize=8.2, frameon=True, facecolor="white", edgecolor="#d1d5db")
    leg.get_frame().set_alpha(0.95)
    return fig


def main():
    parser = argparse.ArgumentParser(description="Render optimized Fig-B architecture (Fig.1) as SVG/PDF/PNG")
    parser.add_argument("--out", type=str, default="figures/fig_b_architecture_v2.svg", help="Output SVG path. PDF/PNG will be emitted with same stem.")
    parser.add_argument("--png-dpi", type=int, default=500, help="PNG export DPI")
    args = parser.parse_args()

    out_svg = Path(args.out)
    out_pdf = out_svg.with_suffix(".pdf")
    out_png = out_svg.with_suffix(".png")
    out_svg.parent.mkdir(parents=True, exist_ok=True)

    fig = build_figure()
    fig.savefig(out_svg, format="svg", bbox_inches="tight")
    fig.savefig(out_pdf, format="pdf", bbox_inches="tight")
    fig.savefig(out_png, format="png", dpi=args.png_dpi, bbox_inches="tight")
    plt.close(fig)

    print(f"[OK] Fig1 optimized SVG: {out_svg}")
    print(f"[OK] Fig1 optimized PDF: {out_pdf}")
    print(f"[OK] Fig1 optimized PNG: {out_png}")


if __name__ == "__main__":
    main()
