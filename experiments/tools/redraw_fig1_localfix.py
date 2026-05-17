#!/usr/bin/env python3
"""Local targeted redraw for Fig.1: preserve correct regions and patch known issues."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def _font(size: int) -> ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]
    for path in candidates:
        p = Path(path)
        if p.exists():
            return ImageFont.truetype(str(p), size)
    return ImageFont.load_default()


def _draw_dashed_line(
    draw: ImageDraw.ImageDraw,
    p0: tuple[int, int],
    p1: tuple[int, int],
    color: tuple[int, int, int],
    width: int = 4,
    dash: int = 14,
    gap: int = 10,
) -> None:
    x0, y0 = p0
    x1, y1 = p1
    if x0 == x1:
        step = 1 if y1 >= y0 else -1
        y = y0
        while (y <= y1 if step > 0 else y >= y1):
            y_end = y + step * min(dash, abs(y1 - y))
            draw.line([(x0, y), (x1, y_end)], fill=color, width=width)
            y = y_end + step * gap
    elif y0 == y1:
        step = 1 if x1 >= x0 else -1
        x = x0
        while (x <= x1 if step > 0 else x >= x1):
            x_end = x + step * min(dash, abs(x1 - x))
            draw.line([(x, y0), (x_end, y1)], fill=color, width=width)
            x = x_end + step * gap
    else:
        draw.line([p0, p1], fill=color, width=width)


def redraw(in_path: Path, out_path: Path) -> None:
    img = Image.open(in_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    # Sample local background tones from source image to preserve identity.
    module3_bg = img.getpixel((760, 1030))
    white_bg = (255, 255, 255)
    black = (24, 24, 24)
    blue = (36, 130, 206)

    # 1) Fix module 3 right-end: remove noisy mesh and draw clean waveguide termination.
    draw.rectangle((1130, 845, 1265, 1302), fill=module3_bg)
    draw.line((1228, 862, 1228, 1278), fill=black, width=4)
    ys = [875, 928, 981, 1034, 1087, 1140, 1193, 1246]
    for y in ys:
        draw.line((1138, y, 1228, y), fill=black, width=4)
        draw.ellipse((1230, y - 10, 1250, y + 10), outline=black, width=3, fill=(230, 238, 248))
        draw.ellipse((1235, y - 5, 1245, y + 5), outline=(86, 138, 185), width=2, fill=white_bg)

    # 2) Fix module 7 internals: replace doodle with clean monitor-control-actuator icon.
    draw.rectangle((2258, 704, 2686, 936), fill=white_bg)
    # Keep box border clean.
    draw.rectangle((2248, 560, 2698, 936), outline=(74, 74, 74), width=3)

    # Icon row.
    draw.rectangle((2300, 730, 2410, 804), outline=(70, 70, 70), width=3, fill=(248, 248, 248))
    draw.line((2312, 790, 2342, 772), fill=(80, 80, 80), width=3)
    draw.line((2342, 772, 2372, 780), fill=(80, 80, 80), width=3)
    draw.line((2372, 780, 2398, 748), fill=(80, 80, 80), width=3)

    draw.rectangle((2440, 730, 2550, 804), outline=(70, 70, 70), width=3, fill=(248, 248, 248))
    f_mid = _font(34)
    draw.text((2464, 736), "PHY", font=f_mid, fill=(70, 70, 70))

    draw.rectangle((2580, 730, 2662, 804), outline=(70, 70, 70), width=3, fill=(248, 248, 248))
    draw.polygon([(2598, 767), (2628, 748), (2628, 760), (2648, 760), (2648, 774), (2628, 774), (2628, 786)], fill=(80, 80, 80))

    draw.line((2412, 767, 2438, 767), fill=(70, 70, 70), width=3)
    draw.line((2552, 767, 2578, 767), fill=(70, 70, 70), width=3)

    # Metric list.
    f_list = _font(34)
    draw.text((2452, 838), "Loss", font=f_list, fill=(45, 45, 45))
    draw.text((2400, 878), "PP_xtalk", font=f_list, fill=(45, 45, 45))
    draw.text((2450, 918), "BER", font=f_list, fill=(45, 45, 45))
    draw.text((2418, 958), "P_laser", font=f_list, fill=(45, 45, 45))

    # 3) Improve annotation readability on right side (same terms, cleaner placement).
    f_label = _font(34)
    # White pads behind labels to remove visual clutter.
    draw.rectangle((2160, 170, 2740, 236), fill=white_bg)
    draw.text((2188, 174), "Accuracy Loop", font=f_label, fill=(35, 35, 35))

    draw.rectangle((2140, 276, 2620, 346), fill=white_bg)
    draw.text((2152, 284), "calibrate(BtoS)", font=f_label, fill=(35, 35, 35))

    draw.rectangle((2140, 366, 2640, 438), fill=white_bg)
    draw.text((2152, 374), "calibrate(Sched)", font=f_label, fill=(35, 35, 35))

    draw.rectangle((1928, 628, 2440, 700), fill=white_bg)
    draw.text((1938, 636), "calibrate(Photon)", font=f_label, fill=(35, 35, 35))

    draw.rectangle((2060, 1032, 2745, 1314), fill=white_bg)
    draw.text((2096, 1052), "monitored_power", font=f_label, fill=(35, 35, 35))
    draw.text((2096, 1132), "update(Laser_Power)", font=f_label, fill=(35, 35, 35))
    draw.text((2096, 1212), "update(Tuning_Power)", font=f_label, fill=(35, 35, 35))

    # Re-ink key dashed connectors near right with uniform style.
    _draw_dashed_line(draw, (2122, 258), (2660, 258), color=blue, width=4)
    _draw_dashed_line(draw, (2660, 258), (2660, 560), color=blue, width=4)
    _draw_dashed_line(draw, (2122, 348), (2530, 348), color=blue, width=4)
    _draw_dashed_line(draw, (2530, 348), (2530, 560), color=blue, width=4)
    _draw_dashed_line(draw, (2112, 445), (2470, 445), color=blue, width=4)
    _draw_dashed_line(draw, (2470, 445), (2470, 905), color=blue, width=4)

    purple = (118, 69, 143)
    _draw_dashed_line(draw, (2482, 948), (2482, 1120), color=purple, width=4)
    _draw_dashed_line(draw, (2482, 1120), (2096, 1120), color=purple, width=4)
    _draw_dashed_line(draw, (2632, 948), (2632, 1280), color=purple, width=4)
    _draw_dashed_line(draw, (2632, 1280), (2104, 1280), color=purple, width=4)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


def main() -> int:
    in_path = Path("figures/fig1_ai_generated_from_v20_ieee_v1.png")
    out_path = Path("figures/fig1_ai_localfix_v1_offline.png")
    redraw(in_path, out_path)
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
