"""Contact sheet (4x3 clock layout) + zoomed eye strip from metadata.json and the renders.

    python tools/contact_sheet.py --meta metadata.json --renders renders
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

TILE = 320
# 4x3 grid: the 10 perimeter cells, ordered clockwise from the top-left-of-center cell, so
# frame k (k * 36 deg clockwise from 12 o'clock) lands in the cell with the nearest angle.
CLOCK_CELLS = [(1, 0), (2, 0), (3, 0), (3, 1), (3, 2), (2, 2), (1, 2), (0, 2), (0, 1), (0, 0)]
CENTER_CELL, LEGEND_CELL = (1, 1), (2, 1)


def font(size: int):
    return ImageFont.load_default(size=size)


def cross(d: ImageDraw.ImageDraw, x: float, y: float, r: float, color, w: int = 2) -> None:
    d.line([(x - r, y), (x + r, y)], fill=color, width=w)
    d.line([(x, y - r), (x, y + r)], fill=color, width=w)


def arrow(d, x0, y0, x1, y1, color, w=2):
    d.line([(x0, y0), (x1, y1)], fill=color, width=w)
    a = math.atan2(y1 - y0, x1 - x0)
    for da in (2.6, -2.6):
        d.line([(x1, y1), (x1 + 8 * math.cos(a + da), y1 + 8 * math.sin(a + da))], fill=color, width=w)


def label(rec: dict) -> str:
    return (f"k={rec['frame']}  theta={math.degrees(rec['theta']):.0f}°\n"
            f"h={rec['input']['h']:+.3f} v={rec['input']['v']:+.3f}")


def tile(path: Path, rec: dict, text: str) -> Image.Image:
    im = Image.open(path).convert("RGB")
    s = TILE / im.width
    im = im.resize((TILE, TILE), Image.LANCZOS)
    d = ImageDraw.Draw(im)
    for side in "LR":
        x, y = rec[f"pupil_center_2D_{side}"]
        cross(d, x * s, y * s, 7, (0, 255, 120))
    d.rectangle([0, 0, TILE, 40], fill=(0, 0, 0))
    d.multiline_text((6, 2), text, fill=(255, 255, 255), font=font(15))
    return im


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", default="metadata.json")
    ap.add_argument("--renders", default="renders")
    args = ap.parse_args()
    meta = json.loads(Path(args.meta).read_text())
    rd = Path(args.renders)
    frames, center = meta["frames"], meta["center"]

    sheet = Image.new("RGB", (4 * TILE, 3 * TILE + 50), (25, 25, 28))
    d = ImageDraw.Draw(sheet)
    d.text((10, 12), "EyeForge: 10 clockwise frames (k=0 top), crosses = projected pupil centers "
           "(pupil_center_2D)", fill=(255, 255, 255), font=font(20))
    for rec in frames:
        cx, cy = CLOCK_CELLS[rec["frame"]]
        sheet.paste(tile(rd / f"frame_{rec['frame']:02d}.png", rec, label(rec)), (cx * TILE, 50 + cy * TILE))
    cx, cy = CENTER_CELL
    sheet.paste(tile(rd / "center.png", center, "center / reset\nh=0 v=0"), (cx * TILE, 50 + cy * TILE))
    lx, ly = LEGEND_CELL
    s = meta["scene"]
    d.multiline_text((lx * TILE + 12, 50 + ly * TILE + 12),
                     f"profile: {s['profile']}\nlens {s['lens_mm']} mm, cam 40 cm\n"
                     f"IPD {s['ipd_m']*1000:.1f} mm\ntarget 35 cm, ±5 cm\n"
                     f"{s['engine']} {s['samples']} spp\n\nzoom: renders/eye_strip.png",
                     fill=(220, 220, 220), font=font(17), spacing=6)
    sheet.save(rd / "contact_sheet.png")

    # zoomed eye strip: one row per pose, arrows = pupil displacement from center (x4)
    eR, eL = center["extended"]["eyeball_center_2D_R"], center["extended"]["eyeball_center_2D_L"]
    x0, x1 = int(eR[0] - 150), int(eL[0] + 150)
    yc = int((eR[1] + eL[1]) / 2)
    y0, y1 = yc - 75, yc + 75
    rows = [("center", "center.png", center)] + [(label(r).replace("\n", "  "), f"frame_{r['frame']:02d}.png", r)
                                                 for r in frames]
    lw = 300
    strip = Image.new("RGB", (lw + x1 - x0, (y1 - y0) * len(rows)), (25, 25, 28))
    sd = ImageDraw.Draw(strip)
    for i, (txt, fn, rec) in enumerate(rows):
        crop = Image.open(rd / fn).convert("RGB").crop((x0, y0, x1, y1))
        cd = ImageDraw.Draw(crop)
        for side in "LR":
            px, py = rec[f"pupil_center_2D_{side}"]
            qx, qy = center[f"pupil_center_2D_{side}"]
            cd.ellipse([qx - x0 - 3, qy - y0 - 3, qx - x0 + 3, qy - y0 + 3], outline=(255, 255, 255))
            if rec is not center:
                arrow(cd, qx - x0, qy - y0, qx - x0 + 4 * (px - qx), qy - y0 + 4 * (py - qy), (255, 210, 0))
            cross(cd, px - x0, py - y0, 6, (0, 255, 120))
        strip.paste(crop, (lw, i * (y1 - y0)))
        sd.text((10, i * (y1 - y0) + 60), txt, fill=(255, 255, 255), font=font(18))
    strip.save(rd / "eye_strip.png")
    print("wrote", rd / "contact_sheet.png", "and", rd / "eye_strip.png")


if __name__ == "__main__":
    main()
