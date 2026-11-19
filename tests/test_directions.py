"""The evaluator's direction check, automated: Left, Right, Up, Down and Reset."""
import json
import math

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from conftest import FAST, REPORTS, run_job

EXPECT = {"left": (0, -1), "right": (0, +1), "up": (1, -1), "down": (1, +1)}  # (axis, sign); y is down
MIN_SHIFT, MAX_OFFAXIS, RESET_TOL = 15.0, 0.30, 0.5


def centroid(rec, side, kind="apparent"):
    return np.array(rec["extended"][f"pupil_center_2D_{kind}_{side}" if kind == "apparent"
                                    else f"pupil_center_2D_geometric_mask_{side}"])


def test_directions():
    out = REPORTS / "direction"
    res = run_job("directions", out)
    rows, ok = [], True
    for name, (ax, sign) in EXPECT.items():
        for side in "LR":
            d = centroid(res[name], side) - centroid(res["center"], side)
            on, off = sign * d[ax], abs(d[1 - ax])
            passed = on >= MIN_SHIFT and off < MAX_OFFAXIS * on
            ok &= passed
            rows.append(dict(direction=name, eye=side, dx=d[0], dy=d[1], on_axis_px=on,
                             off_axis_ratio=off / on if on > 0 else None, passed=bool(passed)))
    reset = [float(np.linalg.norm(centroid(res["reset"], s) - centroid(res["center"], s))) for s in "LR"]
    ok &= max(reset) < RESET_TOL
    report = dict(rows=rows, reset_px=reset, passed=bool(ok), min_shift_px=MIN_SHIFT,
                  max_offaxis_ratio=MAX_OFFAXIS, reset_tol_px=RESET_TOL, fast=FAST)
    (REPORTS / "direction.json").write_text(json.dumps(report, indent=2))
    montage(res, out, rows)
    for r in rows:
        assert r["passed"], r
    assert max(reset) < RESET_TOL, reset


def montage(res, out, rows):
    """direction_check.png: eye-region crops for C/L/R/U/D/reset, arrows = shift (x4)."""
    c = res["center"]["extended"]
    eR, eL = c["eyeball_center_2D_R"], c["eyeball_center_2D_L"]
    x0, x1 = int(eR[0] - 140), int(eL[0] + 140)
    yc = int((eR[1] + eL[1]) / 2)
    y0, y1 = yc - 70, yc + 70
    names = ["center", "left", "right", "up", "down", "reset"]
    W, Hh = x1 - x0, y1 - y0
    sheet = Image.new("RGB", (W, (Hh + 30) * len(names)), (20, 20, 22))
    d = ImageDraw.Draw(sheet)
    f = ImageFont.load_default(size=18)
    for i, n in enumerate(names):
        src = out / f"{n}.png" if (out / f"{n}.png").exists() else out / "masks" / f"{n}_apparent.png"
        crop = Image.open(src).convert("RGB").crop((x0, y0, x1, y1))
        cd = ImageDraw.Draw(crop)
        for s in "LR":
            p0, p = centroid(res["center"], s), centroid(res[n], s)
            q = p0 + 4 * (p - p0)
            cd.line([(p0[0] - x0, p0[1] - y0), (q[0] - x0, q[1] - y0)], fill=(255, 210, 0), width=3)
            cd.ellipse([q[0] - x0 - 4, q[1] - y0 - 4, q[0] - x0 + 4, q[1] - y0 + 4], fill=(255, 210, 0))
            cd.line([(p[0] - x0 - 6, p[1] - y0), (p[0] - x0 + 6, p[1] - y0)], fill=(0, 255, 120), width=2)
            cd.line([(p[0] - x0, p[1] - y0 - 6), (p[0] - x0, p[1] - y0 + 6)], fill=(0, 255, 120), width=2)
        top = i * (Hh + 30)
        sheet.paste(crop, (0, top + 30))
        info = [r for r in rows if r["direction"] == n]
        status = ("PASS" if all(r["passed"] for r in info) else "FAIL") if info else ""
        shift = "  ".join(f"{r['eye']}: dx={r['dx']:+.1f} dy={r['dy']:+.1f}px" for r in info)
        d.text((8, top + 5), f"{n.upper()}  {status}  {shift}  (arrow = shift x4)", fill=(255, 255, 255), font=f)
    sheet.save(REPORTS / "direction_check.png")
