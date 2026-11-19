"""spec vs physio vs physio_full: same inputs, side by side, annotated.

Render (inside Blender, eye region only via a render border):
    blender -b scene.blend --python-exit-code 1 -P tools/compare.py -- renders/compare
Compose the annotated sheets (normal Python with Pillow):
    python tools/compare.py renders/compare
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "wrapper"))

INPUTS = {"center": (0.0, 0.0), "up": (0.0, 1.0), "down": (0.0, -1.0), "oblique": (0.7071, 0.7071)}
PROFILES = ["spec", "physio", "physio_full"]
PAD_X, PAD_Y = 170, 95


def render(out: Path) -> None:
    import bpy
    import eyes
    rig = eyes._rig()
    sc = rig.scene
    c = rig.reset()["extended"]
    eR, eL = c["eyeball_center_2D_R"], c["eyeball_center_2D_L"]
    W, H = sc.render.resolution_x, sc.render.resolution_y
    box = [eR[0] - PAD_X, (eR[1] + eL[1]) / 2 - PAD_Y, eL[0] + PAD_X, (eR[1] + eL[1]) / 2 + PAD_Y]
    sc.render.use_border, sc.render.use_crop_to_border = True, True
    sc.render.border_min_x, sc.render.border_max_x = box[0] / W, box[2] / W
    sc.render.border_min_y, sc.render.border_max_y = 1 - box[3] / H, 1 - box[1] / H  # Blender y is up
    info = {"box_px": box, "renders": {}}
    try:
        for prof in PROFILES:
            rig.profile = prof
            for name, (h, v) in INPUTS.items():
                rec = rig.move(h, v)
                p = out / "raw" / f"{name}_{prof}.png"
                rig.render(str(p))
                e = rec["extended"]
                info["renders"][f"{name}_{prof}"] = dict(
                    path=str(p), h=h, v=v, profile=prof, pupil_mm=e["pupil_diameter_mm"],
                    torsion_deg=[e["torsion_deg_L"], e["torsion_deg_R"]],
                    upper_lid_deg=e["upper_lid_rotation_deg_L"], aperture_mm=e["lid_aperture_mm_L"],
                    pupil_2D=[rec["pupil_center_2D_L"], rec["pupil_center_2D_R"]],
                    eye_rotation_deg=[e["eye_rotation_deg_L"], e["eye_rotation_deg_R"]],
                    kappa_deg=e["kappa_deg"], seconds=e["render_seconds"])
                print(f"[compare] {name} {prof}: {e['render_seconds']} s", flush=True)
    finally:
        sc.render.use_border = False
        rig.profile = "physio"
        rig.reset()
    (out / "compare.json").write_text(json.dumps(info, indent=2))


def compose(out: Path) -> None:
    from PIL import Image, ImageDraw, ImageFont
    info = json.loads((out / "compare.json").read_text())
    f, fs = ImageFont.load_default(size=17), ImageFont.load_default(size=14)
    x0, y0 = info["box_px"][0], info["box_px"][1]
    first = Image.open(next(iter(info["renders"].values()))["path"])
    tw, th = first.size
    sheet = Image.new("RGB", (tw * 3, (th + 64) * len(INPUTS) + 40), (22, 22, 25))
    d = ImageDraw.Draw(sheet)
    for j, prof in enumerate(PROFILES):
        d.text((j * tw + 10, 10), {"spec": "spec: look-at, static lids, 4.0 mm pupil",
                                   "physio": "physio: Listing + lids + near-triad pupil",
                                   "physio_full": "physio_full: + kappa 5° + L2"}[prof],
               fill=(255, 255, 255), font=f)
    for i, name in enumerate(INPUTS):
        for j, prof in enumerate(PROFILES):
            r = info["renders"][f"{name}_{prof}"]
            im = Image.open(r["path"]).convert("RGB")
            dd = ImageDraw.Draw(im)
            for (px, py) in r["pupil_2D"]:
                dd.line([(px - x0 - 6, py - y0), (px - x0 + 6, py - y0)], fill=(0, 255, 120), width=2)
                dd.line([(px - x0, py - y0 - 6), (px - x0, py - y0 + 6)], fill=(0, 255, 120), width=2)
            top = 40 + i * (th + 64)
            sheet.paste(im, (j * tw, top))
            d.multiline_text(
                (j * tw + 8, top + th + 4),
                f"{name} (h={r['h']:+.2f}, v={r['v']:+.2f})  pupil {r['pupil_mm']:.2f} mm  kappa {r['kappa_deg']:.0f}°\n"
                f"torsion L/R {r['torsion_deg'][0]:+.3f}/{r['torsion_deg'][1]:+.3f}°  upper lid "
                f"{r['upper_lid_deg']:+.1f}°  aperture {r['aperture_mm']:.2f} mm",
                fill=(230, 230, 230), font=fs, spacing=4)
    sheet.save(out / "compare_sheet.png")
    print("wrote", out / "compare_sheet.png")


if __name__ == "__main__":
    try:
        import bpy  # noqa: F401
        argv = sys.argv[sys.argv.index("--") + 1:]
        render(Path(argv[0] if argv else ROOT / "renders" / "compare"))
    except ImportError:
        compose(Path(sys.argv[1] if len(sys.argv) > 1 else ROOT / "renders" / "compare"))
