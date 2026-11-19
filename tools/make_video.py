"""Demo video (60-110 s, H.264): API calls, clockwise sequence, direction check, profile
comparison, physiological saccade/blink clip, results.

    python tools/make_video.py [--out docs/demo.mp4]

Needs the outputs of `eyes.py sequence`, the tests (reports/direction_check.png),
`make compare` and renders the animation clip with `eyes.py animate` if it is missing.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
W, H, FPS = 1280, 720, 30
BG, FG, ACC = (18, 18, 22), (235, 235, 235), (255, 200, 0)


def font(size, mono=False):
    for name in (["consola.ttf", "DejaVuSansMono.ttf"] if mono else ["segoeui.ttf", "DejaVuSans.ttf"]):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


def fit(im: Image.Image, w: int, h: int) -> Image.Image:
    s = min(w / im.width, h / im.height)
    return im.resize((max(1, int(im.width * s)), max(1, int(im.height * s))), Image.LANCZOS)


def canvas(title: str | None = None) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)
    if title:
        d.text((30, 18), title, fill=ACC, font=font(30))
    return im, d


def card(img_path: Path, title: str, caption: str = "") -> Image.Image:
    im, d = canvas(title)
    src = fit(Image.open(img_path).convert("RGB"), W - 60, H - 120)
    im.paste(src, ((W - src.width) // 2, 70))
    if caption:
        d.text((30, H - 40), caption, fill=FG, font=font(20))
    return im


class Writer:
    def __init__(self, out: Path):
        ff = shutil.which("ffmpeg")
        if not ff:
            raise SystemExit("ffmpeg not found on PATH")
        out.parent.mkdir(parents=True, exist_ok=True)
        self.p = subprocess.Popen([ff, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                                   "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264",
                                   "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
                                   "-movflags", "+faststart", str(out)], stdin=subprocess.PIPE)
        self.n = 0

    def add(self, im: Image.Image, seconds: float = 1 / FPS) -> None:
        b = im.convert("RGB").tobytes()
        for _ in range(max(1, round(seconds * FPS))):
            self.p.stdin.write(b)
            self.n += 1

    def close(self) -> float:
        self.p.stdin.close()
        if self.p.wait():
            raise SystemExit("ffmpeg failed")
        return self.n / FPS


def api_frame(meta, rec, png: Path, code: list[str], crop) -> Image.Image:
    im, d = canvas("1. The wrapper API: eyes.move(h, v) -> eyes.render()")
    full = Image.open(png).convert("RGB")
    left = fit(full, 600, 600)
    im.paste(left, (30, 80))
    s = left.width / full.width
    dl = ImageDraw.Draw(im)
    for side in "LR":
        x, y = rec[f"pupil_center_2D_{side}"]
        dl.line([(30 + x * s - 8, 80 + y * s), (30 + x * s + 8, 80 + y * s)], fill=(0, 255, 120), width=2)
        dl.line([(30 + x * s, 80 + y * s - 8), (30 + x * s, 80 + y * s + 8)], fill=(0, 255, 120), width=2)
    y = 90
    for line in code:
        d.text((660, y), line, fill=(120, 220, 255) if line.startswith(">>>") else FG, font=font(19, True))
        y += 28
    z = full.crop(crop)
    zd = ImageDraw.Draw(z)
    c = meta["center"]
    for side in "LR":
        px, py = rec[f"pupil_center_2D_{side}"]
        qx, qy = c[f"pupil_center_2D_{side}"]
        zd.ellipse([qx - crop[0] - 3, qy - crop[1] - 3, qx - crop[0] + 3, qy - crop[1] + 3], outline=(255, 255, 255))
        zd.line([(qx - crop[0], qy - crop[1]), (qx - crop[0] + 4 * (px - qx), qy - crop[1] + 4 * (py - qy))],
                fill=ACC, width=3)
    z = fit(z, 590, 200)
    im.paste(z, (660, 470))
    d.text((660, 676), "zoom: white = center pupil, arrow = displacement x4", fill=(170, 170, 170), font=font(16))
    return im


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "docs" / "demo.mp4"))
    args = ap.parse_args()
    meta = json.loads((ROOT / "metadata.json").read_text())
    rd = ROOT / "renders"
    anim = rd / "extra" / "anim"
    if not (anim / "timeline.json").exists():
        print("rendering the physiological clip (eyes.py animate) ...", flush=True)
        subprocess.run([sys.executable, str(ROOT / "wrapper" / "eyes.py"), "animate", "--out", str(anim)],
                       check=True, env={**os.environ})
    tl = json.loads((anim / "timeline.json").read_text())
    val = json.loads((ROOT / "reports" / "validation.json").read_text())
    w = Writer(Path(args.out))

    # 0 title
    im, d = canvas()
    d.text((80, 200), "EyeForge", fill=ACC, font=font(84))
    d.text((80, 320), "Others move pupils. EyeForge simulates eyes.", fill=FG, font=font(36))
    d.text((80, 380), "Listing's law · vergence at 35 cm · lid coupling · refraction-aware labels",
           fill=(190, 190, 190), font=font(26))
    d.text((80, 600), "Team Orchestrators · Thapar Institute of Engineering and Technology · "
           "Samsung PRISM XR Hackathon 2026, Track 1", fill=(160, 160, 160), font=font(20))
    w.add(im, 4.0)

    # 1 API + sequence
    c = meta["center"]["extended"]
    eR, eL = c["eyeball_center_2D_R"], c["eyeball_center_2D_L"]
    yc = (eR[1] + eL[1]) / 2
    crop = (int(eR[0] - 150), int(yc - 60), int(eL[0] + 150), int(yc + 60))
    recs = [("center.png", meta["center"], [">>> import eyes", ">>> eyes.reset()", ">>> eyes.render('center.png')"])]
    for f in meta["frames"]:
        k = f["frame"]
        recs.append((f"frame_{k:02d}.png", f, [
            f">>> θ = 2π·{k}/10   # {math.degrees(f['theta']):.0f}°",
            f">>> eyes.move(h={f['input']['h']:+.3f}, v={f['input']['v']:+.3f})",
            f">>> eyes.render('renders/frame_{k:02d}.png')"]))
    for png, rec, code in recs:
        e = rec["extended"]
        code = code + ["", f"eye_rotation_L  {[round(math.degrees(a), 2) for a in rec['eye_rotation_L']]}°",
                       f"eye_rotation_R  {[round(math.degrees(a), 2) for a in rec['eye_rotation_R']]}°",
                       f"pupil_2D_L      {[round(a, 1) for a in rec['pupil_center_2D_L']]}",
                       f"pupil_2D_R      {[round(a, 1) for a in rec['pupil_center_2D_R']]}",
                       f"vergence {e['vergence_deg']:.2f}°   pupil {e['pupil_diameter_mm']:.2f} mm",
                       f"torsion L/R {e['torsion_deg_L']:+.1e}° / {e['torsion_deg_R']:+.1e}° (Listing)"]
        w.add(api_frame(meta, rec, rd / png, code, crop), 1.7)

    # 2 clockwise loop in the zoomed eye region
    for _ in range(2):
        for f in meta["frames"]:
            im, d = canvas("2. Ten clockwise frames, k = 0 at the top (zoomed eye region, 1024 px source)")
            z = fit(Image.open(rd / f"frame_{f['frame']:02d}.png").convert("RGB").crop(crop), W - 60, 420)
            im.paste(z, ((W - z.width) // 2, 140))
            d.text((40, 620), f"k={f['frame']}  θ={math.degrees(f['theta']):.0f}°  h={f['input']['h']:+.3f}  "
                   f"v={f['input']['v']:+.3f}", fill=FG, font=font(28, True))
            w.add(im, 0.3)
    w.add(card(rd / "contact_sheet.png", "Contact sheet: clock layout, crosses = pupil_center_2D"), 4.0)

    # 3 direction check
    if (ROOT / "reports" / "direction_check.png").exists():
        dr = val["direction"]
        cap = (f"Automated evaluator check: {'ALL PASS' if dr and dr['passed'] else 'see report'} · "
               f"min shift {min(r['on_axis_px'] for r in dr['rows']):.1f} px · reset {max(dr['reset_px']):.3f} px")
        w.add(card(ROOT / "reports" / "direction_check.png", "3. Direction self-test: L / R / U / D / reset", cap), 6.0)

    # 4 profiles
    cmp_sheet = rd / "compare" / "compare_sheet.png"
    if cmp_sheet.exists():
        w.add(card(cmp_sheet, "4. spec (look-at) vs physio (Listing + lids + near triad) vs physio_full",
                   f"Look-at torsion up to {val['torsion_deg']['spec_at_30deg_oblique']:.1f}° at 30° oblique; "
                   "Listing keeps it at 0. Lids follow gaze; pupil 3.5 mm at 35 cm."), 7.0)

    # 5 physiological motion
    x0, y0 = tl["crop_px"][0], tl["crop_px"][1]
    frames = sorted(anim.glob("frame_*.png"))

    def anim_frame(i, title):
        f = tl["frames"][i]
        im, d = canvas(title)
        src = Image.open(frames[i]).convert("RGB")
        z = fit(src, W - 60, 470)
        im.paste(z, ((W - z.width) // 2, 110))
        ev = f["event"]
        col = {"saccade": (255, 120, 80), "blink": (120, 200, 255), "microsaccade": (180, 255, 120)}.get(ev, FG)
        d.text((40, 610), f"t = {f['t']:.2f} s   {ev.upper()}", fill=col, font=font(32, True))
        d.text((40, 660), "saccades: 2.2 ms/deg + 21 ms, minimum-jerk · blinks: close 100 ms / open 150 ms · "
               "microsaccades < 0.5°", fill=(170, 170, 170), font=font(18))
        return im

    for i in range(len(frames)):
        w.add(anim_frame(i, "5. Physiological motion (real time): saccades, blinks, microsaccades"))
    first_blink = next((i for i, f in enumerate(tl["frames"]) if f["event"] == "blink"), None)
    if first_blink is not None:
        for i in range(max(0, first_blink - 6), min(len(frames), first_blink + 12)):
            w.add(anim_frame(i, "5b. Blink, 4x slow motion"), 4 / FPS)

    # 6 results
    im, d = canvas("6. Measured results (reports/validation.md)")
    s = val
    lines = [f"Reprojection error (projection vs rendered mask): mean {s['reprojection_error_px']['mean']:.3f} px, "
             f"max {s['reprojection_error_px']['max']:.3f} px",
             f"Projection P vs Blender: max {s['camera']['max_err_px']:.4f} px" if s.get("camera") else "",
             f"Pupil shift per frame: {s['pupil_shift_px']['min']:.1f} to {s['pupil_shift_px']['max']:.1f} px (lens "
             f"{s['lens_mm']} mm)",
             f"Vergence at center: {s['vergence_center_deg']['measured']:.2f}° (analytic "
             f"{s['vergence_center_deg']['expected']:.2f}°)",
             f"Torsion: Listing {s['torsion_deg']['physio_max']:.0e}° vs look-at {s['torsion_deg']['spec_max']:.2f}°",
             f"Corneal refraction shift of the apparent pupil: mean {s['refraction_shift_px']['mean']:.2f} px",
             (f"MediaPipe iris vs our labels: mean {s['mediapipe']['mean_err_px']:.1f} px"
              if s.get("mediapipe") and s["mediapipe"].get("mean_err_px") else ""),
             "", "Run it:  python wrapper/eyes.py sequence --out renders     |     ./run_all.sh     |     docker run eyeforge"]
    y = 110
    for line in filter(None, lines):
        d.text((60, y), line, fill=FG, font=font(26))
        y += 52
    w.add(im, 7.0)
    print(f"wrote {args.out} ({w.close():.1f} s)")


if __name__ == "__main__":
    main()
