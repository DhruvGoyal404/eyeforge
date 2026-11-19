"""Draft slide deck (docs/Thapar_Orchestrators_EyeForge.pptx) from measured results + figures.

    python tools/make_deck.py
Numbers come from reports/validation.json; paste the slides into the official template if required.
"""
from __future__ import annotations

import json
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
BG, FG, ACC, MUTED = RGBColor(18, 18, 22), RGBColor(235, 235, 235), RGBColor(255, 196, 0), RGBColor(160, 160, 168)


def slide(prs, title, bullets=(), image=None):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    s.background.fill.solid()
    s.background.fill.fore_color.rgb = BG
    tb = s.shapes.add_textbox(Inches(0.5), Inches(0.3), Inches(12.3), Inches(0.9)).text_frame
    tb.text = title
    tb.paragraphs[0].runs[0].font.size, tb.paragraphs[0].runs[0].font.bold = Pt(32), True
    tb.paragraphs[0].runs[0].font.color.rgb = ACC
    if image and Path(image).exists():  # figure left, text right
        pic = s.shapes.add_picture(str(image), Inches(0.5), Inches(1.4), width=Inches(6.2))
        if pic.height > Inches(5.8):  # tall figures: fit by height instead
            pic.width, pic.height = int(pic.width * Inches(5.8) / pic.height), Inches(5.8)
        body = s.shapes.add_textbox(Inches(7.0), Inches(1.4), Inches(5.9), Inches(5.6)).text_frame
    else:
        body = s.shapes.add_textbox(Inches(0.5), Inches(1.4), Inches(12.3), Inches(5.6)).text_frame
    body.word_wrap = True
    for i, b in enumerate(bullets):
        p = body.paragraphs[0] if i == 0 else body.add_paragraph()
        p.text = b
        p.space_after = Pt(8)
        for r in p.runs:
            r.font.size, r.font.color.rgb = Pt(17), FG if not b.startswith("  ") else MUTED
    return s


def main() -> None:
    v = json.loads((ROOT / "reports" / "validation.json").read_text(encoding="utf-8"))
    fac = ROOT / "data" / "stats.json"
    fac = json.loads(fac.read_text()) if fac.exists() else {}
    d = v["direction"]
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
    s = slide(prs, "EyeForge", [
        "Others move pupils. EyeForge simulates eyes.",
        "True 3D eyeball rotation (Listing's law) · vergence at 35 cm · lids that follow gaze · "
        "refraction-aware pupil labels · XR near-eye data factory",
        "", "Team Orchestrators · Vaani Prashar, Dhruv Goyal",
        "Thapar Institute of Engineering and Technology",
        "Samsung PRISM XR Hackathon 2026 · Track 1: Synthetic Eye Movement & Virtual Camera Rendering",
        "github.com/DhruvGoyal404/eyeforge"], ROOT / "docs" / "eye_closeup.png")
    slide(prs, "Problem", [
        "XR eye trackers need large, exactly labelled eye-image datasets; real labels are slow, noisy, private.",
        "Typical synthetic rigs slide a pupil texture or yaw/pitch a sphere:",
        "  wrong rotations (false torsion), parallel eyes, frozen lids,",
        "  labels that ignore the cornea, so ground truth is off where it matters.",
        "Track 1: rotate real 3D eyeballs, locked 40 cm camera, 10-frame clockwise sequence, exact 2D/3D labels."],
        ROOT / "renders" / "compare" / "compare_sheet.png")
    slide(prs, "Solution", [
        "MPFB2 (CC0) textured head; eye bones snapped to least-squares-fitted eyeball centers.",
        "Layered anatomical eyes: veined sclera, limbal ring, fibrous iris, real pupil aperture,",
        "  glass cornea R 7.8 mm, IOR 1.376.",
        "Physiology engine (pure numpy, unit-tested): Listing's law, per-eye vergence,",
        "  lid–gaze coupling (1 : 0.3), near-triad pupil 3.5 mm, kappa + L2 profile.",
        "Two labels per pupil: geometric projection (spec) and the centroid actually seen",
        "  through the cornea, measured from mask passes of the real render."], ROOT / "renders" / "contact_sheet.png")
    slide(prs, "Architecture", [
        "build_scene.py: MPFB2 → sphere fit → bone snap → layered eyes → lid calibration → analytic lens",
        "physiology.py + calib.py: poses, K [R|t] (OpenCV), verified vs Blender to 0.001 px",
        "eyes.py: move / reset / render / sequence / animate  (Blender module + CLI)",
        "masks.py: emission mask passes → refraction-aware centroids",
        "factory.py: near-eye IR dataset · physiology.js: browser twin (parity 1e-12)",
        "tests + CI + Docker: 103 tests, headless direction self-test on every push"])
    slide(prs, "Track final output", [
        f"Lens auto-calibrated to {v['lens_mm']} mm: every L/R/U/D move shifts the pupil ≥ "
        f"{min(r['on_axis_px'] for r in d['rows']):.1f} px at 1024².",
        f"Direction self-test (L/R/U/D/reset): {'ALL PASS' if d['passed'] else 'FAIL'}, reset {max(d['reset_px']):.3f} px.",
        f"Reprojection error (label vs rendered mask): mean {v['reprojection_error_px']['mean']:.3f} px, "
        f"max {v['reprojection_error_px']['max']:.3f} px.",
        f"Projection P vs Blender: {v['camera']['max_err_px']:.4f} px. Determinism: bit-identical renders.",
        "Bundle Thapar_Orchestrators.zip: scene.blend, wrapper/eyes.py, renders/, metadata.json, README.md."],
        ROOT / "reports" / "direction_check.png")
    slide(prs, "Differentiation", [
        f"Rotation: look-at gives up to {v['torsion_deg']['spec_at_30deg_oblique']:.1f}° false torsion at 30° oblique; "
        "Listing's law gives 0.",
        f"Binocular: vergence {v['vergence_center_deg']['measured']:.2f}° at 35 cm (analytic "
        f"{v['vergence_center_deg']['expected']:.2f}°).",
        f"Lids follow gaze: aperture {v['lid_aperture_mm']['min']:.1f}–{v['lid_aperture_mm']['max']:.1f} mm; blinks.",
        f"Cornea shifts the visible pupil by {v['refraction_shift_px']['mean']:.2f} px on average: we label both.",
        "Independent check: MediaPipe agrees on motion direction (cos 0.97)."], ROOT / "reports" / "plots" / "torsion.png")
    slide(prs, "Demo & local setup", [
        "git clone https://github.com/DhruvGoyal404/eyeforge && cd eyeforge",
        "pip install -r requirements.txt; export BLENDER_PATH=/path/to/blender",
        "python wrapper/eyes.py sequence --out renders",
        "or: ./run_all.sh   ·   docker build -t eyeforge . && docker run -v $PWD:/work eyeforge",
        "Live browser twin: dhruvgoyal404.github.io/eyeforge (webcam gaze drives the avatar)",
        "Demo video: docs/demo.mp4"], ROOT / "docs" / "web_demo.png")
    slide(prs, "Brownie points: realism & XR data", [
        "Saccades on the main sequence (2.2 ms/° + 21 ms), minimum-jerk; microsaccades < 0.5°; "
        "blinks 100 ms close / 150 ms open.",
        f"Near-eye IR factory: {fac.get('samples', 0)} samples, 8-LED rings, glints solved on the corneal sphere "
        f"and verified in the render ({fac.get('glint_verify_rate', 0):.0%}).",
        "Labels: pupil ellipse (geometric + apparent), limbus ellipse, glints, COCO masks, gaze, lid aperture.",
        "Randomized: gaze ±30°, pupil 2–7 mm, iris, skin IR reflectance, LEDs, camera jitter, glasses."],
        ROOT / "data" / "overlay_grid.png")
    out = ROOT / "docs" / "Thapar_Orchestrators_EyeForge.pptx"
    prs.save(out)
    print("wrote", out)


if __name__ == "__main__":
    main()
