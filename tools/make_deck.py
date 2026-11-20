"""Submission deck on the OFFICIAL template (docs/template/Submission_Final_forGithub.pptx).

    python tools/make_deck.py   ->  docs/Thapar_Orchestrators_Submission.pptx

Only the template's own text runs are replaced (fonts, colors and layout stay as issued); content
is added inside each slide's body area. Numbers come from reports/validation.json and data/stats.json.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

from lxml import etree
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "docs" / "template" / "Submission_Final_forGithub.pptx"
OUT = ROOT / "docs" / "Thapar_Orchestrators_Submission.pptx"
TEXT, CYAN, AMBER, MUTED, BOX = (RGBColor(0xE8, 0xED, 0xF5), RGBColor(0x00, 0xB4, 0xD8),
                                 RGBColor(0xF5, 0x9E, 0x0B), RGBColor(0x9A, 0xA4, 0xB5), RGBColor(0x13, 0x1C, 0x33))
REPO = "github.com/DhruvGoyal404/eyeforge"


def set_text(shape, text):
    """Replace the text of a template shape, keeping its first run's formatting."""
    p = shape.text_frame.paragraphs[0]
    if not p.runs:  # empty template shape (e.g. the Thank You footer bar): add a styled run
        r = p.add_run()
        r.font.size, r.font.color.rgb, r.font.name = Pt(10), TEXT, "Calibri"
        p.alignment = PP_ALIGN.CENTER
    p.runs[0].text = text
    for r in p.runs[1:]:
        r.text = ""


def bullets(slide, x, y, w, h, items, size=12):
    """items: str (body), ('h', str) accent heading, ('m', str) muted note."""
    tf = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h)).text_frame
    tf.word_wrap = True
    for i, it in enumerate(items):
        kind, txt = it if isinstance(it, tuple) else ("b", it)
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(5)
        r = p.add_run()
        r.text = ("• " + txt) if kind == "b" else txt
        r.font.name = "Calibri"
        r.font.size = Pt(size + (1 if kind == "h" else -1 if kind == "m" else 0))
        r.font.bold = kind == "h"
        r.font.color.rgb = {"h": CYAN, "m": MUTED, "b": TEXT}[kind]
    return tf


def picture(slide, path, x, y, w=None, h=None):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    return slide.shapes.add_picture(str(path), Inches(x), Inches(y),
                                    width=Inches(w) if w else None, height=Inches(h) if h else None)


def box(slide, x, y, w, h, title, sub="", color=CYAN):
    s = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    s.fill.solid()
    s.fill.fore_color.rgb = BOX
    s.line.color.rgb = color
    s.line.width = Pt(1.25)
    tf = s.text_frame
    tf.word_wrap = True
    for i, (t, sz, col, b) in enumerate([(title, 11, color, True), (sub, 8.5, TEXT, False)]):
        if not t:
            continue
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = PP_ALIGN.CENTER
        r = p.add_run()
        r.text, r.font.size, r.font.bold, r.font.color.rgb, r.font.name = t, Pt(sz), b, col, "Calibri"
    return s


def arrow(slide, a, b, horizontal=True):
    if horizontal:
        x1, y1 = a.left + a.width, a.top + a.height // 2
        x2, y2 = b.left, b.top + b.height // 2
    else:  # a sits below b: arrow from a's top edge up to b's bottom edge
        x1, y1 = a.left + a.width // 2, a.top
        x2, y2 = b.left + b.width // 2, b.top + b.height
    c = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, x1, y1, x2, y2)
    c.line.color.rgb = MUTED
    c.line.width = Pt(1.25)
    c.line._get_or_add_ln().append(copy.deepcopy(_ARROW_HEAD))


_ARROW_HEAD = etree.fromstring(
    '<a:tailEnd xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" type="triangle"/>')


def duplicate(prs, index):
    """Copy slide `index` (shapes + background) and place the copy right after it."""
    src = prs.slides[index]
    new = prs.slides.add_slide(src.slide_layout)
    for sh in list(new.shapes):
        sh._element.getparent().remove(sh._element)
    for sh in src.shapes:
        new.shapes._spTree.append(copy.deepcopy(sh._element))
    bg = src._element.cSld.find(qn("p:bg"))
    if bg is not None:
        new._element.cSld.insert(0, copy.deepcopy(bg))
    ids = prs.slides._sldIdLst
    el = ids[-1]
    ids.remove(el)
    ids.insert(index + 1, el)
    return new


def main() -> None:
    v = json.loads((ROOT / "reports" / "validation.json").read_text(encoding="utf-8"))
    fac = json.loads((ROOT / "data" / "stats.json").read_text(encoding="utf-8"))
    d = v["direction"]
    min_shift = min(r["on_axis_px"] for r in d["rows"])
    prs = Presentation(str(TEMPLATE))
    duplicate(prs, 7)  # two Brownie slides: physiology realism + XR dataset factory
    S = prs.slides
    title_of = {sh.name: sh for sh in S[0].shapes}

    # 1 ---- title
    set_text(title_of["Text 7"], "Track 1")
    tf = title_of["Text 10"].text_frame
    tf.paragraphs[0].runs[0].text = "Team Orchestrators"
    tf.paragraphs[1].runs[0].text = "Thapar Institute of Engineering and Technology"
    tf.paragraphs[1].runs[0].font.size = Pt(15)  # fit left of the template's Members Details box
    mem = title_of["Text Box 22"].text_frame
    for name in ("Vaani Prashar", "Dhruv Goyal"):
        p = mem.add_paragraph()
        r = p.add_run()
        r.text, r.font.size, r.font.color.rgb, r.font.name = name, Pt(16), TEXT, "Calibri"
    bullets(S[0], 0.55, 3.75, 8.6, 1.5, [
        ("h", "EyeForge · Synthetic Eye Movement & Virtual Camera Rendering"),
        ("b", "Others move pupils. EyeForge simulates eyes: Listing's-law rotation, vergence at 35 cm, "
              "gaze-coupled lids, refraction-aware pupil labels, XR near-eye data factory."),
        ("m", REPO)], 12)

    # 2 ---- problem
    bullets(S[1], 0.4, 1.1, 5.0, 4.3, [
        ("h", "Eye trackers need perfectly labelled eye images"),
        "XR headsets track gaze with cameras; models need large, exact, private-safe datasets.",
        "Real capture is slow and its labels are noisy (where exactly is the pupil?).",
        ("h", "Typical synthetic eyes are wrong where it matters"),
        "Pupil textures slid around, or a sphere rotated by yaw then pitch → false torsion.",
        "Eyes stay parallel (no vergence), lids are frozen, pupil size never changes.",
        "Labels ignore the cornea, but the camera sees the pupil through it.",
        ("m", "Track 1 asks for true 3D eyeball rotation, a locked 40 cm camera, a 10-frame clockwise "
              "sequence and exact per-frame ground truth.")], 11.5)
    picture(S[1], ROOT / "docs" / "eye_closeup.png", 5.8, 1.1, h=4.2)

    # 3 ---- solution
    picture(S[2], ROOT / "renders" / "contact_sheet.png", 0.35, 1.1, w=4.9)
    bullets(S[2], 5.45, 1.05, 4.3, 4.4, [
        ("h", "Rig: real anatomy on a CC0 head"),
        "MPFB2 head with eye.L / eye.R bones snapped onto least-squares-fitted eyeball centers.",
        "Layered eye: veined sclera, limbal ring, fibrous iris, real pupil, glass cornea "
        "(R 7.8 mm, IOR 1.376).",
        ("h", "Physiology engine"),
        "Listing's law (torsion = 0), per-eye vergence on the 35 cm target, lids follow gaze "
        "(1 : 0.3), near-triad pupil 3.5 mm.",
        ("h", "Two labels per pupil"),
        "Geometric projection (spec) + the centroid actually seen through the cornea, measured "
        "from mask passes of the real render.",
        "API: eyes.move(h, v) · eyes.reset() · eyes.render()"], 11)

    # 4 ---- architecture (native diagram)
    s = S[3]
    a = box(s, 0.3, 1.25, 1.75, 0.85, "MPFB2 + CC0", "head, rig, skin, lashes")
    b = box(s, 2.35, 1.25, 2.1, 0.85, "build_scene.py", "sphere fit · bone snap · layered eyes · lens calib")
    c = box(s, 4.75, 1.25, 1.55, 0.85, "scene.blend", "textures packed")
    e = box(s, 6.6, 1.25, 1.65, 0.85, "eyes.py", "move · reset · render · sequence", AMBER)
    f = box(s, 8.5, 1.25, 1.3, 0.85, "outputs", "10 PNG frames, metadata.json")
    for p_, q_ in ((a, b), (b, c), (c, e), (e, f)):
        arrow(s, p_, q_)
    ph = box(s, 4.75, 2.65, 1.75, 0.85, "physiology.py", "Listing · vergence · lids · pupil")
    ca = box(s, 6.75, 2.65, 1.45, 0.85, "calib.py", "K [R|t], OpenCV P")
    ma = box(s, 8.4, 2.65, 1.4, 0.85, "masks.py", "refraction-aware centroids")
    for p_ in (ph, ca):
        arrow(s, p_, e, horizontal=False)
    arrow(s, ma, f, horizontal=False)
    t1 = box(s, 0.3, 4.05, 2.3, 0.95, "tests + CI + Docker", "103 tests · headless direction self-test")
    t2 = box(s, 2.85, 4.05, 2.3, 0.95, "factory.py", "near-eye IR dataset, glints, COCO masks")
    t3 = box(s, 5.4, 4.05, 2.1, 0.95, "web twin", "Three.js + MediaPipe, physiology.js")
    t4 = box(s, 7.75, 4.05, 2.05, 0.95, "reports + video", "validation.md · demo.mp4")
    for t in (t1, t2, t3, t4):
        t.line.color.rgb = MUTED
    bullets(s, 0.3, 2.65, 4.2, 1.2, [
        ("m", "Pure-numpy modules are unit-tested without Blender; eyes.py is the only bpy code. "
              "Extras never touch the required outputs.")], 10)

    # 5 ---- track final output
    set_text(next(sh for sh in S[4].shapes if sh.name == "Text 2"), "Track Final Output")
    picture(S[4], ROOT / "reports" / "direction_check.png", 0.35, 1.05, h=4.4)
    bullets(S[4], 4.2, 1.05, 5.55, 4.4, [
        ("h", "Required deliverables (Thapar_Orchestrators.zip)"),
        "scene.blend · wrapper/eyes.py · renders/ (10 clockwise frames, θk = 2πk/10, h = sin θ, v = cos θ) "
        "· metadata.json · README.md",
        "Camera locked 40 cm in front of the eye midpoint; target 35 cm, ±5 cm; 1024² PNG, constant light.",
        ("h", "Measured"),
        f"Direction self-test L / R / U / D / reset: {'ALL PASS' if d['passed'] else 'FAIL'} "
        f"(pupil shift ≥ {min_shift:.1f} px, reset {max(d['reset_px']):.3f} px).",
        f"Pupil label vs rendered mask: mean {v['reprojection_error_px']['mean']:.3f} px, "
        f"max {v['reprojection_error_px']['max']:.3f} px.",
        f"Camera matrix P vs Blender projection: {v['camera']['max_err_px']:.4f} px.",
        f"Vergence {v['vergence_center_deg']['measured']:.2f}° = analytic; renders bit-identical on re-run.",
        ("m", f"Lens auto-calibrated to {v['lens_mm']} mm so every move is clearly visible.")], 11)

    # 6 ---- differentiation
    picture(S[5], ROOT / "renders" / "compare" / "compare_sheet.png", 0.35, 1.05, w=5.3)
    picture(S[5], ROOT / "reports" / "plots" / "torsion.png", 0.95, 3.35, h=2.1)
    bullets(S[5], 5.85, 1.05, 3.95, 4.4, [
        ("h", "Typical rig  →  EyeForge"),
        f"Yaw-then-pitch (up to {v['torsion_deg']['spec_at_30deg_oblique']:.1f}° false torsion at 30°) "
        "→ Listing's law, torsion exactly 0.",
        f"Parallel eyes → vergence {v['vergence_center_deg']['measured']:.2f}° on the 35 cm target.",
        f"Frozen lids → lids follow gaze (aperture {v['lid_aperture_mm']['min']:.1f}–"
        f"{v['lid_aperture_mm']['max']:.1f} mm) + blinks.",
        "Fixed pupil → near-triad 3.5 mm at 35 cm.",
        f"Geometric label only → + refraction-aware label (cornea shifts it {v['refraction_shift_px']['mean']:.2f} px).",
        "Eyeballed → 103 automated tests, CI, Docker, MediaPipe cross-check."], 11)

    # 7 ---- demo & local setup
    picture(S[6], ROOT / "docs" / "web_demo.png", 0.35, 1.05, w=4.6)
    bullets(S[6], 5.15, 1.05, 4.65, 4.4, [
        ("h", "Run it"),
        "git clone https://" + REPO,
        "pip install -r requirements.txt; set BLENDER_PATH (Blender 4.2 LTS)",
        "python wrapper/eyes.py sequence --out renders",
        "or ./run_all.sh  ·  or docker build -t eyeforge . && docker run -v $PWD:/work eyeforge",
        ("h", "See it"),
        "Demo video: docs/demo.mp4 (API calls, clockwise sequence, direction check, saccades & blinks).",
        "Live browser twin: dhruvgoyal404.github.io/eyeforge, webcam gaze drives the avatar.",
        "Release v1.0: Thapar_Orchestrators.zip."], 11)

    # 8 ---- brownie 1: realism
    set_text(next(sh for sh in S[7].shapes if sh.name == "Text 2"), "Brownie Slides · Human-like motion")
    picture(S[7], ROOT / "renders" / "eye_strip.png", 0.35, 1.05, h=4.4)
    bullets(S[7], 3.6, 1.05, 6.15, 4.4, [
        ("h", "Physiological motion: eyes.animate(targets, fps=30)"),
        "Saccades on the main sequence: duration = 2.2 ms/° × amplitude + 21 ms, minimum-jerk profile.",
        "Microsaccades under 0.5° during fixation; blinks every 3–5 s (close 100 ms, open 150 ms).",
        ("h", "Physiology profiles"),
        "spec (look-at baseline) · physio (default) · physio_full (+ angle kappa 5°, binocular L2).",
        "Lid bones calibrated on the mesh so the lid margin follows gaze 1 : 1 (upper) and 0.3 : 1 (lower).",
        ("h", "Independent check"),
        "MediaPipe Face Landmarker agrees with our labels on motion direction (cos 0.97)."], 11)

    # 9 ---- brownie 2: XR dataset factory
    set_text(next(sh for sh in S[8].shapes if sh.name == "Text 2"), "Brownie Slides · XR near-eye dataset")
    picture(S[8], ROOT / "data" / "overlay_grid.png", 0.35, 1.05, h=4.4)
    bullets(S[8], 4.95, 1.05, 4.85, 4.4, [
        ("h", "Headset-style IR eye-tracker data"),
        "Near-eye IR camera ~3 cm from each eye with an 8-LED ring; grayscale IR look.",
        "Randomized: gaze ±30°, pupil 2–7 mm, iris, skin reflectance, LEDs, camera jitter, glasses.",
        ("h", "Tracker-ready labels"),
        "Pupil ellipse (geometric + seen through the cornea), limbus ellipse, glints, COCO masks, gaze, lid aperture.",
        f"Glints solved on the corneal sphere; visibility ray-cast past lids and lash hairs; "
        f"{fac['glint_verify_rate']:.1%} of visible glints (images without glasses) confirmed in the render within 3 px "
        f"(mean {fac['glint_offset_px_mean']:.1f} px).",
        ("m", "With glasses on, glint labels are flagged through_glasses: the spectacle lens shifts the "
              "real reflection and these labels are not refraction-corrected."),
        ("m", f"Smoke run: {fac['samples']} samples / {fac['images']} images; parallel-safe and resumable.")], 11)

    # 10 ---- thank you
    set_text(next(sh for sh in S[9].shapes if sh.name == "Text 25"),
             f"{REPO}  ·  Team Orchestrators  ·  Thapar Institute of Engineering and Technology")
    prs.save(OUT)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
