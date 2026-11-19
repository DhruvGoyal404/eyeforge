"""Aggregate metadata.json + test outputs into reports/validation.{md,json} and plots,
and refresh the results table in README.md.

    python tools/make_report.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "wrapper"))
import physiology as ph  # noqa: E402

REP = ROOT / "reports"


def load(name: str):
    p = REP / name
    return json.loads(p.read_text()) if p.exists() else None


def main() -> None:
    meta = json.loads((ROOT / "metadata.json").read_text())
    fr, sc, c = meta["frames"], meta["scene"], meta["center"]
    geom = ph.Geometry(C_L=np.array(sc["eyeball_center_L"]), C_R=np.array(sc["eyeball_center_R"]),
                       F=np.array(sc["F"]), U=np.array(sc["U"]), R_img=np.array(sc["R_img"]),
                       pupil_depth=sc["pupil_depth_m"])
    rows = []
    for f in fr:
        e = f["extended"]
        shift = [math.dist(f[f"pupil_center_2D_{s}"], c[f"pupil_center_2D_{s}"]) for s in "LR"]
        spec = ph.solve(f["input"]["h"], f["input"]["v"], geom, "spec")
        full = ph.solve(f["input"]["h"], f["input"]["v"], geom, "physio_full")
        rows.append(dict(
            k=f["frame"], theta_deg=math.degrees(f["theta"]), h=f["input"]["h"], v=f["input"]["v"],
            reproj_L=e["reprojection_error_px_L"], reproj_R=e["reprojection_error_px_R"],
            refr_L=e["refraction_shift_px_L"], refr_R=e["refraction_shift_px_R"],
            shift_L=shift[0], shift_R=shift[1], vergence_deg=e["vergence_deg"],
            torsion_physio_deg=max(abs(e["torsion_deg_L"]), abs(e["torsion_deg_R"])),
            torsion_spec_deg=max(abs(math.degrees(spec["eyes"][s]["torsion"])) for s in "LR"),
            torsion_full_deg=max(abs(math.degrees(full["eyes"][s]["torsion"])) for s in "LR"),
            aperture_L=e["lid_aperture_mm_L"], aperture_R=e["lid_aperture_mm_R"],
            pupil_mm=e["pupil_diameter_mm"], render_s=e.get("render_seconds")))
    reproj = [x for r in rows for x in (r["reproj_L"], r["reproj_R"])]
    refr = [x for r in rows for x in (r["refr_L"], r["refr_R"])]
    verg_expected = 2 * math.degrees(math.atan(sc["ipd_m"] / 2 / 0.35))
    rt = [r["render_s"] for r in rows if r["render_s"]]
    # Fick (look-at) false torsion grows with eccentricity: show it at a factory-like +-30 deg too
    wide = ph.Geometry(geom.C_L, geom.C_R, geom.F, geom.U, geom.R_img, geom.pupil_depth,
                       target_range=0.35 * math.tan(math.radians(30)))
    t_wide = max(abs(math.degrees(e["torsion"])) for e in ph.solve(0.7071, 0.7071, wide, "spec")["eyes"].values())

    summary = dict(
        frames=len(fr), reprojection_error_px=dict(mean=float(np.mean(reproj)), max=float(np.max(reproj))),
        refraction_shift_px=dict(mean=float(np.mean(refr)), max=float(np.max(refr))),
        pupil_shift_px=dict(min=min(min(r["shift_L"], r["shift_R"]) for r in rows),
                            max=max(max(r["shift_L"], r["shift_R"]) for r in rows)),
        vergence_center_deg=dict(measured=c["extended"]["vergence_deg"], expected=verg_expected),
        torsion_deg=dict(physio_max=max(r["torsion_physio_deg"] for r in rows),
                         spec_max=max(r["torsion_spec_deg"] for r in rows),
                         physio_full_max=max(r["torsion_full_deg"] for r in rows),
                         spec_at_30deg_oblique=t_wide),
        lid_aperture_mm=dict(min=min(min(r["aperture_L"], r["aperture_R"]) for r in rows),
                             max=max(max(r["aperture_L"], r["aperture_R"]) for r in rows),
                             center=c["extended"]["lid_aperture_mm_L"]),
        render_seconds_per_frame=dict(mean=float(np.mean(rt)) if rt else None, max=max(rt) if rt else None),
        pipeline_seconds=meta.get("pipeline_seconds"), lens_mm=sc["lens_mm"], samples=sc["samples"],
        camera=load("camera.json"), direction=load("direction.json"), determinism=load("determinism.json"),
        mediapipe=load("mediapipe.json"), rows=rows)
    REP.mkdir(exist_ok=True)
    (REP / "validation.json").write_text(json.dumps(summary, indent=2))
    plots(rows, meta)
    md = markdown(summary)
    (REP / "validation.md").write_text(md, encoding="utf-8")
    readme = ROOT / "README.md"
    if readme.exists():
        t = readme.read_text(encoding="utf-8")
        a, b = "<!-- VALIDATION:START -->", "<!-- VALIDATION:END -->"
        if a in t and b in t:
            t = t[: t.index(a) + len(a)] + "\n" + headline_table(summary) + "\n" + t[t.index(b):]
            readme.write_text(t, encoding="utf-8")
    fill_slides(summary)
    print("wrote", REP / "validation.md")


def fill_slides(s: dict) -> None:
    """docs/slides_template.md -> docs/slides_content.md with the measured numbers."""
    tpl = ROOT / "docs" / "slides_template.md"
    if not tpl.exists():
        return
    d, cam, det = s["direction"] or {}, s["camera"] or {}, s["determinism"] or {}
    fac = json.loads((ROOT / "data" / "stats.json").read_text()) if (ROOT / "data" / "stats.json").exists() else {}
    vid = ROOT / "docs" / "demo.mp4"
    video_s = 0.0
    if vid.exists():
        import subprocess
        r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(vid)],
                           capture_output=True, text=True)
        video_s = float(r.stdout.strip() or 0)
    vals = dict(lens_mm=s["lens_mm"], min_shift=min(r["on_axis_px"] for r in d["rows"]) if d else s["pupil_shift_px"]["min"],
                direction_status="ALL PASS" if d.get("passed") else "see report",
                reset_px=max(d["reset_px"]) if d else float("nan"),
                reproj_mean=s["reprojection_error_px"]["mean"], reproj_max=s["reprojection_error_px"]["max"],
                cam_err=cam.get("max_err_px", float("nan")), det=det.get("mean_abs_diff_8bit", float("nan")),
                t30=s["torsion_deg"]["spec_at_30deg_oblique"], verg=s["vergence_center_deg"]["measured"],
                verg_exp=s["vergence_center_deg"]["expected"], refr=s["refraction_shift_px"]["mean"],
                video_s=video_s, glint_rate=fac.get("glint_verify_rate", 0.0))
    (ROOT / "docs" / "slides_content.md").write_text(tpl.read_text(encoding="utf-8").format(**vals), encoding="utf-8")


def headline_table(s: dict) -> str:
    d, cam, det, mp = s["direction"], s["camera"], s["determinism"], s["mediapipe"]
    lines = ["| Check | Result | Target |", "|---|---|---|"]
    if d:
        worst = min(r["on_axis_px"] for r in d["rows"])
        lines.append(f"| Direction L/R/U/D + reset | {'PASS' if d['passed'] else 'FAIL'} "
                     f"(min on-axis shift {worst:.1f} px, reset {max(d['reset_px']):.3f} px) | "
                     f">= 15 px, reset < 0.5 px |")
    lines.append(f"| Reprojection error (projection vs mask) | mean {s['reprojection_error_px']['mean']:.3f} px, "
                 f"max {s['reprojection_error_px']['max']:.3f} px | mean < 0.5 px |")
    if cam:
        lines.append(f"| P vs Blender world_to_camera_view | max {cam['max_err_px']:.4f} px | < 0.01 px |")
    lines.append(f"| Vergence at center | {s['vergence_center_deg']['measured']:.3f}° "
                 f"(analytic {s['vergence_center_deg']['expected']:.3f}°) | equal |")
    lines.append(f"| Torsion, physio (Listing) vs spec (look-at) | {s['torsion_deg']['physio_max']:.2e}° vs "
                 f"{s['torsion_deg']['spec_max']:.3f}° (look-at at 30° oblique: "
                 f"{s['torsion_deg']['spec_at_30deg_oblique']:.2f}°) | physio = 0 |")
    lines.append(f"| Refraction shift (apparent vs geometric) | mean {s['refraction_shift_px']['mean']:.2f} px | reported |")
    if det:
        lines.append(f"| Determinism (same frame twice) | mean abs diff {det['mean_abs_diff_8bit']:.4f} / 255 | < 0.05 |")
    if mp and mp.get("detected"):
        lines.append(f"| MediaPipe iris vs our apparent pupil | mean {mp['mean_err_px']:.2f} px, max "
                     f"{mp['max_err_px']:.2f} px on {mp['detected']}/{mp['images']} images | report only |")
    rs = s["render_seconds_per_frame"]
    if rs["mean"]:
        lines.append(f"| Render time | {rs['mean']:.0f} s/frame (Cycles CPU, {s['samples']} spp, i5-1235U) | - |")
    return "\n".join(lines)


def markdown(s: dict) -> str:
    out = ["# EyeForge validation report", "", "Generated by `tools/make_report.py` from `metadata.json` "
           "and the pytest outputs in `reports/`.", "", "## Summary", "", headline_table(s), "",
           "## Per-frame (required sequence, profile `physio`)", "",
           "| k | θ (°) | h | v | reproj L/R (px) | refraction L/R (px) | pupil shift L/R (px) | "
           "vergence (°) | torsion physio / spec / full (°) | lid aperture L (mm) | pupil (mm) | render (s) |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in s["rows"]:
        out.append(f"| {r['k']} | {r['theta_deg']:.0f} | {r['h']:+.3f} | {r['v']:+.3f} | "
                   f"{r['reproj_L']:.3f} / {r['reproj_R']:.3f} | {r['refr_L']:.2f} / {r['refr_R']:.2f} | "
                   f"{r['shift_L']:.1f} / {r['shift_R']:.1f} | {r['vergence_deg']:.2f} | "
                   f"{r['torsion_physio_deg']:.1e} / {r['torsion_spec_deg']:.3f} / {r['torsion_full_deg']:.3f} | "
                   f"{r['aperture_L']:.2f} | {r['pupil_mm']:.2f} | {r['render_s']} |")
    d = s["direction"]
    if d:
        out += ["", "## Direction self-test", "", "| direction | eye | dx (px) | dy (px) | off-axis ratio | pass |",
                "|---|---|---|---|---|---|"]
        for r in d["rows"]:
            out.append(f"| {r['direction']} | {r['eye']} | {r['dx']:+.2f} | {r['dy']:+.2f} | "
                       f"{r['off_axis_ratio']:.3f} | {'yes' if r['passed'] else 'NO'} |")
        out += ["", f"Reset returns to within {max(d['reset_px']):.4f} px of the first center render.", "",
                "![direction check](direction_check.png)"]
    mp = s["mediapipe"]
    if mp:
        out += ["", "## Independent check: MediaPipe Face Landmarker", "", mp.get("note", "")]
        if mp.get("detected"):
            out.append(f"\nIris centers on {mp['detected']}/{mp['images']} images: mean error "
                       f"{mp['mean_err_px']:.2f} px, max {mp['max_err_px']:.2f} px (vs our apparent pupil centroid).")
    out += ["", "## Plots", "", "![errors](plots/errors.png)", "", "![torsion](plots/torsion.png)", "",
            "![trajectory](plots/trajectory.png)", "",
            f"Pipeline time for the sequence (11 renders + 22 mask passes): {s['pipeline_seconds']} s."]
    return "\n".join(out) + "\n"


def plots(rows, meta) -> None:
    pd = REP / "plots"
    pd.mkdir(exist_ok=True)
    k = [r["k"] for r in rows]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
    ax[0].bar(np.array(k) - 0.2, [r["reproj_L"] for r in rows], 0.4, label="L")
    ax[0].bar(np.array(k) + 0.2, [r["reproj_R"] for r in rows], 0.4, label="R")
    ax[0].axhline(0.5, color="r", ls="--", lw=1, label="target 0.5 px")
    ax[0].set(title="Reprojection error (projection vs geometric mask)", xlabel="frame k", ylabel="px")
    ax[0].legend()
    ax[1].bar(np.array(k) - 0.2, [r["refr_L"] for r in rows], 0.4, label="L")
    ax[1].bar(np.array(k) + 0.2, [r["refr_R"] for r in rows], 0.4, label="R")
    ax[1].set(title="Corneal refraction shift (apparent vs geometric)", xlabel="frame k", ylabel="px")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(pd / "errors.png", dpi=110)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.plot(k, [r["torsion_spec_deg"] for r in rows], "o-", label="spec (look-at / Fick)")
    ax.plot(k, [r["torsion_full_deg"] for r in rows], "s-", label="physio_full (L2)")
    ax.plot(k, [r["torsion_physio_deg"] for r in rows], "^-", label="physio (Listing)")
    ax.set(title="|Torsion| about the line of sight vs Listing's law", xlabel="frame k", ylabel="deg")
    ax.legend()
    fig.tight_layout()
    fig.savefig(pd / "torsion.png", dpi=110)
    plt.close(fig)

    c = meta["center"]
    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    for s, col in (("L", "tab:red"), ("R", "tab:green")):
        d = np.array([np.subtract(f[f"pupil_center_2D_{s}"], c[f"pupil_center_2D_{s}"]) for f in meta["frames"]])
        ax.plot(d[:, 0], d[:, 1], "o-", color=col, label=f"eye {s}")
        for i, (x, y) in enumerate(d):
            ax.annotate(str(i), (x, y), fontsize=8)
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.set(title="Pupil displacement vs center (image px)", xlabel="dx (right)", ylabel="dy (down)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(pd / "trajectory.png", dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
