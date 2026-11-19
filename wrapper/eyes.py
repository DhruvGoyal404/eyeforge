"""EyeForge wrapper: rotate both eyeballs about their physical centers and render.

Inside Blender (scene.blend open):
    import sys; sys.path.append("<repo>/wrapper"); import eyes
    eyes.move(0.5, -0.2); eyes.render("out.png"); eyes.reset(); eyes.sequence("renders")

From a normal shell (launches headless Blender for you; set BLENDER_PATH if not on PATH):
    python wrapper/eyes.py sequence --out renders
    python wrapper/eyes.py pose 1 0 --render renders/right.png
or directly:
    blender -b scene.blend -P wrapper/eyes.py -- sequence --out renders
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import calib  # noqa: E402
import masks  # noqa: E402
import physiology as ph  # noqa: E402

try:
    import bpy
    from mathutils import Matrix, Quaternion, Vector
    IN_BLENDER = True
except ImportError:
    IN_BLENDER = False

CONVENTIONS = {
    "units": "meters for all 3D quantities; angles in radians (degrees duplicated under 'extended')",
    "world_up": "+Z (U)",
    "forward_F": "unit vector out of the face, detected from the mesh (MPFB faces world -Y)",
    "image_right_R_img": "camera +X axis in world space; image right is the SUBJECT'S LEFT",
    "eye_labels": "L/R are the subject's own left/right eye; the left eye appears on image right",
    "h_v": "h > 0 moves the pupils toward image right, v > 0 moves them up; inputs clamped to [-1, 1]",
    "eye_midpoint_M": "midpoint of the two least-squares-fitted eyeball centers",
    "fixation_target": "T(h, v) = M + 0.35*F + 0.05*h*R_img + 0.05*v*U",
    "camera": "location M + 0.40*F, looking exactly at M, up = U; never moves",
    "sequence": "theta_k = 2*pi*k/10, h = sin(theta_k), v = cos(theta_k); k = 0 is straight up, "
                "then clockwise as seen in the image",
    "pixel_coordinates": "origin at the image top-left corner, x right, y down, pixel (i, j) center "
                         "at (j + 0.5, i + 0.5) (OpenCV-style continuous coordinates)",
    "camera_matrix": "3x4 P = K [R | t], OpenCV convention (camera +X right, +Y down, +Z forward); "
                     "K, R, t also under extended",
    "eye_rotation": "world-frame Euler angles (Blender 'XYZ': R = Rz @ Ry @ Rx), radians, of the eyeball "
                    "rotation relative to its rest pose (optical axis along F); quaternion (w, x, y, z) "
                    "under extended",
    "pupil_center_2D": "geometric projection through P of the 3D pupil-disk center (no refraction); "
                       "the refraction-aware apparent centroid is under extended",
}


# ============================================================================= Blender side

class EyeRig:
    """Owns the pose state of the scene. The only state kept between calls is the current
    pose (h, v), the profile and the parameters."""

    def __init__(self, scene=None):
        self.scene = scene or bpy.context.scene
        if "eyeforge" not in self.scene:
            raise RuntimeError("this .blend has no EyeForge data; open scene.blend from the repo")
        self.info = json.loads(self.scene["eyeforge"])
        i = self.info
        self.geom = ph.Geometry(C_L=_a(i["C_L"]), C_R=_a(i["C_R"]), F=_a(i["F"]), U=_a(i["U"]),
                                R_img=_a(i["R_img"]), pupil_depth=i["pupil_depth"])
        self.arm = bpy.data.objects["EF_Rig"]
        self.cam = bpy.data.objects["EF_Camera"]
        self.profile = "physio"
        self.params = ph.Params()
        self.h = self.v = 0.0
        self.blink = 0.0  # 0 open .. 1 closed; only used by animate(), never by the required frames
        self.pose = None
        self.record = None
        self._check_camera()
        self._apply_iris_color()

    # ---------------------------------------------------------------- public API
    def set_profile(self, name: str) -> None:
        if name not in ph.PROFILES:
            raise ValueError(f"unknown profile {name!r}; choose from {sorted(ph.PROFILES)}")
        self.profile = name
        self.move(self.h, self.v)

    def set_params(self, **kwargs) -> None:
        for k, v in kwargs.items():
            if not hasattr(self.params, k):
                raise ValueError(f"unknown parameter {k!r}; valid: {sorted(vars(self.params))}")
            setattr(self.params, k, tuple(v) if k == "iris_color" else v)
        self._apply_iris_color()
        self.move(self.h, self.v)

    def move(self, h: float, v: float) -> dict:
        self.pose = ph.solve(h, v, self.geom, self.profile, self.params)
        self.h, self.v = self.pose["h"], self.pose["v"]
        self._apply_pose(self.pose)
        self.record = self._make_record(h, v)
        return self.record

    def reset(self) -> dict:
        return self.move(0.0, 0.0)

    def render(self, path: str | None = None) -> str:
        if self.pose is None:
            self.reset()
        self._check_camera()
        path = Path(path) if path else _default_dir() / f"eyes_h{self.h:+.3f}_v{self.v:+.3f}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        self.scene.render.filepath = str(path.resolve())
        t0 = time.perf_counter()
        bpy.ops.render.render(write_still=True)
        self.record["extended"]["render_seconds"] = round(time.perf_counter() - t0, 2)
        return str(path)

    def render_masks(self, prefix: str) -> dict:
        """Two cheap emission-only passes -> apparent (through the cornea) and geometric
        (cornea hidden) pupil centroids. Adds them to the current record."""
        sc = self.scene
        cy = sc.cycles
        saved = dict(samples=cy.samples, ad=cy.use_adaptive_sampling, dn=cy.use_denoising,
                     vt=sc.view_settings.view_transform, look=sc.view_settings.look,
                     world=sc.world, path=sc.render.filepath)
        mats = bpy.data.materials
        swapped, hidden = [], []
        try:
            cy.samples, cy.use_adaptive_sampling, cy.use_denoising = 16, False, False
            sc.view_settings.view_transform, sc.view_settings.look = "Standard", "None"
            sc.world = _black_world()
            for o in sc.objects:
                lower = o.name.lower()
                if o.type == "LIGHT" or (o.type == "MESH" and ("brow" in lower or "lash" in lower)):
                    if not o.hide_render:
                        o.hide_render = True
                        hidden.append(o)
                elif o.type == "MESH":
                    for slot in o.material_slots:
                        swapped.append((slot, slot.material))
                        n = slot.material.name if slot.material else ""
                        slot.material = (mats["EF_MaskPupil.L"] if n == "EF_Pupil" and o.name.endswith(".L")
                                         else mats["EF_MaskPupil.R"] if n == "EF_Pupil"
                                         else mats["EF_MaskGlass"] if n == "EF_Cornea"
                                         else mats["EF_MaskBlack"])
            out = {}
            for kind in ("apparent", "geometric"):
                corneas = [bpy.data.objects[f"EF_Cornea.{s}"] for s in "LR"]
                for c in corneas:
                    c.hide_render = kind == "geometric"
                p = f"{prefix}_{kind}.png"
                Path(p).parent.mkdir(parents=True, exist_ok=True)
                sc.render.filepath = str(Path(p).resolve())
                bpy.ops.render.render(write_still=True)
                out[kind] = masks.pupil_centroids(masks.load_image(sc.render.filepath))
        finally:
            for c in (bpy.data.objects[f"EF_Cornea.{s}"] for s in "LR"):
                c.hide_render = False
            for slot, m in swapped:
                slot.material = m
            for o in hidden:
                o.hide_render = False
            cy.samples, cy.use_adaptive_sampling, cy.use_denoising = saved["samples"], saved["ad"], saved["dn"]
            sc.view_settings.view_transform, sc.view_settings.look = saved["vt"], saved["look"]
            sc.world, sc.render.filepath = saved["world"], saved["path"]
        self._add_mask_results(out)
        return out

    def sequence(self, out_dir: str = "renders", n: int = 10, meta_path: str | None = None,
                 with_masks: bool = True) -> list[dict]:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        t0 = time.perf_counter()
        self.reset()
        self.render(str(out / "center.png"))
        if with_masks:
            self.render_masks(str(out / "extra" / "masks" / "center"))
        center = self.record
        frames = []
        for k in range(n):
            theta = 2.0 * math.pi * k / n
            rec = self.move(math.sin(theta), math.cos(theta))
            rec["frame"], rec["theta"] = k, theta
            self.render(str(out / f"frame_{k:02d}.png"))
            if with_masks:
                self.render_masks(str(out / "extra" / "masks" / f"frame_{k:02d}"))
            frames.append(rec)
            print(f"[eyes] frame {k}: h={rec['input']['h']:+.3f} v={rec['input']['v']:+.3f} "
                  f"L={rec['pupil_center_2D_L']} R={rec['pupil_center_2D_R']} "
                  f"({rec['extended']['render_seconds']} s)", flush=True)
        self.reset()
        meta = dict(conventions=CONVENTIONS, scene=self.scene_info(frames),
                    pipeline_seconds=round(time.perf_counter() - t0, 1),
                    center=center, frames=frames)
        meta_path = Path(meta_path) if meta_path else out.parent / "metadata.json"
        meta_path.write_text(json.dumps(meta, indent=2))
        print(f"[eyes] wrote {meta_path}", flush=True)
        _try_contact_sheet(meta_path, out)
        return frames

    def animate(self, targets, fps: int = 30, out_dir: str = "renders/extra/anim", samples: int = 6,
                percent: int = 50, seed: int = 0) -> list[dict]:
        """Physiological clip: main-sequence saccades between `targets` [(h, v), ...], blinks and
        microsaccades (wrapper/motion.py), rendered cheaply (eye-region border, low samples).
        Writes frame_%04d.png + timeline.json; restores render settings and the pose."""
        import motion
        tl = motion.plan(targets, fps=fps, seed=seed)
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        sc, r = self.scene, self.scene.render
        saved = (sc.cycles.samples, r.resolution_percentage, r.use_border, r.use_crop_to_border,
                 r.border_min_x, r.border_max_x, r.border_min_y, r.border_max_y)
        c = self.reset()["extended"]
        eR, eL = c["eyeball_center_2D_R"], c["eyeball_center_2D_L"]
        W, H = r.resolution_x, r.resolution_y
        yc = (eR[1] + eL[1]) / 2
        try:
            sc.cycles.samples, r.resolution_percentage = samples, percent
            r.use_border = r.use_crop_to_border = True
            r.border_min_x, r.border_max_x = (eR[0] - 190) / W, (eL[0] + 190) / W
            r.border_min_y, r.border_max_y = 1 - (yc + 110) / H, 1 - (yc - 110) / H
            for i, f in enumerate(tl):
                self.blink = f["blink"]
                rec = self.move(f["h"], f["v"])
                self.render(str(out / f"frame_{i:04d}.png"))
                f["pupil_center_2D_L"], f["pupil_center_2D_R"] = rec["pupil_center_2D_L"], rec["pupil_center_2D_R"]
                f["lid_aperture_mm_L"] = rec["extended"]["lid_aperture_mm_L"]
                if i % 30 == 0:
                    print(f"[eyes] animate {i}/{len(tl)} ({rec['extended']['render_seconds']} s/frame)", flush=True)
        finally:
            (sc.cycles.samples, r.resolution_percentage, r.use_border, r.use_crop_to_border,
             r.border_min_x, r.border_max_x, r.border_min_y, r.border_max_y) = saved
            self.blink = 0.0
            self.reset()
        meta = dict(fps=fps, targets=[list(t) for t in targets], seed=seed,
                    crop_px=[eR[0] - 190, yc - 110, eL[0] + 190, yc + 110], frames=tl)
        (out / "timeline.json").write_text(json.dumps(meta, indent=2))
        return tl

    # ---------------------------------------------------------------- internals
    def _apply_pose(self, pose: dict) -> None:
        A = self.arm.matrix_world
        Ainv = A.inverted()
        R_img = Vector(self.geom.R_img)
        eff = self.info.get("lid_efficiency", {"upper": 1.0, "lower": 1.0})
        keys = self.scene.objects["EF_Body"].data.shape_keys.key_blocks
        for s in "LR":
            C = Vector(self.geom.C_L if s == "L" else self.geom.C_R)
            e = pose["eyes"][s]
            rots = {self.info["eye_bones"][s]: Quaternion(e["quat"].tolist())}
            up_b, lo_b = self.info["lid_bones"][s]
            # lids rotate about the eye center's horizontal axis; Rot(R_img, -a) raises them.
            # Bone angles are divided by the measured weight efficiency so the lid margin itself
            # moves at the requested ratio of the gaze elevation.
            rots[up_b] = Quaternion(R_img, -e["upper_lid"] / eff["upper"])
            rots[lo_b] = Quaternion(R_img, -e["lower_lid"] / eff["lower"])
            if "blink_keys" in self.info:
                keys[self.info["blink_keys"][s]].value = self.blink
            for bone, q in rots.items():
                pb = self.arm.pose.bones[bone]
                Rw = Matrix.Translation(C) @ q.to_matrix().to_4x4() @ Matrix.Translation(-C)
                pb.matrix = Ainv @ Rw @ A @ pb.bone.matrix_local
            r = pose["pupil_mm"] / 2.0  # the pupil mesh has a 1 mm radius
            bpy.data.objects[f"EF_Pupil.{s}"].scale = (r, r, 1.0)
        bpy.context.view_layer.update()

    def projection(self) -> tuple:
        cam = self.cam
        K = calib.intrinsics(cam.data.lens, cam.data.sensor_width, self.scene.render.resolution_x,
                             self.scene.render.resolution_y)
        mw = cam.matrix_world
        R, t = calib.extrinsics(list(mw.translation), [list(r) for r in mw.to_3x3()])
        return K, R, t, calib.projection(K, R, t)

    def _make_record(self, h_in: float, v_in: float) -> dict:
        pose = self.pose
        K, R, t, P = self.projection()
        rec = {"frame": None, "theta": math.atan2(pose["h"], pose["v"]) % (2 * math.pi),
               "input": {"h": pose["h"], "v": pose["v"]}}
        ext = {"profile": self.profile, "raw_input": {"h": h_in, "v": v_in},
               "target_3D": pose["target"].tolist(), "target_distance_m": pose["target_distance"],
               "vergence_rad": pose["vergence"], "vergence_deg": math.degrees(pose["vergence"]),
               "accommodation_D": pose["accommodation_D"], "pupil_diameter_mm": pose["pupil_mm"],
               "kappa_deg": math.degrees(pose["kappa"]), "l2_angle_deg": math.degrees(pose["l2_angle"]),
               "K": K.tolist(), "R": R.tolist(), "t": t.tolist()}
        aperture = self._lid_apertures()
        for s in "LR":
            e = pose["eyes"][s]
            C = self.geom.C_L if s == "L" else self.geom.C_R
            pupil3d = self._pupil_center_world(s)
            drift = float(((pupil3d - e["pupil_center_3d"]) ** 2).sum() ** 0.5)
            if drift > 1e-6:
                raise RuntimeError(f"rig and physiology disagree on the {s} pupil by {drift} m")
            rec[f"eye_rotation_{s}"] = e["euler_xyz"].tolist()
            ext[f"eye_rotation_deg_{s}"] = [math.degrees(a) for a in e["euler_xyz"]]
            ext[f"quaternion_wxyz_{s}"] = e["quat"].tolist()
            ext[f"gaze_vector_{s}"] = e["visual_axis"].tolist()
            ext[f"optical_axis_{s}"] = e["optical_axis"].tolist()
            ext[f"eyeball_center_3D_{s}"] = list(C)
            ext[f"pupil_center_3D_{s}"] = pupil3d.tolist()
            ext[f"torsion_rad_{s}"] = e["torsion"]
            ext[f"torsion_deg_{s}"] = math.degrees(e["torsion"])
            ext[f"upper_lid_rotation_deg_{s}"] = math.degrees(e["upper_lid"])
            ext[f"lower_lid_rotation_deg_{s}"] = math.degrees(e["lower_lid"])
            ext[f"lid_aperture_mm_{s}"] = aperture[s]
            ext[f"eyeball_center_2D_{s}"] = calib.project(P, C).tolist()
        for s in "LR":
            rec[f"pupil_center_2D_{s}"] = calib.project(P, _a(ext[f"pupil_center_3D_{s}"])).tolist()
        rec["camera_matrix"] = P.tolist()
        rec["extended"] = ext
        return rec

    def _pupil_center_world(self, s: str):
        import numpy as np
        o = bpy.data.objects[f"EF_Pupil.{s}"]
        return np.array(o.matrix_world @ Vector((0.0, 0.0, self.info["eye_geometry_mm"]["pupil_plane"] / 1000)))

    def _lid_apertures(self) -> dict:
        """Measured on the deformed mesh: distance between the lid-margin vertices above and
        below each pupil (picked once in build_scene)."""
        body = bpy.data.objects["EF_Body"]
        mods = [m for m in body.modifiers if m.type == "MASK" and m.show_viewport]
        for m in mods:
            m.show_viewport = False  # keep vertex indices; viewport-only, renders unaffected
        try:
            dg = bpy.context.evaluated_depsgraph_get()
            ev = body.evaluated_get(dg)
            me = ev.to_mesh()
            out = {}
            for s, ids in self.info["lid_margin_vertices"].items():
                a, b = me.vertices[ids["upper"]].co, me.vertices[ids["lower"]].co
                out[s] = (body.matrix_world @ a - body.matrix_world @ b).length * 1000
            ev.to_mesh_clear()
        finally:
            for m in mods:
                m.show_viewport = True
        bpy.context.view_layer.update()
        return out

    def _add_mask_results(self, out: dict) -> None:
        rec, ext = self.record, self.record["extended"]
        for s in "LR":
            app, geo = out["apparent"][s], out["geometric"][s]
            proj = rec[f"pupil_center_2D_{s}"]
            ext[f"pupil_center_2D_apparent_{s}"] = [app["x"], app["y"]] if app else None
            ext[f"pupil_center_2D_geometric_mask_{s}"] = [geo["x"], geo["y"]] if geo else None
            ext[f"pupil_area_px_apparent_{s}"] = app["area_px"] if app else 0.0
            ext[f"pupil_area_px_geometric_{s}"] = geo["area_px"] if geo else 0.0
            ext[f"reprojection_error_px_{s}"] = _dist([geo["x"], geo["y"]], proj) if geo else None
            ext[f"refraction_shift_px_{s}"] = (_dist([app["x"], app["y"]], [geo["x"], geo["y"]])
                                               if app and geo else None)
        ext["reprojection_error_px"] = [ext["reprojection_error_px_L"], ext["reprojection_error_px_R"]]
        ext["refraction_shift_px"] = [ext["refraction_shift_px_L"], ext["refraction_shift_px_R"]]

    def _check_camera(self) -> None:
        import numpy as np
        loc = np.array(self.cam.matrix_world.translation)
        R = np.array(self.cam.matrix_world.to_3x3())
        if (np.abs(loc - self.info["cam_loc"]).max() > 1e-6
                or np.abs(R - self.info["cam_R_world_from_blendercam"]).max() > 1e-6
                or abs(self.cam.data.lens - self.info["lens"]["lens_mm"]) > 1e-9):
            raise RuntimeError("EF_Camera was moved or re-lensed; it must stay locked "
                               "(reload scene.blend)")

    def _apply_iris_color(self) -> None:
        node = bpy.data.materials["EF_Iris"].node_tree.nodes["IrisColor"]
        node.outputs[0].default_value = (*self.params.iris_color, 1.0)

    def scene_info(self, frames: list[dict] | None = None) -> dict:
        sc, i = self.scene, self.info
        lights = [dict(name=o.name, type=o.data.type, location=list(o.matrix_world.translation),
                       energy_W=o.data.energy, size_m=getattr(o.data, "size", None),
                       color=list(o.data.color)) for o in sc.objects if o.type == "LIGHT"]
        return {
            "blender_version": bpy.app.version_string, "engine": sc.render.engine,
            "device": sc.cycles.device, "samples": sc.cycles.samples,
            "adaptive_threshold": sc.cycles.adaptive_threshold, "seed": sc.cycles.seed,
            "denoiser": sc.cycles.denoiser if sc.cycles.use_denoising else None,
            "view_transform": sc.view_settings.view_transform,
            "resolution": [sc.render.resolution_x, sc.render.resolution_y],
            "lens_mm": self.cam.data.lens, "sensor_width_mm": self.cam.data.sensor_width,
            "sensor_fit": self.cam.data.sensor_fit, "lens_calibration": i["lens"],
            "camera_location": i["cam_loc"], "camera_distance_m": 0.40,
            "ipd_m": i["ipd"], "eyeball_radius_m": i["eyeball_radius"],
            "socket_sphere_radius_m": i["socket_sphere_radius"], "pupil_depth_m": i["pupil_depth"],
            "eye_geometry_mm": i["eye_geometry_mm"], "cornea_ior": i["cornea_ior"],
            "eyeball_center_L": i["C_L"], "eyeball_center_R": i["C_R"], "M": i["M"],
            "F": i["F"], "U": i["U"], "R_img": i["R_img"],
            "targets_T": [f["extended"]["target_3D"] for f in frames or []],
            "profile": self.profile, "params": ph.params_dict(self.params),
            "lights": lights, "world_color": list(sc.world.node_tree.nodes["Background"]
                                                  .inputs["Color"].default_value)[:3],
            "eye_bones": i["eye_bones"], "lid_bones": i["lid_bones"], "bone_fix": i["bones"],
            "head_asset": "MPFB2 2.0.18 human (MakeHuman CC0 assets), see ASSETS.md",
        }


def _a(x):
    import numpy as np
    return np.asarray(x, float)


def _dist(a, b) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _default_dir() -> Path:
    blend = bpy.data.filepath if IN_BLENDER else ""
    return (Path(blend).parent if blend else Path.cwd()) / "renders"


def _black_world():
    w = bpy.data.worlds.get("EF_MaskWorld")
    if w is None:
        w = bpy.data.worlds.new("EF_MaskWorld")
        w.use_nodes = True
        w.node_tree.nodes["Background"].inputs["Color"].default_value = (0, 0, 0, 1)
    return w


def _try_contact_sheet(meta_path: Path, out: Path) -> None:
    """Contact sheet needs Pillow. Blender's Python lacks it, so fall back to a system Python;
    never let this optional step break the required outputs."""
    tool = ROOT / "tools" / "contact_sheet.py"
    try:
        import PIL  # noqa: F401
        cmd = [sys.executable] if not IN_BLENDER else None
    except ImportError:
        cmd = None
    if cmd is None:
        py = os.environ.get("EYEFORGE_PYTHON") or shutil.which("python3") or shutil.which("python")
        cmd = [py] if py else None
    if cmd is None:
        print("[eyes] contact sheet skipped (no Python with Pillow); run tools/contact_sheet.py")
        return
    r = subprocess.run(cmd + [str(tool), "--meta", str(meta_path), "--renders", str(out)])
    if r.returncode:
        print("[eyes] contact sheet failed; run tools/contact_sheet.py manually")


# ============================================================================= module API

_RIG: EyeRig | None = None


def _rig() -> EyeRig:
    global _RIG
    if not IN_BLENDER:
        raise RuntimeError("eyes.move/reset/render need Blender. Run inside Blender with scene.blend, "
                           "or use the CLI: python wrapper/eyes.py sequence --out renders")
    if _RIG is None or _RIG.scene != bpy.context.scene:
        _RIG = EyeRig()
    return _RIG


def move(h: float, v: float) -> dict:
    """Clamp (h, v) to [-1, 1], rotate both eyes toward T(h, v), update lids/pupil.
    Returns this pose's metadata record."""
    return _rig().move(h, v)


def reset() -> dict:
    """h = v = 0: both eyes converge on the centered target 35 cm ahead."""
    return _rig().reset()


def render(path: str | None = None) -> str:
    """Render a 1024x1024 PNG with the locked camera; returns the file path."""
    return _rig().render(path)


def sequence(out_dir: str = "renders", n: int = 10, **kw) -> list[dict]:
    """The n clockwise frames (theta_k = 2*pi*k/n, h = sin, v = cos) + metadata.json + contact sheet."""
    return _rig().sequence(out_dir, n, **kw)


def animate(targets, fps: int = 30, **kw) -> list[dict]:
    """Saccades (main sequence, min-jerk), blinks, microsaccades between [(h, v), ...]."""
    return _rig().animate(targets, fps, **kw)


def set_profile(name: str) -> None:
    """'spec' | 'physio' (default) | 'physio_full'."""
    _rig().set_profile(name)


def set_params(**kwargs) -> None:
    """pupil_mm, iris_color, kappa_deg, l2_gain, upper_lid_gain, lower_lid_gain, ..."""
    _rig().set_params(**kwargs)


def current() -> dict | None:
    return _rig().record


# ============================================================================= CLI

def find_blender() -> str:
    cands = [os.environ.get("BLENDER_PATH"), shutil.which("blender"),
             r"C:\Program Files\Blender Foundation\Blender 4.2\blender.exe",
             "/Applications/Blender.app/Contents/MacOS/Blender", "/opt/blender/blender"]
    for c in cands:
        if c and Path(c).is_file():
            return c
    raise SystemExit("Blender 4.2 not found: set BLENDER_PATH=/path/to/blender (see README)")


def _cli(argv: list[str]) -> None:
    ap = argparse.ArgumentParser(prog="eyes.py", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", default=str(ROOT / "scene.blend"), help="only used outside Blender")
    ap.add_argument("--profile", default="physio", choices=sorted(ph.PROFILES))
    ap.add_argument("--samples", type=int, help="override Cycles samples")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sequence", help="10 clockwise frames + metadata.json + contact sheet")
    s.add_argument("--out", default="renders")
    s.add_argument("--n", type=int, default=10)
    s.add_argument("--meta", default=None, help="metadata.json path (default: <out>/../metadata.json)")
    s.add_argument("--no-masks", action="store_true")
    p = sub.add_parser("pose", help="move to (h, v), print the record, optionally render")
    p.add_argument("h", type=float)
    p.add_argument("v", type=float)
    p.add_argument("--render", default=None)
    p.add_argument("--masks", default=None, help="prefix for mask passes")
    p.add_argument("--json", default=None, help="write the record here")
    a = sub.add_parser("animate", help="physiological saccade/blink clip (cheap eye-region renders)")
    a.add_argument("--out", default="renders/extra/anim")
    a.add_argument("--fps", type=int, default=30)
    a.add_argument("--percent", type=int, default=50, help="render resolution percentage")
    a.add_argument("--targets", default="0,0;1,0;-0.7,0.7;0,-1;0.7,0.7;-1,0;0,0",
                   help="'h,v;h,v;...' fixation sequence")
    args = ap.parse_args(argv)

    if not IN_BLENDER:
        blender = find_blender()
        cmd = [blender, "-b", args.scene, "--python-exit-code", "1",
               "--python", str(Path(__file__).resolve()), "--"] + argv
        raise SystemExit(subprocess.run(cmd).returncode)

    rig = _rig()
    if args.samples:
        rig.scene.cycles.samples = args.samples
    rig.profile = args.profile
    if args.cmd == "sequence":
        rig.sequence(args.out, args.n, meta_path=args.meta, with_masks=not args.no_masks)
    elif args.cmd == "animate":
        tg = [tuple(float(x) for x in t.split(",")) for t in args.targets.split(";")]
        rig.animate(tg, args.fps, out_dir=args.out, percent=args.percent)
    elif args.cmd == "pose":
        rec = rig.move(args.h, args.v)
        if args.render:
            rig.render(args.render)
        if args.masks:
            rig.render_masks(args.masks)
        text = json.dumps(rec, indent=2)
        if args.json:
            Path(args.json).parent.mkdir(parents=True, exist_ok=True)
            Path(args.json).write_text(text)
        else:
            print(text)


if __name__ == "__main__":
    _argv = sys.argv[sys.argv.index("--") + 1:] if (IN_BLENDER and "--" in sys.argv) else sys.argv[1:]
    try:
        _cli(_argv)
    except SystemExit as e:
        if IN_BLENDER and e.code:
            sys.exit(e.code)
        if not IN_BLENDER:
            raise
    except Exception:
        import traceback
        traceback.print_exc()
        if IN_BLENDER:
            sys.exit(1)  # make headless Blender return non-zero on failure
        raise
