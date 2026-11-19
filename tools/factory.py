"""XR near-eye dataset factory: headset-style IR eye-tracker images + tracker-ready labels.

This is a separate render path: it builds its own near-eye cameras and IR LED rings at runtime,
never saves scene.blend and never touches the required 40 cm renders.

    python tools/factory.py --n 200 --seed 0 --out data            # renders (via Blender) + finalize
    python tools/factory.py --n 5000 --seed 0 --out data --shard 0/4  # 4 parallel workers: 0/4..3/4
    python tools/factory.py --finalize --out data                   # labels.jsonl, COCO, overlay grid

Per sample and eye: data/images/{id}_{eye}.png (grayscale IR), data/masks/{id}_{eye}.png (class
index: 0 bg, 1 skin, 2 sclera, 3 iris, 4 pupil, 5 lashes/brows; seen through the cornea), and
data/labels/{id}.json. Resumable: finished samples are skipped. Deterministic: every sample's
randomization comes from (seed, id) only.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "wrapper"))
IMG_W, IMG_H = 400, 300
CLASSES = {1: "skin", 2: "sclera", 3: "iris", 4: "pupil", 5: "lashes_brows"}
CLASS_RGB = {0: (0, 0, 0), 1: (1, 0, 0), 2: (0, 1, 0), 3: (0, 0, 1), 4: (1, 1, 0), 5: (1, 0, 1)}
MAX_GAZE_DEG = 30.0
CAM_DIST = 0.045      # camera to eyeball center (~28 mm from the cornea), below and in front
LED_RING_R, LED_DIST, N_LED = 0.016, 0.038, 8


# =============================================================================== Blender side

def blender_main(args) -> None:
    import bpy
    from mathutils import Matrix, Vector

    import calib
    import eyes
    import physiology as ph

    rig = eyes._rig()
    sc, info = rig.scene, rig.info
    gm = info["eye_geometry_mm"]
    for o in sc.objects:  # the 40 cm studio lights are off in the IR path
        if o.type == "LIGHT":
            o.hide_render = True
    sc.world = _world(bpy, "EF_IR_World", (0.0, 0.0, 0.0))
    sc.render.resolution_x, sc.render.resolution_y, sc.render.resolution_percentage = IMG_W, IMG_H, 100
    sc.cycles.samples, sc.cycles.use_denoising = 16, True
    sc.view_settings.view_transform, sc.view_settings.look = "Standard", "None"
    sc.view_settings.exposure = 0.0
    ir = _ir_materials(bpy)
    body = sc.objects["EF_Body"]
    skin_slots = [(s, s.material) for s in body.material_slots]
    glasses = _glasses(bpy, rig)
    U, F, Rimg = (np.array(info[k]) for k in ("U", "F", "R_img"))

    out = Path(args.out)
    for d in ("images", "masks", "labels"):
        (out / d).mkdir(parents=True, exist_ok=True)
    k, n_shards = (int(x) for x in args.shard.split("/"))
    for i in range(args.start, args.start + args.n):
        if i % n_shards != k or (out / "labels" / f"{i:06d}.json").exists():
            continue
        rng = np.random.default_rng([args.seed, i])
        # ---- domain randomization
        ecc, phi = math.radians(MAX_GAZE_DEG) * math.sqrt(rng.uniform()), rng.uniform(0, 2 * math.pi)
        ang = ecc * np.array([math.cos(phi), math.sin(phi)])  # uniform over a 30 deg disk
        rng_geom = ph.Geometry(rig.geom.C_L, rig.geom.C_R, rig.geom.F, rig.geom.U, rig.geom.R_img,
                               rig.geom.pupil_depth, target_range=0.35)
        h, v = math.tan(ang[0]), math.tan(ang[1])  # target offset in units of 35 cm
        rig.geom = rng_geom
        pupil_mm = float(rng.uniform(2.0, 7.0))
        iris = _iris_color(rng)
        skin_gray = float(rng.uniform(0.35, 0.8))  # skin tone / IR reflectance
        led_power = float(rng.uniform(0.7, 1.3))
        use_glasses = bool(rng.uniform() < 0.3)
        rig.params = ph.Params(pupil_mm=pupil_mm, iris_color=iris)
        # melanin is nearly transparent in near-IR: irises of any color look mid-gray
        g_ir = 0.18 + 0.8 * float(np.dot(iris, (0.2126, 0.7152, 0.0722)))
        bpy.data.materials["EF_Iris"].node_tree.nodes["IrisColor"].outputs[0].default_value = (g_ir,) * 3 + (1,)
        rig.profile = "physio_full"
        rig.blink = 0.0
        rec = rig.move(max(-1, min(1, h)), max(-1, min(1, v)))
        ir["skin"].node_tree.nodes["Diffuse BSDF"].inputs["Color"].default_value = (skin_gray,) * 3 + (1,)
        for s, _ in skin_slots:
            s.material = ir["skin"]
        glasses.hide_render = not use_glasses
        label = dict(id=i, seed=args.seed, gaze_input=dict(h=rec["input"]["h"], v=rec["input"]["v"]),
                     gaze_deg=[math.degrees(a) for a in ang], pupil_mm=pupil_mm, iris_color=list(iris),
                     skin_ir_reflectance=skin_gray, led_power_scale=led_power, glasses=use_glasses,
                     target_3D=rec["extended"]["target_3D"], vergence_deg=rec["extended"]["vergence_deg"],
                     eyes={},
                     note="geometric 2D labels ignore spectacle refraction when glasses=True; "
                          "masks and pupil_ellipse_apparent are measured from the render")
        for side in "LR":
            C = np.array(info[f"C_{side}"])
            temporal = Rimg if side == "L" else -Rimg
            jitter = rng.normal(0, 0.003, 3)
            d = F * 0.85 - U * 0.45 + temporal * 0.15
            d /= np.linalg.norm(d)
            cam_loc = C + CAM_DIST * d + jitter
            look = C + 0.010 * F + rng.normal(0, 0.0015, 3)  # aim near the iris, not the eyeball center
            cam = _camera(bpy, f"EF_IR_Cam.{side}", cam_loc, look, U, Matrix, Vector, calib,
                          lens=float(rng.uniform(45, 55)))
            sc.camera = cam
            leds = _leds(bpy, side, C, d, U, led_power * 0.005, Matrix, Vector)
            K = calib.intrinsics(cam.data.lens, cam.data.sensor_width, IMG_W, IMG_H)
            R, t = calib.extrinsics(cam_loc, np.array(cam.matrix_world.to_3x3()))
            P = calib.projection(K, R, t)
            # ---- geometry labels (no refraction)
            q = np.array(rec["extended"][f"quaternion_wxyz_{side}"])
            Rq = ph.q_to_matrix(q)
            frame = np.column_stack([Rimg, U, F])  # eye-local axes at rest
            ax_x, ax_y, ax_z = (Rq @ frame[:, j] for j in range(3))
            circle = lambda r, z: [C + z * ax_z + r * (math.cos(a) * ax_x + math.sin(a) * ax_y)  # noqa: E731
                                   for a in np.linspace(0, 2 * math.pi, 72, endpoint=False)]
            pupil_pts = calib.project(P, circle(pupil_mm / 2000, gm["pupil_plane"] / 1000))
            limbus_pts = calib.project(P, circle(info["limbus_radius_mm"] / 1000, gm["z_limbus"] / 1000))
            cc = C + gm["cornea_center"] / 1000 * ax_z
            glints = []
            for j, L in enumerate(leds):
                g = _glint(cc, info["cornea_radius_mm"] / 1000, np.array(L.matrix_world.translation), cam_loc)
                on_cap = float((g - cc) @ ax_z) / (info["cornea_radius_mm"] / 1000) > math.cos(
                    math.asin(info["limbus_radius_mm"] / info["cornea_radius_mm"]))
                glints.append(dict(led=j, xy=calib.project(P, g).tolist(), on_cornea=bool(on_cap),
                                   point_3D=g.tolist()))
            stem = f"{i:06d}_{side}"
            sc.render.filepath = str((out / "images" / f"{stem}.png").resolve())
            sc.render.image_settings.color_mode = "BW"
            bpy.ops.render.render(write_still=True)
            sc.render.image_settings.color_mode = "RGB"
            _class_mask(bpy, sc, ir, out / "masks" / f"{stem}_rgb.png", skin_slots)
            for s, _ in skin_slots:
                s.material = ir["skin"]
            for L in leds:
                bpy.data.objects.remove(L)
            bpy.data.objects.remove(cam)
            e = rec["extended"]
            label["eyes"][side] = dict(
                image=f"images/{stem}.png", mask=f"masks/{stem}.png", K=K.tolist(), R=R.tolist(),
                t=t.tolist(), camera_matrix=P.tolist(), camera_location=cam_loc.tolist(),
                pupil_boundary_2D=pupil_pts.tolist(), limbus_boundary_2D=limbus_pts.tolist(),
                pupil_center_2D=calib.project(P, np.array(e[f"pupil_center_3D_{side}"])).tolist(),
                glints=glints, gaze_vector=e[f"gaze_vector_{side}"], optical_axis=e[f"optical_axis_{side}"],
                eyeball_center_3D=C.tolist(), pupil_center_3D=e[f"pupil_center_3D_{side}"],
                lid_aperture_mm=e[f"lid_aperture_mm_{side}"])
        for s, m in skin_slots:
            s.material = m
        (out / "labels" / f"{i:06d}.json").write_text(json.dumps(label))
        print(f"[factory] sample {i} done", flush=True)


def _glint(cc, r, led, cam, iters: int = 40):
    """Specular reflection point of a point light on a sphere: the surface normal must bisect
    the directions to the LED and to the camera (fixed-point iteration, converges fast)."""
    n = (led - cc) / np.linalg.norm(led - cc) + (cam - cc) / np.linalg.norm(cam - cc)
    n /= np.linalg.norm(n)
    for _ in range(iters):
        p = cc + r * n
        b = (led - p) / np.linalg.norm(led - p) + (cam - p) / np.linalg.norm(cam - p)
        n = b / np.linalg.norm(b)
    return cc + r * n


def _iris_color(rng):
    base = [(0.10, 0.042, 0.012), (0.05, 0.02, 0.006), (0.06, 0.11, 0.2), (0.07, 0.09, 0.05), (0.09, 0.09, 0.09)]
    c = np.array(base[rng.integers(len(base))]) * rng.uniform(0.7, 1.4)
    return tuple(float(x) for x in c)


def _world(bpy, name, col):
    w = bpy.data.worlds.get(name) or bpy.data.worlds.new(name)
    w.use_nodes = True
    w.node_tree.nodes["Background"].inputs["Color"].default_value = (*col, 1)
    return w


def _ir_materials(bpy):
    """IR look: skin is bright and smooth, iris texture flattens toward gray (melanin is
    nearly transparent in near-IR), cornea stays glass for sharp glints."""
    mats = {}
    m = bpy.data.materials.new("EF_IR_Skin")
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    dif = nt.nodes.new("ShaderNodeBsdfDiffuse")
    dif.name = "Diffuse BSDF"
    dif.inputs["Roughness"].default_value = 0.6
    nt.links.new(dif.outputs[0], nt.nodes.new("ShaderNodeOutputMaterial").inputs[0])
    mats["skin"] = m
    for name, col in CLASS_RGB.items():
        mm = bpy.data.materials.new(f"EF_Cls{name}")
        mm.use_nodes = True
        nt = mm.node_tree
        nt.nodes.clear()
        e = nt.nodes.new("ShaderNodeEmission")
        e.inputs["Color"].default_value = (*col, 1)
        nt.links.new(e.outputs[0], nt.nodes.new("ShaderNodeOutputMaterial").inputs[0])
        mats[f"cls{name}"] = mm
    return mats


def _class_mask(bpy, sc, ir, path: Path, skin_slots) -> None:
    """One emission pass, cornea kept as pure refraction -> class colors as actually seen."""
    mats = bpy.data.materials
    swapped, hidden = [], []
    cls = {"EF_Sclera": 2, "EF_Iris": 3, "EF_Pupil": 4}
    saved = (sc.cycles.samples, sc.cycles.use_denoising, sc.world, sc.cycles.use_adaptive_sampling)
    sc.cycles.samples, sc.cycles.use_denoising, sc.cycles.use_adaptive_sampling = 4, False, False
    sc.world = _world(bpy, "EF_MaskWorld", (0, 0, 0))
    for o in sc.objects:
        if o.type == "LIGHT" and not o.hide_render:
            o.hide_render = True
            hidden.append(o)
        if o.type != "MESH" or o.hide_render:
            continue
        for slot in o.material_slots:
            swapped.append((slot, slot.material))
            n = slot.material.name.split(".")[0] if slot.material else ""
            low = o.name.lower()
            if n == "EF_Cornea":
                slot.material = mats["EF_MaskGlass"]
            elif n in cls:
                slot.material = ir[f"cls{cls[n]}"]
            elif "brow" in low or "lash" in low:
                slot.material = ir["cls5"]
            elif "glasses" in low:
                slot.material = mats["EF_MaskGlass"]
            else:
                slot.material = ir["cls1"]
    sc.render.filepath = str(path.resolve())
    bpy.ops.render.render(write_still=True)
    for slot, m in swapped:
        slot.material = m
    for o in hidden:
        o.hide_render = False
    sc.cycles.samples, sc.cycles.use_denoising, sc.world, sc.cycles.use_adaptive_sampling = saved


def _camera(bpy, name, loc, look, up, Matrix, Vector, calib, lens):
    cd = bpy.data.cameras.new(name)
    cd.lens, cd.sensor_width, cd.sensor_fit, cd.clip_start = lens, 36.0, "HORIZONTAL", 0.003
    cam = bpy.data.objects.new(name, cd)
    bpy.context.scene.collection.objects.link(cam)
    cam.matrix_world = Matrix.Translation(Vector(loc)) @ Matrix(calib.look_at(loc, look, up)).to_4x4()
    return cam


def _leds(bpy, side, C, d, U, power, Matrix, Vector):
    """Ring of N_LED IR point LEDs around the eye-camera axis, LED_DIST from the eye center."""
    x = np.cross(d, U)
    x /= np.linalg.norm(x)
    y = np.cross(x, d)
    out = []
    for j in range(N_LED):
        a = 2 * math.pi * j / N_LED
        p = C + LED_DIST * d + LED_RING_R * (math.cos(a) * x + math.sin(a) * y)
        ld = bpy.data.lights.new(f"EF_IR_LED.{side}.{j}", "POINT")
        ld.energy, ld.shadow_soft_size = power, 0.0008
        o = bpy.data.objects.new(ld.name, ld)
        bpy.context.scene.collection.objects.link(o)
        o.matrix_world = Matrix.Translation(Vector(p))
        out.append(o)
    return out


def _glasses(bpy, rig):
    """Thin plano lenses 12 mm in front of the corneas with a dark frame ring (distractor glints)."""
    import bmesh
    from mathutils import Matrix, Vector
    info = rig.info
    F, U, Rimg = (Vector(info[k]) for k in ("F", "U", "R_img"))
    bm = bmesh.new()
    for s in "LR":
        C = Vector(info[f"C_{s}"]) + F * (info["eye_geometry_mm"]["apex"] / 1000 + 0.012)
        rot = Matrix([Rimg, U, F]).transposed().to_4x4()
        bmesh.ops.create_cone(bm, cap_ends=True, segments=48, radius1=0.022, radius2=0.022, depth=0.002,
                              matrix=Matrix.Translation(C) @ rot)
    me = bpy.data.meshes.new("EF_Glasses")
    bm.to_mesh(me)
    bm.free()
    o = bpy.data.objects.new("EF_Glasses", me)
    bpy.context.scene.collection.objects.link(o)
    m = bpy.data.materials.new("EF_GlassesLens")
    m.use_nodes = True
    g = m.node_tree.nodes.new("ShaderNodeBsdfGlass")
    g.inputs["IOR"].default_value = 1.5
    m.node_tree.links.new(g.outputs[0], m.node_tree.nodes["Material Output"].inputs[0])
    o.data.materials.append(m)
    o.hide_render = True
    return o


# =============================================================================== finalize (no bpy)

def finalize(out: Path) -> dict:
    import cv2
    from PIL import Image, ImageDraw
    labels = sorted((out / "labels").glob("*.json"))
    coco = dict(images=[], annotations=[], categories=[dict(id=k, name=v) for k, v in CLASSES.items()])
    ann_id, glint_err, glint_n, glint_found = 0, [], 0, 0
    with open(out / "labels.jsonl", "w") as jl:
        for lp in labels:
            lab = json.loads(lp.read_text())
            for side, e in lab["eyes"].items():
                img = np.asarray(Image.open(out / e["image"]).convert("L"), dtype=np.float32)
                rgb = np.asarray(Image.open(out / e["mask"].replace(".png", "_rgb.png")).convert("RGB"),
                                 dtype=np.float32) / 255
                cols = np.array([CLASS_RGB[c] for c in sorted(CLASS_RGB)], dtype=np.float32)
                idx = np.argmin(((rgb[:, :, None, :] - cols[None, None]) ** 2).sum(-1), axis=-1).astype(np.uint8)
                Image.fromarray(idx).save(out / e["mask"])
                e["pupil_ellipse"] = _ellipse(cv2, e.pop("pupil_boundary_2D"))
                e["limbus_ellipse"] = _ellipse(cv2, e.pop("limbus_boundary_2D"))
                pm = (idx == 4).astype(np.uint8)
                cs, _ = cv2.findContours(pm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                big = max(cs, key=len) if cs else None
                e["pupil_ellipse_apparent"] = (_ellipse(cv2, big[:, 0, :].astype(float) + 0.5)
                                               if big is not None and len(big) >= 6 else None)
                for g in e["glints"]:  # verify each predicted glint against the rendered image
                    x, y = g["xy"]
                    g["visible_in_render"] = False
                    if not g["on_cornea"] or not (3 <= x < img.shape[1] - 3 and 3 <= y < img.shape[0] - 3):
                        continue
                    glint_n += 1
                    win = img[int(y) - 3:int(y) + 4, int(x) - 3:int(x) + 4]
                    ctx = img[max(0, int(y) - 8):int(y) + 9, max(0, int(x) - 8):int(x) + 9]
                    # a glint is a local peak clearly brighter than its surroundings, on the eye
                    if win.max() >= np.median(ctx) + 40 and idx[int(y), int(x)] in (2, 3, 4):
                        yy, xx = np.unravel_index(np.argmax(win), win.shape)
                        err = math.hypot(int(x) - 3 + xx + 0.5 - x, int(y) - 3 + yy + 0.5 - y)
                        g["visible_in_render"], g["render_offset_px"] = True, err
                        glint_err.append(err)
                        glint_found += 1
                img_id = len(coco["images"])
                coco["images"].append(dict(id=img_id, file_name=e["image"], width=img.shape[1], height=img.shape[0]))
                for c in CLASSES:
                    m = (idx == c).astype(np.uint8)
                    if m.sum() == 0:
                        continue
                    ys, xs = np.nonzero(m)
                    polys = [p[:, 0, :].flatten().astype(float).tolist() for p in
                             cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0] if len(p) >= 3]
                    coco["annotations"].append(dict(id=ann_id, image_id=img_id, category_id=c, iscrowd=0,
                                                    area=int(m.sum()), segmentation=polys,
                                                    bbox=[int(xs.min()), int(ys.min()), int(np.ptp(xs)) + 1,
                                                          int(np.ptp(ys)) + 1]))
                    ann_id += 1
            jl.write(json.dumps(lab) + "\n")
    (out / "coco.json").write_text(json.dumps(coco))
    stats = dict(samples=len(labels), images=len(coco["images"]), glints_expected=glint_n,
                 glints_verified=glint_found, glint_verify_rate=glint_found / max(1, glint_n),
                 glint_offset_px_mean=float(np.mean(glint_err)) if glint_err else None)
    (out / "stats.json").write_text(json.dumps(stats, indent=2))
    overlay_grid(out, labels[:24])
    print("[factory]", stats)
    return stats


def _ellipse(cv2, pts):
    (cx, cy), (a, b), ang = cv2.fitEllipse(np.asarray(pts, np.float32))
    return dict(center=[cx, cy], axes=[a / 2, b / 2], angle_deg=ang)


def overlay_grid(out: Path, label_files) -> None:
    """Visual label check: pupil (green), apparent pupil (cyan), limbus (yellow), glints (red)."""
    from PIL import Image, ImageDraw
    tiles = []
    by_id = {json.loads(line)["id"]: json.loads(line) for line in open(out / "labels.jsonl")}
    for lp in label_files:
        lab = by_id[int(lp.stem)]
        for side, e in lab["eyes"].items():
            im = Image.open(out / e["image"]).convert("RGB")
            d = ImageDraw.Draw(im)
            for key, col in (("limbus_ellipse", (255, 220, 0)), ("pupil_ellipse", (0, 255, 0)),
                             ("pupil_ellipse_apparent", (0, 220, 255))):
                el = e.get(key)
                if el:
                    pts = [(el["center"][0] + el["axes"][0] * math.cos(t) * math.cos(math.radians(el["angle_deg"]))
                            - el["axes"][1] * math.sin(t) * math.sin(math.radians(el["angle_deg"])),
                            el["center"][1] + el["axes"][0] * math.cos(t) * math.sin(math.radians(el["angle_deg"]))
                            + el["axes"][1] * math.sin(t) * math.cos(math.radians(el["angle_deg"])))
                           for t in np.linspace(0, 2 * math.pi, 60)]
                    d.line(pts + [pts[0]], fill=col, width=1)
            for g in e["glints"]:
                if g["on_cornea"]:
                    x, y = g["xy"]
                    d.ellipse([x - 3, y - 3, x + 3, y + 3], outline=(255, 0, 0) if g.get("visible_in_render")
                              else (120, 0, 0))
            d.text((4, 4), f"#{lab['id']} {side} gaze {lab['gaze_deg'][0]:+.0f},{lab['gaze_deg'][1]:+.0f}° "
                   f"pupil {lab['pupil_mm']:.1f}mm{' glasses' if lab['glasses'] else ''}", fill=(255, 255, 255))
            tiles.append(im)
    if not tiles:
        return
    cols = 6
    rows = math.ceil(len(tiles) / cols)
    grid = Image.new("RGB", (cols * IMG_W, rows * IMG_H))
    for j, t in enumerate(tiles):
        grid.paste(t, ((j % cols) * IMG_W, (j // cols) * IMG_H))
    grid.save(out / "overlay_grid.png")


# =============================================================================== CLI

def main() -> None:
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "data"))
    ap.add_argument("--shard", default="0/1", help="k/N: this worker renders ids with id % N == k")
    ap.add_argument("--finalize", action="store_true", help="only build labels.jsonl/COCO/overlay")
    args = ap.parse_args(argv)
    try:
        import bpy  # noqa: F401
        in_blender = True
    except ImportError:
        in_blender = False
    if in_blender:
        blender_main(args)
        return
    if not args.finalize:
        import eyes
        cmd = [eyes.find_blender(), "-b", str(ROOT / "scene.blend"), "--python-exit-code", "1",
               "-P", str(Path(__file__).resolve()), "--"] + argv
        subprocess.run(cmd, check=True, env={**os.environ})
    finalize(Path(args.out))


if __name__ == "__main__":
    main()
