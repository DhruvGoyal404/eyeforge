"""Jobs that must run inside Blender; invoked by the pytest suite via tests/conftest.py.

    blender -b scene.blend --python-exit-code 1 -P tests/blender_jobs.py -- <job> <out_dir> [--fast]
"""
import json
import math
import sys
from pathlib import Path

import bpy
import numpy as np
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Quaternion, Vector

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "wrapper"))
import calib  # noqa: E402
import eyes  # noqa: E402

POSES = {"center": (0, 0), "left": (-1, 0), "right": (1, 0), "up": (0, 1), "down": (0, -1)}


def camera(out: Path, fast: bool) -> dict:
    rig = eyes._rig()
    sc, cam = rig.scene, rig.cam
    W, H = sc.render.resolution_x, sc.render.resolution_y
    rng = np.random.default_rng(0)
    M = np.array(rig.info["M"])
    pts = [rig.geom.C_L, rig.geom.C_R] + [M + rng.uniform(-0.06, 0.06, 3) for _ in range(50)]
    euler_err = 0.0
    for h, v in [(0.3, -0.7), (-1, 1), (0.9, 0.2)]:
        rec = rig.move(h, v)
        pts += [np.array(rec["extended"][f"pupil_center_3D_{s}"]) for s in "LR"]
        for s in "LR":
            q = Quaternion(rec["extended"][f"quaternion_wxyz_{s}"])
            e = q.to_euler("XYZ")
            euler_err = max(euler_err, float(np.abs(np.array(e) - rec[f"eye_rotation_{s}"]).max()))
    _, _, _, P = rig.projection()
    errs = []
    for p in pts:
        n = world_to_camera_view(sc, cam, Vector(p))
        ref = np.array([n.x * W, (1 - n.y) * H])
        errs.append(float(np.linalg.norm(calib.project(P, p) - ref)))
    loc = np.array(cam.matrix_world.translation)
    look = -np.array(cam.matrix_world.to_3x3().col[2])
    to_M = (M - loc) / np.linalg.norm(M - loc)
    return dict(max_err_px=max(errs), mean_err_px=float(np.mean(errs)), n_points=len(pts),
                cam_to_M_m=float(np.linalg.norm(M - loc)),
                cam_offset_along_F_m=float((loc - M) @ np.array(rig.info["F"])),
                look_dir_angle_deg=math.degrees(math.acos(min(1.0, float(look @ to_M)))),
                euler_vs_mathutils_max_rad=euler_err)


def directions(out: Path, fast: bool) -> dict:
    rig = eyes._rig()
    res = {}
    for name in list(POSES) + ["reset"]:
        rec = rig.reset() if name == "reset" else rig.move(*POSES[name])
        if not fast:
            rig.render(str(out / f"{name}.png"))
        rig.render_masks(str(out / "masks" / name))
        res[name] = rec
    return res


def determinism(out: Path, fast: bool) -> dict:
    rig = eyes._rig()
    if fast:
        rig.scene.cycles.samples = 8
    paths = []
    for i in range(2):
        rig.move(0.6, -0.4)
        paths.append(rig.render(str(out / f"det_{i}.png")))
    return dict(paths=paths, samples=rig.scene.cycles.samples,
                seconds=rig.record["extended"]["render_seconds"])


def main():
    argv = sys.argv[sys.argv.index("--") + 1:]
    job, out, fast = argv[0], Path(argv[1]), "--fast" in argv
    out.mkdir(parents=True, exist_ok=True)
    res = globals()[job](out, fast)
    (out / f"{job}.json").write_text(json.dumps(res, indent=2))


main()
