"""P (OpenCV convention) vs Blender's own world_to_camera_view; camera placement."""
import json
import math

import numpy as np

import calib
from conftest import REPORTS, run_job


def test_pure_math_roundtrip():
    loc, M, up = np.array([0, -0.5, 1.5]), np.array([0, -0.1, 1.5]), np.array([0, 0, 1])
    Rwc = calib.look_at(loc, M, up)
    K = calib.intrinsics(112.0, 36.0, 1024, 1024)
    R, t = calib.extrinsics(loc, Rwc)
    P = calib.projection(K, R, t)
    assert np.allclose(calib.project(P, M), [512, 512])  # the look-at point is the image center
    assert calib.project(P, M + [0.01, 0, 0])[0] > 512   # +X world = image right here
    assert calib.project(P, M + [0, 0, 0.01])[1] < 512   # up = smaller y (y points down)
    assert np.allclose(R @ R.T, np.eye(3)) and abs(np.linalg.det(R) - 1) < 1e-12


def test_projection_matches_blender():
    res = run_job("camera", REPORTS / "camera")
    (REPORTS / "camera.json").write_text(json.dumps(res, indent=2))
    assert res["max_err_px"] < 0.01, res
    assert abs(res["cam_to_M_m"] - 0.40) < 1e-6
    assert abs(res["cam_offset_along_F_m"] - 0.40) < 1e-6
    assert res["look_dir_angle_deg"] < 1e-3
    assert res["euler_vs_mathutils_max_rad"] < 1e-6
