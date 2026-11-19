"""Camera calibration helpers: pure numpy, no bpy.

OpenCV convention: camera +X right, +Y down, +Z forward. Pixel coordinates are continuous
with the origin at the top-left corner of the image, so pixel (i, j) has its center at
(j + 0.5, i + 0.5) and the principal point of an unshifted camera is (W/2, H/2).
Blender cameras look down their local -Z with +Y up, hence the diag(1, -1, -1) flip.
"""
from __future__ import annotations

import math

import numpy as np

BLENDER_TO_CV = np.diag([1.0, -1.0, -1.0])


def look_at(cam_loc, target, up) -> np.ndarray:
    """World-from-camera rotation (columns = camera X, Y, Z axes) for a Blender camera at
    cam_loc looking at target (local -Z toward target, local +Y as close to `up` as possible)."""
    cam_loc, target, up = (np.asarray(a, float) for a in (cam_loc, target, up))
    z = cam_loc - target
    z /= np.linalg.norm(z)
    x = np.cross(up, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.column_stack([x, y, z])


def intrinsics(lens_mm: float, sensor_w_mm: float, W: int, H: int) -> np.ndarray:
    """K for a Blender camera with sensor_fit HORIZONTAL (or AUTO with W >= H), no shift."""
    f = lens_mm / sensor_w_mm * W
    return np.array([[f, 0.0, W / 2.0], [0.0, f, H / 2.0], [0.0, 0.0, 1.0]])


def extrinsics(cam_loc, R_world_from_blendercam) -> tuple[np.ndarray, np.ndarray]:
    """(R, t) mapping world points into OpenCV camera coordinates: X_cam = R X + t."""
    R = BLENDER_TO_CV @ np.asarray(R_world_from_blendercam, float).T
    return R, -R @ np.asarray(cam_loc, float)


def projection(K, R, t) -> np.ndarray:
    return np.asarray(K) @ np.column_stack([R, t])


def project(P, X) -> np.ndarray:
    """Project world point(s) X (3,) or (N, 3) with a 3x4 P. Returns (2,) or (N, 2) pixels."""
    X = np.asarray(X, float)
    X2 = np.atleast_2d(X)
    x = (np.asarray(P) @ np.column_stack([X2, np.ones(len(X2))]).T).T
    uv = x[:, :2] / x[:, 2:3]
    return uv[0] if X.ndim == 1 else uv


def auto_lens(cam_loc, R_wc, roi_pts, shift_pairs, W: int, sensor_w_mm: float = 36.0,
              margin: float = 0.06, min_shift_px: float = 15.0) -> dict:
    """Pick the focal length analytically.

    Pixel offsets from the principal point scale linearly with f, so with normalized image
    coordinates n = (x/z, y/z):
      * framing:  f_px * max|n_roi| <= W/2 - margin*W   -> upper bound f_fit
      * motion:   f_px * min|dn_shift| >= min_shift_px  -> lower bound f_shift
    shift_pairs: list of (X_center, X_extreme, axis) with axis 0 = image x, 1 = image y.
    We take the longest lens that still frames the ROI with the margin (maximizes the
    visible pupil motion) and require it to satisfy the motion bound.
    """
    R, t = extrinsics(cam_loc, R_wc)

    def norm(X):
        Xc = (R @ np.asarray(X, float).T).T + t
        return Xc[:, :2] / Xc[:, 2:3]

    a = float(np.abs(norm(roi_pts)).max())
    f_fit = (W / 2 - margin * W) / a
    shifts = [abs(float((norm([e]) - norm([c]))[0, ax])) for c, e, ax in shift_pairs]
    f_shift = min_shift_px / min(shifts)
    if f_fit < f_shift:
        raise RuntimeError(f"cannot frame ROI with margin {margin} and reach {min_shift_px}px "
                           f"shift (f_fit={f_fit:.0f}px < f_shift={f_shift:.0f}px)")
    lens = math.floor(f_fit * sensor_w_mm / W * 2) / 2  # round down to 0.5 mm keeps the margin
    f_px = lens / sensor_w_mm * W
    return dict(lens_mm=lens, f_px=f_px, margin=margin, roi_max_norm=a,
                min_shift_px=min(shifts) * f_px, shifts_px=[s * f_px for s in shifts],
                lens_fit_mm=f_fit * sensor_w_mm / W, lens_min_for_shift_mm=f_shift * sensor_w_mm / W)
