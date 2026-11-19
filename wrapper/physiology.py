"""EyeForge physiology engine: pure numpy, no bpy.

Given a normalized gaze input (h, v) and the head geometry, compute for each eye the
world-frame rotation (about its own eyeball center), the visual and optical axes,
vergence, torsion, near-triad pupil size and eyelid coupling angles.

Quaternions are (w, x, y, z) numpy arrays. All vectors are world-frame, meters.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import numpy as np

# ----------------------------------------------------------------------------- quaternions


def q_axis_angle(axis, angle: float) -> np.ndarray:
    axis = np.asarray(axis, float)
    n = np.linalg.norm(axis)
    if n < 1e-15 or abs(angle) < 1e-15:
        return np.array([1.0, 0.0, 0.0, 0.0])
    s = math.sin(angle / 2.0) / n
    return np.array([math.cos(angle / 2.0), axis[0] * s, axis[1] * s, axis[2] * s])


def q_mul(a, b) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ])


def q_conj(q) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]])


def q_normalize(q) -> np.ndarray:
    q = np.asarray(q, float)
    q = q / np.linalg.norm(q)
    return q if q[0] >= 0 else -q  # canonical hemisphere -> deterministic output


def q_to_matrix(q) -> np.ndarray:
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def q_rotate(q, v) -> np.ndarray:
    return q_to_matrix(q) @ np.asarray(v, float)


def q_between(a, b) -> np.ndarray:
    """Single shortest rotation taking unit vector a to unit vector b.

    Axis normalize(a x b), angle acos(a . b). This is exactly Listing's rule when a is the
    primary direction: the rotation axis is perpendicular to a, so there is no torsion.
    """
    a = _unit(a)
    b = _unit(b)
    axis = np.cross(a, b)
    s = np.linalg.norm(axis)
    c = float(np.clip(np.dot(a, b), -1.0, 1.0))
    if s < 1e-12:
        if c > 0:
            return np.array([1.0, 0.0, 0.0, 0.0])
        raise ValueError("q_between: opposite vectors, rotation axis undefined")
    return q_normalize(q_axis_angle(axis / s, math.atan2(s, c)))


def matrix_to_euler_xyz(m) -> np.ndarray:
    """Blender 'XYZ' Euler: M = Rz(rz) @ Ry(ry) @ Rx(rx) (extrinsic X, then Y, then Z)."""
    ry = math.asin(float(np.clip(-m[2][0], -1.0, 1.0)))
    if abs(m[2][0]) < 1 - 1e-9:
        rx = math.atan2(m[2][1], m[2][2])
        rz = math.atan2(m[1][0], m[0][0])
    else:  # gimbal lock, never reached for eye rotations
        rx = math.atan2(-m[1][2], m[1][1])
        rz = 0.0
    return np.array([rx, ry, rz])


def twist_angle(q, axis) -> float:
    """Signed rotation angle of q about a unit axis (q must leave the axis invariant)."""
    axis = _unit(axis)
    return 2.0 * math.atan2(float(np.dot(q[1:], axis)), float(q[0]))


def _unit(v) -> np.ndarray:
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


# ----------------------------------------------------------------------------- inputs

@dataclass(frozen=True)
class Geometry:
    """Head geometry measured from the scene (see tools/build_scene.py)."""
    C_L: np.ndarray          # subject's left eyeball center (appears on image right)
    C_R: np.ndarray          # subject's right eyeball center
    F: np.ndarray            # unit forward, out of the face
    U: np.ndarray            # unit world up (+Z)
    R_img: np.ndarray        # unit image-right (camera +X in world) = subject's left
    pupil_depth: float       # eyeball center -> pupil center along the optical axis (m)
    target_dist: float = 0.35
    target_range: float = 0.05

    @property
    def M(self) -> np.ndarray:
        return 0.5 * (self.C_L + self.C_R)

    @property
    def ipd(self) -> float:
        return float(np.linalg.norm(self.C_L - self.C_R))

    @staticmethod
    def from_dict(d: dict) -> "Geometry":
        a = lambda k: np.asarray(d[k], float)  # noqa: E731
        return Geometry(a("C_L"), a("C_R"), _unit(a("F")), _unit(a("U")), _unit(a("R_img")),
                        float(d["pupil_depth"]), float(d.get("target_dist", 0.35)),
                        float(d.get("target_range", 0.05)))


@dataclass
class Params:
    pupil_mm: float | None = None      # explicit override; None = profile rule
    pupil_fixed_mm: float = 4.0        # "spec" profile fixed pupil
    pupil_base_mm: float = 4.5         # near triad: pupil = base - gain * accommodation(D)
    pupil_near_gain: float = 0.35      # mm per diopter -> 3.5 mm at 35 cm (2.86 D)
    iris_color: tuple = (0.10, 0.042, 0.012)  # linear RGB, hazel-brown
    kappa_deg: float = 5.0             # used only when the profile enables kappa
    l2_gain: float = 0.25              # used only when the profile enables L2
    upper_lid_gain: float = 1.0        # upper lid follows vertical gaze ~1:1
    lower_lid_gain: float = 0.3        # lower lid ~0.3x


PROFILES: dict[str, dict] = {
    # naive baseline: per-eye look-at (Fick gimbal: yaw then pitch), static lids, fixed pupil
    "spec": dict(rotation="lookat", lids=False, pupil="fixed", kappa=False, l2=False),
    # default for the required renders
    "physio": dict(rotation="listing", lids=True, pupil="near_triad", kappa=False, l2=False),
    "physio_full": dict(rotation="listing", lids=True, pupil="near_triad", kappa=True, l2=True),
}


def clamp_input(x: float) -> float:
    x = float(x)
    if not math.isfinite(x):
        raise ValueError(f"gaze input must be finite, got {x!r}")
    return max(-1.0, min(1.0, x))


def target(h: float, v: float, g: Geometry) -> np.ndarray:
    """T(h, v) = M + d*F + r*h*R_img + r*v*U, with h, v clamped to [-1, 1]."""
    h, v = clamp_input(h), clamp_input(v)
    return g.M + g.target_dist * g.F + g.target_range * (h * g.R_img + v * g.U)


# ----------------------------------------------------------------------------- solver

def _lookat_q(gaze, g: Geometry) -> np.ndarray:
    """Fick-gimbal look-at (what a 'track to' constraint does): yaw about U, then pitch."""
    az = math.atan2(float(gaze @ g.R_img), float(gaze @ g.F))   # + toward image right
    el = math.asin(float(np.clip(gaze @ g.U, -1, 1)))           # + up
    # Rot(U, az) maps F toward +R_img; Rot(R_img, -el) maps F toward +U.
    return q_normalize(q_mul(q_axis_angle(g.U, az), q_axis_angle(g.R_img, -el)))


def solve(h: float, v: float, geom: Geometry, profile: str = "physio",
          params: Params | None = None) -> dict:
    """Return the full physiological pose for input (h, v)."""
    if profile not in PROFILES:
        raise ValueError(f"unknown profile {profile!r}; choose from {sorted(PROFILES)}")
    prof = PROFILES[profile]
    p = params or Params()
    hc, vc = clamp_input(h), clamp_input(v)
    T = target(hc, vc, geom)

    gaze = {s: _unit(T - c) for s, c in (("L", geom.C_L), ("R", geom.C_R))}
    vergence = math.acos(float(np.clip(gaze["L"] @ gaze["R"], -1, 1)))
    kappa = math.radians(p.kappa_deg) if prof["kappa"] else 0.0
    l2 = p.l2_gain * vergence if prof["l2"] else 0.0
    # nasal direction for the subject's left eye is -R_img (toward the midline), +R_img for right
    nasal_sign = {"L": -1.0, "R": +1.0}

    eyes = {}
    for s in ("L", "R"):
        gz = gaze[s]
        # rest visual axis: optical axis F rotated nasally by kappa (Rot(U, a) moves F toward +R_img)
        v0 = q_rotate(q_axis_angle(geom.U, nasal_sign[s] * kappa), geom.F)
        listing_q = q_between(v0, gz)  # Listing's law with primary direction v0
        if prof["rotation"] == "lookat":
            q = _lookat_q(gz, geom)  # kappa is never enabled for "spec", so v0 == F
        else:
            # L2: primary direction rotated temporally by l2 under vergence, then Listing from there
            q_prim = q_axis_angle(geom.U, -nasal_sign[s] * l2)
            q = q_normalize(q_mul(q_between(q_rotate(q_prim, v0), gz), q_prim))
        # torsion = residual twist about the line of sight versus pure Listing (from v0)
        torsion = twist_angle(q_mul(q, q_conj(listing_q)), gz)
        optical = q_rotate(q, geom.F)
        el = math.asin(float(np.clip(gz @ geom.U, -1, 1)))
        eyes[s] = dict(
            quat=q, euler_xyz=matrix_to_euler_xyz(q_to_matrix(q)), visual_axis=gz,
            optical_axis=optical, elevation=el,
            azimuth=math.atan2(float(gz @ geom.R_img), float(gz @ geom.F)),
            torsion=torsion,
            pupil_center_3d=(geom.C_L if s == "L" else geom.C_R) + geom.pupil_depth * optical,
            upper_lid=(p.upper_lid_gain * el) if prof["lids"] else 0.0,
            lower_lid=(p.lower_lid_gain * el) if prof["lids"] else 0.0,
        )

    dist = float(np.linalg.norm(T - geom.M))
    if p.pupil_mm is not None:
        pupil = float(p.pupil_mm)
    elif prof["pupil"] == "fixed":
        pupil = p.pupil_fixed_mm
    else:
        pupil = p.pupil_base_mm - p.pupil_near_gain * (1.0 / dist)  # accommodation in diopters
    pupil = float(np.clip(pupil, 1.5, 8.0))

    return dict(h=hc, v=vc, target=T, target_distance=dist, vergence=vergence,
                accommodation_D=1.0 / dist, pupil_mm=pupil, kappa=kappa, l2_angle=l2,
                profile=profile, eyes=eyes)


def params_dict(p: Params) -> dict:
    return asdict(p)
