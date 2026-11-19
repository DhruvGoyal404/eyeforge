"""Pure-math tests for wrapper/physiology.py (no Blender needed)."""
import math

import numpy as np
import pytest

import physiology as ph

# MPFB-like geometry: face looks along -Y, image right (camera +X) = +X = subject's left.
GEOM = ph.Geometry(C_L=np.array([0.0315, 0.0, 1.6]), C_R=np.array([-0.0315, 0.0, 1.6]),
                   F=np.array([0.0, -1.0, 0.0]), U=np.array([0.0, 0.0, 1.0]),
                   R_img=np.array([1.0, 0.0, 0.0]), pupil_depth=0.0103)
GRID = [(h, v) for h in np.linspace(-1, 1, 9) for v in np.linspace(-1, 1, 9)]


@pytest.mark.parametrize("h,v", GRID)
def test_listing_zero_torsion_and_unit_quats(h, v):
    pose = ph.solve(h, v, GEOM, "physio")
    for e in pose["eyes"].values():
        q = e["quat"]
        assert abs(np.linalg.norm(q) - 1) < 1e-12
        assert abs(e["torsion"]) < 1e-9
        # rotation axis perpendicular to the primary direction (Listing's plane)
        if np.linalg.norm(q[1:]) > 1e-9:
            assert abs(np.dot(q[1:] / np.linalg.norm(q[1:]), GEOM.F)) < 1e-9
        # eye actually looks at the target
        assert np.allclose(ph.q_rotate(q, GEOM.F), e["visual_axis"], atol=1e-12)


def test_spec_lookat_hits_target_but_has_false_torsion():
    pose = ph.solve(0.7071, 0.7071, GEOM, "spec")
    for e in pose["eyes"].values():
        assert np.allclose(ph.q_rotate(e["quat"], GEOM.F), e["visual_axis"], atol=1e-12)
        # Fick vs Listing false torsion: |tan(psi/2)| = tan(az/2) * tan(el/2)
        expect = 2 * math.atan(abs(math.tan(e["azimuth"] / 2) * math.tan(e["elevation"] / 2)))
        assert abs(abs(e["torsion"]) - expect) < 1e-9
    assert max(abs(e["torsion"]) for e in pose["eyes"].values()) > math.radians(0.25)
    # pure horizontal gaze stays in the plane of the eyes: no torsion even for look-at
    for h in (-1, 0, 1):
        for e in ph.solve(h, 0, GEOM, "spec")["eyes"].values():
            assert abs(e["torsion"]) < 1e-9


def test_vergence_at_center():
    pose = ph.solve(0, 0, GEOM, "physio")
    half = math.atan(GEOM.ipd / 2 / 0.35)
    assert abs(pose["vergence"] - 2 * half) < 1e-12
    assert abs(math.degrees(half) - 5.14) < 0.02
    # each eye turns inward: left eye (at +X) yaws toward -X (image left)
    assert pose["eyes"]["L"]["azimuth"] < 0 < pose["eyes"]["R"]["azimuth"]


def test_clamping_and_bad_input():
    a = ph.solve(3.0, -7.0, GEOM)
    b = ph.solve(1.0, -1.0, GEOM)
    assert a["h"] == 1.0 and a["v"] == -1.0
    assert np.allclose(a["target"], b["target"])
    with pytest.raises(ValueError):
        ph.solve(float("nan"), 0, GEOM)
    with pytest.raises(ValueError):
        ph.solve(0, 0, GEOM, "nope")


def test_target_definition():
    T = ph.target(0.5, -0.5, GEOM)
    assert np.allclose(T, GEOM.M + 0.35 * GEOM.F + 0.025 * GEOM.R_img - 0.025 * GEOM.U)


def test_near_triad_pupil_and_lids():
    p = ph.solve(0, 0, GEOM, "physio")
    assert abs(p["pupil_mm"] - 3.5) < 0.01
    assert ph.solve(0, 0, GEOM, "spec")["pupil_mm"] == 4.0
    assert ph.solve(0, 0, GEOM, "physio", ph.Params(pupil_mm=5.0))["pupil_mm"] == 5.0
    up = ph.solve(0, 1, GEOM, "physio")["eyes"]["L"]
    assert up["upper_lid"] > 0 and abs(up["lower_lid"] - 0.3 * up["upper_lid"]) < 1e-12
    assert ph.solve(0, 1, GEOM, "spec")["eyes"]["L"]["upper_lid"] == 0.0


def test_kappa_and_l2_full_profile():
    full = ph.solve(0, 0, GEOM, "physio_full")
    for s, e in full["eyes"].items():
        ang = math.degrees(math.acos(np.clip(e["optical_axis"] @ e["visual_axis"], -1, 1)))
        assert abs(ang - 5.0) < 1e-6  # optical axis offset from visual axis by kappa
        # L2 adds torsion relative to pure Listing only for non-horizontal gaze under vergence
        assert abs(e["torsion"]) < 1e-9
    for e in ph.solve(0, 1, GEOM, "physio_full")["eyes"].values():
        assert abs(e["torsion"]) > math.radians(0.05)
    # visual axis still on target
    assert np.allclose(full["eyes"]["L"]["visual_axis"],
                       (full["target"] - GEOM.C_L) / np.linalg.norm(full["target"] - GEOM.C_L))


def test_euler_matches_matrix():
    q = ph.solve(0.3, -0.8, GEOM, "spec")["eyes"]["R"]["quat"]
    rx, ry, rz = ph.matrix_to_euler_xyz(ph.q_to_matrix(q))
    Rx = ph.q_to_matrix(ph.q_axis_angle([1, 0, 0], rx))
    Ry = ph.q_to_matrix(ph.q_axis_angle([0, 1, 0], ry))
    Rz = ph.q_to_matrix(ph.q_axis_angle([0, 0, 1], rz))
    assert np.allclose(Rz @ Ry @ Rx, ph.q_to_matrix(q), atol=1e-12)


def test_deterministic():
    a = ph.solve(0.123, 0.456, GEOM)
    b = ph.solve(0.123, 0.456, GEOM)
    assert np.array_equal(a["eyes"]["L"]["quat"], b["eyes"]["L"]["quat"])
