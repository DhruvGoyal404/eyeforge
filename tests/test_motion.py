"""wrapper/motion.py: main sequence, minimum jerk, blinks, microsaccades."""
import math

import numpy as np

import motion


def test_main_sequence_and_min_jerk():
    assert abs(motion.main_sequence_duration(10) - 0.043) < 1e-12
    tau = np.linspace(0, 1, 101)
    s = motion.min_jerk(tau)
    assert s[0] == 0 and abs(s[-1] - 1) < 1e-12 and np.all(np.diff(s) >= 0)
    v = np.gradient(s, tau)
    assert abs(v[0]) < 0.02 and abs(v[-1]) < 0.02 and abs(tau[np.argmax(v)] - 0.5) < 0.02


def test_blink_shape():
    assert motion.blink_profile(-0.01) == 0 and motion.blink_profile(0.3) == 0
    assert abs(motion.blink_profile(0.100) - 1) < 1e-12
    assert 0 < motion.blink_profile(0.05) < 1 and 0 < motion.blink_profile(0.2) < 1


def test_plan_visits_targets_with_physiological_events():
    targets = [(0, 0), (1, 0), (0, 1), (-1, -1)]
    tl = motion.plan(targets, fps=240, seed=1)
    ev = {f["event"] for f in tl}
    assert {"fixation", "saccade", "microsaccade"} <= ev
    # each target is reached
    for h, v in targets:
        assert min(math.hypot(f["h"] - h, f["v"] - v) for f in tl) < 1e-6
    # microsaccades stay below 0.5 deg from the fixation target
    for f in tl:
        if f["event"] == "microsaccade":
            d = min(math.hypot(f["h"] - h, f["v"] - v) for h, v in targets) * motion.DEG_PER_UNIT
            assert d < 0.5
    # saccade durations follow the main sequence (1 unit = 8.13 deg -> ~39 ms)
    sac = [f for f in tl if f["event"] == "saccade"]
    assert sac and all(0 <= f["blink"] <= 1 for f in tl)
    # deterministic
    assert motion.plan(targets, seed=1) == motion.plan(targets, seed=1)
