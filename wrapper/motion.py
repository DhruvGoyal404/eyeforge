"""Physiological eye-movement timelines: pure numpy, no bpy.

* Saccades follow the main sequence: duration = 2.2 ms/deg * amplitude + 21 ms, with a
  minimum-jerk position profile s(tau) = 10 tau^3 - 15 tau^4 + 6 tau^5.
* Fixations hold the target and carry tiny microsaccades (< 0.5 deg, same main sequence).
* Blinks every 3-5 s: lids close in ~100 ms and reopen in ~150 ms (blink = 0 open .. 1 closed).

plan(targets) returns one sample per video frame: dict(t, h, v, blink, event).
"""
from __future__ import annotations

import math

import numpy as np

DEG_PER_UNIT = math.degrees(math.atan(0.05 / 0.35))  # (h, v) = 1 is ~8.13 deg from M at 35 cm


def main_sequence_duration(amplitude_deg: float) -> float:
    return 0.0022 * amplitude_deg + 0.021


def min_jerk(tau):
    tau = np.clip(tau, 0.0, 1.0)
    return 10 * tau ** 3 - 15 * tau ** 4 + 6 * tau ** 5


def blink_profile(t_since: float, close: float = 0.100, opening: float = 0.150) -> float:
    """0 = open, 1 = closed; smooth (min-jerk) close then slower reopen."""
    if t_since < 0 or t_since > close + opening:
        return 0.0
    if t_since <= close:
        return float(min_jerk(t_since / close))
    return float(1 - min_jerk((t_since - close) / opening))


def _amp_deg(a, b) -> float:
    return math.hypot(b[0] - a[0], b[1] - a[1]) * DEG_PER_UNIT


def plan(targets, fps: int = 30, dwell=(0.7, 1.3), blink_every=(3.0, 5.0), micro_every=(0.35, 0.8),
         micro_amp_deg=(0.1, 0.45), seed: int = 0) -> list[dict]:
    """Continuous gaze timeline visiting `targets` [(h, v), ...] in order."""
    rng = np.random.default_rng(seed)
    segs = []  # (t0, t1, p0, p1, kind)
    t, cur = 0.0, tuple(map(float, targets[0]))
    for nxt in list(targets[1:]) + [None]:
        fix_end = t + float(rng.uniform(*dwell))
        tm = t + float(rng.uniform(*micro_every))
        fix = cur
        while tm < fix_end - 0.06:
            # microsaccades alternate: jump off the target by < 0.5 deg, then back onto it
            if fix == cur:
                ang = rng.uniform(0, 2 * math.pi)
                amp = float(rng.uniform(*micro_amp_deg)) / DEG_PER_UNIT
                p = (cur[0] + amp * math.cos(ang), cur[1] + amp * math.sin(ang))
            else:
                p = cur
            segs.append((t, tm, fix, fix, "fixation"))
            d = main_sequence_duration(_amp_deg(fix, p))
            segs.append((tm, tm + d, fix, p, "microsaccade"))
            t, fix = tm + d, p
            tm = t + float(rng.uniform(*micro_every))
        end = max(t + 0.06, fix_end)
        segs.append((t, end, fix, fix, "fixation"))
        t = end
        if nxt is None:
            break
        nxt = tuple(map(float, nxt))
        d = main_sequence_duration(_amp_deg(fix, nxt))
        segs.append((t, t + d, fix, nxt, "saccade"))
        t, cur = t + d, nxt
    total = t
    blinks, tb = [], float(rng.uniform(1.0, 2.0))
    while tb < total - 0.3:
        blinks.append(tb)
        tb += float(rng.uniform(*blink_every))

    out = []
    for i in range(int(math.floor(total * fps)) + 1):
        ti = i / fps
        seg = next((s for s in segs if s[0] <= ti <= s[1]), segs[-1])
        t0, t1, p0, p1, kind = seg
        s = float(min_jerk((ti - t0) / (t1 - t0))) if t1 > t0 else 1.0
        h, v = p0[0] + s * (p1[0] - p0[0]), p0[1] + s * (p1[1] - p0[1])
        b = max([blink_profile(ti - tb) for tb in blinks] + [0.0])
        out.append(dict(t=ti, h=max(-1.0, min(1.0, h)), v=max(-1.0, min(1.0, v)), blink=b,
                        event="blink" if b > 0 else kind))
    return out
