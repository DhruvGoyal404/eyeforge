"""The 10 required frames and metadata.json."""
import json
import math

import jsonschema
import numpy as np
from PIL import Image

from conftest import ROOT

REQUIRED = ["frame", "theta", "input", "eye_rotation_L", "eye_rotation_R", "pupil_center_2D_L",
            "pupil_center_2D_R", "camera_matrix"]


def test_frames_exist_and_are_1024_png(meta):
    for k in range(10):
        p = ROOT / "renders" / f"frame_{k:02d}.png"
        im = Image.open(p)
        assert im.format == "PNG" and im.size == (1024, 1024), p


def test_theta_h_v(meta):
    fr = meta["frames"]
    assert [f["frame"] for f in fr] == list(range(10))
    for f in fr:
        th = 2 * math.pi * f["frame"] / 10
        assert abs(f["theta"] - th) < 1e-12
        assert abs(f["input"]["h"] - math.sin(th)) < 1e-6
        assert abs(f["input"]["v"] - math.cos(th)) < 1e-6
        assert set(REQUIRED) <= set(f)
        assert np.asarray(f["camera_matrix"]).shape == (3, 4)


def test_schema(meta):
    schema = json.loads((ROOT / "schema" / "metadata.schema.json").read_text())
    jsonschema.validate(meta, schema)


def test_clockwise_from_top_in_image(meta):
    """Measured (rendered) pupil displacement vs center: k=0 points up and the angle
    increases clockwise in image space (x right, y down) by ~36 degrees per frame."""
    c = meta["center"]["extended"]
    angs = []
    for f in meta["frames"]:
        e = f["extended"]
        d = np.mean([np.subtract(e[f"pupil_center_2D_apparent_{s}"], c[f"pupil_center_2D_apparent_{s}"])
                     for s in "LR"], axis=0)
        angs.append(math.degrees(math.atan2(d[0], -d[1])) % 360)  # clockwise from image-up
    assert min(angs[0], 360 - angs[0]) < 10, angs
    unwrapped = np.unwrap(np.radians(angs))
    steps = np.degrees(np.diff(unwrapped))
    assert np.all(steps > 20) and np.all(steps < 52), steps


def test_reprojection_error(meta):
    errs = [x for f in meta["frames"] for x in f["extended"]["reprojection_error_px"]]
    assert np.mean(errs) < 0.5 and max(errs) < 1.0, errs
