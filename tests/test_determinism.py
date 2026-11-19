"""Same inputs -> same pixels."""
import json

import numpy as np
from PIL import Image

from conftest import REPORTS, run_job


def test_render_twice_identical():
    res = run_job("determinism", REPORTS / "determinism")
    a, b = (np.asarray(Image.open(p), dtype=np.float64) for p in res["paths"])
    mad = float(np.abs(a - b).mean())
    res.update(mean_abs_diff_8bit=mad, max_abs_diff_8bit=float(np.abs(a - b).max()))
    (REPORTS / "determinism.json").write_text(json.dumps(res, indent=2))
    assert mad < 0.05, res
