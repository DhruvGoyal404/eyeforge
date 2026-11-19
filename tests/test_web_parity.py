"""The browser demo's physiology.js must give the same quaternions as physiology.py."""
import json
import shutil
import subprocess

import numpy as np
import pytest

import physiology as ph
from conftest import ROOT
from test_physiology import GEOM


def test_js_matches_python():
    node = shutil.which("node")
    if not node:
        pytest.skip("node not installed")
    cases = [[h, v, p] for h in np.linspace(-1, 1, 5) for v in np.linspace(-1, 1, 5)
             for p in ("spec", "physio", "physio_full")]
    geom = {k: getattr(GEOM, k).tolist() for k in ("C_L", "C_R", "F", "U", "R_img")}
    r = subprocess.run([node, str(ROOT / "web" / "parity.mjs")], input=json.dumps(dict(geom=geom, cases=cases)),
                       capture_output=True, text=True, check=True)
    js = json.loads(r.stdout)
    for (h, v, p), j in zip(cases, js):
        py = ph.solve(h, v, GEOM, p)
        assert np.allclose(j["quat_L"], py["eyes"]["L"]["quat"], atol=1e-12)
        assert np.allclose(j["quat_R"], py["eyes"]["R"]["quat"], atol=1e-12)
        assert abs(j["torsion_L"] - py["eyes"]["L"]["torsion"]) < 1e-12
        assert abs(j["vergence"] - py["vergence"]) < 1e-12
        assert abs(j["pupil_mm"] - py["pupil_mm"]) < 1e-12
        assert abs(j["upper_lid_L"] - py["eyes"]["L"]["upper_lid"]) < 1e-12
