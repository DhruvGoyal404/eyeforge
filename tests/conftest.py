import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "wrapper"))
REPORTS = ROOT / "reports"
FAST = os.environ.get("EYEFORGE_FAST") == "1"  # CI: cheap renders, masks only


def blender_or_skip() -> str:
    import eyes
    try:
        return eyes.find_blender()
    except SystemExit:
        pytest.skip("Blender not found (set BLENDER_PATH)")


def run_job(job: str, out: Path) -> dict:
    """Run tests/blender_jobs.py <job> once per session; results are cached in <out>/<job>.json."""
    blender = blender_or_skip()
    cmd = [blender, "-b", str(ROOT / "scene.blend"), "--python-exit-code", "1",
           "-P", str(ROOT / "tests" / "blender_jobs.py"), "--", job, str(out)] + (["--fast"] if FAST else [])
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    return json.loads((out / f"{job}.json").read_text())


@pytest.fixture(scope="session")
def meta():
    p = ROOT / "metadata.json"
    if not p.exists():
        pytest.skip("metadata.json missing: run `python wrapper/eyes.py sequence --out renders` first")
    return json.loads(p.read_text())
