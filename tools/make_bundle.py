"""Build <TEAM_NAME>.zip in the exact layout the organizers require.

    team.zip
    ├── scene.blend          textured mesh, eye bones, materials (textures packed)
    ├── wrapper/eyes.py      eyes.move(h, v) / eyes.reset() / eyes.render()   (+ its helper modules)
    ├── renders/             the 10 clockwise frames frame_00.png ... frame_09.png
    ├── metadata.json        per-frame records
    ├── README.md            setup and run instructions
    └── extras: contact_sheet.png, eye_strip.png, direction_check.png, center.png, demo video, report

    python tools/make_bundle.py [--team Thapar_Orchestrators]
"""
from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEAM_NAME = "Thapar_Orchestrators"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--team", default=TEAM_NAME)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = Path(args.out or ROOT / "dist" / f"{args.team}.zip")
    out.parent.mkdir(parents=True, exist_ok=True)

    required = {"scene.blend": ROOT / "scene.blend", "metadata.json": ROOT / "metadata.json",
                "README.md": ROOT / "README.md"}
    for k in range(10):
        required[f"renders/frame_{k:02d}.png"] = ROOT / "renders" / f"frame_{k:02d}.png"
    for py in sorted((ROOT / "wrapper").glob("*.py")):
        required[f"wrapper/{py.name}"] = py
    optional = {"contact_sheet.png": ROOT / "renders" / "contact_sheet.png",
                "eye_strip.png": ROOT / "renders" / "eye_strip.png",
                "direction_check.png": ROOT / "reports" / "direction_check.png",
                "extras/center.png": ROOT / "renders" / "center.png",
                "extras/validation.md": ROOT / "reports" / "validation.md",
                "tools/contact_sheet.py": ROOT / "tools" / "contact_sheet.py",
                "demo.mp4": ROOT / "docs" / "demo.mp4", "LICENSE": ROOT / "LICENSE",
                "ASSETS.md": ROOT / "ASSETS.md", "requirements.txt": ROOT / "requirements.txt"}
    missing = [k for k, p in required.items() if not p.exists()]
    if missing:
        raise SystemExit(f"missing required files: {missing}")
    meta = json.loads((ROOT / "metadata.json").read_text())
    assert len(meta["frames"]) == 10, "metadata.json must hold 10 frames"

    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for arc, p in {**required, **{k: v for k, v in optional.items() if v.exists()}}.items():
            z.write(p, arc)
    with zipfile.ZipFile(out) as z:
        names = z.namelist()
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB, {len(names)} files)")
    for n in names:
        print("  ", n)


if __name__ == "__main__":
    main()
