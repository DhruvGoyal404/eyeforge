#!/usr/bin/env bash
# EyeForge: full pipeline from a clean clone.
#   renders -> metadata.json -> contact sheet -> tests -> validation report -> bundle zip
# Env: BLENDER_PATH (default: blender on PATH), PYTHON (default: python3), EYEFORGE_FAST=1 for a
# quick CI-style run (masks-only direction test, 8-sample determinism check).
# REBUILD_SCENE=1 MH_ASSETS=<makehuman assets dir> also rebuilds scene.blend with MPFB2 first.
set -euo pipefail
export MSYS_NO_PATHCONV=1
cd "$(dirname "$0")"
PY="${PYTHON:-python3}"
command -v "$PY" >/dev/null || PY=python
export BLENDER_PATH="${BLENDER_PATH:-$(command -v blender || true)}"
[ -n "$BLENDER_PATH" ] || { echo "Blender 4.2 not found: set BLENDER_PATH" >&2; exit 1; }
export EYEFORGE_PYTHON="${EYEFORGE_PYTHON:-$(command -v "$PY")}"
T0=$(date +%s)

if [ "${REBUILD_SCENE:-0}" = "1" ]; then
  "$BLENDER_PATH" -b --python-exit-code 1 -P tools/build_scene.py -- --assets "${MH_ASSETS:?set MH_ASSETS}" --out scene.blend
fi
"$PY" wrapper/eyes.py sequence --out renders
"$PY" tools/contact_sheet.py --meta metadata.json --renders renders
"$PY" -m pytest -q tests
"$PY" tools/make_report.py
"$PY" tools/make_bundle.py
echo "run_all: done in $(( $(date +%s) - T0 )) s"
