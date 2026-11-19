# EyeForge build plan

Deadline: 22 Nov 2026, 23:59 IST. The core (Phase 1) is finished and verified before any extra work.
Each phase ends with its gate checks run headless and a git commit.

| Phase | Scope | Gate |
|---|---|---|
| 1 Core | MPFB2 head, layered eyes, bone fix, locked camera, lens auto-calibration, `eyes.py` API, physio profile, 10-frame sequence, metadata, contact sheet, tests 1-5, README, Dockerfile, bundle builder | `./run_all.sh` produces renders, metadata, contact sheet, report, zip; all tests pass |
| 2 Realism + comparisons | lid coupling tuning, sclera/iris/limbus detail, spec vs physio vs physio_full side-by-sides, MediaPipe independent check, validation report | comparisons show the gain; required renders still pass |
| 3 Video + slides | `wrapper/motion.py` (main-sequence saccades, min-jerk, blinks, microsaccades), `tools/make_video.py`, `docs/slides_content.md` | one command renders the video; slides use measured numbers |
| 4 XR dataset factory | near-eye IR cameras + LED ring, glint/ellipse labels, `tools/factory.py` | 200-sample smoke run, overlay grid checked |
| 5 Live web demo | glTF export, Three.js + MediaPipe in the browser, JS port of physiology with parity test | runs on a laptop webcam |
| 6 Stretch | sim-to-real gaze CNN | needs a real dataset slice from the team |

Design rules
- The required path (`eyes.move/reset/render/sequence`) never imports extras; extras live in
  separate modules and output folders (`renders/compare`, `renders/extra`, `data/`, `web/`).
- `wrapper/physiology.py`, `calib.py`, `masks.py`, `motion.py` are pure numpy and unit-tested
  without Blender; `eyes.py` is the only module that touches bpy.
