"""Independent third-party sanity check: MediaPipe Face Landmarker iris centers vs our pupils.

Report only: never fails the build. The required renders are framed tightly on the eyes, so the
face detector needs context; each render is downscaled 0.5x onto a padded canvas and the iris
landmarks (468 and 473) are mapped back to full-resolution pixels.
"""
import json
import math
import urllib.request

import numpy as np
from PIL import Image

from conftest import ROOT, REPORTS

MODEL = ROOT / "assets" / "models" / "face_landmarker.task"
URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/"
       "float16/1/face_landmarker.task")
SCALE, PAD = 0.5, 256


def detector():
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision
    if not MODEL.exists():
        MODEL.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(URL, MODEL)
    det = vision.FaceLandmarker.create_from_options(
        vision.FaceLandmarkerOptions(base_options=BaseOptions(model_asset_path=str(MODEL)), num_faces=1))
    return det, mp


def iris_centers(det, mp, path):
    im = Image.open(path).convert("RGB")
    W = im.width
    canvas = Image.new("RGB", (W, W), (40, 40, 45))
    canvas.paste(im.resize((int(W * SCALE), int(W * SCALE)), Image.LANCZOS), (PAD, PAD))
    r = det.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.asarray(canvas)))
    if not r.face_landmarks:
        return None
    lm = r.face_landmarks[0]
    return [((lm[i].x * W - PAD) / SCALE, (lm[i].y * W - PAD) / SCALE) for i in (468, 473)]


def test_mediapipe_iris_vs_pupils(meta):
    det, mp = detector()
    recs = [("center.png", meta["center"])] + [(f"frame_{f['frame']:02d}.png", f) for f in meta["frames"]]
    errs, per, disp = [], [], []
    c_iris = None
    for name, rec in recs:
        found = iris_centers(det, mp, ROOT / "renders" / name)
        if found is None:
            per.append(dict(image=name, detected=False))
            continue
        ours = {s: rec["extended"][f"pupil_center_2D_apparent_{s}"] for s in "LR"}
        row = dict(image=name, detected=True)
        for p in found:  # assign each MediaPipe iris to the nearest of our two pupils
            s = min("LR", key=lambda k: math.dist(p, ours[k]))
            e = math.dist(p, ours[s])
            errs.append(e)
            row[f"mediapipe_{s}"], row[f"err_px_{s}"] = list(p), e
        if name == "center.png":
            c_iris = row
        elif c_iris and all(f"mediapipe_{s}" in row for s in "LR"):
            # does MediaPipe see the same motion direction as our labels?
            for s in "LR":
                dm = np.subtract(row[f"mediapipe_{s}"], c_iris[f"mediapipe_{s}"])
                do = np.subtract(ours[s], meta["center"]["extended"][f"pupil_center_2D_apparent_{s}"])
                cos = float(dm @ do / (np.linalg.norm(dm) * np.linalg.norm(do) + 1e-9))
                disp.append(cos)
        per.append(row)
    res = dict(images=len(recs), detected=sum(r["detected"] for r in per),
               mean_err_px=float(np.mean(errs)) if errs else None,
               max_err_px=float(np.max(errs)) if errs else None,
               direction_agreement_cos_mean=float(np.mean(disp)) if disp else None, per_image=per,
               note=("MediaPipe Face Landmarker (float16 v1) run on each render after a 0.5x downscale "
                     "onto a padded canvas; iris landmarks 468/473 compared with our apparent (refraction-"
                     "aware) pupil centroids. MediaPipe is trained on real faces and its landmark noise is "
                     "a few px at this scale, so this is a sanity check, not ground truth."))
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "mediapipe.json").write_text(json.dumps(res, indent=2))
    print(f"MediaPipe: {res['detected']}/{res['images']} detected, mean {res['mean_err_px']} px, "
          f"max {res['max_err_px']} px, direction cos {res['direction_agreement_cos_mean']}")
