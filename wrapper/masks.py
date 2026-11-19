"""Pupil mask images -> 2D centroids. Pure numpy; image loading works with or without bpy.

Mask passes are emission-only renders (see eyes.EyeRig.render_masks): the subject's left
pupil emits pure red, the right pupil pure green, everything else is black. In the
"apparent" pass the cornea stays as clear refracting glass, so the centroid is where the
pupil is actually seen; in the "geometric" pass the cornea is hidden.
"""
from __future__ import annotations

import numpy as np

CHANNEL = {"L": 0, "R": 1}


def centroid(channel: np.ndarray, thresh: float = 0.05) -> dict | None:
    """Intensity-weighted centroid of an HxW mask (row 0 = image top).

    Pixel (i, j) has its center at (j + 0.5, i + 0.5): origin at the top-left image corner.
    Returns {"x", "y", "area_px"} or None when the mask is empty.
    """
    w = np.where(channel > thresh, channel, 0.0).astype(np.float64)
    s = float(w.sum())
    if s < 1.0:
        return None
    ys, xs = np.indices(w.shape)
    return {"x": float((w * (xs + 0.5)).sum() / s), "y": float((w * (ys + 0.5)).sum() / s),
            "area_px": s}


def pupil_centroids(img: np.ndarray) -> dict:
    """img: HxWx(3|4) float in [0, 1], row 0 = top. Returns {"L": ..., "R": ...}."""
    return {s: centroid(img[..., c]) for s, c in CHANNEL.items()}


def load_image(path: str) -> np.ndarray:
    """HxWx3 float32 in [0, 1], row 0 = top. Uses PIL if present, else Blender's image IO."""
    try:
        from PIL import Image
        im = Image.open(path)
        arr = np.asarray(im).astype(np.float32)
        return arr[..., :3] / (65535.0 if arr.max() > 255 else 255.0)
    except ImportError:
        import bpy
        im = bpy.data.images.load(path, check_existing=False)
        try:
            im.colorspace_settings.name = "Non-Color"
            w, h = im.size
            px = np.empty(w * h * 4, np.float32)
            im.pixels.foreach_get(px)
            return px.reshape(h, w, 4)[::-1, :, :3].copy()  # Blender stores bottom row first
        finally:
            bpy.data.images.remove(im)
