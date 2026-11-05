"""Automatic pitch (playing field) recognition.

Segments the grass by color in HSV space, takes the largest connected
region, and extracts its four extreme corners ordered the same way the
manual calibration expects: ``[near-left, far-left, far-right, near-right]``
(near = bottom of the image). Also returns a simplified polygon of the
recognized region so the UI can show what was detected (AR-style overlay).

Pure OpenCV/NumPy on a single frame -- no learned model needed, works on
any grass pitch broadcast.
"""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

# Broadcast grass, OpenCV HSV scale (H in 0..179). Generous range: mowing
# bands, shadows and worn patches still count as field.
_H_LO, _H_HI = 35, 95
_S_MIN, _V_MIN = 40, 40

_DETECT_WIDTH = 640  # run segmentation on a downscaled frame, it's enough
_MIN_AREA_FRAC = 0.10  # field must cover at least this much of the frame


def _quad_from_contour(contour: np.ndarray) -> np.ndarray:
    """4 extreme corners of a (roughly trapezoidal) hull, ordered
    [near-left, far-left, far-right, near-right]."""
    pts = contour.reshape(-1, 2).astype(np.float32)
    s = pts[:, 0] + pts[:, 1]
    d = pts[:, 1] - pts[:, 0]
    return np.array([
        pts[int(np.argmax(d))],   # near-left:  big y, small x
        pts[int(np.argmin(s))],   # far-left:   small x, small y
        pts[int(np.argmin(d))],   # far-right:  small y, big x
        pts[int(np.argmax(s))],   # near-right: big x, big y
    ], dtype=np.float32)


def detect_pitch(frame: np.ndarray) -> Optional[dict]:
    """Detect the playing field on one BGR frame.

    Returns ``{"points": [[x,y]*4], "polygon": [[x,y]*k]}`` in original
    frame pixel coordinates, or None when no plausible field is found.
    """
    h0, w0 = frame.shape[:2]
    scale = _DETECT_WIDTH / max(1, w0)
    small = cv2.resize(frame, (_DETECT_WIDTH, max(1, int(h0 * scale)))) if scale < 1.0 else frame

    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (_H_LO, _S_MIN, _V_MIN), (_H_HI, 255, 255))

    # Fill player/line holes so the field is one connected blob, then trim
    # speckle. A large closing is the key step: persons stand *on* the grass
    # and would otherwise punch holes / split the region.
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    biggest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(biggest) < _MIN_AREA_FRAC * mask.shape[0] * mask.shape[1]:
        return None

    hull = cv2.convexHull(biggest)
    quad = _quad_from_contour(hull)
    polygon = cv2.approxPolyDP(biggest, 0.02 * cv2.arcLength(biggest, True), True)

    inv = 1.0 / scale if scale < 1.0 else 1.0
    return {
        "points": [[float(x * inv), float(y * inv)] for x, y in quad],
        "polygon": [[float(p[0][0] * inv), float(p[0][1] * inv)] for p in polygon],
    }
