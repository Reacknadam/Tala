"""Pitch calibration: pixel <-> meters homography for one video.

The operator clicks the four pitch corners in image order
``[near-left, far-left, far-right, near-right]``; they map to pitch
coordinates ``(0,0), (0,W), (L,W), (L,0)`` in meters (x along the length,
y across the width). Default dimensions are the FIFA standard 105 x 68 m.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

DEFAULT_LENGTH = 105.0
DEFAULT_WIDTH = 68.0


class PitchMapping:
    def __init__(
        self,
        pixel_points: Sequence[Sequence[float]],
        length: float = DEFAULT_LENGTH,
        width: float = DEFAULT_WIDTH,
    ) -> None:
        pts = np.asarray(pixel_points, dtype=np.float32)
        if pts.shape != (4, 2):
            raise ValueError("exactly 4 pixel points are required")
        if cv2.contourArea(pts) < 100.0:
            raise ValueError("degenerate pitch quad (area too small)")
        self.pixel_points = pts
        self.length = float(length)
        self.width = float(width)
        target = np.array(
            [[0, 0], [0, self.width], [self.length, self.width], [self.length, 0]],
            dtype=np.float32,
        )
        self.matrix = cv2.getPerspectiveTransform(pts, target)
        if not np.isfinite(self.matrix).all() or abs(np.linalg.det(self.matrix)) < 1e-12:
            raise ValueError("degenerate pitch quad (singular homography)")
        # pitch -> pixel: lets the UI re-project pitch markings onto the
        # video (AR overlay) and lets callers map off-quad positions.
        self.inv_matrix = np.linalg.inv(self.matrix)

    def to_pitch(self, x: float, y: float, strict: bool = True) -> Optional[Tuple[float, float]]:
        """Map one image point to meters, or None when outside the quad
        (strict=False keeps the projected value for display purposes)."""
        if strict and cv2.pointPolygonTest(self.pixel_points, (float(x), float(y)), False) < 0:
            return None
        point = np.array([[[x, y]]], dtype=np.float32)
        mapped = cv2.perspectiveTransform(point, self.matrix)[0, 0]
        return float(mapped[0]), float(mapped[1])

    def to_pixel(self, x: float, y: float) -> Tuple[float, float]:
        """Inverse map: pitch meters -> calibration-space pixel."""
        point = np.array([[[x, y]]], dtype=np.float32)
        mapped = cv2.perspectiveTransform(point, self.inv_matrix)[0, 0]
        return float(mapped[0]), float(mapped[1])

    def map_points(self, points: Sequence[Tuple[float, float]]) -> List[Optional[Tuple[float, float]]]:
        return [self.to_pitch(x, y) for x, y in points]
