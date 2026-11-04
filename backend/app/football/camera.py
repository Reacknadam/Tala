"""Inter-frame camera compensation for moving broadcast cameras.

A pitch homography is only valid for one camera pose. This class tracks
Shi-Tomasi features frame to frame (Lucas-Kanade), estimates the
prev->current homography with RANSAC, and accumulates the transform that
maps any current-frame point back into the first frame's coordinates,
where the operator's pitch calibration lives. Unlike the reference
project's single-feature translation, this handles pan, tilt, zoom and
perspective change, and one outlier feature cannot dictate the result.
"""
from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np


class CameraCompensator:
    def __init__(self) -> None:
        self._to_ref = np.eye(3, dtype=np.float64)
        self._prev_gray: Optional[np.ndarray] = None
        self._prev_points: Optional[np.ndarray] = None

    @staticmethod
    def _detect(gray: np.ndarray) -> Optional[np.ndarray]:
        return cv2.goodFeaturesToTrack(gray, maxCorners=400, qualityLevel=0.01, minDistance=8, blockSize=7)

    def update(self, frame: np.ndarray) -> np.ndarray:
        """Feed one frame (in order); returns current->reference homography."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self._prev_gray is None or self._prev_points is None:
            self._prev_gray = gray
            self._prev_points = self._detect(gray)
            return self._to_ref.copy()

        new_points, status, _ = cv2.calcOpticalFlowPyrLK(
            self._prev_gray, gray, self._prev_points, None,
            winSize=(21, 21), maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )
        if new_points is not None and status is not None:
            ok = status.reshape(-1) == 1
            old_ok = self._prev_points.reshape(-1, 2)[ok]
            new_ok = new_points.reshape(-1, 2)[ok]
            if len(old_ok) >= 12:
                mapping, _inliers = cv2.findHomography(
                    old_ok.astype(np.float32), new_ok.astype(np.float32), cv2.RANSAC, 3.0)
                if mapping is not None and abs(np.linalg.det(mapping)) > 1e-9:
                    self._to_ref = self._to_ref @ np.linalg.inv(mapping)
            tracked = new_ok.reshape(-1, 1, 2).astype(np.float32) if len(new_ok) else None
        else:
            tracked = None

        # Refresh features in current coordinates before they age out.
        self._prev_points = tracked if tracked is not None and len(tracked) >= 120 else self._detect(gray)
        self._prev_gray = gray
        return self._to_ref.copy()

    def transform(self, x: float, y: float) -> Tuple[float, float]:
        point = np.array([[[x, y]]], dtype=np.float32)
        mapped = cv2.perspectiveTransform(point, self._to_ref.astype(np.float32))[0, 0]
        return float(mapped[0]), float(mapped[1])
