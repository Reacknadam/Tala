"""Unit test for inter-frame camera compensation.

A synthetic textured frame is shifted by a known translation; the
compensator must map points of the shifted frame back to reference
coordinates.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

from app.football.camera import CameraCompensator


class CameraTests(unittest.TestCase):
    def test_recovers_known_translation(self):
        rng = np.random.default_rng(7)
        base = np.zeros((400, 600, 3), dtype=np.uint8)
        for _ in range(300):
            x, y = int(rng.integers(30, 570)), int(rng.integers(30, 370))
            cv2.circle(base, (x, y), int(rng.integers(2, 6)), (255, 255, 255), -1)

        dx, dy = 12.0, -6.0
        shift = np.array([[1, 0, dx], [0, 1, dy], [0, 0, 1]], dtype=np.float64)
        moved = cv2.warpPerspective(base, shift, (600, 400))

        comp = CameraCompensator()
        comp.update(base)
        matrix = comp.update(moved)

        point = np.array([[[300.0, 200.0]]], dtype=np.float32)  # in moved frame
        back = cv2.perspectiveTransform(point, matrix.astype(np.float32))[0, 0]
        self.assertAlmostEqual(float(back[0]), 300.0 - dx, delta=2.0)
        self.assertAlmostEqual(float(back[1]), 200.0 - dy, delta=2.0)


if __name__ == "__main__":
    unittest.main()
