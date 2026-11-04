"""Video file helpers built directly on OpenCV's VideoCapture.

Kept intentionally small: probe metadata, grab a single frame by index, and
encode JPEG thumbnails/crops for the API to return as base64. No caching
layer -- for a handful of hackathon demo videos this is fast enough, and
adding one would be premature optimization.
"""
from __future__ import annotations

import base64
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

VIDEOS_DIR = Path(__file__).resolve().parents[3] / "videos"
VIDEOS_DIR.mkdir(parents=True, exist_ok=True)

THUMB_MAX_SIDE = 220
JPEG_QUALITY = 82


@dataclass
class VideoMeta:
    fps: float
    width: int
    height: int
    frame_count: int
    duration_sec: float


def probe(path: Path) -> VideoMeta:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0.0
    cap.release()
    return VideoMeta(fps=fps, width=width, height=height, frame_count=frame_count, duration_sec=duration)


def read_frame(path: Path, frame_number: int) -> Optional[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            return None
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_number))
        ret, frame = cap.read()
        return frame if ret else None
    finally:
        cap.release()


def crop_bytes_jpeg(frame: np.ndarray, x1: float, y1: float, x2: float, y2: float, max_side: int = THUMB_MAX_SIDE) -> bytes:
    h, w = frame.shape[:2]
    xi1, yi1 = max(0, int(x1)), max(0, int(y1))
    xi2, yi2 = min(w, int(x2)), min(h, int(y2))
    if xi2 <= xi1 or yi2 <= yi1:
        crop = np.zeros((max_side, max_side, 3), dtype=np.uint8)
    else:
        crop = frame[yi1:yi2, xi1:xi2]
        ch, cw = crop.shape[:2]
        scale = max_side / max(ch, cw)
        if scale < 1.0:
            crop = cv2.resize(crop, (max(1, int(cw * scale)), max(1, int(ch * scale))))
    ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return buf.tobytes()


def bytes_to_base64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def frame_to_jpeg_base64(frame: np.ndarray, quality: int = 85) -> str:
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return bytes_to_base64(buf.tobytes())
