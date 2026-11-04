"""Person detection + tracking.

Uses a YOLO11n model (COCO-pretrained, `person` class only) for detection and
Ultralytics' built-in ByteTrack implementation for frame-to-frame association.

This module is deliberately thin: it exposes one call, `track_frame`, that
takes a raw BGR frame and returns a list of `Detection` objects, each already
carrying a stable track_id for the video it belongs to. Track continuity
(`persist=True`) is kept per-video via a distinct tracker instance, because
ByteTrack's internal state must not be shared across unrelated videos/cameras.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
import threading

import numpy as np
from ultralytics import YOLO

MODEL_DIR = Path(__file__).resolve().parents[2] / "models"
DETECTOR_WEIGHTS = MODEL_DIR / "yolo11n.pt"

PERSON_CLASS_ID = 0  # COCO class index for "person"

_model_lock = threading.Lock()
_model: Optional[YOLO] = None


def get_detector() -> YOLO:
    """Lazily load the shared YOLO detector (thread-safe, loaded once)."""
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                _model = YOLO(str(DETECTOR_WEIGHTS))
    return _model


@dataclass
class Detection:
    track_id: int
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float

    def as_crop(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        x1 = max(0, int(self.x1))
        y1 = max(0, int(self.y1))
        x2 = min(w, int(self.x2))
        y2 = min(h, int(self.y2))
        if x2 <= x1 or y2 <= y1:
            return np.zeros((1, 1, 3), dtype=np.uint8)
        return frame[y1:y2, x1:x2]


class VideoTracker:
    """One instance per video. Wraps YOLO.track with persist=True so the
    ByteTrack track table is scoped to this video only."""

    def __init__(self):
        self.model = get_detector()
        self._started = False

    def track_frame(self, frame: np.ndarray, conf: float = 0.35) -> List[Detection]:
        results = self.model.track(
            frame,
            persist=True,
            classes=[PERSON_CLASS_ID],
            conf=conf,
            verbose=False,
            tracker="bytetrack.yaml",
        )
        r = results[0]
        detections: List[Detection] = []
        if r.boxes is None or r.boxes.id is None:
            return detections
        boxes_xyxy = r.boxes.xyxy.cpu().numpy()
        ids = r.boxes.id.cpu().numpy().astype(int)
        confs = r.boxes.conf.cpu().numpy()
        for (x1, y1, x2, y2), tid, c in zip(boxes_xyxy, ids, confs):
            detections.append(Detection(track_id=int(tid), x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2), confidence=float(c)))
        return detections

    def detect_only(self, frame: np.ndarray, conf: float = 0.35) -> List[Detection]:
        """Single-frame detection with no tracker state, used for the
        interactive search-mode click/drag selection on the live frame."""
        results = self.model.predict(frame, classes=[PERSON_CLASS_ID], conf=conf, verbose=False)
        r = results[0]
        detections: List[Detection] = []
        if r.boxes is None:
            return detections
        boxes_xyxy = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        for i, ((x1, y1, x2, y2), c) in enumerate(zip(boxes_xyxy, confs)):
            detections.append(Detection(track_id=-1, x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2), confidence=float(c)))
        return detections
