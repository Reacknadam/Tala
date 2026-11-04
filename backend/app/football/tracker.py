"""Frame-level football detection on the shared YOLO11n weights.

Persons reuse TALA's ByteTrack pipeline; the ball uses the COCO
``sports ball`` class (32) that the same weights already know, so no new
model download is needed. Ball output is the best raw detection per frame;
temporal interpolation happens in the service layer.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

from app.vision.detector import MODEL_DIR, get_detector

PERSON_CLASS_ID = 0
BALL_CLASS_ID = 32

# Fine-tuned football weights (ball / goalkeeper / player / referee classes).
# The generic COCO model cannot see a broadcast ball, so these weights are
# used whenever they are present on disk -- see README "Model weights".
BEST_WEIGHTS = MODEL_DIR / "football_best.pt"

_best_model: Optional[YOLO] = None


def _get_best_model() -> Optional[YOLO]:
    global _best_model
    if _best_model is None and BEST_WEIGHTS.exists():
        _best_model = YOLO(str(BEST_WEIGHTS))
    return _best_model


class FootballTracker:
    """One instance per analyzed video (ByteTrack state is per-video)."""

    def __init__(self) -> None:
        best = _get_best_model()
        self.use_best = best is not None
        self.model = best if best is not None else get_detector()
        names = getattr(self.model, "names", {}) or {}
        self._name_to_id = {str(v): int(k) for k, v in names.items()}

    def _process_frame_best(self, frame: np.ndarray) -> Tuple[List[dict], Optional[dict]]:
        results = self.model.track(frame, persist=True, conf=0.25, verbose=False, tracker="bytetrack.yaml")
        boxes = results[0].boxes
        players: List[dict] = []
        ball: Optional[dict] = None
        ball_conf = -1.0
        if boxes is None or boxes.xyxy is None:
            return players, ball
        xyxy = boxes.xyxy.cpu().numpy()
        cls = boxes.cls.cpu().numpy().astype(int)
        confs = boxes.conf.cpu().numpy()
        ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else [-1] * len(xyxy)
        player_ids = {self._name_to_id.get("player", 2), self._name_to_id.get("goalkeeper", 1)}
        ball_id = self._name_to_id.get("ball", 0)
        for (x1, y1, x2, y2), class_id, score, track_id in zip(xyxy, cls, confs, ids):
            if class_id in player_ids and track_id >= 0:
                players.append({
                    "track_id": int(track_id),
                    "bbox": (float(x1), float(y1), float(x2), float(y2)),
                    "foot": (float((x1 + x2) / 2.0), float(y2)),
                    "confidence": float(score),
                })
            elif class_id == ball_id and score > ball_conf:
                ball_conf = float(score)
                ball = {"x": float((x1 + x2) / 2.0), "y": float((y1 + y2) / 2.0),
                        "bbox": (float(x1), float(y1), float(x2), float(y2)), "confidence": ball_conf}
        if ball is None:
            # Ball-only low-threshold pass (the reference project detects at
            # conf 0.1); size/aspect filter rejects socks and line markings.
            results2 = self.model.predict(frame, classes=[ball_id], conf=0.08, imgsz=1280, verbose=False)
            boxes2 = results2[0].boxes
            if boxes2 is not None and boxes2.xyxy is not None and len(boxes2.xyxy) > 0:
                xyxy2 = boxes2.xyxy.cpu().numpy()
                confs2 = boxes2.conf.cpu().numpy()
                candidates = [
                    i for i, (x1, y1, x2, y2) in enumerate(xyxy2)
                    if (x2 - x1) <= 50 and (y2 - y1) <= 50 and 0.5 <= (x2 - x1) / max(1.0, y2 - y1) <= 2.0
                ]
                if candidates:
                    best = max(candidates, key=lambda i: confs2[i])
                    x1, y1, x2, y2 = xyxy2[best]
                    ball = {"x": float((x1 + x2) / 2.0), "y": float((y1 + y2) / 2.0),
                            "bbox": (float(x1), float(y1), float(x2), float(y2)),
                            "confidence": float(confs2[best])}
        return players, ball

    def process_frame(self, frame: np.ndarray, conf: float = 0.3) -> Tuple[List[dict], Optional[dict]]:
        if self.use_best:
            return self._process_frame_best(frame)
        results = self.model.track(
            frame,
            persist=True,
            classes=[PERSON_CLASS_ID, BALL_CLASS_ID],
            conf=conf,
            verbose=False,
            tracker="bytetrack.yaml",
        )
        boxes = results[0].boxes
        players: List[dict] = []
        ball: Optional[dict] = None
        ball_conf = -1.0
        if boxes is None or boxes.xyxy is None:
            return players, ball
        xyxy = boxes.xyxy.cpu().numpy()
        cls = boxes.cls.cpu().numpy().astype(int)
        confs = boxes.conf.cpu().numpy()
        ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else [-1] * len(xyxy)
        for (x1, y1, x2, y2), class_id, score, track_id in zip(xyxy, cls, confs, ids):
            if class_id == PERSON_CLASS_ID and track_id >= 0:
                players.append({
                    "track_id": int(track_id),
                    "bbox": (float(x1), float(y1), float(x2), float(y2)),
                    "foot": (float((x1 + x2) / 2.0), float(y2)),
                    "confidence": float(score),
                })
            elif class_id == BALL_CLASS_ID and score > ball_conf:
                ball_conf = float(score)
                ball = {"x": float((x1 + x2) / 2.0), "y": float((y1 + y2) / 2.0),
                        "bbox": (float(x1), float(y1), float(x2), float(y2)), "confidence": ball_conf}
        if ball is None:
            center = self._detect_ball_fallback(frame)
            if center is not None:
                ball = {"x": center[0], "y": center[1], "bbox": None, "confidence": 0.1}
        return players, ball

    def _detect_ball_fallback(self, frame: np.ndarray) -> Optional[Tuple[float, float]]:
        """Second ball-only pass: the ball is a few pixels wide in broadcast
        footage, so it needs a lower threshold and larger input size than
        the person pass. False positives are filtered downstream by the
        possession hysteresis."""
        results = self.model.predict(frame, classes=[BALL_CLASS_ID], conf=0.1, imgsz=1280, verbose=False)
        boxes = results[0].boxes
        if boxes is None or boxes.xyxy is None or len(boxes.xyxy) == 0:
            return None
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        # A broadcast ball is a small, roughly square box; this rejects the
        # usual false positives (socks, shorts, line markings, ad boards).
        candidates = [
            i for i, (x1, y1, x2, y2) in enumerate(xyxy)
            if (x2 - x1) <= 50 and (y2 - y1) <= 50 and 0.5 <= (x2 - x1) / max(1.0, y2 - y1) <= 2.0
        ]
        if not candidates:
            return None
        best = max(candidates, key=lambda i: confs[i])
        x1, y1, x2, y2 = xyxy[best]
        return float((x1 + x2) / 2.0), float((y1 + y2) / 2.0)


def dominant_hue(frame: np.ndarray, bbox: Tuple[float, float, float, float]) -> Optional[float]:
    """Peak saturation-weighted hue (degrees) of the torso band, or None."""
    x1, y1, x2, y2 = (int(v) for v in bbox)
    h, w = frame.shape[:2]
    x1, x2 = max(0, x1), min(w, x2)
    top, bottom = max(0, int(y1 + 0.15 * (y2 - y1))), min(h, int(y1 + 0.55 * (y2 - y1)))
    if x2 <= x1 or bottom <= top:
        return None
    crop = frame[top:bottom, x1:x2]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hue, sat = hsv[..., 0].astype(np.float32) * 2.0, hsv[..., 1].astype(np.float32)
    if float(sat.mean()) < 40.0:
        return None
    hist, edges = np.histogram(hue, bins=36, range=(0, 360), weights=sat)
    return float((edges[int(np.argmax(hist))] + edges[int(np.argmax(hist)) + 1]) / 2.0)
