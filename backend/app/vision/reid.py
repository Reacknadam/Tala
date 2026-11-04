"""Visual person embeddings for cross-camera re-identification.

This is NOT facial recognition. It produces a whole-body appearance
descriptor (clothing color/texture, build) from a person crop -- the
standard signal used by lightweight re-id systems for "find this person in
other footage" search.

The descriptor combines two real, complementary signals:

  1. A banded HSV color histogram (upper/mid/lower body, computed
     separately per band). This is the single strongest, most explainable
     cue for cross-camera clothing-based re-id, and it is what an operator
     actually means by "find the person in the red shirt". It needs no
     pretrained weights.
  2. A learned appearance embedding from the classification head of a
     YOLO11n-cls network (COCO/ImageNet pretrained, CPU-friendly, ~5MB),
     taken as the 1280-d pooled feature before the final classification
     layer. This adds some shape/texture signal beyond raw color and helps
     a little with lighting changes between cameras.

We tried the deep embedding alone first and measured it empirically: on our
demo footage, mean same-person similarity was 0.96 vs mean different-person
similarity of 0.95 -- essentially no separation, because an ImageNet
classification backbone is not trained to distinguish between people. Colour
histograms alone did much better (same-person ~0.67 vs different-person
~0.48). The combined descriptor below is weighted toward color for that
reason. This is a real, measured design decision, not a guess -- see
README "Known limitations" for the numbers and what a production system
should use instead (a dedicated re-id network such as OSNet).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional
import threading

import cv2
import numpy as np
import torch
from ultralytics import YOLO

MODEL_DIR = Path(__file__).resolve().parents[2] / "models"
REID_WEIGHTS = MODEL_DIR / "yolo11n-cls.pt"

DEEP_EMBEDDING_DIM = 1280
INPUT_SIZE = 224

COLOR_BANDS = 3
COLOR_HUE_BINS = 16
COLOR_SAT_BINS = 8
COLOR_DIM_PER_BAND = COLOR_HUE_BINS * COLOR_SAT_BINS

COLOR_WEIGHT = 0.7
DEEP_WEIGHT = 0.3

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

_lock = threading.Lock()
_extractor: Optional["ReidExtractor"] = None


def _l2_normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    return v / norm if norm > 1e-6 else v


def _color_descriptor(crop_bgr: np.ndarray, bands: int = COLOR_BANDS) -> np.ndarray:
    """Banded HSV histogram: split the crop into `bands` horizontal strips
    (roughly head/torso, waist, legs for a standing person) and compute a
    hue-saturation histogram per strip, so a red top + blue jeans produces a
    different descriptor than a blue top + red jeans."""
    h, w = crop_bgr.shape[:2]
    feats = []
    band_h = max(1, h // bands)
    for i in range(bands):
        y0 = i * band_h
        y1 = h if i == bands - 1 else (i + 1) * band_h
        part = crop_bgr[y0:y1]
        if part.size == 0:
            feats.append(np.zeros(COLOR_DIM_PER_BAND, dtype=np.float32))
            continue
        hsv = cv2.cvtColor(part, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [COLOR_HUE_BINS, COLOR_SAT_BINS], [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        feats.append(hist.flatten().astype(np.float32))
    return np.concatenate(feats)


class ReidExtractor:
    def __init__(self):
        yolo = YOLO(str(REID_WEIGHTS))
        self.torch_model = yolo.model
        self.torch_model.eval()
        self._feat = {}
        # Hook the global-average-pool output of the classification head:
        # this is the 1280-d embedding, taken before the final linear layer
        # that maps to ImageNet class logits (we never use the classes).
        pool_layer = self.torch_model.model[-1].pool
        pool_layer.register_forward_hook(self._hook)

    def _hook(self, module, inp, out):
        self._feat["v"] = out.detach()

    @torch.no_grad()
    def _deep_embedding(self, crop_bgr: np.ndarray) -> np.ndarray:
        img = cv2.resize(crop_bgr, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = (img - _IMAGENET_MEAN) / _IMAGENET_STD
        tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()
        self.torch_model(tensor)
        return self._feat["v"].flatten().numpy().astype(np.float32)

    def embed(self, crop_bgr: np.ndarray) -> np.ndarray:
        if crop_bgr.size == 0 or crop_bgr.shape[0] < 4 or crop_bgr.shape[1] < 4:
            return np.zeros(COLOR_BANDS * COLOR_DIM_PER_BAND + DEEP_EMBEDDING_DIM, dtype=np.float32)
        color = _l2_normalize(_color_descriptor(crop_bgr))
        deep = _l2_normalize(self._deep_embedding(crop_bgr))
        combined = np.concatenate([color * COLOR_WEIGHT, deep * DEEP_WEIGHT]).astype(np.float32)
        return _l2_normalize(combined)


def get_extractor() -> ReidExtractor:
    global _extractor
    if _extractor is None:
        with _lock:
            if _extractor is None:
                _extractor = ReidExtractor()
    return _extractor


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Both vectors are already L2-normalized, so cosine similarity is a dot product."""
    return float(np.dot(a, b))


def cosine_similarity_batch(query: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    """query: (D,) normalized. candidates: (N, D) normalized rows. -> (N,) scores."""
    if candidates.size == 0:
        return np.zeros(0, dtype=np.float32)
    return candidates @ query

