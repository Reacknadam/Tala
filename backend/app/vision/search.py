"""Cross-camera person search: cosine similarity over stored embeddings.

Deliberately simple: load candidate embeddings for the requested scope into
a NumPy matrix and do one matrix-vector product. For a hackathon-scale
index (thousands of detections) this is well under 10ms and a dedicated
vector database would be pure overhead.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from app.services import storage
from app.vision.reid import cosine_similarity_batch


@dataclass
class SearchHit:
    detection_id: int
    video_id: str
    camera_id: str
    track_id: int
    frame_number: int
    timestamp_sec: float
    similarity: float
    bbox: tuple
    thumbnail: bytes


def search_by_embedding(
    query_embedding: np.ndarray,
    threshold: float = 0.55,
    top_k: int = 25,
    exclude_video_id: Optional[str] = None,
) -> Tuple[List[SearchHit], int]:
    """Returns (ranked hits above threshold capped at top_k, total candidates scanned)."""
    rows = storage.fetch_candidate_detections(exclude_video_id=exclude_video_id)
    if not rows:
        return [], 0

    embeddings = np.frombuffer(b"".join(r["embedding"] for r in rows), dtype=np.float32)
    embeddings = embeddings.reshape(len(rows), -1)

    scores = cosine_similarity_batch(query_embedding, embeddings)

    order = np.argsort(-scores)
    hits: List[SearchHit] = []
    for idx in order:
        score = float(scores[idx])
        if score < threshold:
            break  # scores are sorted descending, nothing further clears the bar
        r = rows[int(idx)]
        hits.append(
            SearchHit(
                detection_id=r["id"],
                video_id=r["video_id"],
                camera_id=r["camera_id"],
                track_id=r["track_id"],
                frame_number=r["frame_number"],
                timestamp_sec=r["timestamp_sec"],
                similarity=score,
                bbox=(r["x1"], r["y1"], r["x2"], r["y2"]),
                thumbnail=r["thumbnail"],
            )
        )
        if len(hits) >= top_k:
            break
    return hits, len(rows)
