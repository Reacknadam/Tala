"""Indexing pipeline: turns a raw video file into searchable embeddings.

For each video we:
  1. Run the tracker on *every* frame (ByteTrack needs frame-to-frame
     continuity to keep track IDs stable -- skipping frames breaks it).
  2. Only persist a detection+embedding record every `sample_every_n_frames`
     frames per track, so storage stays small and indexing stays fast. This
     is the "sample frames at a reasonable interval" step from the spec,
     applied to what we store rather than to what the tracker sees.
  3. For each persisted detection, crop the person, run it through the re-id
     embedding model, and write (bbox, embedding, thumbnail) to SQLite.

Progress is tracked in-memory (`JOBS`) so the frontend can poll
GET /videos/{id}/index/status while a job runs in a background thread.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import cv2

from app.services import storage, video as video_service
from app.vision.detector import VideoTracker
from app.vision.reid import get_extractor

DEFAULT_SAMPLE_EVERY_N_FRAMES = 5
DETECTION_CONF_THRESHOLD = 0.35


@dataclass
class IndexJob:
    video_id: str
    status: str = "queued"  # queued | running | done | error
    frames_processed: int = 0
    total_frames: int = 0
    tracks_found: int = 0
    detections_stored: int = 0
    error: Optional[str] = None
    lock: threading.Lock = field(default_factory=threading.Lock)

    def progress(self) -> float:
        if self.total_frames <= 0:
            return 0.0
        return min(1.0, self.frames_processed / self.total_frames)


JOBS: Dict[str, IndexJob] = {}
_jobs_lock = threading.Lock()


def get_job(video_id: str) -> Optional[IndexJob]:
    with _jobs_lock:
        return JOBS.get(video_id)


def _set_job(job: IndexJob) -> None:
    with _jobs_lock:
        JOBS[job.video_id] = job


def run_indexing_job(video_id: str, sample_every_n_frames: int = DEFAULT_SAMPLE_EVERY_N_FRAMES) -> None:
    """Runs synchronously in whatever thread calls it (see api/videos.py,
    which schedules this via FastAPI BackgroundTasks)."""
    job = IndexJob(video_id=video_id, status="running")
    _set_job(job)

    row = storage.get_video(video_id)
    if row is None:
        job.status = "error"
        job.error = "video not found"
        return

    path = Path(row["path"])
    camera_id = row["camera_id"]
    fps = row["fps"]
    job.total_frames = row["frame_count"]

    storage.set_video_status(video_id, "indexing")
    storage.clear_detections_for_video(video_id)

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        job.status = "error"
        job.error = f"could not open video at {path}"
        storage.set_video_status(video_id, "error", job.error)
        return

    tracker = VideoTracker()
    extractor = get_extractor()

    seen_tracks: Dict[int, int] = {}  # track_id -> frames since last stored sample
    all_track_ids = set()
    frame_idx = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            detections = tracker.track_frame(frame, conf=DETECTION_CONF_THRESHOLD)

            for det in detections:
                all_track_ids.add(det.track_id)
                frames_since = seen_tracks.get(det.track_id)
                should_store = frames_since is None or frames_since >= sample_every_n_frames
                if not should_store:
                    seen_tracks[det.track_id] = frames_since + 1
                    continue

                seen_tracks[det.track_id] = 0
                crop = det.as_crop(frame)
                embedding = extractor.embed(crop)
                thumb = video_service.crop_bytes_jpeg(frame, det.x1, det.y1, det.x2, det.y2)
                timestamp_sec = frame_idx / fps if fps > 0 else 0.0

                storage.insert_detection(
                    video_id=video_id,
                    camera_id=camera_id,
                    track_id=det.track_id,
                    frame_number=frame_idx,
                    timestamp_sec=timestamp_sec,
                    bbox=(det.x1, det.y1, det.x2, det.y2),
                    confidence=det.confidence,
                    embedding=embedding,
                    thumbnail_bytes=thumb,
                )
                job.detections_stored += 1

            frame_idx += 1
            job.frames_processed = frame_idx
            job.tracks_found = len(all_track_ids)

        job.status = "done"
        storage.set_video_status(video_id, "indexed")
    except Exception as exc:  # noqa: BLE001 - surface any real failure to the UI
        job.status = "error"
        job.error = str(exc)
        storage.set_video_status(video_id, "error", str(exc))
    finally:
        cap.release()
