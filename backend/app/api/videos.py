from __future__ import annotations

import re
import uuid
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse

from app.models.schemas import (
    BoundingBox,
    DetectedPerson,
    DetectFrameResponse,
    IndexJobStatus,
    IndexRequest,
    TrackSample,
    TracksResponse,
    VideoInfo,
)
from app.services import indexing, storage, video as video_service
from app.vision.detector import VideoTracker

router = APIRouter()

CHUNK_SIZE = 1024 * 1024  # 1MB streaming chunks


def _row_to_info(row) -> VideoInfo:
    det_count, track_count = storage.count_detections(row["video_id"])
    return VideoInfo(
        video_id=row["video_id"],
        camera_id=row["camera_id"],
        filename=row["filename"],
        duration_sec=row["duration_sec"],
        fps=row["fps"],
        width=row["width"],
        height=row["height"],
        status=row["status"],
        track_count=track_count,
        detection_count=det_count,
        error=row["error"],
    )


@router.get("/videos", response_model=list[VideoInfo])
def list_videos():
    return [_row_to_info(r) for r in storage.list_videos()]


@router.get("/videos/{video_id}", response_model=VideoInfo)
def get_video(video_id: str):
    row = storage.get_video(video_id)
    if row is None:
        raise HTTPException(404, "video not found")
    return _row_to_info(row)


@router.post("/videos", response_model=VideoInfo)
async def upload_video(camera_id: str = Form(...), file: UploadFile = File(...)):
    video_id = str(uuid.uuid4())[:8]
    dest_dir = video_service.VIDEOS_DIR
    dest_dir.mkdir(parents=True, exist_ok=True)
    safe_name = re.sub(r"[^A-Za-z0-9._-]", "_", file.filename or f"{video_id}.mp4")
    dest_path = dest_dir / f"{video_id}_{safe_name}"

    with open(dest_path, "wb") as f:
        while chunk := await file.read(CHUNK_SIZE):
            f.write(chunk)

    try:
        meta = video_service.probe(dest_path)
    except Exception as exc:
        dest_path.unlink(missing_ok=True)
        raise HTTPException(400, f"could not read video: {exc}")

    storage.upsert_video(
        video_id=video_id,
        camera_id=camera_id,
        filename=safe_name,
        path=str(dest_path),
        fps=meta.fps,
        width=meta.width,
        height=meta.height,
        frame_count=meta.frame_count,
        duration_sec=meta.duration_sec,
        status="unindexed",
    )
    return _row_to_info(storage.get_video(video_id))


@router.post("/videos/{video_id}/index", response_model=IndexJobStatus)
def index_video(video_id: str, req: IndexRequest, background_tasks: BackgroundTasks):
    row = storage.get_video(video_id)
    if row is None:
        raise HTTPException(404, "video not found")
    existing = indexing.get_job(video_id)
    if existing and existing.status == "running":
        raise HTTPException(409, "indexing already in progress for this video")

    background_tasks.add_task(indexing.run_indexing_job, video_id, req.sample_every_n_frames)
    return IndexJobStatus(video_id=video_id, status="queued")


@router.get("/videos/{video_id}/index/status", response_model=IndexJobStatus)
def index_status(video_id: str):
    job = indexing.get_job(video_id)
    if job is None:
        row = storage.get_video(video_id)
        if row is None:
            raise HTTPException(404, "video not found")
        return IndexJobStatus(video_id=video_id, status=row["status"])
    return IndexJobStatus(
        video_id=job.video_id,
        status=job.status,
        progress=job.progress(),
        frames_processed=job.frames_processed,
        total_frames=job.total_frames,
        tracks_found=job.tracks_found,
        detections_stored=job.detections_stored,
        error=job.error,
    )


def _iter_range(path: Path, start: int, end: int, chunk_size: int = CHUNK_SIZE):
    with open(path, "rb") as f:
        f.seek(start)
        remaining = end - start + 1
        while remaining > 0:
            data = f.read(min(chunk_size, remaining))
            if not data:
                break
            remaining -= len(data)
            yield data


@router.get("/videos/{video_id}/stream")
def stream_video(video_id: str, request: Request):
    """HTTP range-aware video streaming so the <video> element can seek."""
    row = storage.get_video(video_id)
    if row is None:
        raise HTTPException(404, "video not found")
    path = Path(row["path"])
    if not path.exists():
        raise HTTPException(404, "video file missing on disk")

    file_size = path.stat().st_size
    range_header = request.headers.get("range")
    media_type = "video/mp4" if path.suffix.lower() == ".mp4" else "video/x-msvideo"

    if range_header:
        match = re.match(r"bytes=(\d+)-(\d*)", range_header)
        start = int(match.group(1)) if match else 0
        end = int(match.group(2)) if match and match.group(2) else file_size - 1
        end = min(end, file_size - 1)
        headers = {
            "Content-Range": f"bytes {start}-{end}/{file_size}",
            "Accept-Ranges": "bytes",
            "Content-Length": str(end - start + 1),
        }
        return StreamingResponse(_iter_range(path, start, end), status_code=206, headers=headers, media_type=media_type)

    headers = {"Accept-Ranges": "bytes", "Content-Length": str(file_size)}
    return StreamingResponse(_iter_range(path, 0, file_size - 1), headers=headers, media_type=media_type)


@router.get("/videos/{video_id}/frame")
def get_frame(video_id: str, frame_number: int):
    row = storage.get_video(video_id)
    if row is None:
        raise HTTPException(404, "video not found")
    frame = video_service.read_frame(Path(row["path"]), frame_number)
    if frame is None:
        raise HTTPException(400, "could not read requested frame")
    return {"frame_number": frame_number, "jpeg_base64": video_service.frame_to_jpeg_base64(frame)}


@router.get("/videos/{video_id}/tracks", response_model=TracksResponse)
def get_tracks(video_id: str):
    """Toutes les positions trackées stockées pour cette vidéo (léger : pas 
    d'images, pas d'embeddings). Le frontend s'en sert pour dessiner un 
    cadre qui suit chaque personne trackée pendant la lecture, en 
    interpolant entre les frames échantillonnées lors de l'indexation."""
    row = storage.get_video(video_id)
    if row is None:
        raise HTTPException(404, "video not found")
    samples = storage.fetch_track_samples(video_id)
    return TracksResponse(
        video_id=video_id,
        fps=row["fps"],
        sample_stride_hint=indexing.DEFAULT_SAMPLE_EVERY_N_FRAMES,
        samples=[
            TrackSample(
                track_id=s["track_id"],
                frame_number=s["frame_number"],
                timestamp_sec=s["timestamp_sec"],
                bbox=BoundingBox(x1=s["x1"], y1=s["y1"], x2=s["x2"], y2=s["y2"]),
                confidence=s["confidence"],
            )
            for s in samples
        ],
    )


_click_select_tracker: Optional[VideoTracker] = None


def _get_click_select_tracker() -> VideoTracker:
    global _click_select_tracker
    if _click_select_tracker is None:
        _click_select_tracker = VideoTracker()
    return _click_select_tracker


@router.get("/videos/{video_id}/detect_frame", response_model=DetectFrameResponse)
def detect_frame(video_id: str, frame_number: int):
    """Détection de personnes (sans identité de tracking) sur une seule 
    frame, pour que l'UI permette de cliquer directement sur une personne 
    détectée en Search Mode plutôt que de dessiner un rectangle à la main. 
    Vraie inférence YOLO temps réel sur une frame, pas un lookup — marche 
    sur n'importe quelle frame en pause, indexée ou non."""
    row = storage.get_video(video_id)
    if row is None:
        raise HTTPException(404, "video not found")
    frame = video_service.read_frame(Path(row["path"]), frame_number)
    if frame is None:
        raise HTTPException(400, "could not read requested frame")

    tracker = _get_click_select_tracker()
    detections = tracker.detect_only(frame, conf=0.35)

    return DetectFrameResponse(
        video_id=video_id,
        frame_number=frame_number,
        people=[
            DetectedPerson(
                temp_id=i,
                bbox=BoundingBox(x1=d.x1, y1=d.y1, x2=d.x2, y2=d.y2),
                confidence=round(d.confidence, 3),
            )
            for i, d in enumerate(detections)
        ],
    )
