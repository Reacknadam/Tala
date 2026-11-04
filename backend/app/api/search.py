from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException

from app.models.schemas import BoundingBox, MatchResult, SearchRequest, SearchResponse
from app.services import storage, video as video_service
from app.vision.reid import get_extractor
from app.vision.search import search_by_embedding

router = APIRouter()


@router.post("/search", response_model=SearchResponse)
def search(req: SearchRequest):
    row = storage.get_video(req.video_id)
    if row is None:
        raise HTTPException(404, "source video not found")

    frame = video_service.read_frame(Path(row["path"]), req.frame_number)
    if frame is None:
        raise HTTPException(400, "could not read the requested frame from the source video")

    bbox = req.bbox
    if bbox.x2 <= bbox.x1 or bbox.y2 <= bbox.y1:
        raise HTTPException(400, "invalid bounding box")

    extractor = get_extractor()
    h, w = frame.shape[:2]
    x1, y1 = max(0, int(bbox.x1)), max(0, int(bbox.y1))
    x2, y2 = min(w, int(bbox.x2)), min(h, int(bbox.y2))
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        raise HTTPException(400, "selected region is empty")

    query_embedding = extractor.embed(crop)
    query_thumb = video_service.bytes_to_base64(video_service.crop_bytes_jpeg(frame, bbox.x1, bbox.y1, bbox.x2, bbox.y2))

    exclude = req.video_id if req.exclude_source_video else None
    hits, scanned = search_by_embedding(
        query_embedding, threshold=req.threshold, top_k=req.top_k, exclude_video_id=exclude
    )

    matches = [
        MatchResult(
            match_id=str(h.detection_id),
            video_id=h.video_id,
            camera_id=h.camera_id,
            track_id=h.track_id,
            frame_number=h.frame_number,
            timestamp_sec=h.timestamp_sec,
            similarity=round(h.similarity, 4),
            bbox=BoundingBox(x1=h.bbox[0], y1=h.bbox[1], x2=h.bbox[2], y2=h.bbox[3]),
            thumbnail_base64=video_service.bytes_to_base64(h.thumbnail),
        )
        for h in hits
    ]

    searched_videos = len({r["video_id"] for r in storage.list_videos()}) - (1 if exclude else 0)

    return SearchResponse(
        query_video_id=req.video_id,
        query_bbox=bbox,
        query_thumbnail_base64=query_thumb,
        matches=matches,
        searched_videos=max(0, searched_videos),
        candidates_scanned=scanned,
    )


@router.get("/matches/{match_id}", response_model=MatchResult)
def get_match(match_id: str):
    """Fetch a single stored detection by id. `similarity` is not meaningful
    here (there is no query to compare against outside a /search call) and
    is returned as 0.0 -- this endpoint exists for re-fetching match detail
    (e.g. bbox/thumbnail) given a match_id from a previous search response."""
    try:
        detection_id = int(match_id)
    except ValueError:
        raise HTTPException(400, "invalid match id")
    row = storage.get_detection(detection_id)
    if row is None:
        raise HTTPException(404, "match not found")
    return MatchResult(
        match_id=str(row["id"]),
        video_id=row["video_id"],
        camera_id=row["camera_id"],
        track_id=row["track_id"],
        frame_number=row["frame_number"],
        timestamp_sec=row["timestamp_sec"],
        similarity=0.0,
        bbox=BoundingBox(x1=row["x1"], y1=row["y1"], x2=row["x2"], y2=row["y2"]),
        thumbnail_base64=video_service.bytes_to_base64(row["thumbnail"]),
    )
