"""Football (pitch intelligence) endpoints."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.football import service
from app.models.schemas import FootballAnalyzeRequest, FootballCalibrationRequest
from app.services import storage

router = APIRouter()


def _require_video(video_id: str) -> None:
    if storage.get_video(video_id) is None:
        raise HTTPException(404, "video not found")


@router.put("/football/{video_id}/calibration")
def save_calibration(video_id: str, req: FootballCalibrationRequest):
    _require_video(video_id)
    points = [[p.x, p.y] for p in req.points]
    try:
        from app.football.pitch import PitchMapping
        PitchMapping(points, req.pitch_length, req.pitch_width)  # validate
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    return service.save_calibration(video_id, points, req.pitch_length, req.pitch_width)


@router.get("/football/{video_id}/calibration")
def get_calibration(video_id: str):
    _require_video(video_id)
    data = service.load_calibration(video_id)
    if data is None:
        raise HTTPException(404, "no calibration for this video")
    return data


@router.post("/football/{video_id}/analyze")
def analyze(video_id: str, req: FootballAnalyzeRequest):
    _require_video(video_id)
    try:
        result = service.run_analysis(video_id, req.max_sampled_frames, req.stride)
    except LookupError:
        raise HTTPException(404, "video not found")
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    summary = {k: v for k, v in result.items() if k != "frames"}
    summary["frames_available"] = len(result["frames"])
    return summary


@router.get("/football/{video_id}/analysis")
def get_analysis(video_id: str):
    _require_video(video_id)
    data = service.load_analysis(video_id)
    if data is None:
        raise HTTPException(404, "no analysis for this video (run analyze first)")
    return data
