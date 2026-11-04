"""Pydantic request/response models for the TALA API."""
from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel


class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class VideoInfo(BaseModel):
    video_id: str
    camera_id: str
    filename: str
    duration_sec: float
    fps: float
    width: int
    height: int
    status: str  # "unindexed" | "indexing" | "indexed" | "error"
    track_count: int = 0
    detection_count: int = 0
    error: Optional[str] = None


class IndexRequest(BaseModel):
    sample_every_n_frames: int = 5


class IndexJobStatus(BaseModel):
    video_id: str
    status: str
    progress: float = 0.0
    frames_processed: int = 0
    total_frames: int = 0
    tracks_found: int = 0
    detections_stored: int = 0
    error: Optional[str] = None


class SearchRequest(BaseModel):
    video_id: str
    frame_number: int
    bbox: BoundingBox
    threshold: float = 0.55
    exclude_source_video: bool = False
    top_k: int = 25


class MatchResult(BaseModel):
    match_id: str
    video_id: str
    camera_id: str
    track_id: int
    frame_number: int
    timestamp_sec: float
    similarity: float
    bbox: BoundingBox
    thumbnail_base64: str


class SearchResponse(BaseModel):
    query_video_id: str
    query_bbox: BoundingBox
    query_thumbnail_base64: str
    matches: List[MatchResult]
    searched_videos: int
    candidates_scanned: int


class TrackSample(BaseModel):
    track_id: int
    frame_number: int
    timestamp_sec: float
    bbox: BoundingBox
    confidence: float


class TracksResponse(BaseModel):
    video_id: str
    fps: float
    sample_stride_hint: int
    samples: List[TrackSample]


class DetectedPerson(BaseModel):
    temp_id: int
    bbox: BoundingBox
    confidence: float


class DetectFrameResponse(BaseModel):
    video_id: str
    frame_number: int
    people: List[DetectedPerson]


class FootballPoint(BaseModel):
    x: float
    y: float


class FootballCalibrationRequest(BaseModel):
    points: List[FootballPoint]  # [near-left, far-left, far-right, near-right] in pixels
    pitch_length: float = 105.0
    pitch_width: float = 68.0


class FootballAnalyzeRequest(BaseModel):
    max_sampled_frames: int = 600
    stride: int = 5
