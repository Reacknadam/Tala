"""Football analysis orchestration: video -> pitch-mapped intelligence.

Persists calibration and results as JSON sidecars under
``backend/data/football/`` so the SQLite person-search schema stays intact.
Analysis is synchronous and capped (sampled frames), which keeps it honest
and testable; long matches can raise the caps explicitly.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from app.football.analytics import assign_teams, compute_player_stats, possession_timeline
from app.football.camera import CameraCompensator
from app.football.pitch import PitchMapping
from app.football.tracker import FootballTracker, dominant_hue
from app.services import storage

FOOTBALL_DIR = storage.DATA_DIR / "football"


def _path(video_id: str, kind: str) -> Path:
    FOOTBALL_DIR.mkdir(parents=True, exist_ok=True)
    return FOOTBALL_DIR / f"{video_id}_{kind}.json"


def save_calibration(video_id: str, points: list, length: float, width: float) -> dict:
    data = {"video_id": video_id, "points": points, "pitch_length": length, "pitch_width": width}
    _path(video_id, "calibration").write_text(json.dumps(data), encoding="utf-8")
    return data


def load_calibration(video_id: str) -> Optional[dict]:
    path = _path(video_id, "calibration")
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def load_analysis(video_id: str) -> Optional[dict]:
    path = _path(video_id, "analysis")
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def run_analysis(video_id: str, max_sampled_frames: int = 600, stride: int = 5) -> dict:
    row = storage.get_video(video_id)
    if row is None:
        raise LookupError("video not found")
    calibration = load_calibration(video_id)
    if calibration is None:
        raise ValueError("pitch calibration required before analysis")
    pitch = PitchMapping(calibration["points"], calibration["pitch_length"], calibration["pitch_width"])
    fps = float(row["fps"]) or 25.0
    stride = max(1, min(stride, 10))

    cap = cv2.VideoCapture(str(Path(row["path"])))
    if not cap.isOpened():
        raise ValueError("could not open video file")
    tracker = FootballTracker()
    compensator = CameraCompensator()
    tracks_m: dict = {}
    hues: dict = {}
    sampled: list = []  # UI frames: {frame, players_m, players_px, ball_cur}
    ball_series: list = []  # (frame_idx, reference-space ball center | None), every frame
    frame_idx, sampled_count, truncated = 0, 0, False
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Every frame: the broadcast camera pans/zooms, so positions are
            # first warped back into the calibration (first-frame) space.
            compensator.update(frame)
            # Continuous tracking: ByteTrack needs every frame to keep IDs
            # stable; only the UI output is sub-sampled (stride).
            players, ball = tracker.process_frame(frame)
            ball_series.append((frame_idx, compensator.transform(ball["x"], ball["y"]) if ball else None))
            is_sample = frame_idx % stride == 0
            players_m: dict = {}
            players_px: dict = {}
            for player in players:
                mapped = pitch.to_pitch(*compensator.transform(*player["foot"]))
                tracks_m.setdefault(player["track_id"], []).append(
                    (frame_idx, mapped[0] if mapped else None, mapped[1] if mapped else None))
                hue = dominant_hue(frame, player["bbox"])
                if hue is not None and len(hues.setdefault(player["track_id"], [])) < 30:
                    hues[player["track_id"]].append(hue)
                if is_sample and mapped is not None:
                    players_m[player["track_id"]] = mapped
                    players_px[player["track_id"]] = {"bbox": player["bbox"], "foot": player["foot"]}
            if is_sample:
                sampled.append({
                    "frame": frame_idx, "players_m": players_m, "players_px": players_px,
                    "ball_cur": ({"x": ball["x"], "y": ball["y"], "bbox": ball["bbox"]} if ball else None),
                })
                sampled_count += 1
                if sampled_count >= max_sampled_frames:
                    truncated = True
                    break
            frame_idx += 1
    finally:
        cap.release()

    # Ball: interpolate in reference space between real detections only
    # (no extrapolation past the first/last sighting), then map to meters.
    known = {f: p for f, p in ball_series if p is not None}
    known_frames = sorted(known)
    for item in sampled:
        f = item["frame"]
        point = known.get(f)
        if point is None and len(known_frames) >= 2 and known_frames[0] < f < known_frames[-1]:
            xs = np.interp(f, known_frames, [known[k][0] for k in known_frames])
            ys = np.interp(f, known_frames, [known[k][1] for k in known_frames])
            point = (float(xs), float(ys))
        item["ball_m"] = pitch.to_pitch(*point) if point is not None else None

    # Median-filter meter positions (window 5) per track: projection jitter
    # otherwise dominates path length at full frame rate.
    for tid, series in tracks_m.items():
        pts = [(f, x, y) for f, x, y in series if x is not None and y is not None]
        if len(pts) >= 5:
            smoothed = {}
            for k, (f, _x, _y) in enumerate(pts):
                win = pts[max(0, k - 2):k + 3]
                smoothed[f] = (float(np.median([w[1] for w in win])), float(np.median([w[2] for w in win])))
            tracks_m[tid] = [
                (f, smoothed[f][0], smoothed[f][1]) if f in smoothed else (f, x, y)
                for f, x, y in series
            ]

    assigned = assign_teams(hues)
    teams = {tid: (assigned.get(tid, 0) if len(hues.get(tid, [])) >= 3 else 0) for tid in tracks_m}
    stats = compute_player_stats(tracks_m, fps)

    def _smoothed(samples: list) -> dict:
        # ~0.2 s moving average: per-frame speeds are dominated by jitter.
        out = {}
        vals = [v for _, v in samples]
        for i, (frame_no, _v) in enumerate(samples):
            window = vals[max(0, i - 2):i + 3]
            out[frame_no] = sum(window) / len(window)
        return out

    speed_at = {tid: _smoothed(s["speed_samples"]) for tid, s in stats.items()}
    poss_frames = [{
        "frame": s["frame"],
        "ball": s["ball_m"],
        "players": {tid: (xy[0], xy[1], teams[tid]) for tid, xy in s["players_m"].items() if teams[tid] in (1, 2)},
    } for s in sampled]
    possession = possession_timeline(poss_frames, fps / stride)

    events = []
    for tid, s in stats.items():
        run_start, prev_frame = None, None
        for frame_no, kmh in s["speed_samples"] + [(None, 0.0)]:
            fast = kmh >= 25.0
            if fast and run_start is None:
                run_start = frame_no
            if not fast and run_start is not None:
                if (prev_frame - run_start) / fps >= 0.8:
                    events.append({"type": "high_speed_run", "track_id": tid,
                                   "start_frame": run_start, "end_frame": prev_frame})
                run_start = None
            if fast:
                prev_frame = frame_no

    owners = possession["owners"]

    def _owner_player(item: dict) -> Optional[int]:
        if item["ball_m"] is None:
            return None
        best_id, best_dist = None, 3.0
        for tid, xy in item["players_m"].items():
            dist = math.hypot(xy[0] - item["ball_m"][0], xy[1] - item["ball_m"][1])
            if dist < best_dist:
                best_id, best_dist = tid, dist
        return best_id

    result = {
        "video_id": video_id, "fps": fps, "stride": stride, "truncated": truncated,
        "pitch": {"length": calibration["pitch_length"], "width": calibration["pitch_width"]},
        "frames_sampled": sampled_count,
        "players": [{"track_id": tid, "team": teams[tid], "distance_m": s["distance_m"],
                     "mean_speed_kmh": s["mean_speed_kmh"], "max_speed_kmh": s["max_speed_kmh"]}
                    for tid, s in sorted(stats.items())],
        "possession": {"shares": {str(k): v for k, v in possession["shares"].items()},
                       "segments": possession["segments"]},
        "events": events,
        "frames": [{
            "frame": s["frame"], "t_sec": round(s["frame"] / fps, 2),
            "players": [{"id": tid, "team": teams[tid], "x": round(xy[0], 2), "y": round(xy[1], 2),
                         "speed_kmh": round(speed_at.get(tid, {}).get(s["frame"], 0.0), 1),
                         "bbox_px": [round(v, 1) for v in s["players_px"][tid]["bbox"]],
                         "foot_px": [round(v, 1) for v in s["players_px"][tid]["foot"]]}
                        for tid, xy in sorted(s["players_m"].items())],
            "ball": ({"x": round(s["ball_m"][0], 2), "y": round(s["ball_m"][1], 2),
                      "px": [round(s["ball_cur"]["x"], 1), round(s["ball_cur"]["y"], 1)] if s["ball_cur"] else None,
                      "bbox_px": [round(v, 1) for v in s["ball_cur"]["bbox"]] if s["ball_cur"] and s["ball_cur"]["bbox"] else None}
                     if s["ball_m"] else None),
            "owner_team": owners[i],
            "owner_player_id": _owner_player(s),
        } for i, s in enumerate(sampled)],
    }
    _path(video_id, "analysis").write_text(json.dumps(result), encoding="utf-8")
    return result
