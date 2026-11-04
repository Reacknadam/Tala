"""Pure analytics over pitch-mapped tracks (meters, real fps).

All functions are deterministic and side-effect free so they can be unit
tested without video or models.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

Point = Tuple[float, float]
TrackSeries = Dict[int, List[Tuple[int, Optional[float], Optional[float]]]]

MAX_FRAME_GAP = 10  # never integrate distance across longer dropouts


def compute_player_stats(tracks: TrackSeries, fps: float) -> Dict[int, dict]:
    """Per-track distance (path length) and speeds from meter positions."""
    out: Dict[int, dict] = {}
    for track_id, series in tracks.items():
        ordered = sorted(series, key=lambda s: s[0])
        distance = 0.0
        valid_time = 0.0
        speeds: List[Tuple[int, float]] = []
        for (f0, x0, y0), (f1, x1, y1) in zip(ordered, ordered[1:]):
            if x0 is None or x1 is None or y0 is None or y1 is None:
                continue
            gap = f1 - f0
            if gap <= 0 or gap > MAX_FRAME_GAP or fps <= 0:
                continue
            seg = math.hypot(x1 - x0, y1 - y0)
            dt = gap / fps
            if seg / dt * 3.6 > 45.0:
                continue  # teleport glitch (ID swap / projection jump), not motion
            distance += seg
            valid_time += dt
            speeds.append((f1, seg / dt * 3.6))
        mean = distance / valid_time * 3.6 if valid_time > 0 else 0.0
        out[track_id] = {
            "distance_m": round(distance, 2),
            "mean_speed_kmh": round(mean, 2),
            "max_speed_kmh": round(max((s for _, s in speeds), default=0.0), 2),
            "speed_samples": speeds,
        }
    return out


def assign_teams(hue_samples: Dict[int, Sequence[float]]) -> Dict[int, int]:
    """Split tracks into two teams by mean jersey hue (degrees 0-360).

    Deterministic 2-means on the unit circle (cos/sin encoding); team 1 is
    the cluster whose centroid hue is smallest. No randomness, no sklearn.
    """
    ids = sorted(hue_samples)
    if len(ids) < 2:
        return {track_id: 1 for track_id in ids}

    def centroid_angle(samples: Sequence[float]) -> float:
        vals = [h for h in samples if h is not None]
        if not vals:
            return 0.0
        c = sum(math.cos(math.radians(h)) for h in vals) / len(vals)
        s = sum(math.sin(math.radians(h)) for h in vals) / len(vals)
        return math.degrees(math.atan2(s, c)) % 360.0

    feats = {i: centroid_angle(hue_samples[i]) for i in ids}
    # Init: the two most separated tracks.
    a, b = max(((i, j) for i in ids for j in ids if i < j),
               key=lambda p: _hue_distance(feats[p[0]], feats[p[1]]))
    centers = [feats[a], feats[b]]
    labels: Dict[int, int] = {}
    for _ in range(50):
        changed = False
        for i in ids:
            label = 0 if _hue_distance(feats[i], centers[0]) <= _hue_distance(feats[i], centers[1]) else 1
            if labels.get(i) != label:
                labels[i] = label
                changed = True
        for k in (0, 1):
            members = [feats[i] for i in ids if labels[i] == k]
            if members:
                c = sum(math.cos(math.radians(h)) for h in members) / len(members)
                s = sum(math.sin(math.radians(h)) for h in members) / len(members)
                centers[k] = math.degrees(math.atan2(s, c)) % 360.0
        if not changed:
            break
    team_of_center = {0: 1, 1: 2} if centers[0] <= centers[1] else {0: 2, 1: 1}
    return {i: team_of_center[labels[i]] for i in ids}


def _hue_distance(a: float, b: float) -> float:
    return abs((a - b + 180.0) % 360.0 - 180.0)


def possession_timeline(frames: List[dict], fps: float, hold_sec: float = 0.4, radius_m: float = 3.0) -> dict:
    """Ball owner team per frame with hysteresis.

    Each frame: ``{frame, ball: (x, y) | None, players: {id: (x, y, team)}}``
    in meters. The nearest player within ``radius_m`` is the candidate;
    ownership only changes after the same challenger team stays candidate
    for ``hold_sec`` consecutive seconds, which kills single-frame flicker.
    Shares are computed over frames that have an owner.
    """
    hold_frames = max(1, int(round(hold_sec * fps))) if fps > 0 else 1
    owners: List[Optional[int]] = []
    current: Optional[int] = None
    challenger: Optional[int] = None
    challenger_run = 0

    for item in frames:
        ball = item.get("ball")
        candidate: Optional[int] = None
        if ball is not None:
            best_dist, best_team = radius_m, None
            for _pid, (px, py, team) in item.get("players", {}).items():
                dist = math.hypot(px - ball[0], py - ball[1])
                if dist < best_dist:
                    best_dist, best_team = dist, team
            candidate = best_team

        if candidate is None or candidate == current:
            challenger, challenger_run = None, 0
            if candidate is not None:
                current = candidate
        else:
            if candidate == challenger:
                challenger_run += 1
            else:
                challenger, challenger_run = candidate, 1
            if challenger_run >= hold_frames:
                current, challenger, challenger_run = candidate, None, 0
        owners.append(current)

    owned = [o for o in owners if o is not None]
    total = len(owned)
    shares = {
        team: round(100.0 * owned.count(team) / total, 1) if total else 0.0
        for team in (1, 2)
    }
    segments: List[dict] = []
    for item, owner in zip(frames, owners):
        if owner is None:
            continue
        if segments and segments[-1]["team"] == owner:
            segments[-1]["end_frame"] = item["frame"]
        else:
            segments.append({"team": owner, "start_frame": item["frame"], "end_frame": item["frame"]})
    return {"owners": owners, "shares": shares, "segments": segments}
