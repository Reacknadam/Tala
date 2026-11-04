"""Local SQLite storage for indexed video metadata and person embeddings.

No vector database, no ORM -- a hackathon MVP with a few thousand rows of
embeddings does not need one. Embeddings are stored as raw float32 bytes and
loaded into a NumPy array at search time for a plain cosine-similarity scan.
"""
from __future__ import annotations

import sqlite3
import threading
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = DATA_DIR / "tala.db"

_lock = threading.Lock()

SCHEMA = """
CREATE TABLE IF NOT EXISTS videos (
    video_id TEXT PRIMARY KEY,
    camera_id TEXT NOT NULL,
    filename TEXT NOT NULL,
    path TEXT NOT NULL,
    fps REAL NOT NULL,
    width INTEGER NOT NULL,
    height INTEGER NOT NULL,
    frame_count INTEGER NOT NULL,
    duration_sec REAL NOT NULL,
    status TEXT NOT NULL DEFAULT 'unindexed',
    error TEXT
);

CREATE TABLE IF NOT EXISTS detections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id TEXT NOT NULL,
    camera_id TEXT NOT NULL,
    track_id INTEGER NOT NULL,
    frame_number INTEGER NOT NULL,
    timestamp_sec REAL NOT NULL,
    x1 REAL NOT NULL, y1 REAL NOT NULL, x2 REAL NOT NULL, y2 REAL NOT NULL,
    confidence REAL NOT NULL DEFAULT 0.0,
    embedding BLOB NOT NULL,
    thumbnail BLOB NOT NULL,
    FOREIGN KEY(video_id) REFERENCES videos(video_id)
);

CREATE INDEX IF NOT EXISTS idx_detections_video ON detections(video_id);
"""


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with _lock:
        conn = get_conn()
        try:
            conn.executescript(SCHEMA)
            conn.commit()
        finally:
            conn.close()


def upsert_video(video_id, camera_id, filename, path, fps, width, height, frame_count, duration_sec, status="unindexed"):
    with _lock:
        conn = get_conn()
        try:
            conn.execute(
                """INSERT INTO videos (video_id, camera_id, filename, path, fps, width, height, frame_count, duration_sec, status)
                   VALUES (?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(video_id) DO UPDATE SET status=excluded.status""",
                (video_id, camera_id, filename, path, fps, width, height, frame_count, duration_sec, status),
            )
            conn.commit()
        finally:
            conn.close()


def set_video_status(video_id: str, status: str, error: Optional[str] = None) -> None:
    with _lock:
        conn = get_conn()
        try:
            conn.execute("UPDATE videos SET status=?, error=? WHERE video_id=?", (status, error, video_id))
            conn.commit()
        finally:
            conn.close()


def list_videos() -> List[sqlite3.Row]:
    conn = get_conn()
    try:
        return conn.execute("SELECT * FROM videos ORDER BY camera_id").fetchall()
    finally:
        conn.close()


def get_video(video_id: str) -> Optional[sqlite3.Row]:
    conn = get_conn()
    try:
        return conn.execute("SELECT * FROM videos WHERE video_id=?", (video_id,)).fetchone()
    finally:
        conn.close()


def clear_detections_for_video(video_id: str) -> None:
    with _lock:
        conn = get_conn()
        try:
            conn.execute("DELETE FROM detections WHERE video_id=?", (video_id,))
            conn.commit()
        finally:
            conn.close()


def insert_detection(video_id, camera_id, track_id, frame_number, timestamp_sec, bbox, confidence: float, embedding: np.ndarray, thumbnail_bytes: bytes) -> None:
    x1, y1, x2, y2 = (float(v) for v in bbox)
    with _lock:
        conn = get_conn()
        try:
            conn.execute(
                """INSERT INTO detections (video_id, camera_id, track_id, frame_number, timestamp_sec, x1, y1, x2, y2, confidence, embedding, thumbnail)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                (video_id, camera_id, track_id, frame_number, timestamp_sec, x1, y1, x2, y2, float(confidence),
                 embedding.astype(np.float32).tobytes(), thumbnail_bytes),
            )
            conn.commit()
        finally:
            conn.close()


def fetch_track_samples(video_id: str) -> List[sqlite3.Row]:
    """Positions légères (pas d'embedding/thumbnail) de toutes les détections 
    stockées pour cette vidéo, triées par track puis frame. Sert à l'overlay 
    de tracking live : le frontend interpole entre échantillons consécutifs 
    du même track pour suivre une personne fluidement pendant la lecture, 
    au lieu d'un cadre qui ne bouge qu'une fois toutes les sample_every_n_frames."""
    conn = get_conn()
    try:
        return conn.execute(
            """SELECT track_id, frame_number, timestamp_sec, x1, y1, x2, y2, confidence
               FROM detections WHERE video_id=? ORDER BY track_id, frame_number""",
            (video_id,),
        ).fetchall()
    finally:
        conn.close()


def count_detections(video_id: str) -> Tuple[int, int]:
    """Returns (detection_count, distinct_track_count)."""
    conn = get_conn()
    try:
        row = conn.execute(
            "SELECT COUNT(*) as n, COUNT(DISTINCT track_id) as t FROM detections WHERE video_id=?", (video_id,)
        ).fetchone()
        return row["n"], row["t"]
    finally:
        conn.close()


def fetch_candidate_detections(exclude_video_id: Optional[str] = None) -> List[sqlite3.Row]:
    """All stored detections available for search, optionally excluding one video."""
    conn = get_conn()
    try:
        if exclude_video_id:
            return conn.execute("SELECT * FROM detections WHERE video_id != ?", (exclude_video_id,)).fetchall()
        return conn.execute("SELECT * FROM detections").fetchall()
    finally:
        conn.close()


def get_detection(detection_id: int) -> Optional[sqlite3.Row]:
    conn = get_conn()
    try:
        return conn.execute("SELECT * FROM detections WHERE id=?", (detection_id,)).fetchone()
    finally:
        conn.close()
