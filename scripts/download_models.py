"""Fetch the model weights TALA needs into backend/models/.

- yolo11n.pt / yolo11n-cls.pt: small (~5 MB each), downloaded automatically
  through the ultralytics package.
- football_best.pt: ~165 MB fine-tuned detector (ball/player/goalkeeper/
  referee classes) hosted on Google Drive -- too large for the git repo.
  Downloaded via `gdown` when available, otherwise printed as a manual step.

Usage:  python scripts/download_models.py
"""
from __future__ import annotations

import sys
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parents[1] / "backend" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

FOOTBALL_GDRIVE_ID = "1DC2kCygbBWUKheQ_9cFziCsYVSRw6axK"
FOOTBALL_GDRIVE_URL = f"https://drive.google.com/file/d/{FOOTBALL_GDRIVE_ID}/view"


def _fetch_small(name: str) -> None:
    dest = MODEL_DIR / name
    if dest.exists():
        print(f"  [ok] {name} already present")
        return
    # Ultralytics resolves a bare filename by downloading it, then loads the
    # weights; we simply move the downloaded file into models/.
    from ultralytics import YOLO

    print(f"  [..] downloading {name} via ultralytics")
    model = YOLO(name)
    src = Path(model.ckpt_path or name)
    if src.exists() and src.resolve() != dest.resolve():
        src.replace(dest)
    print(f"  [ok] {name} -> {dest}")


def _fetch_football() -> None:
    dest = MODEL_DIR / "football_best.pt"
    if dest.exists():
        print("  [ok] football_best.pt already present")
        return
    try:
        import gdown
    except ImportError:
        print("  [!!] football_best.pt missing and `gdown` is not installed.")
        print("       Install it (`pip install gdown`) and re-run, or download")
        print(f"       manually: {FOOTBALL_GDRIVE_URL}")
        print(f"       then place the file at: {dest}")
        return
    print("  [..] downloading football_best.pt (~165 MB) via gdown")
    gdown.download(id=FOOTBALL_GDRIVE_ID, output=str(dest), quiet=False)


if __name__ == "__main__":
    print(f"Model dir: {MODEL_DIR}")
    _fetch_small("yolo11n.pt")
    _fetch_small("yolo11n-cls.pt")
    _fetch_football()
    print("Done. (football_best.pt is optional: without it, Football mode")
    print("falls back to generic COCO weights with reduced ball detection.)")
