# TALA — Video Intelligence

**Local-first video intelligence: find a person across all your cameras with one click, and turn a football broadcast into pitch-mapped analytics.**

TALA ingests footage from multiple cameras, detects and tracks every person with YOLO11 + ByteTrack, and builds a visual embedding per tracked appearance. Click on someone in one camera — TALA instantly returns every appearance of that person across the other feeds, with thumbnails, timestamps and one-click seek. A second mode turns a football clip into real analytics: pitch calibration, player/ball tracking in meters, per-player speed & distance, team assignment and possession — all running **100% locally on CPU**, no cloud, no GPU required.

Built for a hackathon: honest engineering over buzzwords. Design decisions below are measured, not guessed.

---

## The two modes

### 1. Search — cross-camera person re-identification

- Upload footage, tag it with a `camera_id`, hit **Index**
- Every frame runs through YOLO11n + ByteTrack; sampled detections get a hybrid appearance embedding (banded HSV histogram + YOLO11n-cls deep feature) stored as float32 blobs in SQLite
- Enable **Search Mode**, draw a box or click a detected person → cosine similarity scan across all indexed detections → ranked matches
- Click a match → instant seek to that moment in the other camera's footage, with a live bbox overlay interpolated between sampled frames

### 2. Football — pitch-mapped analytics

- Click the 4 pitch corners once (calibration is per-video, in meters — default FIFA 105×68)
- Analysis warps every frame back into calibration space via Lucas-Kanade feature tracking + RANSAC homography (handles pan/tilt/zoom), maps player feet and ball to meters
- Output: 2D pitch replay overlay, per-player distance & speed (median-smoothed, teleport-filtered), deterministic team assignment by jersey hue, possession timeline with hysteresis, sprint events
- Uses a fine-tuned detector (`football_best.pt`, ball/player/goalkeeper/referee) when present; falls back to COCO weights

---

## Architecture

```
┌──────────────┐        ┌────────────────────────────────────────────┐
│  React + Vite │  HTTP  │  FastAPI backend (app/)                    │
│  frontend/    │ ─────► │                                            │
│  SPA :5173    │  /api  │  api/      videos · search · football      │
└──────────────┘        │  vision/   detector (YOLO+ByteTrack)        │
                        │            reid     (hybrid embedding)      │
                        │            search   (cosine scan, NumPy)    │
                        │  football/ pitch    (homography, meters)    │
                        │            camera   (LK + RANSAC compens.)  │
                        │            tracker · analytics · service    │
                        │  services/ video · indexing · storage       │
                        └──────┬───────────────────────┬──────────────┘
                               ▼                       ▼
                     videos/ (raw mp4)         backend/data/
                                               tala.db (SQLite:
                                               embeddings+thumbs)
                                               football/*.json
```

**Stack:** Python 3.11+ · FastAPI · OpenCV · PyTorch (CPU) · Ultralytics YOLO11 · SQLite (no ORM, no vector DB — a NumPy matmul over a few thousand embeddings is <10 ms) · React 18 · TypeScript · Vite

---

## Quick start

**Requirements:** Python ≥ 3.11, Node ≥ 18. Everything runs on CPU.

### Windows (one shot)

```powershell
scripts\setup.ps1   # venv + pip install + npm install + model weights
start.bat           # double-click: backend :8000 + frontend :5173 + opens the app
```

### Manual (any OS)

```bash
# Backend
cd backend
python -m venv .venv && source .venv/bin/activate   # .venv\Scripts\activate on Windows
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000           # API docs: http://127.0.0.1:8000/docs

# Frontend (second terminal)
cd frontend
npm install
npm run dev                                        # http://localhost:5173
```

> Have a CUDA GPU? Install torch/torchvision from [pytorch.org](https://pytorch.org/get-started/locally/) **before** `requirements.txt`.

## Model weights

| File | Size | Needed for | How to get |
|---|---|---|---|
| `backend/models/yolo11n.pt` | ~5 MB | person detection/tracking | auto-downloaded by `scripts/download_models.py` |
| `backend/models/yolo11n-cls.pt` | ~5 MB | re-id deep feature | same |
| `backend/models/football_best.pt` | ~165 MB | Football mode (quality) | `python scripts/download_models.py` (gdown) or [manual download](https://drive.google.com/file/d/1DC2kCygbBWUKheQ_9cFziCsYVSRw6axK/view) → place in `backend/models/` |

`football_best.pt` is gitignored (over GitHub's file limit). Without it, Football mode still runs on COCO weights with weaker ball detection.

## Demo walkthrough

1. `npm run dev` → open http://localhost:5173
2. **Search mode:** upload 2+ videos with different `camera_id`s → Index each → toggle *Search Mode* → pause on a frame, click a detected person (or drag a box) → *Search* → click any match to teleport to that moment in the other camera.
3. **Football mode:** open a broadcast clip → *Calibrate* → click the 4 pitch corners (near-left, far-left, far-right, near-right) → *Analyze* → watch the 2D pitch replay with per-player speed, distance, teams and possession.
   - Sample broadcast clip: [public sample video](https://drive.google.com/file/d/1t6agoqggZKx6thamUuPAIdN_1zR9v9S_/view) (the `08fd33_4` footage).

## API overview

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/api/health` | liveness |
| `GET/POST` | `/api/videos` | list / upload (`camera_id` + file) |
| `POST` | `/api/videos/{id}/index` | start background indexing |
| `GET` | `/api/videos/{id}/index/status` | indexing progress |
| `GET` | `/api/videos/{id}/stream` | range-aware video streaming |
| `GET` | `/api/videos/{id}/frame?frame_number=` | single frame as JPEG base64 |
| `GET` | `/api/videos/{id}/tracks` | all track samples (live overlay) |
| `GET` | `/api/videos/{id}/detect_frame` | one-shot person detection on a frame |
| `POST` | `/api/search` | cross-camera re-id search |
| `GET` | `/api/matches/{match_id}` | re-fetch one detection |
| `PUT/GET` | `/api/football/{id}/calibration` | save / load 4-point pitch calibration |
| `POST` | `/api/football/{id}/analyze` | run analysis (`max_sampled_frames`, `stride`) |
| `GET` | `/api/football/{id}/analysis` | stored analysis result |

Full schemas: interactive Swagger at `/docs` when the backend runs.

## Project structure

```
tala/
├── backend/
│   ├── app/
│   │   ├── api/        # HTTP layer (videos, search, football, health)
│   │   ├── football/   # pitch mapping, camera compensation, tracking, analytics
│   │   ├── models/     # pydantic schemas
│   │   ├── services/   # storage (SQLite), indexing pipeline, video helpers
│   │   └── vision/     # YOLO detector, re-id extractor, similarity search
│   ├── models/         # *.pt weights (football_best.pt is downloaded, not committed)
│   ├── data/           # gitignored runtime data (tala.db, football/*.json)
│   ├── tests/          # unittest suite — no video/models needed
│   └── requirements.txt
├── frontend/           # React 18 + TS + Vite
│   └── src/{components,services}/
├── scripts/            # setup.ps1 · dev.ps1/.sh · download_models.py
├── videos/             # gitignored upload dir
└── README.md · LICENSE · AGENTS.md
```

## Testing

```bash
cd backend && python -m unittest discover -s tests -v
```

Pure-function unit tests (homography, camera compensation, team assignment, stats, empty-index regression) — no video files or weights required.

## Design decisions & known limitations

- **Re-id embedding.** We measured a YOLO11n-cls ImageNet feature alone on our demo footage: mean same-person similarity **0.96** vs different-person **0.95** — no separation (an ImageNet classifier isn't trained to tell people apart). Banded HSV color histograms did much better (same ≈0.67 / different ≈0.48), so the production descriptor is **0.7·color + 0.3·deep**, L2-normalized. A production system should swap in a dedicated re-id network (e.g. OSNet).
- **No vector DB.** Thousands of float32 vectors → one NumPy matmul, <10 ms. Correct scope for the problem size.
- **Indexing tracks every frame** (ByteTrack needs continuity) but only *stores* every N frames; the UI interpolates between samples.
- **Camera compensation** estimates a full prev→ref homography (LK + RANSAC), not just translation — one outlier feature can't dictate the result.
- **Known limits:** whole-body appearance ≠ identity (no face recognition — by design); uniform-looking teams degrade color re-id; pitch calibration assumes a planar view region; football analysis is synchronous and capped (`max_sampled_frames`).

## Credits

Football mode was inspired by the public [`football_analysis`](https://github.com/abdullahtarek/football_analysis) tutorial project (MIT). Its fine-tuned detector weights are redistributed via the link above; the analytics pipeline here is a rewrite that fixes its core defects (per-video calibration, real fps, meter-space possession with hysteresis, deterministic team assignment).

## License

MIT — see [LICENSE](LICENSE).
