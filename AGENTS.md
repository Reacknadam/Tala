# AGENTS.md — working notes for coding agents

## Layout

- `backend/` — FastAPI app (`app/`), unit tests (`tests/`), model weights (`models/`), runtime data (`data/`, gitignored)
- `frontend/` — React 18 + TypeScript + Vite
- `videos/` — uploaded footage, gitignored
- `scripts/` — setup / dev / model-download helpers

## Commands

- Backend dev server: `cd backend && uvicorn app.main:app --reload --port 8000`
- Backend tests: `cd backend && python -m unittest discover -s tests -v`
- Frontend dev server: `cd frontend && npm run dev` (http://localhost:5173, proxies `/api` → :8000)
- Frontend build / typecheck: `cd frontend && npm run build`
- One-shot dev launch: `start.bat` (repo root, Windows double-click) or `scripts/dev.ps1` / `scripts/dev.sh`

## Conventions

- Backend tests are `unittest`-based and must stay runnable without video files or model weights (pure functions / mocks only — the heavy pipeline is exercised manually).
- Paths are derived from `__file__` (`storage.DATA_DIR`, `video.VIDEOS_DIR`, `detector.MODEL_DIR`); keep them relative to the repo, never absolute.
- Embeddings are stored as raw float32 blobs in SQLite and scanned with a NumPy matmul — do not add a vector DB.
