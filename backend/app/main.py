from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import football, health, search, videos
from app.services import storage


@asynccontextmanager
async def lifespan(app: FastAPI):
    storage.init_db()
    yield


app = FastAPI(
    title="TALA",
    description="Local-first cross-camera person search and football pitch intelligence",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],  # Vite dev server (the proxy makes requests same-origin anyway)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(health.router, prefix="/api")
app.include_router(videos.router, prefix="/api")
app.include_router(search.router, prefix="/api")
app.include_router(football.router, prefix="/api")
