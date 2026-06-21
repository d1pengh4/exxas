from contextlib import asynccontextmanager
from fastapi import FastAPI
from .core import ssl_patch  # noqa: F401  macOS SSL fix
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from .core.database import init_db, close_db
from .core.config import settings
from .core.rate_limit import RateLimitMiddleware
from .api.v1 import analyze, auth


@asynccontextmanager
async def lifespan(app: FastAPI):
    active_model = settings.OLLAMA_MODEL if settings.LLM_PROVIDER == "ollama" else settings.LLM_MODEL
    logger.info(
        f"EXXAS v2.0 starting — env={settings.ENVIRONMENT} "
        f"llm={settings.LLM_PROVIDER}/{active_model}"
    )
    await init_db()
    logger.info("All databases initialized")
    yield
    await close_db()
    logger.info("EXXAS shutdown")


app = FastAPI(
    title="EXXAS API",
    description="Image-Based Intelligent Geolocalization & OSINT Investigation Platform v2.0",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3001",
        "https://exxas.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(RateLimitMiddleware, redis_url=settings.REDIS_URL)

app.include_router(auth.router, prefix="/api/v1/auth", tags=["auth"])
app.include_router(analyze.router, prefix="/api/v1", tags=["analyze"])


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "version": "2.0.0",
        "llm_provider": settings.LLM_PROVIDER,
        "llm_model": settings.LLM_MODEL,
        "environment": settings.ENVIRONMENT,
    }
