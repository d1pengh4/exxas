from pydantic_settings import BaseSettings
from pydantic import field_validator
from typing import Literal
import os


class Settings(BaseSettings):
    # ── 환경 ──────────────────────────────────────────────
    ENVIRONMENT: Literal["development", "production"] = "development"
    LOG_LEVEL: str = "INFO"

    # ── LLM Provider ──────────────────────────────────────
    LLM_PROVIDER: Literal["claude", "ollama", "openai", "groq"] = "claude"
    LLM_MODEL: str = "claude-sonnet-4-6"
    ANTHROPIC_API_KEY: str = ""
    OPENAI_API_KEY: str = ""
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "qwen2.5vl:7b"
    GROQ_API_KEY: str = ""
    GROQ_MODEL: str = "meta-llama/llama-4-scout-17b-16e-instruct"

    # ── Database ──────────────────────────────────────────
    DATABASE_URL: str = "postgresql+asyncpg://exxas:exxas_dev@localhost:5432/exxas"
    REDIS_URL: str = "redis://localhost:6379/0"

    # ── Milvus ────────────────────────────────────────────
    MILVUS_HOST: str = "localhost"
    MILVUS_PORT: int = 19530

    # ── Neo4j ─────────────────────────────────────────────
    NEO4J_URI: str = "bolt://localhost:7687"
    NEO4J_USER: str = "neo4j"
    NEO4J_PASSWORD: str = "exxas_dev"

    # ── 지도 API ──────────────────────────────────────────
    NAVER_CLIENT_ID: str = ""
    NAVER_CLIENT_SECRET: str = ""
    KAKAO_API_KEY: str = ""
    KAKAO_ACCESS_TOKEN: str = ""   # OAuth Bearer 토큰 (KakaoAK 403 시 폴백)
    GOOGLE_MAPS_API_KEY: str = ""

    # ── OSINT API ─────────────────────────────────────────
    FLICKR_API_KEY: str = ""
    SERP_API_KEY: str = ""
    SERPAPI_KEY: str = ""          # SerpAPI (web_search 도구용)
    BING_SEARCH_API_KEY: str = ""
    BRAVE_SEARCH_API_KEY: str = ""  # Brave Search API (DDG 차단 시 폴백)
    MAPILLARY_TOKEN: str = ""      # Mapillary API (street_view_fetch 도구용)
    HF_TOKEN: str = ""             # HuggingFace token (Llama-3.2-Vision inference)

    # ── 한국 전용 API ──────────────────────────────────────
    JUSO_API_KEY: str = ""         # 행정안전부 도로명주소 API
    PUBLIC_DATA_API_KEY: str = ""  # 공공데이터포털 API (사업자등록/문화재청 등)

    # ── CLOVA OCR (E 업그레이드) ───────────────────────────
    CLOVA_OCR_API_KEY: str = ""    # 네이버 CLOVA OCR Secret Key
    CLOVA_OCR_API_URL: str = ""    # CLOVA OCR Invoke URL

    # ── 위성/기상 ─────────────────────────────────────────
    NASA_API_KEY: str = "DEMO_KEY"
    OPEN_METEO_API_URL: str = "https://api.open-meteo.com/v1"

    # ── 보안 ──────────────────────────────────────────────
    SECRET_KEY: str = "dev-secret-key-change-in-production"
    ADMIN_SECRET_KEY: str = "admin-secret-key-change-in-production"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 1440

    # ── 스토리지 ──────────────────────────────────────────
    UPLOAD_DIR: str = "./uploads"
    MAX_UPLOAD_SIZE_MB: int = 50

    # ── 분석 설정 ─────────────────────────────────────────
    MAX_INVESTIGATION_STEPS: int = 8
    CONFIDENCE_THRESHOLD_HIGH: float = 0.95
    CONFIDENCE_THRESHOLD_MEDIUM: float = 0.70
    CONFIDENCE_THRESHOLD_LOW: float = 0.30

    @field_validator("UPLOAD_DIR")
    @classmethod
    def create_upload_dir(cls, v: str) -> str:
        os.makedirs(v, exist_ok=True)
        return v

    model_config = {"env_file": ".env", "case_sensitive": True, "extra": "ignore"}


settings = Settings()
