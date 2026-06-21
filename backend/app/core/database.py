from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase
from neo4j import AsyncGraphDatabase
from pymilvus import connections as milvus_conn, utility as milvus_util
import redis.asyncio as aioredis
from loguru import logger
from .config import settings


# ── SQLAlchemy ────────────────────────────────────────────
engine = create_async_engine(
    settings.DATABASE_URL,
    echo=settings.ENVIRONMENT == "development",
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,
)

AsyncSessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


async def get_db():
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()


# ── Redis ─────────────────────────────────────────────────
_redis: aioredis.Redis | None = None
_raw_redis: aioredis.Redis | None = None


async def get_redis() -> aioredis.Redis:
    """문자열 Redis (decode_responses=True) — 상태값/JSON 저장용"""
    global _redis
    if _redis is None:
        _redis = aioredis.from_url(settings.REDIS_URL, encoding="utf-8", decode_responses=True)
    return _redis


async def get_raw_redis() -> aioredis.Redis:
    """바이너리 Redis (decode_responses=False) — 이미지 등 바이트 저장용"""
    global _raw_redis
    if _raw_redis is None:
        _raw_redis = aioredis.from_url(settings.REDIS_URL, decode_responses=False)
    return _raw_redis


# ── Milvus ────────────────────────────────────────────────
def _connect_milvus():
    try:
        milvus_conn.connect(
            alias="default",
            host=settings.MILVUS_HOST,
            port=settings.MILVUS_PORT,
            timeout=3,          # 연결 시도 최대 3초
        )
        logger.info("Milvus connected")
        _ensure_milvus_collections()
    except Exception as e:
        logger.warning(f"Milvus not available: {e}")


def _ensure_milvus_collections():
    from pymilvus import Collection, FieldSchema, CollectionSchema, DataType
    for name, fields in _MILVUS_SCHEMAS.items():
        if not milvus_util.has_collection(name):
            col = Collection(name=name, schema=CollectionSchema(fields))
            col.create_index("embedding", {
                "metric_type": "COSINE", "index_type": "IVF_FLAT", "params": {"nlist": 1024}
            })
            logger.info(f"Milvus collection created: {name}")


_MILVUS_SCHEMAS = {}


def _build_milvus_schemas():
    from pymilvus import FieldSchema, DataType
    _MILVUS_SCHEMAS["image_embeddings"] = [
        FieldSchema("id", DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema("image_hash", DataType.VARCHAR, max_length=64),
        FieldSchema("latitude", DataType.FLOAT),
        FieldSchema("longitude", DataType.FLOAT),
        FieldSchema("location", DataType.VARCHAR, max_length=128),
        FieldSchema("embedding", DataType.FLOAT_VECTOR, dim=768),  # fine-tuned CLIP projection_dim
    ]


# ── Neo4j ─────────────────────────────────────────────────
_neo4j_driver = None
_neo4j_unavailable = False   # 첫 연결 실패 후 True → 이후 호출 즉시 None 반환


def get_neo4j_driver():
    global _neo4j_driver, _neo4j_unavailable
    if _neo4j_unavailable:
        return None
    if _neo4j_driver is None:
        try:
            _neo4j_driver = AsyncGraphDatabase.driver(
                settings.NEO4J_URI,
                auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD),
                connection_timeout=3,
            )
            logger.info("Neo4j driver created (연결 테스트는 첫 쿼리 시)")
        except Exception as e:
            _neo4j_unavailable = True
            logger.debug(f"Neo4j 미실행 — 지식 그래프 기능 비활성화: {e}")
    return _neo4j_driver


# ── 초기화 / 종료 ─────────────────────────────────────────
async def init_db():
    # 모델 임포트 (테이블 생성 전에 필요)
    from ..models import User, Analysis  # noqa: F401

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("PostgreSQL tables created/verified")

    try:
        _build_milvus_schemas()
        _connect_milvus()
    except Exception as e:
        logger.warning(f"Milvus init skipped: {e}")

    get_neo4j_driver()


async def close_db():
    global _redis, _raw_redis, _neo4j_driver
    await engine.dispose()
    if _redis:
        await _redis.aclose()
        _redis = None
    if _raw_redis:
        await _raw_redis.aclose()
        _raw_redis = None
    if _neo4j_driver:
        await _neo4j_driver.close()
        _neo4j_driver = None
