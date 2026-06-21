"""
Neo4j 지식 그래프
이미지 → 위치 → 랜드마크 → POI 연결망 자동 생성
분석 누적 → 그래프 촘촘 → 정확도 향상
"""
from typing import Optional
from loguru import logger
from ..core.database import get_neo4j_driver
import app.core.database as _db_module


def _mark_neo4j_unavailable():
    _db_module._neo4j_unavailable = True
    _db_module._neo4j_driver = None


async def add_analysis_to_graph(
    job_id: str,
    image_hash: str,
    location: str,
    latitude: Optional[float],
    longitude: Optional[float],
    evidence_chain: list[dict],
    confidence: float,
) -> None:
    """분석 결과를 지식 그래프에 추가"""
    driver = get_neo4j_driver()
    if driver is None:
        return

    try:
        async with driver.session() as session:
            # 이미지 노드 생성/업데이트
            await session.run(
                """
                MERGE (img:Image {hash: $hash})
                SET img.job_id = $job_id,
                    img.analyzed_at = datetime()
                """,
                hash=image_hash,
                job_id=job_id,
            )

            # 위치 노드 생성/업데이트
            if location and location != "위치 특정 불가":
                await session.run(
                    """
                    MERGE (loc:Location {name: $location})
                    SET loc.latitude = $lat,
                        loc.longitude = $lon,
                        loc.last_seen = datetime()
                    WITH loc
                    MATCH (img:Image {hash: $hash})
                    MERGE (img)-[r:LOCATED_AT]->(loc)
                    SET r.confidence = $confidence,
                        r.job_id = $job_id
                    """,
                    location=location,
                    lat=latitude,
                    lon=longitude,
                    hash=image_hash,
                    confidence=confidence,
                    job_id=job_id,
                )

            # 단서(증거) 노드 연결
            for ev in evidence_chain:
                if ev.get("source") in ("naver_place", "kakao_place", "google_place"):
                    await session.run(
                        """
                        MERGE (poi:POI {description: $desc})
                        SET poi.source = $source
                        WITH poi
                        MATCH (loc:Location {name: $location})
                        MERGE (poi)-[:CONFIRMS]->(loc)
                        """,
                        desc=ev.get("description", ""),
                        source=ev.get("source", ""),
                        location=location,
                    )

            logger.debug(f"Knowledge graph updated: {job_id}")

    except Exception as e:
        logger.debug(f"Neo4j 사용 불가 — 지식 그래프 비활성화: {type(e).__name__}")
        _mark_neo4j_unavailable()


async def query_similar_locations(
    location: str,
    limit: int = 5,
) -> list[dict]:
    """지식 그래프에서 유사 위치 쿼리"""
    driver = get_neo4j_driver()
    if driver is None:
        return []

    try:
        async with driver.session() as session:
            result = await session.run(
                """
                MATCH (loc:Location)-[:LOCATED_AT]-(img:Image)
                WHERE loc.name CONTAINS $keyword
                WITH loc, count(img) as img_count
                ORDER BY img_count DESC
                LIMIT $limit
                RETURN loc.name as location,
                       loc.latitude as lat,
                       loc.longitude as lon,
                       img_count
                """,
                keyword=location.split()[-1] if location else "",
                limit=limit,
            )
            return [dict(record) async for record in result]
    except Exception as e:
        logger.debug(f"Neo4j query skipped: {type(e).__name__}")
        _mark_neo4j_unavailable()
        return []


async def get_location_confidence_history(location: str) -> dict:
    """특정 위치에 대한 과거 분석 신뢰도 통계"""
    driver = get_neo4j_driver()
    if driver is None:
        return {}

    try:
        async with driver.session() as session:
            result = await session.run(
                """
                MATCH (img:Image)-[r:LOCATED_AT]->(loc:Location {name: $location})
                RETURN
                    count(r) as total_analyses,
                    avg(r.confidence) as avg_confidence,
                    max(r.confidence) as max_confidence,
                    min(r.confidence) as min_confidence
                """,
                location=location,
            )
            record = await result.single()
            if record:
                return dict(record)
    except Exception as e:
        logger.debug(f"Neo4j history query skipped: {type(e).__name__}")
        _mark_neo4j_unavailable()

    return {}
