"""
데이터 수집 통합 오케스트레이터
- Flickr, 네이버 로드뷰, 카카오 Place 병렬 수집
- 진행 상황 저장 (중단/재시작 지원)
- 완료 후 HuggingFace 업로드 준비
"""
import asyncio
import json
import os
import sys
from pathlib import Path
from datetime import datetime
from loguru import logger


async def collect_all(
    output_dir: Path,
    flickr_key: str = "",
    naver_client_id: str = "",
    naver_client_secret: str = "",
    kakao_key: str = "",
    max_flickr: int = 200_000,
    max_roadview: int = 200_000,
    max_kakao: int = 100_000,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    progress_path = output_dir / "collection_progress.json"
    progress = {}
    if progress_path.exists():
        with open(progress_path) as f:
            progress = json.load(f)

    stats = {"started_at": datetime.now().isoformat(), "sources": {}}

    async def run_flickr():
        if not flickr_key:
            logger.warning("FLICKR_API_KEY 없음 — Flickr 스킵")
            return 0
        if progress.get("flickr_done"):
            logger.info("Flickr 이미 완료됨 — 스킵")
            return progress.get("flickr_count", 0)
        from .flickr_crawler import crawl_flickr
        n = await crawl_flickr(flickr_key, output_dir, max_flickr)
        progress["flickr_done"] = True
        progress["flickr_count"] = n
        return n

    async def run_roadview():
        if not (naver_client_id and naver_client_secret):
            logger.warning("NAVER_CLIENT_ID/SECRET 없음 — 로드뷰 스킵")
            return 0
        if progress.get("roadview_done"):
            logger.info("로드뷰 이미 완료됨 — 스킵")
            return progress.get("roadview_count", 0)
        from .naver_roadview import crawl_roadview
        n = await crawl_roadview(naver_client_id, naver_client_secret, output_dir, max_roadview)
        progress["roadview_done"] = True
        progress["roadview_count"] = n
        return n

    async def run_kakao():
        if not kakao_key:
            logger.warning("KAKAO_API_KEY 없음 — 카카오 스킵")
            return 0
        if progress.get("kakao_done"):
            logger.info("카카오 이미 완료됨 — 스킵")
            return progress.get("kakao_count", 0)
        from .kakao_place_crawler import crawl_kakao_places
        n = await crawl_kakao_places(kakao_key, output_dir, max_kakao)
        progress["kakao_done"] = True
        progress["kakao_count"] = n
        return n

    # 병렬 수집 (각 소스 독립적)
    logger.info("===== 데이터 수집 시작 =====")
    results = await asyncio.gather(
        run_flickr(),
        run_roadview(),
        run_kakao(),
        return_exceptions=True,
    )

    for source, result in zip(["flickr", "roadview", "kakao"], results):
        if isinstance(result, Exception):
            logger.error(f"{source} 수집 오류: {result}")
            stats["sources"][source] = {"error": str(result)}
        else:
            stats["sources"][source] = {"count": result}
            logger.info(f"{source}: {result}개 수집")

    total = sum(v.get("count", 0) for v in stats["sources"].values())
    stats["total"] = total
    stats["completed_at"] = datetime.now().isoformat()
    logger.info(f"===== 수집 완료: 총 {total}개 =====")

    # 진행 상황 저장
    with open(progress_path, "w") as f:
        json.dump(progress, f, indent=2)

    # 수집 통계 저장
    with open(output_dir / "collection_stats.json", "w") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    return stats


if __name__ == "__main__":
    from loguru import logger as _log
    _log.add("logs/collect_{time}.log", rotation="100 MB")

    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("./ml_data")
    result = asyncio.run(collect_all(
        output_dir=out_dir,
        flickr_key=os.environ.get("FLICKR_API_KEY", ""),
        naver_client_id=os.environ.get("NAVER_CLIENT_ID", ""),
        naver_client_secret=os.environ.get("NAVER_CLIENT_SECRET", ""),
        kakao_key=os.environ.get("KAKAO_API_KEY", ""),
        max_flickr=int(os.environ.get("MAX_FLICKR", "200000")),
        max_roadview=int(os.environ.get("MAX_ROADVIEW", "200000")),
        max_kakao=int(os.environ.get("MAX_KAKAO", "100000")),
    ))
    print(json.dumps(result, ensure_ascii=False, indent=2))
