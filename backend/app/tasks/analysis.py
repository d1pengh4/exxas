"""
Celery 분석 태스크 v2
- 수사 진행 상황 Redis Pub/Sub 브로드캐스트
- DB에 최종 결과 저장
- 실패 시 자동 재시도 (1회)
"""
import json
import asyncio
from loguru import logger
from ..core import ssl_patch  # noqa: F401  macOS SSL fix — must be first
from ..core.celery_app import celery_app
from ..core.config import settings


@celery_app.task(name="run_analysis", bind=True, max_retries=1, default_retry_delay=5)
def run_analysis_task(self, job_id: str):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(_async_run(job_id, self))
    except Exception as exc:
        logger.error(f"Task failed {job_id}: {exc}")
        raise self.retry(exc=exc)
    finally:
        loop.close()


async def _async_run(job_id: str, task):
    import redis.asyncio as aioredis
    from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
    from sqlalchemy import select, update
    from ..models.analysis import Analysis, AnalysisStatus
    from ..pipeline.orchestrator import EXXASOrchestrator

    redis = aioredis.from_url(settings.REDIS_URL, decode_responses=False)
    engine = create_async_engine(settings.DATABASE_URL)
    SessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async def pub(data: dict):
        payload = json.dumps(data, ensure_ascii=False)
        await redis.publish(f"job:{job_id}:updates", payload)
        await redis.setex(f"job:{job_id}:status", 86400, payload)

    async def update_db(status: AnalysisStatus, **kwargs):
        async with SessionLocal() as db:
            await db.execute(
                update(Analysis)
                .where(Analysis.job_id == job_id)
                .values(status=status, **kwargs)
            )
            await db.commit()

    try:
        # 상태: running
        await pub({"job_id": job_id, "status": "running", "message": "수사 시작 중..."})
        await update_db(AnalysisStatus.running)

        # 이미지 로드
        raw_image = await redis.get(f"job:{job_id}:image")
        raw_type = await redis.get(f"job:{job_id}:media_type")
        if not raw_image:
            raise ValueError("이미지를 찾을 수 없습니다")

        media_type = raw_type.decode() if raw_type else "image/jpeg"

        # 플랜 기능 플래그 로드
        raw_flags = await redis.get(f"job:{job_id}:plan_flags")
        plan_flags = json.loads(raw_flags) if raw_flags else {}
        osint_enabled = plan_flags.get("osint_enabled", False)

        # 수사 진행 상황을 주기적으로 브로드캐스트하기 위해
        # Orchestrator의 investigator에 step 콜백 주입
        orch = EXXASOrchestrator()

        # 도구 실행 시 진행 상황 브로드캐스트 (공식 콜백, 몽키패칭 불필요)
        async def _on_tool_start(step_num: int, tool_name: str) -> None:
            await pub({
                "job_id": job_id,
                "status": "running",
                "message": f"[Step {step_num}] {tool_name} 실행 중...",
                "step": step_num,
            })

        orch.investigator.on_tool_start = _on_tool_start

        # Free 플랜: 역방향 검색 도구 비활성화
        if not osint_enabled:
            orch.investigator.restricted_tools = {"reverse_image_search"}

        # job_id 주입 (지식 그래프 업데이트용)
        orch._job_id = job_id

        # 파이프라인 스테이지 진행상황 콜백 주입
        async def stage_progress_cb(stage_id: str, stage_name: str, status: str):
            await pub({
                "job_id": job_id,
                "status": "running",
                "message": f"{stage_name} {status}...",
                "pipeline_stage": {"id": stage_id, "name": stage_name, "status": status},
            })
        orch._progress_cb = stage_progress_cb

        # 분석 실행
        result = await orch.analyze(raw_image, media_type)

        # Stage 0 캐시에서 해시/조작 탐지 정보 추출
        pre = orch._stage0_cache
        phash = pre.phash if pre else ""
        dhash = pre.dhash if pre else ""
        manipulation = pre.manipulation_suspected if pre else False
        ai_generated = pre.ai_generated_suspected if pre else False

        # 역방향 지오코딩 (address 필드 채우기)
        address = result.address or ""
        if not address and result.latitude and result.longitude:
            try:
                from ..services.geocoding import reverse_geocode
                address = await reverse_geocode(result.latitude, result.longitude) or ""
            except Exception:
                pass

        # GPS 즉시 경로일 때 location 이름도 reverse geocode로 보완
        if result.location.startswith("GPS (") and address:
            result.location = address

        # "위치 특정 불가"지만 실제 주소/좌표가 있으면 address로 대체
        if result.location in ("위치 특정 불가", "", None) and address:
            result.location = address
        elif result.location in ("위치 특정 불가", "", None) and result.latitude and result.longitude:
            result.location = f"{result.latitude:.4f}, {result.longitude:.4f}" 

        # 실제 사용 모델 이름 결정
        actual_model = settings.OLLAMA_MODEL if settings.LLM_PROVIDER == "ollama" else settings.LLM_MODEL

        # DB 업데이트 (누락 필드 포함)
        await update_db(
            AnalysisStatus.completed,
            location=result.location,
            latitude=result.latitude,
            longitude=result.longitude,
            address=address,
            confidence=result.confidence,
            confidence_label=result.confidence_label,
            exploration_mode=result.exploration_mode,
            total_steps=result.total_steps,
            elapsed_seconds=result.elapsed_seconds,
            hallucination_check_passed=result.hallucination_check_passed,
            manipulation_suspected=manipulation,
            image_hash_phash=phash,
            image_hash_dhash=dhash,
            evidence_chain=result.evidence_chain,
            hypothesis_tree=result.hypothesis_tree,
            final_reasoning=result.final_reasoning,
            llm_provider=settings.LLM_PROVIDER,
            llm_model=actual_model,
        )

        # Redis 최종 상태 + Pub/Sub
        final_data = {
            "job_id": job_id,
            "status": "completed",
            "location": result.location,
            "address": address,
            "latitude": result.latitude,
            "longitude": result.longitude,
            "confidence": result.confidence,
            "confidence_label": result.confidence_label,
            "exploration_mode": result.exploration_mode,
            "total_steps": result.total_steps,
            "elapsed_seconds": result.elapsed_seconds,
            "evidence_chain": result.evidence_chain,
            "hypothesis_tree": result.hypothesis_tree,
            "final_reasoning": result.final_reasoning,
            "hallucination_check_passed": result.hallucination_check_passed,
            "image_manipulation_suspected": manipulation,
            "ai_generated_suspected": ai_generated,
        }
        await pub(final_data)

        # 임시 이미지 삭제
        await redis.delete(f"job:{job_id}:image", f"job:{job_id}:media_type")
        logger.info(f"Task completed: {job_id} → {result.location} ({result.confidence:.2%})")

        # phash 기반 분석 결과 캐싱 (24시간, 신뢰도 0.6 이상만)
        if phash and result.confidence and result.confidence >= 0.6:
            try:
                cache_payload = {
                    "location": result.location,
                    "address": address,
                    "latitude": result.latitude,
                    "longitude": result.longitude,
                    "confidence": result.confidence,
                    "confidence_label": result.confidence_label,
                    "evidence_chain": result.evidence_chain,
                    "final_reasoning": result.final_reasoning,
                    "hypothesis_tree": result.hypothesis_tree,
                    "hallucination_check_passed": result.hallucination_check_passed,
                    "image_manipulation_suspected": manipulation,
                    "ai_generated_suspected": ai_generated,
                }
                await redis.setex(f"cache:phash:{phash}", 86400, json.dumps(cache_payload, ensure_ascii=False))
            except Exception:
                pass

    except Exception as e:
        logger.opt(exception=True).error("Task error {}: {}", job_id, str(e))
        err_data = {"job_id": job_id, "status": "failed", "error": str(e)}
        await pub(err_data)
        await update_db(AnalysisStatus.failed, error_message=str(e)[:500])
        raise

    finally:
        await redis.aclose()
        await engine.dispose()


# ── 주기적 태스크 ──────────────────────────────────────────
@celery_app.task(name="daily_retraining")
def daily_retraining():
    """매일 새벽 2시 LoRA 재학습"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(_async_retrain("daily"))
    finally:
        loop.close()


async def _async_retrain(trigger: str):
    from ..services.selflearn import get_selflearn_service
    svc = get_selflearn_service()
    job = await svc.schedule_retraining(trigger)
    result = await svc.run_lora_finetune(job)
    logger.info(f"Retraining {trigger}: {result}")


