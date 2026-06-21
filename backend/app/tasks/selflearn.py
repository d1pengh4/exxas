"""
자가학습 Celery 태스크
- RLHF 가중치 업데이트 (일간 새벽 3시)
- VPR DB 상태 체크 (6시간 주기)
"""
import asyncio
from loguru import logger
from ..core import ssl_patch  # noqa: F401
from ..core.celery_app import celery_app


@celery_app.task(name="update_ensemble_weights")
def update_ensemble_weights_task():
    """RLHF 피드백 기반 앙상블 가중치 업데이트"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        result = loop.run_until_complete(_async_update_weights())
        logger.info(f"[RLHF] 가중치 업데이트 완료: {result}")
        return result
    except Exception as e:
        logger.error(f"[RLHF] 가중치 업데이트 실패: {e}")
        return {"error": str(e)}
    finally:
        loop.close()


async def _async_update_weights():
    from ..services.selflearn import get_selflearn_service
    svc = get_selflearn_service()
    return await svc.update_ensemble_weights()


@celery_app.task(name="vpr_health_check")
def vpr_health_check_task():
    """Milvus VPR DB 상태 체크 + 임베딩 수 로깅"""
    try:
        from pymilvus import connections, Collection, utility
        connections.connect(alias="default", host="localhost", port=19530)
        if utility.has_collection("image_embeddings"):
            col = Collection("image_embeddings")
            count = col.num_entities
            logger.info(f"[VPR Health] Milvus image_embeddings: {count}개 임베딩")
            return {"status": "ok", "embedding_count": count}
        else:
            logger.warning("[VPR Health] Milvus 컬렉션 없음 — seed_milvus.py 실행 필요")
            return {"status": "no_collection"}
    except Exception as e:
        logger.debug(f"[VPR Health] Milvus 미연결 (정상): {e}")
        return {"status": "milvus_unavailable"}


@celery_app.task(name="daily_retraining")
def daily_retraining_task():
    """일간 LoRA 재학습 + 가중치 업데이트 연계"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        result = loop.run_until_complete(_async_daily_retrain())
        return result
    except Exception as e:
        logger.error(f"[Daily Retrain] 실패: {e}")
        return {"error": str(e)}
    finally:
        loop.close()


async def _async_daily_retrain():
    from ..services.selflearn import get_selflearn_service
    svc = get_selflearn_service()
    job = await svc.schedule_retraining("daily")
    retrain_result = await svc.run_lora_finetune(job)
    # 재학습 후 가중치도 업데이트
    weight_result = await svc.update_ensemble_weights()
    return {
        "retraining": retrain_result,
        "weights_updated": weight_result.get("updated", False),
    }
