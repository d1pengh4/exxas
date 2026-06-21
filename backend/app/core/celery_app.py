from celery import Celery
from celery.schedules import crontab
from celery.signals import worker_process_init
from kombu import Queue, Exchange
from .config import settings

celery_app = Celery(
    "exxas",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL,
    include=["app.tasks.analysis"],
)

# ── 우선순위 큐 정의 ──────────────────────────────────────
_analysis_exchange = Exchange("analysis", type="direct")
_default_exchange = Exchange("default", type="direct")

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="Asia/Seoul",
    enable_utc=True,
    task_track_started=True,
    result_expires=86400,  # 24시간
    worker_prefetch_multiplier=1,
    task_acks_late=True,
    include=["app.tasks.analysis", "app.tasks.selflearn"],
    # 우선순위 큐 설정
    task_queues=(
        Queue("analysis_high", _analysis_exchange, routing_key="analysis.high",
              queue_arguments={"x-max-priority": 10}),
        Queue("analysis_low",  _analysis_exchange, routing_key="analysis.low",
              queue_arguments={"x-max-priority": 10}),
        Queue("celery",        _default_exchange,  routing_key="celery"),
    ),
    task_default_queue="analysis_low",
    task_default_exchange="analysis",
    task_default_routing_key="analysis.low",
    task_routes={
        "run_analysis": {"queue": "analysis_low"},       # 기본 — analyze.py에서 오버라이드
        "daily_retraining": {"queue": "celery"},
        "update_ensemble_weights": {"queue": "celery"},
        "vpr_health_check": {"queue": "celery"},
    },
    beat_schedule={
        # 매일 새벽 2시 (Asia/Seoul) LoRA 재학습
        "daily-retrain": {
            "task": "daily_retraining",
            "schedule": crontab(hour=2, minute=0),
        },
        # 매일 새벽 3시 — RLHF 피드백 기반 앙상블 가중치 업데이트
        "daily-rlhf-weights": {
            "task": "update_ensemble_weights",
            "schedule": crontab(hour=3, minute=0),
        },
        # 6시간마다 — Stage2 캐시 만료 예고 및 VPR DB 상태 체크
        "vpr-health-check": {
            "task": "vpr_health_check",
            "schedule": crontab(hour="*/6", minute=30),
        },
    },
)


@worker_process_init.connect
def preload_ml_models(**kwargs):
    """fork pool worker 시작 시 ML 모델 사전 로딩 (첫 작업 타임아웃 방지) — 순차 실행"""
    import os
    import logging
    _log = logging.getLogger("celery.preload")

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    # 1) torch 먼저 import (다른 모델들이 의존)
    try:
        import torch  # noqa: F401
    except Exception:
        pass

    # 2) 로컬 fine-tuned CLIP (Stage 5 주력) — 가장 오래 걸림
    try:
        from app.pipeline.stage5_embedding import _get_local_clip
        _get_local_clip()
        _log.info("[preload] Local CLIP (fine-tuned) loaded")
    except Exception as e:
        _log.warning(f"[preload] Local CLIP load failed: {e}")

    # 3) YOLO (Stage 4)
    try:
        from app.pipeline.stage4_infra import _get_yolo_model
        _get_yolo_model()
        _log.info("[preload] YOLO loaded")
    except Exception as e:
        _log.warning(f"[preload] YOLO load failed: {e}")

    # 4) EasyOCR Reader (Stage 3) — 초기화 5~10s → preload로 타임아웃 방지
    try:
        from app.pipeline.stage3_ocr_gis import preload_easyocr
        preload_easyocr()
        _log.info("[preload] EasyOCR loaded")
    except Exception as e:
        _log.warning(f"[preload] EasyOCR load failed: {e}")

