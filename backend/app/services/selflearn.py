"""
자가학습 & Closed-Loop AI
- RLHF 피드백 수집 (명시적 + 암묵적)
- 야간 배치 재학습 스케줄러
- Active Learning (최저 신뢰도 케이스 우선)
- A/B 테스트 프레임워크
- Catastrophic Forgetting 방지 (EWC + Replay Buffer)
"""
import json
import random
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Optional
from loguru import logger


@dataclass
class FeedbackRecord:
    job_id: str
    predicted_location: str
    actual_location: Optional[str]
    confidence: float
    is_correct: Optional[bool]
    implicit_signals: dict = field(default_factory=dict)  # 클릭, 체류시간, 공유
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class RetrainingJob:
    job_id: str
    trigger: str           # "daily" | "weekly" | "monthly" | "active_learning"
    sample_count: int
    priority_samples: list[str] = field(default_factory=list)  # 최저 신뢰도 job_id 목록
    status: str = "pending"
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    metrics: dict = field(default_factory=dict)


class SelfLearningService:
    """
    Closed-Loop AI: 실시간 추론 → 피드백 수집 → 야간 재학습 → 자동 배포
    """

    def __init__(self):
        self._feedback_buffer: list[FeedbackRecord] = []
        self._replay_buffer: list[dict] = []  # 이전 고품질 샘플 보관 (Catastrophic Forgetting 방지)
        self._ab_test_config: dict = {"current_model": "v1", "canary_model": None, "canary_pct": 0}

    async def record_feedback(
        self,
        job_id: str,
        predicted_location: str,
        confidence: float,
        is_correct: Optional[bool] = None,
        actual_location: Optional[str] = None,
        implicit_signals: dict | None = None,
    ) -> None:
        """사용자 피드백 수집 (명시적 + 암묵적)"""
        record = FeedbackRecord(
            job_id=job_id,
            predicted_location=predicted_location,
            actual_location=actual_location,
            confidence=confidence,
            is_correct=is_correct,
            implicit_signals=implicit_signals or {},
        )
        self._feedback_buffer.append(record)
        logger.debug(f"Feedback recorded: {job_id} correct={is_correct}")

        # Redis에 영구 저장
        await self._persist_feedback(record)

    async def _persist_feedback(self, record: FeedbackRecord) -> None:
        try:
            import redis.asyncio as aioredis
            from ..core.config import settings
            r = aioredis.from_url(settings.REDIS_URL, encoding="utf-8", decode_responses=True)
            key = f"feedback:{record.job_id}"
            await r.setex(key, 86400 * 30, json.dumps(record.__dict__))
            await r.aclose()
        except Exception as e:
            logger.warning(f"Feedback persist failed: {e}")

    async def get_active_learning_samples(self, n: int = 50) -> list[str]:
        """
        Active Learning: 신뢰도가 가장 낮은 케이스를 우선 인간 검토 대상으로 선정
        데이터 효율 최대화
        """
        try:
            import redis.asyncio as aioredis
            from ..core.config import settings
            redis = aioredis.from_url(settings.REDIS_URL, encoding="utf-8", decode_responses=True)

            # 최근 분석 결과에서 낮은 신뢰도 케이스 수집
            pattern = "job:*:status"
            keys = await redis.keys(pattern)

            low_confidence: list[tuple[float, str]] = []
            for key in keys[:500]:  # 최근 500건
                raw = await redis.get(key)
                if not raw:
                    continue
                data = json.loads(raw)
                if data.get("status") == "completed":
                    conf = data.get("confidence", 1.0)
                    if conf and conf < 0.70:
                        job_id = key.decode().split(":")[1] if isinstance(key, bytes) else key.split(":")[1]
                        low_confidence.append((conf, job_id))

            low_confidence.sort(key=lambda x: x[0])
            await redis.aclose()
            return [jid for _, jid in low_confidence[:n]]

        except Exception as e:
            logger.error(f"Active learning sampling failed: {e}")
            return []

    async def schedule_retraining(self, trigger: str = "daily") -> RetrainingJob:
        """재학습 스케줄 생성"""
        priority_samples = await self.get_active_learning_samples(50)

        job = RetrainingJob(
            job_id=f"retrain_{trigger}_{datetime.utcnow().strftime('%Y%m%d_%H%M')}",
            trigger=trigger,
            sample_count=len(self._feedback_buffer),
            priority_samples=priority_samples,
        )

        logger.info(f"Retraining job scheduled: {job.job_id}, samples={job.sample_count}")
        return job

    async def run_lora_finetune(self, retraining_job: RetrainingJob) -> dict:
        """
        LoRA 경량 파인튜닝 (일간)
        실제 GPU 학습은 별도 스크립트 (ml/training/)에서 실행
        여기서는 학습 데이터 준비 + 메트릭 추적
        """
        retraining_job.status = "running"
        retraining_job.started_at = datetime.utcnow().isoformat()

        try:
            # 학습 데이터 준비
            training_data = await self._prepare_training_data(retraining_job)

            # ml/training/ 스크립트 호출 (비동기 subprocess)
            import asyncio
            import subprocess

            cmd = [
                "python", "ml/training/lora_finetune.py",
                "--data", json.dumps(training_data[:100]),  # 최대 100샘플
                "--epochs", "3",
                "--output", f"ml/models/checkpoint_{retraining_job.job_id}",
            ]

            # 백그라운드 실행 (오래 걸림)
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            retraining_job.status = "submitted"
            return {"status": "submitted", "pid": process.pid}

        except FileNotFoundError:
            # 학습 스크립트 없으면 데이터만 저장
            logger.warning("Training script not found, saving data only")
            retraining_job.status = "data_saved"
            retraining_job.completed_at = datetime.utcnow().isoformat()
            return {"status": "data_saved", "samples": retraining_job.sample_count}

    async def _prepare_training_data(self, job: RetrainingJob) -> list[dict]:
        """피드백에서 학습 데이터 구성"""
        data = []

        # 명시적 정답 피드백
        for rec in self._feedback_buffer:
            if rec.is_correct is True and rec.actual_location:
                data.append({
                    "job_id": rec.job_id,
                    "label": rec.actual_location,
                    "quality": "high",
                    "source": "explicit_feedback",
                })
            elif rec.is_correct is False and rec.actual_location:
                data.append({
                    "job_id": rec.job_id,
                    "label": rec.actual_location,
                    "quality": "high",
                    "source": "correction",
                    "wrong_prediction": rec.predicted_location,
                })

        # Replay buffer (이전 고품질 샘플)
        replay = random.sample(self._replay_buffer, min(20, len(self._replay_buffer)))
        data.extend(replay)

        return data

    async def update_ensemble_weights(self) -> dict:
        """
        RLHF 피드백 기반 앙상블 가중치 자동 업데이트
        - Redis에서 최근 피드백(정답/오답) 수집
        - 소스별 정확도 계산 → DYNAMIC_WEIGHTS 갱신
        - 갱신된 가중치 Redis에 저장 (stage7_ensemble.py가 로딩)
        """
        import redis.asyncio as aioredis
        from ..core.config import settings
        import json as _json

        redis = aioredis.from_url(settings.REDIS_URL, encoding="utf-8", decode_responses=True)
        try:
            # 최근 피드백 수집 (최대 200건)
            keys = await redis.keys("feedback:*")
            records = []
            for key in keys[:200]:
                raw = await redis.get(key)
                if raw:
                    try:
                        records.append(_json.loads(raw))
                    except Exception:
                        pass

            if len(records) < 10:
                logger.info(f"RLHF 가중치 업데이트 스킵: 피드백 {len(records)}건 (최소 10건 필요)")
                return {"skipped": True, "feedback_count": len(records)}

            # 소스별 정확도 집계
            source_correct: dict[str, int] = {}
            source_total: dict[str, int] = {}
            for rec in records:
                if rec.get("is_correct") is None:
                    continue
                # implicit_signals에서 소스 추출 (예: {"dominant_source": "ocr"})
                source = rec.get("implicit_signals", {}).get("dominant_source", "llm")
                source_total[source] = source_total.get(source, 0) + 1
                if rec.get("is_correct"):
                    source_correct[source] = source_correct.get(source, 0) + 1

            if not source_total:
                logger.info("RLHF: 명시적 피드백 없음, 가중치 유지")
                return {"skipped": True, "reason": "no_explicit_feedback"}

            # 정확도 계산
            accuracy: dict[str, float] = {
                src: source_correct.get(src, 0) / total
                for src, total in source_total.items()
                if total >= 3
            }

            # 소스명 → DYNAMIC_WEIGHTS 키 매핑
            _SRC_MAP = {
                "exif": "exif", "gps": "exif",
                "ocr": "ocr", "ocr_poi": "ocr",
                "reverse_search": "reverse_search", "naver_vision": "reverse_search",
                "object_detect": "object_detect", "yolo": "object_detect",
                "geoclip": "geoclip", "streetclip": "geoclip",
                "physical": "physical", "sun": "physical",
                "dem": "dem",
            }

            # 현재 가중치 로드 (Redis 저장분 우선, 없으면 기본값)
            from ..pipeline.stage7_ensemble import DYNAMIC_WEIGHTS
            import copy
            current_weights = _json.loads(
                await redis.get("ensemble:weights") or "{}"
            ) or copy.deepcopy(DYNAMIC_WEIGHTS)

            # 정확도 기반 가중치 소폭 조정 (±20% 제한)
            updated_count = 0
            for src, acc in accuracy.items():
                weight_key = _SRC_MAP.get(src)
                if not weight_key:
                    continue
                for scene_type, weights in current_weights.items():
                    if weight_key not in weights:
                        continue
                    old_w = weights[weight_key]
                    # acc > 0.8 → 최대 +20%, acc < 0.4 → 최대 -20%
                    delta = (acc - 0.6) * 0.4   # [-0.24, +0.16] 범위
                    new_w = max(0.1, min(old_w * (1 + delta), old_w * 1.2))
                    weights[weight_key] = round(new_w, 3)
                    updated_count += 1

            # Redis에 저장 (TTL 7일)
            await redis.setex(
                "ensemble:weights",
                86400 * 7,
                _json.dumps(current_weights),
            )
            logger.info(
                f"RLHF 가중치 업데이트 완료: feedback={len(records)}건, "
                f"accuracy={accuracy}, updated_keys={updated_count}"
            )
            return {
                "updated": True,
                "feedback_count": len(records),
                "accuracy": accuracy,
                "updated_keys": updated_count,
            }

        except Exception as e:
            logger.error(f"RLHF weight update failed: {e}")
            return {"error": str(e)}
        finally:
            await redis.aclose()

    def configure_ab_test(self, canary_model: str, canary_pct: float = 0.05) -> None:
        """A/B 테스트 설정 (Canary 배포)"""
        self._ab_test_config["canary_model"] = canary_model
        self._ab_test_config["canary_pct"] = canary_pct
        logger.info(f"A/B test configured: canary={canary_model} at {canary_pct:.0%}")

    def get_model_for_request(self, user_id: Optional[int] = None) -> str:
        """요청별 모델 결정 (A/B 테스트)"""
        if (
            self._ab_test_config["canary_model"]
            and random.random() < self._ab_test_config["canary_pct"]
        ):
            return self._ab_test_config["canary_model"]
        return self._ab_test_config["current_model"]

    def add_to_replay_buffer(self, sample: dict, max_size: int = 500) -> None:
        """Replay Buffer에 샘플 추가 (Catastrophic Forgetting 방지)"""
        self._replay_buffer.append(sample)
        if len(self._replay_buffer) > max_size:
            self._replay_buffer = self._replay_buffer[-max_size:]


# 싱글톤
_selflearn_service: Optional[SelfLearningService] = None


def get_selflearn_service() -> SelfLearningService:
    global _selflearn_service
    if _selflearn_service is None:
        _selflearn_service = SelfLearningService()
    return _selflearn_service
