from sqlalchemy import Column, Integer, String, Boolean, DateTime, Enum
from sqlalchemy.sql import func
from ..core.database import Base
import enum


class PlanType(str, enum.Enum):
    free = "free"
    pro = "pro"
    expert = "expert"
    enterprise = "enterprise"


PLAN_LIMITS = {
    PlanType.free: 5,
    PlanType.pro: 100,
    PlanType.expert: 999999,
    PlanType.enterprise: 999999,
}

# 플랜별 기능 접근 권한
PLAN_FEATURES = {
    PlanType.free: {
        "osint_reverse_search": False,    # 역방향 이미지 검색
        "manipulation_detection": False,  # 조작 탐지
        "full_pipeline": False,           # 전체 7레이어
        "batch_upload": False,            # 배치 업로드
        "api_access": False,              # API 접근
        "report_download": False,         # 리포트 다운로드
        "max_batch": 1,
        "history_days": 7,
    },
    PlanType.pro: {
        "osint_reverse_search": False,
        "manipulation_detection": True,
        "full_pipeline": True,
        "batch_upload": True,
        "api_access": False,
        "report_download": True,
        "max_batch": 10,
        "history_days": 365,
    },
    PlanType.expert: {
        "osint_reverse_search": True,
        "manipulation_detection": True,
        "full_pipeline": True,
        "batch_upload": True,
        "api_access": True,
        "report_download": True,
        "max_batch": 50,
        "history_days": 99999,
    },
    PlanType.enterprise: {
        "osint_reverse_search": True,
        "manipulation_detection": True,
        "full_pipeline": True,
        "batch_upload": True,
        "api_access": True,
        "report_download": True,
        "max_batch": 200,
        "history_days": 99999,
    },
}


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    name = Column(String(100), default="")
    is_active = Column(Boolean, default=True)
    is_verified = Column(Boolean, default=False)

    plan = Column(Enum(PlanType), default=PlanType.free)
    monthly_usage = Column(Integer, default=0)
    usage_reset_at = Column(DateTime(timezone=True), server_default=func.now())

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    @property
    def monthly_limit(self) -> int:
        return PLAN_LIMITS.get(self.plan, 5)

    @property
    def can_analyze(self) -> bool:
        from datetime import datetime, timezone
        # 사용량 리셋 기준: 매월 1일 자정(UTC) 이후 최초 체크 시 리셋
        now = datetime.now(timezone.utc)
        if self.usage_reset_at:
            reset_at = self.usage_reset_at
            if not reset_at.tzinfo:
                reset_at = reset_at.replace(tzinfo=timezone.utc)
            # reset_at이 이번 달 1일보다 이전이면 → 사용량 리셋 필요
            this_month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            if reset_at < this_month_start:
                # 리셋은 API 레이어에서 처리 (여기서는 판단만)
                return True  # 리셋 필요 → 분석 허용
        return self.monthly_usage < self.monthly_limit

    def has_feature(self, feature: str) -> bool:
        features = PLAN_FEATURES.get(self.plan, PLAN_FEATURES[PlanType.free])
        return bool(features.get(feature, False))

    @property
    def max_batch_size(self) -> int:
        features = PLAN_FEATURES.get(self.plan, PLAN_FEATURES[PlanType.free])
        return int(features.get("max_batch", 1))
