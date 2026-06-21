from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, Text, ForeignKey, JSON, Enum
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from ..core.database import Base
import enum


class AnalysisStatus(str, enum.Enum):
    queued = "queued"
    running = "running"
    completed = "completed"
    failed = "failed"


class ExplorationMode(str, enum.Enum):
    fast = "fast"
    elimination = "elimination"
    inductive = "inductive"
    indoor = "indoor"
    nature = "nature"
    urban = "urban"
    default = "default"


class Analysis(Base):
    __tablename__ = "analyses"

    id = Column(Integer, primary_key=True, index=True)
    job_id = Column(String(36), unique=True, index=True, nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)

    # 상태
    status = Column(Enum(AnalysisStatus), default=AnalysisStatus.queued)
    error_message = Column(Text, nullable=True)

    # 입력
    image_hash_phash = Column(String(64), index=True)
    image_hash_dhash = Column(String(64))
    image_size_bytes = Column(Integer)
    image_media_type = Column(String(50))
    manipulation_suspected = Column(Boolean, default=False)

    # 결과
    location = Column(String(500))
    latitude = Column(Float, nullable=True)
    longitude = Column(Float, nullable=True)
    address = Column(String(1000), nullable=True)
    confidence = Column(Float, nullable=True)
    confidence_label = Column(String(20), nullable=True)

    # 수사 메타
    exploration_mode = Column(Enum(ExplorationMode), nullable=True)
    total_steps = Column(Integer, default=0)
    elapsed_seconds = Column(Float, nullable=True)
    hallucination_check_passed = Column(Boolean, nullable=True)

    # 상세 데이터 (JSON)
    evidence_chain = Column(JSON, nullable=True)
    hypothesis_tree = Column(JSON, nullable=True)
    stage_results = Column(JSON, nullable=True)   # 각 Stage 요약
    ensemble_breakdown = Column(JSON, nullable=True)  # Stage 7 결과

    # LLM
    final_reasoning = Column(Text, nullable=True)
    llm_provider = Column(String(50))
    llm_model = Column(String(100))

    # 피드백 (자가학습용)
    user_feedback_correct = Column(Boolean, nullable=True)  # True=정답 확인, False=오답
    user_feedback_actual_location = Column(String(500), nullable=True)
    feedback_at = Column(DateTime(timezone=True), nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    user = relationship("User", back_populates="analyses", lazy="select")


# User 모델에 관계 추가
from .user import User
User.analyses = relationship("Analysis", back_populates="user", lazy="dynamic")
