"""
인증 API v2 — 완전한 DB 기반 JWT 인증
"""
from datetime import datetime, timedelta
from fastapi import APIRouter, HTTPException, Depends, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel, EmailStr
from passlib.context import CryptContext
from jose import JWTError, jwt
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from ...core.config import settings
from ...core.database import get_db
from ...models.user import User, PlanType, PLAN_LIMITS

router = APIRouter()
pwd_ctx = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2 = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/token")
ALGO = "HS256"


# ── Pydantic 스키마 ────────────────────────────────────────
class RegisterRequest(BaseModel):
    email: EmailStr
    password: str
    name: str = ""


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int


class UserResponse(BaseModel):
    id: int
    email: str
    name: str
    plan: str
    monthly_usage: int
    monthly_limit: int
    can_analyze: bool


class FeedbackRequest(BaseModel):
    job_id: str
    is_correct: bool
    actual_location: str = ""


# ── 유틸 ──────────────────────────────────────────────────
def _make_token(data: dict) -> str:
    exp = datetime.utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    return jwt.encode({**data, "exp": exp}, settings.SECRET_KEY, algorithm=ALGO)


async def _get_user_by_email(db: AsyncSession, email: str) -> User | None:
    result = await db.execute(select(User).where(User.email == email))
    return result.scalar_one_or_none()


async def _get_user_by_id(db: AsyncSession, user_id: int) -> User | None:
    result = await db.execute(select(User).where(User.id == user_id))
    return result.scalar_one_or_none()


async def get_current_user(
    token: str = Depends(oauth2),
    db: AsyncSession = Depends(get_db),
) -> User:
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[ALGO])
        uid = int(payload["sub"])
    except (JWTError, KeyError, ValueError):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "유효하지 않은 토큰")

    user = await _get_user_by_id(db, uid)
    if not user or not user.is_active:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "계정을 찾을 수 없습니다")
    return user


async def get_optional_user(
    token: str | None = Depends(OAuth2PasswordBearer(tokenUrl="/api/v1/auth/token", auto_error=False)),
    db: AsyncSession = Depends(get_db),
) -> User | None:
    if not token:
        return None
    try:
        return await get_current_user(token, db)
    except HTTPException:
        return None


# ── 엔드포인트 ─────────────────────────────────────────────
@router.post("/register", response_model=Token, status_code=status.HTTP_201_CREATED)
async def register(req: RegisterRequest, db: AsyncSession = Depends(get_db)):
    existing = await _get_user_by_email(db, req.email)
    if existing:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "이미 사용 중인 이메일입니다")

    user = User(
        email=req.email,
        hashed_password=pwd_ctx.hash(req.password),
        name=req.name or req.email.split("@")[0],
        plan=PlanType.free,
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)

    return Token(
        access_token=_make_token({"sub": str(user.id)}),
        expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
    )


@router.post("/token", response_model=Token)
async def login(form: OAuth2PasswordRequestForm = Depends(), db: AsyncSession = Depends(get_db)):
    user = await _get_user_by_email(db, form.username)
    if not user or not pwd_ctx.verify(form.password, user.hashed_password):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "이메일 또는 비밀번호가 잘못되었습니다")
    if not user.is_active:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "비활성화된 계정입니다")

    return Token(
        access_token=_make_token({"sub": str(user.id)}),
        expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
    )


@router.get("/me", response_model=UserResponse)
async def get_me(user: User = Depends(get_current_user)):
    return UserResponse(
        id=user.id,
        email=user.email,
        name=user.name,
        plan=user.plan,
        monthly_usage=user.monthly_usage,
        monthly_limit=user.monthly_limit,
        can_analyze=user.can_analyze,
    )


class AdminSetPlan(BaseModel):
    email: EmailStr
    plan: str  # "free" | "pro" | "expert"
    admin_key: str


@router.post("/admin/set-plan")
async def admin_set_plan(req: AdminSetPlan, db: AsyncSession = Depends(get_db)):
    """어드민 전용 — 사용자 플랜 변경 (결제 연동 전 수동 처리)"""
    if req.admin_key != settings.ADMIN_SECRET_KEY:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "관리자 키가 올바르지 않습니다")

    allowed = {p.value for p in PlanType}
    if req.plan not in allowed:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, f"유효한 플랜: {allowed}")

    user = await _get_user_by_email(db, req.email)
    if not user:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "사용자를 찾을 수 없습니다")

    old_plan = user.plan
    user.plan = PlanType(req.plan)
    await db.commit()

    return {"status": "ok", "email": req.email, "old_plan": old_plan, "new_plan": req.plan}


@router.post("/feedback")
async def submit_feedback(
    req: FeedbackRequest,
    user: User | None = Depends(get_optional_user),
    db: AsyncSession = Depends(get_db),
):
    """사용자 피드백 수집 (자가학습용)"""
    from ...services.selflearn import get_selflearn_service
    svc = get_selflearn_service()
    await svc.record_feedback(
        job_id=req.job_id,
        predicted_location="",
        confidence=0.0,
        is_correct=req.is_correct,
        actual_location=req.actual_location,
    )
    return {"status": "recorded", "job_id": req.job_id}
