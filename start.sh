#!/bin/bash
# EXXAS v2.0 로컬 개발 환경 시작 스크립트
set -e

ROOT=$(cd "$(dirname "$0")" && pwd)
cd "$ROOT"

# ── 색상 ──────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

log() { echo -e "${CYAN}[EXXAS]${NC} $1"; }
ok()  { echo -e "${GREEN}[OK]${NC}   $1"; }
warn(){ echo -e "${YELLOW}[WARN]${NC} $1"; }
err() { echo -e "${RED}[ERR]${NC}  $1"; exit 1; }

# ── 0. .env 확인 ──────────────────────────────────────────
if [ ! -f .env ]; then
  err ".env 파일이 없습니다. .env.example 을 복사 후 API 키를 입력하세요."
fi

LLM_PROVIDER=$(grep "^LLM_PROVIDER=" .env | cut -d= -f2 | tr -d '[:space:]')
if [ "$LLM_PROVIDER" = "ollama" ]; then
  if ! command -v ollama &>/dev/null; then
    warn "Ollama 미설치. https://ollama.ai 에서 설치 후 'ollama pull qwen2.5vl:7b' 실행 필요"
  fi
elif grep -q "^ANTHROPIC_API_KEY=$" .env; then
  warn "ANTHROPIC_API_KEY 가 비어있습니다. Claude API 호출이 실패합니다."
  warn "LLM_PROVIDER=ollama 로 변경하면 로컬 LLM 사용 가능합니다."
fi

# ── 1. Docker 인프라 (PostgreSQL + Redis) ─────────────────
log "인프라 컨테이너 시작 (postgres + redis)..."
docker compose up postgres redis -d

log "DB 준비 대기 중 (최대 30초)..."
for i in $(seq 1 30); do
  if docker compose exec -T postgres pg_isready -U exxas -q 2>/dev/null; then
    ok "PostgreSQL 준비 완료"
    break
  fi
  sleep 1
done

# Milvus/Neo4j는 선택적 (없어도 동작, 경고만)
log "선택적: Milvus + Neo4j 시작 시도..."
docker compose up milvus neo4j -d 2>/dev/null || warn "Milvus/Neo4j 시작 실패 (VPR/지식그래프 기능 비활성)"

# ── 2. 백엔드 Python 환경 ─────────────────────────────────
cd "$ROOT/backend"

# Milvus VPR DB 초기 시드 (최초 1회만, 컬렉션이 비어 있을 때)
_seed_milvus() {
  if docker compose ps milvus 2>/dev/null | grep -q "running"; then
    log "Milvus VPR DB 초기화 확인..."
    if python3 -c "
from pymilvus import connections, utility
connections.connect(host='localhost', port=19530)
print(utility.has_collection('vpr_embeddings'))
" 2>/dev/null | grep -q "False"; then
      log "VPR DB 시드 중 (최초 1회)..."
      cd "$ROOT/backend" && .venv/bin/python -m app.data.seed_vpr_db 2>&1 | tail -5 && ok "VPR DB 시드 완료" || warn "VPR DB 시드 실패 (무시됨)"
    else
      log "VPR DB 이미 초기화됨, 시드 스킵"
    fi
  fi
}
_seed_milvus

if [ ! -d ".venv" ]; then
  log "Python 가상환경 생성..."
  python3 -m venv .venv
fi

log "Python 패키지 설치 중..."
.venv/bin/pip install -q --upgrade pip
.venv/bin/pip install -q -r requirements.txt
ok "Python 의존성 설치 완료"

# ── 3. 백엔드 서버 ────────────────────────────────────────
log "FastAPI 백엔드 시작 (port 8000)..."
.venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload &
BACKEND_PID=$!

# ── 4. Celery Worker + Beat ───────────────────────────────
log "Celery Worker 시작..."
.venv/bin/celery -A app.core.celery_app worker --loglevel=info --concurrency=2 &
CELERY_PID=$!

log "Celery Beat 스케줄러 시작 (일간 재학습)..."
.venv/bin/celery -A app.core.celery_app beat --loglevel=info &
CELERY_BEAT_PID=$!

# ── 5. 프론트엔드 ─────────────────────────────────────────
cd "$ROOT/frontend"

if [ ! -d "node_modules" ]; then
  log "npm install 실행..."
  npm install
fi

log "Next.js 프론트엔드 시작 (port 3000)..."
npm run dev &
FRONTEND_PID=$!

# ── 완료 ──────────────────────────────────────────────────
echo ""
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}  EXXAS v2.0 실행 중${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "  프론트엔드:  ${CYAN}http://localhost:3000${NC}"
echo -e "  API 문서:    ${CYAN}http://localhost:8000/docs${NC}"
echo -e "  헬스체크:    ${CYAN}http://localhost:8000/health${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo "종료: Ctrl+C"

# ── 종료 처리 ─────────────────────────────────────────────
cleanup() {
  log "종료 중..."
  kill $FRONTEND_PID $BACKEND_PID $CELERY_PID $CELERY_BEAT_PID 2>/dev/null
  docker compose stop postgres redis milvus neo4j 2>/dev/null
  ok "정상 종료"
}

trap cleanup INT TERM
wait
