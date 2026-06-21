#!/bin/bash
# EXXAS AI 학습 파이프라인 자동화 스크립트
# 사용법: ./ml/run_pipeline.sh [collect|process|upload|status|deploy <model_id>|rollback]

set -e
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"

VENV=".venv/bin/python"
ML_DATA="./ml_data"
PROCESSED="./ml_data/processed"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'
log()  { echo -e "${CYAN}[PIPELINE]${NC} $1"; }
ok()   { echo -e "${GREEN}[OK]${NC}    $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC}  $1"; }
err()  { echo -e "${RED}[ERR]${NC}   $1"; exit 1; }

# .env 로드
if [ -f "$ROOT/../.env" ]; then
    set -a; source "$ROOT/../.env"; set +a
    ok ".env 로드됨"
else
    warn ".env 없음 — 환경변수 직접 설정 필요"
fi

CMD=${1:-"status"}

case "$CMD" in

# ──────────────────────────────────────────────────────
collect)
    log "===== 1단계: 데이터 수집 시작 ====="
    log "Flickr 목표: ${MAX_FLICKR:-200000}개"
    log "로드뷰 목표: ${MAX_ROADVIEW:-200000}개"
    log "카카오 목표: ${MAX_KAKAO:-100000}개"

    mkdir -p "$ML_DATA" logs

    $VENV -m ml.data.collect_all "$ML_DATA" 2>&1 | tee logs/collect_$(date +%Y%m%d_%H%M%S).log
    ok "데이터 수집 완료"
    ;;

# ──────────────────────────────────────────────────────
process)
    log "===== 2단계: 데이터 처리 시작 ====="

    if [ ! -d "$ML_DATA" ]; then
        err "수집 데이터 없음. 먼저 'collect' 실행"
    fi

    mkdir -p "$PROCESSED"
    $VENV -m ml.processing.processor "$ML_DATA" "$PROCESSED" 2>&1 | tee logs/process_$(date +%Y%m%d_%H%M%S).log
    ok "데이터 처리 완료 — $PROCESSED 확인"

    # 통계 출력
    if [ -f "$PROCESSED/dataset_stats.json" ]; then
        log "데이터셋 통계:"
        $VENV -c "import json; d=json.load(open('$PROCESSED/dataset_stats.json')); \
            print(f'  Total: {d[\"after_balance\"]:,}  Train: {d[\"train\"]:,}  Val: {d[\"val\"]:,}  Test: {d[\"test\"]:,}')"
    fi
    ;;

# ──────────────────────────────────────────────────────
upload)
    log "===== 3단계: HuggingFace 업로드 ====="

    if [ -z "$HF_TOKEN" ]; then
        err "HF_TOKEN 환경변수 없음"
    fi
    HF_DATASET_REPO=${HF_DATASET_REPO:-"exxas/korean-location-clips"}
    log "업로드 대상: $HF_DATASET_REPO"

    $VENV -m ml.processing.processor "$ML_DATA" "$PROCESSED" --upload \
        2>&1 | tee logs/upload_$(date +%Y%m%d_%H%M%S).log
    ok "HuggingFace 업로드 완료"
    log "다음: Colab에서 korean_clip_colab.ipynb 실행"
    ;;

# ──────────────────────────────────────────────────────
benchmark)
    log "===== 벤치마크 실행 ====="
    NEW_MODEL=${2:-""}
    if [ -z "$NEW_MODEL" ]; then
        err "Usage: ./run_pipeline.sh benchmark <new_model_id>"
    fi
    BASELINE=${3:-"openai/clip-vit-large-patch14"}

    $VENV -m ml.evaluation.benchmark \
        "$PROCESSED/test.jsonl" \
        "$ML_DATA" \
        "$NEW_MODEL" \
        "$BASELINE" \
        2>&1 | tee logs/benchmark_$(date +%Y%m%d_%H%M%S).log
    ;;

# ──────────────────────────────────────────────────────
deploy)
    log "===== 모델 배포 ====="
    NEW_MODEL=${2:-""}
    if [ -z "$NEW_MODEL" ]; then
        err "Usage: ./run_pipeline.sh deploy <hf_model_id>"
    fi

    # 벤치마크 결과 확인
    BENCH="ml_data/benchmark_result.json"
    if [ ! -f "$BENCH" ]; then
        warn "벤치마크 결과 없음 — 로컬 벤치마크 실행"
        $VENV -m ml.evaluation.benchmark \
            "$PROCESSED/test.jsonl" "$ML_DATA" "$NEW_MODEL" \
            2>&1 | tee logs/pre_deploy_bench.log
    fi

    log "모델 교체 시작: $NEW_MODEL"
    $VENV -m ml.deployment.safe_swap swap "$NEW_MODEL" "$BENCH" "$ML_DATA"

    ok "===== 배포 완료 ====="
    log "서버 재시작 필요:"
    log "  kill \$(lsof -ti:8000) && .venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000 &"
    log "  kill \$(pgrep -f 'celery.*worker') && .venv/bin/celery -A app.core.celery_app worker --loglevel=info &"
    ;;

# ──────────────────────────────────────────────────────
rollback)
    log "===== 롤백 ====="
    STEPS=${2:-1}
    $VENV -m ml.deployment.safe_swap rollback "$STEPS"
    ok "롤백 완료 — 서버 재시작 필요"
    ;;

# ──────────────────────────────────────────────────────
status)
    log "===== 파이프라인 상태 ====="
    echo ""

    # 데이터 수집 상태
    if [ -f "$ML_DATA/collection_stats.json" ]; then
        ok "수집 완료:"
        $VENV -c "import json; d=json.load(open('$ML_DATA/collection_stats.json')); \
            [print(f'  {k}: {v.get(\"count\",0):,}') for k,v in d.get('sources',{}).items()]"
    else
        warn "수집 데이터 없음"
    fi

    # 처리 상태
    if [ -f "$PROCESSED/dataset_stats.json" ]; then
        ok "처리 완료:"
        $VENV -c "import json; d=json.load(open('$PROCESSED/dataset_stats.json')); \
            print(f'  Train: {d[\"train\"]:,}  Val: {d[\"val\"]:,}  Test: {d[\"test\"]:,}')"
    else
        warn "처리된 데이터 없음"
    fi

    # 현재 활성 모델
    echo ""
    log "현재 활성 모델:"
    $VENV -m ml.deployment.safe_swap status 2>/dev/null || echo "  레지스트리 없음"

    # 벤치마크 결과
    if [ -f "ml_data/benchmark_result.json" ]; then
        echo ""
        ok "마지막 벤치마크:"
        $VENV -c "import json; d=json.load(open('ml_data/benchmark_result.json')); \
            print(f'  배포: {d[\"deploy\"]}'); \
            print(f'  이유: {d.get(\"reason\",\"\")}')"
    fi
    ;;

# ──────────────────────────────────────────────────────
all)
    log "===== 전체 파이프라인 순차 실행 ====="
    bash "$0" collect
    bash "$0" process
    bash "$0" upload
    ok "수집/처리/업로드 완료"
    echo ""
    log "다음 단계:"
    echo -e "  ${CYAN}1.${NC} Colab에서 ml/training/korean_clip_colab.ipynb 실행"
    echo -e "  ${CYAN}2.${NC} 학습 완료 후: ${CYAN}./ml/run_pipeline.sh deploy <HF_MODEL_ID>${NC}"
    ;;

*)
    echo ""
    echo "EXXAS AI 학습 파이프라인"
    echo ""
    echo "사용법: ./ml/run_pipeline.sh <command>"
    echo ""
    echo "Commands:"
    echo "  all                  전체 파이프라인 (collect → process → upload)"
    echo "  collect              1. 데이터 수집 (Flickr + 로드뷰 + 카카오)"
    echo "  process              2. 데이터 처리 (필터 + 정규화 + 분할)"
    echo "  upload               3. HuggingFace 업로드"
    echo "  benchmark <model>    신규 모델 벤치마크 비교"
    echo "  deploy <model>       벤치마크 통과 시 모델 교체"
    echo "  rollback [steps]     이전 모델로 롤백"
    echo "  status               현재 상태 확인"
    echo ""
    echo "환경변수 (.env):"
    echo "  FLICKR_API_KEY, NAVER_CLIENT_ID/SECRET, KAKAO_API_KEY"
    echo "  HF_TOKEN, HF_DATASET_REPO"
    echo "  MAX_FLICKR=200000, MAX_ROADVIEW=200000, MAX_KAKAO=100000"
    ;;
esac
