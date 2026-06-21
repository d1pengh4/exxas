# EXXAS v2.0

**사진 한 장으로 한국의 정확한 위치를 찾아내는 OSINT 플랫폼**

Image-Based Geolocalization & OSINT platform that identifies Korean locations down to GPS/street-address precision from a single photograph.

---

## Overview

EXXAS는 업로드된 사진을 분석해 촬영 위치를 추정합니다. 단순한 이미지 분류가 아닌, 8단계 ML 파이프라인과 ReAct LLM 에이전트를 결합해 EXIF, OCR, 역방향 이미지 검색, 물리적 분석(태양 방위각, 그림자, DEM) 등 30개 이상의 도구를 종합해 최종 좌표와 주소를 도출합니다.

## Key Features

- **8-Stage ML Pipeline** — 전처리(ELA, AI생성 감지) → EXIF/GPS → 역방향 이미지 검색 ×8 → OCR+POI → 인프라 감지(YOLOv8) → 임베딩(GeoCLIP, StreetCLIP, DINOv2) → 물리 분석(태양/그림자/기상) → Bayesian 앙상블
- **ReAct LLM Agent** — 30+ OSINT 도구를 자율적으로 호출하는 조사 에이전트 (한국 특화)
- **Korea Specializer** — 지하철 DB, 브랜드 DB, 랜드마크 DB, 도로명주소 API 통합
- **License Plate OCR** — 한국 번호판 → 즉시 시/도 추정
- **Real-time SSE** — 분석 진행 상황 실시간 스트리밍
- **Bayesian Hypothesis Tree** — 증거 기반 위치 가설 가중치 시각화
- **RLHF** — 사용자 피드백으로 앙상블 가중치 자동 업데이트

## Architecture

```
Frontend (Next.js 14)
    │ POST /api/v1/analyze
    ▼
Backend (FastAPI)
    │ Celery task enqueue
    ▼
Worker (Celery)
    │
    ├── EXXASOrchestrator (Stage 0–7 Pipeline)
    │       Stage 0: ELA + AI 생성 감지
    │       Stage 1: EXIF / GPS 추출
    │       Stage 2: 역방향 이미지 검색 ×8
    │       Stage 3: OCR + POI 룩업 (Naver/Kakao/Google)
    │       Stage 4: YOLOv8 객체 감지 + CLIP 장면 분류
    │       Stage 5: GeoCLIP / StreetCLIP / DINOv2 VPR
    │       Stage 6: 태양 방위각 + 기상 + DEM 분석
    │       Stage 7: Dynamic Bayesian Ensemble → lat/lon
    │
    └── EXXASInvestigator (ReAct Agent)
            Groq qwen3-32b / Claude / OpenAI / Ollama
            30+ OSINT Tools
            Hypothesis Tree

    SSE Stream → Frontend
```

## Tech Stack

| Layer | Stack |
|---|---|
| Frontend | Next.js 14, Tailwind CSS, Leaflet |
| Backend | FastAPI, Celery, SQLAlchemy async |
| Database | PostgreSQL + PostGIS, Redis |
| Vector DB | Milvus (optional) |
| Knowledge Graph | Neo4j (optional) |
| ML/Vision | GeoCLIP, StreetCLIP, DINOv2, YOLOv8, OpenCLIP ViT-L/14 |
| OCR | PaddleOCR, EasyOCR, CLOVA OCR |
| LLM | Groq (qwen3-32b), Claude, OpenAI, Ollama |

## Getting Started

### Prerequisites

- Python 3.12+
- Node.js 18+
- Docker & Docker Compose
- macOS Apple Silicon (arm64) or Linux

### 1. Clone & Configure

```bash
git clone https://github.com/d1pengh4/exxas.git
cd exxas
cp .env.example .env
# .env 파일에 API 키 입력 (아래 API Keys 섹션 참고)
```

### 2. Start Infrastructure

```bash
docker compose up postgres redis -d
```

### 3. Backend

```bash
cd backend
python3 -m venv .venv
arch -arm64 .venv/bin/pip install -r requirements.txt   # macOS arm64
.venv/bin/pip install -r requirements.txt                # Linux

# API 서버
arch -arm64 .venv/bin/python3 -u -m uvicorn app.main:app --host 0.0.0.0 --port 8000

# Celery 워커 (별도 터미널)
VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
arch -arm64 .venv/bin/python3.12 -m celery -A app.core.celery_app worker \
  --loglevel=info --concurrency=1 -Q analysis_high,analysis_low,celery -n worker1@%h
```

### 4. Frontend

```bash
cd frontend
npm install
npm run dev
```

### 5. Access

| Service | URL |
|---|---|
| Frontend | http://localhost:3000 |
| API Docs | http://localhost:8000/docs |
| Health | http://localhost:8000/health |

---

## API Keys

`.env.example`를 복사해 `.env`를 만들고 아래 키를 설정합니다.

| Key | 필수 | 용도 |
|---|---|---|
| `GROQ_API_KEY` | 권장 | LLM 추론 (무료) |
| `ANTHROPIC_API_KEY` | 선택 | Claude LLM |
| `NAVER_CLIENT_ID/SECRET` | 권장 | 지도, 장소, 블로그, 이미지 검색 |
| `KAKAO_API_KEY` | 권장 | 지도, 비전 API |
| `JUSO_API_KEY` | 권장 | 행정안전부 도로명주소 API |
| `SERP_API_KEY` | 선택 | Google 역이미지 검색 |
| `SERPAPI_KEY` | 선택 | 웹 검색 (없으면 DuckDuckGo) |
| `FLICKR_API_KEY` | 선택 | Flickr 지오태그 사진 검색 |
| `MAPILLARY_TOKEN` | 선택 | 스트리트뷰 |
| `PUBLIC_DATA_API_KEY` | 선택 | 공공데이터포털 |

> Groq API는 무료 티어 제공: https://console.groq.com
> Naver Developers: https://developers.naver.com
> Kakao Developers: https://developers.kakao.com

---

## Local CLIP Model

Stage 5 임베딩과 한국 지역 분류에 로컬 CLIP 모델을 사용합니다.

모델 파일(`model.safetensors`, ~1.6GB)은 크기 때문에 저장소에 포함되지 않습니다.
HuggingFace에서 ViT-L/14 CLIP 모델을 다운로드해 `modelforder/model/`에 배치하세요:

```bash
from huggingface_hub import snapshot_download
snapshot_download(repo_id="openai/clip-vit-large-patch14", local_dir="modelforder/model")
```

---

## Optional Services

### Milvus (VPR 벡터 검색)

```bash
docker compose up milvus -d
# 시작 후 한 번만 실행
cd backend && python -m app.data.seed_vpr_db
```

### Neo4j (지식 그래프)

```bash
docker compose up neo4j -d
```

---

## Project Structure

```
exxas/
├── backend/
│   ├── app/
│   │   ├── agents/          # ReAct 조사 에이전트
│   │   ├── api/v1/          # FastAPI 라우터
│   │   ├── core/            # 설정, DB, Celery
│   │   ├── data/            # 한국 지하철/랜드마크 DB
│   │   ├── models/          # SQLAlchemy 모델
│   │   ├── pipeline/        # Stage 0–7 분석 파이프라인
│   │   ├── services/        # Korea specializer, OSINT, geocoding 등
│   │   └── tasks/           # Celery 태스크
│   └── requirements.txt
├── frontend/
│   └── src/
│       ├── app/             # Next.js 페이지
│       ├── components/      # UI 컴포넌트
│       └── lib/             # API 클라이언트
├── ml/training/             # LoRA 파인튜닝 스크립트
├── modelforder/model/       # 로컬 CLIP 모델 (git 미포함)
├── docker-compose.yml
├── .env.example
└── start.sh
```

---

## Notes

- **macOS arm64**: `arch -arm64` prefix 필수. uvloop 설치 금지(`pip install uvicorn[standard]` 사용 X).
- **scipy BLAS hang**: `VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1` 환경변수 설정 필수.
- **Celery concurrency**: macOS에서 `--concurrency=1` 사용 (BLAS 충돌 방지).
- **GeoCLIP cold start**: 첫 실행 시 모델 로딩에 최대 6분 소요.

## License

MIT
