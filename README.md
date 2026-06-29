# EXXAS v2.0

**An OSINT platform that pinpoints exact locations in Korea from a single photograph**

Image-Based Geolocalization & OSINT platform that identifies Korean locations down to GPS/street-address precision from a single photograph.

---

## Overview

EXXAS analyzes an uploaded photo to estimate where it was taken. Rather than simple image classification, it combines an 8-stage ML pipeline with a ReAct LLM agent, synthesizing 30+ tools — EXIF, OCR, reverse image search, physical analysis (sun azimuth, shadows, DEM), and more — to derive the final coordinates and address.

## Key Features

- **8-Stage ML Pipeline** — Preprocessing (ELA, AI-generated detection) → EXIF/GPS → Reverse image search ×8 → OCR+POI → Infrastructure detection (YOLOv8) → Embeddings (GeoCLIP, StreetCLIP, DINOv2) → Physical analysis (sun/shadow/weather) → Bayesian ensemble
- **ReAct LLM Agent** — An investigation agent that autonomously calls 30+ OSINT tools (Korea-specialized)
- **Korea Specializer** — Integrates subway DB, brand DB, landmark DB, and road-address API
- **License Plate OCR** — Korean license plates → instant city/province estimation
- **Real-time SSE** — Real-time streaming of analysis progress
- **Bayesian Hypothesis Tree** — Evidence-based visualization of location hypothesis weights
- **RLHF** — Automatically updates ensemble weights from user feedback

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
    │       Stage 0: ELA + AI-generated detection
    │       Stage 1: EXIF / GPS extraction
    │       Stage 2: Reverse image search ×8
    │       Stage 3: OCR + POI lookup (Naver/Kakao/Google)
    │       Stage 4: YOLOv8 object detection + CLIP scene classification
    │       Stage 5: GeoCLIP / StreetCLIP / DINOv2 VPR
    │       Stage 6: Sun azimuth + weather + DEM analysis
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
# Fill in API keys in the .env file (see API Keys section below)
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

# API server
arch -arm64 .venv/bin/python3 -u -m uvicorn app.main:app --host 0.0.0.0 --port 8000

# Celery worker (separate terminal)
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

Copy `.env.example` to `.env` and set the keys below.

| Key | Required | Purpose |
|---|---|---|
| `GROQ_API_KEY` | Recommended | LLM inference (free) |
| `ANTHROPIC_API_KEY` | Optional | Claude LLM |
| `NAVER_CLIENT_ID/SECRET` | Recommended | Maps, places, blog, image search |
| `KAKAO_API_KEY` | Recommended | Maps, Vision API |
| `JUSO_API_KEY` | Recommended | Korean road-address API (Ministry of the Interior and Safety) |
| `SERP_API_KEY` | Optional | Google reverse image search |
| `SERPAPI_KEY` | Optional | Web search (falls back to DuckDuckGo) |
| `FLICKR_API_KEY` | Optional | Flickr geo-tagged photo search |
| `MAPILLARY_TOKEN` | Optional | Street view |
| `PUBLIC_DATA_API_KEY` | Optional | Korea Public Data Portal |

> Groq API offers a free tier: https://console.groq.com
> Naver Developers: https://developers.naver.com
> Kakao Developers: https://developers.kakao.com

---

## Local CLIP Model

Stage 5 embedding and Korean region classification use a local CLIP model.

The model file (`model.safetensors`, ~1.6GB) is not included in the repository due to its size.
Download the ViT-L/14 CLIP model from HuggingFace and place it in `modelforder/model/`:

```bash
from huggingface_hub import snapshot_download
snapshot_download(repo_id="openai/clip-vit-large-patch14", local_dir="modelforder/model")
```

---

## Optional Services

### Milvus (VPR vector search)

```bash
docker compose up milvus -d
# Run once after startup
cd backend && python -m app.data.seed_vpr_db
```

### Neo4j (knowledge graph)

```bash
docker compose up neo4j -d
```

---

## Project Structure

```
exxas/
├── backend/
│   ├── app/
│   │   ├── agents/          # ReAct investigation agent
│   │   ├── api/v1/          # FastAPI routers
│   │   ├── core/            # Settings, DB, Celery
│   │   ├── data/            # Korean subway/landmark DB
│   │   ├── models/          # SQLAlchemy models
│   │   ├── pipeline/        # Stage 0–7 analysis pipeline
│   │   ├── services/        # Korea specializer, OSINT, geocoding, etc.
│   │   └── tasks/           # Celery tasks
│   └── requirements.txt
├── frontend/
│   └── src/
│       ├── app/             # Next.js pages
│       ├── components/      # UI components
│       └── lib/             # API client
├── ml/training/             # LoRA fine-tuning scripts
├── modelforder/model/       # Local CLIP model (not in git)
├── docker-compose.yml
├── .env.example
└── start.sh
```

---

## Notes

- **macOS arm64**: The `arch -arm64` prefix is required. Do not install uvloop (do not use `pip install uvicorn[standard]`).
- **scipy BLAS hang**: The `VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1` environment variables must be set.
- **Celery concurrency**: Use `--concurrency=1` on macOS (avoids BLAS conflicts).
- **GeoCLIP cold start**: Initial model loading can take up to 6 minutes on first run.

## License

MIT
