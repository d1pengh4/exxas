# CLAUDE.md

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

EXXAS v2.0 — 사진 기반 위치 수사 플랫폼 (Image-Based Geolocalization & OSINT). A photo is uploaded and the system identifies its Korean location down to GPS/street-address precision using a multi-stage ML pipeline driven by a ReAct LLM agent.

## Commands

### Full stack start (recommended)
```bash
cp .env.example .env   # fill in API keys first
./start.sh             # starts all services
```

### Manual start — macOS arm64 (component by component)

On macOS (Apple Silicon), always prefix Python/Celery commands with `arch -arm64` and scipy env vars:

```bash
# Infrastructure
docker compose up postgres redis -d

# Backend (from /backend)
python3 -m venv .venv && arch -arm64 .venv/bin/pip install -r requirements.txt
arch -arm64 .venv/bin/python3 -u -m uvicorn app.main:app --host 0.0.0.0 --port 8000

# Celery worker (macOS arm64 — concurrency=1 to avoid BLAS hang)
cd backend && VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  arch -arm64 .venv/bin/python3.12 -m celery -A app.core.celery_app worker \
  --loglevel=info --concurrency=1 -Q analysis_high,analysis_low,celery -n worker1@%h

# Frontend (from /frontend)
npm install && npm run dev
```

### Frontend lint/build
```bash
cd frontend && npm run lint
cd frontend && npm run build
```

### Optional services (VPR + knowledge graph)
```bash
docker compose up milvus neo4j -d
# After Milvus starts, seed VPR embeddings (run once):
python -m app.data.seed_vpr_db
```

### URLs
- Frontend: http://localhost:3000
- API docs: http://localhost:8000/docs
- Health: http://localhost:8000/health

## Architecture

### Request flow
1. Frontend uploads image → `POST /api/v1/analyze`
2. API stores image bytes in Redis (`get_raw_redis`), creates DB record, enqueues Celery task
3. Celery worker (`tasks/analysis.py`) runs `EXXASOrchestrator`
4. Orchestrator runs Stage 0–7 pipeline, then hands results to `EXXASInvestigator`
5. Investigator (ReAct loop) calls registered tools iteratively until confidence threshold met
6. Progress streamed to frontend via SSE (`GET /api/v1/analyze/{id}/stream`)

### Pipeline stages (`backend/app/pipeline/`)
| Stage | File | Function |
|---|---|---|
| 0 | `stage0_preprocess.py` | ELA manipulation detection, perceptual hash, AI-generated image detection (DCT+noise+CLIP) |
| 1 | `stage1_exif.py` | GPS extraction, EXIF forensics, PRNU fingerprinting |
| 2 | `stage2_internet.py` | Reverse image search ×8 (Naver Vision Landmark/OCR, Kakao Vision, Google Lens, Yandex, TinEye, Bing, Naver SmartLens) via catbox.moe; Redis 24h cache; Wayback CDX |
| 3 | `stage3_ocr_gis.py` | PaddleOCR + EasyOCR `["ko","en"]`, Naver/Kakao/Google POI lookup, Korea specializer |
| 4 | `stage4_infra.py` | YOLOv8n object detect, 35-country infra DB, CLIP scene tagging |
| 5 | `stage5_embedding.py` | GeoCLIP + OpenCLIP ViT-L-14 (90-label Korea region) + StreetCLIP + DINOv2-base VPR; Milvus matching |
| 6 | `stage6_physical.py` | Sun/moon azimuth inversion (ephem), Open-Meteo, DEM ridge matching, night star analysis |
| 7 | `stage7_ensemble.py` | Dynamic-weight Bayesian fusion → final lat/lon; loads RLHF-updated weights from Redis |

### Agent layer (`backend/app/agents/`)
- **`investigator.py`**: ReAct loop. Korea-specialized — all images assumed Korean. Vision pre-step: Groq uses scout model (`llama-4-scout`) for image description → qwen3 gets text context; Ollama sends image directly. Registered tools (30+):
  - Core: `exif_extract`, `ocr_extract`, `object_detect`, `geoclip_embed`, `reverse_image_search`
  - Korea: `korea_analyze`, `juso_lookup`, `roadview_check`, `license_plate_lookup`, `transit_lookup`
  - Place: `naver_place_search`, `kakao_place_search`, `naver_local_search`, `kakao_local_search`, `osm_poi_search`
  - OSINT: `web_search`, `search_naver_blog`, `naver_news_search`, `street_view_fetch`, `deep_crawl_url`, `crawl_social`, `reverse_chain`, `flickr_search`, `news_image_search`
  - Indoor/Receipt: `receipt_parse`, `brand_locate`, `barcode_lookup`, `interior_osint`, `phone_lookup`, `biz_reg_lookup`, `auto_chain`
  - Advanced: `sun_analysis`, `shadow_analysis`, `vpr_compare`, `skyline_match`, `weather_cross_check`, `clova_ocr`, `knowledge_graph_query`, `osint_fuse`
- **`hypothesis_tree.py`**: Bayesian hypothesis tree; `Evidence` dataclass carries `metadata` dict with `lat/lon/full_address`.
- **`llm_provider.py`**: Abstraction over Claude/OpenAI/Ollama/Groq. Currently configured with Groq (`qwen/qwen3-32b`). Strips `<think>` tags from qwen3 output. Falls back: qwen3 → scout → 70b → gpt-oss-120b.

### Services (`backend/app/services/`)
- **`korea_specializer.py`**: 852-line Korea engine — subway station DB, brand DB, landmark DB, Juso API, Naver geocoding, road-view lookup.
- **`osint_chain.py`**: DuckDuckGo/SerpAPI web search, Naver Blog search, OSM Overpass POI, Mapillary street view, URL deep crawl.
- **`confidence.py`**: `ConfidenceCalculator` — maps evidence sources to groups, computes weighted final score.
- **`selflearn.py`**: Active learning / feedback persistence. Uses its own `aioredis` instance (does not use shared `get_redis`). `update_ensemble_weights()` writes RLHF feedback to Redis for stage7 to load.
- **`license_plate.py`**: Korean license plate OCR → immediate city/province geocoding (신형/구형/영업용).
- **`knowledge_graph.py`**: Neo4j-backed past investigation history query.
- **`geocoding.py`**: Reverse geocoding utilities (lat/lon → address).
- **`indoor_analyzer.py`**: Indoor scene analysis (building interiors, signage).
- **`ai_detector.py`**: AI-generated image detection.
- **`weather_cross.py`**: Cross-reference weather data with physical analysis.
- **`skyline_match.py`**: Skyline/building silhouette matching.
- **`report.py`**: `InvestigationReport` dataclass — renders investigation output to human-readable MD/JSON.
- **`transit_db.py`**: Transit database utilities.

### Data layer (`backend/app/core/database.py`)
- `get_redis()` — string Redis (`decode_responses=True`), for status/JSON
- `get_raw_redis()` — binary Redis (`decode_responses=False`), for image bytes. **Always use this when storing/loading image data.**
- PostgreSQL via SQLAlchemy async (PostGIS extension)
- Milvus and Neo4j connections are optional — app degrades gracefully if unavailable

### Local CLIP model (`modelforder/model/`)
- Type: `CLIPModel` (ViT-L/14), `projection_dim=768`
- Load via `from transformers import CLIPModel, CLIPProcessor`
- Path resolution: `Path(__file__).resolve().parents[3] / "modelforder" / "model"`

### Static data (`backend/app/data/`)
- `korea_stations_db.py`: Subway/transit station coordinates DB
- `korea_landmarks_db.py`: Landmark coordinates DB (300+ entries)
- `seed_vpr_db.py`: Script to seed Milvus VPR with Korean landmark embeddings

### ML training (`ml/training/`, `backend/ml/training/`)
- `lora_finetune.py`: LoRA fine-tuning script triggered by Celery daily at 02:00

### Frontend (`frontend/src/`)
- `app/page.tsx` — main upload page with auth header, plan badge, batch toggle
- `app/history/page.tsx` — past analysis list
- `app/plans/page.tsx` — plan comparison
- `components/InvestigationProgress.tsx` — SSE progress display with tool labels
- `components/AnalysisResultView.tsx` — result display, manipulation warning, feedback/report buttons
- `components/HypothesisTreeView.tsx` — Bayesian hypothesis tree visualization
- `components/MapView.tsx` — Leaflet map (no Mapbox, free tiles only)
- `lib/api.ts` — all API calls including auth, feedback, report download

### Auth & plans (`backend/app/models/user.py`, `api/v1/auth.py`)
- JWT-based auth; plans: `free`, `pro`, `enterprise`
- `PLAN_FEATURES` dict controls feature access (osint, manipulation, batch, api, report)
- Admin plan change: `POST /api/v1/auth/admin/set-plan` (requires `ADMIN_SECRET_KEY` header)
- Free plan blocks reverse image search in `tasks/analysis.py`

### Rate limiting (`backend/app/core/rate_limit.py`)
- Prefix matching sorted by length descending — longer prefixes take priority (e.g. `/analyze/batch` over `/analyze`)

### Celery schedules (`backend/app/core/celery_app.py`)
- `worker_process_init` signal: preloads YOLO + CLIP models on startup
- Daily 02:00: LoRA fine-tune trigger
- Daily 03:00: RLHF weights update (`daily-rlhf-weights`)
- Every 6h: VPR health check (`vpr-health-check`)

## Key Configuration

All settings in `backend/app/core/config.py` (`Settings` class, `extra="ignore"`), loaded from `.env`:

| Key | Purpose |
|---|---|
| `LLM_PROVIDER` | `claude` / `openai` / `ollama` / `groq` (default: `groq`) |
| `GROQ_API_KEY` / `GROQ_MODEL` | Groq LLM (currently active, default model `qwen/qwen3-32b`) |
| `ANTHROPIC_API_KEY` | Claude API (LLM_PROVIDER=claude 시) |
| `NAVER_CLIENT_ID/SECRET` | Naver Maps + Place + Blog + Image API |
| `KAKAO_API_KEY` | Kakao Maps + Vision API |
| `KAKAO_ACCESS_TOKEN` | Kakao OAuth Bearer (KakaoAK 403 폴백) |
| `JUSO_API_KEY` | 행정안전부 도로명주소 API |
| `PUBLIC_DATA_API_KEY` | 공공데이터포털 (사업자등록/문화재청) |
| `CLOVA_OCR_API_KEY` / `CLOVA_OCR_API_URL` | 네이버 CLOVA OCR (선택, 없으면 PaddleOCR 폴백) |
| `SERPAPI_KEY` | Web search (falls back to DuckDuckGo → Brave) |
| `SERP_API_KEY` | Google reverse image search in stage2 |
| `BRAVE_SEARCH_API_KEY` | Brave Search API (DuckDuckGo IP 차단 시 폴백) |
| `MAPILLARY_TOKEN` | Street view fetch (optional) |
| `FLICKR_API_KEY` | Flickr geo-tagged photo search |
| `BING_SEARCH_API_KEY` | Bing Visual Search in stage2 |
| `HF_TOKEN` | HuggingFace private model access (선택) |
| `NASA_API_KEY` | NASA API (기본값 `DEMO_KEY`) |
| `ADMIN_SECRET_KEY` | Header value for `POST /api/v1/auth/admin/set-plan` |
| `MAX_INVESTIGATION_STEPS` | ReAct loop max steps (default 8) |

## Important Patterns & Gotchas

- **Image storage in Redis**: always use `get_raw_redis()` (binary), never `get_redis()` (string) for image bytes
- **Async CPU work**: wrap all torch/YOLO/PaddleOCR/EasyOCR calls in `asyncio.to_thread(...)` to avoid blocking the event loop
- **EasyOCR languages**: only `["ko", "en"]` — do not add `ja` or `zh` (incompatible with `ko`)
- **GeoCLIP transformers 5.x**: `geoclip/model/image_encoder.py` is patched in `.venv` to handle `BaseModelOutputWithPooling.image_embeds`
- **Celery job_id injection**: `orch._job_id = job_id` must be set in `tasks/analysis.py` before calling the orchestrator
- **0.0 coordinate guard**: use `if ens.final_lon is not None:` not `if ens.final_lon:` — zero longitude is valid
- **qwen3-32b think tags**: `llm_provider.py` strips `<think>...</think>` with `re.sub`
- **macOS SSL**: `core/ssl_patch.py` must be imported before any HTTPS calls; it is imported at the top of `main.py` and `tasks/analysis.py`
- **macOS scipy BLAS hang**: `ssl_patch.py` sets `VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1` via `os.environ.setdefault()` — always launch Celery with these vars too
- **macOS uvloop**: if uvloop is installed, uvicorn blocks forever — run `rm -rf .venv/lib/python3.12/site-packages/uvloop*`; never use `pip install uvicorn[standard]`
- **macOS arm64 wheels**: pip may install x86_64 wheels for torchvision, jiter, aiohttp, frozenlist, multidict, yarl, propcache — replace manually with `arch -arm64 pip download --platform macosx_11_0_arm64`
- **Celery preload**: YOLO + CLIP models are preloaded on worker startup via `worker_process_init` signal in `celery_app.py`
- **Stage7 weights**: `_load_weights_from_redis()` loads RLHF-updated dynamic weights; falls back to static defaults if Redis key absent
- **Korea longitude validation**: stage7 rejects coordinates outside 124–132°E to filter GeoCLIP noise
- **Wayback CDX**: `_check_wayback(image_url)` searches by the actual catbox.moe URL, not by hash — call only after `_upload_to_temp_host` returns a URL
- **Web search priority**: `web_search()` → SerpAPI (if key) → DuckDuckGo → Brave Search API (if `BRAVE_SEARCH_API_KEY` set)
- **Parallel investigator**: When initial signals are weak, orchestrator spawns 2 investigators with different focus hints (`_focus: geoclip/ocr`) via `_parallel_investigate()`; timeout 150s
- **Milvus VPR seed**: `python -m app.data.seed_vpr_db` — run once after Milvus starts; `start.sh` does this automatically when collection is absent
- **PaddleOCR / EasyOCR**: incompatible with NumPy 2.x / current easyocr.model — both skip gracefully on import failure
- **Stage 5 cold start**: GeoCLIP first run can take ~6 minutes; models are cached after preload
