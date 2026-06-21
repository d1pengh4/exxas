# EXXAS v2.0 — 개발 계획서

## 전체 구조

```
EXXAS/
├── backend/          # FastAPI + Celery
├── frontend/         # Next.js + Tailwind
├── ml/               # ML 모델 학습/관리
├── docker-compose.yml
└── .env.example
```

---

## Phase 1 — 기반 인프라 (M1~2)
- [ ] Docker Compose 환경 (PostgreSQL+PostGIS, Redis, Milvus, Neo4j)
- [ ] FastAPI 앱 골격 (config, DB 연결, 헬스체크)
- [ ] 사용자 인증 (JWT)
- [ ] 이미지 업로드 API + S3/로컬 스토리지
- [ ] Stage 0: 전처리 (해시, ELA, 품질 평가)
- [ ] Stage 1: EXIF 포렌식 완전 구현
- [ ] Next.js 프론트엔드 기본 UI (업로드 + 결과)

## Phase 2 — LLM 수사관 코어 (M2~3)
- [ ] Anthropic Claude API 연동 (로컬 LLM 교체 가능 인터페이스)
- [ ] ReAct 루프 엔진 (OBSERVE→THINK→HYPOTHESIZE→ACT→EVALUATE)
- [ ] 프롬프트 3층 아키텍처 (정체성/상태/행동)
- [ ] 가설 트리 엔진 (베이지안 업데이트, 가지치기, 분기)
- [ ] Tool Calling 프레임워크 (도구 등록/실행/결과 피드백)
- [ ] Hallucination 방지 4종 메커니즘

## Phase 3 — AI 코어 통합 (M3~4)
- [ ] Stage 5: GeoCLIP + StreetCLIP 임베딩 (로컬 CPU/GPU)
- [ ] VPR: CosPlace, NetVLAD, SelaVPR
- [ ] Stage 4: YOLOv12 인프라 핑거프린팅 (170개국 DB)
- [ ] Stage 3: PaddleOCR 다국어 + GIS 매핑 (네이버/카카오/Google)
- [ ] Milvus 벡터 DB 연동

## Phase 4 — OSINT 레이어 (M4~5)
- [ ] Stage 2: 역방향 이미지 검색 5종 병렬 (Google/Yandex/TinEye/Baidu/Bing)
- [ ] Wayback Machine CDX API
- [ ] 소셜 크롤러 (네이버 블로그/카페, Flickr, Twitter)
- [ ] 지도 API 교차검증 (네이버/카카오/Google Places/OSM)
- [ ] GDELT 뉴스 이미지 인덱스
- [ ] 단서 체인 추적 자동화

## Phase 5 — 물리·천문 역산 (M5~6)
- [ ] Stage 6: SunCalc 태양 고도/그림자 역산
- [ ] DEM 능선 매칭 (NASA SRTM + Copernicus)
- [ ] 달·별 분석 (Stellarium API, SunCalc)
- [ ] ERA5 기상 DB 교차검증
- [ ] 식생 계절 분석

## Phase 6 — Agentic 통합 완성 (M6~7)
- [ ] 탐색 모드 3종 (속전속결/포위축소/귀납추론)
- [ ] 충돌 해소 프로토콜
- [ ] 탐색 종료 조건
- [ ] Explainable AI 리포트 생성
- [ ] 자기 평가 시스템 (이중 검증 루프)

## Phase 7 — 앙상블 & 신뢰도 엔진 (M7~8)
- [ ] 신뢰도 상한 테이블
- [ ] 독립 소스 수렴 보너스
- [ ] 반증 패널티
- [ ] Bayesian Fusion (동적 가중치)
- [ ] 입력 유형 자동 분류 (도시/자연/야간/실내/SNS압축)

## Phase 8 — 자가학습 파이프라인 (M8~9)
- [ ] RLHF 피드백 수집 DB
- [ ] 야간 배치 재학습 스케줄러
- [ ] A/B 테스트 프레임워크
- [ ] Active Learning (최저 신뢰도 케이스 우선)
- [ ] Neo4j 지식 그래프 자동 생성

## Phase 9 — Beta & 정식 출시 (M9~12)
- [ ] 보안 아키텍처 (Zone A/B/C 격리)
- [ ] Freemium 구독 결제 시스템
- [ ] B2B API 라이선스 포털
- [ ] 클로즈드 베타 300명
- [ ] 성능 최적화 (캐시 히트율, 배치 처리)

---

## LLM 전략

| 단계 | LLM | 이유 |
|------|-----|------|
| 개발~MVP | Claude API (claude-sonnet-4-6) | 저비용, 고성능, 즉시 시작 가능 |
| 스케일업 | Ollama (Qwen2.5-VL-7B/14B) | 로컬 실행, API 비용 0 |
| 프로덕션 | Qwen2.5-VL-72B (서버) | 최고 성능, 온프레미스 |

인터페이스 통일: `LLMProvider` 추상 클래스 → 교체 시 코드 변경 없음

---

## 현재 진행 상태

- [x] 기획서 v2.0 완성
- [ ] Phase 1 시작
