"""
한국 전용 학습 데이터 시딩 스크립트
===========================================
실행: cd backend && python scripts/seed_korea_training.py

작업:
1. Redis — 역명/랜드마크/대학/병원/행정구역 → GPS 즉시 조회 캐시
2. Redis — 브랜드 → 지역 정보 캐시
3. Milvus — 한국 도시 VPR 임베딩 (가상 임베딩으로 초기 시드)
4. 통계 출력
"""
import asyncio
import json
import sys
import os
import math
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Redis 연결 ─────────────────────────────────────────────

async def get_redis():
    import redis.asyncio as aioredis
    return await aioredis.from_url("redis://localhost:6379/0", decode_responses=True)


# ── 1. 지하철역 캐시 시딩 ─────────────────────────────────

async def seed_subway_stations(redis):
    from app.data.korea_stations_db import KR_SUBWAY_STATIONS
    pipe = redis.pipeline()
    for name, info in KR_SUBWAY_STATIONS.items():
        key = f"kr:station:{name}"
        value = json.dumps({
            "lat": info["lat"],
            "lon": info["lon"],
            "line": info.get("line", ""),
            "city": info.get("city", ""),
        }, ensure_ascii=False)
        pipe.set(key, value, ex=86400 * 365)  # 1년 TTL
        # 별칭도 등록 (역 이름 + 역)
        pipe.set(f"kr:station:{name}역", value, ex=86400 * 365)

    results = await pipe.execute()
    count = sum(1 for r in results if r)
    print(f"[Redis] 지하철역 캐시: {len(KR_SUBWAY_STATIONS)}역 → {count}개 키 등록")
    return len(KR_SUBWAY_STATIONS)


# ── 2. 랜드마크/대학/병원 캐시 시딩 ─────────────────────

async def seed_landmarks(redis):
    from app.data.korea_landmarks_db import KR_LANDMARKS, KR_UNIVERSITIES, KR_HOSPITALS, KR_DISTRICTS
    pipe = redis.pipeline()
    total = 0

    for name, info in KR_LANDMARKS.items():
        key = f"kr:landmark:{name}"
        value = json.dumps({
            "lat": info["lat"],
            "lon": info["lon"],
            "city": info.get("city", ""),
            "category": info.get("category", ""),
        }, ensure_ascii=False)
        pipe.set(key, value, ex=86400 * 365)
        total += 1
        # aliases도 등록
        for alias in info.get("aliases", []):
            if alias:
                pipe.set(f"kr:landmark:{alias}", value, ex=86400 * 365)
                total += 1

    for name, info in KR_UNIVERSITIES.items():
        key = f"kr:university:{name}"
        value = json.dumps({
            "lat": info["lat"], "lon": info["lon"],
            "city": info.get("city", ""), "category": "대학교",
        }, ensure_ascii=False)
        pipe.set(key, value, ex=86400 * 365)
        # landmark key에도 등록 (통합 조회)
        pipe.set(f"kr:landmark:{name}", value, ex=86400 * 365)
        total += 2
        for alias in info.get("aliases", []):
            if alias:
                pipe.set(f"kr:landmark:{alias}", value, ex=86400 * 365)
                total += 1

    for name, info in KR_HOSPITALS.items():
        key = f"kr:hospital:{name}"
        value = json.dumps({
            "lat": info["lat"], "lon": info["lon"],
            "city": info.get("city", ""), "category": "병원",
        }, ensure_ascii=False)
        pipe.set(key, value, ex=86400 * 365)
        pipe.set(f"kr:landmark:{name}", value, ex=86400 * 365)
        total += 2
        for alias in info.get("aliases", []):
            if alias:
                pipe.set(f"kr:landmark:{alias}", value, ex=86400 * 365)
                total += 1

    for name, info in KR_DISTRICTS.items():
        key = f"kr:district:{name}"
        value = json.dumps({
            "lat": info["lat"], "lon": info["lon"],
            "city": f"{info['city']} {name}", "category": "행정구역",
        }, ensure_ascii=False)
        pipe.set(key, value, ex=86400 * 365)
        total += 1

    results = await pipe.execute()
    registered = sum(1 for r in results if r)
    print(f"[Redis] 랜드마크/대학/병원/구 캐시: {total}개 키 등록 (새로운: {registered})")
    return total


# ── 3. 브랜드 캐시 시딩 ───────────────────────────────────

async def seed_brands(redis):
    from app.services.korea_specializer import ALL_KR_BRANDS
    pipe = redis.pipeline()
    for brand, info in ALL_KR_BRANDS.items():
        key = f"kr:brand:{brand}"
        value = json.dumps(info, ensure_ascii=False)
        pipe.set(key, value, ex=86400 * 365)

    results = await pipe.execute()
    count = sum(1 for r in results if r)
    print(f"[Redis] 브랜드 캐시: {len(ALL_KR_BRANDS)}개 → {count}개 키 등록")
    return len(ALL_KR_BRANDS)


# ── 4. 한국 지명 → 좌표 역조회 세트 생성 ─────────────────

async def seed_city_index(redis):
    """도시명 → 대표 좌표 매핑 (지오코딩 없이 즉시 도시 좌표 반환)"""
    CITY_COORDS = {
        "서울": (37.5665, 126.9780),
        "부산": (35.1796, 129.0756),
        "인천": (37.4563, 126.7052),
        "대구": (35.8714, 128.6014),
        "대전": (36.3504, 127.3845),
        "광주": (35.1595, 126.8526),
        "울산": (35.5384, 129.3114),
        "세종": (36.4800, 127.2890),
        "수원": (37.2636, 127.0286),
        "성남": (37.4201, 127.1266),
        "고양": (37.6584, 126.8320),
        "용인": (37.2341, 127.2031),
        "부천": (37.5034, 126.7660),
        "안산": (37.3219, 126.8309),
        "안양": (37.3943, 126.9568),
        "남양주": (37.6360, 127.2162),
        "화성": (37.1996, 126.8316),
        "평택": (36.9922, 127.1126),
        "의정부": (37.7382, 127.0339),
        "파주": (37.7600, 126.7798),
        "김포": (37.6152, 126.7158),
        "하남": (37.5392, 127.2148),
        "포항": (36.0190, 129.3435),
        "경주": (35.8414, 129.2112),
        "창원": (35.2342, 128.6811),
        "진주": (35.1798, 128.1076),
        "통영": (34.8545, 128.4330),
        "여수": (34.7604, 127.6622),
        "순천": (34.9506, 127.4872),
        "제주": (33.4996, 126.5312),
        "춘천": (37.8813, 127.7298),
        "원주": (37.3422, 127.9202),
        "강릉": (37.7515, 128.8761),
        "속초": (38.2044, 128.5912),
        "청주": (36.6424, 127.4890),
        "천안": (36.8151, 127.1139),
        "전주": (35.8242, 127.1480),
        "익산": (35.9483, 126.9547),
        "목포": (34.8118, 126.3922),
        "안동": (36.5684, 128.7294),
        "구미": (36.1195, 128.3446),
    }
    pipe = redis.pipeline()
    for city, (lat, lon) in CITY_COORDS.items():
        key = f"kr:city:{city}"
        value = json.dumps({"lat": lat, "lon": lon, "city": city}, ensure_ascii=False)
        pipe.set(key, value, ex=86400 * 365)

    results = await pipe.execute()
    count = sum(1 for r in results if r)
    print(f"[Redis] 도시 좌표 캐시: {len(CITY_COORDS)}개 → {count}개 키 등록")
    return len(CITY_COORDS)


# ── 5. Milvus VPR 시딩 ────────────────────────────────────

def _gaussian_embedding(lat: float, lon: float, dim: int = 512) -> list[float]:
    """
    좌표 기반 가상 임베딩 생성.
    동일 도시 이미지끼리 클러스터링되도록 lat/lon으로 시드 설정.
    """
    seed = int(lat * 1000) * 100000 + int(lon * 1000)
    rng = random.Random(seed)
    vec = [rng.gauss(lat / 90.0, 0.1) + rng.gauss(lon / 180.0, 0.1) for _ in range(dim)]
    # L2 정규화
    norm = math.sqrt(sum(x * x for x in vec))
    return [x / norm for x in vec] if norm > 0 else vec


async def seed_milvus_vpr():
    """한국 주요 위치 임베딩으로 Milvus VPR DB 초기 시드"""
    try:
        from pymilvus import MilvusClient
        client = MilvusClient(uri="http://localhost:19530")
        col = "image_embeddings"
        if not client.has_collection(col):
            print("[Milvus] 컬렉션 없음 — app 먼저 실행해서 스키마 생성 후 재시도")
            return 0

        # 시딩할 한국 위치 목록 (핵심 포인트들)
        from app.data.korea_stations_db import KR_SUBWAY_STATIONS
        from app.data.korea_landmarks_db import KR_LANDMARKS, KR_UNIVERSITIES

        seed_locations = []

        # 주요 지하철역 (각 역마다 3개 임베딩 = 다양한 각도)
        priority_stations = [
            "강남", "홍대입구", "이태원", "명동", "서울역", "잠실", "건대입구",
            "신촌", "한양대", "성수", "왕십리", "판교", "수서", "고속터미널",
            "광화문", "안국", "경복궁", "동대문역사문화공원", "합정", "뚝섬",
            "해운대", "서면", "부산역", "동대구", "반월당", "광주역", "대전역",
        ]
        for name in priority_stations:
            info = KR_SUBWAY_STATIONS.get(name)
            if info:
                # 반경 200m 내 3개 다른 앵글
                for offset_i, (dlat, dlon) in enumerate([
                    (0, 0), (0.001, 0.001), (-0.001, 0.002)
                ]):
                    seed_locations.append({
                        "location": f"{name}역 ({info['city']})",
                        "lat": info["lat"] + dlat,
                        "lon": info["lon"] + dlon,
                    })

        # 주요 랜드마크 (각 1개)
        priority_landmarks = [
            "롯데월드타워", "N서울타워", "코엑스", "경복궁", "DDP",
            "광화문광장", "해운대", "광안리", "성수동", "홍대",
            "전주한옥마을", "불국사", "수원화성", "인천국제공항",
        ]
        for name in priority_landmarks:
            info = KR_LANDMARKS.get(name)
            if info:
                seed_locations.append({
                    "location": f"{name} ({info['city']})",
                    "lat": info["lat"],
                    "lon": info["lon"],
                })

        # 임베딩 생성 및 삽입
        import hashlib
        batch = []
        for loc in seed_locations:
            emb = _gaussian_embedding(loc["lat"], loc["lon"])
            img_hash = hashlib.md5(f"{loc['location']}_{loc['lat']:.4f}".encode()).hexdigest()
            batch.append({
                "image_hash": img_hash,
                "latitude": float(loc["lat"]),
                "longitude": float(loc["lon"]),
                "location": loc["location"][:128],
                "embedding": emb,
            })

        # 배치 삽입 (50개씩)
        inserted = 0
        for i in range(0, len(batch), 50):
            chunk = batch[i:i+50]
            try:
                client.insert(col, chunk)
                inserted += len(chunk)
            except Exception as e:
                print(f"  [Milvus] 배치 {i//50+1} 삽입 오류: {e}")

        print(f"[Milvus] VPR 임베딩 시딩: {inserted}/{len(seed_locations)}개 삽입")
        return inserted

    except Exception as e:
        print(f"[Milvus] 연결 실패 (비필수): {e}")
        return 0


# ── 6. 검색 테스트 ────────────────────────────────────────

async def test_lookups(redis):
    print("\n[테스트] Redis 조회 확인")
    tests = [
        ("kr:station:강남", "강남역"),
        ("kr:station:홍대입구역", "홍대입구역"),
        ("kr:landmark:경복궁", "경복궁"),
        ("kr:landmark:연세대", "연세대"),
        ("kr:landmark:세브란스병원", "세브란스병원"),
        ("kr:district:강남구", "강남구"),
        ("kr:city:부산", "부산"),
    ]
    ok = 0
    for key, label in tests:
        val = await redis.get(key)
        if val:
            data = json.loads(val)
            print(f"  ✓ {label}: ({data.get('lat'):.4f}, {data.get('lon'):.4f}) {data.get('city','')}")
            ok += 1
        else:
            print(f"  ✗ {label}: 미등록")
    print(f"  결과: {ok}/{len(tests)} 성공")


# ── 메인 ─────────────────────────────────────────────────

async def main():
    print("=" * 60)
    print("EXXAS 한국 학습 데이터 시딩 시작")
    print("=" * 60)

    redis = await get_redis()
    try:
        n1 = await seed_subway_stations(redis)
        n2 = await seed_landmarks(redis)
        n3 = await seed_brands(redis)
        n4 = await seed_city_index(redis)
        n5 = await seed_milvus_vpr()

        await test_lookups(redis)

        print("\n" + "=" * 60)
        print(f"시딩 완료:")
        print(f"  지하철역:    {n1}개")
        print(f"  랜드마크/구: {n2}개 키")
        print(f"  브랜드:      {n3}개")
        print(f"  도시:        {n4}개")
        print(f"  Milvus VPR:  {n5}개 임베딩")
        print("=" * 60)
    finally:
        await redis.aclose()


if __name__ == "__main__":
    asyncio.run(main())
