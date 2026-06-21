"""
수집 데이터 처리 파이프라인
1. 중복 제거 (pHash)
2. 품질 필터 (해상도, 블러)
3. 주소 정규화 (법정동 기준)
4. 학습 페어 생성 (image, text) — CLIP 학습용
5. HuggingFace Dataset 업로드
"""
import json
import math
import sys
from pathlib import Path
from typing import Iterator
from loguru import logger


# ── 중복 제거 ────────────────────────────────────────────

def compute_phash(image_path: Path) -> str | None:
    try:
        from PIL import Image
        import imagehash
        img = Image.open(image_path).convert("RGB")
        return str(imagehash.phash(img))
    except Exception:
        return None


def is_duplicate(phash: str, seen: dict[str, str], threshold: int = 8) -> bool:
    """해밍 거리 threshold 이하면 중복"""
    try:
        import imagehash
        h = imagehash.hex_to_hash(phash)
        for existing_hash in seen:
            if h - imagehash.hex_to_hash(existing_hash) <= threshold:
                return True
    except Exception:
        pass
    return False


# ── 품질 필터 ────────────────────────────────────────────

def check_quality(image_path: Path, min_size: int = 224, max_blur_threshold: float = 50.0) -> bool:
    """
    True = 품질 양호, False = 스킵
    - 최소 해상도: 224x224
    - 블러: Laplacian 분산 50 이상 (낮으면 블러)
    """
    try:
        import numpy as np
        from PIL import Image
        img = Image.open(image_path).convert("RGB")
        w, h = img.size
        if w < min_size or h < min_size:
            return False
        # 블러 감지
        import numpy as np
        gray = np.array(img.convert("L"), dtype=np.float32)
        lap_var = float(np.var(
            np.abs(gray[1:-1, 1:-1]*4 - gray[:-2, 1:-1] - gray[2:, 1:-1] - gray[1:-1, :-2] - gray[1:-1, 2:])
        ))
        return lap_var >= max_blur_threshold
    except Exception:
        return False


# ── 주소 정규화 ──────────────────────────────────────────

def normalize_address(address: str) -> dict:
    """
    주소에서 시/도, 시/군/구, 동/읍/면 추출
    반환: {"city": "서울특별시", "district": "서초구", "dong": "서초동", "full": "서울특별시 서초구 서초동"}
    """
    CITY_PATTERNS = [
        "서울특별시", "부산광역시", "대구광역시", "인천광역시", "광주광역시",
        "대전광역시", "울산광역시", "세종특별자치시", "경기도", "강원도",
        "강원특별자치도", "충청북도", "충청남도", "전라북도", "전북특별자치도",
        "전라남도", "경상북도", "경상남도", "제주특별자치도",
    ]
    result = {"city": "", "district": "", "dong": "", "full": address}
    addr = address.strip()

    for city in CITY_PATTERNS:
        if addr.startswith(city):
            result["city"] = city
            rest = addr[len(city):].strip()
            parts = rest.split()
            if parts:
                result["district"] = parts[0]
            if len(parts) > 1:
                result["dong"] = parts[1]
            break

    result["full"] = f"{result['city']} {result['district']} {result['dong']}".strip()
    return result


def _build_clip_text(meta: dict) -> list[str]:
    """
    (이미지, 텍스트) 페어용 텍스트 생성 — 다양한 표현으로 증강
    """
    texts = []
    addr = meta.get("address", "")
    name = meta.get("place_name", "")
    norm = normalize_address(addr)
    region = meta.get("region", "")

    if norm["full"]:
        texts.append(norm["full"])
    if norm["city"] and norm["district"]:
        texts.append(f"{norm['city']} {norm['district']}")
    if name:
        texts.append(f"{name}")
    if name and norm["city"]:
        texts.append(f"{norm['city']} {name}")
    if name and norm["district"]:
        texts.append(f"{norm['district']} {name}")
    if addr:
        texts.append(addr)
    if region and not any(region in t for t in texts):
        texts.append(region)

    # 빈 문자열 제거 + 중복 제거 (순서 유지)
    seen = set()
    unique = []
    for t in texts:
        t = t.strip()
        if t and t not in seen:
            seen.add(t)
            unique.append(t)
    return unique


# ── 메인 처리 파이프라인 ──────────────────────────────────

def process_dataset(
    data_dir: Path,
    output_dir: Path,
    max_samples: int = 500_000,
    test_ratio: float = 0.05,
) -> dict:
    """
    수집된 raw 데이터 → 학습/검증/테스트 셋 생성
    출력: output_dir/train.jsonl, val.jsonl, test.jsonl
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("데이터 처리 시작...")

    # 모든 메타데이터 파일 수집
    meta_files = list(data_dir.rglob("metadata.jsonl"))
    logger.info(f"메타 파일 {len(meta_files)}개 발견")

    # 1단계: 전체 목록 로드
    all_records = []
    for mf in meta_files:
        with open(mf) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    # 이미지 파일 존재 확인
                    if rec.get("image_path"):
                        img = data_dir / rec["image_path"]
                        if img.exists():
                            all_records.append(rec)
                except Exception:
                    pass
    logger.info(f"원본 레코드: {len(all_records)}개")

    # 2단계: 품질 필터 + 중복 제거
    seen_hashes: dict[str, str] = {}
    valid_records = []

    for i, rec in enumerate(all_records):
        if len(valid_records) >= max_samples:
            break

        img_path = data_dir / rec["image_path"]

        # 품질 체크
        if not check_quality(img_path):
            continue

        # 중복 체크
        phash = compute_phash(img_path)
        if phash is None:
            continue
        if is_duplicate(phash, seen_hashes):
            continue
        seen_hashes[phash] = rec["id"]

        # 텍스트 페어 생성
        texts = _build_clip_text(rec)
        if not texts:
            continue

        rec["phash"] = phash
        rec["clip_texts"] = texts
        rec["primary_text"] = texts[0]  # 학습 메인 텍스트
        valid_records.append(rec)

        if i % 5000 == 0:
            logger.info(f"  처리 중: {i}/{len(all_records)}, 유효: {len(valid_records)}")

    logger.info(f"품질 필터 후: {len(valid_records)}개")

    # 3단계: 지역 균형 맞추기 (특정 도시 과다 방지)
    from collections import defaultdict
    city_buckets: dict[str, list] = defaultdict(list)
    for rec in valid_records:
        city = normalize_address(rec.get("address", "")).get("city", "unknown")
        city_buckets[city].append(rec)

    # 각 도시 최대 비율 40% (서울 편향 방지)
    max_per_city = max(len(valid_records) // len(city_buckets), 1000)
    balanced = []
    for city, recs in city_buckets.items():
        balanced.extend(recs[:max_per_city])

    import random
    random.shuffle(balanced)
    logger.info(f"지역 균형 후: {len(balanced)}개 ({len(city_buckets)}개 도시)")

    # 4단계: 분할 (train/val/test)
    n = len(balanced)
    n_test = max(int(n * test_ratio), 500)
    n_val = max(int(n * test_ratio), 500)
    test_set = balanced[:n_test]
    val_set = balanced[n_test:n_test + n_val]
    train_set = balanced[n_test + n_val:]

    for split, records in [("train", train_set), ("val", val_set), ("test", test_set)]:
        out_path = output_dir / f"{split}.jsonl"
        with open(out_path, "w") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        logger.info(f"  {split}: {len(records)}개 → {out_path}")

    stats = {
        "total_raw": len(all_records),
        "after_quality_filter": len(valid_records),
        "after_balance": len(balanced),
        "train": len(train_set),
        "val": len(val_set),
        "test": len(test_set),
        "cities": {k: len(v) for k, v in city_buckets.items()},
    }
    with open(output_dir / "dataset_stats.json", "w") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    logger.info(f"데이터 처리 완료: train={len(train_set)}, val={len(val_set)}, test={len(test_set)}")
    return stats


def upload_to_huggingface(
    processed_dir: Path,
    data_raw_dir: Path,
    repo_id: str,
    hf_token: str,
    private: bool = True,
):
    """처리된 데이터셋을 HuggingFace Hub에 업로드"""
    try:
        from datasets import Dataset, DatasetDict, Image as HFImage
        from huggingface_hub import login, HfApi
        import io
        from PIL import Image
    except ImportError:
        logger.error("datasets, huggingface_hub 패키지 필요: pip install datasets huggingface_hub")
        return

    login(token=hf_token, add_to_git_credential=False)
    logger.info(f"HuggingFace 업로드 시작: {repo_id}")

    processed_dir = Path(processed_dir)
    data_raw_dir = Path(data_raw_dir)

    splits = {}
    for split in ["train", "val", "test"]:
        path = processed_dir / f"{split}.jsonl"
        if not path.exists():
            continue

        records = []
        with open(path) as f:
            for line in f:
                try:
                    records.append(json.loads(line))
                except Exception:
                    pass

        # HuggingFace Dataset 형식으로 변환
        hf_records = []
        for rec in records:
            img_path = data_raw_dir / rec["image_path"]
            if not img_path.exists():
                continue
            try:
                img = Image.open(img_path).convert("RGB")
                # 최대 512px로 리사이즈 (저장 공간 절약)
                img.thumbnail((512, 512), Image.LANCZOS)
                hf_records.append({
                    "image": img,
                    "text": rec["primary_text"],
                    "all_texts": rec.get("clip_texts", [rec["primary_text"]]),
                    "latitude": float(rec.get("latitude", 0)),
                    "longitude": float(rec.get("longitude", 0)),
                    "address": rec.get("address", ""),
                    "place_name": rec.get("place_name", ""),
                    "source": rec.get("source", ""),
                })
            except Exception as e:
                logger.debug(f"이미지 로드 실패: {img_path}: {e}")

        if hf_records:
            splits[split] = Dataset.from_list(hf_records)
            logger.info(f"  {split}: {len(hf_records)}개")

    if not splits:
        logger.error("업로드할 데이터 없음")
        return

    ds_dict = DatasetDict(splits)
    ds_dict.push_to_hub(repo_id, token=hf_token, private=private)
    logger.info(f"HuggingFace 업로드 완료: https://huggingface.co/datasets/{repo_id}")


if __name__ == "__main__":
    import os
    data_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("./ml_data")
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("./ml_data/processed")

    stats = process_dataset(data_dir, out_dir)
    print(json.dumps(stats, ensure_ascii=False, indent=2))

    hf_token = os.environ.get("HF_TOKEN", "")
    repo_id = os.environ.get("HF_DATASET_REPO", "exxas/korean-location-clips")
    if hf_token and "--upload" in sys.argv:
        upload_to_huggingface(out_dir, data_dir, repo_id, hf_token)
