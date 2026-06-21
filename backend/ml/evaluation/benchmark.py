"""
모델 평가 벤치마크
- 기존 모델 vs 신규 모델 비교
- 지역 수준별 정확도: 시/도, 시/군/구, 동
- Top-1 / Top-5 retrieval accuracy
- 배포 판단: new > baseline + MIN_IMPROVEMENT 조건
"""
import json
import sys
from pathlib import Path
from typing import Any
from loguru import logger

# 배포 허용 최소 개선폭
MIN_IMPROVEMENT_DISTRICT = 0.05   # 구/군 수준에서 5%p 이상 향상 필요
MIN_IMPROVEMENT_CITY = 0.03       # 시/도 수준에서 3%p 이상


def _extract_city_district(address: str) -> tuple[str, str]:
    """주소에서 시/도, 구/군 추출"""
    CITIES = [
        "서울특별시", "부산광역시", "대구광역시", "인천광역시", "광주광역시",
        "대전광역시", "울산광역시", "세종특별자치시", "경기도", "강원도",
        "강원특별자치도", "충청북도", "충청남도", "전라북도", "전북특별자치도",
        "전라남도", "경상북도", "경상남도", "제주특별자치도",
    ]
    city, district = "", ""
    for c in CITIES:
        if address.startswith(c):
            city = c
            rest = address[len(c):].strip().split()
            if rest:
                district = rest[0]
            break
    return city, district


class ModelBenchmark:
    def __init__(self, model_name: str, device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self._model = None
        self._processor = None

    def load(self):
        from transformers import CLIPModel, CLIPProcessor
        logger.info(f"모델 로딩: {self.model_name}")
        self._processor = CLIPProcessor.from_pretrained(self.model_name)
        self._model = CLIPModel.from_pretrained(self.model_name).to(self.device)
        self._model.eval()
        logger.info("모델 로딩 완료")

    def encode_images(self, image_paths: list[Path]) -> "torch.Tensor":
        import torch
        from PIL import Image
        embeddings = []
        batch_size = 32
        for i in range(0, len(image_paths), batch_size):
            batch = image_paths[i:i + batch_size]
            images = []
            for p in batch:
                try:
                    images.append(Image.open(p).convert("RGB"))
                except Exception:
                    images.append(Image.new("RGB", (224, 224)))
            inputs = self._processor(images=images, return_tensors="pt", padding=True).to(self.device)
            with torch.no_grad():
                feats = self._model.get_image_features(**inputs)
                feats = feats / feats.norm(dim=-1, keepdim=True)
            embeddings.append(feats.cpu())
        return torch.cat(embeddings, dim=0)

    def encode_texts(self, texts: list[str]) -> "torch.Tensor":
        import torch
        embeddings = []
        batch_size = 64
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            inputs = self._processor(text=batch, return_tensors="pt", padding=True, truncation=True, max_length=77).to(self.device)
            with torch.no_grad():
                feats = self._model.get_text_features(**inputs)
                feats = feats / feats.norm(dim=-1, keepdim=True)
            embeddings.append(feats.cpu())
        return torch.cat(embeddings, dim=0)


def run_benchmark(
    test_jsonl: Path,
    data_dir: Path,
    model_name: str,
    device: str = "cpu",
    max_samples: int = 2000,
) -> dict:
    """
    테스트셋에서 image → text retrieval 정확도 측정
    반환: {"city_top1", "city_top5", "district_top1", "district_top5", "model": model_name}
    """
    import torch

    # 테스트 데이터 로드
    records = []
    with open(test_jsonl) as f:
        for line in f:
            try:
                records.append(json.loads(line))
            except Exception:
                pass
    records = records[:max_samples]
    logger.info(f"벤치마크 샘플: {len(records)}개")

    image_paths = [data_dir / r["image_path"] for r in records]
    texts = [r["primary_text"] for r in records]

    bm = ModelBenchmark(model_name, device)
    bm.load()

    logger.info("이미지 임베딩 생성...")
    img_emb = bm.encode_images(image_paths)
    logger.info("텍스트 임베딩 생성...")
    txt_emb = bm.encode_texts(texts)

    # 유사도 행렬 (N x N)
    similarity = img_emb @ txt_emb.T  # (N, N)

    # Top-K 정확도 계산
    def topk_accuracy(sim: "torch.Tensor", k: int, level: str) -> float:
        n = sim.shape[0]
        correct = 0
        topk_indices = sim.topk(k, dim=1).indices  # (N, k)
        for i, record in enumerate(records):
            gt_city, gt_district = _extract_city_district(record.get("address", ""))
            for j in topk_indices[i].tolist():
                candidate = records[j]
                cand_city, cand_district = _extract_city_district(candidate.get("address", ""))
                if level == "city" and gt_city and gt_city == cand_city:
                    correct += 1
                    break
                elif level == "district" and gt_district and gt_district == cand_district:
                    correct += 1
                    break
        return correct / n

    results = {
        "model": model_name,
        "n_samples": len(records),
        "city_top1": round(topk_accuracy(similarity, 1, "city"), 4),
        "city_top5": round(topk_accuracy(similarity, 5, "city"), 4),
        "district_top1": round(topk_accuracy(similarity, 1, "district"), 4),
        "district_top5": round(topk_accuracy(similarity, 5, "district"), 4),
    }
    logger.info(f"벤치마크 결과 [{model_name}]:")
    logger.info(f"  City   Top-1: {results['city_top1']:.1%}  Top-5: {results['city_top5']:.1%}")
    logger.info(f"  District Top-1: {results['district_top1']:.1%}  Top-5: {results['district_top5']:.1%}")
    return results


def compare_and_decide(
    baseline_results: dict,
    new_results: dict,
    save_path: Path | None = None,
) -> dict:
    """
    기존 모델 vs 신규 모델 비교 → 배포 여부 결정
    반환: {"deploy": True/False, "reason": str, "improvements": dict}
    """
    improvements = {
        "city_top1": new_results["city_top1"] - baseline_results["city_top1"],
        "city_top5": new_results["city_top5"] - baseline_results["city_top5"],
        "district_top1": new_results["district_top1"] - baseline_results["district_top1"],
        "district_top5": new_results["district_top5"] - baseline_results["district_top5"],
    }

    passed_district = improvements["district_top5"] >= MIN_IMPROVEMENT_DISTRICT
    passed_city = improvements["city_top5"] >= MIN_IMPROVEMENT_CITY
    no_regression = improvements["city_top1"] >= -0.02  # 시/도 Top-1 2%p 이상 하락 금지

    deploy = passed_district and passed_city and no_regression

    reasons = []
    if passed_district:
        reasons.append(f"구/군 Top-5 +{improvements['district_top5']:.1%} (기준: +{MIN_IMPROVEMENT_DISTRICT:.0%})")
    else:
        reasons.append(f"구/군 Top-5 향상 부족: {improvements['district_top5']:.1%} (기준: +{MIN_IMPROVEMENT_DISTRICT:.0%})")
    if passed_city:
        reasons.append(f"시/도 Top-5 +{improvements['city_top5']:.1%}")
    else:
        reasons.append(f"시/도 Top-5 향상 부족: {improvements['city_top5']:.1%}")
    if not no_regression:
        reasons.append(f"시/도 Top-1 하락: {improvements['city_top1']:.1%} (회귀 감지)")

    decision = {
        "deploy": deploy,
        "reason": " | ".join(reasons),
        "improvements": {k: round(v, 4) for k, v in improvements.items()},
        "baseline": baseline_results,
        "new_model": new_results,
        "thresholds": {
            "min_district_top5": MIN_IMPROVEMENT_DISTRICT,
            "min_city_top5": MIN_IMPROVEMENT_CITY,
        },
    }

    logger.info(f"=== 배포 결정: {'✅ 배포 승인' if deploy else '❌ 배포 거부'} ===")
    logger.info(f"이유: {decision['reason']}")
    for k, v in improvements.items():
        sign = "+" if v >= 0 else ""
        logger.info(f"  {k}: {sign}{v:.1%}")

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(decision, f, ensure_ascii=False, indent=2)

    return decision


if __name__ == "__main__":
    # 사용법: python -m ml.evaluation.benchmark <test.jsonl> <data_dir> <model_name>
    if len(sys.argv) < 4:
        print("Usage: python benchmark.py <test.jsonl> <data_dir> <model_name> [baseline_model]")
        sys.exit(1)

    test_file = Path(sys.argv[1])
    data_directory = Path(sys.argv[2])
    new_model = sys.argv[3]
    baseline = sys.argv[4] if len(sys.argv) > 4 else "openai/clip-vit-large-patch14"

    logger.info(f"기존 모델 벤치마크: {baseline}")
    base_res = run_benchmark(test_file, data_directory, baseline)

    logger.info(f"신규 모델 벤치마크: {new_model}")
    new_res = run_benchmark(test_file, data_directory, new_model)

    decision = compare_and_decide(base_res, new_res, save_path=Path("ml_data/benchmark_result.json"))
    print(json.dumps(decision, ensure_ascii=False, indent=2))
    sys.exit(0 if decision["deploy"] else 1)
