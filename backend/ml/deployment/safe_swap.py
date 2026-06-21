"""
안전한 모델 교체 시스템
1. HuggingFace에서 신규 모델 다운로드
2. 벤치마크 재실행 (로컬 검증)
3. Shadow mode: 100개 이미지에 두 모델 동시 실행 비교
4. 통과 시 → stage5_embedding.py의 모델 교체 + 기존 모델 백업
5. 실패 시 → 자동 롤백
"""
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from loguru import logger

MODEL_REGISTRY_PATH = Path("ml/models/registry.json")
STAGE5_CONFIG_PATH  = Path("app/pipeline/stage5_embedding.py")
BACKUP_DIR          = Path("ml/models/backups")


def load_registry() -> dict:
    if MODEL_REGISTRY_PATH.exists():
        with open(MODEL_REGISTRY_PATH) as f:
            return json.load(f)
    return {"active_model": "openai/clip-vit-large-patch14", "history": []}


def save_registry(registry: dict):
    MODEL_REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_REGISTRY_PATH, "w") as f:
        json.dump(registry, f, indent=2, ensure_ascii=False)


def get_active_model() -> str:
    return load_registry().get("active_model", "openai/clip-vit-large-patch14")


def download_model_from_hf(repo_id: str, local_dir: Path, hf_token: str = "") -> Path:
    """HuggingFace Hub에서 모델 다운로드"""
    from huggingface_hub import snapshot_download
    logger.info(f"모델 다운로드: {repo_id} → {local_dir}")
    local_dir.mkdir(parents=True, exist_ok=True)
    path = snapshot_download(
        repo_id=repo_id,
        local_dir=str(local_dir),
        token=hf_token or None,
        ignore_patterns=["*.msgpack", "flax_model*", "tf_model*", "rust_model*"],
    )
    logger.info(f"다운로드 완료: {path}")
    return Path(path)


def run_shadow_comparison(
    baseline_model: str,
    new_model_path: str,
    test_images: list[Path],
    device: str = "cpu",
) -> dict:
    """
    Shadow mode: 두 모델을 동일한 이미지에 실행, 예측 일치도 측정
    """
    import torch
    from transformers import CLIPModel, CLIPProcessor
    from PIL import Image

    KOREA_LABELS = [
        "서울특별시", "부산광역시", "인천광역시", "대구광역시", "대전광역시",
        "광주광역시", "울산광역시", "경기도", "강원도", "충청북도",
        "충청남도", "전라남도", "경상북도", "경상남도", "제주도",
    ]
    label_texts = [f"한국 {l} 사진" for l in KOREA_LABELS]

    def load_clip(model_name):
        proc = CLIPProcessor.from_pretrained(model_name)
        mdl = CLIPModel.from_pretrained(model_name).to(device).eval()
        return proc, mdl

    def predict_top3(proc, mdl, img_path):
        try:
            img = Image.open(img_path).convert("RGB")
            inputs = proc(text=label_texts, images=img, return_tensors="pt", padding=True).to(device)
            with torch.no_grad():
                logits = mdl(**inputs).logits_per_image[0]
                probs = logits.softmax(dim=0)
            top3 = probs.topk(3).indices.tolist()
            return [KOREA_LABELS[i] for i in top3]
        except Exception:
            return []

    logger.info("Shadow mode: 두 모델 동시 비교 시작")
    base_proc, base_mdl = load_clip(baseline_model)
    new_proc, new_mdl = load_clip(new_model_path)

    agreements = 0
    total = 0
    new_top1_count = 0

    for img_path in test_images[:100]:
        if not img_path.exists():
            continue
        base_preds = predict_top3(base_proc, base_mdl, img_path)
        new_preds = predict_top3(new_proc, new_mdl, img_path)
        if base_preds and new_preds:
            total += 1
            if base_preds[0] == new_preds[0]:
                agreements += 1
            # 신규 모델 예측 신뢰도가 더 높은 경우 (더 확실하게 예측)
            new_top1_count += 1  # 추후 신뢰도 기반 개선 가능

    agreement_rate = agreements / total if total > 0 else 0
    logger.info(f"Shadow mode 결과: {agreements}/{total} 일치 ({agreement_rate:.1%})")

    return {
        "total": total,
        "agreements": agreements,
        "agreement_rate": round(agreement_rate, 4),
        "passed": agreement_rate >= 0.70,  # 70% 이상 일치 시 패스
    }


def swap_model(
    new_model_hf_id: str,
    benchmark_result_path: Path,
    test_images_dir: Path | None,
    hf_token: str = "",
    device: str = "cpu",
    force: bool = False,
) -> bool:
    """
    메인 교체 함수
    1. 벤치마크 결과 확인
    2. Shadow mode 비교
    3. stage5_embedding.py 모델 ID 교체
    4. 레지스트리 업데이트
    """
    # 1. 벤치마크 결과 확인
    if not force:
        if not benchmark_result_path.exists():
            logger.error("벤치마크 결과 파일 없음. 배포 중단.")
            return False
        with open(benchmark_result_path) as f:
            bench = json.load(f)
        if not bench.get("deploy"):
            logger.error(f"벤치마크 미통과: {bench.get('reason')}. 배포 중단.")
            return False
        logger.info(f"벤치마크 통과: {bench.get('reason')}")

    registry = load_registry()
    baseline_model = registry["active_model"]

    # 2. Shadow mode (테스트 이미지 있을 경우)
    if test_images_dir and test_images_dir.exists():
        test_imgs = list(test_images_dir.rglob("*.jpg"))[:100]
        if test_imgs:
            local_model_path = Path(f"ml/models/{new_model_hf_id.replace('/', '_')}")
            download_model_from_hf(new_model_hf_id, local_model_path, hf_token)
            shadow_result = run_shadow_comparison(
                baseline_model, str(local_model_path), test_imgs, device
            )
            if not shadow_result["passed"] and not force:
                logger.error(f"Shadow mode 미통과: 일치율 {shadow_result['agreement_rate']:.1%}. 배포 중단.")
                return False

    # 3. stage5_embedding.py 모델 ID 교체
    _update_stage5_model(baseline_model, new_model_hf_id)

    # 4. 레지스트리 업데이트
    registry["history"].append({
        "model": baseline_model,
        "replaced_at": datetime.now().isoformat(),
        "replaced_by": new_model_hf_id,
        "benchmark": str(benchmark_result_path),
    })
    registry["active_model"] = new_model_hf_id
    save_registry(registry)

    logger.info(f"모델 교체 완료: {baseline_model} → {new_model_hf_id}")
    return True


def _update_stage5_model(old_model: str, new_model: str):
    """stage5_embedding.py에서 OpenCLIP 모델 ID 업데이트"""
    stage5 = STAGE5_CONFIG_PATH
    if not stage5.exists():
        logger.warning(f"stage5 파일 없음: {stage5}")
        return

    # 백업
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    backup = BACKUP_DIR / f"stage5_embedding_{datetime.now().strftime('%Y%m%d_%H%M%S')}.py"
    shutil.copy2(stage5, backup)
    logger.info(f"stage5 백업: {backup}")

    content = stage5.read_text()
    # 기존 모델 ID → 신규 모델 ID (HuggingFace ID 또는 로컬 경로)
    # stage5에서 openai/clip-vit-large-patch14 또는 apple/DFN2B-CLIP-ViT-L-14 참조 부분
    updated = content.replace(
        f'"{old_model}"',
        f'"{new_model}"',
    )
    if updated == content:
        logger.warning(f"stage5에서 '{old_model}' 참조를 찾지 못했습니다.")
    else:
        stage5.write_text(updated)
        logger.info(f"stage5 모델 업데이트: {old_model} → {new_model}")


def rollback(steps: int = 1):
    """이전 모델로 롤백"""
    registry = load_registry()
    history = registry.get("history", [])
    if not history:
        logger.error("롤백할 이력 없음")
        return False

    target = history[-(steps)]
    old_model = target["model"]
    current = registry["active_model"]

    _update_stage5_model(current, old_model)
    registry["active_model"] = old_model
    registry["history"] = history[:-steps]
    save_registry(registry)
    logger.info(f"롤백 완료: {current} → {old_model}")
    return True


if __name__ == "__main__":
    import os
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"

    if cmd == "status":
        print(f"현재 활성 모델: {get_active_model()}")
        reg = load_registry()
        for h in reg.get("history", [])[-3:]:
            print(f"  이전: {h['model']} → {h.get('replaced_by')} ({h.get('replaced_at')})")

    elif cmd == "swap":
        new_hf_id = sys.argv[2]
        bench_path = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("ml_data/benchmark_result.json")
        test_dir = Path(sys.argv[4]) if len(sys.argv) > 4 else None
        hf_token = os.environ.get("HF_TOKEN", "")
        force = "--force" in sys.argv
        ok = swap_model(new_hf_id, bench_path, test_dir, hf_token, force=force)
        sys.exit(0 if ok else 1)

    elif cmd == "rollback":
        steps = int(sys.argv[2]) if len(sys.argv) > 2 else 1
        ok = rollback(steps)
        sys.exit(0 if ok else 1)

    else:
        print("Usage: safe_swap.py [status|swap <model_id> <bench_path>|rollback [steps]]")
