"""
LoRA 경량 파인튜닝 스크립트
SelfLearningService.run_lora_finetune()에서 서브프로세스로 호출됨.

사용법:
  python ml/training/lora_finetune.py \
    --data '[{"job_id":"...", "label":"서울 강남구", "quality":"high"}]' \
    --epochs 3 \
    --output ml/models/checkpoint_20260601

환경 요구사항:
  - peft>=0.10.0
  - transformers>=4.42.0
  - torch>=2.2.0
  - GPU 권장 (MPS/CUDA 자동 감지)
"""
import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime


def main():
    parser = argparse.ArgumentParser(description="EXXAS LoRA 파인튜닝")
    parser.add_argument("--data", type=str, required=True, help="JSON 학습 데이터 (문자열)")
    parser.add_argument("--epochs", type=int, default=3, help="학습 에폭 수")
    parser.add_argument("--output", type=str, required=True, help="체크포인트 저장 경로")
    parser.add_argument("--base_model", type=str, default="geolocal/StreetCLIP", help="기반 모델")
    parser.add_argument("--lr", type=float, default=1e-4, help="학습률")
    parser.add_argument("--batch_size", type=int, default=4, help="배치 크기")
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank")
    args = parser.parse_args()

    # 데이터 파싱
    try:
        data = json.loads(args.data)
    except json.JSONDecodeError as e:
        print(f"[ERROR] JSON 파싱 실패: {e}", file=sys.stderr)
        sys.exit(1)

    if not data:
        print("[WARN] 학습 데이터 없음 — 종료", file=sys.stderr)
        sys.exit(0)

    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"[LoRA] 학습 시작: {len(data)}개 샘플, {args.epochs}에폭")
    print(f"[LoRA] 기반 모델: {args.base_model}")
    print(f"[LoRA] 출력 경로: {output_path}")

    # 장치 감지
    try:
        import torch
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        print(f"[LoRA] 장치: {device}")
    except ImportError:
        print("[ERROR] PyTorch 미설치", file=sys.stderr)
        sys.exit(1)

    # PEFT/Transformers 로드
    try:
        from transformers import CLIPModel, CLIPProcessor
        from peft import LoraConfig, get_peft_model, TaskType
        import torch.nn as nn
        from torch.utils.data import Dataset, DataLoader
    except ImportError as e:
        print(f"[ERROR] 의존성 미설치: {e}", file=sys.stderr)
        print("[INFO] pip install peft transformers", file=sys.stderr)
        sys.exit(1)

    # 모델 로드
    print(f"[LoRA] 모델 로딩...")
    try:
        processor = CLIPProcessor.from_pretrained(args.base_model)
        model = CLIPModel.from_pretrained(args.base_model)
    except Exception as e:
        print(f"[ERROR] 모델 로드 실패: {e}", file=sys.stderr)
        sys.exit(1)

    # LoRA 설정 적용 (vision encoder)
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_r * 2,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    model = model.to(device)

    # 학습 데이터셋 구성
    class GeoDataset(Dataset):
        def __init__(self, records, processor):
            self.records = [r for r in records if r.get("label")]
            self.processor = processor
            self._load_images()

        def _load_images(self):
            import redis
            from pathlib import Path
            import io
            from PIL import Image

            # Redis에서 이미지 바이트 로드 (job_id 기반)
            redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
            try:
                r = redis.from_url(redis_url, decode_responses=False)
            except Exception:
                r = None

            self.items = []
            for rec in self.records:
                job_id = rec.get("job_id", "")
                label = rec.get("label", "")
                img = None

                if r:
                    try:
                        img_bytes = r.get(f"job:{job_id}:image")
                        if img_bytes:
                            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                    except Exception:
                        pass

                if img is None:
                    # 이미지를 못 가져오면 빈 이미지로 대체 (degraded 학습)
                    img = Image.new("RGB", (224, 224), color=(128, 128, 128))

                self.items.append({"image": img, "label": label})

        def __len__(self):
            return len(self.items)

        def __getitem__(self, idx):
            item = self.items[idx]
            inputs = self.processor(
                images=item["image"],
                text=[item["label"]],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=77,
            )
            return {k: v.squeeze(0) for k, v in inputs.items()}

    dataset = GeoDataset(data, processor)
    if len(dataset) == 0:
        print("[WARN] 유효한 학습 샘플 없음 — 종료")
        sys.exit(0)

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    # 학습 루프
    optimizer = __import__("torch").optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
    )
    import torch

    model.train()
    for epoch in range(args.epochs):
        total_loss = 0.0
        for step, batch in enumerate(loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch, return_loss=True)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            if step % 10 == 0:
                print(f"[LoRA] Epoch {epoch+1}/{args.epochs} step {step} loss={loss.item():.4f}")

        avg_loss = total_loss / max(len(loader), 1)
        print(f"[LoRA] Epoch {epoch+1} 완료 — avg_loss={avg_loss:.4f}")

    # 저장
    model.save_pretrained(str(output_path))
    processor.save_pretrained(str(output_path))

    # 메타데이터 저장
    meta = {
        "base_model": args.base_model,
        "epochs": args.epochs,
        "samples": len(dataset),
        "completed_at": datetime.utcnow().isoformat(),
        "avg_final_loss": avg_loss,
        "device": device,
        "lora_r": args.lora_r,
    }
    with open(output_path / "training_meta.json", "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"[LoRA] 완료 → {output_path}")
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
