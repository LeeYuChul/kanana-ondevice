"""
3단계: TFLite → .task 패키징 (MediaPipe LlmInference 포맷)
MediaPipe .task 파일명 규칙에 맞게 수정됨.
"""

import argparse
import json
import logging
import sys
import zipfile
from pathlib import Path
from datetime import datetime

SCRIPT_DIR   = Path(__file__).parent
ONDEVICE_DIR = SCRIPT_DIR.parent
LOG_DIR      = ONDEVICE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

log_file = LOG_DIR / f"bundle_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_file, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)


def load_export_meta(quant: str) -> dict:
    meta_path = ONDEVICE_DIR / "outputs" / quant / "export_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"export_meta.json 없음: {meta_path}")
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    log.info(f"메타데이터 로드: {meta_path}")
    return meta


def find_tokenizer_model(tokenizer_path: Path) -> Path | None:
    """SentencePiece 또는 tokenizer.json 파일 찾기"""
    tok_dir = Path(tokenizer_path)
    for pattern in ("*.spm", "*.model", "tokenizer.model"):
        matches = list(tok_dir.glob(pattern))
        if matches:
            return matches[0]
    tj = tok_dir / "tokenizer.json"
    if tj.exists():
        return tj
    return None


def bundle_task(meta: dict, output_path: Path):
    """
    MediaPipe LlmInference .task 포맷:
    ├── TF_LITE_PREFILL_DECODE   (메인 모델)
    ├── TOKENIZER_MODEL          (토크나이저)
    └── METADATA                 (메타데이터 JSON)
    """
    tflite_path = Path(meta["tflite_path"])
    tokenizer_path = Path(meta["tokenizer_path"])

    if not tflite_path.exists():
        raise FileNotFoundError(f"TFLite 파일 없음: {tflite_path}")

    tok_file = find_tokenizer_model(tokenizer_path)

    metadata = {
        "model_type": "LLM_CAUSAL",
        "model_id": meta["model_id"],
        "quant_mode": meta["quant_mode"],
        "start_token_id": meta.get("bos_token_id"),
        "stop_token_ids": [meta.get("eos_token_id")],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # ZIP_STORED (비압축) 필수!
    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_STORED) as zf:
        # 메인 모델 → TF_LITE_PREFILL_DECODE (확장자 없음)
        zf.write(tflite_path, "TF_LITE_PREFILL_DECODE")
        log.info(f"  ➕ TF_LITE_PREFILL_DECODE ({tflite_path.stat().st_size/1e6:.1f} MB)")

        # 토크나이저 → TOKENIZER_MODEL (확장자 없음)
        if tok_file and tok_file.exists():
            zf.write(tok_file, "TOKENIZER_MODEL")
            log.info(f"  ➕ TOKENIZER_MODEL ({tok_file.name})")
        else:
            log.warning("⚠️  토크나이저 파일 없음")

        # 메타데이터 → METADATA (확장자 없음)
        zf.writestr("METADATA", json.dumps(metadata, ensure_ascii=False, indent=2))
        log.info("  ➕ METADATA")

    size_mb = output_path.stat().st_size / 1e6
    log.info(f"✅ .task 생성 완료: {output_path} ({size_mb:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(description="TFLite → .task 패키징")
    parser.add_argument("--quant", choices=["int8", "int4"], required=True)
    args = parser.parse_args()

    meta = load_export_meta(args.quant)
    output_dir = ONDEVICE_DIR / "outputs" / args.quant
    task_path = output_dir / f"kanana_{args.quant}.task"
    
    log.info(f"\n[.task 패키징 시작] → {task_path}")
    bundle_task(meta, task_path)
    log.info(f"\n로그 저장: {log_file}")


if __name__ == "__main__":
    main()
