"""
4단계: 변환 산출물 검증 스크립트
bundle_task.py 이후 실행.

검증 항목:
  1. 파일 생성 여부 + 파일 크기
  2. ZIP 구조 (.task / .litertlm)
  3. 메타데이터 내용 (bos/eos token)
  4. TFLite 인터프리터 로드 가능 여부
  5. 샘플 추론 실행 가능 여부

사용법:
  python verify_artifact.py --quant int8
  python verify_artifact.py --quant int4
  python verify_artifact.py --quant int8 --format litertlm
"""

import argparse
import json
import logging
import sys
import tempfile
import zipfile
from pathlib import Path
from datetime import datetime

SCRIPT_DIR   = Path(__file__).parent
ONDEVICE_DIR = SCRIPT_DIR.parent
LOG_DIR      = ONDEVICE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

log_file = LOG_DIR / f"verify_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_file, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)


def section(title: str):
    log.info("")
    log.info("─" * 55)
    log.info(f"  {title}")
    log.info("─" * 55)


# ────────────────────────────────────────────────────────────
# CHECK 1: 파일 존재 + 크기
# ────────────────────────────────────────────────────────────
def check_file_exists(path: Path) -> bool:
    section(f"CHECK 1: 파일 존재 확인")
    if path.exists():
        size_mb = path.stat().st_size / 1e6
        log.info(f"✅ 파일 존재: {path}")
        log.info(f"   크기: {size_mb:.1f} MB")
        if size_mb < 1:
            log.warning("⚠️  파일이 너무 작음 — 변환 실패일 수 있음")
            return False
        return True
    else:
        log.error(f"❌ 파일 없음: {path}")
        return False


# ────────────────────────────────────────────────────────────
# CHECK 2: ZIP 구조 확인
# ────────────────────────────────────────────────────────────
def check_zip_structure(path: Path, fmt: str) -> dict:
    section("CHECK 2: ZIP 내부 구조")
    result = {"model_tflite": False, "tokenizer": False, "metadata": False}

    try:
        with zipfile.ZipFile(path, "r") as zf:
            names = zf.namelist()
            log.info(f"ZIP 내 파일 목록:")
            for n in names:
                info = zf.getinfo(n)
                log.info(f"  {n:<40} {info.file_size/1e6:.2f} MB")

            result["model_tflite"] = "model.tflite" in names
            result["tokenizer"] = any("tokenizer" in n for n in names)

            if fmt == "task":
                result["metadata"] = "metadata/metadata.json" in names
            else:
                result["metadata"] = "config.json" in names

    except zipfile.BadZipFile:
        log.error("❌ ZIP 파일 손상")
        return result
    except Exception as e:
        log.error(f"❌ ZIP 읽기 오류: {e}")
        return result

    log.info(f"\n  model.tflite 포함: {'✅' if result['model_tflite'] else '❌'}")
    log.info(f"  tokenizer 포함   : {'✅' if result['tokenizer']    else '⚠️ 없음'}")
    log.info(f"  metadata 포함    : {'✅' if result['metadata']     else '❌'}")
    return result


# ────────────────────────────────────────────────────────────
# CHECK 3: 메타데이터 검증
# ────────────────────────────────────────────────────────────
def check_metadata(path: Path, fmt: str) -> bool:
    section("CHECK 3: 메타데이터 검증")
    try:
        with zipfile.ZipFile(path, "r") as zf:
            meta_file = "metadata/metadata.json" if fmt == "task" else "config.json"
            if meta_file not in zf.namelist():
                log.warning(f"⚠️  {meta_file} 없음")
                return False
            meta = json.loads(zf.read(meta_file))
    except Exception as e:
        log.error(f"❌ 메타데이터 읽기 실패: {e}")
        return False

    log.info("메타데이터 내용:")
    for k, v in meta.items():
        log.info(f"  {k:<20}: {v}")

    bos = meta.get("start_token_id") or meta.get("bos_token_id")
    eos = meta.get("stop_token_ids") or meta.get("eos_token_id")

    if bos is None:
        log.warning("⚠️  start_token_id(bos) 없음 — flutter_gemma 초기화 시 문제 가능")
    else:
        log.info(f"✅ bos_token_id: {bos}")

    if not eos:
        log.warning("⚠️  stop_token_ids(eos) 없음 — 무한 생성 가능")
    else:
        log.info(f"✅ eos_token_id: {eos}")

    return True


# ────────────────────────────────────────────────────────────
# CHECK 4: TFLite 인터프리터 로드
# ────────────────────────────────────────────────────────────
def check_tflite_load(path: Path) -> bool:
    section("CHECK 4: TFLite 인터프리터 로드")
    try:
        try:
            from ai_edge_litert.interpreter import Interpreter
        except ImportError:
            try:
                import tflite_runtime.interpreter as tflite
                Interpreter = tflite.Interpreter
            except ImportError:
                import tensorflow as tf
                Interpreter = tf.lite.Interpreter
    except Exception:
        log.warning("⚠️  TFLite 인터프리터 없음 (ai_edge_litert / tflite_runtime / tensorflow 미설치)")
        log.warning("   pip install ai-edge-litert 또는 pip install tflite-runtime")
        return False

    try:
        with zipfile.ZipFile(path, "r") as zf:
            tflite_bytes = zf.read("model.tflite")
    except Exception as e:
        log.error(f"❌ TFLite 바이트 추출 실패: {e}")
        return False

    try:
        interp = Interpreter(model_content=tflite_bytes)
        interp.allocate_tensors()
        inp_details = interp.get_input_details()
        out_details = interp.get_output_details()
        log.info(f"✅ TFLite 로드 성공")
        log.info(f"   입력 텐서 수: {len(inp_details)}")
        log.info(f"   출력 텐서 수: {len(out_details)}")
        for i, d in enumerate(inp_details):
            log.info(f"   input[{i}]  shape={d['shape']} dtype={d['dtype'].__name__}")
        return True
    except Exception as e:
        log.error(f"❌ TFLite 로드 실패: {e}")
        return False


# ────────────────────────────────────────────────────────────
# CHECK 5: 샘플 추론
# ────────────────────────────────────────────────────────────
def check_inference(path: Path, meta_json_path: Path) -> bool:
    section("CHECK 5: 샘플 추론")

    if not meta_json_path.exists():
        log.warning(f"export_meta.json 없음: {meta_json_path}")
        return False

    with open(meta_json_path, encoding="utf-8") as f:
        export_meta = json.load(f)

    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            export_meta["tokenizer_path"], trust_remote_code=True
        )
    except Exception as e:
        log.warning(f"tokenizer 로드 실패: {e}")
        return False

    try:
        try:
            from ai_edge_litert.interpreter import Interpreter
        except ImportError:
            import tflite_runtime.interpreter as tflite
            Interpreter = tflite.Interpreter
    except Exception:
        log.warning("TFLite 인터프리터 없음, 추론 스킵")
        return False

    import numpy as np

    try:
        with zipfile.ZipFile(path, "r") as zf:
            tflite_bytes = zf.read("model.tflite")

        interp = Interpreter(model_content=tflite_bytes)
        interp.allocate_tensors()

        test_prompt = "안녕하세요"
        ids = tokenizer.encode(test_prompt, add_special_tokens=False)

        inp_details = interp.get_input_details()
        # 고정 shape에 맞게 패딩/트런케이션 (ai_edge_litert는 resize_input_tensor 미지원)
        seq_len = int(inp_details[0]["shape"][1])
        ids = ids[:seq_len]
        pad_len = seq_len - len(ids)
        token_ids  = np.array([ids + [0] * pad_len], dtype=np.int64)
        attn_mask  = np.array([[1] * len(ids) + [0] * pad_len], dtype=np.int64)

        interp.set_tensor(inp_details[0]["index"], token_ids)
        if len(inp_details) > 1:
            interp.set_tensor(inp_details[1]["index"], attn_mask)
        interp.invoke()

        out_details = interp.get_output_details()
        output = interp.get_tensor(out_details[0]["index"])
        log.info(f"✅ 추론 성공! 출력 shape: {output.shape}")
        return True
    except Exception as e:
        log.error(f"❌ 추론 실패: {e}")
        return False


# ────────────────────────────────────────────────────────────
# 최종 리포트
# ────────────────────────────────────────────────────────────
def print_report(checks: dict, artifact_path: Path):
    section("최종 검증 리포트")
    all_pass = all(checks.values())
    for k, v in checks.items():
        icon = "✅" if v else ("⚠️" if k in ("tokenizer", "inference") else "❌")
        log.info(f"  {k:<25}: {icon}")

    log.info("")
    if all_pass:
        log.info("🟢 모든 검증 통과 — Flutter에서 로드 가능 예상")
        log.info(f"   artifact: {artifact_path}")
    else:
        failed = [k for k, v in checks.items() if not v]
        log.warning(f"🔴 실패 항목: {failed}")
        log.warning("   bundle_task.py 재실행 또는 export 단계를 확인하세요.")

    log.info(f"\n로그 저장: {log_file}")


def main():
    parser = argparse.ArgumentParser(description="변환 산출물 검증")
    parser.add_argument("--quant",  choices=["int8", "int4"], required=True)
    parser.add_argument("--format", choices=["task", "litertlm"], default="task")
    args = parser.parse_args()

    ext  = ".task" if args.format == "task" else ".litertlm"
    path = ONDEVICE_DIR / "outputs" / args.quant / f"kanana_{args.quant}{ext}"
    meta_path = ONDEVICE_DIR / "outputs" / args.quant / "export_meta.json"

    checks = {}
    checks["file_exists"] = check_file_exists(path)

    if checks["file_exists"]:
        struct_result         = check_zip_structure(path, args.format)
        checks["model_tflite"] = struct_result["model_tflite"]
        checks["tokenizer"]    = struct_result["tokenizer"]
        checks["metadata"]     = check_metadata(path, args.format)
        checks["tflite_load"]  = check_tflite_load(path)
        checks["inference"]    = check_inference(path, meta_path)
    else:
        for k in ("model_tflite", "tokenizer", "metadata", "tflite_load", "inference"):
            checks[k] = False

    print_report(checks, path)


if __name__ == "__main__":
    main()
