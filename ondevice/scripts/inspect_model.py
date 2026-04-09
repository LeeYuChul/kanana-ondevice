"""
1단계: Kanana 모델 사전 점검 스크립트
- 아키텍처 이름 확인
- transformers 로드 가능 여부
- tokenizer 형식 확인
- special token 확인
- LiteRT 변환 가능성 판단
"""

import json
import sys
import logging
from pathlib import Path
from datetime import datetime

# ── 로그 설정 ──────────────────────────────────────────────
LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
log_file = LOG_DIR / f"inspect_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_file, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

MODEL_ID = "kakaocorp/kanana-nano-2.1b-base"

# MediaPipe/LiteRT가 공식 지원하는 아키텍처 목록
SUPPORTED_ARCH = {
    "GemmaForCausalLM",
    "Gemma2ForCausalLM",
    "LlamaForCausalLM",
    "Phi3ForCausalLM",
    "MistralForCausalLM",
    "Qwen2ForCausalLM",
    "Falcon3ForCausalLM",
}

# Llama 계열 아키텍처 (구조 매핑 가능성 있음)
LLAMA_LIKE_ARCH = {
    "LlamaForCausalLM",
    "MistralForCausalLM",
    "Qwen2ForCausalLM",
    "Falcon3ForCausalLM",
}


def section(title: str):
    log.info("")
    log.info("=" * 60)
    log.info(f"  {title}")
    log.info("=" * 60)


def check_transformers_load():
    section("STEP 1: transformers 로드 테스트")
    try:
        from transformers import AutoConfig, AutoTokenizer
        log.info(f"모델 ID: {MODEL_ID}")

        log.info("AutoConfig 로드 중...")
        config = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
        log.info(f"✅ AutoConfig 로드 성공")

        log.info("AutoTokenizer 로드 중...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
        log.info(f"✅ AutoTokenizer 로드 성공")

        return config, tokenizer
    except Exception as e:
        log.error(f"❌ 로드 실패: {e}")
        return None, None


def inspect_config(config):
    section("STEP 2: 아키텍처 분석")
    if config is None:
        log.error("config 없음, 스킵")
        return None

    arch_list = getattr(config, "architectures", [])
    arch = arch_list[0] if arch_list else "unknown"
    model_type = getattr(config, "model_type", "unknown")

    log.info(f"architectures : {arch_list}")
    log.info(f"model_type    : {model_type}")
    log.info(f"hidden_size   : {getattr(config, 'hidden_size', 'N/A')}")
    log.info(f"num_layers    : {getattr(config, 'num_hidden_layers', 'N/A')}")
    log.info(f"num_heads     : {getattr(config, 'num_attention_heads', 'N/A')}")
    log.info(f"num_kv_heads  : {getattr(config, 'num_key_value_heads', 'N/A')}")
    log.info(f"vocab_size    : {getattr(config, 'vocab_size', 'N/A')}")
    log.info(f"max_pos_emb   : {getattr(config, 'max_position_embeddings', 'N/A')}")
    log.info(f"rope_theta    : {getattr(config, 'rope_theta', 'N/A')}")
    log.info(f"intermediate  : {getattr(config, 'intermediate_size', 'N/A')}")

    # LiteRT 지원 여부 판단
    if arch in SUPPORTED_ARCH:
        log.info(f"\n✅ [직접 지원] '{arch}' 는 LiteRT/MediaPipe 공식 지원 아키텍처입니다.")
        log.info("   → export_8bit.py 바로 진행 가능")
        verdict = "DIRECT"
    elif arch in LLAMA_LIKE_ARCH or model_type in ("llama", "mistral", "qwen2"):
        log.info(f"\n⚠️  [매핑 필요] '{arch}' 는 Llama 계열이므로 구조 매핑 후 변환 가능성 있음")
        log.info("   → ai_edge_torch 커스텀 래퍼 필요")
        verdict = "MAPPING"
    else:
        log.info(f"\n❌ [미지원] '{arch}' 는 LiteRT 공식 지원 목록에 없습니다.")
        log.info("   → GGUF/ONNX 등 대안 런타임 검토 권장")
        verdict = "UNSUPPORTED"

    return {"arch": arch, "model_type": model_type, "verdict": verdict}


def inspect_tokenizer(tokenizer):
    section("STEP 3: Tokenizer 분석")
    if tokenizer is None:
        log.error("tokenizer 없음, 스킵")
        return None

    tok_class = type(tokenizer).__name__
    log.info(f"tokenizer class   : {tok_class}")
    log.info(f"vocab_size        : {tokenizer.vocab_size}")

    # SentencePiece 여부 (MediaPipe 번들링에 영향)
    is_sp = hasattr(tokenizer, "sp_model") or "sentencepiece" in tok_class.lower()
    log.info(f"SentencePiece 기반: {'✅ YES' if is_sp else '❌ NO (BPE 등 다른 방식)'}")

    # Special tokens
    log.info("\n[Special Tokens]")
    special = {
        "bos_token"      : tokenizer.bos_token,
        "eos_token"      : tokenizer.eos_token,
        "unk_token"      : tokenizer.unk_token,
        "pad_token"      : tokenizer.pad_token,
        "bos_token_id"   : tokenizer.bos_token_id,
        "eos_token_id"   : tokenizer.eos_token_id,
        "unk_token_id"   : tokenizer.unk_token_id,
        "pad_token_id"   : tokenizer.pad_token_id,
    }
    for k, v in special.items():
        log.info(f"  {k:<18}: {v}")

    # 인코딩 테스트 (add_special_tokens=False: BOS 토큰 자동 추가 방지)
    test_str = "안녕하세요, 저는 Kanana 모델입니다."
    ids = tokenizer.encode(test_str, add_special_tokens=False)
    decoded = tokenizer.decode(ids, skip_special_tokens=True)
    log.info(f"\n[인코딩 테스트]")
    log.info(f"  입력   : {test_str}")
    log.info(f"  토큰 수: {len(ids)}")
    log.info(f"  복원   : {decoded}")
    round_trip_ok = decoded.strip() == test_str.strip()
    log.info(f"  라운드트립: {'✅ OK' if round_trip_ok else '⚠️ 불일치 (확인 필요)'}")

    return {
        "class"        : tok_class,
        "is_sp"        : is_sp,
        "bos_token_id" : tokenizer.bos_token_id,
        "eos_token_id" : tokenizer.eos_token_id,
    }


def check_ai_edge_torch():
    section("STEP 4: ai_edge_torch / litert-torch 설치 확인")
    # 신규 패키지명 우선 시도
    for pkg_name in ("litert.torch", "ai_edge_torch"):
        try:
            mod = __import__(pkg_name)
            version = getattr(mod, "__version__", "버전정보없음(설치됨)")
            log.info(f"✅ {pkg_name} 설치됨: {version}")
            return True
        except ImportError:
            continue
    log.warning("❌ litert-torch / ai_edge_torch 미설치")
    log.warning("   설치 명령: pip install litert-torch")
    return False


def check_mediapipe():
    section("STEP 5: mediapipe / litert 설치 확인")
    mp_ok, lr_ok = False, False
    try:
        import mediapipe as mp
        log.info(f"✅ mediapipe 설치됨: {mp.__version__}")
        mp_ok = True
    except ImportError:
        log.warning("❌ mediapipe 미설치  →  pip install mediapipe")

    try:
        import litert
        log.info(f"✅ litert 설치됨: {litert.__version__}")
        lr_ok = True
    except ImportError:
        try:
            import ai_edge_litert
            log.info(f"✅ ai_edge_litert 설치됨: {ai_edge_litert.__version__}")
            lr_ok = True
        except ImportError:
            log.warning("❌ litert/ai_edge_litert 미설치  →  pip install ai-edge-litert")

    return mp_ok, lr_ok


def print_summary(arch_info, tok_info, aet_ok, mp_ok, lr_ok):
    section("최종 점검 요약")

    verdict = arch_info["verdict"] if arch_info else "UNKNOWN"

    rows = [
        ("아키텍처 로드",          "✅" if arch_info else "❌"),
        ("LiteRT 변환 가능성",     {"DIRECT": "✅ 직접 변환", "MAPPING": "⚠️ 매핑 필요", "UNSUPPORTED": "❌ 미지원", "UNKNOWN": "❓"}.get(verdict, "❓")),
        ("Tokenizer 로드",         "✅" if tok_info else "❌"),
        ("SentencePiece",          "✅" if (tok_info and tok_info["is_sp"]) else "❌"),
        ("ai_edge_torch 설치",     "✅" if aet_ok else "❌ (필요)"),
        ("mediapipe 설치",         "✅" if mp_ok  else "❌ (필요)"),
        ("litert 설치",            "✅" if lr_ok  else "❌ (필요)"),
    ]
    for label, status in rows:
        log.info(f"  {label:<25}: {status}")

    log.info("")
    if verdict == "DIRECT":
        log.info("🟢 다음 단계: export_8bit.py 바로 실행 가능")
    elif verdict == "MAPPING":
        log.info("🟡 다음 단계: configs/convert_8bit.yaml 에서 arch_mapping 옵션 설정 후 export_8bit.py 실행")
    else:
        log.info("🔴 다음 단계: GGUF(llama.cpp) 또는 ONNX 런타임 대안 검토 필요")

    if not aet_ok:
        log.info("\n📦 필수 패키지 설치:")
        log.info("  pip install ai-edge-torch ai-edge-litert mediapipe")

    log.info(f"\n📄 로그 파일 저장됨: {log_file}")


def main():
    log.info("Kanana 모델 온디바이스 변환 사전 점검 시작")
    log.info(f"대상 모델: {MODEL_ID}")

    config, tokenizer = check_transformers_load()
    arch_info = inspect_config(config)
    tok_info  = inspect_tokenizer(tokenizer)
    aet_ok    = check_ai_edge_torch()
    mp_ok, lr_ok = check_mediapipe()

    print_summary(arch_info, tok_info, aet_ok, mp_ok, lr_ok)

    # 결과를 JSON으로도 저장
    result = {
        "model_id"  : MODEL_ID,
        "arch_info" : arch_info,
        "tok_info"  : tok_info,
        "deps"      : {"ai_edge_torch": aet_ok, "mediapipe": mp_ok, "litert": lr_ok},
    }
    result_path = LOG_DIR / "inspect_result.json"
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    log.info(f"📄 JSON 결과 저장됨: {result_path}")


if __name__ == "__main__":
    main()
