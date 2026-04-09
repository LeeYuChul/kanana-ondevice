#!/usr/bin/env python3
"""
Kanana-Nano-2.1B-Base → MediaPipe-compatible multi-signature TFLite 변환 스크립트

이 스크립트는 litert_torch의 DecoderOnlyModel 래퍼와 convert_to_tflite 함수를 사용하여
MediaPipe LlmInference가 요구하는 multi-signature TFLite 모델을 생성합니다.

요구 시그니처:
- prefill_256 (또는 다중 prefill 길이)
- decode

사용법:
    cd /home/mjudcd/문서/Study/kanana-test
    source venv/bin/activate
    python export_kanana_mediapipe.py --quantize dynamic_int4_block128 --output_path ./outputs
"""

import os
import argparse
from functools import partial

import torch

# litert_torch imports
import litert_torch.generative.layers.model_config as cfg
from litert_torch.generative.utilities import model_builder
from litert_torch.generative.utilities import converter
from litert_torch.generative.utilities import loader
from litert_torch.generative.utilities import export_config as export_config_lib

# HuggingFace snapshot download
from huggingface_hub import snapshot_download


# Kanana 모델 설정 (config.json 기반)
def get_kanana_model_config() -> cfg.ModelConfig:
    """Kanana-Nano-2.1B-Base 모델 설정 반환"""
    
    # Attention 설정
    attn_config = cfg.AttentionConfig(
        num_heads=24,           # num_attention_heads
        head_dim=128,           # head_dim (hidden_size / num_heads = 1792 / 24 ≈ 74.67, but config says 128)
        num_query_groups=8,     # num_key_value_heads (GQA)
        rotary_base=500000,     # rope_theta
        rotary_percentage=1.0,  # 전체 head_dim에 RoPE 적용
    )
    
    # FeedForward 설정
    ff_config = cfg.FeedForwardConfig(
        type=cfg.FeedForwardType.GATED,
        activation=cfg.ActivationConfig(cfg.ActivationType.SILU),  # hidden_act: silu
        intermediate_size=8064,  # intermediate_size
    )
    
    # Normalization 설정
    norm_config = cfg.NormalizationConfig(
        type=cfg.NormalizationType.RMS_NORM,
        epsilon=1e-05,  # rms_norm_eps
    )
    
    # Transformer Block 설정
    block_config = cfg.TransformerBlockConfig(
        attn_config=attn_config,
        ff_config=ff_config,
        pre_attention_norm_config=norm_config,
        post_attention_norm_config=norm_config,
    )
    
    # 전체 모델 설정
    config = cfg.ModelConfig(
        vocab_size=128256,          # vocab_size
        num_layers=32,              # num_hidden_layers
        max_seq_len=8192,           # max_position_embeddings
        embedding_dim=1792,         # hidden_size
        block_configs=block_config,
        final_norm_config=norm_config,
        lm_head_share_weight_with_embedding=True,  # tie_word_embeddings
    )
    
    return config


class Kanana(model_builder.DecoderOnlyModel):
    """Kanana-Nano-2.1B 모델 래퍼
    
    DecoderOnlyModel을 상속하여 MediaPipe 호환 시그니처를 생성합니다.
    """
    pass


def load_kanana_weights(
    pytorch_model: model_builder.DecoderOnlyModel,
    checkpoint_path: str,
) -> model_builder.DecoderOnlyModel:
    """HuggingFace 체크포인트에서 가중치를 로드합니다.
    
    Args:
        pytorch_model: 초기화된 DecoderOnlyModel
        checkpoint_path: HuggingFace 모델 경로 또는 로컬 경로
        
    Returns:
        가중치가 로드된 모델
    """
    print(f"Loading weights from: {checkpoint_path}")
    
    # HuggingFace 허브에서 다운로드 (캐시 사용)
    if not os.path.isdir(checkpoint_path):
        print(f"Downloading from HuggingFace Hub: {checkpoint_path}")
        local_path = snapshot_download(
            repo_id=checkpoint_path,
            allow_patterns=["*.safetensors", "*.bin"],
        )
    else:
        local_path = checkpoint_path
    
    print(f"Local checkpoint path: {local_path}")
    
    # TENSOR_NAMES: 표준 Llama 포맷과 동일 (Kanana도 LlamaForCausalLM)
    tensor_names = model_builder.TENSOR_NAMES
    
    # ModelLoader를 사용하여 가중치 로드
    model_loader = loader.ModelLoader(
        file_name=local_path,
        names=tensor_names,
    )
    
    missing_keys, unexpected_keys = model_loader.load(pytorch_model, strict=False)
    
    if missing_keys:
        print(f"Warning: Missing keys: {missing_keys[:5]}...")
    if unexpected_keys:
        print(f"Warning: Unexpected keys: {unexpected_keys[:5]}...")
    
    print("✅ Weights loaded successfully")
    return pytorch_model


def convert_kanana_to_tflite(
    checkpoint_path: str,
    output_path: str,
    output_name: str = "kanana_nano_2_1b",
    quantize: str = "dynamic_int4_block128",
    prefill_seq_lens: list = None,
    kv_cache_max_len: int = 1280,
):
    """Kanana 모델을 MediaPipe 호환 multi-signature TFLite로 변환
    
    Args:
        checkpoint_path: HuggingFace 모델 경로
        output_path: 출력 디렉토리
        output_name: 출력 파일 이름 prefix
        quantize: 양자화 타입 (dynamic_int4_block128, dynamic_int8, etc.)
        prefill_seq_lens: prefill 시퀀스 길이 리스트
        kv_cache_max_len: KV 캐시 최대 길이
    """
    if prefill_seq_lens is None:
        prefill_seq_lens = [256]  # 기본값: 256 토큰 prefill
    
    os.makedirs(output_path, exist_ok=True)
    
    print("=" * 60)
    print("Kanana → MediaPipe TFLite Conversion")
    print("=" * 60)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Output: {output_path}/{output_name}_*.tflite")
    print(f"Quantization: {quantize}")
    print(f"Prefill lengths: {prefill_seq_lens}")
    print(f"KV cache max: {kv_cache_max_len}")
    print("=" * 60)
    
    # 1. 모델 설정 생성
    print("\n[1/4] Creating model config...")
    config = get_kanana_model_config()
    
    # 2. DecoderOnlyModel 인스턴스 생성
    print("\n[2/4] Building DecoderOnlyModel...")
    # mask_cache_size는 kv_cache_max_len과 같거나 작아야 함
    # mask_as_input=False일 때 필요
    mask_cache_size = kv_cache_max_len
    pytorch_model = Kanana(config, mask_cache_size=mask_cache_size)
    pytorch_model.eval()
    
    # 3. 가중치 로드
    print("\n[3/4] Loading HuggingFace weights...")
    pytorch_model = load_kanana_weights(pytorch_model, checkpoint_path)
    
    # 4. Multi-signature TFLite 변환
    print("\n[4/4] Converting to multi-signature TFLite...")
    print("  This will create signatures: ", end="")
    if len(prefill_seq_lens) == 1:
        print(f"'prefill', 'decode'")
    else:
        sigs = [f"'prefill_{l}'" for l in prefill_seq_lens] + ["'decode'"]
        print(", ".join(sigs))
    
    # ExportConfig 생성 (기본값 사용)
    export_config = export_config_lib.ExportConfig()
    
    converter.convert_to_tflite(
        pytorch_model=pytorch_model,
        output_path=output_path,
        output_name_prefix=output_name,
        prefill_seq_len=prefill_seq_lens,
        kv_cache_max_len=kv_cache_max_len,
        quantize=quantize,
        config=config,
        export_config=export_config,
    )
    
    print("\n" + "=" * 60)
    print("✅ Conversion complete!")
    print("=" * 60)
    
    # 결과 파일 출력
    quant_suffix = {
        "none": "f32",
        "dynamic_int8": "q8",
        "dynamic_int4_block32": "q4_b32",
        "dynamic_int4_block128": "q4_b128",
    }.get(quantize, "q")
    
    expected_file = f"{output_path}/{output_name}_{quant_suffix}_ekv{kv_cache_max_len}.tflite"
    print(f"\nExpected output: {expected_file}")
    
    if os.path.exists(expected_file):
        size_mb = os.path.getsize(expected_file) / (1024 * 1024)
        print(f"File size: {size_mb:.1f} MB")
    
    return expected_file


def main():
    parser = argparse.ArgumentParser(
        description="Convert Kanana-Nano-2.1B to MediaPipe-compatible TFLite"
    )
    parser.add_argument(
        "--checkpoint", "-c",
        default="kakaocorp/kanana-nano-2.1b-base",
        help="HuggingFace model path or local checkpoint path"
    )
    parser.add_argument(
        "--output_path", "-o",
        default="./outputs",
        help="Output directory for TFLite files"
    )
    parser.add_argument(
        "--output_name", "-n",
        default="kanana_nano_2_1b",
        help="Output filename prefix"
    )
    parser.add_argument(
        "--quantize", "-q",
        default="dynamic_int4_block128",
        choices=[
            "none", 
            "dynamic_int8", 
            "dynamic_int4_block32", 
            "dynamic_int4_block128"
        ],
        help="Quantization type"
    )
    parser.add_argument(
        "--prefill_seq_lens", "-p",
        nargs="+",
        type=int,
        default=[256],
        help="Prefill sequence lengths (space-separated)"
    )
    parser.add_argument(
        "--kv_cache_max_len", "-k",
        type=int,
        default=1280,
        help="Maximum KV cache length"
    )
    
    args = parser.parse_args()
    
    convert_kanana_to_tflite(
        checkpoint_path=args.checkpoint,
        output_path=args.output_path,
        output_name=args.output_name,
        quantize=args.quantize,
        prefill_seq_lens=args.prefill_seq_lens,
        kv_cache_max_len=args.kv_cache_max_len,
    )


if __name__ == "__main__":
    main()
