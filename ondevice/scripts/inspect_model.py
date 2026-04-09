from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

from huggingface_hub import hf_hub_download
from transformers import AutoConfig, AutoTokenizer

from common import save_json


def has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def inspect(model_id: str) -> dict:
    cfg, config_load_error, raw_config = _load_config_with_fallback(model_id)
    tokenizer, tokenizer_error = _load_tokenizer_with_fallback(model_id)

    model_type = getattr(cfg, "model_type", None) or raw_config.get("model_type", "unknown")
    architectures = getattr(cfg, "architectures", None) or raw_config.get("architectures", [])

    tokenizer_type = type(tokenizer).__name__ if tokenizer is not None else "unavailable"
    special_tokens = _special_tokens(tokenizer)

    sentencepiece_like = any(
        k in tokenizer_type.lower()
        for k in ["sentencepiece", "llama", "t5", "xglm", "spm"]
    )

    report = {
        "model_id": model_id,
        "model_type": model_type,
        "architectures": architectures,
        "tokenizer_class": tokenizer_type,
        "sentencepiece_like": sentencepiece_like,
        "special_tokens": special_tokens,
        "tooling_check": {
            "bitsandbytes_installed": has_module("bitsandbytes"),
            "ai_edge_torch_installed": has_module("ai_edge_torch"),
            "litert_installed": has_module("litert"),
        },
        "tokenizer_load": {
            "loaded": tokenizer_error is None,
            "error": tokenizer_error,
        },
        "config_load": {
            "loaded_with_autoconfig": config_load_error is None,
            "autoconfig_error": config_load_error,
            "raw_config_available": bool(raw_config),
        },
        "raw_architecture_sanity": _raw_architecture_sanity(raw_config),
        "conversion_path_hint": _path_hint(model_type, architectures),
    }
    return report


def _load_tokenizer_with_fallback(model_id: str):
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        return tokenizer, None
    except Exception as e:  # pragma: no cover - runtime/environment dependent
        return None, str(e)


def _special_tokens(tokenizer) -> dict:
    if tokenizer is None:
        return {}
    return {
        "bos_token": tokenizer.bos_token,
        "eos_token": tokenizer.eos_token,
        "pad_token": tokenizer.pad_token,
        "unk_token": tokenizer.unk_token,
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "unk_token_id": tokenizer.unk_token_id,
    }


def _load_config_with_fallback(model_id: str):
    raw_config = _load_raw_config_json(model_id)
    try:
        cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        return cfg, None, raw_config
    except Exception as e:  # pragma: no cover - runtime/environment dependent
        return None, str(e), raw_config


def _load_raw_config_json(model_id: str) -> dict:
    try:
        config_path = hf_hub_download(repo_id=model_id, filename="config.json")
    except Exception:
        return {}
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _raw_architecture_sanity(raw_config: dict) -> dict:
    if not raw_config:
        return {"checked": False}

    hidden_size = raw_config.get("hidden_size")
    num_attention_heads = raw_config.get("num_attention_heads")
    divisible = None
    if isinstance(hidden_size, int) and isinstance(num_attention_heads, int) and num_attention_heads > 0:
        divisible = (hidden_size % num_attention_heads) == 0

    return {
        "checked": True,
        "hidden_size": hidden_size,
        "num_attention_heads": num_attention_heads,
        "hidden_size_divisible_by_num_attention_heads": divisible,
    }


def _path_hint(model_type: str, architectures: list[str]) -> str:
    joined = " ".join([model_type, *architectures]).lower()
    if any(key in joined for key in ["llama", "qwen", "gemma"]):
        return "direct_or_mapped_supported_family"
    return "custom_mapping_or_runtime_fallback"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--out", default="ondevice/logs/inspect_report.json")
    args = parser.parse_args()

    report = inspect(args.model_id)
    save_json(report, args.out)

    print("[inspect] done")
    print(f"- report: {Path(args.out).resolve()}")
    print(f"- model_type: {report['model_type']}")
    print(f"- tokenizer_class: {report['tokenizer_class']}")
    print(f"- hint: {report['conversion_path_hint']}")


if __name__ == "__main__":
    main()
