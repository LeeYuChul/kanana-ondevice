from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

from transformers import AutoConfig, AutoTokenizer

from common import save_json


def has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def inspect(model_id: str) -> dict:
    cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    model_type = getattr(cfg, "model_type", "unknown")
    architectures = getattr(cfg, "architectures", [])

    tokenizer_type = type(tokenizer).__name__
    special_tokens = {
        "bos_token": tokenizer.bos_token,
        "eos_token": tokenizer.eos_token,
        "pad_token": tokenizer.pad_token,
        "unk_token": tokenizer.unk_token,
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "unk_token_id": tokenizer.unk_token_id,
    }

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
        "conversion_path_hint": _path_hint(model_type, architectures),
    }
    return report


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
