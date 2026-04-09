from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from common import load_yaml, normalize_dtype, pick_device, save_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    model_id = cfg["model_id"]
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    dtype = normalize_dtype(cfg.get("dtype", "bfloat16"))
    compute_dtype = normalize_dtype(cfg.get("bnb_4bit_compute_dtype", "bfloat16"))
    device = pick_device(cfg.get("device", "auto"))

    qconfig = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_quant_type=cfg.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_use_double_quant=bool(cfg.get("bnb_4bit_use_double_quant", True)),
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        quantization_config=qconfig,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    prompt = cfg.get("sample_prompt", "안녕하세요")
    inputs = tokenizer(prompt, return_tensors="pt")
    if device == "cuda":
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

    model.eval()
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=int(cfg.get("max_new_tokens", 64)),
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    decoded = tokenizer.decode(output[0], skip_special_tokens=True)

    model.save_pretrained(out_dir)
    if cfg.get("export", {}).get("save_tokenizer", True):
        tokenizer.save_pretrained(out_dir)

    manifest = {
        "model_id": model_id,
        "quantization": "int4",
        "output_dir": str(out_dir),
        "sample_prompt": prompt,
        "sample_output": decoded,
        "torch_dtype": str(dtype),
        "compute_dtype": str(compute_dtype),
        "device": device,
    }
    save_json(manifest, out_dir / "export_manifest.json")
    print(f"[export_4bit] done -> {out_dir}")


if __name__ == "__main__":
    main()
