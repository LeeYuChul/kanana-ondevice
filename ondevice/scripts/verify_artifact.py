from __future__ import annotations

import argparse
import json
import tarfile
import tempfile
from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer


def verify_bundle(artifact: Path) -> dict:
    if not artifact.exists():
        raise FileNotFoundError(str(artifact))

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        with tarfile.open(artifact, "r:gz") as tar:
            tar.extractall(td_path)

        tokenizer_files = [
            "tokenizer.json",
            "tokenizer_config.json",
            "tokenizer.model",
        ]
        has_tokenizer = any((td_path / f).exists() for f in tokenizer_files)

        has_weights = any(
            p.name.endswith((".safetensors", ".bin"))
            for p in td_path.glob("**/*")
            if p.is_file()
        )

        report = {
            "artifact": str(artifact),
            "has_tokenizer": has_tokenizer,
            "has_weights": has_weights,
            "files": sorted(str(p.relative_to(td_path)) for p in td_path.glob("**/*") if p.is_file()),
        }
    return report


def optional_inference(model_dir: Path, prompt: str, max_new_tokens: int) -> str:
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=True)

    inputs = tokenizer(prompt, return_tensors="pt")
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return tokenizer.decode(out[0], skip_special_tokens=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--sample-prompt", default="안녕하세요")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--run-inference", action="store_true")
    args = parser.parse_args()

    artifact = Path(args.artifact)
    report = verify_bundle(artifact)

    if args.run_inference:
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            with tarfile.open(artifact, "r:gz") as tar:
                tar.extractall(td_path)
            text = optional_inference(td_path, args.sample_prompt, args.max_new_tokens)
            report["sample_output"] = text

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
