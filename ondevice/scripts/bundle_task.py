from __future__ import annotations

import argparse
import json
import tarfile
from pathlib import Path


def write_metadata(input_dir: Path, fmt: str) -> Path:
    tokenizer_config = input_dir / "tokenizer_config.json"
    generation_config = input_dir / "generation_config.json"
    manifest = input_dir / "export_manifest.json"

    data = {
        "format": fmt,
        "model_dir": str(input_dir),
        "has_tokenizer": tokenizer_config.exists(),
        "has_generation_config": generation_config.exists(),
        "has_export_manifest": manifest.exists(),
        "notes": [
            "This is a lightweight bundle scaffold for MediaPipe/LiteRT integration.",
            "If official converter is available, replace this step with converter-generated artifact.",
        ],
    }

    meta_path = input_dir / "bundle_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return meta_path


def make_bundle(input_dir: Path, output: Path, fmt: str) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    meta_path = write_metadata(input_dir, fmt)

    with tarfile.open(output, "w:gz") as tar:
        for p in input_dir.iterdir():
            if p.name.endswith(".task") or p.name.endswith(".litertlm"):
                continue
            tar.add(p, arcname=p.name)
        tar.add(meta_path, arcname=meta_path.name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--format", choices=["task", "litertlm"], required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output = Path(args.output)

    suffix = ".task" if args.format == "task" else ".litertlm"
    if output.suffix != suffix:
        raise ValueError(f"Output file must end with {suffix}: {output}")

    make_bundle(input_dir, output, args.format)
    print(f"[bundle_task] created {output}")


if __name__ == "__main__":
    main()
