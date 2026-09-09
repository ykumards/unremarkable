#!/usr/bin/env python3
"""Fetch pinned TinyStories assets; no Python packages required."""
import argparse
import hashlib
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
MODEL_BASE = "https://huggingface.co/karpathy/tinyllamas/resolve/0bd21da7698eaf29a0d7de3992de8a46ef624add/"
SMOLLM_BASE = "https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct/resolve/main/"
CODE_BASE = "https://raw.githubusercontent.com/karpathy/llama2.c/350e04fe35433e6d2941dce5a1f53308f87058eb/"
ASSETS = {
    "15M": [
        ("stories15M.bin", MODEL_BASE + "stories15M.bin", "cd590644d963867a2b6e5a1107f51fad663c41d79c149fbecbbb1f95fa81f49a"),
        ("tokenizer.bin", CODE_BASE + "tokenizer.bin", "50a52ef822ee9e83de5ce9d0be0a025a773d019437f58b5ff9dcafb063ece361"),
    ],
    "260K": [
        ("stories260K.bin", MODEL_BASE + "stories260K/stories260K.bin", "b0a507e7ad0f626624f17112325e66691f9076d622e1d3274d103d00299f2696"),
        ("tok512.bin", MODEL_BASE + "stories260K/tok512.bin", "037cb335abb25d1fa9e8ecae30ed2a3a8ace9302862ebcdc05d51a6bbb10c312"),
    ],
}
# Instruct model, converted to this engine's format by tools/export_hf.py.
SMOLLM = [
    ("config.json", "8eb740e8bbe4cff95ea7b4588d17a2432deb16e8075bc5828ff7ba9be94d982a"),
    ("tokenizer.json", "9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c"),
    ("model.safetensors", "5af571cbf074e6d21a03528d2330792e532ca608f24ac70a143f6b369968ab8c"),
]


def checksum(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def download(name, url, digest, subdirectory=""):
    destination = ROOT / "models" / subdirectory / name
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if checksum(destination) != digest:
            raise SystemExit(f"Checksum mismatch: {destination}; move it aside and retry.")
        print(f"Verified {name}")
        return
    with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as stream:
        temporary = Path(stream.name)
    try:
        subprocess.run(["curl", "--fail", "--location", "--retry", "3",
                        "--connect-timeout", "30", "--max-time", "600",
                        "--output", str(temporary), url], check=True)
        if checksum(temporary) != digest:
            raise SystemExit(f"Checksum mismatch downloading {name}")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Downloaded and verified {name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=[*ASSETS, "smollm2"])
    args = parser.parse_args()
    if args.model == "smollm2":
        for name, digest in SMOLLM:
            download(name, SMOLLM_BASE + name, digest, "hf/SmolLM2-135M-Instruct")
        raise SystemExit(0)
    for asset in ASSETS[args.model]:
        download(*asset)
