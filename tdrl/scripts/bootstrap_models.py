#!/usr/bin/env python3
"""Public, credential-free bootstrap download; never performs model inference."""

import argparse
import json
import shutil
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tdrl.provenance import file_hash  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--lock", type=Path, default=Path("configs/models.lock.json"))
    args = parser.parse_args()
    lock = json.loads((args.root / args.lock).read_text())
    for model in lock["models"]:
        mid, revision = model["model_id"], model["revision"]
        request = urllib.request.Request(
            f"https://huggingface.co/api/models/{mid}/revision/{revision}"
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            info = json.load(response)
        if info["sha"] != revision:
            raise ValueError("bootstrap revision mismatch")
        directory = args.root / "models" / mid.split("/")[-1] / revision
        directory.mkdir(parents=True, exist_ok=True)
        for row in info["siblings"]:
            filename = row["rfilename"]
            if not filename.endswith((".json", ".jinja", ".safetensors", ".txt", ".model")):
                continue
            if Path(filename).is_absolute() or ".." in Path(filename).parts:
                raise ValueError("unsafe bootstrap path")
            dest = directory / filename
            expected = model["weight_sha256"].get(filename)
            if dest.exists() and (not expected or file_hash(dest) == expected):
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            part = dest.with_suffix(dest.suffix + ".part")
            url = f"https://huggingface.co/{mid}/resolve/{revision}/{filename}"
            with urllib.request.urlopen(url, timeout=300) as response, part.open("wb") as output:
                shutil.copyfileobj(response, output, 8 * 1024 * 1024)
            if expected and file_hash(part) != expected:
                raise ValueError("download weight checksum mismatch")
            part.replace(dest)
            print(f"Downloaded {mid}@{revision}/{filename}", flush=True)
    print("Bootstrap complete. Run tdrl verify-models before submitting any allocation.")


if __name__ == "__main__":
    main()
