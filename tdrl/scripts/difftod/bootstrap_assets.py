"""Bootstrap only: download explicitly pinned DiffTOD assets and write byte hashes."""

import argparse
import json
import os
import re
import sys
import tempfile
from pathlib import Path

import torch


def sha256(path):
    import hashlib

    value = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def unwrap_state(value):
    if not isinstance(value, dict):
        raise ValueError("downloaded SEDD shard has an unknown format")
    for key in ("state_dict", "model"):
        if key in value and isinstance(value[key], dict):
            value = value[key]
            break
    if not value or not all(torch.is_tensor(tensor) for tensor in value.values()):
        raise ValueError("downloaded SEDD shard does not contain model tensors")
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sedd-revision", required=True)
    parser.add_argument("--tokenizer-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    args = parser.parse_args()
    for revision in (args.sedd_revision, args.tokenizer_revision):
        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError("bootstrap requires immutable 40-character commit revisions")
    from huggingface_hub import snapshot_download

    from safetensors.torch import load_file

    sedd_dir = args.output / "sedd-medium" / args.sedd_revision
    token_dir = args.output / "gpt2" / args.tokenizer_revision
    snapshot_download(
        "louaaron/sedd-medium",
        revision=args.sedd_revision,
        local_dir=sedd_dir,
        allow_patterns=["*.safetensors", "*.bin", "*.json", "*.yaml"],
    )
    snapshot_download(
        "openai-community/gpt2",
        revision=args.tokenizer_revision,
        local_dir=token_dir,
        allow_patterns=["*.json", "*.txt", "*.model"],
    )
    safetensors = sorted(sedd_dir.glob("*.safetensors"))
    weights = safetensors or sorted(sedd_dir.glob("*.bin"))
    if not weights:
        raise ValueError("downloaded SEDD snapshot contains no supported weight files")
    state = {}
    for source in weights:
        shard = (
            load_file(str(source), device="cpu")
            if source.suffix == ".safetensors"
            else torch.load(source, map_location="cpu", weights_only=True)
        )
        for key, tensor in unwrap_state(shard).items():
            if key in state:
                raise ValueError(f"duplicate SEDD tensor across shards: {key}")
            state[key] = tensor
    checkpoint = sedd_dir / "checkpoint.pth"
    torch.save(state, checkpoint)
    token_files = {
        path.name: sha256(path)
        for path in sorted(token_dir.iterdir())
        if path.is_file() and not path.name.startswith(".")
    }
    lock = {
        "schema_version": 1,
        "status": "pinned_and_verified",
        "sedd": {
            "model_id": "louaaron/sedd-medium",
            "revision": args.sedd_revision,
            "checkpoint_sha256": sha256(checkpoint),
            "source_files": {path.name: sha256(path) for path in weights},
        },
        "tokenizer": {
            "model_id": "openai-community/gpt2",
            "revision": args.tokenizer_revision,
            "files": token_files,
        },
    }
    args.lock.parent.mkdir(parents=True, exist_ok=True)
    if args.lock.exists():
        previous = json.loads(args.lock.read_text(encoding="utf-8"))
        if previous.get("status") != "blocked_until_bootstrap":
            raise FileExistsError("refusing to replace an already pinned DiffTOD asset lock")
    descriptor, temporary = tempfile.mkstemp(dir=args.lock.parent, prefix=".difftod-lock-")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(lock, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, args.lock)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    print(json.dumps({"checkpoint": str(checkpoint), "lock": str(args.lock)}, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"DiffTOD bootstrap failed: {error}", file=sys.stderr)
        raise
