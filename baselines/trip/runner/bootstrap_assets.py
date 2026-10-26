import argparse
import json
import re
from pathlib import Path

from tdrl.baselines.trip.assets import directory_hashes
from tdrl.provenance import atomic_json


def main():
    parser = argparse.ArgumentParser(description="Bootstrap immutable local TRIP BERT assets")
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--model-id", default="bert-base-uncased")
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}", args.revision):
        raise ValueError("TRIP bootstrap requires a full lowercase 40-character commit SHA")
    if args.lock.exists():
        previous = json.loads(args.lock.read_text(encoding="utf-8"))
        if previous.get("status") == "pinned_and_verified":
            raise FileExistsError("refusing to replace an already pinned TRIP asset lock")
    from huggingface_hub import snapshot_download

    destination = args.output / args.model_id.replace("/", "--") / args.revision
    snapshot_download(
        repo_id=args.model_id,
        revision=args.revision,
        local_dir=destination,
        allow_patterns=("*.json", "*.txt", "*.safetensors", "*.model"),
    )
    hashes = directory_hashes(destination)
    if not any(name.endswith((".safetensors", ".bin")) for name in hashes):
        raise ValueError("downloaded TRIP encoder snapshot contains no weights")
    payload = {
        "schema_version": 1,
        "status": "pinned_and_verified",
        "model_id": args.model_id,
        "revision": args.revision,
        "encoder": hashes,
        "tokenizer": hashes,
    }
    atomic_json(args.lock, payload)
    print(json.dumps({"path": str(destination), "files": len(hashes)}, indent=2))


if __name__ == "__main__":
    main()
