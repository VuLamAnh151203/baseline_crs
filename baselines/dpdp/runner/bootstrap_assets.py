import argparse
import json
import re
from pathlib import Path

from tdrl.baselines.dpdp.assets import directory_hashes
from tdrl.provenance import atomic_json


def main():
    parser = argparse.ArgumentParser(description="Bootstrap immutable local DPDP RoBERTa assets")
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--model-id", default="FacebookAI/roberta-large")
    args = parser.parse_args()
    if args.model_id != "FacebookAI/roberta-large":
        raise ValueError("DPDP requires FacebookAI/roberta-large")
    if not re.fullmatch(r"[0-9a-f]{40}", args.revision):
        raise ValueError("DPDP bootstrap requires a full lowercase 40-character commit SHA")
    if args.lock.exists():
        previous = json.loads(args.lock.read_text(encoding="utf-8"))
        if previous.get("status") == "pinned_and_verified":
            raise FileExistsError("refusing to replace an already pinned DPDP asset lock")
    from huggingface_hub import snapshot_download

    # Reuse the same immutable snapshot layout as the shared TDRL/PPDPP model cache.
    destination = args.output / args.model_id.split("/")[-1] / args.revision
    snapshot_download(
        repo_id=args.model_id,
        revision=args.revision,
        local_dir=destination,
        allow_patterns=("*.json", "*.txt", "*.safetensors", "*.model", "merges.txt", "vocab.json"),
    )
    hashes = directory_hashes(destination)
    if not any(name.endswith((".safetensors", ".bin")) for name in hashes):
        raise ValueError("downloaded DPDP encoder snapshot contains no weights")
    atomic_json(
        args.lock,
        {
            "schema_version": 1,
            "status": "pinned_and_verified",
            "model_id": args.model_id,
            "revision": args.revision,
            "encoder": hashes,
            "tokenizer": hashes,
        },
    )
    print(json.dumps({"path": str(destination), "files": len(hashes)}, indent=2))


if __name__ == "__main__":
    main()
