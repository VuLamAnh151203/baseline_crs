#!/usr/bin/env python3
"""Download and lock the two native EPL encoders; never performs inference."""

import argparse
import json
import re
from pathlib import Path

from tdrl.baselines.epl.assets import directory_hashes, verify_assets
from tdrl.provenance import atomic_json

MODELS = {
    "policy_encoder": (
        "google-bert/bert-base-cased",
        "cd5ef92a9fb2f889e972770a36d4ed042daf221e",
        "bert-base-cased",
    ),
    "retrieval_encoder": (
        "sentence-transformers/all-MiniLM-L6-v2",
        "c9745ed1d9f207416be6d2e6f8de32d1f16199bf",
        "all-MiniLM-L6-v2",
    ),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    args = parser.parse_args()
    destinations = {
        name: args.output / slug / revision
        for name, (_, revision, slug) in MODELS.items()
    }
    if args.lock.exists():
        verify_assets(
            args.lock,
            destinations["policy_encoder"],
            destinations["retrieval_encoder"],
        )
        print(json.dumps({"status": "already_verified", "lock": str(args.lock)}))
        return
    from huggingface_hub import snapshot_download

    records = {}
    for name, (model_id, revision, _) in MODELS.items():
        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError("EPL bootstrap requires immutable full commit revisions")
        destination = destinations[name]
        snapshot_download(
            repo_id=model_id,
            revision=revision,
            local_dir=destination,
            allow_patterns=(
                "*.json",
                "*.txt",
                "*.safetensors",
                "*.bin",
                "*.model",
                "*.py",
                "modules.json",
            ),
        )
        hashes = directory_hashes(destination)
        if not any(path.endswith((".safetensors", ".bin")) for path in hashes):
            raise ValueError(f"downloaded {name} has no model weights")
        records[name] = {
            "model_id": model_id,
            "revision": revision,
            "files": hashes,
        }
    atomic_json(
        args.lock,
        {"schema_version": 1, "status": "pinned_and_verified", **records},
    )
    verify_assets(
        args.lock,
        destinations["policy_encoder"],
        destinations["retrieval_encoder"],
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "lock": str(args.lock),
                "paths": {name: str(path) for name, path in destinations.items()},
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
