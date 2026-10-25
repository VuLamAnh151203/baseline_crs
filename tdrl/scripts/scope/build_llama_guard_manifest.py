#!/usr/bin/env python3
"""Create a local hash manifest for an already authorized Llama Guard snapshot."""

import argparse
import json
from pathlib import Path

from tdrl.baselines.scope.assets import LLAMA_GUARD_ID, LLAMA_GUARD_REVISION
from tdrl.provenance import atomic_json, file_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    snapshot = args.snapshot.resolve()
    files = {
        "config.json",
        "model.safetensors.index.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        *(f"model-{index:05d}-of-00004.safetensors" for index in range(1, 5)),
    }
    missing = [str(snapshot / name) for name in sorted(files) if not (snapshot / name).is_file()]
    if missing:
        raise FileNotFoundError("missing Llama Guard snapshot files: " + ", ".join(missing))
    payload = {
        "schema_version": 1,
        "model_id": LLAMA_GUARD_ID,
        "revision": LLAMA_GUARD_REVISION,
        "files": {
            name: {
                "bytes": (snapshot / name).stat().st_size,
                "sha256": file_hash(snapshot / name),
            }
            for name in sorted(files)
        },
    }
    atomic_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
