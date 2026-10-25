"""Create an explicit DiffTOD sampling candidate for H100 budget matching."""

import argparse
import hashlib
import json
import os
from pathlib import Path


def sha256(path):
    value = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def digest(value):
    encoded = json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def exclusive_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--candidates", type=int, choices=range(1, 6), required=True)
    args = parser.parse_args()
    if args.steps < 1:
        raise ValueError("matched DiffTOD steps must be positive")
    manifest_path = args.output.with_suffix(args.output.suffix + ".manifest.json")
    if args.output.exists() or manifest_path.exists():
        raise FileExistsError("refusing to replace a DiffTOD matched-budget candidate")
    config = json.loads(args.base.read_text(encoding="utf-8"))
    if set(config) != {"schema_version", "model", "training", "sampling"}:
        raise ValueError("base DiffTOD configuration is incomplete")
    natural_work = config["sampling"]["steps"] * config["sampling"]["candidates"]
    candidate_work = args.steps * args.candidates
    if candidate_work >= natural_work:
        raise ValueError("matched candidate must be cheaper than the natural configuration")
    config["sampling"]["steps"] = args.steps
    config["sampling"]["candidates"] = args.candidates
    exclusive_json(args.output, config)
    manifest = {
        "schema_version": 1,
        "status": "candidate_not_selected",
        "base_sha256": sha256(args.base),
        "config_file_sha256": sha256(args.output),
        "config_sha256": digest(config),
        "steps": args.steps,
        "candidates": args.candidates,
        "sequence_forward_proxy": candidate_work,
        "selection_rule": "measure on development H100; freeze only through selection lock",
    }
    exclusive_json(manifest_path, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
