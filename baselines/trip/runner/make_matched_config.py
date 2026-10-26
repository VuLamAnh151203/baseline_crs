import argparse
import json
from dataclasses import asdict
from pathlib import Path

from tdrl.baselines.trip.config import TripConfig
from tdrl.provenance import atomic_json, file_hash


def main():
    parser = argparse.ArgumentParser(description="Create a compute-matched TRIP decoding candidate")
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--beam-size", type=int, required=True)
    parser.add_argument("--max-plan-tokens", type=int, required=True)
    args = parser.parse_args()
    base = TripConfig.from_path(args.base)
    raw = asdict(base)
    raw["decoding"]["beam_size"] = args.beam_size
    raw["decoding"]["max_plan_tokens"] = args.max_plan_tokens
    candidate = TripConfig(
        schema_version=raw["schema_version"],
        model=type(base.model)(**raw["model"]),
        training=type(base.training)(**raw["training"]),
        decoding=type(base.decoding)(**raw["decoding"]),
        max_context_tokens=raw["max_context_tokens"],
        max_plan_tokens=raw["max_plan_tokens"],
    )
    if candidate.training_sha256 != base.training_sha256:
        raise ValueError("matched budget must not change fitted TRIP weights")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError("refusing to overwrite a TRIP matched-budget candidate")
    args.output.write_text(json.dumps(raw, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    atomic_json(
        args.output.with_suffix(args.output.suffix + ".manifest.json"),
        {
            "schema_version": 1,
            "status": "candidate_not_selected",
            "base_sha256": file_hash(args.base),
            "output_sha256": file_hash(args.output),
            "training_sha256": candidate.training_sha256,
            "config_sha256": candidate.sha256,
            "selection_rule": "closest development H100-seconds without test access",
        },
    )


if __name__ == "__main__":
    main()
