import argparse
import json
from dataclasses import asdict
from pathlib import Path

from tdrl.baselines.dpdp.config import DPDPConfig
from tdrl.provenance import atomic_json, file_hash


def main():
    parser = argparse.ArgumentParser(description="Create a compute-matched DPDP search candidate")
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--budget", choices=("natural", "matched"), default="matched")
    parser.add_argument("--simulations", type=int, required=True)
    parser.add_argument("--depth", type=int, required=True)
    parser.add_argument("--gate-threshold", type=float)
    args = parser.parse_args()
    if args.budget == "matched":
        if args.simulations not in {2, 4, 6} or args.depth not in {1, 2, 3}:
            raise ValueError("matched DPDP grid is simulations=[2,4,6], depth=[1,2,3]")
    elif (args.simulations, args.depth) != (10, 5):
        raise ValueError("natural DPDP budget is fixed at simulations=10, depth=5")
    base = DPDPConfig.from_path(args.base)
    raw = asdict(base)
    raw["mcts_simulations"] = args.simulations
    raw["mcts_max_depth"] = args.depth
    if args.budget == "matched" and args.gate_threshold not in {None, base.gate_threshold}:
        raise ValueError("matched budget must keep the development-selected gate threshold")
    if args.gate_threshold is not None:
        raw["gate_threshold"] = args.gate_threshold
    raw["actions"] = list(raw["actions"])
    candidate = DPDPConfig(**{**raw, "actions": tuple(raw["actions"])})
    if candidate.training_sha256 != base.training_sha256:
        raise ValueError("matched budget must not change fitted DPDP weights")
    if args.output.exists():
        raise FileExistsError("refusing to overwrite a DPDP matched-budget candidate")
    args.output.parent.mkdir(parents=True, exist_ok=True)
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
            "budget": args.budget,
            "selection_rule": "closest development H100-seconds without test access",
        },
    )


if __name__ == "__main__":
    main()
