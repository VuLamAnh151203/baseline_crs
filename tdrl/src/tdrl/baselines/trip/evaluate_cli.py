import argparse
import json
from pathlib import Path

from ...comparison_trip import TripComparison
from .config import TripConfig
from .selection import selection_payload, write_selection


def _artifacts(args):
    checkpoints = {
        "TopDial": args.checkpoint_topdial,
        "DuRecDial2.0": args.checkpoint_durecdial,
    }
    configs = {
        "TopDial": {
            "natural": args.natural_config_topdial,
            "matched": args.matched_config_topdial,
        },
        "DuRecDial2.0": {
            "natural": args.natural_config_durecdial,
            "matched": args.matched_config_durecdial,
        },
    }
    return checkpoints, configs


def _arguments(command):
    command.add_argument("--config", type=Path, required=True)
    command.add_argument("--checkpoint-topdial", type=Path, required=True)
    command.add_argument("--checkpoint-durecdial", type=Path, required=True)
    command.add_argument("--natural-config-topdial", type=Path, required=True)
    command.add_argument("--matched-config-topdial", type=Path, required=True)
    command.add_argument("--natural-config-durecdial", type=Path, required=True)
    command.add_argument("--matched-config-durecdial", type=Path, required=True)
    command.add_argument("--encoder", type=Path, required=True)
    command.add_argument("--tokenizer", type=Path, required=True)
    command.add_argument("--model-lock", type=Path, required=True)


def main():
    parser = argparse.ArgumentParser(description="TRIP shared-setting evaluation")
    sub = parser.add_subparsers(dest="command", required=True)
    command = sub.add_parser("evaluate")
    _arguments(command)
    command.add_argument("--root", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--split", choices=("dev", "test"), required=True)
    command.add_argument("--selection-lock", type=Path)
    command.add_argument("--fixed-test-setting", type=Path)
    command.add_argument("--fixed-test-count", type=int, choices=(400, 1000))
    command = sub.add_parser("lock-selection")
    _arguments(command)
    command.add_argument("--development-evidence", type=Path, required=True)
    command.add_argument("--base-selection", type=Path)
    command.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    checkpoints, configs = _artifacts(args)
    if args.command == "evaluate":
        result = TripComparison(
            args.root,
            args.output,
            args.config,
            checkpoints,
            configs,
            args.encoder,
            args.tokenizer,
            args.model_lock,
            args.selection_lock,
            args.fixed_test_setting,
            args.fixed_test_count,
        ).run(args.split)
    else:
        config_hashes = {
            dataset: {
                budget: TripConfig.from_path(path).sha256
                for budget, path in values.items()
            }
            for dataset, values in configs.items()
        }
        payload = selection_payload(
            args.config,
            checkpoints,
            config_hashes,
            args.model_lock,
            args.development_evidence,
            args.base_selection,
        )
        result = {"path": str(args.output), "sha256": write_selection(args.output, payload)}
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
