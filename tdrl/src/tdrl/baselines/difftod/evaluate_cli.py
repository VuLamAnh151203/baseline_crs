import argparse
import json
from pathlib import Path

from ...comparison_difftod import DiffTODComparison
from .config import DiffTODConfig
from .selection import selection_payload, write_selection


def main():
    parser = argparse.ArgumentParser(description="DiffTOD shared-setting evaluation")
    sub = parser.add_subparsers(dest="command", required=True)
    command = sub.add_parser("evaluate")
    command.add_argument("--root", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--config", type=Path, required=True)
    command.add_argument("--plans", type=Path, required=True)
    command.add_argument("--checkpoint-topdial", type=Path, required=True)
    command.add_argument("--checkpoint-durecdial", type=Path, required=True)
    command.add_argument("--natural-config-topdial", type=Path, required=True)
    command.add_argument("--matched-config-topdial", type=Path, required=True)
    command.add_argument("--natural-config-durecdial", type=Path, required=True)
    command.add_argument("--matched-config-durecdial", type=Path, required=True)
    command.add_argument("--split", choices=("dev", "test"), required=True)
    command.add_argument("--selection-lock", type=Path)
    command.add_argument("--fixed-test-setting", type=Path)
    command.add_argument("--fixed-test-count", type=int, choices=(400, 1000))
    command = sub.add_parser("lock-selection")
    command.add_argument("--config", type=Path, required=True)
    command.add_argument("--checkpoint-topdial", type=Path, required=True)
    command.add_argument("--checkpoint-durecdial", type=Path, required=True)
    command.add_argument("--natural-config-topdial", type=Path, required=True)
    command.add_argument("--matched-config-topdial", type=Path, required=True)
    command.add_argument("--natural-config-durecdial", type=Path, required=True)
    command.add_argument("--matched-config-durecdial", type=Path, required=True)
    command.add_argument("--development-evidence", type=Path, required=True)
    command.add_argument("--base-selection", type=Path)
    command.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "lock-selection":
        checkpoints = {
            "TopDial": args.checkpoint_topdial,
            "DuRecDial2.0": args.checkpoint_durecdial,
        }
        plan_configs = {
            "TopDial": {
                "natural": args.natural_config_topdial,
                "matched": args.matched_config_topdial,
            },
            "DuRecDial2.0": {
                "natural": args.natural_config_durecdial,
                "matched": args.matched_config_durecdial,
            },
        }
        plan_config_sha256 = {
            dataset: {
                budget: DiffTODConfig.from_path(path).sha256
                for budget, path in paths.items()
            }
            for dataset, paths in plan_configs.items()
        }
        payload = selection_payload(
            args.config,
            checkpoints,
            plan_config_sha256,
            args.development_evidence,
            args.base_selection,
        )
        result = {"sha256": write_selection(args.output, payload), "path": str(args.output)}
    else:
        checkpoints = {
            "TopDial": args.checkpoint_topdial,
            "DuRecDial2.0": args.checkpoint_durecdial,
        }
        plan_configs = {
            "TopDial": {
                "natural": args.natural_config_topdial,
                "matched": args.matched_config_topdial,
            },
            "DuRecDial2.0": {
                "natural": args.natural_config_durecdial,
                "matched": args.matched_config_durecdial,
            },
        }
        runner = DiffTODComparison(
            args.root,
            args.output,
            args.config,
            args.plans,
            checkpoints,
            plan_configs,
            args.selection_lock,
            args.fixed_test_setting,
            args.fixed_test_count,
        )
        result = runner.run(args.split)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
