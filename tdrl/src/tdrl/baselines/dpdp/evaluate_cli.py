import argparse
import json
from pathlib import Path

import torch

from ...comparison_dpdp import DPDPComparison
from .config import DPDPConfig
from .selection import selection_payload, write_selection


def _artifacts(args):
    checkpoints = {
        "TopDial": args.checkpoint_topdial,
        "DuRecDial2.0": args.checkpoint_durecdial,
    }
    configs = {
        "TopDial": {"natural": args.natural_config_topdial, "matched": args.matched_config_topdial},
        "DuRecDial2.0": {
            "natural": args.natural_config_durecdial,
            "matched": args.matched_config_durecdial,
        },
    }
    return checkpoints, configs


def _arguments(command):
    command.add_argument("--root", type=Path, required=True)
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
    command.add_argument("--runtime-config", type=Path)


def main():
    parser = argparse.ArgumentParser(description="DPDP shared-setting evaluation")
    sub = parser.add_subparsers(dest="command", required=True)
    command = sub.add_parser("evaluate")
    _arguments(command)
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--split", choices=("dev", "test"), required=True)
    command.add_argument("--selection-lock", type=Path)
    command.add_argument("--fixed-test-setting", type=Path)
    command.add_argument("--fixed-test-count", type=int, choices=(400, 1000))
    command.add_argument(
        "--dataset",
        action="append",
        choices=("TopDial", "DuRecDial2.0"),
        help="Evaluate only this dataset branch; repeat for more than one.",
    )
    command = sub.add_parser("lock-selection")
    _arguments(command)
    command.add_argument("--development-evidence", type=Path, required=True)
    command.add_argument("--base-selection", type=Path)
    command.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    checkpoints, configs = _artifacts(args)
    if args.command == "evaluate":
        result = DPDPComparison(
            args.root,
            args.output,
            args.config,
            checkpoints,
            configs,
            args.encoder,
            args.tokenizer,
            args.model_lock,
            args.selection_lock,
            runtime_config=args.runtime_config,
            fixed_test_setting=args.fixed_test_setting,
            fixed_test_count=args.fixed_test_count,
        ).run(args.split, datasets=args.dataset)
    else:
        parsed = {
            dataset: {budget: DPDPConfig.from_path(path) for budget, path in values.items()}
            for dataset, values in configs.items()
        }
        config_hashes = {
            dataset: {budget: value.sha256 for budget, value in values.items()}
            for dataset, values in parsed.items()
        }
        offline_checkpoint_sha256 = {}
        for dataset, path in checkpoints.items():
            payload = torch.load(path, map_location="cpu", weights_only=False)
            parent = payload.get("metadata", {}).get("parent_checkpoint_sha256")
            if not isinstance(parent, str) or len(parent) != 64:
                raise ValueError("DPDP final checkpoint lacks offline parent provenance")
            offline_checkpoint_sha256[dataset] = parent
        training_hashes = {}
        for dataset, values in parsed.items():
            if values["natural"].gate_threshold != values["matched"].gate_threshold:
                raise ValueError("DPDP natural/matched budgets must share the frozen gate")
            hashes = {value.training_sha256 for value in values.values()}
            if len(hashes) != 1:
                raise ValueError("DPDP natural/matched configs changed fitted weights")
            training_hashes[dataset] = hashes.pop()
        gate_thresholds = {
            dataset: {
                budget: value.gate_threshold for budget, value in values.items()
            }
            for dataset, values in parsed.items()
        }
        search_budgets = {
            dataset: {
                budget: {
                    "simulations": value.mcts_simulations,
                    "depth": value.mcts_max_depth,
                }
                for budget, value in values.items()
            }
            for dataset, values in parsed.items()
        }
        payload = selection_payload(
            args.root,
            args.config,
            checkpoints,
            offline_checkpoint_sha256,
            config_hashes,
            training_hashes,
            gate_thresholds,
            search_budgets,
            args.model_lock,
            args.development_evidence,
            args.base_selection,
        )
        result = {"path": str(args.output), "sha256": write_selection(args.output, payload)}
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
