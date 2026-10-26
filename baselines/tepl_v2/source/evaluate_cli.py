import argparse
import json
from pathlib import Path

from ...comparison_epl import EPLComparison
from .selection import selection_payload, write_selection


def _arguments(command):
    command.add_argument("--root", type=Path, required=True)
    command.add_argument("--comparison-config", type=Path, required=True)
    command.add_argument("--checkpoint-topdial", type=Path, required=True)
    command.add_argument("--checkpoint-durecdial", type=Path, required=True)
    command.add_argument("--epl-config-topdial", type=Path, required=True)
    command.add_argument("--epl-config-durecdial", type=Path, required=True)
    command.add_argument("--memory-topdial", type=Path, required=True)
    command.add_argument("--memory-durecdial", type=Path, required=True)
    command.add_argument("--vectors-topdial", type=Path, required=True)
    command.add_argument("--vectors-durecdial", type=Path, required=True)
    command.add_argument("--encoder", type=Path, required=True)
    command.add_argument("--tokenizer", type=Path, required=True)
    command.add_argument("--retrieval-encoder", type=Path, required=True)
    command.add_argument("--model-lock", type=Path, required=True)
    command.add_argument("--source-lock", type=Path, required=True)
    command.add_argument("--runtime-config", type=Path)


def _artifacts(args):
    return {
        "checkpoints": {
            "TopDial": args.checkpoint_topdial,
            "DuRecDial2.0": args.checkpoint_durecdial,
        },
        "configs": {
            "TopDial": args.epl_config_topdial,
            "DuRecDial2.0": args.epl_config_durecdial,
        },
        "memories": {
            "TopDial": args.memory_topdial,
            "DuRecDial2.0": args.memory_durecdial,
        },
        "vectors": {
            "TopDial": args.vectors_topdial,
            "DuRecDial2.0": args.vectors_durecdial,
        },
    }


def main():
    parser = argparse.ArgumentParser(description="EPL shared-setting evaluation")
    sub = parser.add_subparsers(dest="command", required=True)
    command = sub.add_parser("evaluate")
    _arguments(command)
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--split", choices=("dev", "test"), required=True)
    command.add_argument("--selection-lock", type=Path)
    command.add_argument("--fixed-test-setting", type=Path)
    command.add_argument("--fixed-test-count", type=int, choices=(400, 1000))
    command.add_argument(
        "--dataset", action="append", choices=("TopDial", "DuRecDial2.0")
    )
    command = sub.add_parser("lock-selection")
    _arguments(command)
    command.add_argument("--development-evidence", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    artifacts = _artifacts(args)
    if args.command == "evaluate":
        result = EPLComparison(
            args.root,
            args.output,
            args.comparison_config,
            artifacts["checkpoints"],
            artifacts["configs"],
            artifacts["memories"],
            artifacts["vectors"],
            args.encoder,
            args.tokenizer,
            args.retrieval_encoder,
            args.model_lock,
            args.source_lock,
            args.selection_lock,
            args.runtime_config,
            args.fixed_test_setting,
            args.fixed_test_count,
        ).run(args.split, args.dataset)
    else:
        payload = selection_payload(
            args.root,
            args.comparison_config,
            artifacts["configs"],
            artifacts["checkpoints"],
            artifacts["memories"],
            artifacts["vectors"],
            args.model_lock,
            args.source_lock,
            args.development_evidence,
        )
        result = {"path": str(args.output), "sha256": write_selection(args.output, payload)}
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
