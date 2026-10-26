"""Command-line entry points for isolated ProCoT evaluation workers."""

import argparse
import json
from pathlib import Path

from ...comparison_procot import ProCoTComparison
from .selection import selection_payload, write_selection


def _common(command):
    command.add_argument("--root", type=Path, required=True)
    command.add_argument("--comparison-config", type=Path, required=True)
    command.add_argument("--source-lock", type=Path, required=True)


def main():
    parser = argparse.ArgumentParser(description="ProCoT shared-setting evaluation")
    sub = parser.add_subparsers(dest="command", required=True)
    command = sub.add_parser("evaluate")
    _common(command)
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--split", choices=("dev", "test"), required=True)
    command.add_argument("--selection-lock", type=Path)
    command.add_argument("--runtime-config", type=Path)
    command.add_argument("--fixed-test-setting", type=Path)
    command.add_argument("--fixed-test-count", type=int, choices=(400, 1000))
    command.add_argument(
        "--dataset", action="append", choices=("TopDial", "DuRecDial2.0")
    )
    command = sub.add_parser("lock-selection")
    _common(command)
    command.add_argument("--development-evidence", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "evaluate":
        result = ProCoTComparison(
            args.root,
            args.output,
            args.comparison_config,
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
            args.source_lock,
            args.development_evidence,
        )
        result = {
            "path": str(args.output),
            "sha256": write_selection(args.output, payload),
        }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
