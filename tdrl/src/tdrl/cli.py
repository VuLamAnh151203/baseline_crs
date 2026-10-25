import argparse
import json
from pathlib import Path

from .audit import dependency_audit, static_audit
from .backend import verify_model_cache
from .budget import experiment_matrix, feasibility
from .data import prepare_canonical
from .provenance import atomic_json, validate_manifest
from .runtime import smoke


def main():
    parser = argparse.ArgumentParser(description="TDRL (temporary method name)")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("check")
    sub.add_parser("runtime-audit")
    sub.add_parser("verify-models")
    p = sub.add_parser("matrix")
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("budget")
    p.add_argument("--measurements", type=Path, required=True)
    p.add_argument("--remaining-hours", type=float, required=True)
    p = sub.add_parser("validate-manifest")
    p.add_argument("path", type=Path)
    p = sub.add_parser("prepare")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--split", required=True)
    p.add_argument("--source-manifest", type=Path, required=True)
    p.add_argument("--selection", type=Path)
    p.add_argument("--selection-sha256")
    p = sub.add_parser("smoke")
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("comparison")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--config", type=Path, default=Path("configs/comparison-judge-v2.json"))
    p.add_argument("--fixed-test-setting", type=Path)
    p.add_argument("--fixed-test-count", type=int, choices=(400, 1000))
    p = sub.add_parser("recover-comparison")
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--audit", type=Path, required=True)
    p.add_argument("--config", type=Path, default=Path("configs/comparison.json"))
    p = sub.add_parser("continue-selection")
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--config", type=Path, default=Path("configs/comparison.json"))
    p = sub.add_parser("repair-development")
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--config", type=Path, default=Path("configs/comparison.json"))
    p = sub.add_parser("repair-ppdpp")
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--config", type=Path, default=Path("configs/comparison-judge-v2.json"))
    p = sub.add_parser("parallel-comparison")
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--config", type=Path, default=Path("configs/comparison-judge-v2-sample2000-seed17.json")
    )
    p = sub.add_parser("two-gpu-task")
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--shared", type=Path, required=True)
    p.add_argument("--slot", type=int, choices=[0, 1], required=True)
    p.add_argument(
        "--mode",
        choices=[
            "bootstrap",
            "offline",
            "policy",
            "policy-dev",
            "select-shard",
            "lock",
            "test",
            "report",
        ],
        required=True,
    )
    p.add_argument("--dataset", choices=["TopDial", "DuRecDial2.0"])
    p.add_argument("--role", choices=["policy", "full"])
    p.add_argument(
        "--config", type=Path, default=Path("configs/comparison-judge-v2-sample500-seed17.json")
    )
    args = parser.parse_args()
    root = args.root.resolve()
    if args.command == "check":
        errors = static_audit(root)
        print(json.dumps({"status": "failed" if errors else "passed", "errors": errors}, indent=2))
        raise SystemExit(bool(errors))
    if args.command == "runtime-audit":
        report = dependency_audit(root / "runtime/uv.lock", allow_local_vllm=True)
        print(json.dumps(report, indent=2))
        raise SystemExit(report["status"] != "passed")
    if args.command == "matrix":
        cells = experiment_matrix(
            {"TopDial": ["test_seen", "test_unseen"], "DuRecDial2.0": ["test"]}
        )
        atomic_json(
            args.output,
            {
                "schema_version": 1,
                "status": "proposed",
                "split_topology": "requires_official_audit",
                "cells": cells,
            },
        )
        print(f"Wrote {len(cells)} proposed cells; counts remain unverified.")
    elif args.command == "budget":
        data = json.loads(args.measurements.read_text())
        print(
            json.dumps(
                feasibility(
                    data["cells"],
                    data["counts"],
                    data["seconds_per_dialogue"],
                    args.remaining_hours,
                ),
                indent=2,
            )
        )
    elif args.command == "verify-models":
        print(
            json.dumps(
                verify_model_cache(root / "configs/models.lock.json", root / "models"), indent=2
            )
        )
    elif args.command == "validate-manifest":
        validate_manifest(json.loads(args.path.read_text()))
        print("valid")
    elif args.command == "prepare":
        report = prepare_canonical(
            args.input,
            args.output,
            args.dataset,
            args.split,
            json.loads(args.source_manifest.read_text()),
            args.selection,
            args.selection_sha256,
        )
        print(json.dumps(report, indent=2))
    elif args.command == "smoke":
        smoke(root, args.output)
    elif args.command == "comparison":
        from .comparison import Comparison

        Comparison(
            root,
            args.output,
            root / args.config,
            args.fixed_test_setting,
            args.fixed_test_count,
        ).run()

    elif args.command == "recover-comparison":
        from .recovery import AuditedRecovery

        AuditedRecovery(root, args.output, root / args.config, args.source, args.audit).run()

    elif args.command == "continue-selection":
        from .recovery import SelectionContinuation

        SelectionContinuation(root, args.output, root / args.config, args.source).run()

    elif args.command == "repair-development":
        from .recovery import SelectionContinuation

        SelectionContinuation(root, args.output, root / args.config, args.source, repair=True).run()
    elif args.command == "repair-ppdpp":
        from .fidelity import FidelityRepair

        FidelityRepair(root, args.output, root / args.config, args.source).run()
    elif args.command == "parallel-comparison":
        from .parallel_comparison import ParallelComparison

        ParallelComparison(root, args.output, root / args.config, args.source).run()

    elif args.command == "two-gpu-task":
        from .two_gpu import run_task

        run_task(
            root,
            args.output,
            root / args.config,
            args.source,
            args.shared,
            args.slot,
            args.mode,
            args.dataset,
            args.role,
        )


if __name__ == "__main__":
    main()
