#!/usr/bin/env python3
"""Run native EnPL/ProCoT under the pinned external CARP evaluator."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

from tdrl.backend import verify_model_cache
from tdrl.carp_current import AdapterFactory, SUPPORTED_METHODS
from tdrl.contracts import ModelProvenance
from tdrl.provenance import atomic_json, digest, file_hash

PINNED_CARP_COMMIT = "e5bf7034d7526bcc9cf6869666e8bebc80d36cc0"
CARP_FILES = {
    "src/carp/harness/prompts.py": "1a0b17b4556d6827d697bfdf4c7ebff845ead5108d96445d71e09cd55ed14da0",
    "src/carp/harness/simulator.py": "f3de61150b752f1e54dbb8256a375dc2ac73935a39438d74bb96e82a237f73a2",
    "src/carp/harness/runner.py": "4ef1fb2b241dac7931a200b22b507181064155d0cc46482cb58d0992addd7a0f",
    "src/carp/harness/quality.py": "a8cb500b2746e3b39155971a5d61f3b286107645bd12ce80c6970d0598ad0650",
}
DATASETS = {
    "TopDial": {
        "filename": "topdial_test200_seed2911.jsonl",
        "kind": "topdial",
        "sha256": "bcb3a8e56f2d767eb0f544b4dbeed23f57e051f4d99f0048e7fe6e717b693d4c",
        "min_success_turn": 1,
    },
    "DuRecDial2.0": {
        "filename": "durecdial_test200_seed2911.jsonl",
        "kind": "durecdial",
        "sha256": "6d8fdc3e9e6d68581224e59789946e5de4b75717276a1c6d330cbe0afaff572e",
        "min_success_turn": 2,
    },
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def git_commit(path: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()


def verify_carp(carp_root: Path) -> dict[str, Any]:
    commit = git_commit(carp_root)
    if commit != PINNED_CARP_COMMIT:
        raise ValueError(f"CARP commit mismatch: {commit}")
    observed = {name: file_hash(carp_root / name) for name in CARP_FILES}
    if observed != CARP_FILES:
        raise ValueError("CARP evaluator source hash mismatch")
    datasets = {}
    for label, expected in DATASETS.items():
        path = carp_root / "data/eval" / expected["filename"]
        observed_hash = file_hash(path)
        if observed_hash != expected["sha256"]:
            raise ValueError(f"CARP dataset hash mismatch: {label}")
        count = sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
        if count != 200:
            raise ValueError(f"CARP dataset count mismatch: {label}")
        datasets[label] = {"path": str(path.resolve()), "count": count, **expected}
    return {"commit": commit, "source_sha256": observed, "datasets": datasets}


def verify_assets(args: argparse.Namespace) -> None:
    root = args.root.resolve()
    carp_root = args.carp_root.resolve()
    snapshots = verify_model_cache(root / "configs/models.lock.json", args.model_root.resolve())
    generation = snapshots["Qwen/Qwen3.5-9B"]
    carp = verify_carp(carp_root)
    enpl_bank = args.enpl_artifact.resolve() / "bank/plan-bank.jsonl"
    enpl_manifest = args.enpl_artifact.resolve() / "bank/bank-manifest.json"
    payload = {
        "schema_version": 1,
        "status": "verified",
        "integration": "native_enpl_procot_to_carp_current_v1",
        "baseline_commit": git_commit(root.parent),
        "carp": carp,
        "generation_model": generation,
        "enpl": {
            "bank_path": str(enpl_bank),
            "bank_sha256": file_hash(enpl_bank),
            "manifest_path": str(enpl_manifest),
            "manifest_sha256": file_hash(enpl_manifest),
        },
        "evaluation": {
            "max_turns": 10,
            "sample_seed": 42,
            "experimental_root_seed": 17,
            "simulator_moderator_temperature": 0.0,
            "simulator_moderator_max_output_tokens": 3072,
            "thinking": "disabled",
            "quality_scale": [0, 3],
        },
    }
    payload["contract_sha256"] = digest(payload)
    atomic_json(args.output.resolve(), payload)


def generation_provenance(manifest: dict[str, Any]) -> ModelProvenance:
    row = manifest["generation_model"]
    return ModelProvenance(
        row["model_id"],
        row["revision"],
        row["weight_sha256"],
        row["template_sha256"],
    )


def evaluate(args: argparse.Namespace) -> None:
    import carp.harness.runner as carp_runner

    manifest = read_json(args.asset_manifest.resolve())
    if manifest.get("status") != "verified" or manifest.get("contract_sha256") != digest(
        {key: value for key, value in manifest.items() if key != "contract_sha256"}
    ):
        raise ValueError("invalid asset manifest")
    current_carp = verify_carp(args.carp_root.resolve())
    if current_carp != manifest["carp"]:
        raise ValueError("CARP checkout changed after asset verification")
    dataset = DATASETS[args.dataset]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    contract = {
        "schema_version": 1,
        "integration": "native_enpl_procot_to_carp_current_v1",
        "asset_contract_sha256": manifest["contract_sha256"],
        "method": args.method,
        "dataset": args.dataset,
        "dataset_sha256": dataset["sha256"],
        "count": args.count,
        "root_seed": 17,
        "carp_sample_seed": 42,
        "carp_commit": PINNED_CARP_COMMIT,
    }
    factory = AdapterFactory(
        project_root=args.root.resolve(),
        output_root=output,
        method=args.method,
        model_provenance=generation_provenance(manifest),
        generation_url=args.generation_url,
        enpl_artifact=args.enpl_artifact.resolve(),
        root_seed=17,
        contract=contract,
    )
    original_builder = carp_runner.build_assistant_adapter

    def build_adapter(name: str, **kwargs: Any):
        if name == args.method:
            return factory.build(opening_adapter=original_builder("llm", **kwargs))
        return original_builder(name, **kwargs)

    carp_runner.build_assistant_adapter = build_adapter
    config = carp_runner.EvalConfig(
        dataset_path=args.carp_root.resolve() / "data/eval" / dataset["filename"],
        dataset_kind=dataset["kind"],
        output_dir=output,
        assistant_adapter=args.method,
        assistant_model="Qwen/Qwen3.5-9B",
        user_model="Qwen/Qwen3.5-9B",
        moderator_model="Qwen/Qwen3.5-9B",
        max_turns=10,
        min_success_turn=dataset["min_success_turn"],
        sample_size=args.count,
        sample_seed=42,
        resume=True,
        temperature=0.0,
        max_output_tokens=3072,
        thinking_budget=0,
        thinking_level="none",
        request_timeout_seconds=180.0,
        max_retries=40,
        retry_backoff_seconds=10.0,
        llm_base_url=args.generation_url,
        assistant_llm_base_url=args.generation_url,
        embedding_base_url=args.embedding_url,
        num_workers=args.workers,
    )
    summary = carp_runner.run_evaluation(config)
    if summary["num_scenarios"] != args.count or summary["error_count"] != 0:
        raise RuntimeError(f"incomplete CARP evaluation: {summary}")
    atomic_json(output / "integration-contract.json", contract)


def finalize(args: argparse.Namespace) -> None:
    manifest = read_json(args.asset_manifest.resolve())
    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "complete",
        "asset_contract_sha256": manifest["contract_sha256"],
        "count_per_dataset": args.count,
        "methods": {},
    }
    for method in SUPPORTED_METHODS:
        report["methods"][method] = {}
        for dataset in DATASETS:
            run_dir = args.run_root.resolve() / method / dataset
            summary = read_json(run_dir / "summary.json")
            rows = [
                json.loads(line)
                for line in (run_dir / "results.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            if (
                summary.get("num_scenarios") != args.count
                or summary.get("error_count") != 0
                or len(rows) != args.count
                or len({row["scenario_id"] for row in rows}) != args.count
            ):
                raise ValueError(f"incomplete or duplicate results: {method}/{dataset}")
            quality = read_json(args.run_root.resolve() / "quality" / dataset / "summary.json")
            quality_row = quality["runs"][method]
            if quality_row["num_dialogues"] != args.count or quality_row["num_scored"] != args.count:
                raise ValueError(f"incomplete quality results: {method}/{dataset}")
            report["methods"][method][dataset] = {
                **summary,
                "ovr_mean": quality_row["ovr_mean"],
                "results_sha256": file_hash(run_dir / "results.jsonl"),
            }
    report["report_sha256"] = digest(report)
    atomic_json(args.run_root.resolve() / "report.json", report)


def parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--root", type=Path, required=True)
    common.add_argument("--carp-root", type=Path, required=True)
    common.add_argument("--enpl-artifact", type=Path, required=True)
    command = argparse.ArgumentParser()
    sub = command.add_subparsers(dest="command", required=True)
    verify = sub.add_parser("verify-assets", parents=[common])
    verify.add_argument("--model-root", type=Path, required=True)
    verify.add_argument("--output", type=Path, required=True)
    verify.set_defaults(handler=verify_assets)
    run = sub.add_parser("evaluate", parents=[common])
    run.add_argument("--asset-manifest", type=Path, required=True)
    run.add_argument("--method", choices=SUPPORTED_METHODS, required=True)
    run.add_argument("--dataset", choices=DATASETS, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--count", type=int, required=True)
    run.add_argument("--workers", type=int, default=16)
    run.add_argument("--generation-url", default="http://127.0.0.1:8000/v1")
    run.add_argument("--embedding-url", default="http://127.0.0.1:8001/v1")
    run.set_defaults(handler=evaluate)
    final = sub.add_parser("finalize")
    final.add_argument("--asset-manifest", type=Path, required=True)
    final.add_argument("--run-root", type=Path, required=True)
    final.add_argument("--count", type=int, required=True)
    final.set_defaults(handler=finalize)
    return command


if __name__ == "__main__":
    parsed = parser().parse_args()
    if parsed.command == "evaluate" and not 1 <= parsed.count <= 200:
        raise SystemExit("--count must be in [1, 200]")
    parsed.handler(parsed)
