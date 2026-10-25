#!/usr/bin/env python3
"""Run one CARP-legacy baseline shard or coordinate both dataset shards."""

from __future__ import annotations

import argparse
import gc
import json
import os
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

from tdrl.baselines.direct_llm.planner import planner_for_method
from tdrl.baselines.dpdp.planner import DPDPPlanner
from tdrl.baselines.dpdp.runtime import load_inference as load_dpdp
from tdrl.baselines.enpl.bank import load_bank
from tdrl.baselines.enpl.config import load_enpl_config
from tdrl.baselines.enpl.planner import EnPLPlanner
from tdrl.baselines.enpl.retrieval import DomainEntityOverlapRetriever
from tdrl.baselines.epl.memory import ExperienceMemory, NativeMiniLMEmbedding
from tdrl.baselines.epl.planner import EPLPlanner
from tdrl.baselines.epl.runtime import load_inference as load_epl
from tdrl.baselines.procot.planner import ProCoTPlanner
from tdrl.baselines.scope.config import ScopeConfig
from tdrl.baselines.scope.planner import ScopeLitePlanner
from tdrl.baselines.scope.reward import load_reward_checkpoint
from tdrl.baselines.scope.semantic import LlamaGuardSemanticEncoder
from tdrl.baselines.scope.transition import TransitionEnsemble
from tdrl.carp_legacy import (
    DATASETS,
    PROTOCOL,
    RESOURCE_CONTRACT,
    load_carp_scenarios,
    prompt_contract_hash,
    run_carp_dialogue,
    summarize_carp_results,
)
from tdrl.contracts import Decoding
from tdrl.provenance import atomic_json, digest, file_hash
from tdrl.runtime import Servers

METHODS = (
    "direct_llm_v2",
    "tepl_v2",
    "enpl_domain3",
    "procot",
    "dpdp",
    "scope_crs_lite",
)
METHOD_LABELS = {
    "direct_llm_v2": "DirectLLM-v2",
    "tepl_v2": "T-EPL-v2",
    "enpl_domain3": "EnPL-domain3",
    "procot": "ProCoT",
    "dpdp": "DPDP",
    "scope_crs_lite": "SCOPE-CRS-lite",
}
ARTIFACTS = {
    "tepl_v2": Path("/mnt/data/u26anhvl/crs/epl/epl-fixed1000-60111"),
    "enpl_domain3": Path("/mnt/data/u26anhvl/crs/enpl/enpl-fixed1000-60429"),
    "dpdp": Path("/mnt/data/u26anhvl/crs/dpdp/deadline250-60714"),
    "scope_crs_lite": Path("/mnt/data/u26anhvl/crs/scope/scope-fixed1000-60461"),
}

SCOPE_ASSET_ROOT = Path("/mnt/data/u26anhvl/crs/scope/assets")
SCOPE_LLAMA_GUARD = (
    SCOPE_ASSET_ROOT
    / "Meta-Llama-Guard-2-8B"
    / "7d257f3c1a0ec6ed99b2cb715027149dfb9784ef"
)
SCOPE_TRANSITIONS = SCOPE_ASSET_ROOT / "transition_models/deterministic"


def now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def assistant_decoding(root: Path, method: str):
    config_name = {
        "direct_llm_v2": "comparison.direct_llm-v2.json",
        "tepl_v2": "comparison.epl.json",
        "enpl_domain3": "comparison.enpl.json",
        "procot": "comparison.procot.json",
        "dpdp": "comparison.dpdp.json",
        "scope_crs_lite": "comparison.scope-lite.json",
    }[method]
    return Decoding(**read_json(root / "configs" / config_name)["assistant_decoding"])


def build_planner_factory(root: Path, run_root: Path, method: str, dataset: str, generation):
    decoding = assistant_decoding(root, method)
    provenance = {
        "method": METHOD_LABELS[method],
        "assistant_decoding": as_plain(decoding),
    }
    if method == "direct_llm_v2":
        return (
            lambda scenario: planner_for_method(
                "DirectLLM-v2", generation, scenario.method_scenario(), dataset, decoding
            ),
            provenance,
        )
    if method == "procot":
        return (
            lambda scenario: ProCoTPlanner(
                generation, scenario.method_scenario(), decoding
            ),
            provenance,
        )
    if method == "enpl_domain3":
        source = ARTIFACTS[method]
        config_path = root / "configs/enpl.tdrl-adapted.json"
        entries, manifest = load_bank(
            source / "bank/plan-bank.jsonl",
            source / "bank/bank-manifest.json",
            require_complete=True,
        )
        config = load_enpl_config(config_path)
        retriever = DomainEntityOverlapRetriever(
            entries,
            config["retrieval"]["k"],
            config["retrieval"]["distinct_source_conversations"],
        )
        provenance.update(
            {
                "bank_sha256": file_hash(source / "bank/plan-bank.jsonl"),
                "bank_manifest_sha256": file_hash(source / "bank/bank-manifest.json"),
                "bank_status": manifest["status"],
                "method_config_sha256": file_hash(config_path),
            }
        )
        def make_enpl(scenario):
            return EnPLPlanner(
                generation,
                scenario.method_scenario(),
                retriever,
                Decoding(**config["plan_decoding"]),
                decoding,
                config["context"]["max_input_characters"],
            )

        return make_enpl, provenance
    if method == "tepl_v2":
        source = ARTIFACTS[method]
        config_path = root / "configs/epl" / ("topdial.json" if dataset == "TopDial" else "durecdial.json")
        checkpoint = source / "training" / dataset / "checkpoint-best.pt"
        memory_path = source / "memory" / dataset / "experiences.jsonl"
        vector_path = source / "memory" / dataset / "vectors.pt"
        lock_path = root / "configs/epl.models.lock.json"
        lock = read_json(lock_path)
        asset_root = Path("/mnt/data/u26anhvl/crs/epl/models")
        encoder = asset_root / "bert-base-cased" / lock["policy_encoder"]["revision"]
        retrieval = asset_root / "all-MiniLM-L6-v2" / lock["retrieval_encoder"]["revision"]
        model, tokenizer, config, _, _ = load_epl(
            config_path, checkpoint, encoder, encoder, retrieval, lock_path, "cuda"
        )
        embedding = NativeMiniLMEmbedding(
            encoder,
            retrieval,
            lock_path,
            run_root / "intrinsic-embedding-journal" / dataset,
        )
        memory = ExperienceMemory(memory_path, vector_path, embedding, config)
        comparison = read_json(root / "configs/comparison.epl.json")
        provenance.update(
            {
                "checkpoint_sha256": file_hash(checkpoint),
                "memory_sha256": file_hash(memory_path),
                "vectors_sha256": file_hash(vector_path),
                "method_config_sha256": file_hash(config_path),
            }
        )
        def make_tepl(scenario):
            return EPLPlanner(
                generation,
                model,
                tokenizer,
                memory,
                config,
                scenario.method_scenario(),
                provenance["checkpoint_sha256"],
                "profile",
                decoding,
                Decoding(**comparison["simulator_decoding"]),
            )

        return make_tepl, provenance
    if method == "dpdp":
        source = ARTIFACTS[method]
        config_path = source / "configs" / dataset / "natural.json"
        checkpoint = source / "selfplay" / dataset / "checkpoint-final.pt"
        lock_path = root / "configs/dpdp.models.lock.json"
        lock = read_json(lock_path)
        encoder = root / "models/roberta-large" / lock["revision"]
        model, tokenizer, config, _, _ = load_dpdp(
            config_path, checkpoint, encoder, encoder, lock_path, "cuda"
        )
        model.to(dtype=torch.bfloat16).eval()
        provenance.update(
            {
                "checkpoint_sha256": file_hash(checkpoint),
                "method_config_sha256": file_hash(config_path),
            }
        )
        def make_dpdp(scenario):
            return DPDPPlanner(
                generation,
                model,
                tokenizer,
                config,
                scenario.method_scenario(),
                provenance["checkpoint_sha256"],
                decoding,
                "profile",
            )

        return make_dpdp, provenance
    if method == "scope_crs_lite":
        source = ARTIFACTS[method]
        config_name = "topdial.json" if dataset == "TopDial" else "durecdial.json"
        config_path = root / "configs/scope" / config_name
        reward_path = source / "reward-training" / dataset / "checkpoint-best.pt"
        llama_manifest = SCOPE_LLAMA_GUARD / "scope-asset-manifest.json"
        config = ScopeConfig.from_path(config_path)
        encoder = LlamaGuardSemanticEncoder(
            SCOPE_LLAMA_GUARD,
            llama_manifest,
            output_dim=config.embedding_dim,
            projection_seed=config.projection_seed,
        )
        transitions = TransitionEnsemble(
            SCOPE_TRANSITIONS, dimension=config.embedding_dim
        )
        reward, reward_metadata = load_reward_checkpoint(reward_path, config)
        provenance.update(
            {
                "reward_checkpoint_sha256": reward_metadata["sha256"],
                "method_config_sha256": file_hash(config_path),
                "llama_guard_manifest_sha256": file_hash(llama_manifest),
                "transition_manifest_sha256": file_hash(
                    SCOPE_TRANSITIONS / "manifest.json"
                ),
                "runtime_override": {
                    "gpu_memory_utilization": 0.48,
                    "max_num_seqs": 4,
                },
            }
        )

        def make_scope(scenario):
            method_scenario = scenario.method_scenario()
            return ScopeLitePlanner(
                generation,
                encoder,
                transitions,
                reward,
                method_scenario,
                dataset,
                config,
                decoding,
                run_root / "scope-plans" / method_scenario["id"],
            )

        return make_scope, provenance
    raise ValueError(f"unsupported method: {method}")


def as_plain(decoding):
    return {
        "temperature": decoding.temperature,
        "top_p": decoding.top_p,
        "top_k": decoding.top_k,
        "max_tokens": decoding.max_tokens,
    }


def runtime_for_method(base_runtime, method):
    """Keep CARP semantics fixed while leaving room for intrinsic method models."""
    runtime = json.loads(json.dumps(base_runtime))
    if method == "scope_crs_lite":
        runtime["generation"].update(
            {
                "gpu_memory_utilization": 0.48,
                "max_num_seqs": 4,
            }
        )
    return runtime


def run_worker(args):
    root, run_root = Path(args.root).resolve(), Path(args.run_root).resolve()
    dataset_root = run_root / args.dataset
    config = read_json(root / "configs/carp-legacy-v1.json")
    runtime_template_path = root / "configs/carp-legacy-runtime.json"
    runtime_template_sha256 = file_hash(runtime_template_path)
    if config["mode"] != PROTOCOL or config["runtime_shuffle_seed"] != 42:
        raise ValueError("CARP evaluator config changed")
    carp_root = Path(args.carp_data_root).resolve().parents[1]
    for relative, expected in config["reference_source"]["files"].items():
        source_path = carp_root / relative
        if not source_path.is_file() or file_hash(source_path) != expected:
            raise ValueError(f"CARP reference source changed: {source_path}")
    scenarios = load_carp_scenarios(Path(args.carp_data_root), args.dataset, 42)[: args.count]
    if len(scenarios) != args.count:
        raise ValueError("requested more CARP scenarios than available")
    runtime = read_json(args.runtime_config)
    results = []
    with Servers(root, dataset_root / "servers", runtime, kinds=("generation",)) as servers:
        generation = servers.backends()["generation"]
        make_planner, method_provenance = build_planner_factory(
            root, dataset_root, args.method, args.dataset, generation
        )
        for scenario in scenarios:
            planner = make_planner(scenario)
            contract = {
                "phase": "mini" if args.count < 200 else "test",
                "method": METHOD_LABELS[args.method],
                "dataset": args.dataset,
                "root_seed": config["experimental_root_seed"],
                "resource_sha256": RESOURCE_CONTRACT[args.dataset]["sha256"],
                "runtime_template_sha256": runtime_template_sha256,
                "method_provenance": method_provenance,
                "request_namespace": digest(
                    {
                        "mode": PROTOCOL,
                        "method": METHOD_LABELS[args.method],
                        "dataset": args.dataset,
                        "resource": RESOURCE_CONTRACT[args.dataset]["sha256"],
                        "prompt": prompt_contract_hash(),
                    }
                ),
            }
            result = run_carp_dialogue(
                scenario,
                planner,
                generation,
                dataset_root / "dialogues" / scenario.id,
                contract,
                root_seed=config["experimental_root_seed"],
                max_turns=config["max_turns"],
                max_output_tokens=config["max_output_tokens"],
            )
            results.append(result)
            atomic_json(
                dataset_root / "progress.json",
                {
                    "schema_version": 1,
                    "status": "running",
                    "method": METHOD_LABELS[args.method],
                    "dataset": args.dataset,
                    "completed": len(results),
                    "expected": len(scenarios),
                    "last_scenario": scenario.id,
                    "updated_at": now(),
                },
            )
            del planner
            gc.collect()
        compute = servers.compute_metadata()
    report = {
        "schema_version": 1,
        "status": "complete",
        "mode": PROTOCOL,
        "method": METHOD_LABELS[args.method],
        "dataset": args.dataset,
        "diagnostic_only": args.count < 200,
        "selected_ids": [row.id for row in scenarios],
        "resource_sha256": RESOURCE_CONTRACT[args.dataset]["sha256"],
        "runtime_template_sha256": runtime_template_sha256,
        "prompt_contract_sha256": prompt_contract_hash(),
        "request_contract": config["request_contract"],
        "summary": summarize_carp_results(results),
        "server_compute": compute,
        "completed_at": now(),
    }
    atomic_json(dataset_root / "report.json", report)
    return report


def ports(job_id: str):
    start, stop = 42_000, 58_000
    blocks = (stop - start) // 4
    initial = int(job_id) % blocks
    for offset in range(blocks):
        base = start + ((initial + offset) % blocks) * 4
        values = tuple(base + index for index in range(4))
        probes = []
        try:
            for value in values:
                probe = socket.socket()
                probes.append(probe)
                probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                probe.bind(("127.0.0.1", value))
                probe.listen(1)
            return {"TopDial": values[:2], "DuRecDial2.0": values[2:]}
        except OSError:
            pass
        finally:
            for probe in probes:
                probe.close()
    raise OSError("no free CARP evaluator port block")


def reusable_dataset_report(path, root, method, dataset, count):
    path = Path(path)
    if not path.is_file():
        return None
    try:
        report = read_json(path)
    except (OSError, ValueError):
        return None
    expected = {
        "status": "complete",
        "mode": PROTOCOL,
        "method": METHOD_LABELS[method],
        "dataset": dataset,
        "diagnostic_only": count < 200,
        "resource_sha256": RESOURCE_CONTRACT[dataset]["sha256"],
        "runtime_template_sha256": file_hash(
            Path(root) / "configs/carp-legacy-runtime.json"
        ),
        "prompt_contract_sha256": prompt_contract_hash(),
    }
    if any(report.get(field) != value for field, value in expected.items()):
        return None
    if len(report.get("selected_ids", [])) != count:
        return None
    return report


def run_controller(args):
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("CARP evaluator inference must run inside Slurm")
    root, run_root = Path(args.root).resolve(), Path(args.run_root).resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    dirty = subprocess.check_output(
        ("git", "status", "--porcelain", "--untracked-files=all"), cwd=root, text=True
    ).strip()
    if dirty:
        raise ValueError("CARP evaluator requires a clean committed checkout")
    assigned = ports(os.environ["SLURM_JOB_ID"])
    base_runtime = read_json(root / "configs/carp-legacy-runtime.json")
    if base_runtime.get("mode") != PROTOCOL:
        raise ValueError("CARP evaluator runtime mode changed")
    expected_generation = {
        "gpu_memory_utilization": 0.85,
        "max_model_len": 16384,
        "max_num_seqs": 64,
    }
    for field, expected in expected_generation.items():
        if base_runtime["generation"].get(field) != expected:
            raise ValueError(f"CARP generation runtime changed: {field}")
    children = []
    completed_reports = {}
    for dataset in DATASETS:
        prior_report = reusable_dataset_report(
            run_root / dataset / "report.json",
            root,
            args.method,
            dataset,
            args.count,
        )
        if prior_report is not None:
            completed_reports[dataset] = prior_report
            print(
                f"Reusing complete CARP dataset report: method={args.method} "
                f"dataset={dataset} count={args.count}",
                flush=True,
            )
            continue
        runtime = runtime_for_method(base_runtime, args.method)
        runtime["generation"]["port"], runtime["embedding"]["port"] = assigned[dataset]
        runtime_path = run_root / "runtime" / f"{dataset}.json"
        atomic_json(runtime_path, runtime)
        log_path = run_root / "logs" / f"{dataset}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handle = log_path.open("a", encoding="utf-8")
        worker = [
            sys.executable, str(Path(__file__).resolve()), "worker",
            "--root", str(root), "--run-root", str(run_root), "--method", args.method,
            "--dataset", dataset, "--count", str(args.count),
            "--runtime-config", str(runtime_path), "--carp-data-root", args.carp_data_root,
        ]
        if args.method == "scope_crs_lite":
            # Allocation 62050 contains an accounting-only orphan step.  Pinning
            # physical devices under --overlap uses the actually idle GPUs and
            # remains valid in clean follow-up holder allocations as well.
            gpu_index = DATASETS.index(dataset)
            command = [
                "srun", "--overlap", "--nodes=1", "--ntasks=1",
                "--cpus-per-task=8", "--mem=72G", "env",
                f"CUDA_VISIBLE_DEVICES={gpu_index}", *worker,
            ]
        else:
            command = [
                "srun", "--exclusive", "--exact", "--nodes=1", "--ntasks=1",
                "--cpus-per-task=8", "--gpus-per-task=1", "--gpu-bind=single:1",
                "--mem=72G", *worker,
            ]
        children.append((dataset, subprocess.Popen(command, cwd=root, stdout=handle, stderr=subprocess.STDOUT), handle, command))
    failures = []
    for dataset, child, handle, command in children:
        status = child.wait()
        handle.close()
        if status:
            failures.append({"dataset": dataset, "exit_code": status, "command": command})
    if failures:
        atomic_json(run_root / "failure.json", {"schema_version": 1, "failures": failures, "at": now()})
        raise RuntimeError(f"CARP evaluator dataset worker failed: {failures}")
    reports = [
        completed_reports.get(dataset)
        or read_json(run_root / dataset / "report.json")
        for dataset in DATASETS
    ]
    atomic_json(
        run_root / "report.json",
        {
            "schema_version": 1,
            "status": "complete",
            "mode": PROTOCOL,
            "method": METHOD_LABELS[args.method],
            "count_per_dataset": args.count,
            "diagnostic_only": args.count < 200,
            "dataset_reports": {row["dataset"]: row for row in reports},
            "source_commit": subprocess.check_output(("git", "rev-parse", "HEAD"), cwd=root, text=True).strip(),
            "slurm_job_id": os.environ["SLURM_JOB_ID"],
            "completed_at": now(),
        },
    )


def parse_args():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("controller", "worker"):
        child = subparsers.add_parser(command)
        child.add_argument("--root", required=True)
        child.add_argument("--run-root", required=True)
        child.add_argument("--method", choices=METHODS, required=True)
        child.add_argument("--count", type=int, default=1)
        child.add_argument(
            "--carp-data-root",
            default="/home/u26anhvl/crs/carp_www/data/eval",
        )
        if command == "worker":
            child.add_argument("--dataset", choices=DATASETS, required=True)
            child.add_argument("--runtime-config", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    if not 1 <= args.count <= 200:
        raise ValueError("CARP evaluator count must be in [1, 200]")
    if args.command == "worker":
        run_worker(args)
    else:
        run_controller(args)


if __name__ == "__main__":
    main()
