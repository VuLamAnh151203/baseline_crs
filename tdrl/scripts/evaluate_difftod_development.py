#!/usr/bin/env python3
"""Fixed fitted DiffTOD development baseline; confirmation and official tests stay closed."""

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import torch
from tokenizers import Tokenizer

from tdrl.backend import require_h100_slurm
from tdrl.confirmation import confirmation_candidate, confirmation_cases
from tdrl.contracts import Decoding
from tdrl.difftod import DiffTODPlanner
from tdrl.difftod_evaluation import load_selected_policy, verify_selected_policy
from tdrl.evaluation import initial_state, run_dialogue
from tdrl.feasibility import full_turn_dialogue_proxy
from tdrl.judge import LocalJudge
from tdrl.metrics import aggregate
from tdrl.planner import ASSISTANT_POLICY, INSTRUCTION
from tdrl.provenance import atomic_json, digest, file_hash, validate_manifest
from tdrl.runtime import Servers
from tdrl.stages import GpuMonitor, Stage, code_provenance, counters
from tdrl.state import SERIALIZATION_HASH
from tdrl.training import load_prepared


def verify_gates(paths, contract, remaining):
    if len(paths) != 2:
        raise ValueError("two independent fitted-policy GPU gates required")
    checks = []
    for directory in paths:
        manifest = json.loads((directory / "manifest.json").read_text())
        validate_manifest(manifest)
        if manifest["status"] != "complete" or manifest["stage"] != "DiffTOD-development-smoke":
            raise ValueError("complete fitted-policy smoke required")
        if file_hash(directory / "checks.json") != manifest["outputs"]["checks.json"]:
            raise ValueError("fitted-policy smoke checksum mismatch")
        check = json.loads((directory / "checks.json").read_text())
        if check["status"] != "passed" or check["contract"] != contract:
            raise ValueError("fitted-policy runtime/config/replay gate mismatch")
        checks.append(check)
    if len({c["gpu"]["uuid"] for c in checks}) != 2:
        raise ValueError("distinct H100 UUIDs required")
    if max(c["required_seconds_with_safety"] for c in checks) > remaining:
        raise ValueError(
            "post-fit matrix exceeds allocation time; explicit scope/budget decision required"
        )
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fitting", type=Path, required=True)
    parser.add_argument("--slot", type=int, choices=[0, 1], required=True)
    parser.add_argument("--mode", choices=["smoke", "evaluate"], required=True)
    parser.add_argument("--gates", type=Path, nargs=2, default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--confirmation-lock", type=Path)
    args = parser.parse_args()
    root = Path.cwd()
    cfg_path = root / "configs/difftod-development-v1.json"
    cfg = json.loads(cfg_path.read_text())
    if cfg.get("schema_version") != 1 or cfg["seed"] != 17:
        raise ValueError("unknown DiffTOD development contract")
    base = json.loads((root / cfg["comparison_config"]).read_text())
    native = json.loads((root / "configs/difftod-topdial-fit-v1.json").read_text())
    _, chosen, fitting_manifest = verify_selected_policy(args.fitting)
    if fitting_manifest["provenance"]["config"]["sha256"] != file_hash(
        root / "configs/difftod-topdial-fit-v1.json"
    ):
        raise ValueError("DiffTOD fitting contract changed")
    population = load_prepared(root, "TopDial", "dev")
    cases = sorted(population, key=lambda r: digest([17, r["id"]]))[16:80]
    if len(cases) != 64:
        raise ValueError("fixed discovery development population incomplete")
    discovery_ids = [s["id"] for s in cases]
    candidate = None
    if args.confirmation_lock:
        if args.mode != "evaluate":
            raise ValueError("confirmation cannot run in smoke mode")
        candidate, membership = confirmation_candidate(
            root,
            args.confirmation_lock,
            code_provenance(root)["commit"],
            root / cfg["comparison_config"],
        )
        if chosen["sha256"] != candidate["difftod_policy_sha256"]:
            raise ValueError("DiffTOD policy changed after candidate freeze")
        cases = confirmation_cases(
            population, membership, file_hash(root / "data/processed/TopDial/dev.jsonl")
        )
    gpu = require_h100_slurm()
    stage = Stage(
        root,
        args.output,
        "DiffTOD-development-confirmation" if candidate else "DiffTOD-development-" + args.mode,
        cfg_path,
        {
            "slot": args.slot,
            "fitting_manifest_sha256": file_hash(args.fitting / "manifest.json"),
            "comparison_sha256": file_hash(root / cfg["comparison_config"]),
            "scenario_ids": [s["id"] for s in cases],
            "serialization_sha256": SERIALIZATION_HASH,
            **(
                {"confirmation_lock_sha256": file_hash(args.confirmation_lock)} if candidate else {}
            ),
        },
    )
    if stage.reused:
        return
    contract = {
        k: v for k, v in stage.contract.items() if k not in {"slot", "confirmation_lock_sha256"}
    }
    gate_contract = {**contract, "scenario_ids": discovery_ids}
    allocation = json.loads((root / "artifacts/two-gpu-58159/allocation.json").read_text())
    if allocation["job_id"] != "58159" and allocation["job_id"] != 58159:
        raise ValueError("allocation clock provenance mismatch")
    remaining = allocation["walltime_seconds"] - (time.time() - allocation["started_unix"])
    if args.mode == "evaluate":
        verify_gates(args.gates, gate_contract, remaining)
    compute = {}
    started = time.monotonic()
    try:
        with GpuMonitor(args.output / "gpu-telemetry.jsonl") as monitor:
            torch.use_deterministic_algorithms(True)
            torch.manual_seed(17)
            torch.cuda.manual_seed_all(17)
            model, provenance = load_selected_policy(root, args.fitting)
            pin = json.loads((root / "configs/difftod-policy-model.lock.json").read_text())
            tokenizer = Tokenizer.from_file(
                str(root / "models/difftod/gpt2" / pin["tokenizer"]["revision"] / "tokenizer.json")
            )
            settings = json.loads((root / "configs/runtime.json").read_text())
            for kind in ["generation", "embedding"]:
                settings[kind]["port"] += args.slot * 100
            selected = (
                [load_prepared(root, "TopDial", "train")[0]] if args.mode == "smoke" else cases
            )
            results, timings = [], []
            with Servers(root, args.output / "servers", settings) as servers:
                backends = servers.backends()
                judge = LocalJudge(backends["generation"], base["judge"])
                for protocol in cfg["protocols"]:
                    for scenario in selected:
                        actor = DiffTODPlanner(
                            model,
                            tokenizer,
                            backends["generation"],
                            backends["embedding"],
                            scenario,
                            Decoding(**base["assistant_decoding"]),
                            Decoding(**native["guidance_decoding"]),
                            args.output / "policy-journal",
                            provenance,
                            root / "artifacts/difftod-nltk-data",
                        )
                        result = run_dialogue(
                            scenario,
                            actor,
                            backends["generation"],
                            backends["embedding"],
                            protocol,
                            17,
                            args.output / "dialogues" / protocol / digest(scenario["id"]),
                            {
                                "phase": "DiffTOD-fitted-confirmation"
                                if candidate
                                else "DiffTOD-fitted-development-" + args.mode,
                                "method": "DiffTOD",
                                "request_namespace": digest(contract),
                                **(
                                    {"confirmation_lock_sha256": file_hash(args.confirmation_lock)}
                                    if candidate
                                    else {}
                                ),
                                "fitted_policy": provenance,
                            },
                            turn_budget=base["turn_budget"],
                            simulator_decoding=Decoding(**base["simulator_decoding"]),
                            judge=judge,
                        )
                        if args.mode == "smoke":
                            first_plan = result["turns_trace"][0]["plan"]
                            request = first_plan["replay_ids"][0].removesuffix("/live")
                            actor.trace = None
                            actor.seed_context = f"{scenario['id']}/{protocol}/0"
                            executed = len(backends["generation"].requests)
                            embedded = len(backends["embedding"].requests)
                            replay = actor.act(initial_state(scenario), [], 17, request)
                            if (
                                {k: v for k, v in asdict(replay).items() if k != "compute"}
                                != {k: v for k, v in first_plan.items() if k != "compute"}
                                or actor.trace != first_plan["compute"]["diffusion"]
                                or len(backends["generation"].requests) != executed
                                or len(backends["embedding"].requests) != embedded
                            ):
                                raise ValueError("fitted DiffTOD exact plan/journal replay failed")
                            if actor.trace["policy_forwards"] != 5125:
                                raise ValueError("DiffTOD native sampling budget changed")
                            timings.append(
                                full_turn_dialogue_proxy(
                                    result["wall_s"],
                                    result["actual_turns"],
                                    base["turn_budget"],
                                    first_plan["compute"]["diffusion"]["policy_sampling_wall_s"]
                                    + first_plan["compute"]["diffusion"]["guidance_record"][
                                        "latency_s"
                                    ],
                                )
                            )
                        results.append(result)
                compute = {
                    **counters(backends["generation"], backends["embedding"]),
                    **servers.compute_metadata(),
                }
            if list(args.output.rglob("pending-*.json")) or list(
                args.output.rglob("*.pending.json")
            ):
                raise ValueError("unresolved request outcome; preserve journals")
            if args.mode == "smoke":
                reserve = (
                    native["evaluation_dialogue_reserve"]
                    + cfg["additional_discovery_dialogue_reserve"]
                )
                required = (
                    max(timings) * reserve + native["checkpoint_io_reserve_seconds"]
                ) * native["feasibility_safety_factor"]
                check = {
                    "schema_version": 1,
                    "status": "passed",
                    "gpu": gpu,
                    "contract": contract,
                    "exact_plan_journal_replay": True,
                    "optimizer_updates": 0,
                    "full_turn_dialogue_proxy_seconds": timings,
                    "required_seconds_with_safety": required,
                    "dialogue_reserve": reserve,
                    "remaining_allocation_seconds": allocation["walltime_seconds"]
                    - (time.time() - allocation["started_unix"]),
                    "scope_limits": "One training case per protocol; conservative common-method proxy. Official tests remain separately gated.",
                }
                atomic_json(args.output / "checks.json", check)
                if required > check["remaining_allocation_seconds"]:
                    raise ValueError(
                        "post-fit matrix exceeds allocation time; explicit scope/budget decision required"
                    )
            else:
                if (
                    len(results) != 128
                    or len({(r["scenario_id"], r["protocol"]) for r in results}) != 128
                ):
                    raise ValueError("incomplete/duplicate discovery dialogues")
                atomic_json(args.output / "results.json", results)
                atomic_json(
                    args.output / "metrics.json",
                    {
                        p: aggregate([r for r in results if r["protocol"] == p])
                        for p in cfg["protocols"]
                    },
                )
        atomic_json(args.output / "gpu-summary.json", monitor.summary())
        stage.write(
            "complete",
            provenance_updates={
                "gpu": gpu,
                "checkpoint": provenance,
                "scenarios": {
                    "ids": [s["id"] for s in selected],
                    "confirmation_lock_sha256": file_hash(args.confirmation_lock)
                    if candidate
                    else None,
                },
                "dataset": {
                    "dataset": "TopDial",
                    "split": "training-smoke" if args.mode == "smoke" else "development",
                    "processed_sha256": file_hash(
                        root
                        / "data/processed/TopDial"
                        / ("train.jsonl" if args.mode == "smoke" else "dev.jsonl")
                    ),
                },
                "seeds": {"root": 17, "protocols": cfg["protocols"]},
                "prompts": {
                    "assistant_sha256": digest(ASSISTANT_POLICY),
                    "embedding_instruction_sha256": digest(INSTRUCTION),
                },
            },
            compute_updates={**compute, "h100_seconds": time.monotonic() - started},
        )
    except Exception as error:
        stage.write(
            "failed",
            error=error,
            compute_updates={**compute, "h100_seconds": time.monotonic() - started},
        )
        raise


if __name__ == "__main__":
    main()
