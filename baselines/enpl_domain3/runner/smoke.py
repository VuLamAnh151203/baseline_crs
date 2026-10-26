#!/usr/bin/env python3
"""Real two-call EnPL dialogue and exact request-journal replay gate."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from tdrl.baselines.enpl.bank import load_bank
from tdrl.baselines.enpl.config import load_enpl_config
from tdrl.baselines.enpl.planner import EnPLPlanner
from tdrl.baselines.enpl.retrieval import DomainEntityOverlapRetriever
from tdrl.baselines.enpl.selection import verify_reference_lock, verify_selection
from tdrl.contracts import Decoding
from tdrl.evaluation import dialogue_request_base, initial_state, run_dialogue
from tdrl.judge import LocalJudge
from tdrl.provenance import atomic_json, digest, file_hash
from tdrl.runtime import Servers


def restore_turn_seed_context(planner, scenario_id, protocol, turn):
    """Match the seed context that run_dialogue sets before planner.act."""
    if not isinstance(turn, int) or turn < 0:
        raise ValueError("invalid EnPL replay turn")
    planner.seed_context = f"{scenario_id}/{protocol}/{turn}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", choices=("TopDial", "DuRecDial2.0"), required=True)
    parser.add_argument("--selection-lock", type=Path, required=True)
    parser.add_argument("--source-lock", type=Path, required=True)
    parser.add_argument("--enpl-config", type=Path, required=True)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--bank-manifest", type=Path, required=True)
    parser.add_argument("--runtime-config", type=Path, required=True)
    parser.add_argument("--comparison-config", type=Path, required=True)
    args = parser.parse_args()
    verify_reference_lock(args.source_lock)
    selection = verify_selection(args.selection_lock)
    for key, path in (
        ("comparison_config_sha256", args.comparison_config),
        ("enpl_config_sha256", args.enpl_config),
        ("source_lock_sha256", args.source_lock),
        ("bank_sha256", args.bank),
        ("bank_manifest_sha256", args.bank_manifest),
    ):
        if selection["payload"][key] != file_hash(path):
            raise ValueError(f"EnPL smoke changed locked {key}")
    method = load_enpl_config(args.enpl_config)
    entries, _manifest = load_bank(args.bank, args.bank_manifest)
    retriever = DomainEntityOverlapRetriever(entries)
    comparison = json.loads(args.comparison_config.read_text(encoding="utf-8"))
    scenario = json.loads(
        next(
            line
            for line in (args.root / "data/processed" / args.dataset / "dev.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        )
    )
    settings = json.loads(args.runtime_config.read_text(encoding="utf-8"))
    checks = []
    with Servers(args.root, args.output / "servers", settings) as servers:
        backends = servers.backends()
        generation, embedding = backends["generation"], backends["embedding"]
        judge = LocalJudge(generation, comparison["judge"])
        for protocol in comparison["protocols"]:
            planner = EnPLPlanner(
                generation,
                scenario,
                retriever,
                Decoding(**method["plan_decoding"]),
                Decoding(**comparison["assistant_decoding"]),
                method["context"]["max_input_characters"],
            )
            contract = {
                "phase": "enpl-post-lock-smoke",
                "method": "EnPL",
                "selection_sha256": selection["sha256"],
                "budget": "natural",
                "target_nodes": [],
                "request_namespace": digest(
                    [selection["sha256"], args.dataset, protocol, "enpl-smoke-v2"]
                ),
            }
            result = run_dialogue(
                scenario,
                planner,
                generation,
                embedding,
                protocol,
                17,
                args.output / protocol,
                contract,
                turn_budget=1,
                simulator_decoding=Decoding(**comparison["simulator_decoding"]),
                judge=judge,
            )
            first = result["turns_trace"][0]["plan"]
            replay_contract = {**contract, "judge_sha256": judge.protocol_hash}
            request_id = (
                dialogue_request_base(replay_contract, protocol, 17, scenario["id"])
                + "/turn/0"
            )
            # A completed snapshot makes run_dialogue return before its turn loop,
            # so explicitly restore the per-turn seed context for an idempotent
            # replay in a later allocation.
            restore_turn_seed_context(planner, scenario["id"], protocol, 0)
            before = (len(generation.requests), len(embedding.requests))
            replay = asdict(planner.act(initial_state(scenario), [], 17, request_id))
            after = (len(generation.requests), len(embedding.requests))
            if (
                {key: value for key, value in first.items() if key != "compute"}
                != {key: value for key, value in replay.items() if key != "compute"}
                or before != after
            ):
                raise ValueError("EnPL exact two-call plan replay failed")
            roles = [
                item["role"]
                for turn in result["turns_trace"]
                for item in turn["generation"]
            ]
            if roles.count("planner") != 1 or roles.count("assistant") != 1:
                raise ValueError("EnPL two-call contract failed")
            if result["embedding_requests"] != 0:
                raise ValueError("EnPL domain-entity retrieval used embeddings")
            compute = first["compute"]
            if not compute["plan_format_valid"] or compute["plan_steps"] < 2:
                raise ValueError("EnPL smoke generated a malformed conversation plan")
            checks.append(
                {
                    "dataset": args.dataset,
                    "protocol": protocol,
                    "scenario_id": scenario["id"],
                    "exact_non_compute_plan": True,
                    "planner_requests": roles.count("planner"),
                    "assistant_requests": roles.count("assistant"),
                    "additional_generation_requests_on_replay": after[0] - before[0],
                    "additional_embedding_requests_on_replay": after[1] - before[1],
                    "examples_used": first["compute"]["examples_used"],
                    "retrieval_domain": compute["retrieval_domain"],
                    "plan_format_valid": compute["plan_format_valid"],
                    "plan_finish_reason": compute["plan_finish_reason"],
                    "plan_truncated": compute["plan_truncated"],
                    "plan_steps": compute["plan_steps"],
                }
            )
        server_compute = servers.compute_metadata()
    payload = {
        "schema_version": 1,
        "status": "passed",
        "selection_sha256": selection["sha256"],
        "checks": checks,
        "server_compute": server_compute,
    }
    atomic_json(args.output / "checks.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
