#!/usr/bin/env python3
"""Real ProCoT dialogue and exact request-journal replay gate."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from tdrl.baselines.procot.planner import ProCoTPlanner
from tdrl.baselines.procot.selection import verify_reference_lock, verify_selection
from tdrl.contracts import Decoding
from tdrl.evaluation import dialogue_request_base, initial_state, run_dialogue
from tdrl.judge import LocalJudge
from tdrl.provenance import atomic_json, digest, file_hash
from tdrl.runtime import Servers


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", choices=("TopDial", "DuRecDial2.0"), required=True)
    parser.add_argument("--selection-lock", type=Path, required=True)
    parser.add_argument("--source-lock", type=Path, required=True)
    parser.add_argument("--runtime-config", type=Path, required=True)
    parser.add_argument("--comparison-config", type=Path, required=True)
    args = parser.parse_args()
    verify_reference_lock(args.source_lock)
    selection = verify_selection(args.selection_lock)
    if selection["payload"]["comparison_config_sha256"] != file_hash(
        args.comparison_config
    ):
        raise ValueError("ProCoT smoke config differs from method lock")
    if selection["payload"]["source_lock_sha256"] != file_hash(args.source_lock):
        raise ValueError("ProCoT smoke source lock differs from method lock")
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
            planner = ProCoTPlanner(
                generation, scenario, Decoding(**comparison["assistant_decoding"])
            )
            contract = {
                "phase": "procot-post-lock-smoke",
                "method": "ProCoT",
                "selection_sha256": selection["sha256"],
                "budget": "natural",
                "target_nodes": [],
                "request_namespace": digest(
                    [selection["sha256"], args.dataset, protocol, "procot-smoke-v1"]
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
            before = (len(generation.requests), len(embedding.requests))
            replay = asdict(planner.act(initial_state(scenario), [], 17, request_id))
            after = (len(generation.requests), len(embedding.requests))
            if (
                {key: value for key, value in first.items() if key != "compute"}
                != {key: value for key, value in replay.items() if key != "compute"}
                or before != after
            ):
                raise ValueError("ProCoT exact plan replay failed")
            assistant_calls = sum(
                item["role"] == "assistant"
                for turn in result["turns_trace"]
                for item in turn["generation"]
            )
            if assistant_calls != 1 or result["embedding_requests"] != 0:
                raise ValueError("ProCoT one-call/no-embedding contract failed")
            compute = first["compute"]
            if compute["generation_requests"] != 1:
                raise ValueError("ProCoT planner compute accounting is invalid")
            if not (
                compute["format_valid"]
                and compute["current_topics"] is not None
                and compute["next_topics"] is not None
            ):
                raise ValueError("ProCoT smoke violated the three-part output contract")
            checks.append(
                {
                    "dataset": args.dataset,
                    "protocol": protocol,
                    "scenario_id": scenario["id"],
                    "exact_non_compute_plan": True,
                    "assistant_requests": assistant_calls,
                    "additional_generation_requests_on_replay": after[0] - before[0],
                    "additional_embedding_requests_on_replay": after[1] - before[1],
                    "policy_embedding_requests": result["embedding_requests"],
                    "format_valid": compute["format_valid"],
                    "current_topics_present": compute["current_topics"] is not None,
                    "next_topics_present": compute["next_topics"] is not None,
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
