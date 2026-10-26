#!/usr/bin/env python3
import argparse
import json
from dataclasses import asdict
from pathlib import Path

from tdrl.baselines.epl.memory import ExperienceMemory, NativeMiniLMEmbedding
from tdrl.baselines.epl.planner import EPLPlanner
from tdrl.baselines.epl.runtime import load_inference
from tdrl.baselines.epl.selection import verify_selection
from tdrl.contracts import Decoding
from tdrl.evaluation import dialogue_request_base, initial_state, run_dialogue
from tdrl.judge import LocalJudge
from tdrl.provenance import atomic_json, digest, file_hash
from tdrl.runtime import Servers


def main():
    parser = argparse.ArgumentParser(description="Real dialogue and exact replay gate for EPL")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", choices=("TopDial", "DuRecDial2.0"), required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--memory", type=Path, required=True)
    parser.add_argument("--vectors", type=Path, required=True)
    parser.add_argument("--encoder", type=Path, required=True)
    parser.add_argument("--retrieval-encoder", type=Path, required=True)
    parser.add_argument("--model-lock", type=Path, required=True)
    parser.add_argument("--selection-lock", type=Path, required=True)
    parser.add_argument("--runtime-config", type=Path, required=True)
    parser.add_argument("--comparison-config", type=Path, required=True)
    args = parser.parse_args()
    selection = verify_selection(args.selection_lock)
    if selection["payload"]["checkpoint_sha256"][args.dataset] != file_hash(args.checkpoint):
        raise ValueError("EPL smoke checkpoint differs from selection")
    if selection["payload"]["memory_sha256"][args.dataset] != file_hash(args.memory):
        raise ValueError("EPL smoke memory differs from selection")
    if selection["payload"]["memory_vector_sha256"][args.dataset] != file_hash(args.vectors):
        raise ValueError("EPL smoke memory vectors differ from selection")
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
    model, tokenizer, config, _, _ = load_inference(
        args.config,
        args.checkpoint,
        args.encoder,
        args.encoder,
        args.retrieval_encoder,
        args.model_lock,
        "cuda",
    )
    model.eval()
    checks = []
    with Servers(
        args.root, args.output / "servers", settings, kinds=("generation",)
    ) as servers:
        backends = servers.backends()
        embedding = NativeMiniLMEmbedding(
            args.encoder,
            args.retrieval_encoder,
            args.model_lock,
            args.output / "embedding-journal",
        )
        memory = ExperienceMemory(args.memory, args.vectors, embedding, config)
        judge = LocalJudge(backends["generation"], comparison["judge"])
        for protocol in comparison["protocols"]:
            planner = EPLPlanner(
                backends["generation"],
                model,
                tokenizer,
                memory,
                config,
                scenario,
                file_hash(args.checkpoint),
                protocol,
                Decoding(**comparison["assistant_decoding"]),
                Decoding(**comparison["simulator_decoding"]),
            )
            contract = {
                "phase": "epl-post-selection-smoke",
                "method": "EPL",
                "selection_sha256": selection["sha256"],
                "budget": "natural",
                "target_nodes": [],
                "request_namespace": digest(
                    [selection["sha256"], args.dataset, protocol, "epl-smoke-v1"]
                ),
            }
            result = run_dialogue(
                scenario,
                planner,
                backends["generation"],
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
            compute_trace = first["compute"]
            expected_action = max(
                range(len(config.actions)),
                key=lambda index: (compute_trace["puct_root_q_values"][index], -index),
            )
            if (
                compute_trace.get("method_revision") != "source-corrected-v2"
                or compute_trace.get("search_revision")
                != "source-puct-mean-return-v2"
                or compute_trace.get("final_action_rule")
                != "maximum_backed_up_mean_return"
                or compute_trace.get("rollout_terminal_rule")
                != "source-farewell-or-search-horizon-v2"
                or compute_trace["action_index"] != expected_action
                or not first["replay_ids"][-1].endswith("/final-realization")
            ):
                raise ValueError("EPL corrected source-search contract failed")
            request_id = dialogue_request_base(contract, protocol, 17, scenario["id"]) + "/turn/0"
            before = (len(backends["generation"].requests), len(embedding.requests))
            replay = asdict(planner.act(initial_state(scenario), [], 17, request_id))
            after = (len(backends["generation"].requests), len(embedding.requests))
            if (
                {key: value for key, value in first.items() if key != "compute"}
                != {key: value for key, value in replay.items() if key != "compute"}
                or before != after
            ):
                raise ValueError("EPL exact dialogue plan replay failed")
            checks.append(
                {
                    "dataset": args.dataset,
                    "protocol": protocol,
                    "scenario_id": scenario["id"],
                    "exact_non_compute_plan": True,
                    "source_search_contract": True,
                    "selected_action": compute_trace["selected_action"],
                    "root_q_values": compute_trace["puct_root_q_values"],
                    "additional_generation_requests": after[0] - before[0],
                    "additional_embedding_requests": after[1] - before[1],
                }
            )
        compute = servers.compute_metadata()
    payload = {
        "schema_version": 1,
        "status": "passed",
        "selection_sha256": selection["sha256"],
        "checks": checks,
        "server_compute": compute,
    }
    atomic_json(args.output / "checks.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
