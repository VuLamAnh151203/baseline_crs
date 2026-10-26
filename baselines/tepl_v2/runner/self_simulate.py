#!/usr/bin/env python3
import argparse
import json
from dataclasses import asdict
from pathlib import Path

from tdrl.baselines.epl.contracts import Experience
from tdrl.baselines.epl.planner import EPLPolicyPlanner
from tdrl.baselines.epl.runtime import load_inference
from tdrl.baselines.epl.serialization import policy_text
from tdrl.contracts import Decoding, Message
from tdrl.evaluation import run_dialogue
from tdrl.judge import LocalJudge
from tdrl.provenance import atomic_json, digest, file_hash
from tdrl.runtime import Servers


class NoEmbedding:
    requests = []

    def embed(self, *args, **kwargs):
        raise RuntimeError("EPL policy-only self-simulation must not request embeddings")


def main():
    parser = argparse.ArgumentParser(description="Build EPL memory by train-only self-simulation")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--dataset", choices=("TopDial", "DuRecDial2.0"), required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--encoder", type=Path, required=True)
    parser.add_argument("--retrieval-encoder", type=Path, required=True)
    parser.add_argument("--model-lock", type=Path, required=True)
    parser.add_argument("--runtime-config", type=Path, required=True)
    parser.add_argument("--comparison-config", type=Path, required=True)
    parser.add_argument("--max-cases", type=int)
    args = parser.parse_args()
    train_path = args.root / "data/processed" / args.dataset / "train.jsonl"
    rows = [
        json.loads(line)
        for line in train_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    model, tokenizer, config, _, _ = load_inference(
        args.config,
        args.checkpoint,
        args.encoder,
        args.encoder,
        args.retrieval_encoder,
        args.model_lock,
        "cuda",
    )
    if config.dataset != args.dataset:
        raise ValueError("EPL self-simulation dataset/config mismatch")
    case_count = config.memory_cases if args.max_cases is None else args.max_cases
    if case_count < 1 or case_count > config.memory_cases:
        raise ValueError(f"invalid EPL self-simulation case limit: {case_count}")
    selected = sorted(
        rows,
        key=lambda row: digest(
            [args.dataset, row["id"], config.self_simulation_seed]
        ),
    )[
        :case_count
    ]
    if len(selected) != case_count:
        raise ValueError("insufficient EPL training scenarios for self-simulation")
    comparison = json.loads(args.comparison_config.read_text(encoding="utf-8"))
    settings = json.loads(args.runtime_config.read_text(encoding="utf-8"))
    output = args.run_root / "self-simulation" / args.dataset
    model.eval()
    experiences = []
    no_embedding = NoEmbedding()
    with Servers(args.root, output / "servers", settings, kinds=("generation",)) as servers:
        generation = servers.backends()["generation"]
        judge = LocalJudge(generation, comparison["judge"])
        for scenario in selected:
            planner = EPLPolicyPlanner(
                generation,
                model,
                tokenizer,
                None,
                config,
                scenario,
                file_hash(args.checkpoint),
                "profile",
                Decoding(**comparison["assistant_decoding"]),
                Decoding(**comparison["simulator_decoding"]),
            )
            contract = {
                "phase": "epl-train-only-self-simulation",
                "method": "EPL",
                "selection_sha256": "training-only",
                "budget": "natural",
                "target_nodes": [],
                "request_namespace": digest(
                    [
                        args.dataset,
                        scenario["id"],
                        config.sha256,
                        file_hash(args.checkpoint),
                        "epl-self-simulation-v1",
                    ]
                ),
            }
            result = run_dialogue(
                scenario,
                planner,
                generation,
                no_embedding,
                "profile",
                config.self_simulation_seed,
                output / "dialogues" / digest(scenario["id"]),
                contract,
                turn_budget=config.puct_max_depth,
                simulator_decoding=Decoding(**comparison["simulator_decoding"]),
                judge=judge,
            )
            if result["status"] != "complete":
                raise ValueError("partial EPL self-simulation dialogue")
            messages = tuple(Message(**row) for row in result["history"])
            for index, message in enumerate(messages):
                if message.role != "assistant" or not messages[:index]:
                    continue
                experiences.append(
                    Experience(
                        experience_id=f"selfsim:{scenario['id']}:{index}",
                        scenario_id=scenario["id"],
                        state_text=policy_text(scenario, messages[:index]),
                        continuation=tuple(asdict(item) for item in messages[index:]),
                        score=1.0 if result["accepted"] else 0.0,
                        target=tuple(scenario["target"]),
                    )
                )
        compute = servers.compute_metadata()
    if not experiences or len({row.experience_id for row in experiences}) != len(experiences):
        raise ValueError("empty or duplicate EPL self-simulation experience")
    memory_path = args.run_root / "memory" / args.dataset / "experiences.jsonl"
    memory_path.parent.mkdir(parents=True, exist_ok=True)
    raw = "".join(
        json.dumps(row.to_dict(), ensure_ascii=False, sort_keys=True) + "\n"
        for row in experiences
    )
    if memory_path.exists() and memory_path.read_text(encoding="utf-8") != raw:
        raise ValueError("immutable EPL self-simulation memory changed")
    memory_path.write_text(raw, encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "dataset": args.dataset,
        "source_split": "train",
        "source_sha256": file_hash(train_path),
        "checkpoint_sha256": file_hash(args.checkpoint),
        "selected_scenario_ids": [row["id"] for row in selected],
        "selected_scenario_ids_sha256": digest([row["id"] for row in selected]),
        "configured_memory_cases": config.memory_cases,
        "diagnostic_case_limit": args.max_cases,
        "experience_count": len(experiences),
        "memory_sha256": file_hash(memory_path),
        "score": "shared semantic judge on train-only self-simulation",
        "test_bodies": "never_opened",
        "server_compute": compute,
    }
    atomic_json(memory_path.with_suffix(".manifest.json"), manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
