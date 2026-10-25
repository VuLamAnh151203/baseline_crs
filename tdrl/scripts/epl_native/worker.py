#!/usr/bin/env python3
import argparse
import json
import os
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import torch

from tdrl.baselines.epl.checkpoint import load_checkpoint
from tdrl.baselines.epl.contracts import EPLCheckpointMetadata, Experience
from tdrl.baselines.epl.memory import NativeMiniLMEmbedding, embed_experiences
from tdrl.baselines.epl.model import load_pretrained_policy
from tdrl.baselines.epl.runtime import SOURCE_REVISION, load_tokenizer
from tdrl.baselines.epl.training import EPLTrainer
from tdrl.baselines.epl_native.config import EPLNativeConfig
from tdrl.baselines.epl_native.dataset import (
    EPLNativeDataset,
    build_vocabulary,
    dataset_signature,
)
from tdrl.baselines.epl_native.memory import NativeExperienceMemory
from tdrl.baselines.epl_native.planner import (
    METHOD_CONTRACT,
    METHOD_REVISION,
    EPLNativePlanner,
    EPLNativePolicyPlanner,
)
from tdrl.baselines.epl_native.serialization import policy_text
from tdrl.backend import require_h100_slurm
from tdrl.contracts import Decoding, Message
from tdrl.evaluation import run_dialogue
from tdrl.judge import LocalJudge
from tdrl.provenance import atomic_json, digest, file_hash
from tdrl.runtime import Servers


class NoEmbedding:
    requests = []

    def embed(self, *args, **kwargs):
        raise RuntimeError("EPL native policy-only self-simulation requested embeddings")


def _rows(path):
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _experiences(result, scenario):
    messages = tuple(Message(**row) for row in result["history"])
    rows = []
    for index, message in enumerate(messages):
        if message.role != "assistant" or not messages[:index]:
            continue
        rows.append(
            Experience(
                experience_id=f"native-selfsim:{scenario['id']}:{index}",
                scenario_id=scenario["id"],
                state_text=policy_text(scenario, messages[:index]),
                continuation=tuple(asdict(item) for item in messages[index:]),
                score=1.0 if result["accepted"] else 0.0,
                target=tuple(scenario["target"]),
            )
        )
    if not rows:
        raise ValueError("EPL native self-simulation produced no experience")
    return rows


@torch.inference_mode()
def _policy_diagnostics(trainer, dataset, vocabulary):
    trainer.model.eval()
    gold, predicted = Counter(), Counter()
    correct = total = 0
    for batch in trainer._loader(dataset, 0, False):
        input_ids = batch["input_ids"].to(trainer.device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(
            trainer.device, non_blocking=True
        )
        labels = batch["labels"].to(trainer.device, non_blocking=True)
        predictions = trainer.model(input_ids, attention_mask)["logits"].argmax(-1)
        for label in labels.cpu().tolist():
            gold[int(label)] += 1
        for label in predictions.cpu().tolist():
            predicted[int(label)] += 1
        correct += int((predictions == labels).sum())
        total += len(labels)
    if total != len(dataset) or not total:
        raise ValueError("incomplete EPL native policy diagnostics")

    def rows(counter):
        return [
            {
                "action": vocabulary.items[index].to_dict(),
                "count": count,
            }
            for index, count in sorted(counter.items())
        ]

    result = {
        "examples": total,
        "top1_accuracy": correct / total,
        "gold_unique_actions": len(gold),
        "predicted_unique_actions": len(predicted),
        "largest_prediction_fraction": max(predicted.values()) / total,
        "gold_counts": rows(gold),
        "prediction_counts": rows(predicted),
    }
    result["noncollapsed"] = result["predicted_unique_actions"] >= 2
    return result


def main():
    parser = argparse.ArgumentParser(description="Run one EPL native-core mini dataset worker")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--dataset", choices=("TopDial", "DuRecDial2.0"), required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--runtime-config", type=Path, required=True)
    parser.add_argument("--comparison-config", type=Path, required=True)
    parser.add_argument("--encoder", type=Path, required=True)
    parser.add_argument("--retrieval-encoder", type=Path, required=True)
    parser.add_argument("--model-lock", type=Path, required=True)
    args = parser.parse_args()
    root, run_root = args.root.resolve(), args.run_root.resolve()
    output = run_root / "workers" / args.dataset
    output.mkdir(parents=True, exist_ok=True)
    gpu = require_h100_slurm()
    gpu["cuda_visible_devices"] = os.environ.get("CUDA_VISIBLE_DEVICES")
    gpu_work_started_at = datetime.now(timezone.utc).isoformat()

    vocabulary = build_vocabulary(root, args.dataset)
    vocabulary_path = output / "action-vocabulary.json"
    vocabulary.write(vocabulary_path)
    config = EPLNativeConfig.from_paths(args.config, vocabulary_path)
    tokenizer = load_tokenizer(args.encoder)
    training = EPLNativeDataset(
        root, args.dataset, "train", tokenizer, config, vocabulary
    )
    validation = EPLNativeDataset(
        root, args.dataset, "dev", tokenizer, config, vocabulary
    )
    metadata = EPLCheckpointMetadata(
        schema_version=1,
        dataset=args.dataset,
        config_sha256=config.sha256,
        training_sha256=config.training_sha256,
        train_sha256=digest(dataset_signature(training)),
        dev_sha256=digest(dataset_signature(validation)),
        action_taxonomy_sha256=digest(config.actions),
        model_lock_sha256=file_hash(args.model_lock),
        source_revision=SOURCE_REVISION,
    )
    torch.manual_seed(config.training_seed)
    torch.cuda.manual_seed_all(config.training_seed)
    model = load_pretrained_policy(args.encoder, config, len(tokenizer), "cuda")
    trainer = EPLTrainer(model, config, metadata, tokenizer.pad_token_id, "cuda")
    training_result = trainer.fit(training, validation, output / "training")
    if training_result.get("status") != "complete":
        raise InterruptedError("EPL native mini training stopped")
    checkpoint = output / "training/checkpoint-best.pt"
    load_checkpoint(checkpoint, model, metadata, map_location="cuda")
    model.eval()
    policy_diagnostics = _policy_diagnostics(trainer, validation, vocabulary)
    if not policy_diagnostics["noncollapsed"]:
        raise ValueError("EPL native policy collapsed to one action on development")

    comparison = json.loads(args.comparison_config.read_text(encoding="utf-8"))
    runtime = json.loads(args.runtime_config.read_text(encoding="utf-8"))
    train_rows = _rows(root / "data/processed" / args.dataset / "train.jsonl")
    dev_rows = _rows(root / "data/processed" / args.dataset / "dev.jsonl")
    train_scenario = sorted(
        train_rows, key=lambda row: digest([args.dataset, row["id"], "native-selfsim"])
    )[0]
    dev_scenario = sorted(
        dev_rows, key=lambda row: digest([args.dataset, row["id"], "native-dev"])
    )[0]
    assistant_decoding = Decoding(**comparison["assistant_decoding"])
    simulator_decoding = Decoding(**comparison["simulator_decoding"])
    memory_path = output / "memory/experiences.jsonl"
    vector_path = output / "memory/vectors.pt"
    evaluation_rows = []
    with Servers(root, output / "servers", runtime, kinds=("generation",)) as servers:
        generation = servers.backends()["generation"]
        judge = LocalJudge(generation, comparison["judge"])
        self_planner = EPLNativePolicyPlanner(
            generation,
            model,
            tokenizer,
            None,
            config,
            vocabulary,
            train_scenario,
            file_hash(checkpoint),
            "profile",
            assistant_decoding,
            simulator_decoding,
        )
        self_result = run_dialogue(
            train_scenario,
            self_planner,
            generation,
            NoEmbedding(),
            "profile",
            config.self_simulation_seed,
            output / "self-simulation/dialogue",
            {
                "phase": "epl-native-train-only-self-simulation",
                "method": EPLNativeConfig.METHOD,
                "selection_sha256": "training-only",
                "budget": "natural",
                "target_nodes": [],
                "request_namespace": digest(
                    [args.dataset, train_scenario["id"], config.sha256, "selfsim-v1"]
                ),
            },
            config.puct_max_depth,
            simulator_decoding,
            judge=judge,
        )
        experiences = _experiences(self_result, train_scenario)
        memory_path.parent.mkdir(parents=True, exist_ok=True)
        memory_path.write_text(
            "".join(
                json.dumps(row.to_dict(), ensure_ascii=False, sort_keys=True) + "\n"
                for row in experiences
            ),
            encoding="utf-8",
        )
        embedding = NativeMiniLMEmbedding(
            args.encoder,
            args.retrieval_encoder,
            args.model_lock,
            output / "memory/embedding-journal",
        )
        embed_experiences(
            memory_path,
            vector_path,
            embedding,
            config,
            f"epl-native/{args.dataset}/{file_hash(memory_path)}",
        )
        memory = NativeExperienceMemory(
            memory_path, vector_path, embedding, config
        )
        for protocol in comparison["protocols"]:
            destination = output / "development" / protocol / dev_scenario["id"]
            planner = EPLNativePlanner(
                generation,
                model,
                tokenizer,
                memory,
                config,
                vocabulary,
                dev_scenario,
                file_hash(checkpoint),
                protocol,
                assistant_decoding,
                simulator_decoding,
            )
            contract = {
                "phase": "dev",
                "method": EPLNativeConfig.METHOD,
                "selection_sha256": "development",
                "budget": "natural",
                "target_nodes": [],
                "request_namespace": digest(
                    [args.dataset, protocol, dev_scenario["id"], config.sha256]
                ),
            }
            result = run_dialogue(
                dev_scenario,
                planner,
                generation,
                embedding,
                protocol,
                comparison["seeds"][0],
                destination,
                contract,
                comparison["turn_budget"],
                simulator_decoding,
                judge=judge,
            )
            if result["status"] != "complete":
                raise ValueError("partial EPL native development dialogue")
            before = file_hash(destination / "dialogue.json")
            replay = run_dialogue(
                dev_scenario,
                EPLNativePlanner(
                    generation,
                    model,
                    tokenizer,
                    memory,
                    config,
                    vocabulary,
                    dev_scenario,
                    file_hash(checkpoint),
                    protocol,
                    assistant_decoding,
                    simulator_decoding,
                ),
                generation,
                embedding,
                protocol,
                comparison["seeds"][0],
                destination,
                contract,
                comparison["turn_budget"],
                simulator_decoding,
                judge=judge,
            )
            if replay["status"] != "complete" or file_hash(destination / "dialogue.json") != before:
                raise ValueError("EPL native exact dialogue replay failed")
            evaluation_rows.append(result)
        server_compute = servers.compute_metadata()

    selected = Counter()
    for result in evaluation_rows:
        for turn in result["turns_trace"]:
            compute = turn["plan"]["compute"]
            if compute.get("selected_goal"):
                selected[(compute["selected_goal"], compute["selected_topic_role"])] += 1
    summary = {
        "schema_version": 1,
        "status": "complete",
        "method": EPLNativeConfig.METHOD,
        "method_revision": METHOD_REVISION,
        "method_contract": METHOD_CONTRACT,
        "dataset": args.dataset,
        "gpu": gpu,
        "gpu_work_started_at": gpu_work_started_at,
        "gpu_work_ended_at": datetime.now(timezone.utc).isoformat(),
        "diagnostic_only": True,
        "official_test_opened": False,
        "training_examples": len(training),
        "development_policy_examples": len(validation),
        "training_result": training_result,
        "policy_diagnostics": policy_diagnostics,
        "action_vocabulary_sha256": vocabulary.sha256,
        "action_count": len(vocabulary.actions),
        "goals": list(vocabulary.goals),
        "checkpoint_sha256": file_hash(checkpoint),
        "memory_sha256": file_hash(memory_path),
        "vectors_sha256": file_hash(vector_path),
        "development_scenario_id": dev_scenario["id"],
        "development_dialogues": len(evaluation_rows),
        "selected_action_counts": [
            {"goal": key[0], "topic_role": key[1], "count": value}
            for key, value in sorted(selected.items())
        ],
        "replay": "exact_completed_dialogue_hash",
        "server_compute": server_compute,
    }
    atomic_json(output / "worker-summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
