#!/usr/bin/env python3
import argparse
import gc
import json
import time
from dataclasses import asdict
from pathlib import Path

import torch

from tdrl.baseline_test_setting import (
    evaluation_axes,
    fixed_test_scenarios,
    load_fixed_test_setting,
)
from tdrl.baselines.epl.checkpoint import load_checkpoint
from tdrl.baselines.epl.contracts import EPLCheckpointMetadata, Experience
from tdrl.baselines.epl.memory import NativeMiniLMEmbedding, embed_experiences
from tdrl.baselines.epl.model import load_pretrained_policy
from tdrl.baselines.epl.runtime import SOURCE_REVISION, load_tokenizer
from tdrl.baselines.epl_native.actions import NativeAction, NativeActionVocabulary
from tdrl.baselines.epl_native.config import EPLNativeConfig
from tdrl.baselines.epl_native.dataset import EPLNativeDataset, dataset_signature
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
from tdrl.metrics import aggregate, validate_records
from tdrl.provenance import atomic_json, digest, file_hash
from tdrl.runtime import Servers
from tdrl.stages import code_provenance
from tdrl.test_prefix import seal_sample


class NoEmbedding:
    requests = []

    def embed(self, *args, **kwargs):
        raise RuntimeError("EPL-native policy self-play requested embeddings")


def restore_committed_actions(planner, destination):
    """Restore the policy-history feature before resuming a partial dialogue."""
    snapshot = Path(destination) / "dialogue.json"
    if not snapshot.exists():
        return
    stored = json.loads(snapshot.read_text(encoding="utf-8"))
    planner.committed_actions = [
        NativeAction.from_key(key)
        for turn in stored.get("turns_trace", [])
        for key in turn["plan"].get("path", [])[-1:]
    ]


def rows(path):
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_policy(args):
    vocabulary = NativeActionVocabulary.read(args.vocabulary)
    if vocabulary.dataset != args.dataset:
        raise ValueError("EPL-native vocabulary/dataset mismatch")
    config = EPLNativeConfig.from_paths(args.config, args.vocabulary)
    tokenizer = load_tokenizer(args.encoder)
    training = EPLNativeDataset(
        args.root, args.dataset, "train", tokenizer, config, vocabulary
    )
    development = EPLNativeDataset(
        args.root, args.dataset, "dev", tokenizer, config, vocabulary
    )
    metadata = EPLCheckpointMetadata(
        schema_version=1,
        dataset=args.dataset,
        config_sha256=config.sha256,
        training_sha256=config.training_sha256,
        train_sha256=digest(dataset_signature(training)),
        dev_sha256=digest(dataset_signature(development)),
        action_taxonomy_sha256=digest(config.actions),
        model_lock_sha256=file_hash(args.model_lock),
        source_revision=SOURCE_REVISION,
    )
    model = load_pretrained_policy(args.encoder, config, len(tokenizer), "cuda")
    load_checkpoint(args.checkpoint, model, metadata, map_location="cuda")
    model.eval()
    return model, tokenizer, config, vocabulary


def experience_rows(result, scenario):
    messages = tuple(Message(**row) for row in result["history"])
    result_rows = []
    for index, message in enumerate(messages):
        if message.role != "assistant" or not messages[:index]:
            continue
        result_rows.append(
            Experience(
                experience_id=f"native-selfsim:{scenario['id']}:{index}",
                scenario_id=scenario["id"],
                state_text=policy_text(scenario, messages[:index]),
                continuation=tuple(asdict(item) for item in messages[index:]),
                score=1.0 if result["accepted"] else 0.0,
                target=tuple(scenario["target"]),
            )
        )
    return result_rows


def prepare_memory(args):
    model, tokenizer, config, vocabulary = load_policy(args)
    comparison = json.loads(args.comparison_config.read_text(encoding="utf-8"))
    scenarios = sorted(
        rows(args.root / "data/processed" / args.dataset / "train.jsonl"),
        key=lambda row: digest([args.dataset, row["id"], config.self_simulation_seed]),
    )[: config.memory_cases]
    if len(scenarios) != config.memory_cases:
        raise ValueError("insufficient EPL-native self-play scenarios")
    output = args.run_root / "self-simulation" / args.dataset
    experiences = []
    with Servers(
        args.root,
        output / "servers",
        json.loads(args.runtime_config.read_text(encoding="utf-8")),
        kinds=("generation",),
    ) as servers:
        generation = servers.backends()["generation"]
        judge = LocalJudge(generation, comparison["judge"])
        for scenario in scenarios:
            planner = EPLNativePolicyPlanner(
                generation,
                model,
                tokenizer,
                None,
                config,
                vocabulary,
                scenario,
                file_hash(args.checkpoint),
                "profile",
                Decoding(**comparison["assistant_decoding"]),
                Decoding(**comparison["simulator_decoding"]),
            )
            contract = {
                "phase": "epl-native-train-only-self-simulation",
                "method": "EPL-native",
                "selection_sha256": "training-only",
                "budget": "natural",
                "target_nodes": [],
                "request_namespace": digest(
                    [
                        args.dataset,
                        scenario["id"],
                        config.sha256,
                        file_hash(args.checkpoint),
                        "epl-native-self-simulation-v1",
                    ]
                ),
            }
            destination = output / "dialogues" / digest(scenario["id"])
            restore_committed_actions(planner, destination)
            result = run_dialogue(
                scenario,
                planner,
                generation,
                NoEmbedding(),
                "profile",
                config.self_simulation_seed,
                destination,
                contract,
                config.puct_max_depth,
                Decoding(**comparison["simulator_decoding"]),
                judge=judge,
            )
            if result["status"] != "complete":
                raise ValueError("partial EPL-native self-play dialogue")
            experiences.extend(experience_rows(result, scenario))
        compute = servers.compute_metadata()
    if not experiences or len({row.experience_id for row in experiences}) != len(
        experiences
    ):
        raise ValueError("empty or duplicate EPL-native experiences")
    memory = args.run_root / "memory" / args.dataset / "experiences.jsonl"
    memory.parent.mkdir(parents=True, exist_ok=True)
    content = "".join(
        json.dumps(row.to_dict(), ensure_ascii=False, sort_keys=True) + "\n"
        for row in experiences
    )
    if memory.exists() and memory.read_text(encoding="utf-8") != content:
        raise ValueError("immutable EPL-native memory changed")
    memory.write_text(content, encoding="utf-8")
    embedding = NativeMiniLMEmbedding(
        args.encoder,
        args.retrieval_encoder,
        args.model_lock,
        memory.parent / "embedding-journal",
    )
    vectors = memory.parent / "vectors.pt"
    payload = embed_experiences(
        memory,
        vectors,
        embedding,
        config,
        f"epl-native/{args.dataset}/{file_hash(memory)}",
    )
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "dataset": args.dataset,
        "configured_memory_cases": config.memory_cases,
        "selected_scenario_ids": [row["id"] for row in scenarios],
        "selected_scenario_ids_sha256": digest([row["id"] for row in scenarios]),
        "experience_count": len(payload["experience_ids"]),
        "checkpoint_sha256": file_hash(args.checkpoint),
        "memory_sha256": file_hash(memory),
        "vectors_sha256": file_hash(vectors),
        "official_test_opened": False,
        "server_compute": compute,
    }
    atomic_json(memory.parent / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


def provenance(args):
    return {
        "method_revision": METHOD_REVISION,
        "method_contract": METHOD_CONTRACT,
        "comparison_config_sha256": file_hash(args.comparison_config),
        "config_sha256": file_hash(args.config),
        "vocabulary_sha256": file_hash(args.vocabulary),
        "checkpoint_sha256": file_hash(args.checkpoint),
        "memory_sha256": file_hash(args.memory),
        "vectors_sha256": file_hash(args.vectors),
        "model_lock_sha256": file_hash(args.model_lock),
        "qwen_model_lock_sha256": file_hash(args.root / "configs/models.lock.json"),
        "code": code_provenance(args.root),
    }


def evaluate(args):
    model, tokenizer, config, vocabulary = load_policy(args)
    comparison = json.loads(args.comparison_config.read_text(encoding="utf-8"))
    setting = None
    selection_sha256 = "development"
    manifest = None
    if args.split == "test":
        setting = load_fixed_test_setting(args.root, args.fixed_setting, 400)
        envelope = json.loads(args.selection_lock.read_text(encoding="utf-8"))
        if digest(envelope["payload"]) != envelope.get("sha256"):
            raise ValueError("invalid EPL-native selection lock")
        expected = envelope["payload"]["datasets"][args.dataset]
        if expected != provenance(args):
            raise ValueError("EPL-native test artifacts differ from selection lock")
        selection_sha256 = envelope["sha256"]
        scenario_map, manifest = fixed_test_scenarios(args.root, setting)
        scenarios = scenario_map[args.dataset]
        seeds, budgets = evaluation_axes(setting, comparison, "test")
    else:
        scenarios = sorted(
            rows(args.root / "data/processed" / args.dataset / "dev.jsonl"),
            key=lambda row: digest([args.dataset, row["id"]]),
        )[: comparison["development_scenarios"]]
        seeds, budgets = comparison["seeds"], comparison["budgets"]
    output = args.run_root / "evaluation" / f"{args.split}-workers" / args.dataset
    leaf = "test-fixed1000" if args.split == "test" else "dev"
    split_output = output / leaf
    records = []
    started = time.monotonic()
    with Servers(
        args.root,
        split_output / "servers",
        json.loads(args.runtime_config.read_text(encoding="utf-8")),
        kinds=("generation",),
    ) as servers:
        generation = servers.backends()["generation"]
        embedding = NativeMiniLMEmbedding(
            args.encoder,
            args.retrieval_encoder,
            args.model_lock,
            split_output / "embedding-journal",
        )
        memory = NativeExperienceMemory(args.memory, args.vectors, embedding, config)
        judge = LocalJudge(generation, comparison["judge"])
        if manifest:
            seal_sample(split_output / "sealed-sample.json", manifest["sample"])
        for protocol in comparison["protocols"]:
            for seed in seeds:
                for scenario in scenarios:
                    for budget in budgets:
                        if budget != "natural":
                            raise ValueError("EPL-native supports natural budget only")
                        planner = EPLNativePlanner(
                            generation,
                            model,
                            tokenizer,
                            memory,
                            config,
                            vocabulary,
                            scenario,
                            file_hash(args.checkpoint),
                            protocol,
                            Decoding(**comparison["assistant_decoding"]),
                            Decoding(**comparison["simulator_decoding"]),
                        )
                        contract = {
                            "phase": args.split,
                            "method": "EPL-native",
                            "selection_sha256": selection_sha256,
                            "budget": budget,
                            "target_nodes": [],
                            "request_namespace": digest(
                                {
                                    **provenance(args),
                                    "dataset": args.dataset,
                                    "selection": selection_sha256,
                                    "fixed_resource": (
                                        setting["payload"]["test"]["resource_sha256"]
                                        if setting
                                        else None
                                    ),
                                }
                            ),
                        }
                        destination = (
                            split_output
                            / args.dataset
                            / protocol
                            / str(seed)
                            / budget
                            / scenario["id"]
                        )
                        restore_committed_actions(planner, destination)
                        before = len(generation.requests)
                        before_embeddings = len(embedding.requests)
                        result = run_dialogue(
                            scenario,
                            planner,
                            generation,
                            embedding,
                            protocol,
                            seed,
                            destination,
                            contract,
                            comparison["turn_budget"],
                            Decoding(**comparison["simulator_decoding"]),
                            judge=judge,
                        )
                        if result["status"] != "complete":
                            raise ValueError("partial EPL-native evaluation dialogue")
                        if args.expect_replay and len(generation.requests) != before:
                            raise ValueError("EPL-native replay issued new generation requests")
                        if (
                            args.expect_replay
                            and len(embedding.requests) != before_embeddings
                        ):
                            raise ValueError("EPL-native replay issued new embedding requests")
                        if args.split == "test":
                            seal_sample(
                                destination / "completed-seal.json",
                                {
                                    "dialogue_sha256": file_hash(
                                        destination / "dialogue.json"
                                    ),
                                    "selection_sha256": selection_sha256,
                                },
                            )
                        records.append(
                            {
                                "dataset": args.dataset,
                                "protocol": protocol,
                                "budget": budget,
                                **result,
                            }
                        )
        compute = servers.compute_metadata()
    summaries = []
    scenario_ids = [row["id"] for row in scenarios]
    for protocol in comparison["protocols"]:
        group = [
            row
            for row in records
            if row["protocol"] == protocol and row["budget"] == "natural"
        ]
        validate_records(group, scenario_ids, seeds, selection_sha256)
        summaries.append(
            {
                "dataset": args.dataset,
                "protocol": protocol,
                "method": "EPL-native",
                "budget": "natural",
                **aggregate(group),
            }
        )
    report = {
        "schema_version": 1,
        "status": "complete",
        "split": args.split,
        "dataset": args.dataset,
        "selection_sha256": selection_sha256,
        "provenance": provenance(args),
        "dialogues": len(records),
        "wall_s": time.monotonic() - started,
        "summary": summaries,
        "server_compute": compute,
        "expect_replay": args.expect_replay,
    }
    atomic_json(output / f"{leaf}-report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    del model
    gc.collect()
    torch.cuda.empty_cache()


def parser():
    result = argparse.ArgumentParser(description="EPL-native full worker")
    result.add_argument("command", choices=("prepare-memory", "evaluate"))
    result.add_argument("--root", type=Path, required=True)
    result.add_argument("--run-root", type=Path, required=True)
    result.add_argument("--dataset", choices=("TopDial", "DuRecDial2.0"), required=True)
    result.add_argument("--config", type=Path, required=True)
    result.add_argument("--vocabulary", type=Path, required=True)
    result.add_argument("--checkpoint", type=Path, required=True)
    result.add_argument("--encoder", type=Path, required=True)
    result.add_argument("--retrieval-encoder", type=Path, required=True)
    result.add_argument("--model-lock", type=Path, required=True)
    result.add_argument("--runtime-config", type=Path, required=True)
    result.add_argument("--comparison-config", type=Path, required=True)
    result.add_argument("--memory", type=Path)
    result.add_argument("--vectors", type=Path)
    result.add_argument("--split", choices=("dev", "test"), default="dev")
    result.add_argument("--selection-lock", type=Path)
    result.add_argument("--fixed-setting", type=Path)
    result.add_argument("--expect-replay", action="store_true")
    return result


def main():
    args = parser().parse_args()
    require_h100_slurm()
    if args.command == "prepare-memory":
        prepare_memory(args)
    else:
        for name in ("memory", "vectors"):
            if getattr(args, name) is None:
                raise ValueError(f"evaluate requires --{name}")
        evaluate(args)


if __name__ == "__main__":
    main()
