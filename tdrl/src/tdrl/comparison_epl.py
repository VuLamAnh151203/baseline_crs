"""T-EPL adaptation using the frozen shared interactive evaluator."""

import gc
import json
import time
from pathlib import Path

import torch

from .baseline_test_setting import (
    evaluation_axes,
    evaluation_output_directory,
    fixed_test_scenarios,
    load_fixed_test_setting,
)
from .baselines.epl.config import EPLConfig
from .baselines.epl.memory import ExperienceMemory, NativeMiniLMEmbedding
from .baselines.epl.planner import EPL_METHOD_CONTRACT, EPL_METHOD_REVISION, EPLPlanner
from .baselines.epl.runtime import load_inference
from .baselines.epl.selection import verify_selection
from .contracts import Decoding
from .evaluation import run_dialogue
from .judge import LocalJudge
from .metrics import aggregate, validate_records
from .provenance import atomic_json, digest, file_hash
from .runtime import Servers
from .stages import code_provenance
from .test_prefix import seal_sample


def validate_config(config):
    exact = {
        "schema_version",
        "scope",
        "datasets",
        "seeds",
        "protocols",
        "turn_budget",
        "budgets",
        "development_scenarios",
        "assistant_decoding",
        "simulator_decoding",
        "judge",
        "test_sampling",
    }
    expected_scope = {
        "methods": ["EPL"],
        "claim": "single_method_shared_fixed1000_evaluation",
    }
    if set(config) != exact or config["schema_version"] != 2 or config["scope"] != expected_scope:
        raise ValueError("unknown/incomplete EPL comparison configuration")
    if config["datasets"] != ["TopDial", "DuRecDial2.0"]:
        raise ValueError("EPL comparison requires both datasets")
    if config["seeds"] != [17] or config["protocols"] != ["profile", "resistant"]:
        raise ValueError("EPL comparison requires the frozen seed/protocols")
    if config["turn_budget"] != 10 or config["budgets"] != ["natural"]:
        raise ValueError("EPL fixed evaluation is natural-budget only")
    if not isinstance(config["development_scenarios"], int) or config["development_scenarios"] < 1:
        raise ValueError("invalid EPL development scenario count")
    for name in ("assistant_decoding", "simulator_decoding"):
        Decoding(**config[name])
    if config["judge"].get("schema_version") != 2 or config["judge"].get("critic_samples") != 10:
        raise ValueError("EPL comparison requires the shared semantic judge")
    Decoding(**config["judge"]["outcome_decoding"])
    Decoding(**config["judge"]["critic_decoding"])
    if config["test_sampling"] != {
        "schema_version": 1,
        "method": "pooled_uniform_without_replacement",
        "total_scenarios": 500,
        "seed": 42,
    }:
        raise ValueError("EPL legacy sample metadata changed")


def _processed(root, dataset):
    path = root / "data/processed" / dataset / "dev.jsonl"
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows or len({row["id"] for row in rows}) != len(rows):
        raise ValueError("prepared EPL development scenarios are empty or duplicated")
    return rows


class EPLComparison:
    def __init__(
        self,
        root,
        output,
        comparison_config,
        checkpoints,
        epl_configs,
        memories,
        vectors,
        encoder,
        tokenizer,
        retrieval_encoder,
        asset_lock,
        source_lock,
        selection_lock=None,
        runtime_config=None,
        fixed_test_setting=None,
        fixed_test_count=None,
    ):
        self.root, self.output = Path(root), Path(output)
        self.comparison_config_path = Path(comparison_config)
        self.checkpoints = {key: Path(value) for key, value in checkpoints.items()}
        self.epl_configs = {key: Path(value) for key, value in epl_configs.items()}
        self.memories = {key: Path(value) for key, value in memories.items()}
        self.vectors = {key: Path(value) for key, value in vectors.items()}
        self.encoder, self.tokenizer = Path(encoder), Path(tokenizer)
        self.retrieval_encoder = Path(retrieval_encoder)
        self.asset_lock, self.source_lock = Path(asset_lock), Path(source_lock)
        self.selection_lock = Path(selection_lock) if selection_lock else None
        self.fixed_test_setting = load_fixed_test_setting(
            self.root, fixed_test_setting, fixed_test_count
        )
        self.fixed_test_cache = self.fixed_test_manifest = None
        self.config = json.loads(self.comparison_config_path.read_text(encoding="utf-8"))
        validate_config(self.config)
        datasets = set(self.config["datasets"])
        for mapping in (self.checkpoints, self.epl_configs, self.memories, self.vectors):
            if set(mapping) != datasets:
                raise ValueError("EPL comparison requires both dataset artifacts")
        self.runtime_path = Path(runtime_config) if runtime_config else self.root / "configs/runtime.json"
        self.runtime = json.loads(self.runtime_path.read_text(encoding="utf-8"))

    def _scenarios(self, dataset, split):
        if split == "dev":
            return sorted(
                _processed(self.root, dataset), key=lambda row: digest([dataset, row["id"]])
            )[: self.config["development_scenarios"]]
        if split != "test" or self.selection_lock is None or self.fixed_test_setting is None:
            raise ValueError("official EPL test requires selection and fixed1000 setting")
        if self.fixed_test_cache is None:
            self.fixed_test_cache, self.fixed_test_manifest = fixed_test_scenarios(
                self.root, self.fixed_test_setting
            )
        return self.fixed_test_cache[dataset]

    def _provenance(self):
        parsed = {dataset: EPLConfig.from_path(path) for dataset, path in self.epl_configs.items()}
        return {
            "method_revision": EPL_METHOD_REVISION,
            "method_contract": EPL_METHOD_CONTRACT,
            "comparison_config_sha256": file_hash(self.comparison_config_path),
            "config_sha256": {dataset: file_hash(path) for dataset, path in self.epl_configs.items()},
            "checkpoint_sha256": {
                dataset: file_hash(path) for dataset, path in self.checkpoints.items()
            },
            "memory_sha256": {dataset: file_hash(path) for dataset, path in self.memories.items()},
            "memory_vector_sha256": {
                dataset: file_hash(path) for dataset, path in self.vectors.items()
            },
            "training_config_sha256": {
                dataset: config.training_sha256 for dataset, config in parsed.items()
            },
            "search_budgets": {
                dataset: {
                    "simulations": config.puct_simulations,
                    "depth": config.puct_max_depth,
                    "retrieval_k": config.retrieval_k,
                }
                for dataset, config in parsed.items()
            },
            "asset_lock_sha256": file_hash(self.asset_lock),
            "source_lock_sha256": file_hash(self.source_lock),
            "qwen_model_lock_sha256": file_hash(self.root / "configs/models.lock.json"),
            "code": code_provenance(self.root),
        }

    def run(self, split, datasets=None):
        if split not in {"dev", "test"}:
            raise ValueError("EPL comparison split must be dev or test")
        datasets = tuple(datasets or self.config["datasets"])
        if not datasets or len(set(datasets)) != len(datasets) or any(
            dataset not in self.config["datasets"] for dataset in datasets
        ):
            raise ValueError("invalid EPL evaluation dataset shard")
        provenance = self._provenance()
        selection_sha256 = "development"
        if split == "test":
            envelope = verify_selection(self.selection_lock)
            selection_sha256 = envelope["sha256"]
            for key in (
                "comparison_config_sha256",
                "config_sha256",
                "checkpoint_sha256",
                "memory_sha256",
                "memory_vector_sha256",
                "asset_lock_sha256",
                "source_lock_sha256",
                "qwen_model_lock_sha256",
                "code",
            ):
                if envelope["payload"][key] != provenance[key]:
                    raise ValueError(f"EPL test runtime changed selected {key}")
        seeds, budgets = evaluation_axes(self.fixed_test_setting, self.config, split)
        split_output = evaluation_output_directory(self.output, split, self.fixed_test_setting)
        rows = []
        started = time.monotonic()
        with Servers(
            self.root, split_output / "servers", self.runtime, kinds=("generation",)
        ) as servers:
            backends = servers.backends()
            embedding = NativeMiniLMEmbedding(
                self.encoder,
                self.retrieval_encoder,
                self.asset_lock,
                split_output / "epl-embedding-journal",
            )
            for dataset in datasets:
                model, tokenizer, epl_config, _, _ = load_inference(
                    self.epl_configs[dataset],
                    self.checkpoints[dataset],
                    self.encoder,
                    self.tokenizer,
                    self.retrieval_encoder,
                    self.asset_lock,
                    "cuda",
                )
                model.eval()
                memory = ExperienceMemory(
                    self.memories[dataset], self.vectors[dataset], embedding, epl_config
                )
                scenarios = self._scenarios(dataset, split)
                if split == "test":
                    seal_sample(split_output / "sealed-sample.json", self.fixed_test_manifest["sample"])
                for protocol in self.config["protocols"]:
                    for seed in seeds:
                        for scenario in scenarios:
                            for budget in budgets:
                                if budget != "natural":
                                    raise ValueError("EPL official test supports natural budget only")
                                planner = EPLPlanner(
                                    backends["generation"],
                                    model,
                                    tokenizer,
                                    memory,
                                    epl_config,
                                    scenario,
                                    provenance["checkpoint_sha256"][dataset],
                                    protocol,
                                    Decoding(**self.config["assistant_decoding"]),
                                    Decoding(**self.config["simulator_decoding"]),
                                )
                                contract = {
                                    "phase": split,
                                    "method": "EPL",
                                    "selection_sha256": selection_sha256,
                                    "budget": budget,
                                    "target_nodes": [],
                                    "request_namespace": digest(
                                        {
                                            **provenance,
                                            "dataset": dataset,
                                            "selection": selection_sha256,
                                            "fixed1000_resource_sha256": (
                                                self.fixed_test_setting["payload"]["test"]["resource_sha256"]
                                                if self.fixed_test_setting
                                                else None
                                            ),
                                        }
                                    ),
                                }
                                destination = (
                                    split_output
                                    / dataset
                                    / protocol
                                    / str(seed)
                                    / budget
                                    / scenario["id"]
                                )
                                torch.cuda.reset_peak_memory_stats()
                                result = run_dialogue(
                                    scenario,
                                    planner,
                                    backends["generation"],
                                    embedding,
                                    protocol,
                                    seed,
                                    destination,
                                    contract,
                                    self.config["turn_budget"],
                                    Decoding(**self.config["simulator_decoding"]),
                                    judge=LocalJudge(backends["generation"], self.config["judge"]),
                                )
                                if result["status"] != "complete":
                                    raise ValueError("partial EPL dialogue cannot enter a report")
                                seal_sample(
                                    destination / "completed-seal.json",
                                    {
                                        "dialogue_sha256": file_hash(destination / "dialogue.json"),
                                        "selection_sha256": selection_sha256,
                                    },
                                )
                                result["epl_puct_simulations"] = sum(
                                    turn["plan"]["compute"].get("puct_simulations", 0)
                                    for turn in result["turns_trace"]
                                )
                                result["epl_planner_seconds"] = sum(
                                    turn["plan"]["compute"].get("planner_latency_s", 0.0)
                                    for turn in result["turns_trace"]
                                )
                                result["epl_peak_memory_bytes"] = torch.cuda.max_memory_allocated()
                                rows.append(
                                    {
                                        "dataset": dataset,
                                        "protocol": protocol,
                                        "budget": budget,
                                        **result,
                                    }
                                )
                del memory, model
                gc.collect()
                torch.cuda.empty_cache()
            compute = servers.compute_metadata()
        summary = []
        for dataset in datasets:
            scenario_ids = [row["id"] for row in self._scenarios(dataset, split)]
            for protocol in self.config["protocols"]:
                group = [
                    row
                    for row in rows
                    if row["dataset"] == dataset
                    and row["protocol"] == protocol
                    and row["budget"] == "natural"
                ]
                validate_records(group, scenario_ids, seeds, selection_sha256)
                summary.append(
                    {
                        "dataset": dataset,
                        "protocol": protocol,
                        "method": "EPL",
                        "budget": "natural",
                        **aggregate(group),
                        "puct_simulations_mean": sum(row["epl_puct_simulations"] for row in group)
                        / len(group),
                        "planner_seconds_mean": sum(row["epl_planner_seconds"] for row in group)
                        / len(group),
                        "peak_memory_bytes_max": max(row["epl_peak_memory_bytes"] for row in group),
                    }
                )
        report = {
            "schema_version": 1,
            "status": "complete",
            "split": split,
            "method": "EPL",
            "selection_sha256": selection_sha256,
            **provenance,
            "dialogues": len(rows),
            "wall_s": time.monotonic() - started,
            "server_compute": compute,
            "summary": summary,
            "evaluation_seeds": seeds,
            "evaluation_budgets": budgets,
            "scope_datasets": list(datasets),
            "runtime_config_sha256": file_hash(self.runtime_path),
        }
        if split == "test":
            report["dataset_setting"] = self.fixed_test_setting["payload"]
            report["test_sample"] = self.fixed_test_manifest
        report_name = "epl-report.json"
        if split == "test":
            report_name = f"epl-report-fixed1000-prefix-{self.fixed_test_setting['count']}.json"
        atomic_json(split_output / report_name, report)
        return report
