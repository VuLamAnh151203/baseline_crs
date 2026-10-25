"""SCOPE-CRS-lite under the frozen shared interactive evaluation contract."""

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
from .baselines.scope.assets import (
    verify_llama_guard_snapshot,
    verify_transition_manifest,
)
from .baselines.scope.config import ScopeConfig
from .baselines.scope.planner import POLICY_VERSION, ScopeLitePlanner
from .baselines.scope.reward import load_reward_checkpoint
from .baselines.scope.selection import (
    prompt_contract,
    verify_reference_lock,
    verify_selection,
)
from .baselines.scope.semantic import LlamaGuardSemanticEncoder, NoSharedEmbedding
from .baselines.scope.transition import TransitionEnsemble
from .contracts import Decoding
from .evaluation import run_dialogue
from .judge import LocalJudge
from .metrics import aggregate, validate_records
from .provenance import atomic_json, digest, file_hash
from .runtime import Servers
from .stages import code_provenance
from .test_prefix import seal_sample

DATASETS = ("TopDial", "DuRecDial2.0")


def validate_config(config):
    required = {
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
    if set(config) != required or config["schema_version"] != 1:
        raise ValueError("unknown/incomplete SCOPE comparison configuration")
    if config["scope"] != {
        "methods": ["SCOPE-CRS-lite"],
        "claim": "task_adapted_scope_shared_fixed1000_evaluation",
    } or config["datasets"] != list(DATASETS):
        raise ValueError("unexpected SCOPE comparison scope")
    if config["seeds"] != [17] or config["protocols"] != ["profile", "resistant"]:
        raise ValueError("SCOPE requires frozen seed/protocols")
    if config["budgets"] != ["natural"] or not 1 <= config["turn_budget"] <= 10:
        raise ValueError("SCOPE supports natural budget with at most ten turns")
    if type(config["development_scenarios"]) is not int or config["development_scenarios"] < 1:
        raise ValueError("invalid SCOPE development count")
    for name in ("assistant_decoding", "simulator_decoding"):
        Decoding(**config[name])
    judge = config["judge"]
    if judge.get("schema_version") != 2 or judge.get("critic_samples") != 10:
        raise ValueError("SCOPE requires the shared semantic judge")
    Decoding(**judge["outcome_decoding"])
    Decoding(**judge["critic_decoding"])
    if config["test_sampling"] != {
        "schema_version": 1,
        "method": "pooled_uniform_without_replacement",
        "total_scenarios": 500,
        "seed": 42,
    }:
        raise ValueError("SCOPE legacy sample metadata changed")


def _processed(root, dataset):
    path = Path(root) / "data/processed" / dataset / "dev.jsonl"
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows or len({row["id"] for row in rows}) != len(rows):
        raise ValueError("empty/duplicate SCOPE development scenarios")
    return rows


def _role_counts(result):
    generations = [item for turn in result["turns_trace"] for item in turn["generation"]]
    roles, tokens = {}, {}
    for item in generations:
        role = item["role"]
        roles[role] = roles.get(role, 0) + 1
        tokens[role] = tokens.get(role, 0) + item["total_tokens"]
    return roles, tokens


class ScopeComparison:
    def __init__(
        self,
        root,
        output,
        comparison_config,
        source_lock,
        run_root,
        scope_configs,
        transitions,
        llama_guard,
        llama_guard_manifest,
        selection_lock=None,
        runtime_config=None,
        fixed_test_setting=None,
        fixed_test_count=None,
    ):
        self.root, self.output = Path(root), Path(output)
        self.run_root = Path(run_root)
        self.comparison_config_path = Path(comparison_config)
        self.source_lock_path = Path(source_lock)
        verify_reference_lock(self.source_lock_path)
        self.selection_lock = Path(selection_lock) if selection_lock else None
        self.runtime_path = Path(runtime_config)
        self.runtime = json.loads(self.runtime_path.read_text(encoding="utf-8"))
        self.config = json.loads(self.comparison_config_path.read_text(encoding="utf-8"))
        validate_config(self.config)
        self.scope_config_paths = {key: Path(value) for key, value in scope_configs.items()}
        if set(self.scope_config_paths) != set(DATASETS):
            raise ValueError("SCOPE requires configs for both datasets")
        self.scope_configs = {
            dataset: ScopeConfig.from_path(path)
            for dataset, path in self.scope_config_paths.items()
        }
        self.transition_root = Path(transitions)
        self.transition_asset = verify_transition_manifest(self.transition_root)
        self.llama_guard = Path(llama_guard)
        self.llama_manifest = Path(llama_guard_manifest)
        self.llama_asset = verify_llama_guard_snapshot(
            self.llama_guard, self.llama_manifest
        )
        self.reward_paths = {
            dataset: self.run_root / "reward-training" / dataset / "checkpoint-best.pt"
            for dataset in DATASETS
        }
        self.fixed_test_setting = load_fixed_test_setting(
            self.root, fixed_test_setting, fixed_test_count
        )
        self.fixed_test_cache = self.fixed_test_manifest = None

    def _scenarios(self, dataset, split):
        if split == "dev":
            return sorted(
                _processed(self.root, dataset), key=lambda row: digest([dataset, row["id"]])
            )[: self.config["development_scenarios"]]
        if split != "test" or self.selection_lock is None or self.fixed_test_setting is None:
            raise ValueError("official SCOPE test requires method lock and fixed setting")
        if self.fixed_test_cache is None:
            self.fixed_test_cache, self.fixed_test_manifest = fixed_test_scenarios(
                self.root, self.fixed_test_setting
            )
        return self.fixed_test_cache[dataset]

    def _provenance(self):
        return {
            "comparison_config_sha256": file_hash(self.comparison_config_path),
            "source_lock_sha256": file_hash(self.source_lock_path),
            "qwen_model_lock_sha256": file_hash(self.root / "configs/models.lock.json"),
            "transition_manifest_sha256": self.transition_asset["sha256"],
            "llama_guard_manifest_sha256": self.llama_asset["manifest_sha256"],
            "scope_config_sha256_by_dataset": {
                dataset: config.sha256 for dataset, config in self.scope_configs.items()
            },
            "reward_checkpoint_sha256_by_dataset": {
                dataset: file_hash(path) for dataset, path in self.reward_paths.items()
            },
            "prompt_contract": prompt_contract(),
            "policy_contract": {
                "version": POLICY_VERSION,
                "candidate_generation": "five independent shared-Qwen requests",
                "transition": "released LMSYS four-seed ensemble",
                "reward": "dataset-specific goal-conditioned train/dev head",
                "planning": "semantic-space-online-mcts",
                "shared_qwen_embedding_calls_per_turn": 0,
            },
            "code": code_provenance(self.root),
        }

    def run(self, split, datasets=None):
        if split not in {"dev", "test"}:
            raise ValueError("SCOPE split must be dev or test")
        datasets = tuple(datasets or self.config["datasets"])
        if not datasets or len(set(datasets)) != len(datasets) or any(
            dataset not in DATASETS for dataset in datasets
        ):
            raise ValueError("invalid SCOPE dataset shard")
        provenance = self._provenance()
        selection_sha256 = "development"
        if split == "test":
            envelope = verify_selection(self.selection_lock)
            selection_sha256 = envelope["sha256"]
            locked = envelope["payload"]
            for key in (
                "comparison_config_sha256",
                "source_lock_sha256",
                "qwen_model_lock_sha256",
                "transition_manifest_sha256",
                "llama_guard_manifest_sha256",
                "scope_config_sha256_by_dataset",
                "code",
            ):
                if locked[key] != provenance[key]:
                    raise ValueError(f"SCOPE test runtime changed locked {key}")
            if locked["prompt_contract"] != provenance["prompt_contract"]:
                raise ValueError("SCOPE prompt contract changed after selection")
            for dataset, path in self.reward_paths.items():
                if locked["reward_checkpoints"][dataset]["sha256"] != file_hash(path):
                    raise ValueError("SCOPE reward checkpoint changed after selection")
        seeds, budgets = evaluation_axes(self.fixed_test_setting, self.config, split)
        split_output = evaluation_output_directory(
            self.output, split, self.fixed_test_setting
        )
        rows, started = [], time.monotonic()
        with Servers(
            self.root, split_output / "servers", self.runtime, kinds=("generation",)
        ) as servers:
            generation = servers.backends()["generation"]
            shared_embedding = NoSharedEmbedding()
            for dataset in datasets:
                scope_config = self.scope_configs[dataset]
                encoder = LlamaGuardSemanticEncoder(
                    self.llama_guard,
                    self.llama_manifest,
                    output_dim=scope_config.embedding_dim,
                    projection_seed=scope_config.projection_seed,
                )
                transitions = TransitionEnsemble(
                    self.transition_root, dimension=scope_config.embedding_dim
                )
                reward, reward_metadata = load_reward_checkpoint(
                    self.reward_paths[dataset], scope_config
                )
                scenarios = self._scenarios(dataset, split)
                if split == "test":
                    seal_sample(
                        split_output / "sealed-sample.json",
                        self.fixed_test_manifest["sample"],
                    )
                for protocol in self.config["protocols"]:
                    for seed in seeds:
                        for scenario in scenarios:
                            for budget in budgets:
                                if budget != "natural":
                                    raise ValueError("SCOPE supports natural budget only")
                                semantic_start = len(encoder.requests)
                                destination = (
                                    split_output
                                    / dataset
                                    / protocol
                                    / str(seed)
                                    / budget
                                    / scenario["id"]
                                )
                                planner = ScopeLitePlanner(
                                    generation,
                                    encoder,
                                    transitions,
                                    reward,
                                    scenario,
                                    dataset,
                                    scope_config,
                                    Decoding(**self.config["assistant_decoding"]),
                                    destination / "scope-plans",
                                )
                                contract = {
                                    "phase": split,
                                    "method": "SCOPE-CRS-lite",
                                    "selection_sha256": selection_sha256,
                                    "budget": budget,
                                    "target_nodes": [],
                                    "request_namespace": digest(
                                        {
                                            **provenance,
                                            "dataset": dataset,
                                            "selection": selection_sha256,
                                            "fixed1000_resource_sha256": (
                                                self.fixed_test_setting["payload"]["test"][
                                                    "resource_sha256"
                                                ]
                                                if self.fixed_test_setting
                                                else None
                                            ),
                                        }
                                    ),
                                }
                                result = run_dialogue(
                                    scenario,
                                    planner,
                                    generation,
                                    shared_embedding,
                                    protocol,
                                    seed,
                                    destination,
                                    contract,
                                    self.config["turn_budget"],
                                    Decoding(**self.config["simulator_decoding"]),
                                    judge=LocalJudge(generation, self.config["judge"]),
                                )
                                if result["status"] != "complete":
                                    raise ValueError("partial SCOPE dialogue cannot enter report")
                                roles, role_tokens = _role_counts(result)
                                if roles.get("assistant", 0) != 5 * result["actual_turns"]:
                                    raise ValueError("SCOPE must generate five candidates per turn")
                                if result["embedding_requests"] != 0:
                                    raise ValueError("SCOPE unexpectedly used shared embeddings")
                                compute = [
                                    turn["plan"]["compute"] for turn in result["turns_trace"]
                                ]
                                if any(item["candidate_count"] != 5 for item in compute):
                                    raise ValueError("SCOPE candidate budget changed")
                                seal_sample(
                                    destination / "completed-seal.json",
                                    {
                                        "dialogue_sha256": file_hash(
                                            destination / "dialogue.json"
                                        ),
                                        "selection_sha256": selection_sha256,
                                        "reward_checkpoint_sha256": reward_metadata["sha256"],
                                    },
                                )
                                rows.append(
                                    {
                                        "dataset": dataset,
                                        "protocol": protocol,
                                        "budget": budget,
                                        "scope_candidate_requests": roles.get("assistant", 0),
                                        "scope_planner_seconds": sum(
                                            item["planner_latency_s"] for item in compute
                                        ),
                                        "scope_search_seconds": sum(
                                            item["search_wall_s"] for item in compute
                                        ),
                                        "scope_search_simulations": sum(
                                            item["search_simulations"] for item in compute
                                        ),
                                        "scope_intrinsic_semantic_requests": len(
                                            encoder.requests[semantic_start:]
                                        ),
                                        "generation_role_counts": roles,
                                        "generation_role_tokens": role_tokens,
                                        **result,
                                    }
                                )
                del reward, transitions, encoder
                torch.cuda.empty_cache()
            compute_metadata = servers.compute_metadata()
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
                role_names = sorted(
                    {name for row in group for name in row["generation_role_counts"]}
                )
                summary.append(
                    {
                        "dataset": dataset,
                        "protocol": protocol,
                        "method": "SCOPE-CRS-lite",
                        "budget": "natural",
                        **aggregate(group),
                        "candidate_requests_per_dialogue": sum(
                            row["scope_candidate_requests"] for row in group
                        )
                        / len(group),
                        "planner_seconds_per_dialogue": sum(
                            row["scope_planner_seconds"] for row in group
                        )
                        / len(group),
                        "search_seconds_per_dialogue": sum(
                            row["scope_search_seconds"] for row in group
                        )
                        / len(group),
                        "search_simulations_per_dialogue": sum(
                            row["scope_search_simulations"] for row in group
                        )
                        / len(group),
                        "intrinsic_semantic_requests_per_dialogue": sum(
                            row["scope_intrinsic_semantic_requests"] for row in group
                        )
                        / len(group),
                        "shared_embedding_requests_per_dialogue": 0.0,
                        "generation_requests_by_role": {
                            role: sum(
                                row["generation_role_counts"].get(role, 0) for row in group
                            )
                            for role in role_names
                        },
                        "generation_tokens_by_role": {
                            role: sum(
                                row["generation_role_tokens"].get(role, 0) for row in group
                            )
                            for role in role_names
                        },
                    }
                )
        report = {
            "schema_version": 1,
            "status": "complete",
            "split": split,
            "method": "SCOPE-CRS-lite",
            "selection_sha256": selection_sha256,
            **provenance,
            "dialogues": len(rows),
            "wall_s": time.monotonic() - started,
            "server_compute": compute_metadata,
            "summary": summary,
            "evaluation_seeds": seeds,
            "evaluation_budgets": budgets,
            "scope_datasets": list(datasets),
            "runtime_config_sha256": file_hash(self.runtime_path),
        }
        if split == "test":
            report["dataset_setting"] = self.fixed_test_setting["payload"]
            report["test_sample"] = self.fixed_test_manifest
        name = "scope-report.json"
        if split == "test":
            name = f"scope-report-fixed1000-prefix-{self.fixed_test_setting['count']}.json"
        atomic_json(split_output / name, report)
        return report
