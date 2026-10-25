"""DirectLLM under the frozen shared interactive evaluation contract."""

import json
import time
from pathlib import Path

from .baseline_test_setting import (
    evaluation_axes,
    evaluation_output_directory,
    fixed_test_scenarios,
    load_fixed_test_setting,
)
from .baselines.direct_llm.planner import (
    DIRECT_LLM_METHODS,
    planner_for_method,
    policy_version,
    prompt_hashes,
)
from .baselines.direct_llm.selection import verify_selection
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
    if set(config) != exact or config["schema_version"] != 1:
        raise ValueError("unknown/incomplete DirectLLM comparison configuration")
    scope = config.get("scope", {})
    methods = scope.get("methods") if isinstance(scope, dict) else None
    if (
        not isinstance(methods, list)
        or len(methods) != 1
        or methods[0] not in DIRECT_LLM_METHODS
        or scope.get("claim")
        != "post_hoc_single_method_shared_fixed1000_evaluation"
        or set(scope) != {"methods", "claim"}
        or config["datasets"] != list(DATASETS)
    ):
        raise ValueError("unexpected DirectLLM comparison scope")
    if config["seeds"] != [17] or config["protocols"] != ["profile", "resistant"]:
        raise ValueError("DirectLLM requires the frozen seed and protocols")
    if config["turn_budget"] != 10 or config["budgets"] != ["natural"]:
        raise ValueError("DirectLLM fixed evaluation is natural-budget only")
    if (
        not isinstance(config["development_scenarios"], int)
        or config["development_scenarios"] < 1
    ):
        raise ValueError("invalid DirectLLM development scenario count")
    for name in ("assistant_decoding", "simulator_decoding"):
        Decoding(**config[name])
    judge = config["judge"]
    if judge.get("schema_version") != 2 or judge.get("critic_samples") != 10:
        raise ValueError("DirectLLM requires the shared semantic judge")
    Decoding(**judge["outcome_decoding"])
    Decoding(**judge["critic_decoding"])
    if config["test_sampling"] != {
        "schema_version": 1,
        "method": "pooled_uniform_without_replacement",
        "total_scenarios": 500,
        "seed": 42,
    }:
        raise ValueError("DirectLLM legacy sample metadata changed")


def _processed(root, dataset):
    path = Path(root) / "data/processed" / dataset / "dev.jsonl"
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows or len({row["id"] for row in rows}) != len(rows):
        raise ValueError("prepared DirectLLM development scenarios are empty or duplicated")
    return rows


def _role_counts(result):
    generations = [item for turn in result["turns_trace"] for item in turn["generation"]]
    roles = {}
    role_tokens = {}
    for item in generations:
        role = item["role"]
        roles[role] = roles.get(role, 0) + 1
        role_tokens[role] = role_tokens.get(role, 0) + item["total_tokens"]
    return roles, role_tokens


class DirectLLMComparison:
    def __init__(
        self,
        root,
        output,
        comparison_config,
        selection_lock=None,
        runtime_config=None,
        fixed_test_setting=None,
        fixed_test_count=None,
    ):
        self.root, self.output = Path(root), Path(output)
        self.comparison_config_path = Path(comparison_config)
        self.selection_lock = Path(selection_lock) if selection_lock else None
        self.runtime_path = (
            Path(runtime_config) if runtime_config else self.root / "configs/runtime.json"
        )
        self.runtime = json.loads(self.runtime_path.read_text(encoding="utf-8"))
        self.config = json.loads(
            self.comparison_config_path.read_text(encoding="utf-8")
        )
        validate_config(self.config)
        self.method = self.config["scope"]["methods"][0]
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
            raise ValueError("official DirectLLM test requires method lock and fixed setting")
        if self.fixed_test_cache is None:
            self.fixed_test_cache, self.fixed_test_manifest = fixed_test_scenarios(
                self.root, self.fixed_test_setting
            )
        return self.fixed_test_cache[dataset]

    def _provenance(self):
        return {
            "comparison_config_sha256": file_hash(self.comparison_config_path),
            "prompt_sha256": prompt_hashes(self.method),
            "qwen_model_lock_sha256": file_hash(self.root / "configs/models.lock.json"),
            "policy_contract": {
                "version": policy_version(self.method),
                "training": "none",
                "checkpoint": "none",
                "retrieval": "none",
                "planning": "none",
                "assistant_calls_per_executed_turn": 1,
                "policy_embedding_calls_per_turn": 0,
            },
            "code": code_provenance(self.root),
        }

    def run(self, split, datasets=None):
        if split not in {"dev", "test"}:
            raise ValueError("DirectLLM comparison split must be dev or test")
        datasets = tuple(datasets or self.config["datasets"])
        if (
            not datasets
            or len(set(datasets)) != len(datasets)
            or any(dataset not in self.config["datasets"] for dataset in datasets)
        ):
            raise ValueError("invalid DirectLLM evaluation dataset shard")
        provenance = self._provenance()
        selection_sha256 = "development"
        if split == "test":
            envelope = verify_selection(self.selection_lock)
            selection_sha256 = envelope["sha256"]
            for key in (
                "comparison_config_sha256",
                "qwen_model_lock_sha256",
                "code",
            ):
                if envelope["payload"][key] != provenance[key]:
                    raise ValueError(f"DirectLLM test runtime changed locked {key}")
            locked = envelope["payload"]["prompt_contract"]
            locked_prompt = locked.get(
                "system_prompt_sha256", locked.get("system_prompt_sha256_by_dataset")
            )
            if (
                locked["policy_version"] != provenance["policy_contract"]["version"]
                or locked_prompt != provenance["prompt_sha256"]
            ):
                raise ValueError("DirectLLM test prompt changed after method lock")
        seeds, budgets = evaluation_axes(self.fixed_test_setting, self.config, split)
        split_output = evaluation_output_directory(
            self.output, split, self.fixed_test_setting
        )
        rows = []
        started = time.monotonic()
        with Servers(self.root, split_output / "servers", self.runtime) as servers:
            backends = servers.backends()
            generation, embedding = backends["generation"], backends["embedding"]
            for dataset in datasets:
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
                                    raise ValueError("DirectLLM supports natural budget only")
                                planner = planner_for_method(
                                    self.method,
                                    generation,
                                    scenario,
                                    dataset,
                                    Decoding(**self.config["assistant_decoding"]),
                                )
                                contract = {
                                    "phase": split,
                                    "method": self.method,
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
                                destination = (
                                    split_output
                                    / dataset
                                    / protocol
                                    / str(seed)
                                    / budget
                                    / scenario["id"]
                                )
                                result = run_dialogue(
                                    scenario,
                                    planner,
                                    generation,
                                    embedding,
                                    protocol,
                                    seed,
                                    destination,
                                    contract,
                                    self.config["turn_budget"],
                                    Decoding(**self.config["simulator_decoding"]),
                                    judge=LocalJudge(generation, self.config["judge"]),
                                )
                                if result["status"] != "complete":
                                    raise ValueError(
                                        "partial DirectLLM dialogue cannot enter a report"
                                    )
                                roles, role_tokens = _role_counts(result)
                                if roles.get("assistant", 0) != result["actual_turns"]:
                                    raise ValueError(
                                        "DirectLLM must make one assistant call per executed turn"
                                    )
                                if result["embedding_requests"] != 0:
                                    raise ValueError("DirectLLM policy must not request embeddings")
                                seal_sample(
                                    destination / "completed-seal.json",
                                    {
                                        "dialogue_sha256": file_hash(
                                            destination / "dialogue.json"
                                        ),
                                        "selection_sha256": selection_sha256,
                                    },
                                )
                                rows.append(
                                    {
                                        "dataset": dataset,
                                        "protocol": protocol,
                                        "budget": budget,
                                        "direct_assistant_requests": roles.get("assistant", 0),
                                        "direct_planner_seconds": sum(
                                            turn["plan"]["compute"]["planner_latency_s"]
                                            for turn in result["turns_trace"]
                                        ),
                                        "generation_role_counts": roles,
                                        "generation_role_tokens": role_tokens,
                                        **result,
                                    }
                                )
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
                role_names = sorted(
                    {name for row in group for name in row["generation_role_counts"]}
                )
                summary.append(
                    {
                        "dataset": dataset,
                        "protocol": protocol,
                        "method": self.method,
                        "budget": "natural",
                        **aggregate(group),
                        "assistant_requests_per_dialogue": sum(
                            row["direct_assistant_requests"] for row in group
                        )
                        / len(group),
                        "planner_seconds_per_dialogue": sum(
                            row["direct_planner_seconds"] for row in group
                        )
                        / len(group),
                        "embedding_requests_per_dialogue": 0.0,
                        "generation_requests_by_role": {
                            role: sum(
                                row["generation_role_counts"].get(role, 0)
                                for row in group
                            )
                            for role in role_names
                        },
                        "generation_tokens_by_role": {
                            role: sum(
                                row["generation_role_tokens"].get(role, 0)
                                for row in group
                            )
                            for role in role_names
                        },
                    }
                )
        report = {
            "schema_version": 1,
            "status": "complete",
            "split": split,
            "method": self.method,
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
        report_name = "direct-llm-report.json"
        if split == "test":
            report_name = (
                "direct-llm-report-fixed1000-prefix-"
                f"{self.fixed_test_setting['count']}.json"
            )
        atomic_json(split_output / report_name, report)
        return report
