"""Independent EnPL reconstruction under the shared interactive evaluator."""

import json
import os
import time
from pathlib import Path

from .baseline_test_setting import (
    evaluation_axes,
    evaluation_output_directory,
    fixed_test_scenarios,
    load_fixed_test_setting,
)
from .baselines.enpl.bank import load_bank
from .baselines.enpl.config import load_enpl_config
from .baselines.enpl.planner import EnPLPlanner
from .baselines.enpl.retrieval import DomainEntityOverlapRetriever
from .baselines.enpl.selection import (
    prompt_contract,
    verify_reference_lock,
    verify_selection,
)
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
    scope = {
        "methods": ["EnPL"],
        "claim": "independent_reconstruction_adapted_shared_fixed1000_evaluation",
    }
    if set(config) != exact or config["schema_version"] != 1:
        raise ValueError("unknown/incomplete EnPL comparison configuration")
    if config["scope"] != scope or config["datasets"] != list(DATASETS):
        raise ValueError("unexpected EnPL comparison scope")
    if config["seeds"] != [17] or config["protocols"] != ["profile", "resistant"]:
        raise ValueError("EnPL requires the frozen seed and protocols")
    if config["turn_budget"] != 10 or config["budgets"] != ["natural"]:
        raise ValueError("EnPL fixed evaluation is natural-budget only")
    if (
        not isinstance(config["development_scenarios"], int)
        or config["development_scenarios"] < 1
    ):
        raise ValueError("invalid EnPL development scenario count")
    for name in ("assistant_decoding", "simulator_decoding"):
        Decoding(**config[name])
    judge = config["judge"]
    if judge.get("schema_version") != 2 or judge.get("critic_samples") != 10:
        raise ValueError("EnPL requires the shared semantic judge")
    Decoding(**judge["outcome_decoding"])
    Decoding(**judge["critic_decoding"])
    if config["test_sampling"] != {
        "schema_version": 1,
        "method": "pooled_uniform_without_replacement",
        "total_scenarios": 500,
        "seed": 42,
    }:
        raise ValueError("EnPL legacy sample metadata changed")


def _processed(root, dataset):
    path = Path(root) / "data/processed" / dataset / "dev.jsonl"
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows or len({row["id"] for row in rows}) != len(rows):
        raise ValueError("prepared EnPL development scenarios are empty or duplicated")
    return rows


def _role_counts(result):
    generations = [item for turn in result["turns_trace"] for item in turn["generation"]]
    roles, tokens = {}, {}
    for item in generations:
        role = item["role"]
        roles[role] = roles.get(role, 0) + 1
        tokens[role] = tokens.get(role, 0) + item["total_tokens"]
    return roles, tokens


class EnPLComparison:
    def __init__(
        self,
        root,
        output,
        comparison_config,
        enpl_config,
        source_lock,
        bank,
        bank_manifest,
        selection_lock=None,
        runtime_config=None,
        fixed_test_setting=None,
        fixed_test_count=None,
        recovery_amendment=None,
    ):
        self.root, self.output = Path(root), Path(output)
        self.comparison_path = Path(comparison_config)
        self.enpl_config_path = Path(enpl_config)
        self.source_lock_path = Path(source_lock)
        self.bank_path, self.bank_manifest_path = Path(bank), Path(bank_manifest)
        verify_reference_lock(self.source_lock_path)
        self.method_config = load_enpl_config(self.enpl_config_path)
        self.entries, self.bank_manifest = load_bank(
            self.bank_path, self.bank_manifest_path, require_complete=True
        )
        expected_data = self.method_config["data"]
        if (
            self.bank_manifest["source_zip_sha256"]
            != expected_data["expected_source_zip_sha256"]
            or self.bank_manifest["source_member_sha256"]
            != expected_data["expected_source_member_sha256"]
        ):
            raise ValueError("EnPL bank uses another released source")
        build = self.bank_manifest["build_config"]
        filtering = self.method_config["filtering"]
        if (
            build["expected_candidates_per_record"]
            != filtering["expected_candidates_per_record"]
            or build["keep_per_conversation"] != filtering["keep_per_conversation"]
            or build["extractor"]["implementation"] != filtering["extractor"]
            or build["extractor"]["keyword_budget"] != filtering["keyword_budget"]
        ):
            raise ValueError("EnPL bank was built with another filtering contract")
        if (
            self.bank_manifest["prepared_train_sha256"]
            != file_hash(self.root / "data/processed/DuRecDial2.0/train.jsonl")
            or self.bank_manifest["prepared_dev_sha256"]
            != file_hash(self.root / "data/processed/DuRecDial2.0/dev.jsonl")
        ):
            raise ValueError("EnPL bank mapping inputs changed")
        self.selection_lock = Path(selection_lock) if selection_lock else None
        self.runtime_path = (
            Path(runtime_config) if runtime_config else self.root / "configs/runtime.json"
        )
        self.runtime = json.loads(self.runtime_path.read_text(encoding="utf-8"))
        self.config = json.loads(self.comparison_path.read_text(encoding="utf-8"))
        validate_config(self.config)
        self.fixed_test_setting = load_fixed_test_setting(
            self.root, fixed_test_setting, fixed_test_count
        )
        self.fixed_test_cache = self.fixed_test_manifest = None
        self.recovery_amendment_path = (
            Path(recovery_amendment) if recovery_amendment else None
        )
        self.recovery_amendment = None
        if self.recovery_amendment_path:
            self.recovery_amendment = json.loads(
                self.recovery_amendment_path.read_text(encoding="utf-8")
            )
            if self.recovery_amendment.get("schema_version") != 1:
                raise ValueError("invalid EnPL recovery amendment")

    def _scenarios(self, dataset, split):
        if split == "dev":
            return sorted(
                _processed(self.root, dataset), key=lambda row: digest([dataset, row["id"]])
            )[: self.config["development_scenarios"]]
        if split != "test" or self.selection_lock is None or self.fixed_test_setting is None:
            raise ValueError("official EnPL test requires method lock and fixed setting")
        if self.fixed_test_cache is None:
            self.fixed_test_cache, self.fixed_test_manifest = fixed_test_scenarios(
                self.root, self.fixed_test_setting
            )
        return self.fixed_test_cache[dataset]

    def _provenance(self):
        code = code_provenance(self.root)
        if self.recovery_amendment:
            envelope = verify_selection(self.selection_lock)
            if (
                self.recovery_amendment.get("selection_sha256") != envelope["sha256"]
                or self.recovery_amendment.get("original_code")
                != envelope["payload"]["code"]
            ):
                raise ValueError("EnPL recovery does not match frozen selection")
            code = envelope["payload"]["code"]
        return {
            "comparison_config_sha256": file_hash(self.comparison_path),
            "enpl_config_sha256": file_hash(self.enpl_config_path),
            "source_lock_sha256": file_hash(self.source_lock_path),
            "bank_sha256": file_hash(self.bank_path),
            "bank_manifest_sha256": file_hash(self.bank_manifest_path),
            "qwen_model_lock_sha256": file_hash(self.root / "configs/models.lock.json"),
            "prompt_contract": prompt_contract(),
            "code": code,
        }

    def _authorize_recovery(self, generation, split_output, datasets):
        if not self.recovery_amendment or "test-fixed1000" not in str(split_output):
            return
        records = {
            row["record"]["request_id"]: row
            for row in self.recovery_amendment.get("unresolved_requests", [])
        }
        pending = sorted(
            (split_output / "servers/request-ledger/generation").glob("pending-*.json")
        )
        if len(pending) > 1:
            raise ValueError("multiple pending requests in one EnPL dataset worker")
        if not pending:
            return
        path = pending[0]
        record = json.loads(path.read_text(encoding="utf-8"))
        expected = records.get(record.get("request_id"))
        if not expected or file_hash(path) != expected["sha256"]:
            raise ValueError("EnPL pending request differs from recovery amendment")
        if not any(f"/{dataset}/" in str(path) for dataset in datasets):
            raise ValueError("EnPL pending request belongs to another dataset worker")
        archive = split_output / "servers/recovery-ledger" / path.name
        archive.parent.mkdir(parents=True, exist_ok=True)
        if archive.exists():
            raise ValueError("EnPL unresolved request was already archived")
        os.replace(path, archive)
        generation.authorized_reissue = record
        if record["request_id"].endswith("/enpl-plan"):
            generation.audited_final_text_repair_request_id = record["request_id"]
        atomic_json(
            split_output / "servers/recovery-authorization.json",
            {
                "schema_version": 1,
                "amendment_sha256": file_hash(self.recovery_amendment_path),
                "archived_pending_sha256": file_hash(archive),
                "request_id": record["request_id"],
                "request_sha256": record["request_sha256"],
            },
        )

    def run(self, split, datasets=None):
        if split not in {"dev", "test"}:
            raise ValueError("EnPL comparison split must be dev or test")
        datasets = tuple(datasets or self.config["datasets"])
        if (
            not datasets
            or len(set(datasets)) != len(datasets)
            or any(dataset not in self.config["datasets"] for dataset in datasets)
        ):
            raise ValueError("invalid EnPL evaluation dataset shard")
        provenance = self._provenance()
        selection_sha256 = "development"
        if split == "test":
            envelope = verify_selection(self.selection_lock)
            selection_sha256 = envelope["sha256"]
            for key, value in provenance.items():
                if envelope["payload"][key] != value:
                    raise ValueError(f"EnPL test runtime changed locked {key}")
        seeds, budgets = evaluation_axes(self.fixed_test_setting, self.config, split)
        split_output = evaluation_output_directory(
            self.output, split, self.fixed_test_setting
        )
        retriever = DomainEntityOverlapRetriever(
            self.entries,
            self.method_config["retrieval"]["k"],
            self.method_config["retrieval"]["distinct_source_conversations"],
        )
        rows = []
        started = time.monotonic()
        with Servers(self.root, split_output / "servers", self.runtime) as servers:
            backends = servers.backends()
            generation, embedding = backends["generation"], backends["embedding"]
            self._authorize_recovery(generation, split_output, datasets)
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
                                    raise ValueError("EnPL supports natural budget only")
                                planner = EnPLPlanner(
                                    generation,
                                    scenario,
                                    retriever,
                                    Decoding(**self.method_config["plan_decoding"]),
                                    Decoding(**self.config["assistant_decoding"]),
                                    self.method_config["context"][
                                        "max_input_characters"
                                    ],
                                )
                                contract = {
                                    "phase": split,
                                    "method": "EnPL",
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
                                    raise ValueError("partial EnPL dialogue cannot enter report")
                                roles, role_tokens = _role_counts(result)
                                if roles.get("planner", 0) != result["actual_turns"]:
                                    raise ValueError("EnPL must replan once per executed turn")
                                if roles.get("assistant", 0) != result["actual_turns"]:
                                    raise ValueError("EnPL must respond once per executed turn")
                                if result["embedding_requests"] != 0:
                                    raise ValueError("EnPL retrieval must not use embeddings")
                                compute = [
                                    turn["plan"]["compute"]
                                    for turn in result["turns_trace"]
                                ]
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
                                        "enpl_planner_requests": roles.get("planner", 0),
                                        "enpl_response_requests": roles.get("assistant", 0),
                                        "enpl_planner_seconds": sum(
                                            item["planner_latency_s"] for item in compute
                                        ),
                                        "enpl_examples_used": sum(
                                            item["examples_used"] for item in compute
                                        ),
                                        "enpl_zero_overlap_examples": sum(
                                            sum(item["zero_overlap_fallbacks"])
                                            for item in compute
                                        ),
                                        "enpl_format_valid_turns": sum(
                                            item["response_format_valid"] for item in compute
                                        ),
                                        "enpl_plan_valid_turns": sum(
                                            item["plan_format_valid"] for item in compute
                                        ),
                                        "enpl_plan_truncated_turns": sum(
                                            item["plan_truncated"] for item in compute
                                        ),
                                        "generation_role_counts": roles,
                                        "generation_role_tokens": role_tokens,
                                        **result,
                                    }
                                )
            if self.recovery_amendment:
                if getattr(generation, "authorized_reissue", None) is not None:
                    raise ValueError("EnPL authorized recovery request was not consumed")
                remaining = list(
                    (split_output / "servers/request-ledger/generation").glob(
                        "pending-*.json"
                    )
                )
                if remaining:
                    raise ValueError("EnPL recovery left an unresolved request")
            server_compute = servers.compute_metadata()
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
                turns = sum(row["actual_turns"] for row in group)
                examples = sum(row["enpl_examples_used"] for row in group)
                summary.append(
                    {
                        "dataset": dataset,
                        "protocol": protocol,
                        "method": "EnPL",
                        "budget": "natural",
                        **aggregate(group),
                        "planner_requests_per_dialogue": sum(
                            row["enpl_planner_requests"] for row in group
                        )
                        / len(group),
                        "response_requests_per_dialogue": sum(
                            row["enpl_response_requests"] for row in group
                        )
                        / len(group),
                        "planner_seconds_per_dialogue": sum(
                            row["enpl_planner_seconds"] for row in group
                        )
                        / len(group),
                        "examples_per_turn": examples / turns,
                        "zero_overlap_example_rate": sum(
                            row["enpl_zero_overlap_examples"] for row in group
                        )
                        / examples,
                        "response_format_valid_turn_rate": sum(
                            row["enpl_format_valid_turns"] for row in group
                        )
                        / turns,
                        "plan_format_valid_turn_rate": sum(
                            row["enpl_plan_valid_turns"] for row in group
                        )
                        / turns,
                        "plan_truncated_turn_rate": sum(
                            row["enpl_plan_truncated_turns"] for row in group
                        )
                        / turns,
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
            "method": "EnPL",
            "selection_sha256": selection_sha256,
            **provenance,
            "dialogues": len(rows),
            "wall_s": time.monotonic() - started,
            "server_compute": server_compute,
            "summary": summary,
            "evaluation_seeds": seeds,
            "evaluation_budgets": budgets,
            "scope_datasets": list(datasets),
            "runtime_config_sha256": file_hash(self.runtime_path),
        }
        if self.recovery_amendment_path:
            report["recovery_amendment_sha256"] = file_hash(
                self.recovery_amendment_path
            )
        if split == "test":
            report["dataset_setting"] = self.fixed_test_setting["payload"]
            report["test_sample"] = self.fixed_test_manifest
        name = "enpl-report.json"
        if split == "test":
            name = f"enpl-report-fixed1000-prefix-{self.fixed_test_setting['count']}.json"
        atomic_json(split_output / name, report)
        return report
