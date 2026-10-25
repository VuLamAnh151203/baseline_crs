"""Isolated DPDP extension using the frozen shared interactive evaluator."""

import gc
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from .baseline_test_setting import (
    evaluation_axes,
    evaluation_output_directory,
    fixed_test_scenarios,
    load_fixed_test_setting,
)
from .baselines.dpdp.config import DPDPConfig
from .baselines.dpdp.planner import DPDPPlanner
from .baselines.dpdp.runtime import load_inference
from .baselines.dpdp.selection import verify_selection
from .contracts import Decoding
from .evaluation import run_dialogue
from .judge import LocalJudge
from .metrics import aggregate, validate_records
from .official_data import SPLITS, read_release
from .provenance import atomic_json, digest, file_hash
from .runtime import Servers
from .stages import code_provenance
from .test_prefix import seal_sample
from .test_sampling import sample_test_scenarios, validate_sampling


def validate_config(config):
    expected = {
        "methods": ["PPDPP", "DiffTOD", "TRIP", "DPDP", "TDRL-R", "TDRL"],
        "claim": "six_method_automated_comparison",
    }
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
    if set(config) != exact or config["schema_version"] != 2 or config["scope"] != expected:
        raise ValueError("unknown/incomplete DPDP comparison configuration")
    if config["datasets"] != ["TopDial", "DuRecDial2.0"]:
        raise ValueError("DPDP comparison requires both datasets")
    if config["seeds"] != [17] or config["protocols"] != ["profile", "resistant"]:
        raise ValueError("DPDP comparison requires frozen seeds and protocols")
    if config["turn_budget"] != 10 or config["budgets"] != ["natural", "matched"]:
        raise ValueError("DPDP comparison requires frozen turns and both budgets")
    if not isinstance(config["development_scenarios"], int) or config["development_scenarios"] < 1:
        raise ValueError("invalid DPDP development scenario count")
    for name in ("assistant_decoding", "simulator_decoding"):
        Decoding(**config[name])
    if config["judge"].get("schema_version") != 2 or config["judge"].get("critic_samples") != 10:
        raise ValueError("DPDP comparison requires the shared semantic judge")
    Decoding(**config["judge"]["outcome_decoding"])
    Decoding(**config["judge"]["critic_decoding"])
    validate_sampling(config["test_sampling"])
    if config["test_sampling"]["total_scenarios"] != 500:
        raise ValueError("DPDP comparison requires the frozen 500-scenario test subset")


def _processed(root, dataset):
    path = root / "data/processed" / dataset / "dev.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not rows or len({row["id"] for row in rows}) != len(rows):
        raise ValueError("prepared DPDP evaluation scenarios are empty or duplicated")
    return rows


class DPDPComparison:
    def __init__(
        self,
        root,
        output,
        config_path,
        checkpoints,
        dpdp_configs,
        encoder,
        tokenizer,
        asset_lock,
        selection_lock=None,
        runtime_config=None,
        fixed_test_setting=None,
        fixed_test_count=None,
    ):
        self.root, self.output = Path(root), Path(output)
        self.config_path = Path(config_path)
        self.checkpoints = {key: Path(value) for key, value in checkpoints.items()}
        self.dpdp_configs = {
            dataset: {budget: Path(path) for budget, path in values.items()}
            for dataset, values in dpdp_configs.items()
        }
        self.encoder, self.tokenizer, self.asset_lock = map(Path, (encoder, tokenizer, asset_lock))
        self.selection_lock = Path(selection_lock) if selection_lock else None
        self.fixed_test_setting = load_fixed_test_setting(
            root, fixed_test_setting, fixed_test_count
        )
        self.fixed_test_manifest = None
        self.fixed_test_cache = None
        self.test_sample_manifest = None
        self.test_sample_cache = None
        self.config = json.loads(self.config_path.read_text(encoding="utf-8"))
        validate_config(self.config)
        if set(self.checkpoints) != set(self.config["datasets"]):
            raise ValueError("DPDP comparison requires one checkpoint per dataset")
        if set(self.dpdp_configs) != set(self.config["datasets"]) or any(
            set(values) != {"natural", "matched"} for values in self.dpdp_configs.values()
        ):
            raise ValueError("DPDP comparison requires natural/matched configs per dataset")
        self.runtime_path = (
            Path(runtime_config) if runtime_config else self.root / "configs/runtime.json"
        )
        self.runtime = json.loads(self.runtime_path.read_text(encoding="utf-8"))

    def _scenarios(self, dataset, split):
        if split == "dev":
            rows = _processed(self.root, dataset)
            return sorted(rows, key=lambda row: digest([dataset, row["id"]]))[
                : self.config["development_scenarios"]
            ]
        if split != "test" or self.selection_lock is None:
            raise ValueError("official DPDP test requires a verified selection lock")
        if self.fixed_test_setting is not None:
            if self.fixed_test_cache is None:
                self.fixed_test_cache, self.fixed_test_manifest = fixed_test_scenarios(
                    self.root, self.fixed_test_setting
                )
            return self.fixed_test_cache[dataset]
        if self.test_sample_cache is None:
            populations = {
                name: [
                    record
                    for member in SPLITS[name]
                    if member.startswith("test")
                    for record in read_release(self.root, name, member, selection_verified=True)
                ]
                for name in self.config["datasets"]
            }
            self.test_sample_cache, self.test_sample_manifest = sample_test_scenarios(
                populations, self.config["test_sampling"]
            )
        return self.test_sample_cache[dataset]

    def _provenance(self):
        configs = {
            dataset: {budget: DPDPConfig.from_path(path) for budget, path in values.items()}
            for dataset, values in self.dpdp_configs.items()
        }
        for values in configs.values():
            if values["natural"].gate_threshold != values["matched"].gate_threshold:
                raise ValueError("DPDP natural/matched budgets must share the frozen gate")
        offline_checkpoint_sha256 = {}
        for dataset, path in self.checkpoints.items():
            payload = torch.load(path, map_location="cpu", weights_only=False)
            parent = payload.get("metadata", {}).get("parent_checkpoint_sha256")
            if not isinstance(parent, str) or len(parent) != 64:
                raise ValueError("DPDP final checkpoint lacks offline parent provenance")
            offline_checkpoint_sha256[dataset] = parent
        return {
            "config_sha256": file_hash(self.config_path),
            "checkpoint_sha256": {
                dataset: file_hash(path) for dataset, path in self.checkpoints.items()
            },
            "offline_checkpoint_sha256": offline_checkpoint_sha256,
            "dpdp_config_sha256": {
                dataset: {budget: value.sha256 for budget, value in values.items()}
                for dataset, values in configs.items()
            },
            "training_config_sha256": {
                dataset: next(iter({value.training_sha256 for value in values.values()}))
                for dataset, values in configs.items()
                if len({value.training_sha256 for value in values.values()}) == 1
            },
            "gate_thresholds": {
                dataset: {budget: value.gate_threshold for budget, value in values.items()}
                for dataset, values in configs.items()
            },
            "search_budgets": {
                dataset: {
                    budget: {
                        "simulations": value.mcts_simulations,
                        "depth": value.mcts_max_depth,
                    }
                    for budget, value in values.items()
                }
                for dataset, values in configs.items()
            },
            "asset_lock_sha256": file_hash(self.asset_lock),
            "code": code_provenance(self.root),
            "qwen_model_lock_sha256": file_hash(self.root / "configs/models.lock.json"),
        }

    def run(self, split, datasets=None):
        if split not in {"dev", "test"}:
            raise ValueError("DPDP comparison split must be dev or test")
        datasets = tuple(datasets or self.config["datasets"])
        if (
            not datasets
            or len(set(datasets)) != len(datasets)
            or any(dataset not in self.config["datasets"] for dataset in datasets)
        ):
            raise ValueError("invalid DPDP evaluation dataset shard")
        provenance = self._provenance()
        if set(provenance["training_config_sha256"]) != set(self.config["datasets"]):
            raise ValueError("DPDP natural/matched configs must share fitted weights")
        selection_sha256 = "development"
        if split == "test":
            envelope = verify_selection(self.selection_lock)
            selection_sha256 = envelope["sha256"]
            for key, value in provenance.items():
                if envelope["payload"][key] != value:
                    raise ValueError("DPDP test runtime differs from frozen selection")
        seeds, budgets = evaluation_axes(self.fixed_test_setting, self.config, split)
        split_output = evaluation_output_directory(self.output, split, self.fixed_test_setting)
        if split == "test" and self.fixed_test_setting is None:
            split_output = self.output / "test-sample500"
        rows = []
        resumed_complete = 0
        resumed_partial = 0
        executed = 0
        started = time.monotonic()
        progress_path = split_output / "evaluation-progress.jsonl"
        expected_dialogues = sum(
            len(self._scenarios(dataset, split))
            * len(self.config["protocols"])
            * len(seeds)
            * len(budgets)
            for dataset in datasets
        )
        with Servers(self.root, split_output / "servers", self.runtime) as servers:
            backends = servers.backends()
            for dataset in datasets:
                configs = {
                    budget: DPDPConfig.from_path(path)
                    for budget, path in self.dpdp_configs[dataset].items()
                }
                model, tokenizer, _, _, _ = load_inference(
                    self.dpdp_configs[dataset]["natural"],
                    self.checkpoints[dataset],
                    self.encoder,
                    self.tokenizer,
                    self.asset_lock,
                    "cuda",
                )
                model.to(dtype=torch.bfloat16).eval()
                scenarios = self._scenarios(dataset, split)
                if split == "test" and self.fixed_test_setting is not None:
                    seal_sample(
                        split_output / "sealed-sample.json",
                        self.fixed_test_manifest["sample"],
                    )
                for protocol in self.config["protocols"]:
                    for seed in seeds:
                        for scenario in scenarios:
                            for budget in budgets:
                                planner = DPDPPlanner(
                                    backends["generation"],
                                    model,
                                    tokenizer,
                                    configs[budget],
                                    scenario,
                                    provenance["checkpoint_sha256"][dataset],
                                    Decoding(**self.config["assistant_decoding"]),
                                    protocol,
                                )
                                contract = {
                                    "phase": split,
                                    "method": "DPDP",
                                    "selection_sha256": selection_sha256,
                                    "budget": budget,
                                    "target_nodes": [],
                                    "request_namespace": digest(
                                        {
                                            **provenance,
                                            "dataset": dataset,
                                            "budget": budget,
                                            "selection": selection_sha256,
                                            "test_sample_resource": (
                                                self.fixed_test_setting["payload"]["test"][
                                                    "resource_sha256"
                                                ]
                                                if self.fixed_test_setting
                                                else (
                                                    self.test_sample_manifest.get(
                                                        "frozen_subset", {}
                                                    ).get("sha256")
                                                    if self.test_sample_manifest
                                                    else None
                                                )
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
                                snapshot = destination / "dialogue.json"
                                prior_status = None
                                if snapshot.exists():
                                    prior_status = json.loads(
                                        snapshot.read_text(encoding="utf-8")
                                    ).get("status")
                                if prior_status != "complete":
                                    torch.cuda.reset_peak_memory_stats()
                                result = run_dialogue(
                                    scenario,
                                    planner,
                                    backends["generation"],
                                    backends["embedding"],
                                    protocol,
                                    seed,
                                    destination,
                                    contract,
                                    self.config["turn_budget"],
                                    Decoding(**self.config["simulator_decoding"]),
                                    judge=LocalJudge(backends["generation"], self.config["judge"]),
                                )
                                if result.get("status") != "complete":
                                    raise ValueError("partial DPDP dialogue cannot enter a report")
                                derived = {
                                    "dpdp_mcts_turns": sum(
                                        turn["plan"]["compute"].get("selected_system") == "system2"
                                        for turn in result["turns_trace"]
                                    ),
                                    "dpdp_policy_turns": sum(
                                        turn["plan"]["compute"].get("selected_system") == "system1"
                                        for turn in result["turns_trace"]
                                    ),
                                    "dpdp_mcts_simulations": sum(
                                        turn["plan"]["compute"].get("mcts_simulations", 0)
                                        for turn in result["turns_trace"]
                                    ),
                                    "dpdp_planner_seconds": sum(
                                        turn["plan"]["compute"].get("planner_latency_s", 0.0)
                                        for turn in result["turns_trace"]
                                    ),
                                }
                                for key, value in derived.items():
                                    if key in result and result[key] != value:
                                        raise ValueError(
                                            f"resumed DPDP dialogue changed derived field {key}"
                                        )
                                    result[key] = value
                                if "dpdp_peak_memory_bytes" not in result:
                                    result["dpdp_peak_memory_bytes"] = (
                                        torch.cuda.max_memory_allocated()
                                    )
                                atomic_json(destination / "dialogue.json", result)
                                seal_sample(
                                    destination / "completed-seal.json",
                                    {
                                        "dialogue_sha256": file_hash(destination / "dialogue.json"),
                                        "selection_sha256": selection_sha256,
                                    },
                                )
                                resumed_complete += int(prior_status == "complete")
                                resumed_partial += int(prior_status == "partial")
                                executed += int(prior_status != "complete")
                                progress = {
                                    "timestamp": datetime.now(timezone.utc).isoformat(),
                                    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                                    "split": split,
                                    "dataset": dataset,
                                    "protocol": protocol,
                                    "seed": seed,
                                    "budget": budget,
                                    "scenario_id": scenario["id"],
                                    "prior_status": prior_status or "new",
                                    "status": "complete",
                                    "dialogues_collected": len(rows) + 1,
                                }
                                with progress_path.open("a", encoding="utf-8") as handle:
                                    handle.write(
                                        json.dumps(progress, ensure_ascii=False, sort_keys=True)
                                        + "\n"
                                    )
                                    handle.flush()
                                    os.fsync(handle.fileno())
                                atomic_json(
                                    split_output / "evaluation-status.json",
                                    {
                                        "schema_version": 1,
                                        "status": "running",
                                        "updated_at": progress["timestamp"],
                                        "slurm_job_id": progress["slurm_job_id"],
                                        "split": split,
                                        "scope_datasets": list(datasets),
                                        "completed_dialogues": len(rows) + 1,
                                        "expected_dialogues": expected_dialogues,
                                        "percent": round(
                                            100 * (len(rows) + 1) / expected_dialogues, 2
                                        ),
                                        "current": progress,
                                    },
                                )
                                print(json.dumps(progress, sort_keys=True), flush=True)
                                rows.append(
                                    {
                                        "dataset": dataset,
                                        "protocol": protocol,
                                        "budget": budget,
                                        **result,
                                    }
                                )
                del model
                gc.collect()
                torch.cuda.empty_cache()
            compute = servers.compute_metadata()
        summary = []
        for dataset in datasets:
            scenario_ids = [row["id"] for row in self._scenarios(dataset, split)]
            for protocol in self.config["protocols"]:
                for budget in budgets:
                    group = [
                        row
                        for row in rows
                        if row["dataset"] == dataset
                        and row["protocol"] == protocol
                        and row["budget"] == budget
                    ]
                    validate_records(group, scenario_ids, seeds, selection_sha256)
                    summary.append(
                        {
                            "dataset": dataset,
                            "protocol": protocol,
                            "method": "DPDP",
                            "budget": budget,
                            **aggregate(group),
                            "mcts_turns_mean": sum(row["dpdp_mcts_turns"] for row in group)
                            / len(group),
                            "policy_turns_mean": sum(row["dpdp_policy_turns"] for row in group)
                            / len(group),
                            "mcts_simulations_mean": sum(
                                row["dpdp_mcts_simulations"] for row in group
                            )
                            / len(group),
                            "planner_seconds_mean": sum(
                                row["dpdp_planner_seconds"] for row in group
                            )
                            / len(group),
                            "peak_memory_bytes_max": max(
                                row["dpdp_peak_memory_bytes"] for row in group
                            ),
                        }
                    )
        report = {
            "schema_version": 1,
            "status": "complete",
            "split": split,
            "method": "DPDP",
            "selection_sha256": selection_sha256,
            **provenance,
            "dialogues": len(rows),
            "wall_s": time.monotonic() - started,
            "server_compute": compute,
            "summary": summary,
            "gate_evidence": {
                dataset: {
                    budget: {
                        "evaluation_scenario_ids": [
                            scenario["id"] for scenario in self._scenarios(dataset, split)
                        ],
                        "confidence_gaps": [
                            turn["plan"]["compute"]["confidence_gap"]
                            for row in rows
                            if row["dataset"] == dataset and row["budget"] == budget
                            for turn in row["turns_trace"]
                            if "confidence_gap" in turn["plan"]["compute"]
                        ],
                        "policy_turns": sum(
                            row["dpdp_policy_turns"]
                            for row in rows
                            if row["dataset"] == dataset and row["budget"] == budget
                        ),
                        "mcts_turns": sum(
                            row["dpdp_mcts_turns"]
                            for row in rows
                            if row["dataset"] == dataset and row["budget"] == budget
                        ),
                    }
                    for budget in budgets
                }
                for dataset in datasets
            },
            "evaluation_seeds": seeds,
            "evaluation_budgets": budgets,
            "scope_datasets": list(datasets),
            "runtime_config_sha256": file_hash(self.runtime_path),
            "resume_accounting": {
                "reused_complete_dialogues": resumed_complete,
                "continued_partial_dialogues": resumed_partial,
                "executed_or_continued_dialogues": executed,
                "progress_log": str(progress_path),
            },
        }
        if split == "test" and self.fixed_test_setting is not None:
            report["dataset_setting"] = self.fixed_test_setting["payload"]
            report["test_sample"] = self.fixed_test_manifest
        elif split == "test":
            report["test_sample"] = self.test_sample_manifest
        report_path = split_output / "dpdp-report.json"
        if split == "test" and self.fixed_test_setting is not None:
            report_path = split_output / (
                f"dpdp-report-fixed1000-prefix-{self.fixed_test_setting['count']}.json"
            )
        atomic_json(report_path, report)
        atomic_json(
            split_output / "evaluation-status.json",
            {
                "schema_version": 1,
                "status": "complete",
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                "split": split,
                "scope_datasets": list(datasets),
                "completed_dialogues": len(rows),
                "expected_dialogues": expected_dialogues,
                "percent": 100.0,
                "report": str(report_path),
                "report_sha256": file_hash(report_path),
            },
        )
        return report
