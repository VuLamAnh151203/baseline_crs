"""Isolated TRIP extension using the frozen shared interactive evaluator."""

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
from .baselines.trip.config import TripConfig
from .baselines.trip.planner import TripPlanner
from .baselines.trip.runtime import load_inference
from .baselines.trip.selection import verify_selection
from .contracts import Decoding
from .evaluation import run_dialogue
from .metrics import aggregate, validate_records
from .official_data import SPLITS, read_release
from .provenance import atomic_json, digest, file_hash
from .runtime import Servers


def validate_config(config):
    expected = {
        "methods": ["PPDPP", "DiffTOD", "TRIP", "TDRL-R", "TDRL"],
        "claim": "five_method_automated_comparison",
    }
    if config.get("schema_version") != 1 or config.get("scope") != expected:
        raise ValueError("unknown TRIP comparison configuration")
    if config.get("datasets") != ["TopDial", "DuRecDial2.0"]:
        raise ValueError("TRIP comparison requires both datasets")
    if config.get("seeds") != [17, 29, 43] or config.get("protocols") != [
        "profile",
        "resistant",
    ]:
        raise ValueError("TRIP comparison requires frozen seeds and protocols")
    if config.get("turn_budget") != 10 or config.get("budgets") != ["natural", "matched"]:
        raise ValueError("TRIP comparison requires frozen turns and both budgets")
    if (
        not isinstance(config.get("development_scenarios"), int)
        or config["development_scenarios"] < 1
    ):
        raise ValueError("invalid TRIP development scenario count")
    for name in ("assistant_decoding", "simulator_decoding"):
        Decoding(**config[name])


def _processed(root, dataset):
    path = root / "data/processed" / dataset / "dev.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not rows or len({row["id"] for row in rows}) != len(rows):
        raise ValueError("prepared TRIP evaluation scenarios are empty or duplicated")
    return rows


class TripComparison:
    def __init__(
        self,
        root,
        output,
        config_path,
        checkpoints,
        trip_configs,
        encoder,
        tokenizer,
        asset_lock,
        selection_lock=None,
        fixed_test_setting=None,
        fixed_test_count=None,
    ):
        self.root, self.output = Path(root), Path(output)
        self.config_path = Path(config_path)
        self.checkpoints = {key: Path(value) for key, value in checkpoints.items()}
        self.trip_configs = {
            dataset: {budget: Path(path) for budget, path in values.items()}
            for dataset, values in trip_configs.items()
        }
        self.encoder, self.tokenizer, self.asset_lock = map(
            Path, (encoder, tokenizer, asset_lock)
        )
        self.selection_lock = Path(selection_lock) if selection_lock else None
        self.fixed_test_setting = load_fixed_test_setting(
            root, fixed_test_setting, fixed_test_count
        )
        self.fixed_test_manifest = None
        self.fixed_test_cache = None
        self.config = json.loads(self.config_path.read_text(encoding="utf-8"))
        validate_config(self.config)
        if set(self.checkpoints) != set(self.config["datasets"]):
            raise ValueError("TRIP comparison requires one checkpoint per dataset")
        if set(self.trip_configs) != set(self.config["datasets"]) or any(
            set(values) != {"natural", "matched"} for values in self.trip_configs.values()
        ):
            raise ValueError("TRIP comparison requires natural/matched configs per dataset")
        self.runtime = json.loads(
            (self.root / "configs/runtime.json").read_text(encoding="utf-8")
        )

    def _scenarios(self, dataset, split):
        if split == "dev":
            rows = _processed(self.root, dataset)
            return sorted(rows, key=lambda row: digest([dataset, row["id"]]))[
                : self.config["development_scenarios"]
            ]
        if split != "test" or self.selection_lock is None:
            raise ValueError("official TRIP test requires a verified selection lock")
        if self.fixed_test_setting is not None:
            if self.fixed_test_cache is None:
                self.fixed_test_cache, self.fixed_test_manifest = fixed_test_scenarios(
                    self.root, self.fixed_test_setting
                )
            return self.fixed_test_cache[dataset]
        return [
            record
            for member in SPLITS[dataset]
            if member.startswith("test")
            for record in read_release(self.root, dataset, member, selection_verified=True)
        ]

    def _provenance(self):
        return {
            "config_sha256": file_hash(self.config_path),
            "checkpoint_sha256": {
                dataset: file_hash(path) for dataset, path in self.checkpoints.items()
            },
            "trip_config_sha256": {
                dataset: {
                    budget: TripConfig.from_path(path).sha256
                    for budget, path in values.items()
                }
                for dataset, values in self.trip_configs.items()
            },
            "asset_lock_sha256": file_hash(self.asset_lock),
        }

    def run(self, split):
        if split not in {"dev", "test"}:
            raise ValueError("TRIP comparison split must be dev or test")
        provenance = self._provenance()
        selection_sha256 = "development"
        if split == "test":
            envelope = verify_selection(self.selection_lock)
            selection_sha256 = envelope["sha256"]
            for key, value in provenance.items():
                if envelope["payload"][key] != value:
                    raise ValueError("TRIP test runtime differs from frozen selection")
        seeds, budgets = evaluation_axes(self.fixed_test_setting, self.config, split)
        split_output = evaluation_output_directory(
            self.output, split, self.fixed_test_setting
        )
        rows = []
        started = time.monotonic()
        with Servers(self.root, split_output / "servers", self.runtime) as servers:
            backends = servers.backends()
            for dataset in self.config["datasets"]:
                configs = {
                    budget: TripConfig.from_path(path)
                    for budget, path in self.trip_configs[dataset].items()
                }
                if configs["natural"].training_sha256 != configs["matched"].training_sha256:
                    raise ValueError("TRIP natural/matched configs must share fitted weights")
                model, tokenizer, token_ids, _, _ = load_inference(
                    self.trip_configs[dataset]["natural"],
                    self.checkpoints[dataset],
                    self.encoder,
                    self.tokenizer,
                    self.asset_lock,
                    "cuda",
                )
                scenarios = self._scenarios(dataset, split)
                for protocol in self.config["protocols"]:
                    for seed in seeds:
                        for scenario in scenarios:
                            for budget in budgets:
                                planner = TripPlanner(
                                    backends["generation"],
                                    model,
                                    tokenizer,
                                    token_ids,
                                    configs[budget],
                                    scenario,
                                    provenance["checkpoint_sha256"][dataset],
                                    Decoding(**self.config["assistant_decoding"]),
                                )
                                contract = {
                                    "phase": split,
                                    "method": "TRIP",
                                    "selection_sha256": selection_sha256,
                                    "budget": budget,
                                    "target_nodes": [],
                                    "request_namespace": digest(
                                        {
                                            **provenance,
                                            "dataset": dataset,
                                            "budget": budget,
                                            "selection": selection_sha256,
                                            "fixed_test_resource": (
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
                                )
                                result.setdefault(
                                    "trip_sequence_forwards",
                                    sum(
                                        turn["plan"]["compute"].get(
                                            "trip_sequence_forwards", 0
                                        )
                                        for turn in result["turns_trace"]
                                    ),
                                )
                                result.setdefault(
                                    "trip_planner_seconds",
                                    sum(
                                        turn["plan"]["compute"].get(
                                            "planner_latency_s", 0.0
                                        )
                                        for turn in result["turns_trace"]
                                    ),
                                )
                                result.setdefault(
                                    "trip_peak_memory_bytes", torch.cuda.max_memory_allocated()
                                )
                                atomic_json(destination / "dialogue.json", result)
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
        for dataset in self.config["datasets"]:
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
                    trip_compute = {
                        "trip_sequence_forwards_mean": sum(
                            row["trip_sequence_forwards"] for row in group
                        )
                        / len(group),
                        "trip_planner_seconds_mean": sum(
                            row["trip_planner_seconds"] for row in group
                        )
                        / len(group),
                        "trip_peak_memory_bytes_max": max(
                            row["trip_peak_memory_bytes"] for row in group
                        ),
                    }
                    summary.append(
                        {
                            "dataset": dataset,
                            "protocol": protocol,
                            "method": "TRIP",
                            "budget": budget,
                            **aggregate(group),
                            **trip_compute,
                        }
                    )
        report = {
            "schema_version": 1,
            "status": "complete",
            "split": split,
            "method": "TRIP",
            "selection_sha256": selection_sha256,
            **provenance,
            "dialogues": len(rows),
            "wall_s": time.monotonic() - started,
            "server_compute": compute,
            "summary": summary,
            "evaluation_seeds": seeds,
            "evaluation_budgets": budgets,
        }
        if split == "test" and self.fixed_test_setting is not None:
            report["dataset_setting"] = self.fixed_test_setting["payload"]
            report["test_sample"] = self.fixed_test_manifest
        report_path = split_output / "trip-report.json"
        if split == "test" and self.fixed_test_setting is not None:
            report_path = split_output / (
                f"trip-report-fixed1000-prefix-{self.fixed_test_setting['count']}.json"
            )
        atomic_json(report_path, report)
        return report
