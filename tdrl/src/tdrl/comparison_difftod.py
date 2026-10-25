"""Isolated DiffTOD extension using the frozen shared interactive evaluator."""

import json
import time
from pathlib import Path

from .baseline_test_setting import (
    evaluation_axes,
    evaluation_output_directory,
    fixed_test_scenarios,
    load_fixed_test_setting,
)
from .baselines.difftod.config import DiffTODConfig
from .baselines.difftod.plan_store import PlanStore
from .baselines.difftod.planner import DiffTODPlanner
from .baselines.difftod.selection import verify_selection
from .contracts import Decoding
from .evaluation import run_dialogue, visible_context
from .metrics import aggregate, validate_records
from .official_data import SPLITS, read_release
from .provenance import atomic_json, digest, file_hash
from .runtime import Servers


def validate_config(config):
    expected_scope = {
        "methods": ["PPDPP", "DiffTOD", "TDRL-R", "TDRL"],
        "claim": "four_method_automated_comparison",
    }
    if config.get("schema_version") != 1 or config.get("scope") != expected_scope:
        raise ValueError("unknown DiffTOD comparison configuration")
    if config.get("datasets") != ["TopDial", "DuRecDial2.0"]:
        raise ValueError("DiffTOD comparison requires both datasets")
    if config.get("seeds") != [17, 29, 43] or config.get("protocols") != [
        "profile",
        "resistant",
    ]:
        raise ValueError("DiffTOD comparison requires frozen seeds and protocols")
    if config.get("turn_budget") != 10:
        raise ValueError("DiffTOD comparison requires the frozen ten-turn budget")
    if config.get("budgets") != ["natural", "matched"]:
        raise ValueError("DiffTOD comparison requires natural and matched budgets")
    for name in ("assistant_decoding", "simulator_decoding"):
        Decoding(**config[name])


def _processed(root, dataset, split):
    path = root / "data/processed" / dataset / f"{split}.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not rows or len({row["id"] for row in rows}) != len(rows):
        raise ValueError("prepared DiffTOD evaluation scenarios are empty or duplicated")
    return rows


class DiffTODComparison:
    def __init__(
        self,
        root,
        output,
        config_path,
        plan_root,
        checkpoints,
        plan_configs,
        selection_lock=None,
        fixed_test_setting=None,
        fixed_test_count=None,
    ):
        self.root = Path(root)
        self.output = Path(output)
        self.config_path = Path(config_path)
        self.plan_store = PlanStore(plan_root)
        if set(checkpoints) != {"TopDial", "DuRecDial2.0"}:
            raise ValueError("DiffTOD comparison requires one checkpoint per dataset")
        self.checkpoints = {dataset: Path(path) for dataset, path in checkpoints.items()}
        if set(plan_configs) != {"TopDial", "DuRecDial2.0"} or any(
            set(value) != {"natural", "matched"} for value in plan_configs.values()
        ):
            raise ValueError("DiffTOD comparison requires natural/matched configs per dataset")
        self.plan_config_sha256 = {
            dataset: {
                budget: DiffTODConfig.from_path(Path(path)).sha256
                for budget, path in paths.items()
            }
            for dataset, paths in plan_configs.items()
        }
        self.selection_lock = Path(selection_lock) if selection_lock else None
        self.fixed_test_setting = load_fixed_test_setting(
            self.root, fixed_test_setting, fixed_test_count
        )
        self.fixed_test_manifest = None
        self.fixed_test_cache = None
        self.config = json.loads(self.config_path.read_text(encoding="utf-8"))
        validate_config(self.config)
        self.runtime = json.loads((self.root / "configs/runtime.json").read_text(encoding="utf-8"))

    def _scenarios(self, dataset, split):
        if split == "dev":
            rows = _processed(self.root, dataset, "dev")
            count = self.config["development_scenarios"]
            return sorted(rows, key=lambda row: digest([dataset, row["id"]]))[:count]
        if split != "test" or self.selection_lock is None:
            raise ValueError("official test requires the DiffTOD selection lock")
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

    def run(self, split):
        if split not in {"dev", "test"}:
            raise ValueError("DiffTOD comparison split must be dev or test")
        checkpoint_sha256 = {
            dataset: file_hash(path) for dataset, path in self.checkpoints.items()
        }
        selection_sha256 = "development"
        if split == "test":
            envelope = verify_selection(self.selection_lock)
            selection_sha256 = envelope["sha256"]
            payload = envelope["payload"]
            if (
                payload["config_sha256"] != file_hash(self.config_path)
                or payload["checkpoint_sha256"] != checkpoint_sha256
                or payload["plan_config_sha256"] != self.plan_config_sha256
            ):
                raise ValueError("DiffTOD test runtime differs from frozen selection")
        seeds, budgets = evaluation_axes(self.fixed_test_setting, self.config, split)
        split_output = evaluation_output_directory(
            self.output, split, self.fixed_test_setting
        )
        rows = []
        started = time.monotonic()
        with Servers(self.root, split_output / "servers", self.runtime) as servers:
            backends = servers.backends()
            for dataset in self.config["datasets"]:
                scenarios = self._scenarios(dataset, split)
                for protocol in self.config["protocols"]:
                    for seed in seeds:
                        for scenario in scenarios:
                            for budget in budgets:
                                artifact = self.plan_store.load(
                                    dataset, split, scenario["id"], seed, budget
                                )
                                if artifact.checkpoint_sha256 != checkpoint_sha256[dataset]:
                                    raise ValueError("DiffTOD plan/checkpoint mismatch")
                                expected_config = self.plan_config_sha256[dataset][budget]
                                if artifact.config_sha256 != expected_config:
                                    raise ValueError("DiffTOD plan/configuration mismatch")
                                planner = DiffTODPlanner(
                                    backends["generation"],
                                    self.plan_store,
                                    dataset,
                                    split,
                                    scenario["id"],
                                    seed,
                                    budget,
                                    visible_context(scenario),
                                    Decoding(**self.config["assistant_decoding"]),
                                )
                                contract = {
                                    "phase": split,
                                    "method": "DiffTOD",
                                    "selection_sha256": selection_sha256,
                                    "budget": budget,
                                    "request_namespace": digest(
                                        {
                                            "config": file_hash(self.config_path),
                                            "selection": selection_sha256,
                                            "checkpoint": checkpoint_sha256[dataset],
                                            "plan_config": expected_config,
                                            "budget": budget,
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
                                dialogue_output = (
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
                                    backends["generation"],
                                    backends["embedding"],
                                    protocol,
                                    seed,
                                    dialogue_output,
                                    contract,
                                    self.config["turn_budget"],
                                    Decoding(**self.config["simulator_decoding"]),
                                )
                                protocol_count = len(self.config["protocols"])
                                budget_count = len(budgets)
                                plan_share = artifact.wall_s / protocol_count
                                guidance_share = artifact.guidance_wall_s / (
                                    protocol_count
                                    if self.fixed_test_setting is not None
                                    else protocol_count * budget_count
                                )
                                interactive_h100 = result["h100_seconds"] - result.get(
                                    "offline_plan_h100_s", 0.0
                                ) - result.get("offline_guidance_h100_s", 0.0)
                                if interactive_h100 < 0:
                                    raise ValueError("invalid resumed DiffTOD compute accounting")
                                result["offline_plan_h100_s"] = plan_share
                                result["offline_guidance_h100_s"] = guidance_share
                                result["h100_seconds"] = (
                                    interactive_h100 + plan_share + guidance_share
                                )
                                atomic_json(dialogue_output / "dialogue.json", result)
                                rows.append(
                                    {
                                        "dataset": dataset,
                                        "protocol": protocol,
                                        "budget": budget,
                                        **result,
                                    }
                                )
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
                    summary.append(
                        {
                            "dataset": dataset,
                            "protocol": protocol,
                            "method": "DiffTOD",
                            "budget": budget,
                            **aggregate(group),
                        }
                    )
        report = {
            "schema_version": 1,
            "status": "complete",
            "split": split,
            "method": "DiffTOD",
            "selection_sha256": selection_sha256,
            "config_sha256": file_hash(self.config_path),
            "checkpoint_sha256": checkpoint_sha256,
            "plan_config_sha256": self.plan_config_sha256,
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
        report_path = split_output / "difftod-report.json"
        if split == "test" and self.fixed_test_setting is not None:
            report_path = split_output / (
                f"difftod-report-fixed1000-prefix-{self.fixed_test_setting['count']}.json"
            )
        atomic_json(report_path, report)
        return report
