"""Independent one-GPU tasks inside the user's two-H100 allocation."""

import gc
import json
import os

import torch

from . import comparison as comparison_module
from .backend import require_h100_slurm
from .branch_execution import selection_task_name
from .comparison import Comparison
from .fidelity import budget_preflight, judge_smoke
from .parallel_comparison import ParallelComparison, concurrency_smoke
from .provenance import atomic_json, file_hash
from .runtime import smoke
from .stages import combine_compute


def merged_selection_inputs(shards):
    candidates, rows, thresholds, ids = {}, {}, {}, {}
    for shard in shards:
        if shard["schema_version"] != 1:
            raise ValueError("unknown selection shard")
        for head, metrics in shard["candidates"].items():
            if head not in candidates:
                candidates[head] = dict(metrics)
                candidates[head]["development_sr"] = {}
            elif any(
                candidates[head][key] != metrics[key]
                for key in [
                    "parameters",
                    "invariants_passed",
                    "stability_passed",
                    "provenance_passed",
                ]
            ):
                raise ValueError("selection shard architecture/invariant mismatch")
            overlap = set(candidates[head]["development_sr"]) & set(metrics["development_sr"])
            if overlap:
                raise ValueError("duplicate selection dataset")
            candidates[head]["development_sr"].update(metrics["development_sr"])
            rows.setdefault(head, {}).update(shard["rows"][head])
            thresholds.setdefault(head, {}).update(shard["thresholds"][head])
        ids.update(shard["scenario_ids"])
    if set(ids) != {"TopDial", "DuRecDial2.0"}:
        raise ValueError("both selection datasets required")
    return candidates, rows, thresholds, ids


class Task(ParallelComparison):
    def __init__(self, root, output, cfg, source, shared, slot, dataset=None):
        super().__init__(root, output, cfg, source, fit_dataset=dataset or "none")
        self.shared, self.slot = shared, slot
        self.settings = json.loads(json.dumps(self.settings))
        for kind in ["generation", "embedding"]:
            self.settings[kind]["port"] += slot * 100
        self.parallel_evaluation = False
        self.extra["two_gpu_execution"] = {"slot": slot, "cpu_count": 8, "port_offset": slot * 100}
        self.allocation_start = json.loads((output / "allocation.json").read_text())["started_unix"]
        gpu = require_h100_slurm()
        atomic_json(
            output / "task-gpu.json",
            {**gpu, "slot": slot, "slurm_job_id": os.environ["SLURM_JOB_ID"]},
        )
        for name in ["embeddings", "learning", "PPDPP-SFT", "PPDPP-RL"]:
            for ds in self.cfg["datasets"]:
                path, target = output / name / ds, shared / name / ds
                if not path.exists() and target.exists():
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.symlink_to(target.resolve(), target_is_directory=True)

    def stage(self, name, fn, extra=None):
        if name in self.imports:
            from types import SimpleNamespace

            return SimpleNamespace(directory=self.output / name, reused=True)
        return Comparison.stage(self, name, fn, extra)


def run_task(root, output, config, source, shared, slot, mode, dataset=None, role=None):
    task = Task(root, output, config, source, shared, slot, dataset if mode == "policy" else "none")
    try:
        if mode == "bootstrap":
            task.stage("gate0", task.repository)
            task.stage("gate1", task.data)
            smoke(root, output / "gate2", judge_config=task.cfg["judge"])
            task.stage("judge-smoke", lambda st: judge_smoke(task, st))
            task.stage("budget-preflight", lambda st: budget_preflight(task, st))
            task.stage("concurrency-smoke", lambda st: concurrency_smoke(task, st))
        else:
            bootstraps = [shared / "tasks" / f"bootstrap-{i}" for i in range(2)]
            validated = []
            for bootstrap in bootstraps:
                if json.loads((bootstrap / "task-status.json").read_text())["status"] != "complete":
                    raise ValueError("both GPU bootstrap gates required")
                validated.append(json.loads((bootstrap / "task-gpu.json").read_text())["uuid"])
            if len(set(validated)) != 2 or require_h100_slurm()["uuid"] not in validated:
                raise ValueError("worker GPU was not independently validated")
            if mode == "offline":
                task.stage("embeddings/" + dataset, lambda st: task.embeddings(st, dataset))
                for head in ["MRNFixed", "IQE"]:
                    task.stage(
                        f"learning/{dataset}/{head}",
                        lambda st, h=head: task.learning(st, dataset, h),
                    )
            elif mode == "policy":
                task.stage("PPDPP-SFT/" + dataset, lambda st: task.sft(st, dataset))
                task.stage("PPDPP-RL/" + dataset, lambda st: task.rl(st, dataset))
            elif mode == "select-shard":
                comparison_module.DATASETS = (dataset,)
                task.selection_shard = True
                task.stage("selection-shard", task.select)
            elif mode == "policy-dev":
                from . import recovery

                recovery.DATASETS = (dataset,)
                task.stage(
                    "PPDPP-development",
                    lambda st: recovery.SelectionContinuation.ppdpp_development(task, st),
                )
            elif mode == "lock":
                shards = [
                    json.loads(
                        (
                            shared
                            / "tasks"
                            / selection_task_name(ds)
                            / "selection-shard/selection-shard.json"
                        ).read_text()
                    )
                    for ds in task.cfg["datasets"]
                ]
                task.selection_inputs = merged_selection_inputs(shards)
                task.output = shared
                selection_dir = shared / "selection"
                selection_dir.mkdir(parents=True, exist_ok=True)
                for ds in task.cfg["datasets"]:
                    shard_dir = shared / "tasks" / selection_task_name(ds) / "selection-shard"
                    for path in shard_dir.glob("*.json"):
                        if path.name.startswith(("compute-", "trials-")):
                            if not (selection_dir / path.name).exists():
                                (selection_dir / path.name).symlink_to(path.resolve())
                task.recovery_evidence["both_gpu_smoke"] = {
                    "path": str(shared / "both-gpu-smoke.json"),
                    "sha256": file_hash(shared / "both-gpu-smoke.json"),
                }
                task.stage("selection", task.select)
            elif mode == "test":
                task.evaluation_methods = (
                    [("PPDPP", False), ("TDRL-R", False), ("TDRL", True)]
                    if role == "policy"
                    else [("TDRL", False)]
                )
                # Each worker verifies the identical global lock before sampling.
                if not (output / "selection").exists():
                    (output / "selection").symlink_to(
                        shared / "selection", target_is_directory=True
                    )
                task.stage("test", task.test)
            elif mode == "report":
                left, right = [
                    shared / "tasks" / ("test-" + r) / "test" for r in ["policy", "full"]
                ]
                a, b = [json.loads((p / "scenario-order.json").read_text()) for p in [left, right]]
                if a != b:
                    raise ValueError("GPU method shards did not use identical test IDs")
                rows = [
                    r for p in [left, right] for r in json.loads((p / "records.json").read_text())
                ]
                merged = shared / "test"
                merged.mkdir(parents=True, exist_ok=True)
                for name in ["scenario-order.json", "test-sample.json"]:
                    atomic_json(merged / name, json.loads((left / name).read_text()))
                atomic_json(merged / "records.json", rows)
                task.output = shared
                task.two_gpu_execution = True

                def merge_test(stage):
                    return combine_compute(
                        [
                            json.loads((p / "manifest.json").read_text())["compute"]
                            for p in [left, right]
                        ]
                    )

                task.stage("test", merge_test)
                task.stage("report", task.report)
            else:
                raise ValueError("unknown two-GPU task")
        atomic_json(
            output / "task-status.json", {"schema_version": 1, "status": "complete", "mode": mode}
        )
    except Exception as error:
        if task.active_stage:
            task.active_stage.write("partial", error=str(error))
        atomic_json(
            output / "task-status.json",
            {"schema_version": 1, "status": "partial", "mode": mode, "error": str(error)},
        )
        raise
    finally:
        gc.collect()
        torch.cuda.empty_cache()
