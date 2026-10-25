"""Audited execution amendment: share Qwen servers for PPDPP and TDRL Full."""

import ast
import gc
import json
import os
import shutil
import subprocess
import time
from datetime import datetime
from types import SimpleNamespace

import numpy as np
import torch

from .comparison import Comparison, prompts
from .concurrency import run_method_pair
from .evaluation import initial_state
from .fidelity import budget_preflight, replay_policy
from .planner import INSTRUCTION
from .policy_training import load_policy
from .provenance import atomic_json, digest, file_hash, validate_manifest
from .runtime import Servers
from .stages import code_provenance, counters
from .training import load_embeddings, load_head, load_prepared, make_graph


def verify_execution_amendment(root, source, cfg):
    old_cfg = json.loads((root / "configs/comparison-judge-v2-sample2000-seed17.json").read_text())
    if {**cfg, "test_sampling": old_cfg["test_sampling"]} != old_cfg:
        raise ValueError("parallel execution must preserve the complete scientific config")
    status = json.loads((source / "status.json").read_text())
    if "stop requested" not in status.get("error", ""):
        raise ValueError("source has not stopped at a durable checkpoint")
    if (source / "selection").exists() or (source / "test").exists():
        raise ValueError("execution amendment requires pre-selection/pre-test source")
    if list(source.rglob("pending-*.json")):
        raise ValueError("source retains unresolved inference outcomes")
    evidence = source / "fidelity-amendment.json"
    if not evidence.exists():
        evidence = source / "parallel-amendment.json"
    old = json.loads(evidence.read_text())["code"]["commit"]
    modules = [
        "policy_training",
        "policy_optimizer",
        "ppdpp",
        "training",
        "reachability",
        "learning",
        "state",
        "judge",
        "backend",
        "runtime",
    ]
    checked = {}
    for module in modules:
        path = "tdrl/src/tdrl/" + module + ".py"
        previous = subprocess.check_output(["git", "show", old + ":" + path], cwd=root)
        current = (root.parent / path).read_bytes()
        if previous != current:
            if module != "judge":
                raise ValueError("fitting/inference module changed: " + module)
            from .critic_execution import execution_evidence, unchanged_judge_protocol

            execution_evidence(root)
            unchanged_judge_protocol(root, old)
        checked[module] = file_hash(root.parent / path)
    path = "tdrl/src/tdrl/evaluation.py"
    previous = ast.parse(subprocess.check_output(["git", "show", old + ":" + path], cwd=root))
    current = ast.parse((root.parent / path).read_text())
    for name in ["initial_state", "visible_context", "policy_input", "PPDPPPlanner"]:
        before = next(n for n in previous.body if getattr(n, "name", None) == name)
        after = next(n for n in current.body if getattr(n, "name", None) == name)
        if ast.dump(before) != ast.dump(after):
            raise ValueError("policy input/forward behavior changed: " + name)
    return {"source_commit": old, "unchanged_modules": checked, "policy_interfaces": "identical"}


class ParallelComparison(Comparison):
    def __init__(self, root, output, config_path, source, fit_dataset=None):
        source = source.resolve()
        super().__init__(root, output, config_path)
        job_id = os.environ.get("SLURM_JOB_ID")
        two_gpu = os.environ.get("TDRL_ALLOCATION_GPU_COUNT") == "2"
        registered = (root / "logs/active-allocation-job-id.txt").read_text().strip()
        if code_provenance(root)["dirty"] or job_id != registered:
            raise ValueError(
                "parallel comparison requires committed source in registered allocation"
            )
        self.source, self.parallel_evaluation = source, True
        audit = verify_execution_amendment(root, source, self.cfg)
        source_allocation = json.loads((source / "allocation.json").read_text())
        if source_allocation["job_id"] != "57711":
            raise ValueError("source allocation mismatch")
        allocation = (
            json.loads((root / f"artifacts/allocation-{job_id}/allocation.json").read_text())
            if two_gpu
            else source_allocation
        )
        amendment = output / "parallel-amendment.json"
        self.imports, self.partial_sft, self.partial_rl = set(), {}, {}
        imported = {}
        for dataset in self.cfg["datasets"]:
            for name in [
                f"embeddings/{dataset}",
                f"learning/{dataset}/MRNFixed",
                f"learning/{dataset}/IQE",
                f"PPDPP-SFT/{dataset}",
                f"PPDPP-RL/{dataset}",
            ]:
                if name.startswith("PPDPP-RL/") and fit_dataset and dataset != fit_dataset:
                    continue
                src = source / name
                if not (src / "manifest.json").exists():
                    continue
                manifest = json.loads((src / "manifest.json").read_text())
                validate_manifest(manifest)
                dst = output / name
                if manifest["status"] == "complete":
                    for path, sha in manifest["outputs"].items():
                        if file_hash(src / path) != sha:
                            raise ValueError("imported fitting artifact checksum mismatch")
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    if not dst.exists():
                        dst.symlink_to(src, target_is_directory=True)
                    if dst.resolve() != src.resolve():
                        raise ValueError("imported stage location changed")
                    self.imports.add(name)
                    imported[name] = {
                        "manifest_sha256": file_hash(src / "manifest.json"),
                        "status": "complete",
                        "original_compute": manifest["compute"],
                    }
                elif name.startswith("PPDPP-SFT/") and (src / "resume.pt").exists():
                    original = json.loads((src / "contract.json").read_text())
                    self.partial_sft[name] = original
                    if not dst.exists():
                        shutil.copytree(
                            src,
                            dst,
                            ignore=shutil.ignore_patterns("manifest.json", "contract.json"),
                        )
                    # Resume may advance the destination; never rewrite it from the source.
                    imported[name] = {
                        "resume_sha256": file_hash(src / "resume.pt"),
                        "original_contract": original,
                        "status": "partial_sft",
                    }
                elif name.startswith("PPDPP-RL/") and (src / "checkpoint.pt").exists():
                    for path, sha in manifest["outputs"].items():
                        if file_hash(src / path) != sha:
                            raise ValueError("partial RL artifact checksum mismatch")
                    original = json.loads((src / "contract.json").read_text())
                    saved = torch.load(
                        src / "checkpoint.pt", map_location="cpu", weights_only=False
                    )
                    if saved["provenance"] != original:
                        raise ValueError("imported RL checkpoint provenance mismatch")
                    self.partial_rl[name] = original
                    if not dst.exists():
                        shutil.copytree(
                            src,
                            dst,
                            ignore=shutil.ignore_patterns("manifest.json", "contract.json"),
                        )
                    imported[name] = {
                        "checkpoint_sha256": file_hash(src / "checkpoint.pt"),
                        "next_episode": saved["next_episode"],
                        "original_contract": original,
                        "status": "partial_rl",
                        "source_manifest_sha256": file_hash(src / "manifest.json"),
                        "source_generation_ledger": {
                            "requests": len(
                                list(
                                    (src / "servers/request-ledger/generation").glob(
                                        "completed-*.json"
                                    )
                                )
                            ),
                            "input_tokens": sum(
                                json.loads(p.read_text())["input_tokens"]
                                for p in (src / "servers/request-ledger/generation").glob(
                                    "completed-*.json"
                                )
                            ),
                            "output_tokens": sum(
                                json.loads(p.read_text())["output_tokens"]
                                for p in (src / "servers/request-ledger/generation").glob(
                                    "completed-*.json"
                                )
                            ),
                            "timing_reference": str(source / "status.json"),
                            "stage_started_at": manifest["started_at"],
                            "exclusive_h100_seconds": (
                                datetime.fromisoformat(
                                    json.loads((source / "status.json").read_text())["ended_at"]
                                ).timestamp()
                                - datetime.fromisoformat(manifest["started_at"]).timestamp()
                            ),
                        },
                    }
                    del saved
        payload = {
            "schema_version": 1,
            "authorization": "user 2026-10-08: PPDPP and TDRL Full concurrently; measured ETA",
            "source": str(source),
            "source_status_sha256": file_hash(source / "status.json"),
            "code": code_provenance(root),
            "config_sha256": file_hash(config_path),
            "test_sampling_authorization": "user 2026-10-08: paired 500 scenarios, fixed seed 42"
            if self.cfg["test_sampling"]["total_scenarios"] == 500
            else "original 2000-scenario scope",
            "allocation": allocation,
            "source_allocation": source_allocation,
            "fit_compatibility": audit,
            "imports": imported,
            "prompts": prompts(),
            "parallel_methods": ["PPDPP", "TDRL"],
            "other_cells": "sequential",
            "accounting": "equal_active_dialogue_share_v1",
            "generation_batching": "GDN batch invariance unavailable; seeded requests and exact durable journal replay, no serial-parallel text equality claim",
        }
        if amendment.exists() and json.loads(amendment.read_text()) != payload:
            from .critic_execution import execution_evidence

            previous = json.loads(amendment.read_text())
            execution_evidence(root, previous["code"]["commit"])
            if {**previous, "code": payload["code"], "fit_compatibility": payload["fit_compatibility"]} != payload:
                raise ValueError("parallel continuation provenance mismatch")
            # Preserve the original immutable fitting-import contract and its hash.
        else:
            atomic_json(amendment, payload)
        atomic_json(output / "allocation.json", allocation)
        shutil.copy2(
            root / f"artifacts/allocation-{job_id}/slurm-job.txt"
            if two_gpu
            else source / "slurm-job.txt",
            output / "slurm-job.txt",
        )
        marker = output / "stop.requested"
        if not marker.is_symlink() and not marker.exists():
            marker.symlink_to(
                root / f"artifacts/allocation-{job_id}/stop.requested"
                if two_gpu
                else root / "artifacts/recovery-57711/stop.requested"
            )
        self.recovery_evidence = {
            "path": str(amendment),
            "sha256": file_hash(amendment),
            "source_job_id": "57711",
        }
        self.extra["parallel_amendment"] = self.recovery_evidence
        self.diagnostic_done = False

    def stage(self, name, fn, extra=None):
        if name.startswith(("embeddings/", "learning/", "PPDPP-")):
            super().stage("budget-preflight", lambda st: budget_preflight(self, st))
            super().stage("concurrency-smoke", lambda st: concurrency_smoke(self, st))
        if name in self.imports:
            return SimpleNamespace(directory=self.output / name, reused=True)
        if name == "selection" and not self.diagnostic_done:
            self.diagnostic_done = True
            from .recovery import SelectionContinuation

            super().stage(
                "PPDPP-development", lambda st: SelectionContinuation.ppdpp_development(self, st)
            )
        return super().stage(name, fn, extra)

    def sft(self, stage, dataset):
        name = "PPDPP-SFT/" + dataset
        if name in self.partial_sft:
            # Continue exact old SFT contract/RNG/optimizer; new manifest records the amendment.
            proxy = SimpleNamespace(directory=stage.directory, contract=self.partial_sft[name])
            return super().sft(proxy, dataset)
        return super().sft(stage, dataset)

    def rl(self, stage, dataset):
        name = "PPDPP-RL/" + dataset
        if name not in self.partial_rl:
            return super().rl(stage, dataset)
        self.rl_training_contract = self.partial_rl[name]
        try:
            return super().rl(stage, dataset)
        finally:
            del self.rl_training_contract


def concurrency_smoke(comparison, stage):
    """Development-only throughput diagnostic; official test bodies remain sealed."""
    dataset = "TopDial"
    records, vectors = load_embeddings(comparison.output / "embeddings" / dataset / "bank")
    head = load_head(comparison.output / "learning" / dataset / "MRNFixed/checkpoint.pt")
    partial = comparison.output / "PPDPP-SFT" / dataset / "resume.pt"
    policy, tokenizer, _ = load_policy(comparison.root, partial if partial.exists() else None)
    replay = replay_policy(policy, tokenizer, load_prepared(comparison.root, dataset, "dev")[0])
    policy.eval()
    cases = sorted(
        load_prepared(comparison.root, dataset, "dev"), key=lambda r: digest([17, r["id"]])
    )[:2]
    measurements = []
    with Servers(comparison.root, stage.directory / "servers", comparison.settings) as servers:
        b = servers.backends()
        text = initial_state(cases[0]).serialize()
        first = b["embedding"].embed(
            [text], INSTRUCTION, "concurrency/replay-1", "concurrency/replay-1"
        )
        again = b["embedding"].embed(
            [text], INSTRUCTION, "concurrency/replay-2", "concurrency/replay-2"
        )
        if not torch.equal(first.vectors, again.vectors):
            raise ValueError("concurrency smoke live embedding replay mismatch")
        for index, scenario in enumerate(cases):
            graph, goals = make_graph(
                records,
                vectors,
                head,
                1000.0,
                comparison.cfg["node_budget"],
                17,
                scenario["target"][0],
            )
            if not goals:
                raise ValueError("concurrency smoke requires training-backed goals")
            cover = 0.0 if index == 0 else 1000000.0

            def evaluate(
                method,
                mode,
                matched=False,
                index=index,
                scenario=scenario,
                graph=graph,
                goals=goals,
                cover=cover,
                policy=policy,
            ):
                path = (
                    stage.directory
                    / mode
                    / str(index)
                    / method
                    / ("matched" if matched else "natural")
                )
                result = comparison.evaluate(
                    b,
                    scenario,
                    method,
                    "profile",
                    17,
                    path,
                    {
                        "phase": "concurrency-smoke-" + mode,
                        "budget": "matched" if matched else "natural",
                    },
                    graph,
                    goals,
                    cover,
                    policy,
                    tokenizer,
                    matched,
                )
                return result, path

            start = time.monotonic()
            serial = [evaluate(method, "serial")[0] for method in ["PPDPP", "TDRL"]]
            serial_wall = time.monotonic() - start
            pair_path = stage.directory / "pair-measurement" / str(index)
            parallel = run_method_pair(lambda method: evaluate(method, "parallel"), pair_path)
            before = len(b["generation"].requests), len(b["embedding"].requests)
            replayed = run_method_pair(lambda method: evaluate(method, "parallel"), pair_path)
            if replayed != parallel or before != (
                len(b["generation"].requests),
                len(b["embedding"].requests),
            ):
                raise ValueError(
                    "concurrent completed-pair replay reissued requests or changed records"
                )
            gen_ids, emb_ids = [], []
            for result in parallel:
                gen_ids.append(
                    {r["request_id"] for t in result["turns_trace"] for r in t["generation"]}
                )
                emb_ids.append(
                    {r["request_id"] for t in result["turns_trace"] for r in t["embedding"]}
                )
                if result["generation_requests"] != len(gen_ids[-1]) or result[
                    "embedding_requests"
                ] != len(emb_ids[-1]):
                    raise ValueError("concurrent compute counter mismatch")
            if gen_ids[0] & gen_ids[1] or emb_ids[0] & emb_ids[1]:
                raise ValueError("concurrent worker request contamination")
            for turn in parallel[1]["turns_trace"]:
                own_planner_calls = [
                    r
                    for r in turn["generation"]
                    if r["role"] not in {"simulator", "judge", "critic"}
                ]
                if turn["plan"]["compute"]["generation_requests"] != len(own_planner_calls):
                    raise ValueError("nested planner request contamination")
                if turn["plan"]["compute"]["embedding_requests"] != len(turn["embedding"]):
                    raise ValueError("nested planner embedding contamination")
            other = [evaluate("TDRL-R", "serial")[0], evaluate("TDRL", "serial", True)[0]]
            timing = json.loads((pair_path / "pair-timing.json").read_text())
            measurements.append(
                {
                    "scenario_id": scenario["id"],
                    "coverage_probe": "off-map" if index == 0 else "covered",
                    "serial_pair_wall_s": serial_wall,
                    "parallel_pair_wall_s": timing["wall_s"],
                    "pair_speedup": serial_wall / timing["wall_s"],
                    "other_cells_wall_s": sum(r["wall_s"] for r in other),
                    "serial_pair_turns": [r["actual_turns"] for r in serial],
                    "parallel_pair_turns": [r["actual_turns"] for r in parallel],
                }
            )
            atomic_json(stage.directory / "measurements.json", measurements)
        compute = {**counters(b["generation"], b["embedding"]), **servers.compute_metadata()}
    base = json.loads((comparison.output / "budget-preflight/projection.json").read_text())
    # Direct scenario timings include Full's actual lookahead, unlike the PPDPP-only proxy.
    per_scenario = float(
        np.quantile(
            [r["parallel_pair_wall_s"] + r["other_cells_wall_s"] for r in measurements], 0.95
        )
    )
    scaling = {
        d: values["warm_p95_seconds"] / base["development_timing"][dataset]["warm_p95_seconds"]
        for d, values in base["development_timing"].items()
    }
    test_seconds = (
        per_scenario
        * sum(base["evaluated_count_projection"][d] * scaling[d] for d in scaling)
        * len(comparison.cfg["protocols"])
        * len(comparison.cfg["seeds"])
    )
    required = (test_seconds + base["projected_native_rl_seconds"]) * comparison.cfg[
        "feasibility_safety_factor"
    ] + comparison.cfg["reserve_seconds"]
    remaining = comparison.cfg["walltime_seconds"] - (time.time() - comparison.allocation_start)
    atomic_json(
        stage.directory / "checks.json",
        {
            "status": "passed",
            "scientific_result": False,
            "measurements": measurements,
            "controller_replay": replay,
            "isolated_counters": True,
            "exact_completed_pair_replay": True,
            "projected_test_seconds": test_seconds,
            "dataset_scaling_from_ppdpp_proxy": scaling,
            "required_seconds_with_safety_and_reserve": required,
            "remaining_allocation_seconds": remaining,
            "fits_proxy": required <= remaining,
            "estimate_limit": "two TopDial development probes; partial SFT, MRNFixed, provisional extreme thresholds; no DuRecDial/resistant timing or selected-head claims",
        },
    )
    del policy, head, vectors
    gc.collect()
    torch.cuda.empty_cache()
    if required > remaining:
        raise ValueError(
            "concurrent method-specific development proxy exceeds allocation; explicit scope/budget decision required"
        )
    return compute
