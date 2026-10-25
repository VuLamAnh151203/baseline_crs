"""Gated three-method automated comparison, sequential inside one allocation.

This scope does not certify the full four-baseline/human/paper project. Any failed gate
leaves partial artifacts and stops, including infeasibility and ambiguous selection.
"""

import gc
import json
import os
import time
from datetime import datetime, timezone

import numpy as np
import torch

from .audit import dependency_audit, static_audit
from .baseline_test_setting import (
    evaluation_axes,
    load_fixed_test_setting,
    select_fixed_test_populations,
)
from .contracts import Decoding
from .evaluation import (
    RESISTANT_POLICY,
    USER_POLICY,
    PPDPPPlanner,
    initial_state,
    run_dialogue,
    visible_context,
)
from .judge import JUDGE_PROMPT, LocalJudge
from .metrics import aggregate, paired_bootstrap, validate_records
from .near_duplicates import audit_train_dev
from .official_data import SPLITS, prepare_releases, read_release
from .planner import ASSISTANT_POLICY, INSTRUCTION, ConditionalPlanner, PlannerConfig
from .policy_training import load_policy, policy_snapshot, reinforce_policy, supervised_policy
from .provenance import atomic_json, digest, file_hash
from .runtime import Servers, smoke
from .selection import choose_head, verify_selection, write_selection
from .stages import Stage, code_provenance, combine_compute, counters
from .state import SERIALIZATION_HASH
from .test_sampling import sample_test_scenarios, validate_sampling
from .training import (
    embed_records,
    heldout_temporal_diagnostics,
    load_embeddings,
    load_head,
    load_prepared,
    make_graph,
    state_records,
    train_head,
)

DATASETS = ("TopDial", "DuRecDial2.0")
METHODS = ("PPDPP", "TDRL-R", "TDRL")


def validate_config(cfg):
    validate_sampling(cfg.get("test_sampling"))
    if cfg["schema_version"] not in {1, 2} or cfg["scope"] != {
        "methods": list(METHODS),
        "claim": "three_method_automated_comparison",
    }:
        raise ValueError("unknown comparison config/scope")
    if (
        tuple(cfg["datasets"]) != DATASETS
        or cfg["seeds"] not in ([17, 29, 43], [17])
        or cfg["protocols"] != ["profile", "resistant"]
    ):
        raise ValueError("both datasets, authorized seeds and both protocols are required")
    if cfg["seeds"] == [17] and (
        cfg.get("test_sampling") is None or cfg["head"]["seed"] != 17 or cfg["ppdpp"]["seed"] != 17
    ):
        raise ValueError("single-seed scope requires sampled test and training seed 17")
    if not 0 < cfg["walltime_seconds"] <= 20 * 86400 or cfg["turn_budget"] < 1:
        raise ValueError("invalid allocation/turn budget")
    if cfg["matched_generation_call_cap"] != 1:
        raise ValueError("this matched configuration requires one planner call per turn")
    if cfg["schema_version"] == 2 and (
        cfg.get("offline_terminal_protocol") != "censored_v2"
        or cfg["ppdpp"].get("implementation_version") != 2
        or cfg["ppdpp"]["rl_episodes"] != 1000
        or cfg["ppdpp"]["rl_lr"] != 1e-6
        or cfg["ppdpp"]["gamma"] != 0.999
        or cfg.get("judge", {}).get("critic_samples") != 10
    ):
        raise ValueError("invalid PPDPP fidelity/judge protocol")


def prompts():
    return {
        "assistant": digest(ASSISTANT_POLICY),
        "simulator": digest(USER_POLICY),
        "resistant": digest(RESISTANT_POLICY),
        "embedding": digest(INSTRUCTION),
        "scenario_context_version": 1,
        "judge": digest(JUDGE_PROMPT),
    }


def evaluation_contract(contract, method, goals, config_sha256, matched):
    payload = {**contract, "method": method, "target_nodes": goals or []}
    payload["request_namespace"] = digest(
        {
            "schema_version": 1,
            "dialogue_contract": payload,
            "config_sha256": config_sha256,
            "matched": matched,
        }
    )
    return payload


class Comparison:
    def __init__(
        self, root, output, config_path, fixed_test_setting=None, fixed_test_count=None
    ):
        self.root, self.output, self.config_path = root, output, config_path
        self.cfg = json.loads(config_path.read_text())
        validate_config(self.cfg)
        self.fixed_test_setting = load_fixed_test_setting(
            root, fixed_test_setting, fixed_test_count
        )
        self.test_eval_seeds, self.test_eval_budgets = evaluation_axes(
            self.fixed_test_setting, self.cfg, "test"
        )
        prefix_tag = (
            f"-fixed1000-prefix-{self.fixed_test_setting['count']}"
            if self.fixed_test_setting
            else ""
        )
        self.test_stage_name = "test-fixed1000" if self.fixed_test_setting else "test"
        self.test_stage_directory = self.output / self.test_stage_name
        self.test_sample_path = self.test_stage_directory / f"test-sample{prefix_tag}.json"
        self.test_scenario_order_path = (
            self.test_stage_directory / f"scenario-order{prefix_tag}.json"
        )
        self.started = time.monotonic()
        self.output.mkdir(parents=True, exist_ok=True)
        self.models = json.loads((root / "configs/models.lock.json").read_text())
        self.settings = json.loads((root / "configs/runtime.json").read_text())
        self.extra = {"serialization_sha256": SERIALIZATION_HASH, "models": self.models}
        self.assistant = Decoding(**self.cfg["assistant_decoding"])
        self.simulator = Decoding(**self.cfg["simulator_decoding"])
        self.active_stage = None
        self.allocation_start = None

    def stage(self, name, fn, extra=None):
        if (self.output / "stop.requested").exists():
            raise InterruptedError("allocation stop requested; checkpoint artifacts retained")
        archive_hashes = {d: file_hash(self.root / "data/raw" / d / "data.zip") for d in DATASETS}
        stage = Stage(
            self.root,
            self.output / name,
            name,
            self.config_path,
            {**self.extra, "dataset_archives": archive_hashes, **(extra or {})},
        )
        self.active_stage = stage
        if not stage.reused:
            stage.write("partial")
            if name.startswith(("learning/", "PPDPP-SFT/")):
                torch.cuda.reset_peak_memory_stats()
            compute = fn(stage) or {}
            compute.setdefault(
                "h100_seconds", time.time() - datetime.fromisoformat(stage.started).timestamp()
            )
            stage.write("complete", compute_updates=compute)
        self.active_stage = None
        return stage

    def run(self):
        try:
            allocation_path = self.output / "allocation.json"
            job_id = os.environ.get("SLURM_JOB_ID")
            if not job_id:
                raise RuntimeError("comparison requires one Slurm H100 allocation")
            if allocation_path.exists():
                allocation = json.loads(allocation_path.read_text())
                if allocation["job_id"] != job_id:
                    raise ValueError("this pipeline must stay inside its original allocation")
            else:
                allocation = {"job_id": job_id, "started_unix": time.time()}
                atomic_json(allocation_path, allocation)
            self.allocation_start = allocation["started_unix"]
            self.stage("gate0", self.repository)
            self.stage("gate1", self.data)
            smoke(self.root, self.output / "gate2", judge_config=self.cfg.get("judge"))
            # A successful runtime gate is checked before any fitting or scientific evaluation.
            gate2 = json.loads((self.output / "gate2/manifest.json").read_text())
            if gate2["status"] != "complete":
                raise ValueError("Gate 2 not complete")
            if self.cfg["schema_version"] == 2:
                from .fidelity import judge_smoke

                self.stage("judge-smoke", lambda st: judge_smoke(self, st))
            for dataset in DATASETS:
                self.stage("embeddings/" + dataset, lambda st, d=dataset: self.embeddings(st, d))
                for head in ("MRNFixed", "IQE"):
                    self.stage(
                        f"learning/{dataset}/{head}",
                        lambda st, d=dataset, h=head: self.learning(st, d, h),
                    )
                self.stage("PPDPP-SFT/" + dataset, lambda st, d=dataset: self.sft(st, d))
                self.stage("PPDPP-RL/" + dataset, lambda st, d=dataset: self.rl(st, d))
            self.stage("selection", self.select)
            self.stage(self.test_stage_name, self.test)
            self.stage("report", self.report)
            atomic_json(
                self.output / "status.json",
                {
                    "schema_version": 1,
                    "status": "complete",
                    "scope": self.cfg["scope"],
                    "human_paper_full_baseline_gates": "not_completed_by_this_scope",
                    "ended_at": datetime.now(timezone.utc).isoformat(),
                },
            )
        except BaseException as error:
            if self.active_stage:
                self.active_stage.write("failed", error=error)
            atomic_json(
                self.output / "status.json",
                {
                    "schema_version": 1,
                    "status": "partial",
                    "error": str(error),
                    "scope": self.cfg["scope"],
                    "ended_at": datetime.now(timezone.utc).isoformat(),
                },
            )
            raise

    def repository(self, stage):
        errors = static_audit(self.root)
        audit = dependency_audit(self.root / "runtime/uv.lock", allow_local_vllm=True)
        if errors or audit["status"] != "passed":
            raise ValueError(f"repository gate failed: {errors}, {audit}")
        policy_snapshot(self.root)
        atomic_json(
            stage.directory / "checks.json",
            {
                "status": "passed",
                "dependency": audit,
                "scope": self.cfg["scope"],
                "code": code_provenance(self.root),
            },
        )
        return {"h100_seconds": 0.0}

    def data(self, stage):
        manifests = prepare_releases(self.root)
        atomic_json(stage.directory / "data.json", manifests)
        return {"h100_seconds": 0.0}

    def embeddings(self, stage, dataset):
        with Servers(
            self.root, stage.directory / "servers", self.settings, kinds=("embedding",)
        ) as servers:
            backend = servers.backends()["embedding"]
            for split in ("train", "dev"):
                scenarios = load_prepared(self.root, dataset, split)
                records = state_records(
                    scenarios,
                    dataset,
                    split,
                    self.cfg.get("offline_terminal_protocol", "legacy_v1"),
                )
                contract = {
                    **stage.contract,
                    "input_sha256": file_hash(
                        self.root / "data/processed" / dataset / (split + ".jsonl")
                    ),
                    "split": split,
                }
                embed_records(
                    records,
                    backend,
                    stage.directory / ("bank" if split == "train" else "dev-bank"),
                    contract,
                )
            result = counters(type("Empty", (), {"requests": []})(), backend)
            result.update(servers.compute_metadata())
        return result

    def learning(self, stage, dataset, head):
        records, vectors = load_embeddings(self.output / "embeddings" / dataset / "bank")
        trained = train_head(
            head, records, vectors, self.cfg["head"], stage.directory, stage.contract
        )
        dev_records, dev_vectors = load_embeddings(
            self.output / "embeddings" / dataset / "dev-bank"
        )
        heldout_temporal_diagnostics(trained, dev_records, dev_vectors, stage.directory)
        del trained, dev_vectors
        peak = {
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        }
        del vectors
        gc.collect()
        torch.cuda.empty_cache()
        return peak

    def ppdpp_cfg(self):
        return {
            **self.cfg["ppdpp"],
            "turn_budget": self.cfg["turn_budget"],
            "assistant_decoding": self.cfg["assistant_decoding"],
            "simulator_decoding": self.cfg["simulator_decoding"],
            "judge": self.cfg.get("judge"),
        }

    def sft(self, stage, dataset):
        policy, _, _ = supervised_policy(
            self.root,
            load_prepared(self.root, dataset, "train"),
            self.ppdpp_cfg(),
            stage.directory,
            stage.contract,
            validation_scenarios=load_prepared(self.root, dataset, "dev")
            if self.cfg["schema_version"] == 2
            else None,
        )
        peak = {
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        }
        del policy
        gc.collect()
        torch.cuda.empty_cache()
        return peak

    def rl(self, stage, dataset):
        # Initialize/resume policy first; server startup budgets account for its residency.
        policy, tokenizer, pin = load_policy(
            self.root,
            stage.directory / "checkpoint.pt"
            if (stage.directory / "checkpoint.pt").exists()
            else self.output / "PPDPP-SFT" / dataset / "checkpoint.pt",
        )
        if self.cfg["schema_version"] == 2:
            from .fidelity import policy_replay_gate

            policy_replay_gate(policy, tokenizer, self, dataset, stage)
        with Servers(self.root, stage.directory / "servers", self.settings) as servers:
            b = servers.backends()
            if os.environ.get("TDRL_POLICY_DETERMINISM") == "strict-backward-v1":
                policy._recovery_execution = {
                    "code": stage.code,
                    "job_id": os.environ["SLURM_JOB_ID"],
                    "deterministic_backward": True,
                    "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
                }
            if hasattr(self, "configure_rl_backends"):
                self.configure_rl_backends(b)
            reinforce_policy(
                policy,
                tokenizer,
                pin,
                load_prepared(self.root, dataset, "train"),
                b["generation"],
                b["embedding"],
                self.ppdpp_cfg(),
                stage.directory,
                getattr(self, "rl_training_contract", stage.contract),
                recovery_check=getattr(self, "rl_recovery_check", None),
            )
            result = counters(b["generation"], b["embedding"])
            result.update(servers.compute_metadata())
        atomic_json(
            stage.directory / "fidelity.json",
            {
                "status": "validated",
                "mechanism": "pooled_roberta_dropout_classifier_SFT_then_REINFORCE",
                "sft": str(self.output / "PPDPP-SFT" / dataset / "checkpoint.pt"),
                "rl_sha256": file_hash(stage.directory / "checkpoint.pt"),
                "adaptations": [
                    "six weak recommendation strategy labels",
                    "visible target header",
                    "local sampled graded critic; common deterministic semantic outcome judge"
                    if self.cfg["schema_version"] == 2
                    else "deterministic unpressured acceptance reward replacing native graded critic",
                    "dropout disabled and argmax at evaluation",
                ],
                "claim_limit": "recommendation-task adaptation; not native task equivalence",
                "config_sha256": file_hash(self.config_path),
                "implementation_version": self.cfg["ppdpp"].get("implementation_version", 1),
                "native_code_restored": [
                    "graded critic 10 samples",
                    "uniform case sampling",
                    "HF AdamW update formula with explicit defaults",
                    "SFT warmup/linear decay",
                    "bias/LayerNorm no-decay",
                    "development macro-precision checkpoint",
                    "1000 RL episodes, lr 1e-6, gamma .999",
                    "no RL gradient clipping",
                    "explicit strategy instructions",
                ]
                if self.cfg["schema_version"] == 2
                else [],
                "retained_departures": [
                    "six weak CRS labels, not native human strategy labels",
                    "visible target/profile header and local token serialization",
                    "BF16 policy forward with FP32 optimizer masters",
                    "Qwen backbone and common profile/resistant/autonomy protocol",
                    "one deterministic outcome judgment; ten sampled training critic judgments",
                    "live embedding measured but unused by intrinsic PPDPP policy",
                    "common 10-turn/128-token role decoding; full-budget failed turn metric",
                ],
            },
        )
        del policy
        gc.collect()
        torch.cuda.empty_cache()
        return result

    def evaluate(
        self,
        b,
        scenario,
        method,
        protocol,
        seed,
        directory,
        contract,
        graph=None,
        goals=None,
        h_cover=0.0,
        policy=None,
        tokenizer=None,
        matched=False,
    ):
        if method == "PPDPP":
            planner = PPDPPPlanner(
                policy.eval(),
                tokenizer,
                b["generation"],
                b["embedding"],
                visible_context(scenario),
                self.assistant,
            )
        else:
            planner = ConditionalPlanner(
                b["generation"],
                b["embedding"],
                graph,
                PlannerConfig(
                    h_cover,
                    self.cfg["candidates"],
                    self.cfg["lookahead"],
                    routing="retrieval_only" if method == "TDRL-R" else "conditional",
                    generation_call_cap=1 if matched else None,
                ),
                self.assistant,
                self.simulator,
                visible_context=visible_context(scenario),
            )
        return run_dialogue(
            scenario,
            planner,
            b["generation"],
            b["embedding"],
            protocol,
            seed,
            directory,
            evaluation_contract(contract, method, goals, file_hash(self.config_path), matched),
            self.cfg["turn_budget"],
            self.simulator,
            judge=LocalJudge(b["generation"], self.cfg["judge"])
            if self.cfg["schema_version"] == 2
            else None,
        )

    def select(self, stage):
        candidates, all_rows, best_thresholds, sampled_ids = {}, {}, {}, {}
        selection_inputs = getattr(self, "selection_inputs", None)
        if selection_inputs:
            candidates, all_rows, best_thresholds, sampled_ids = selection_inputs
        policies = {}
        for dataset in () if selection_inputs else DATASETS:
            rows = load_prepared(self.root, dataset, "dev")
            scenarios = sorted(rows, key=lambda s: digest([17, s["id"]]))[
                : self.cfg["development_scenarios"]
            ]
            sampled_ids[dataset] = [s["id"] for s in scenarios]
            records, vectors = load_embeddings(self.output / "embeddings" / dataset / "bank")
            for kind in ("MRNFixed", "IQE"):
                head = load_head(self.output / "learning" / dataset / kind / "checkpoint.pt")
                rng = torch.Generator().manual_seed(17)
                ids = torch.randint(len(vectors), (1024, 2), generator=rng)
                with torch.no_grad():
                    costs = (
                        head.distance(
                            vectors[ids[:, 0]].to("cuda", torch.bfloat16),
                            vectors[ids[:, 1]].to("cuda", torch.bfloat16),
                        )
                        .float()
                        .cpu()
                    )
                edges = [float(torch.quantile(costs, q)) for q in self.cfg["threshold_quantiles"]]
                trials = []
                with Servers(
                    self.root, stage.directory / "servers" / dataset / kind, self.settings
                ) as servers:
                    b = servers.backends()
                    graph_cache = {}
                    # Cover thresholds are measured on development-to-training distances, independent of edges.
                    initial = b["embedding"].embed(
                        [initial_state(s).serialize() for s in scenarios],
                        INSTRUCTION,
                        "dev-calibration",
                        f"dev/{dataset}/{kind}/cover",
                    )
                    covers = []
                    for edge in edges:
                        minimum = []
                        for i, scenario in enumerate(scenarios):
                            target_type = scenario["target"][0]
                            key = (edge, target_type)
                            if key not in graph_cache:
                                graph_cache[key] = make_graph(
                                    records,
                                    vectors,
                                    head,
                                    edge,
                                    self.cfg["node_budget"],
                                    17,
                                    target_type,
                                )
                            graph, goals = graph_cache[key]
                            retrieved = graph.retrieve(initial.vectors[i], goals) if graph else []
                            if retrieved:
                                minimum.append(retrieved[0][0])
                        if not minimum:
                            raise ValueError(
                                "development has no training-backed guidance; cannot calibrate coverage"
                            )
                        covers = [
                            float(np.quantile(minimum, q)) for q in self.cfg["threshold_quantiles"]
                        ]
                        for cover in covers:
                            rows = []
                            trial_id = digest([dataset, kind, edge, cover])[:16]
                            for scenario in scenarios:
                                graph, goals = graph_cache[(edge, scenario["target"][0])]
                                result = self.evaluate(
                                    b,
                                    scenario,
                                    "TDRL",
                                    "profile",
                                    17,
                                    stage.directory
                                    / "trials"
                                    / dataset
                                    / kind
                                    / trial_id
                                    / scenario["id"],
                                    {
                                        "phase": "development",
                                        "thresholds": [edge, cover],
                                        "head_sha256": head.checkpoint_hash,
                                    },
                                    graph,
                                    goals,
                                    cover,
                                )
                                rows.append(result)
                            trials.append(
                                {
                                    "H_edge": edge,
                                    "H_cover": cover,
                                    "rows": rows,
                                    "metrics": aggregate(rows),
                                    "trial_id": trial_id,
                                }
                            )
                    phase_counts = counters(b["generation"], b["embedding"])
                    phase_counts.update(servers.compute_metadata())
                    atomic_json(stage.directory / f"compute-{dataset}-{kind}.json", phase_counts)
                best = max(
                    trials,
                    key=lambda t: (
                        t["metrics"]["sr"],
                        -t["metrics"]["h100_seconds_per_dialogue"],
                        -t["H_edge"],
                        -t["H_cover"],
                    ),
                )
                best_thresholds.setdefault(kind, {})[dataset] = {
                    k: best[k] for k in ("H_edge", "H_cover")
                }
                all_rows.setdefault(kind, {})[dataset] = best["rows"]
                candidates.setdefault(
                    kind,
                    {
                        "development_sr": {},
                        "invariants_passed": True,
                        "stability_passed": True,
                        "provenance_passed": True,
                        "parameters": sum(p.numel() for p in head.parameters()),
                    },
                )["development_sr"][dataset] = best["metrics"]["sr"]
                atomic_json(stage.directory / f"trials-{dataset}-{kind}.json", trials)
                del head, graph_cache
                gc.collect()
                torch.cuda.empty_cache()
            del vectors
        if getattr(self, "selection_shard", False):
            atomic_json(
                stage.directory / "selection-shard.json",
                {
                    "schema_version": 1,
                    "candidates": candidates,
                    "rows": all_rows,
                    "thresholds": best_thresholds,
                    "scenario_ids": sampled_ids,
                },
            )
            return combine_compute(
                [json.loads(p.read_text()) for p in stage.directory.glob("compute-*.json")]
            )
        # Compute tie-breaker uses the same best-development scenario sets for each head.
        for kind in candidates:
            rows = [r for d in DATASETS for r in all_rows[kind][d]]
            value = aggregate(rows)["h100_seconds_per_success"]
            candidates[kind]["h100_seconds_per_success"] = value
            candidates[kind]["development_successes"] = {
                d: sum(r["accepted"] for r in all_rows[kind][d]) for d in DATASETS
            }
        left = [float(r["accepted"]) for d in DATASETS for r in all_rows["MRNFixed"][d]]
        right = [float(r["accepted"]) for d in DATASETS for r in all_rows["IQE"][d]]
        ci = paired_bootstrap(left, right)["ci95"]
        atomic_json(
            stage.directory / "decision-inputs.json",
            {"candidates": candidates, "difference_ci": ci},
        )
        chosen = choose_head(candidates, ci)
        for d in DATASETS:
            policies[d] = file_hash(self.output / "PPDPP-RL" / d / "fidelity.json")
        gate_hashes = {
            "0": file_hash(self.output / "gate0/manifest.json"),
            "1": file_hash(self.output / "gate1/manifest.json"),
            "2": file_hash(self.output / "gate2/manifest.json"),
            "3": {
                d: {
                    h: file_hash(self.output / "learning" / d / h / "manifest.json")
                    for h in candidates
                }
                for d in DATASETS
            },
        }
        payload = {
            "schema_version": 2,
            "scope": self.cfg["scope"],
            "selected_head": chosen,
            "candidates": candidates,
            "difference_ci": ci,
            "thresholds": best_thresholds[chosen],
            "prompts": prompts(),
            "models": self.models,
            "budgets": {
                "config": self.cfg,
                "PPDPP_checkpoint_sha256": {
                    d: file_hash(self.output / "PPDPP-RL" / d / "checkpoint.pt") for d in DATASETS
                },
            },
            "baseline_adaptations": {"PPDPP": {"status": "validated", "artifact_sha256": policies}},
            "scenario_manifests": sampled_ids,
            "config_sha256": file_hash(self.config_path),
            "package_lock_sha256": file_hash(self.root / "runtime/uv.lock"),
            "code": code_provenance(self.root),
            "gate_evidence": gate_hashes,
        }
        if hasattr(self, "recovery_evidence"):
            payload["schema_version"] = 3
            payload["recovery_amendment"] = self.recovery_evidence
        lock_path = stage.directory / "selection.lock"
        if lock_path.exists():
            if verify_selection(lock_path, digest(payload)) != payload:
                raise ValueError("resumed selection changed")
        else:
            write_selection(lock_path, payload)
        return combine_compute(
            [json.loads(p.read_text()) for p in stage.directory.glob("compute-*.json")]
        )

    def test(self, stage):
        if self.fixed_test_setting is not None and (
            getattr(self, "parallel_evaluation", False)
            or getattr(self, "evaluation_methods", None)
        ):
            raise ValueError("shared fixed-test setting requires the unsharded comparison runner")
        lock_path = self.output / "selection/selection.lock"
        sha = json.loads(lock_path.read_text())["sha256"]
        lock = verify_selection(lock_path, sha)
        if (
            lock["config_sha256"] != file_hash(self.config_path)
            or lock["models"] != self.models
            or lock["prompts"] != prompts()
            or lock["code"] != code_provenance(self.root)
        ):
            raise ValueError("test runtime differs from locked selection")
        # First opening of official test bodies, after the verified immutable selection lock.
        scenarios_by_data = {
            d: [
                r
                for split in SPLITS[d]
                if split.startswith("test")
                for r in read_release(self.root, d, split, selection_verified=True)
            ]
            for d in DATASETS
        }
        for dataset, test_scenarios in scenarios_by_data.items():
            if len({r["id"] for r in test_scenarios}) != len(test_scenarios):
                raise ValueError("official test membership has duplicate IDs across splits")
            audits = {}
            for split in ("train", "dev"):
                available = load_prepared(self.root, dataset, split)
                exact = sorted(
                    {digest(r["messages"]) for r in available}
                    & {digest(r["messages"]) for r in test_scenarios}
                )
                near = audit_train_dev(available, test_scenarios)
                audits[split] = {"exact": exact, "near": near}
            atomic_json(stage.directory / f"contamination-{dataset}.json", audits)
            if any(v["exact"] or v["near"]["collisions"] for v in audits.values()):
                raise ValueError(
                    "post-lock test contamination detected; retain artifacts and request amendment decision"
                )
        if self.fixed_test_setting is not None:
            scenarios_by_data, sample_manifest = select_fixed_test_populations(
                scenarios_by_data, self.fixed_test_setting
            )
            sample_manifest["dataset_setting"] = self.fixed_test_setting["payload"]
            sample_manifest["dataset_setting_sha256"] = self.fixed_test_setting["sha256"]
        else:
            scenarios_by_data, sample_manifest = sample_test_scenarios(
                scenarios_by_data, self.cfg.get("test_sampling")
            )
        sample_manifest["selection_sha256"] = sha
        sample_manifest["config_sha256"] = file_hash(self.config_path)
        sample_path = self.test_sample_path
        self.test_sample_path = sample_path
        if sample_path.exists() and json.loads(sample_path.read_text()) != sample_manifest:
            raise ValueError("resumed test sample membership/provenance changed")
        atomic_json(sample_path, sample_manifest)
        self.test_scenario_order_path = stage.directory / self.test_scenario_order_path.name
        atomic_json(
            self.test_scenario_order_path,
            {d: [s["id"] for s in rows] for d, rows in scenarios_by_data.items()},
        )
        progress = []
        for dataset, scenarios in scenarios_by_data.items():
            records, vectors = load_embeddings(self.output / "embeddings" / dataset / "bank")
            head = load_head(
                self.output / "learning" / dataset / lock["selected_head"] / "checkpoint.pt"
            )
            policy, tokenizer, _ = load_policy(
                self.root, self.output / "PPDPP-RL" / dataset / "checkpoint.pt"
            )
            thresholds = lock["thresholds"][dataset]
            graphs = {}
            with Servers(
                self.root, stage.directory / "servers" / dataset, self.settings
            ) as servers:
                b = servers.backends()
                for protocol in self.cfg["protocols"]:
                    for seed in self.test_eval_seeds:
                        for scenario in scenarios:
                            if (self.output / "stop.requested").exists():
                                raise InterruptedError(
                                    "allocation stop requested; paired artifacts retained"
                                )
                            target = scenario["target"][0]
                            if target not in graphs:
                                graphs[target] = make_graph(
                                    records,
                                    vectors,
                                    head,
                                    thresholds["H_edge"],
                                    self.cfg["node_budget"],
                                    17,
                                    target,
                                )
                            graph, goals = graphs[target]

                            def evaluate_method(
                                method,
                                matched=False,
                                b=b,
                                scenario=scenario,
                                protocol=protocol,
                                seed=seed,
                                dataset=dataset,
                                graph=graph,
                                goals=goals,
                                thresholds=thresholds,
                                policy=policy,
                                tokenizer=tokenizer,
                            ):
                                budget = "matched" if matched else "natural"
                                result = self.evaluate(
                                    b,
                                    scenario,
                                    method,
                                    protocol,
                                    seed,
                                    stage.directory
                                    / dataset
                                    / protocol
                                    / str(seed)
                                    / method
                                    / budget
                                    / scenario["id"],
                                    {
                                        "phase": "test",
                                        "selection_sha256": sha,
                                        "budget": budget,
                                        "config_sha256": file_hash(self.config_path),
                                        "fixed_test_setting_sha256": (
                                            self.fixed_test_setting["sha256"]
                                            if self.fixed_test_setting
                                            else None
                                        ),
                                    },
                                    graph,
                                    goals,
                                    thresholds["H_cover"],
                                    policy,
                                    tokenizer,
                                    matched,
                                )
                                return result, (
                                    stage.directory
                                    / dataset
                                    / protocol
                                    / str(seed)
                                    / method
                                    / budget
                                    / scenario["id"]
                                )

                            method_shard = getattr(self, "evaluation_methods", None)
                            if method_shard:
                                results = [
                                    (method, matched, evaluate_method(method, matched)[0])
                                    for method, matched in method_shard
                                ]
                            elif getattr(self, "parallel_evaluation", False):
                                from .concurrency import run_method_pair

                                paired = run_method_pair(
                                    evaluate_method,
                                    stage.directory
                                    / dataset
                                    / protocol
                                    / str(seed)
                                    / "paired-timing"
                                    / scenario["id"],
                                )
                                results = [
                                    ("PPDPP", False, paired[0]),
                                    ("TDRL", False, paired[1]),
                                    ("TDRL-R", False, evaluate_method("TDRL-R")[0]),
                                    ("TDRL", True, evaluate_method("TDRL", True)[0]),
                                ]
                            else:
                                cells = (
                                    [
                                        ("PPDPP", False),
                                        ("TDRL-R", False),
                                        ("TDRL", False),
                                    ]
                                    if self.fixed_test_setting
                                    else [
                                        ("PPDPP", False),
                                        ("TDRL-R", False),
                                        ("TDRL", False),
                                        ("TDRL", True),
                                    ]
                                )
                                results = [
                                    (method, matched, evaluate_method(method, matched)[0])
                                    for method, matched in cells
                                ]
                            for method, matched, result in results:
                                progress.append(
                                    {
                                        "dataset": dataset,
                                        "method": method,
                                        "budget": "matched" if matched else "natural",
                                        "protocol": protocol,
                                        "record": result,
                                    }
                                )
                            # Exact observed speed, declared projection; no scenario/seed is dropped.
                            per_config = {}
                            for p in progress:
                                key = (p["method"], p["budget"])
                                per_config.setdefault(key, []).append(p["record"]["wall_s"])
                            per_scenario = sum(
                                float(np.quantile(v, 0.95)) for v in per_config.values()
                            )
                            if getattr(self, "parallel_evaluation", False) and not method_shard:
                                pairs = [
                                    json.loads(p.read_text())["wall_s"]
                                    for p in stage.directory.rglob("pair-timing.json")
                                ]
                                per_scenario = float(np.quantile(pairs, 0.95)) + sum(
                                    float(np.quantile(per_config[key], 0.95))
                                    for key in [("TDRL-R", "natural"), ("TDRL", "matched")]
                                )
                            expected = (
                                sum(len(v) for v in scenarios_by_data.values())
                                * len(self.test_eval_seeds)
                                * len(self.cfg["protocols"])
                            )
                            completed = len(progress) // (
                                len(method_shard)
                                if method_shard
                                else (3 if self.fixed_test_setting else 4)
                            )
                            forecast = (
                                (expected - completed)
                                * per_scenario
                                * self.cfg["feasibility_safety_factor"]
                            )
                            remaining = self.cfg["walltime_seconds"] - (
                                time.time() - self.allocation_start
                            )
                            estimate = {
                                "status": "measured_projection",
                                "completed_scenario_seed_protocol": completed,
                                "required_scenario_seed_protocol": expected,
                                "remaining_seconds": forecast,
                                "allocation_remaining_seconds": remaining,
                                "per_scenario_p95_sum": per_scenario,
                            }
                            atomic_json(self.output / "eta.json", estimate)
                            if (
                                completed >= 2
                                and forecast + self.cfg["reserve_seconds"] > remaining
                            ):
                                raise ValueError(
                                    "full mandatory scoped matrix exceeds allocation: explicit scope/budget decision required"
                                )
                atomic_json(
                    stage.directory / f"compute-{dataset}.json",
                    {**counters(b["generation"], b["embedding"]), **servers.compute_metadata()},
                )
            del head, policy, vectors, graphs
            gc.collect()
            torch.cuda.empty_cache()
        atomic_json(stage.directory / "records.json", progress)
        return combine_compute(
            [json.loads(p.read_text()) for p in stage.directory.glob("compute-*.json")]
        )

    def report(self, stage):
        source = self.test_stage_directory / "records.json"
        rows = json.loads(source.read_text())
        sha = json.loads((self.output / "selection/selection.lock").read_text())["sha256"]
        report = {
            "scope": self.cfg["scope"],
            "execution": {
                "allocated_gpu_count": 2 if getattr(self, "two_gpu_execution", False) else 1,
                "parallel_methods": ["PPDPP", "TDRL"]
                if getattr(self, "parallel_evaluation", False)
                else [],
                "shared_compute_accounting": "equal_active_dialogue_share_v1"
                if getattr(self, "parallel_evaluation", False)
                else "serial_dialogue_wall_time",
                "generation_replay": "durable request journal; serial-parallel text equality not assumed",
            },
            "test_population": json.loads(self.test_sample_path.read_text()),
            "experimental_seeds": self.test_eval_seeds,
            "evaluation_budgets": self.test_eval_budgets,
            "seed_uncertainty": (
                "single experimental seed; scenario-paired CI does not measure training/seed variability"
                if len(self.test_eval_seeds) == 1
                else "evaluation seed repetitions; one training seed"
            ),
            "evidence": {
                "path": str(source),
                "sha256": file_hash(source),
                "run_id": json.loads((source.parent / "manifest.json").read_text())["run_id"],
            },
            "claim_limit": "interactive simulation only; human and full baseline project pending",
            "results": [],
        }
        if self.fixed_test_setting is not None:
            report["dataset_setting"] = self.fixed_test_setting["payload"]
            report["dataset_setting_sha256"] = self.fixed_test_setting["sha256"]
        for dataset in DATASETS:
            for protocol in self.cfg["protocols"]:
                selected = [
                    p for p in rows if p["dataset"] == dataset and p["protocol"] == protocol
                ]
                expected_order = json.loads(self.test_scenario_order_path.read_text())
                ids = expected_order[dataset]
                groups = {}
                cells = (
                    [("PPDPP", "natural"), ("TDRL-R", "natural"), ("TDRL", "natural")]
                    if self.fixed_test_setting
                    else [
                        ("PPDPP", "natural"),
                        ("TDRL-R", "natural"),
                        ("TDRL", "natural"),
                        ("TDRL", "matched"),
                    ]
                )
                for method, budget in cells:
                    group = [
                        p["record"]
                        for p in selected
                        if p["method"] == method and p["budget"] == budget
                    ]
                    validate_records(group, ids, self.test_eval_seeds, sha)
                    groups[(method, budget)] = group
                    report["results"].append(
                        {
                            "dataset": dataset,
                            "protocol": protocol,
                            "method": method,
                            "budget": budget,
                            "aggregate": aggregate(group),
                            "individual_seeds": {
                                str(s): aggregate([r for r in group if r["seed"] == s])
                                for s in self.test_eval_seeds
                            },
                            "generation_calls": sum(r["generation_requests"] for r in group),
                            "embedding_calls": sum(r["embedding_requests"] for r in group),
                            "input_tokens": sum(r["input_tokens"] for r in group),
                            "output_tokens": sum(r["output_tokens"] for r in group),
                            "matched_alias": "PPDPP/TDRL-R natural traces meet the same one-call planner cap"
                            if method != "TDRL"
                            else None,
                        }
                    )
                paired = {}
                for method in ("PPDPP", "TDRL-R"):

                    def matrix(group, scenario_ids=ids):
                        lookup = {
                            (r["scenario_id"], r["seed"]): float(r["accepted"]) for r in group
                        }
                        return [
                            [lookup[(i, s)] for s in self.test_eval_seeds]
                            for i in scenario_ids
                        ]

                    paired[method] = paired_bootstrap(
                        matrix(groups[("TDRL", "natural")]), matrix(groups[(method, "natural")])
                    )
                atomic_json(stage.directory / f"paired-{dataset}-{protocol}.json", paired)
        atomic_json(stage.directory / "report.json", report)
        lines = [
            "# Three-method interactive simulation comparison",
            "",
            str(report["claim_limit"]),
            "",
            f"Source: `{source}`; SHA-256 `{file_hash(source)}`.",
            "",
            "| Dataset | Protocol | Method | Budget | SR | Average turns | H100 s/dialogue |",
            "|---|---|---|---|---:|---:|---:|",
        ]
        for r in report["results"]:
            m = r["aggregate"]
            lines.append(
                f"| {r['dataset']} | {r['protocol']} | {r['method']} | {r['budget']} | "
                f"{m['sr']:.4f} | {m['average_turns']:.2f} | {m['h100_seconds_per_dialogue']:.2f} |"
            )
        (stage.directory / "report.md").write_text("\n".join(lines) + "\n")
        return {"h100_seconds": 0.0}
