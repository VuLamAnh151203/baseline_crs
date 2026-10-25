#!/usr/bin/env python3
"""Fixed 1000-scenario TDRL/DiffTOD prefix; count does not change replay identity."""

import argparse
import json
import os
import time
from dataclasses import asdict, replace
from pathlib import Path

import torch
from tokenizers import Tokenizer

from tdrl.backend import require_h100_slurm
from tdrl.contracts import Decoding
from tdrl.difftod import DiffTODPlanner
from tdrl.difftod_evaluation import load_selected_policy
from tdrl.evaluation import initial_state, run_dialogue, simulator_prompt, visible_context
from tdrl.feasibility import full_turn_dialogue_proxy
from tdrl.judge import LocalJudge
from tdrl.metrics import aggregate
from tdrl.official_data import SPLITS, read_release
from tdrl.planner import ConditionalPlanner, PlannerConfig
from tdrl.prefix_selection import verify_prefix_selection
from tdrl.provenance import atomic_json, digest, file_hash, validate_manifest
from tdrl.runtime import Servers
from tdrl.stages import GpuMonitor, Stage, code_provenance, counters
from tdrl.state import SERIALIZATION_HASH
from tdrl.strategy_annotations import apply_strategy_overlay
from tdrl.test_prefix import dialogue_contract, prefix_cases, seal_sample
from tdrl.training import load_embeddings, load_head, load_prepared, make_graph


def verify_gates(output, selection_sha, dataset):
    checks = []
    for method in ("TDRL", "DiffTOD"):
        for slot in (0, 1):
            p = output / f"smoke-{dataset}-{method}-{slot}"
            m = json.loads((p / "manifest.json").read_text())
            validate_manifest(m)
            if m["status"] != "complete" or m["stage"] != "Fixed1000-prefix-smoke":
                raise ValueError("fresh prefix runtime smoke required")
            if file_hash(p / "checks.json") != m["outputs"]["checks.json"]:
                raise ValueError("prefix gate checksum mismatch")
            c = json.loads((p / "checks.json").read_text())
            if (
                c["selection_sha256"] != selection_sha
                or c["status"] != "passed"
                or not c["exact_plan_replay"]
                or c["optimizer_updates"] != 0
            ):
                raise ValueError("prefix gate contract/replay mismatch")
            checks.append(c)
        if len({c["gpu"]["uuid"] for c in checks[-2:]}) != 2:
            raise ValueError("both distinct H100 runtime gates required")
    return checks


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--method", choices=["TDRL", "DiffTOD"], required=True)
    p.add_argument("--mode", choices=["smoke", "evaluate"], required=True)
    p.add_argument("--slot", choices=[0, 1], type=int, required=True)
    p.add_argument("--count", type=int, default=400)
    p.add_argument("--dataset", choices=["TopDial", "DuRecDial2.0"], required=True)
    a = p.parse_args()
    root = Path.cwd()
    config = root / "configs/fixed1000-prefix-tdrl-difftod-v1.json"
    cfg = json.loads(config.read_text())
    selection, candidate, selection_sha = verify_prefix_selection(
        root, a.output / "selection.lock", code_provenance(root)["commit"]
    )
    if not 1 <= a.count <= 1000:
        raise ValueError("prefix count must be in [1,1000]")
    candidate_cfg = json.loads((root / cfg["candidate_config"]).read_text())
    base = json.loads((root / candidate_cfg["comparison_config"]).read_text())
    os.environ["TDRL_CANDIDATE_WORKERS"] = str(candidate_cfg["candidate_workers"])
    gpu = require_h100_slurm()
    # Selection and fresh gates precede the first read of official test bodies.
    if a.mode == "evaluate":
        verify_gates(a.output, selection_sha, a.dataset)
    if a.mode == "smoke":
        cases = [(a.dataset, load_prepared(root, a.dataset, "train")[0])]
        membership = {"claim": "training-only-runtime-smoke"}
    else:
        populations = {
            d: [
                r
                for split in SPLITS[d]
                if split.startswith("test")
                for r in read_release(root, d, split, selection_verified=True)
            ]
            for d in ("TopDial", "DuRecDial2.0")
        }
        cases, sample, membership = prefix_cases(populations, a.count)
        seal_sample(a.output / "sealed-sample.json", sample)
        atomic_json(
            a.output / f"membership-{a.count}-{a.dataset}.json", {**membership, "sample": sample}
        )
        cases = [(d, r) for d, r in cases if d == a.dataset]
        if not cases:
            raise ValueError("requested prefix does not contain this dataset")
    directory = a.output / (
        f"smoke-{a.dataset}-{a.method}-{a.slot}"
        if a.mode == "smoke"
        else f"prefix-{a.count}-{a.dataset}-{a.method}"
    )
    stage = Stage(
        root,
        directory,
        "Fixed1000-prefix-" + a.mode,
        config,
        {
            "selection_sha256": selection_sha,
            "method": a.method,
            "slot": a.slot,
            "membership": membership,
            "serialization_sha256": SERIALIZATION_HASH,
        },
    )
    if stage.reused:
        return
    started = time.monotonic()
    compute = {}
    settings = json.loads((root / "configs/runtime.json").read_text())
    for kind in ("generation", "embedding"):
        settings[kind]["port"] += a.slot * 100
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(17)
    torch.cuda.manual_seed_all(17)
    model = tokenizer = head = records = vectors = None
    policy_provenance = {}
    legacy = json.loads((root / cfg["durecdial_source"] / "selection/selection.lock").read_text())[
        "payload"
    ]
    thresholds = (
        candidate["thresholds"] if a.dataset == "TopDial" else legacy["thresholds"][a.dataset]
    )
    if a.method == "DiffTOD":
        loader = load_selected_policy
        if a.dataset == "DuRecDial2.0":
            from tdrl.difftod_durecdial_evaluation import load_selected_policy as loader
        fitting = cfg["difftod_fitting"] if a.dataset == "TopDial" else cfg["durecdial_fitting"]
        model, policy_provenance = loader(root, root / fitting)
        expected_policy = (
            candidate["difftod_policy_sha256"]
            if a.dataset == "TopDial"
            else selection["durecdial_policy_sha256"]
        )
        if policy_provenance["checkpoint_sha256"] != expected_policy:
            raise ValueError("DiffTOD selected checkpoint changed")
        pin = json.loads((root / "configs/difftod-policy-model.lock.json").read_text())
        tokenizer = Tokenizer.from_file(
            str(root / "models/difftod/gpt2" / pin["tokenizer"]["revision"] / "tokenizer.json")
        )
        native = json.loads(
            (
                root
                / (
                    "configs/difftod-topdial-fit-v1.json"
                    if a.dataset == "TopDial"
                    else "configs/difftod-durecdial-fit-v1.json"
                )
            ).read_text()
        )
    else:
        if a.dataset == "TopDial":
            head_path = root / candidate_cfg["head_source"] / "MRNFixed/checkpoint.pt"
            expected_head = candidate["head_sha256"]
            bank = root / candidate_cfg["source"] / "embeddings/TopDial/bank"
        else:
            head_path = (
                root
                / cfg["durecdial_source"]
                / f"learning/{a.dataset}/{cfg['durecdial_head']}/checkpoint.pt"
            )
            expected_head = selection["durecdial_head_sha256"]
            bank = root / cfg["durecdial_source"] / f"embeddings/{a.dataset}/bank"
        head = load_head(head_path)
        if head.checkpoint_hash != expected_head:
            raise ValueError("frozen TDRL head changed")
        records, vectors = load_embeddings(bank)
        if a.dataset == "TopDial":
            records = apply_strategy_overlay(
                records,
                root / candidate_cfg["strategy_overlay"],
                file_hash(bank / "index.json"),
                exclude_farewell=candidate_cfg["exclude_farewell_guidance"],
            )
    graphs = {}
    results = []
    try:
        with GpuMonitor(directory / "gpu-telemetry.jsonl") as monitor:
            with Servers(root, directory / "servers", settings) as servers:
                b = servers.backends()
                judge = LocalJudge(b["generation"], base["judge"])
                for protocol in cfg["protocols"]:
                    for dataset, scenario in cases:
                        goals = []
                        if a.method == "DiffTOD":
                            actor = DiffTODPlanner(
                                model,
                                tokenizer,
                                b["generation"],
                                b["embedding"],
                                scenario,
                                Decoding(**base["assistant_decoding"]),
                                Decoding(**native["guidance_decoding"]),
                                directory / "policy-journal"
                                if a.mode == "smoke"
                                else a.output / f"{a.dataset}-DiffTOD-policy-journal",
                                policy_provenance,
                                root / "artifacts/difftod-nltk-data",
                            )
                        else:
                            key = tuple(scenario["target"])
                            if key not in graphs:
                                graphs[key] = make_graph(
                                    records,
                                    vectors,
                                    head,
                                    thresholds["H_edge"],
                                    base["node_budget"],
                                    17,
                                    scenario["target"][0],
                                    target_topic=scenario["target"][1]
                                    if a.dataset == "TopDial"
                                    else None,
                                    actionable=a.dataset == "TopDial",
                                    use_observed_guidance=a.dataset == "TopDial",
                                    reference_scope=candidate_cfg["reference_scope"]
                                    if a.dataset == "TopDial"
                                    else "goal_type",
                                )
                            graph, goals = graphs[key]
                            actor = ConditionalPlanner(
                                b["generation"],
                                b["embedding"],
                                graph,
                                PlannerConfig(
                                    thresholds["H_cover"],
                                    routing="conditional",
                                    revision=2 if a.dataset == "TopDial" else 1,
                                    reference_selection=candidate_cfg["reference_selection"]
                                    if a.dataset == "TopDial"
                                    else "entry_plus_path",
                                    preserve_retrieval_candidate=a.dataset == "TopDial",
                                    semantic_rollout_terminals=a.dataset == "TopDial",
                                    use_observed_guidance=a.dataset == "TopDial",
                                    ground_target_facts=a.dataset == "TopDial",
                                ),
                                Decoding(**base["assistant_decoding"]),
                                Decoding(**base["simulator_decoding"]),
                                visible_context=visible_context(scenario),
                                rollout_judge=judge,
                                rollout_scenario=scenario,
                            )
                        if a.mode == "smoke" and a.method == "TDRL":
                            for routing in ("retrieval_only", "always"):
                                actor.cfg = replace(actor.cfg, routing=routing)
                                actor.semantic_judging = True
                                actor.seed_context = f"prefix-smoke/{protocol}/{routing}"
                                actor.simulator_prompt = simulator_prompt(scenario, protocol)
                                rid = f"prefix-smoke/{a.slot}/{protocol}/{routing}"
                                first_gate = actor.act(initial_state(scenario), goals, 17, rid)
                                before_gate = (
                                    len(b["generation"].requests),
                                    len(b["embedding"].requests),
                                )
                                replay_gate = actor.act(initial_state(scenario), goals, 17, rid)
                                if (
                                    {k: v for k, v in asdict(first_gate).items() if k != "compute"}
                                    != {
                                        k: v
                                        for k, v in asdict(replay_gate).items()
                                        if k != "compute"
                                    }
                                    or before_gate
                                    != (len(b["generation"].requests), len(b["embedding"].requests))
                                    or first_gate.compute["recovery_triggered"]
                                    != (routing == "always")
                                ):
                                    raise ValueError("covered/off-map prefix smoke replay failed")
                            actor.cfg = replace(actor.cfg, routing="conditional")
                        contract = dialogue_contract(
                            selection_sha, a.method, dataset, scenario["id"], goals
                        )
                        if a.mode == "smoke":
                            contract = {
                                **contract,
                                "phase": "fixed1000-prefix-training-smoke",
                                "request_namespace": digest([contract, a.slot]),
                            }
                        dialog_dir = (
                            (directory if a.mode == "smoke" else a.output)
                            / "dialogues"
                            / a.method
                            / dataset
                            / protocol
                            / digest(scenario["id"])
                        )
                        result = run_dialogue(
                            scenario,
                            actor,
                            b["generation"],
                            b["embedding"],
                            protocol,
                            17,
                            dialog_dir,
                            contract,
                            turn_budget=base["turn_budget"],
                            simulator_decoding=Decoding(**base["simulator_decoding"]),
                            judge=judge,
                        )
                        if a.mode == "evaluate":
                            seal_sample(
                                dialog_dir / "completed-seal.json",
                                {
                                    "dialogue_sha256": file_hash(dialog_dir / "dialogue.json"),
                                    "selection_sha256": selection_sha,
                                },
                            )
                        if a.mode == "smoke":
                            first = result["turns_trace"][0]["plan"]
                            if a.method == "DiffTOD":
                                actor.trace = None
                            req = first["replay_ids"][0].removesuffix("/live")
                            actor.seed_context = f"{scenario['id']}/{protocol}/0"
                            before = (len(b["generation"].requests), len(b["embedding"].requests))
                            replay = asdict(actor.act(initial_state(scenario), goals, 17, req))
                            if {k: v for k, v in replay.items() if k != "compute"} != {
                                k: v for k, v in first.items() if k != "compute"
                            } or before != (
                                len(b["generation"].requests),
                                len(b["embedding"].requests),
                            ):
                                raise ValueError("prefix exact plan/journal replay failed")
                        results.append(result)
                        atomic_json(
                            directory / "progress.json",
                            {
                                "complete": len(results),
                                "expected": len(cases) * 2,
                                "latest_scenario": scenario["id"],
                                "requested_prefix_count": a.count,
                            },
                        )
                compute = {
                    **counters(b["generation"], b["embedding"]),
                    **servers.compute_metadata(),
                }
            if list(directory.rglob("pending-*.json")) or list(directory.rglob("*.pending.json")):
                raise ValueError("unresolved runtime requests; preserve journals")
            if len(results) != len(cases) * 2 or any(r["infrastructure_errors"] for r in results):
                raise ValueError("incomplete/error prefix evaluation")
            atomic_json(directory / "records.json", results)
            atomic_json(
                directory / "metrics.json",
                {
                    p: aggregate([r for r in results if r["protocol"] == p])
                    for p in cfg["protocols"]
                },
            )
            if a.mode == "smoke":
                atomic_json(
                    directory / "checks.json",
                    {
                        "status": "passed",
                        "gpu": gpu,
                        "selection_sha256": selection_sha,
                        "exact_plan_replay": True,
                        "optimizer_updates": 0,
                        "method": a.method,
                        "full_turn_proxy_seconds": max(
                            full_turn_dialogue_proxy(
                                r["wall_s"], r["actual_turns"], base["turn_budget"]
                            )
                            for r in results
                        ),
                    },
                )
        atomic_json(directory / "gpu-summary.json", monitor.summary())
        stage.write(
            "complete",
            provenance_updates={
                "gpu": gpu,
                "checkpoint": policy_provenance,
                "scenarios": membership,
                "seeds": {"root": 17, "protocols": cfg["protocols"]},
            },
            compute_updates={**compute, "h100_seconds": time.monotonic() - started},
        )
    except Exception as error:
        stage.write(
            "failed",
            error=error,
            compute_updates={**compute, "h100_seconds": time.monotonic() - started},
        )
        raise


if __name__ == "__main__":
    main()
