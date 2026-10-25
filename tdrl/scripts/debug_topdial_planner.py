#!/usr/bin/env python3
"""Paired TopDial development diagnostics; never load official test bodies."""

import argparse
import ast
import gc
import json
import os
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from tdrl.acceptance import normalize
from tdrl.backend import require_h100_slurm
from tdrl.confirmation import confirmation_candidate, confirmation_cases, verify_planner_gates
from tdrl.contracts import Decoding
from tdrl.evaluation import (
    PPDPPPlanner,
    initial_state,
    run_dialogue,
    simulator_prompt,
    visible_context,
)
from tdrl.fidelity import replay_policy
from tdrl.judge import LocalJudge
from tdrl.metrics import aggregate
from tdrl.planner import ASSISTANT_POLICY, INSTRUCTION, ConditionalPlanner, PlannerConfig
from tdrl.policy_optimizer import policy_optimizer
from tdrl.policy_training import load_policy
from tdrl.provenance import atomic_json, digest, file_hash
from tdrl.runtime import Servers
from tdrl.stages import Stage, code_provenance, counters
from tdrl.state import SERIALIZATION_HASH
from tdrl.strategies import ACTION_INSTRUCTIONS
from tdrl.strategy_annotations import apply_strategy_overlay
from tdrl.training import load_embeddings, load_head, load_prepared, make_graph


def smoke_fixtures(cfg, cases, records):
    """Choose gates from anchor availability, never distance or dialogue outcomes."""
    if cfg.get("reference_scope") != "exact_target_when_supported":
        return [("first", cases[0])]
    anchors = {
        (r["target"][0], normalize(r["target"][1]))
        for r in records
        if r["split"] == "train" and r["guidance"] and r["goal_reference"]
    }
    selected = {}
    for case in cases:
        key = (case["target"][0], normalize(case["target"][1]))
        condition = "exact_target" if key in anchors else "goal_type_transfer"
        selected.setdefault(condition, case)
    if set(selected) != {"exact_target", "goal_type_transfer"}:
        raise ValueError("target-pool smoke requires exact-anchor and transfer fixtures")
    return [
        (condition, selected[condition]) for condition in ["exact_target", "goal_type_transfer"]
    ]


def audit_fitting(root, source):
    old = json.loads((source / "report/manifest.json").read_text())["provenance"]["code"]["commit"]
    unchanged = {}
    for name in [
        "policy_training",
        "policy_optimizer",
        "ppdpp",
        "reachability",
        "state",
        "backend",
        "runtime",
    ]:
        path = "tdrl/src/tdrl/" + name + ".py"
        previous = subprocess.check_output(["git", "show", old + ":" + path], cwd=root)
        if previous != (root / "src/tdrl" / (name + ".py")).read_bytes():
            raise ValueError("fitting/runtime module changed: " + name)
        unchanged[name] = file_hash(root / "src/tdrl" / (name + ".py"))
    for name, excluded in [
        ("training", {"make_graph", "train_head", "load_head"}),
        ("evaluation", {"ACTION_INSTRUCTIONS"}),
    ]:
        path = "tdrl/src/tdrl/" + name + ".py"
        before = ast.parse(subprocess.check_output(["git", "show", old + ":" + path], cwd=root))
        after = ast.parse((root / "src/tdrl" / (name + ".py")).read_text())

        def definitions(tree, excluded=excluded):
            return {
                n.name: ast.dump(n, include_attributes=False)
                for n in tree.body
                if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name not in excluded
            }

        if definitions(before) != definitions(after):
            raise ValueError("fitting/controller/dialogue semantics changed: " + name)
        if name == "training":
            # Numerical learner code is not executed by a frozen-checkpoint dialogue
            # diagnostic. Verify the used loader's forward/weight/provenance path;
            # revision schema validation may additionally reject unknown revisions.
            def loader(tree):
                node = next(n for n in tree.body if getattr(n, "name", None) == "load_head")
                node.body = [n for n in node.body if not isinstance(n, ast.If)]
                return ast.dump(node, include_attributes=False)

            if loader(before) != loader(after):
                raise ValueError("frozen head loader/forward changed")
        if name == "evaluation":
            previous_actions = next(
                n
                for n in before.body
                if isinstance(n, ast.Assign)
                and any(getattr(t, "id", None) == "ACTION_INSTRUCTIONS" for t in n.targets)
            )
            if ast.literal_eval(previous_actions.value) != ACTION_INSTRUCTIONS:
                raise ValueError("shared PPDPP action realization changed")
    hashes = {}
    for name in [
        "PPDPP-RL/TopDial",
        "learning/TopDial/MRNFixed",
        "learning/TopDial/IQE",
        "embeddings/TopDial",
    ]:
        m = json.loads((source / name / "manifest.json").read_text())
        if m["status"] != "complete":
            raise ValueError("fitting is not complete: " + name)
        for filename, expected in m["outputs"].items():
            if filename not in {"checkpoint.pt", "contract.json", "fidelity.json", "index.json"}:
                continue
            if file_hash(source / name / filename) != expected:
                raise ValueError("fitting artifact checksum mismatch")
        hashes[name] = file_hash(source / name / "manifest.json")
    return {
        "source_commit": old,
        "unchanged_modules": unchanged,
        "fitting_manifests": hashes,
        "PPDPP_messages": "identical action instructions and policy interfaces",
    }


def diagnostic_contract(
    variant,
    goals,
    code,
    config_sha256,
    head_sha256=None,
    thresholds=None,
    confirmation_lock_sha256=None,
):
    payload = {
        "phase": "development-confirmation" if confirmation_lock_sha256 else "development-debug",
        "variant": variant,
        "method": "PPDPP" if variant == "PPDPP" else "TDRL-R" if variant.endswith("-R") else "TDRL",
        "config_sha256": config_sha256,
        "code": code,
        "target_nodes": goals,
        "head_sha256": head_sha256,
        "thresholds": thresholds,
    }
    if confirmation_lock_sha256:
        payload["confirmation_lock_sha256"] = confirmation_lock_sha256
    payload["request_namespace"] = digest({"schema_version": 1, "dialogue_contract": payload})
    return payload


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--slot", type=int, choices=[0, 1], required=True)
    p.add_argument("--mode", choices=["smoke", "evaluate"], required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--variants-only", nargs="+")
    p.add_argument("--config", type=Path, default=Path("configs/topdial-debug-v2.json"))
    p.add_argument("--confirmation-lock", type=Path)
    a = p.parse_args()
    root = Path.cwd()
    path = root / a.config
    cfg = json.loads(path.read_text())
    if "candidate_workers" in cfg:
        os.environ.setdefault("TDRL_CANDIDATE_WORKERS", str(cfg["candidate_workers"]))
        if int(os.environ["TDRL_CANDIDATE_WORKERS"]) != cfg["candidate_workers"]:
            raise ValueError("candidate execution differs from frozen configuration")
    candidate_workers = int(os.environ.get("TDRL_CANDIDATE_WORKERS", "1"))
    if candidate_workers not in {1, 3}:
        raise ValueError("unsupported candidate worker count")
    head_types = cfg.get("head_types", ["MRNFixed", "IQE"])
    if (
        not head_types
        or len(set(head_types)) != len(head_types)
        or not set(head_types) <= {"MRNFixed", "IQE"}
    ):
        raise ValueError("unknown/duplicate diagnostic heads")
    base = json.loads((root / cfg["comparison_config"]).read_text())
    source = root / cfg["source"]
    all_cases = sorted(
        load_prepared(root, "TopDial", "dev"), key=lambda x: digest([cfg["seed"], x["id"]])
    )
    calibration = all_cases[: cfg["calibration_count"]]
    cases = all_cases[
        cfg["development_offset"] : cfg["development_offset"] + cfg["development_count"]
    ]
    confirmation_payload = None
    if a.confirmation_lock:
        if a.mode != "evaluate" or head_types != ["MRNFixed"]:
            raise ValueError("confirmation requires evaluation of the frozen MRN candidate")
        confirmation_payload, membership = confirmation_candidate(
            root,
            a.confirmation_lock,
            code_provenance(root)["commit"],
            root / cfg["comparison_config"],
        )
        if file_hash(path) != confirmation_payload["files"]["candidate_config"]["sha256"]:
            raise ValueError("planner configuration changed after freeze")
        cases = confirmation_cases(
            all_cases, membership, file_hash(root / "data/processed/TopDial/dev.jsonl")
        )
        calibration = []
    assert len(cases) == 64 and not set(s["id"] for s in cases) & set(s["id"] for s in calibration)
    audit = audit_fitting(root, source)
    numerical_source = root / cfg["head_source"] if cfg.get("head_source") else None
    if numerical_source:
        imported = {}
        for h in head_types:
            directory = numerical_source / h
            manifest = json.loads((directory / "manifest.json").read_text())
            if manifest["status"] != "complete":
                raise ValueError("new numerical head not complete")
            for filename in [
                "checkpoint.pt",
                "numeric-gate.json",
                "diagnostics.json",
                "heldout-temporal.json",
            ]:
                if file_hash(directory / filename) != manifest["outputs"][filename]:
                    raise ValueError("new head output checksum mismatch")
            gate = json.loads((directory / "numeric-gate.json").read_text())
            if (
                gate["status"] != "passed"
                or gate["forward_dtype"] != "bfloat16"
                or not gate["initial_forward_unchanged"]
                or not gate["exact_gradient_update_replay"]
            ):
                raise ValueError("new numerical head replay/precision gate failed")
            imported[h] = {
                "manifest_sha256": file_hash(directory / "manifest.json"),
                "checkpoint_sha256": file_hash(directory / "checkpoint.pt"),
            }
        audit["numerical_head_imports"] = imported
    stage = Stage(
        root,
        a.output,
        "TopDial-planner-confirmation" if confirmation_payload else "TopDial-planner-" + a.mode,
        path,
        {
            "fit_audit": audit,
            "scenario_ids": [s["id"] for s in cases],
            "calibration_ids": [s["id"] for s in calibration],
            "slot": a.slot,
            "candidate_workers": candidate_workers,
            "serialization_sha256": SERIALIZATION_HASH,
            "models": json.loads((root / "configs/models.lock.json").read_text()),
            "confirmation_lock_sha256": file_hash(a.confirmation_lock)
            if confirmation_payload
            else None,
            "strategy_overlay_manifest_sha256": file_hash(
                root / cfg["strategy_overlay"] / "manifest.json"
            )
            if cfg.get("strategy_overlay")
            else None,
        },
    )
    if stage.reused:
        return
    started = time.monotonic()
    gpu = require_h100_slurm()
    atomic_json(
        a.output / "scenario-ids.json",
        {
            "calibration": [s["id"] for s in calibration],
            "diagnostic": [s["id"] for s in cases],
            "sampling": "locked confirmation ranks 80:144"
            if confirmation_payload
            else "digest([17,id]) sorted development, ranks 16:80; frozen before inference",
            "config_sha256": file_hash(path),
        },
    )
    settings = json.loads((root / "configs/runtime.json").read_text())
    for kind in ["generation", "embedding"]:
        settings[kind]["port"] += a.slot * 100
    policy, tokenizer, _ = load_policy(root, source / "PPDPP-RL/TopDial/checkpoint.pt")
    optimizer = policy_optimizer(policy, base["ppdpp"], "rl")
    optimizer.reserve()
    controller = replay_policy(policy, tokenizer, load_prepared(root, "TopDial", "train")[0])
    policy.eval()
    records, vectors = load_embeddings(source / "embeddings/TopDial/bank")
    if cfg.get("strategy_overlay"):
        records = apply_strategy_overlay(
            records,
            root / cfg["strategy_overlay"],
            file_hash(source / "embeddings/TopDial/bank/index.json"),
            exclude_farewell=cfg.get("exclude_farewell_guidance", False),
            exclude_non_target_recommendations=cfg.get("exclude_non_target_recommendations", False),
        )
    heads = {
        h: load_head(
            (numerical_source / h / "checkpoint.pt")
            if numerical_source
            else (source / f"learning/TopDial/{h}/checkpoint.pt")
        )
        for h in head_types
    }
    thresholds = {}
    if confirmation_payload:
        if heads["MRNFixed"].checkpoint_hash != confirmation_payload["head_sha256"]:
            raise ValueError("head changed after candidate freeze")
        thresholds = {"MRNFixed": dict(confirmation_payload["thresholds"])}
    for h in [] if confirmation_payload else heads:
        trials = json.loads((source / f"selection/trials-TopDial-{h}.json").read_text())
        best = max(
            trials,
            key=lambda x: (
                x["metrics"]["sr"],
                -x["metrics"]["h100_seconds_per_dialogue"],
                -x["H_edge"],
                -x["H_cover"],
            ),
        )
        thresholds[h] = {k: best[k] for k in ["H_edge", "H_cover"]}
    if numerical_source and not confirmation_payload:
        # Connectivity scale is recalibrated on a frozen training-only pair sample.
        rng = np.random.default_rng(17)
        pairs = rng.integers(0, len(vectors), size=(1024, 2))
        for h, head in heads.items():
            with torch.no_grad():
                distances = (
                    head.distance(
                        vectors[pairs[:, 0]].to("cuda", torch.bfloat16),
                        vectors[pairs[:, 1]].to("cuda", torch.bfloat16),
                    )
                    .float()
                    .cpu()
                    .numpy()
                )
            thresholds[h]["H_edge"] = float(np.quantile(distances, cfg["edge_quantile"]))
    graph_cache = {}

    def graph_for(head, scenario, revision):
        key = (
            head,
            scenario["target"][0],
            scenario["target"][1] if revision == 2 else None,
            revision,
        )
        if key not in graph_cache:
            graph_cache[key] = make_graph(
                records,
                vectors,
                heads[head],
                thresholds[head]["H_edge"],
                base["node_budget"],
                17,
                scenario["target"][0],
                target_topic=scenario["target"][1] if revision == 2 else None,
                actionable=revision == 2,
                use_observed_guidance=cfg.get("use_observed_guidance", False),
                reference_scope=cfg.get("reference_scope", "goal_type")
                if revision == 2
                else "goal_type",
            )
        return graph_cache[key]

    variants = a.variants_only or cfg["variants"][str(a.slot)]
    if not set(variants) <= set(cfg["variants"][str(a.slot)]):
        raise ValueError("unknown diagnostic variants")
    try:
        with Servers(root, a.output / "servers", settings) as servers:
            b = servers.backends()
            judge = LocalJudge(b["generation"], base["judge"])
            if a.mode == "smoke":
                smoke = {}
                fixtures = smoke_fixtures(cfg, cases, records)
                for h, condition, smoke_case in [
                    (h, condition, scenario) for h in heads for condition, scenario in fixtures
                ]:
                    graph, goals = graph_for(h, smoke_case, 2)
                    if not goals:
                        raise ValueError("smoke needs training-backed guidance")
                    if condition != "first" and graph.provenance["goal_conditioning"] != condition:
                        raise ValueError("target-pool smoke conditioning mismatch")
                    if condition == "exact_target":
                        allowed = {
                            r["node_id"]
                            for r in records
                            if r["split"] == "train"
                            and r["target"][0] == smoke_case["target"][0]
                            and normalize(r["target"][1]) == normalize(smoke_case["target"][1])
                        }
                        if not set(graph.references) <= allowed:
                            raise ValueError(
                                "target-pool smoke retained another target's reference"
                            )
                    for routing, cover in [("conditional", 1000000.0), ("always", 0.0)]:
                        planner = ConditionalPlanner(
                            b["generation"],
                            b["embedding"],
                            graph,
                            PlannerConfig(
                                cover,
                                routing=routing,
                                revision=2,
                                reference_selection=cfg.get(
                                    "reference_selection", "entry_plus_path"
                                ),
                                preserve_retrieval_candidate=cfg.get(
                                    "preserve_retrieval_candidate", False
                                ),
                                semantic_rollout_terminals=cfg.get(
                                    "semantic_rollout_terminals", False
                                ),
                                use_observed_guidance=cfg.get("use_observed_guidance", False),
                                ground_target_facts=cfg.get("ground_target_facts", False),
                            ),
                            Decoding(**base["assistant_decoding"]),
                            Decoding(**base["simulator_decoding"]),
                            simulator_prompt(smoke_case, "profile"),
                            visible_context(smoke_case),
                            rollout_judge=judge,
                            rollout_scenario=smoke_case,
                        )
                        planner.seed_context = "repair-smoke/profile/" + condition
                        planner.semantic_judging = True
                        rid = f"repair-smoke/{condition}/{h}/{routing}"
                        first = planner.act(initial_state(smoke_case), goals, 17, rid)
                        count = len(b["generation"].requests)
                        again = planner.act(initial_state(smoke_case), goals, 17, rid)

                        def decision(plan):
                            return {k: v for k, v in asdict(plan).items() if k != "compute"}

                        if (
                            decision(first) != decision(again)
                            or len(b["generation"].requests) != count
                            or first.compute["rollout_judgments"]
                            != again.compute["rollout_judgments"]
                        ):
                            raise ValueError("repair exact plan/journal replay failed")
                        if first.compute["recovery_triggered"] != (routing == "always"):
                            raise ValueError("repair routing gate failed")
                        if routing == "always" and planner.cfg.preserve_retrieval_candidate:
                            retrieval = ConditionalPlanner(
                                b["generation"],
                                b["embedding"],
                                graph,
                                planner.cfg.retrieval_control(),
                                Decoding(**base["assistant_decoding"]),
                                visible_context=visible_context(smoke_case),
                            )
                            retrieval.seed_context = planner.seed_context
                            retrieval.semantic_judging = True
                            control_id = rid + "/retrieval-control"
                            retrieval.act(initial_state(smoke_case), goals, 17, control_id)
                            retrieval_candidate = next(
                                r
                                for r in b["generation"].requests
                                if r.request_id == rid + "/candidate/0"
                            )
                            control = next(
                                r
                                for r in b["generation"].requests
                                if r.request_id == control_id + "/final"
                            )
                            if (
                                retrieval_candidate.seed,
                                retrieval_candidate.prompt_sha256,
                                retrieval_candidate.decoding,
                            ) != (
                                control.seed,
                                control.prompt_sha256,
                                control.decoding,
                            ):
                                raise ValueError(
                                    "candidate zero differs from retrieval realization contract"
                                )
                            if (
                                first.compute["generation_requests"]
                                > base["candidates"]
                                * (3 if planner.cfg.semantic_rollout_terminals else 2)
                                * base["lookahead"]
                            ):
                                raise ValueError(
                                    "retrieval candidate exceeded declared rollout budget"
                                )
                        if (
                            planner.cfg.reference_selection == "nearest_entry"
                            and first.reference_id is not None
                            and first.directed_distances[first.reference_id]
                            != min(first.directed_distances.values())
                        ):
                            raise ValueError("nearest-entry H100 reference binding failed")
                        smoke[h + "/" + condition + "/" + routing] = asdict(first)
                planner = ConditionalPlanner(
                    b["generation"],
                    b["embedding"],
                    None,
                    PlannerConfig(0, revision=2),
                    decoding=Decoding(**base["assistant_decoding"]),
                    visible_context=visible_context(cases[0]),
                )
                # One complete real turn verifies shared simulator/outcome judging with live embeddings.
                dialogue = run_dialogue(
                    cases[0],
                    planner,
                    b["generation"],
                    b["embedding"],
                    "profile",
                    17,
                    a.output / "live-dialogue",
                    {"phase": "development-smoke", "method": "TDRL"},
                    turn_budget=1,
                    simulator_decoding=Decoding(**base["simulator_decoding"]),
                    judge=judge,
                )
                atomic_json(
                    a.output / "checks.json",
                    {
                        "status": "passed",
                        "gpu": require_h100_slurm(),
                        "controller_replay": controller,
                        "exact_journal_replay": True,
                        "optimizer_updates": 0,
                        "plans": smoke,
                        "live_dialogue_status": dialogue["status"],
                        "fit_audit": audit,
                    },
                )
            else:
                gate_paths = [a.output.parent / f"smoke-{i}/checks.json" for i in [0, 1]]
                verify_planner_gates(gate_paths, stage.contract)
                vectors_initial = (
                    None
                    if confirmation_payload
                    else (
                        b["embedding"]
                        .embed(
                            [initial_state(s).serialize() for s in calibration],
                            INSTRUCTION,
                            "repair/calibration",
                            "repair/calibration",
                        )
                        .vectors
                    )
                )
                covers = {}
                if confirmation_payload:
                    covers = {"MRNFixed": confirmation_payload["thresholds"]["H_cover"]}
                for h in [] if confirmation_payload else heads:
                    distances = []
                    for i, s in enumerate(calibration):
                        graph, goals = graph_for(h, s, 2)
                        retrieved = graph.retrieve(vectors_initial[i], goals) if graph else []
                        if retrieved:
                            distances.append(retrieved[0][0])
                    if not distances:
                        raise ValueError("no valid coverage calibration")
                    covers[h] = float(np.quantile(distances, cfg["coverage_quantile"]))
                atomic_json(
                    a.output / "thresholds.json",
                    {
                        "legacy": thresholds,
                        "repaired_H_cover": covers,
                        "quantile": cfg["coverage_quantile"],
                        "calibration_ids": [s["id"] for s in calibration],
                        "frozen_from": file_hash(a.confirmation_lock)
                        if confirmation_payload
                        else None,
                    },
                )
                results = []
                for variant in variants:
                    for protocol in cfg["protocols"]:
                        for scenario in cases:
                            if variant == "PPDPP":
                                planner = PPDPPPlanner(
                                    policy,
                                    tokenizer,
                                    b["generation"],
                                    b["embedding"],
                                    visible_context(scenario),
                                    Decoding(**base["assistant_decoding"]),
                                )
                            else:
                                revision, h, method = variant.split("-")
                                revision = 2 if revision == "repair" else 1
                                graph, goals = graph_for(h, scenario, revision)
                                planner = ConditionalPlanner(
                                    b["generation"],
                                    b["embedding"],
                                    graph,
                                    PlannerConfig(
                                        covers[h] if revision == 2 else thresholds[h]["H_cover"],
                                        routing="retrieval_only"
                                        if method == "R"
                                        else "conditional",
                                        revision=revision,
                                        reference_selection=cfg.get(
                                            "reference_selection", "entry_plus_path"
                                        ),
                                        preserve_retrieval_candidate=cfg.get(
                                            "preserve_retrieval_candidate", False
                                        ),
                                        semantic_rollout_terminals=cfg.get(
                                            "semantic_rollout_terminals", False
                                        ),
                                        use_observed_guidance=cfg.get(
                                            "use_observed_guidance", False
                                        ),
                                        ground_target_facts=cfg.get("ground_target_facts", False),
                                    ),
                                    Decoding(**base["assistant_decoding"]),
                                    Decoding(**base["simulator_decoding"]),
                                    visible_context=visible_context(scenario),
                                    rollout_judge=judge,
                                    rollout_scenario=scenario,
                                )
                            result = run_dialogue(
                                scenario,
                                planner,
                                b["generation"],
                                b["embedding"],
                                protocol,
                                17,
                                a.output / "dialogues" / variant / protocol / scenario["id"],
                                diagnostic_contract(
                                    variant,
                                    [] if variant == "PPDPP" else goals,
                                    stage.code,
                                    file_hash(path),
                                    None if variant == "PPDPP" else heads[h].checkpoint_hash,
                                    None
                                    if variant == "PPDPP"
                                    else {
                                        "H_edge": thresholds[h]["H_edge"],
                                        "H_cover": covers[h]
                                        if revision == 2
                                        else thresholds[h]["H_cover"],
                                    },
                                    confirmation_lock_sha256=file_hash(a.confirmation_lock)
                                    if confirmation_payload
                                    else None,
                                ),
                                turn_budget=base["turn_budget"],
                                simulator_decoding=Decoding(**base["simulator_decoding"]),
                                judge=judge,
                            )
                            results.append(
                                {"variant": variant, "protocol": protocol, "record": result}
                            )
                            atomic_json(
                                a.output / "progress.json",
                                {
                                    "complete": len(results),
                                    "expected": len(variants) * len(cfg["protocols"]) * len(cases),
                                    "latest_variant": variant,
                                    "latest_scenario": scenario["id"],
                                },
                            )
                atomic_json(a.output / "records.json", results)
                atomic_json(
                    a.output / "summary.json",
                    {
                        "scope": cfg["scope"],
                        "results": {
                            v + "/" + p: aggregate(
                                [
                                    x["record"]
                                    for x in results
                                    if x["variant"] == v and x["protocol"] == p
                                ]
                            )
                            for v in variants
                            for p in cfg["protocols"]
                        },
                    },
                )
            compute = {**counters(b["generation"], b["embedding"]), **servers.compute_metadata()}
        stage.write(
            "complete",
            provenance_updates={
                "gpu": gpu,
                "dataset": {
                    "dataset": "TopDial",
                    "split": "development",
                    "processed_sha256": file_hash(root / "data/processed/TopDial/dev.jsonl"),
                    "training_bank_index_sha256": file_hash(
                        source / "embeddings/TopDial/bank/index.json"
                    ),
                },
                "seeds": {
                    "root": 17,
                    "protocols": cfg["protocols"],
                    "candidate_streams": "derived_seed with scenario/protocol/turn/purpose/index",
                },
                "models": json.loads((root / "configs/models.lock.json").read_text()),
                "scenarios": {
                    "ids": [s["id"] for s in cases],
                    "calibration_ids": [s["id"] for s in calibration],
                    "scope": "disjoint development confirmation"
                    if confirmation_payload
                    else "development diagnostic; original test lock unchanged",
                    "confirmation_lock_sha256": file_hash(a.confirmation_lock)
                    if confirmation_payload
                    else None,
                    "test_opened": False,
                },
                "prompts": {
                    "assistant_sha256": digest(ASSISTANT_POLICY),
                    "strategy_instructions_sha256": digest(ACTION_INSTRUCTIONS),
                    "embedding_instruction_sha256": digest(INSTRUCTION),
                },
                "checkpoint": {
                    "PPDPP": file_hash(source / "PPDPP-RL/TopDial/checkpoint.pt"),
                    "MRNFixed": file_hash(numerical_source / "MRNFixed/checkpoint.pt")
                    if numerical_source
                    else file_hash(source / "learning/TopDial/MRNFixed/checkpoint.pt"),
                    "IQE": file_hash(numerical_source / "IQE/checkpoint.pt")
                    if numerical_source
                    else file_hash(source / "learning/TopDial/IQE/checkpoint.pt"),
                },
            },
            compute_updates={**compute, "h100_seconds": time.monotonic() - started},
        )
    except Exception as error:
        failed_compute = {"h100_seconds": time.monotonic() - started}
        if "b" in locals():
            failed_compute.update(counters(b["generation"], b["embedding"]))
        stage.write(
            "partial",
            error=str(error),
            provenance_updates={"gpu": gpu},
            compute_updates=failed_compute,
        )
        raise
    heads.clear()
    del optimizer, policy
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
