#!/usr/bin/env python3
"""Real H100 candidate scheduling benchmark and exact cached replay for both heads."""

import argparse
import json
import os
import time
from dataclasses import asdict
from pathlib import Path

from tdrl.backend import LocalBackend, require_h100_slurm
from tdrl.contracts import Decoding
from tdrl.evaluation import initial_state, simulator_prompt, visible_context
from tdrl.fidelity import replay_policy
from tdrl.judge import LocalJudge
from tdrl.planner import ConditionalPlanner, PlannerConfig
from tdrl.policy_optimizer import policy_optimizer
from tdrl.policy_training import load_policy
from tdrl.provenance import Journal, atomic_json, digest, file_hash
from tdrl.runtime import Servers
from tdrl.training import load_embeddings, load_head, load_prepared, make_graph

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--output", type=Path, required=True)
p.add_argument("--shared", type=Path, required=True)
p.add_argument("--slot", type=int, choices=[0, 1], required=True)
p.add_argument("--checkpoint", type=Path, required=True)
a = p.parse_args()
root = Path.cwd()
cfg = json.loads((root / "configs/comparison-judge-v2-sample500-seed17.json").read_text())
settings = json.loads((root / "configs/runtime.json").read_text())
for kind in ["generation", "embedding"]:
    settings[kind]["port"] += a.slot * 100
policy, tokenizer, _ = load_policy(root, a.checkpoint)
optimizer = policy_optimizer(policy, cfg["ppdpp"], "rl")
optimizer.reserve()
controller = replay_policy(policy, tokenizer, load_prepared(root, "DuRecDial2.0", "train")[0])
scenario = sorted(load_prepared(root, "TopDial", "dev"), key=lambda s: digest([17, s["id"]]))[0]
state = initial_state(scenario)
records, vectors = load_embeddings(a.shared / "embeddings/TopDial/bank")
heads = {}
with Servers(root, a.output / "servers", settings) as servers:
    b = servers.backends()
    for kind in ["MRNFixed", "IQE"]:
        head = load_head(a.shared / f"learning/TopDial/{kind}/checkpoint.pt")
        graph, goals = make_graph(
            records, vectors, head, 1000.0, cfg["node_budget"], 17, scenario["target"][0]
        )
        if not goals:
            raise ValueError("branch smoke requires training-backed goals")
        measurements = {}
        for workers in [1, 3]:
            directory = a.output / kind / str(workers)
            backend = LocalBackend(
                b["generation"].url,
                b["generation"].model,
                "generation",
                Journal(directory / "journal", asdict(b["generation"].model)),
                directory / "ledger",
            )
            planner = ConditionalPlanner(
                backend,
                b["embedding"],
                graph,
                PlannerConfig(0.0, cfg["candidates"], cfg["lookahead"]),
                Decoding(**cfg["assistant_decoding"]),
                Decoding(**cfg["simulator_decoding"]),
                simulator_prompt(scenario, "profile"),
                visible_context(scenario),
            )
            planner.semantic_judging = True
            planner.seed_context = f"{scenario['id']}/profile/0"
            os.environ["TDRL_CANDIDATE_WORKERS"] = str(workers)
            start = time.monotonic()
            plan = planner.act(state, goals, 17, f"branch-smoke/{kind}/turn")
            wall = time.monotonic() - start
            if (
                not plan.compute["recovery_triggered"]
                or len(plan.rollout_scores) != 3
                or not 0 <= plan.selected_candidate < 3
                or len(backend.requests) > 12
                or plan.compute["embedding_requests"] != 4
            ):
                raise ValueError("branch smoke violated frozen routing/budget")
            atomic_json(directory / "plan.json", asdict(plan))
            if workers == 3:
                count = len(backend.requests)
                again = planner.act(state, goals, 17, f"branch-smoke/{kind}/turn")

                def decision(value):
                    return {k: v for k, v in asdict(value).items() if k != "compute"}

                if decision(plan) != decision(again) or len(backend.requests) != count:
                    raise ValueError("parallel branches failed exact cached plan replay")
            measurements[str(workers)] = {
                "wall_s": wall,
                "generation_requests": len(backend.requests),
                "seed_by_request": {r.request_id: r.seed for r in backend.requests},
            }
            backend.client.close()
        common = (
            measurements["1"]["seed_by_request"].keys()
            & measurements["3"]["seed_by_request"].keys()
        )
        if any(
            measurements["1"]["seed_by_request"][rid] != measurements["3"]["seed_by_request"][rid]
            for rid in common
        ):
            raise ValueError("candidate seeds changed")
        heads[kind] = {"checkpoint_sha256": head.checkpoint_hash, "measurements": measurements}
        del graph, head
    judge = LocalJudge(b["generation"], cfg["judge"])
    os.environ["TDRL_CRITIC_WORKERS"] = "10"
    feedback = judge.feedback(state, scenario, 17, "branch-smoke/critic")
    count = len(b["generation"].requests)
    if (
        feedback != judge.feedback(state, scenario, 17, "branch-smoke/critic")
        or len(b["generation"].requests) != count
        or count != 10
    ):
        raise ValueError("inherited critic exact replay failed")
    compute = servers.compute_metadata()
if list(a.output.rglob("pending-*.json")):
    raise ValueError("unresolved branch smoke request")
atomic_json(
    a.output / "checks.json",
    {
        "schema_version": 1,
        "status": "passed",
        "gpu": require_h100_slurm(),
        "heads": heads,
        "exact_journal_replay": True,
        "critic_exact_replay": True,
        "pending_requests": 0,
        "optimizer_updates": 0,
        "controller_replay": controller,
        "checkpoint_sha256": file_hash(a.checkpoint),
        "compute": compute,
        "serial_parallel_text_equality_claim": False,
    },
)
