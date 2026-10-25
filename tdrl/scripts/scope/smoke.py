#!/usr/bin/env python3
"""Real one-turn H100 smoke for SCOPE assets, planning, replay and shared evaluation."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch

from tdrl.baselines.scope.config import ScopeConfig
from tdrl.baselines.scope.planner import ScopeLitePlanner
from tdrl.baselines.scope.reward import GoalConditionedReward, load_reward_checkpoint
from tdrl.baselines.scope.semantic import LlamaGuardSemanticEncoder, NoSharedEmbedding
from tdrl.baselines.scope.transition import TransitionEnsemble
from tdrl.contracts import Decoding, Message
from tdrl.evaluation import run_dialogue
from tdrl.judge import LocalJudge
from tdrl.provenance import atomic_json, digest
from tdrl.runtime import Servers
from tdrl.state import DialogueState


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", choices=("TopDial", "DuRecDial2.0"), required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--comparison-config", type=Path, required=True)
    parser.add_argument("--runtime-config", type=Path, required=True)
    parser.add_argument("--transitions", type=Path, required=True)
    parser.add_argument("--llama-guard", type=Path, required=True)
    parser.add_argument("--llama-guard-manifest", type=Path, required=True)
    parser.add_argument("--reward-checkpoint", type=Path)
    args = parser.parse_args()
    root, output = args.root.resolve(), args.output.resolve()
    config = ScopeConfig.from_path(args.config)
    comparison = json.loads(args.comparison_config.read_text(encoding="utf-8"))
    runtime = json.loads(args.runtime_config.read_text(encoding="utf-8"))
    rows = [
        json.loads(line)
        for line in (root / "data/processed" / args.dataset / "dev.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    scenario = min(rows, key=lambda row: digest([args.dataset, row["id"]]))
    with Servers(root, output / "servers", runtime, kinds=("generation",)) as servers:
        generation = servers.backends()["generation"]
        encoder = LlamaGuardSemanticEncoder(
            args.llama_guard,
            args.llama_guard_manifest,
            output_dim=config.embedding_dim,
            projection_seed=config.projection_seed,
        )
        transitions = TransitionEnsemble(args.transitions, dimension=config.embedding_dim)
        if args.reward_checkpoint:
            reward, reward_info = load_reward_checkpoint(args.reward_checkpoint, config)
            reward_mode = "fitted"
        else:
            torch.manual_seed(17)
            reward = GoalConditionedReward(
                config.embedding_dim, config.reward_hidden_dims
            ).cuda().eval()
            reward_info = {"sha256": "diagnostic-random-untrained"}
            reward_mode = "untrained-runtime-gate-only"
        planner = ScopeLitePlanner(
            generation,
            encoder,
            transitions,
            reward,
            scenario,
            args.dataset,
            config,
            Decoding(**comparison["assistant_decoding"]),
            output / "plans",
        )
        contract = {
            "phase": "scope-h100-smoke",
            "method": "SCOPE-CRS-lite",
            "selection_sha256": "diagnostic",
            "budget": "natural",
            "target_nodes": [],
            "request_namespace": digest(
                ["scope-smoke", args.dataset, config.sha256, reward_info["sha256"]]
            ),
        }
        result = run_dialogue(
            scenario,
            planner,
            generation,
            NoSharedEmbedding(),
            "resistant",
            17,
            output / "dialogue",
            contract,
            turn_budget=1,
            simulator_decoding=Decoding(**comparison["simulator_decoding"]),
            judge=LocalJudge(generation, comparison["judge"]),
        )
        if result["status"] != "complete" or result["actual_turns"] != 1:
            raise ValueError("SCOPE real-dialogue smoke incomplete")
        state = DialogueState((Message("user", "I want a suitable recommendation."),))
        replay_planner = ScopeLitePlanner(
            generation,
            encoder,
            transitions,
            reward,
            scenario,
            args.dataset,
            config,
            Decoding(**comparison["assistant_decoding"]),
            output / "replay-plans",
        )
        first = replay_planner.act(state, [], 17, "scope-smoke/replay-plan")
        requests_after_first = len(generation.requests)
        second = replay_planner.act(state, [], 17, "scope-smoke/replay-plan")
        if asdict(first) != asdict(second) or len(generation.requests) != requests_after_first:
            raise ValueError("SCOPE exact plan/generation replay failed")
        summary = {
            "schema_version": 1,
            "status": "complete",
            "dataset": args.dataset,
            "reward_mode": reward_mode,
            "reward_checkpoint_sha256": reward_info["sha256"],
            "real_dialogue": str(output / "dialogue/dialogue.json"),
            "replay_exact": True,
            "selected_candidate": first.selected_candidate,
            "search_simulations": first.compute["search_simulations"],
            "transition": transitions.provenance(),
            "semantic_encoder": encoder.provenance(),
            "server_compute": servers.compute_metadata(),
            "claim_limit": "runtime/replay diagnostic only",
        }
        atomic_json(output / "summary.json", summary)
        print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
