#!/usr/bin/env python3
"""Train one dataset-specific SCOPE-CRS-lite reward head."""

import argparse
import json
from pathlib import Path

from tdrl.baselines.scope.assets import (
    verify_llama_guard_snapshot,
    verify_transition_manifest,
)
from tdrl.baselines.scope.config import ScopeConfig
from tdrl.baselines.scope.dataset import RewardDataset
from tdrl.baselines.scope.reward import RewardMetadata
from tdrl.baselines.scope.training import RewardTrainer
from tdrl.provenance import atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--dataset", choices=("TopDial", "DuRecDial2.0"), required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--transitions", type=Path, required=True)
    parser.add_argument("--llama-guard", type=Path, required=True)
    parser.add_argument("--llama-guard-manifest", type=Path, required=True)
    args = parser.parse_args()
    run_root = args.run_root.resolve()
    config = ScopeConfig.from_path(args.config)
    if config.dataset != args.dataset:
        raise ValueError("SCOPE reward training dataset/config mismatch")
    transition = verify_transition_manifest(args.transitions)
    llama = verify_llama_guard_snapshot(args.llama_guard, args.llama_guard_manifest)
    training = RewardDataset(run_root / "reward-data" / args.dataset / "train")
    development = RewardDataset(run_root / "reward-data" / args.dataset / "dev")
    metadata = RewardMetadata(
        schema_version=1,
        method="SCOPE-CRS-lite",
        dataset=args.dataset,
        config_sha256=config.sha256,
        train_manifest_sha256=training.manifest_sha256,
        dev_manifest_sha256=development.manifest_sha256,
        llama_guard_manifest_sha256=llama["manifest_sha256"],
        transition_manifest_sha256=transition["sha256"],
        architecture="GoalConditionedReward([state,goal,abs(state-goal)])",
        selection_rule="minimum development BCE; train/dev only",
    )
    output = run_root / "reward-training" / args.dataset
    result = RewardTrainer(config, metadata).fit(
        training,
        development,
        output,
        stop_requested=lambda: (run_root / "stop.requested").exists(),
    )
    summary = {
        "schema_version": 1,
        "status": "complete",
        "dataset": args.dataset,
        "result": result,
        "metadata": metadata.__dict__,
        "official_test_opened": False,
    }
    atomic_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
