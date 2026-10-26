#!/usr/bin/env python3
"""Build train/dev SCOPE reward examples on one allocated H100."""

import argparse
import json
from pathlib import Path

from tdrl.baselines.scope.config import ScopeConfig
from tdrl.baselines.scope.dataset import prepare_split
from tdrl.baselines.scope.semantic import LlamaGuardSemanticEncoder
from tdrl.judge import LocalJudge
from tdrl.provenance import atomic_json
from tdrl.runtime import Servers


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--dataset", choices=("TopDial", "DuRecDial2.0"), required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--comparison-config", type=Path, required=True)
    parser.add_argument("--runtime-config", type=Path, required=True)
    parser.add_argument("--llama-guard", type=Path, required=True)
    parser.add_argument("--llama-guard-manifest", type=Path, required=True)
    args = parser.parse_args()
    root, run_root = args.root.resolve(), args.run_root.resolve()
    config = ScopeConfig.from_path(args.config)
    if config.dataset != args.dataset:
        raise ValueError("SCOPE reward preparation dataset/config mismatch")
    comparison = json.loads(args.comparison_config.read_text(encoding="utf-8"))
    runtime = json.loads(args.runtime_config.read_text(encoding="utf-8"))
    output = run_root / "reward-data" / args.dataset
    with Servers(root, output / "servers", runtime, kinds=("generation",)) as servers:
        generation = servers.backends()["generation"]
        judge = LocalJudge(generation, comparison["judge"])
        encoder = LlamaGuardSemanticEncoder(
            args.llama_guard,
            args.llama_guard_manifest,
            output_dim=config.embedding_dim,
            projection_seed=config.projection_seed,
        )
        manifests = {}
        for split, limit in (
            ("train", config.max_train_dialogues),
            ("dev", config.max_dev_dialogues),
        ):
            manifests[split] = prepare_split(
                root / "data/processed" / args.dataset / f"{split}.jsonl",
                output / split,
                args.dataset,
                split,
                encoder,
                judge if config.terminal_semantic_judge else None,
                config.training_seed,
                config.counterfactual_negatives,
                limit,
                stop_requested=lambda: (run_root / "stop.requested").exists(),
            )
        summary = {
            "schema_version": 1,
            "status": "complete",
            "dataset": args.dataset,
            "config_sha256": config.sha256,
            "semantic_encoder": encoder.provenance(),
            "manifests": manifests,
            "official_test_opened": False,
        }
        atomic_json(output / "summary.json", summary)
        print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
