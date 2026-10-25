#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import torch

from tdrl.baselines.epl.assets import verify_assets
from tdrl.baselines.epl.config import EPLConfig
from tdrl.baselines.epl.contracts import EPLCheckpointMetadata
from tdrl.baselines.epl.dataset import EPLDataset
from tdrl.baselines.epl.model import load_pretrained_policy
from tdrl.baselines.epl.runtime import SOURCE_REVISION, load_tokenizer
from tdrl.baselines.epl.training import EPLTrainer
from tdrl.provenance import atomic_json, digest, file_hash


def main():
    parser = argparse.ArgumentParser(description="Fit one EPL policy and train-only experience memory")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--dataset", choices=("TopDial", "DuRecDial2.0"), required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--encoder", type=Path, required=True)
    parser.add_argument("--retrieval-encoder", type=Path, required=True)
    parser.add_argument("--model-lock", type=Path, required=True)
    parser.add_argument("--max-train-examples", type=int)
    parser.add_argument("--max-dev-examples", type=int)
    args = parser.parse_args()
    root, run_root = args.root.resolve(), args.run_root.resolve()
    config = EPLConfig.from_path(args.config)
    if config.dataset != args.dataset:
        raise ValueError("EPL training dataset/config mismatch")
    verify_assets(args.model_lock, args.encoder, args.retrieval_encoder)
    train_path = root / "data/processed" / args.dataset / "train.jsonl"
    dev_path = root / "data/processed" / args.dataset / "dev.jsonl"
    tokenizer = load_tokenizer(args.encoder)
    training = EPLDataset(train_path, args.dataset, tokenizer, config)
    validation = EPLDataset(dev_path, args.dataset, tokenizer, config)
    for name, dataset, limit in (
        ("train", training, args.max_train_examples),
        ("dev", validation, args.max_dev_examples),
    ):
        if limit is not None:
            if limit < 1 or limit > len(dataset):
                raise ValueError(f"invalid EPL {name} example limit: {limit}")
            dataset.examples = dataset.examples[:limit]
    train_signature = (
        file_hash(train_path)
        if args.max_train_examples is None
        else digest([file_hash(train_path), [row["id"] for row in training.examples]])
    )
    dev_signature = (
        file_hash(dev_path)
        if args.max_dev_examples is None
        else digest([file_hash(dev_path), [row["id"] for row in validation.examples]])
    )
    metadata = EPLCheckpointMetadata(
        schema_version=1,
        dataset=args.dataset,
        config_sha256=config.sha256,
        training_sha256=config.training_sha256,
        train_sha256=train_signature,
        dev_sha256=dev_signature,
        action_taxonomy_sha256=digest(config.actions),
        model_lock_sha256=file_hash(args.model_lock),
        source_revision=SOURCE_REVISION,
    )
    torch.manual_seed(config.training_seed)
    torch.cuda.manual_seed_all(config.training_seed)
    output = run_root / "training" / args.dataset
    result_path = output / "training-result.json"
    if result_path.exists() and (output / "checkpoint-best.pt").exists():
        result = json.loads(result_path.read_text(encoding="utf-8"))
    else:
        model = load_pretrained_policy(args.encoder, config, len(tokenizer), "cuda")
        result = EPLTrainer(
            model, config, metadata, tokenizer.pad_token_id, "cuda"
        ).fit(
            training,
            validation,
            output,
            stop_requested=lambda: (run_root / "stop.requested").exists(),
        )
        del model
        torch.cuda.empty_cache()
    if result.get("status") != "complete":
        raise InterruptedError("EPL training stopped at a resumable checkpoint")
    summary = {
        "schema_version": 1,
        "status": "complete",
        "dataset": args.dataset,
        "training": result,
        "checkpoint_sha256": file_hash(output / "checkpoint-best.pt"),
        "training_examples": len(training),
        "development_examples": len(validation),
        "diagnostic_subset": any(
            value is not None for value in (args.max_train_examples, args.max_dev_examples)
        ),
        "memory": "pending_train_only_self_simulation",
        "test_bodies": "never_opened",
    }
    atomic_json(output / "branch-summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
