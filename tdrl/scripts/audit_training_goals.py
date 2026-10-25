#!/usr/bin/env python3
"""Read-only inventory of independently judged goals in existing PPDPP training."""

import argparse
import json
from collections import Counter
from pathlib import Path

from tdrl.judge import LocalJudge
from tdrl.provenance import atomic_json, digest, file_hash, validate_manifest
from tdrl.training_goal_audit import audited_terminal


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, default=Path("artifacts/two-gpu-58159/PPDPP-RL/TopDial")
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("retain existing audit; use a fresh output path")
    source = args.source
    manifest = json.loads((source / "manifest.json").read_text())
    validate_manifest(manifest)
    if manifest["status"] != "complete":
        raise ValueError("source fitting is not complete")

    def verified(name):
        path = source / name
        if file_hash(path) != manifest["outputs"].get(name):
            raise ValueError("source checksum mismatch: " + name)
        return json.loads(path.read_text())

    schedule = verified("sampling.json")
    if schedule["seed"] != 17 or len(schedule["episode_scenario_ids"]) != 1000:
        raise ValueError("wrong fitting root seed or episode budget")
    train_path = Path("data/processed/TopDial/train.jsonl")
    data_manifest = json.loads(Path("data/processed/data-manifest.json").read_text())
    if file_hash(train_path) != data_manifest["datasets"]["TopDial"]["prepared_sha256"]["train"]:
        raise ValueError("official training content changed")
    train = [json.loads(line) for line in train_path.read_text().splitlines()]
    by_id = {s["id"]: s for s in train}
    if len(by_id) != len(train) or digest(sorted(by_id)) != schedule["training_ids_sha256"]:
        raise ValueError("training membership differs from fitting schedule")
    comparison = Path("configs/comparison-judge-v2-sample500-seed17.json")
    cfg = json.loads(comparison.read_text())
    models = json.loads(Path("configs/models.lock.json").read_text())
    if manifest["provenance"]["models"] != models:
        raise ValueError("model provenance differs from frozen source")
    judge_hash = LocalJudge(None, cfg["judge"]).protocol_hash
    goals, hashes = [], {}
    for episode, scenario_id in enumerate(schedule["episode_scenario_ids"]):
        if scenario_id not in by_id:
            raise ValueError("non-training scenario in source")
        name = f"episodes/{episode:06d}/dialogue.json"
        record = verified(name)
        if record["seed"] != 17 + episode or record["protocol"] != (
            "profile" if episode % 2 == 0 else "resistant"
        ):
            raise ValueError("training RNG/protocol schedule mismatch")
        goal = audited_terminal(by_id[scenario_id], record, judge_hash)
        hashes[name] = manifest["outputs"][name]
        if goal is not None:
            goals.append({**goal, "episode": episode, "dialogue_sha256": hashes[name]})
    report = {
        "schema_version": 1,
        "status": "complete",
        "scope": "training-only goal evidence inventory; zero inference/refitting or bank changes",
        "source": str(source),
        "source_manifest_sha256": file_hash(source / "manifest.json"),
        "training_content_sha256": file_hash(train_path),
        "comparison_sha256": file_hash(comparison),
        "judge_protocol_sha256": judge_hash,
        "episodes": 1000,
        "successful_episodes": len(goals),
        "unique_successful_scenarios": len({g["scenario_id"] for g in goals}),
        "unique_successful_targets": len({tuple(g["target"]) for g in goals}),
        "outcome_counts": dict(Counter(g["outcome_label"] for g in goals)),
        "source_dialogue_sha256": hashes,
        "goals": goals,
        "limitation": "Online PPDPP-generated training experience is a different bank resource from official offline conversations. Reuse requires an explicit new experiment contract and compute disclosure, not silent replacement.",
    }
    atomic_json(args.output, report)


if __name__ == "__main__":
    main()
