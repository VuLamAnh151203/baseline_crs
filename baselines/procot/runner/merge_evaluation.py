#!/usr/bin/env python3
"""Merge the two immutable ProCoT dataset evaluation shards."""

import argparse
import json
from pathlib import Path

from tdrl.provenance import atomic_json, file_hash

COMMON_FIELDS = (
    "schema_version",
    "status",
    "split",
    "method",
    "selection_sha256",
    "comparison_config_sha256",
    "source_lock_sha256",
    "prompt_sha256",
    "qwen_model_lock_sha256",
    "policy_contract",
    "code",
    "evaluation_seeds",
    "evaluation_budgets",
)


def merge(paths, output):
    if len(paths) != 2 or len(set(map(str, paths))) != 2:
        raise ValueError("ProCoT merge requires two distinct dataset shards")
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    reference = reports[0]
    if reference.get("status") != "complete":
        raise ValueError("ProCoT evaluation shard is incomplete")
    for report in reports[1:]:
        if report.get("status") != "complete" or any(
            report.get(field) != reference.get(field) for field in COMMON_FIELDS
        ):
            raise ValueError("ProCoT evaluation shards have different provenance")
    if any(len(report.get("scope_datasets", [])) != 1 for report in reports):
        raise ValueError("ProCoT merge accepts single-dataset shards only")
    datasets = [report["scope_datasets"][0] for report in reports]
    if set(datasets) != {"TopDial", "DuRecDial2.0"}:
        raise ValueError("ProCoT evaluation shards do not cover both datasets")
    summary = [row for report in reports for row in report["summary"]]
    expected = {
        (dataset, protocol, "natural")
        for dataset in ("TopDial", "DuRecDial2.0")
        for protocol in ("profile", "resistant")
    }
    observed = {
        (row.get("dataset"), row.get("protocol"), row.get("budget"))
        for row in summary
    }
    if observed != expected or len(summary) != len(expected):
        raise ValueError("merged ProCoT evaluation matrix is incomplete")
    merged = {
        **{field: reference[field] for field in COMMON_FIELDS},
        "dialogues": sum(int(report["dialogues"]) for report in reports),
        "wall_s": max(float(report["wall_s"]) for report in reports),
        "parallel_worker_wall_s": {
            report["scope_datasets"][0]: float(report["wall_s"]) for report in reports
        },
        "server_compute": {
            report["scope_datasets"][0]: report["server_compute"] for report in reports
        },
        "summary": sorted(summary, key=lambda row: (row["dataset"], row["protocol"])),
        "scope_datasets": ["TopDial", "DuRecDial2.0"],
        "runtime_config_sha256_by_dataset": {
            report["scope_datasets"][0]: report["runtime_config_sha256"]
            for report in reports
        },
        "shards": [
            {
                "dataset": report["scope_datasets"][0],
                "path": str(path),
                "sha256": file_hash(path),
            }
            for path, report in zip(paths, reports, strict=True)
        ],
    }
    if reference["split"] == "test":
        manifests = [report.get("test_sample") for report in reports]
        settings = [report.get("dataset_setting") for report in reports]
        if manifests[0] != manifests[1] or manifests[0] is None:
            raise ValueError("ProCoT test shards used different fixed samples")
        if settings[0] != settings[1] or settings[0] is None:
            raise ValueError("ProCoT test shards used different fixed settings")
        merged["test_sample"] = manifests[0]
        merged["dataset_setting"] = settings[0]
    atomic_json(output, merged)
    return merged


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(merge(args.input, args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
