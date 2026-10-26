#!/usr/bin/env python3
"""Merge disjoint DPDP dataset evaluation shards after strict provenance checks."""

from __future__ import annotations

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
    "config_sha256",
    "checkpoint_sha256",
    "offline_checkpoint_sha256",
    "dpdp_config_sha256",
    "training_config_sha256",
    "gate_thresholds",
    "search_budgets",
    "asset_lock_sha256",
    "code",
    "qwen_model_lock_sha256",
    "evaluation_seeds",
    "evaluation_budgets",
)


def merge(paths: list[Path], output: Path) -> dict:
    if len(paths) != 2 or len(set(map(str, paths))) != 2:
        raise ValueError("DPDP merge requires exactly two distinct dataset shards")
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    reference = reports[0]
    if reference.get("status") != "complete":
        raise ValueError("DPDP evaluation shard is incomplete")
    for report in reports[1:]:
        if any(report.get(field) != reference.get(field) for field in COMMON_FIELDS):
            raise ValueError("DPDP evaluation shard provenance mismatch")
    datasets = [dataset for report in reports for dataset in report.get("scope_datasets", [])]
    if set(datasets) != {"TopDial", "DuRecDial2.0"} or len(datasets) != 2:
        raise ValueError("DPDP evaluation shards are missing or overlap datasets")
    summary = [row for report in reports for row in report.get("summary", [])]
    budgets = reference.get("evaluation_budgets")
    if (
        not isinstance(budgets, list)
        or not budgets
        or any(budget not in {"natural", "matched"} for budget in budgets)
    ):
        raise ValueError("invalid DPDP evaluation budgets")
    expected = {
        (dataset, protocol, budget)
        for dataset in ("TopDial", "DuRecDial2.0")
        for protocol in ("profile", "resistant")
        for budget in budgets
    }
    observed = {(row.get("dataset"), row.get("protocol"), row.get("budget")) for row in summary}
    if observed != expected or len(summary) != len(expected):
        raise ValueError("merged DPDP evaluation matrix is incomplete")
    gate_evidence = {}
    for report in reports:
        overlap = set(gate_evidence).intersection(report.get("gate_evidence", {}))
        if overlap:
            raise ValueError(f"duplicate DPDP gate evidence: {sorted(overlap)}")
        gate_evidence.update(report.get("gate_evidence", {}))
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
        "summary": sorted(
            summary, key=lambda row: (row["dataset"], row["protocol"], row["budget"])
        ),
        "gate_evidence": gate_evidence,
        "scope_datasets": ["TopDial", "DuRecDial2.0"],
        "runtime_config_sha256_by_dataset": {
            report["scope_datasets"][0]: report["runtime_config_sha256"] for report in reports
        },
        "resume_accounting_by_dataset": {
            report["scope_datasets"][0]: report.get("resume_accounting", {}) for report in reports
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
        if manifests[0] != manifests[1] or manifests[0] is None:
            raise ValueError("DPDP test shards used different frozen samples")
        merged["test_sample"] = manifests[0]
        settings = [report.get("dataset_setting") for report in reports]
        if settings[0] != settings[1]:
            raise ValueError("DPDP test shards used different dataset settings")
        if settings[0] is not None:
            merged["dataset_setting"] = settings[0]
    atomic_json(output, merged)
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(merge(args.input, args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
