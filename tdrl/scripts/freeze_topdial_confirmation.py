#!/usr/bin/env python3
"""Freeze only a reproduced paired discovery candidate; never read confirmation outcomes."""

import argparse
import json
from pathlib import Path

from compare_difftod_development import shared_provenance, verified_records

from tdrl.confirmation_lock import freeze_candidate
from tdrl.paired_development import compare_records
from tdrl.provenance import file_hash, validate_manifest
from tdrl.stages import code_provenance


def relative_path(value):
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError("candidate artifact must be repository-relative")
    return path


def reproduce_discovery(root, report):
    if report.get("schema_version") != 1 or len(report["sources"]) != 3:
        raise ValueError("complete discovery source references required")
    sources = []
    for index, ref in enumerate(report["sources"]):
        directory = root / relative_path(ref["path"])
        filename = "results.json" if index == 0 else "records.json"
        stage = "DiffTOD-development-evaluate" if index == 0 else "TopDial-planner-evaluate"
        if (
            file_hash(directory / "manifest.json") != ref["manifest_sha256"]
            or file_hash(directory / filename) != ref["records_sha256"]
        ):
            raise ValueError("discovery source reference checksum mismatch")
        rows, manifest = verified_records(directory, filename, stage)
        sources.append((directory, rows, manifest))
    baseline = sources[0][1]
    groups = {}
    for _, rows, manifest in sources[1:]:
        shared_provenance(manifest, sources[0][2])
        for row in rows:
            if row["variant"] not in {"repair-MRNFixed-R", "repair-MRNFixed-Full"}:
                raise ValueError("unexpected discovery variant")
            groups.setdefault(row["variant"], []).append(row["record"])
    if set(groups) != {"repair-MRNFixed-R", "repair-MRNFixed-Full"}:
        raise ValueError("both MRN candidates required")
    comparisons = {key: compare_records(rows, baseline) for key, rows in groups.items()}
    if comparisons != report["comparisons"]:
        raise ValueError("discovery report does not reproduce from immutable records")
    return sources, comparisons


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("--candidate-config", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    root = Path.cwd()
    code = code_provenance(root)
    if code["dirty"]:
        raise ValueError("candidate freeze requires clean committed source")
    report_path = root / relative_path(args.report)
    sources, comparisons = reproduce_discovery(root, json.loads(report_path.read_text()))
    cfg_path = root / relative_path(args.candidate_config)
    cfg = json.loads(cfg_path.read_text())
    thresholds = []
    heads = []
    for directory, _, manifest in sources[1:]:
        prov = manifest["provenance"]
        if (
            prov["config"]["sha256"] != file_hash(cfg_path)
            or prov["code"]["commit"] != code["commit"]
        ):
            raise ValueError("candidate source/config changed since discovery")
        path = directory / "thresholds.json"
        if file_hash(path) != manifest["outputs"]["thresholds.json"]:
            raise ValueError("discovery thresholds changed")
        t = json.loads(path.read_text())
        thresholds.append(
            {
                "H_edge": t["legacy"]["MRNFixed"]["H_edge"],
                "H_cover": t["repaired_H_cover"]["MRNFixed"],
            }
        )
        heads.append(prov["checkpoint"]["MRNFixed"])
    if thresholds[0] != thresholds[1] or heads[0] != heads[1]:
        raise ValueError("both MRN methods require identical frozen head/thresholds")
    diff = sources[0][2]["provenance"]["checkpoint"]
    if (
        diff["status"] != "fitted_ema_selected_by_development_loss"
        or diff["completed_updates"] != 100001
    ):
        raise ValueError("fixed fully fitted DiffTOD required")
    overlay = root / relative_path(cfg["strategy_overlay"]) / "manifest.json"
    m = json.loads(overlay.read_text())
    validate_manifest(m)
    if m["status"] != "complete" or m["stage"] != "TopDial-semantic-strategy-annotate":
        raise ValueError("complete semantic overlay required")
    for directory, _, manifest in sources[1:]:
        contract_path = directory / "contract.json"
        if file_hash(contract_path) != manifest["outputs"]["contract.json"]:
            raise ValueError("discovery contract checksum mismatch")
        contract = json.loads(contract_path.read_text())
        if contract["strategy_overlay_manifest_sha256"] != file_hash(overlay):
            raise ValueError("strategy overlay changed since discovery")
    for _, _, manifest in sources:
        if manifest["provenance"]["packages"]["lock_sha256"] != file_hash(root / "runtime/uv.lock"):
            raise ValueError("runtime package lock changed since discovery")
    files = {
        "candidate_config": cfg_path,
        "comparison_config": root / relative_path(cfg["comparison_config"]),
        "confirmation_membership": root / "data/manifests/topdial-development-confirmation64.json",
        "strategy_overlay_manifest": overlay,
        "discovery_report": report_path,
    }
    membership = json.loads(files["confirmation_membership"].read_text())
    if (
        membership.get("schema_version") != 1
        or membership["dataset"] != "TopDial"
        or membership["split"] != "dev"
        or membership["root_seed"] != 17
        or membership["population_sha256"]
        != sources[0][2]["provenance"]["dataset"]["processed_sha256"]
        or len(set(membership["scenario_ids"])) != 64
        or set(membership["scenario_ids"]) & set(sources[0][2]["provenance"]["scenarios"]["ids"])
    ):
        raise ValueError("confirmation population/membership must remain disjoint and frozen")
    payload = {
        "schema_version": 1,
        "scope": "TopDial disjoint development confirmation",
        "root_seed": 17,
        "methods": ["TDRL-R", "TDRL", "DiffTOD"],
        "head_type": "MRNFixed",
        "code_commit": code["commit"],
        "package_lock_sha256": file_hash(root / "runtime/uv.lock"),
        "thresholds": thresholds[0],
        "head_sha256": heads[0],
        "difftod_policy_sha256": diff["checkpoint_sha256"],
        "discovery_macro_differences": {
            method: comparisons[variant]["paired_macro_protocol_sr"]["difference"]
            for method, variant in [
                ("TDRL-R", "repair-MRNFixed-R"),
                ("TDRL", "repair-MRNFixed-Full"),
            ]
        },
        "files": {
            key: {"path": str(path.relative_to(root)), "sha256": file_hash(path)}
            for key, path in files.items()
        },
    }
    sha = freeze_candidate(args.output, payload)
    print(
        json.dumps(
            {"status": "frozen", "path": str(args.output), "sha256": sha, "scope": payload["scope"]}
        )
    )


if __name__ == "__main__":
    main()
