#!/usr/bin/env python3
"""Audit method-specific warm smoke proxies before any development matrix launch."""

import argparse
import json
import time
from pathlib import Path

from tdrl.feasibility import full_turn_dialogue_proxy, matrix_projection, planner_turn_proxy
from tdrl.provenance import atomic_json, file_hash, validate_manifest


def read_gate(directory, stage, names):
    manifest = json.loads((directory / "manifest.json").read_text())
    validate_manifest(manifest)
    gpu = manifest["provenance"]["gpu"]
    if (
        manifest["status"] != "complete"
        or manifest["stage"] != stage
        or gpu.get("count") != 1
        or "H100" not in gpu.get("model", "")
    ):
        raise ValueError("Complete actual H100 gate required")
    values = {}
    for name in names:
        if file_hash(directory / name) != manifest["outputs"].get(name):
            raise ValueError("Smoke evidence checksum mismatch")
        values[name] = json.loads((directory / name).read_text())
    if values["checks.json"]["status"] != "passed":
        raise ValueError("Smoke failed")
    return manifest, values


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--difftod-gates", type=Path, nargs=2, required=True)
    p.add_argument("--v8-gates", type=Path, nargs=2, required=True)
    p.add_argument("--v9-gates", type=Path, nargs=2, required=True)
    p.add_argument("--allocation", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    root = Path.cwd()
    base = json.loads((root / "configs/comparison-judge-v2-sample500-seed17.json").read_text())
    native = json.loads((root / "configs/difftod-topdial-fit-v1.json").read_text())
    development = json.loads((root / "configs/difftod-development-v1.json").read_text())
    sources = []
    commits = set()
    packages = set()
    model_contracts = []
    diff = []
    methods = {}
    for paths, kind in [
        (args.difftod_gates, "DiffTOD"),
        (args.v8_gates, "v8"),
        (args.v9_gates, "v9"),
    ]:
        uuids = set()
        for path in paths:
            stage = "DiffTOD-development-smoke" if kind == "DiffTOD" else "TopDial-planner-smoke"
            names = ["checks.json", "contract.json"] + (
                [] if kind == "DiffTOD" else ["live-dialogue/dialogue.json"]
            )
            manifest, values = read_gate(path, stage, names)
            commits.add(manifest["provenance"]["code"]["commit"])
            packages.add(manifest["provenance"]["packages"]["lock_sha256"])
            model_contracts.append(manifest["provenance"]["models"])
            uuids.add(manifest["provenance"]["gpu"]["uuid"])
            sources.append(
                {"path": str(path), "manifest_sha256": file_hash(path / "manifest.json")}
            )
            if kind == "DiffTOD":
                if manifest["provenance"]["config"]["sha256"] != file_hash(
                    root / "configs/difftod-development-v1.json"
                ):
                    raise ValueError("Outdated DiffTOD timing estimator/config")
                for file in sorted((path / "dialogues").glob("*/*/dialogue.json")):
                    relative = str(file.relative_to(path))
                    if file_hash(file) != manifest["outputs"].get(relative):
                        raise ValueError("DiffTOD dialogue timing checksum mismatch")
                    dialogue = json.loads(file.read_text())
                    trajectory = dialogue["turns_trace"][0]["plan"]["compute"]["diffusion"]
                    one_time = (
                        trajectory["policy_sampling_wall_s"]
                        + trajectory["guidance_record"]["latency_s"]
                    )
                    diff.append(
                        full_turn_dialogue_proxy(
                            dialogue["wall_s"],
                            dialogue["actual_turns"],
                            base["turn_budget"],
                            one_time,
                        )
                    )
            else:
                if manifest["provenance"]["config"]["sha256"] != file_hash(
                    root / f"configs/topdial-semantic-strategy-debug-{kind}.json"
                ):
                    raise ValueError("Planner configuration mismatch")
                dialogue = values["live-dialogue/dialogue.json"]
                overhead = max(
                    0,
                    (
                        dialogue["wall_s"]
                        - sum(
                            t["plan"]["compute"]["planner_latency_s"]
                            for t in dialogue["turns_trace"]
                        )
                    )
                    / dialogue["actual_turns"],
                )
                for key, plan in values["checks.json"]["plans"].items():
                    method = "Full" if key.endswith("/always") else "R"
                    proxy = planner_turn_proxy(plan["compute"], overhead) * base["turn_budget"]
                    tag = kind + "-" + method
                    methods[tag] = max(methods.get(tag, 0), proxy)
        if len(uuids) != 2:
            raise ValueError("Each method needs both physical GPU measurements")
    if (
        len(commits) != 1
        or packages != {file_hash(root / "runtime/uv.lock")}
        or any(x != model_contracts[0] for x in model_contracts)
    ):
        raise ValueError("Mixed smoke code/package/model contracts")
    allocation = json.loads(args.allocation.read_text())
    if str(allocation["job_id"]) != "58159":
        raise ValueError("Wrong allocation clock")
    remaining = allocation["walltime_seconds"] - (time.time() - allocation["started_unix"])
    result = matrix_projection(
        max(diff),
        methods,
        remaining,
        reserve=native["evaluation_dialogue_reserve"],
        extra_candidate_dialogues=development["additional_discovery_dialogue_reserve"],
        io_reserve=native["checkpoint_io_reserve_seconds"],
        safety=native["feasibility_safety_factor"],
    )
    atomic_json(
        args.output,
        {
            "schema_version": 1,
            "status": "passed" if result["fits"] else "failed",
            "scope": "TopDial development goal and retained original matrix reserve; no official-test authorization",
            "sources": sources,
            "inference_commit": next(iter(commits)),
            "estimator_sha256": file_hash(Path(__file__)),
            "limits": "Few smoke states; full unsuccessful turn limit; planner call-count scaling is a conservative proxy, not a measured matrix throughput guarantee.",
            **result,
        },
    )
    if not result["fits"]:
        raise ValueError(
            "Scope exceeds remaining allocation; explicit budget/scope decision required"
        )


if __name__ == "__main__":
    main()
