#!/usr/bin/env python3
"""Judge frozen MRN hypothetical user prefixes on an allocated, free H100."""

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import torch

from tdrl.backend import require_h100_slurm
from tdrl.judge import LocalJudge
from tdrl.planner import INSTRUCTION, derived_seed
from tdrl.provenance import atomic_json, file_hash
from tdrl.rollout_audit import audit_state, mrn_branches, point_id, summarize
from tdrl.runtime import Servers
from tdrl.stages import Stage, counters


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/mrn-rollout-terminal-audit-v1.json")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--slot", type=int, choices=[0, 1], required=True)
    args = parser.parse_args()
    root = Path.cwd()
    cfg = json.loads(args.config.read_text())
    if cfg["schema_version"] != 1 or cfg["seed"] != 17:
        raise ValueError("unknown audit configuration")
    source = root / cfg["input"]
    if file_hash(source) != cfg["input_sha256"]:
        raise ValueError("frozen input checksum changed")
    payload = json.loads(source.read_text())
    # Audit only the MRN source records, never load the stopped IQE records.
    source_records = root / "artifacts/topdial-targetpool-debug-v4-58159/eval-0/records.json"
    relative = str(source_records.relative_to(root))
    manifest = json.loads((source_records.parent / "manifest.json").read_text())
    expected = payload["source_sha256"][relative]
    if (
        manifest["status"] != "complete"
        or manifest["outputs"]["records.json"] != expected
        or file_hash(source_records) != expected
    ):
        raise ValueError("MRN source incomplete or changed")
    branches = mrn_branches(payload)
    if (
        len(branches) != cfg["branches"]
        or sum(len(b["points"]) for b in branches) != cfg["prefixes"]
    ):
        raise ValueError("frozen MRN subset changed")
    comparison = root / cfg["comparison_config"]
    if file_hash(comparison) != cfg["comparison_sha256"]:
        raise ValueError("frozen judge configuration changed")
    base = json.loads(comparison.read_text())
    require_h100_slurm()
    stage = Stage(
        root,
        args.output,
        "MRN-rollout-terminal-audit",
        args.config,
        {
            "input_sha256": cfg["input_sha256"],
            "comparison_sha256": cfg["comparison_sha256"],
            "source_records_sha256": expected,
            "slot": args.slot,
        },
    )
    if stage.reused:
        return
    stage.write("partial")
    settings = json.loads((root / "configs/runtime.json").read_text())
    for kind in ["generation", "embedding"]:
        settings[kind]["port"] += args.slot * 100
    try:
        with Servers(root, args.output / "servers", settings) as servers:
            b = servers.backends()
            judge = LocalJudge(b["generation"], base["judge"])
            state, scenario = audit_state(branches[0], branches[0]["points"][0])
            embedding = b["embedding"].embed(
                [state.serialize()], INSTRUCTION, "audit-smoke", "audit-smoke/embedding"
            )
            again = b["embedding"].embed(
                [state.serialize()],
                INSTRUCTION,
                "audit-smoke-replay",
                "audit-smoke/embedding-replay",
            )
            if embedding.vectors.shape != (1, 4096) or not torch.equal(
                embedding.vectors, again.vectors
            ):
                raise ValueError("live full-size embedding exact replay failed")
            first = judge.assess(state, scenario, 17, "audit-smoke/judge")
            if judge.assess(state, scenario, 17, "audit-smoke/judge") != first:
                raise ValueError("judge exact journal replay failed")
            atomic_json(
                args.output / "smoke.json",
                {
                    "status": "passed",
                    "judge_sha256": judge.protocol_hash,
                    "embedding_dimensions": 4096,
                    "exact_embedding_and_journal_replay": True,
                },
            )
            judgments = {}
            for branch in branches:
                for point in branch["points"]:
                    key = point_id(cfg["input_sha256"], branch, point)
                    state, scenario = audit_state(branch, point)
                    judgments[key] = asdict(
                        judge.assess(
                            state, scenario, derived_seed(17, key, "judge"), "audit/" + key
                        )
                    )
                    atomic_json(args.output / "judgments.json", judgments)
            report = summarize(branches, judgments, cfg["input_sha256"])
            report.update(
                judge_protocol_sha256=judge.protocol_hash, input_sha256=cfg["input_sha256"]
            )
            atomic_json(args.output / "report.json", report)
            compute = {**counters(b["generation"], b["embedding"]), **servers.compute_metadata()}
        compute["h100_seconds"] = time.monotonic() - stage.start
        stage.write(
            "complete",
            provenance_updates={
                "dataset": {"name": "TopDial", "split": "dev", "source_records_sha256": expected},
                "scenarios": {
                    "ids": sorted({x["scenario_id"] for x in branches}),
                    "frozen_input_sha256": cfg["input_sha256"],
                },
                "seeds": {"root": 17, "derived": "point-hash then purpose=judge"},
                "prompts": {
                    "judge_protocol_sha256": judge.protocol_hash,
                    "comparison_config_sha256": cfg["comparison_sha256"],
                },
            },
            compute_updates=compute,
        )
    except BaseException as error:
        stage.write("partial", error={"type": type(error).__name__, "message": str(error)})
        raise


if __name__ == "__main__":
    main()
