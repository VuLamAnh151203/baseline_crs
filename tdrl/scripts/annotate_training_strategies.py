#!/usr/bin/env python3
"""Resumable semantic training-bank annotation, gated by real runtime/replay/ETA."""

import argparse
import json
import time
from pathlib import Path

import torch
from audit_training_strategies import LABELS, PROMPT

from tdrl.backend import require_h100_slurm
from tdrl.contracts import Decoding, Message
from tdrl.planner import INSTRUCTION, derived_seed
from tdrl.provenance import atomic_json, digest, file_hash, validate_manifest
from tdrl.runtime import Servers
from tdrl.stages import Stage, counters
from tdrl.state import SERIALIZATION_HASH
from tdrl.strategy_annotations import strategy_payload


def verified_manifest(directory):
    m = json.loads((directory / "manifest.json").read_text())
    validate_manifest(m)
    if m["status"] != "complete":
        raise ValueError("complete source stage required")
    return m


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--slot", type=int, choices=[0, 1], required=True)
    p.add_argument("--mode", choices=["preflight", "annotate"], required=True)
    p.add_argument("--gate", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    root = Path.cwd()
    cfg_path = root / "configs/topdial-semantic-strategy-v1.json"
    cfg = json.loads(cfg_path.read_text())
    if cfg.get("schema_version") != 1 or cfg["seed"] != 17:
        raise ValueError("unknown semantic strategy annotation contract")
    bank_path = root / cfg["bank"]
    bank_manifest = verified_manifest((root / cfg["bank_manifest"]).parent)
    relative = str(bank_path.relative_to((root / cfg["bank_manifest"]).parent))
    bank_sha = file_hash(bank_path)
    if bank_manifest["outputs"][relative] != bank_sha:
        raise ValueError("training bank changed")
    if file_hash(root / "data/processed/TopDial/train.jsonl") != cfg["train_sha256"]:
        raise ValueError("training population changed")
    bank = json.loads(bank_path.read_text())
    if bank.get("schema_version") != 1 or bank["serialization_sha256"] != SERIALIZATION_HASH:
        raise ValueError("unknown training bank serialization/schema")
    records = [r for r in bank["records"] if r["split"] == "train" and r["guidance"]]
    if len(records) != cfg["expected_records"] or len({r["node_id"] for r in records}) != len(
        records
    ):
        raise ValueError("incomplete/duplicate actionable training bank")
    records.sort(key=lambda r: digest([17, r["node_id"]]))
    gpu = require_h100_slurm()
    stage = Stage(
        root,
        args.output,
        "TopDial-semantic-strategy-" + args.mode,
        cfg_path,
        {"slot": args.slot, "bank_sha256": bank_sha, "prompt_sha256": digest(PROMPT)},
    )
    if stage.reused:
        return
    allocation = json.loads((root / "artifacts/two-gpu-58159/allocation.json").read_text())
    remaining = allocation["walltime_seconds"] - (time.time() - allocation["started_unix"])
    if args.mode == "annotate":
        if args.gate is None:
            raise ValueError("measured annotation preflight required")
        m = verified_manifest(args.gate)
        if m["stage"] != "TopDial-semantic-strategy-preflight" or m["provenance"]["config"][
            "sha256"
        ] != file_hash(cfg_path):
            raise ValueError("annotation preflight contract mismatch")
        if file_hash(args.gate / "checks.json") != m["outputs"]["checks.json"]:
            raise ValueError("annotation gate checksum mismatch")
        gate = json.loads((args.gate / "checks.json").read_text())
        if (
            gate["status"] != "passed"
            or gate["contract"] != stage.contract
            or gate["gpu"]["uuid"] != gpu["uuid"]
        ):
            raise ValueError("annotation runtime/seed/replay/source gate mismatch")
        if gate["required_seconds_with_safety"] > remaining:
            raise ValueError(
                "annotation plus scoped comparison exceeds remaining clock; explicit budget/scope decision required"
            )
    selected = records[: cfg["preflight_count"]] if args.mode == "preflight" else records
    started = time.monotonic()
    compute = {}
    previous = args.output / "manifest.json"
    prior_seconds = 0
    if previous.exists():
        old = json.loads(previous.read_text())
        prior_seconds = old["compute"]["h100_seconds"] or 0
        atomic_json(args.output / f"prior-manifest-{time.time_ns()}.json", old)
    try:
        settings = json.loads((root / "configs/runtime.json").read_text())
        for kind in ["generation", "embedding"]:
            settings[kind]["port"] += args.slot * 100
        output, elapsed = [], []
        with Servers(root, args.output / "servers", settings) as servers:
            b = servers.backends()
            rid = "strategy-annotation-smoke/live"
            emb = b["embedding"].embed([selected[0]["state"]], INSTRUCTION, rid, rid)
            replay = b["embedding"].embed([selected[0]["state"]], INSTRUCTION, rid, rid)
            if emb.vectors.shape != (1, 4096) or not torch.equal(emb.vectors, replay.vectors):
                raise ValueError("real dual-residency live embedding exact replay gate failed")
            for record in selected:
                rid = "semantic-strategy/" + digest(stage.contract) + "/" + record["node_id"]
                messages = [
                    Message("system", PROMPT),
                    Message("user", json.dumps(strategy_payload(record), ensure_ascii=False)),
                ]
                seed = derived_seed(17, rid, "strategy_label")
                begin = time.monotonic()
                result = b["generation"].generate(
                    messages, "strategy_annotation", Decoding(**cfg["decoding"]), seed, rid
                )
                elapsed.append(time.monotonic() - begin)
                if args.mode == "preflight":
                    repeated = b["generation"].generate(
                        messages, "strategy_annotation", Decoding(**cfg["decoding"]), seed, rid
                    )
                    if result != repeated:
                        raise ValueError("semantic strategy exact journal replay failed")
                label = result.text.strip()
                if result.finish_reason == "length" or label not in LABELS:
                    raise ValueError("unknown/truncated semantic strategy; no fallback")
                output.append(
                    {
                        "node_id": record["node_id"],
                        "weak_label": record["action"],
                        "semantic_label": label,
                        "request_id": rid,
                        "prompt_sha256": result.prompt_sha256,
                    }
                )
                if len(output) % cfg["progress_every"] == 0:
                    atomic_json(
                        args.output / "progress.json",
                        {
                            "schema_version": 1,
                            "status": "partial",
                            "complete": len(output),
                            "expected": len(selected),
                            "wall_s": time.monotonic() - started,
                        },
                    )
                if (args.output / "stop.requested").exists():
                    raise ValueError(
                        "annotation stopped; completed request journals retained for exact resume"
                    )
            compute = {**counters(b["generation"], b["embedding"]), **servers.compute_metadata()}
        if list(args.output.rglob("pending-*.json")) or list(args.output.rglob("*.pending.json")):
            raise ValueError("unresolved request outcome; preserve evidence before retry")
        atomic_json(args.output / "labels.json", output)
        pairs = {}
        for row in output:
            key = row["weak_label"] + "->" + row["semantic_label"]
            pairs[key] = pairs.get(key, 0) + 1
        atomic_json(
            args.output / "report.json",
            {
                "schema_version": 1,
                "counts": pairs,
                "nodes": len(output),
                "scope": "training semantic guidance annotation; automatic labels, not human or success evidence",
            },
        )
        if args.mode == "preflight":
            reserved = root / cfg["comparison_reserve_preflight"]
            m = verified_manifest(reserved)
            if file_hash(reserved / "projection.json") != m["outputs"]["projection.json"]:
                raise ValueError("comparison preflight reserve changed")
            reserve = json.loads((reserved / "projection.json").read_text())[
                "required_seconds_with_safety"
            ]
            annotation = (
                max(elapsed) * len(records) * cfg["feasibility_safety_factor"]
                + compute["cold_load_s"]
            )
            required = annotation + reserve
            atomic_json(
                args.output / "checks.json",
                {
                    "schema_version": 1,
                    "status": "passed",
                    "contract": stage.contract,
                    "gpu": gpu,
                    "exact_journal_replay": True,
                    "required_seconds_with_safety": required,
                    "annotation_seconds": annotation,
                    "comparison_reserve_seconds": reserve,
                    "remaining_allocation_seconds": remaining,
                    "sample_requests": len(elapsed),
                    "max_request_seconds": max(elapsed),
                    "scope_limits": "128 frozen training nodes; max-request projection plus conservative original full comparison reserve. Post-fit comparison feasibility still required.",
                },
            )
            if required > remaining:
                raise ValueError(
                    "annotation plus scoped comparison exceeds remaining clock; explicit budget/scope decision required"
                )
        stage.write(
            "complete",
            provenance_updates={
                "gpu": gpu,
                "dataset": {
                    "split": "train",
                    "bank_sha256": bank_sha,
                    "train_sha256": cfg["train_sha256"],
                },
                "seeds": {"root": 17},
                "prompts": {"strategy_sha256": digest(PROMPT)},
            },
            compute_updates={**compute, "h100_seconds": prior_seconds + time.monotonic() - started},
        )
    except Exception as error:
        if "b" in locals():
            compute.update(counters(b["generation"], b["embedding"]))
        stage.write(
            "failed",
            error=error,
            compute_updates={**compute, "h100_seconds": prior_seconds + time.monotonic() - started},
        )
        raise


if __name__ == "__main__":
    main()
