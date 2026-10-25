#!/usr/bin/env python3
"""Gated resumable native-budget TopDial diffusion fitting; no test-resource access."""

import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from preflight_difftod import cpu_state, fingerprint

from tdrl.backend import require_h100_slurm
from tdrl.diffusion_model import load_initial_policy
from tdrl.diffusion_optimizer import DiffusionOptimizer
from tdrl.diffusion_resume import restore_optimizer
from tdrl.diffusion_training import ChunkBatches, ema_policy, policy_loss, validation_loss
from tdrl.provenance import atomic_json, digest, file_hash, validate_manifest
from tdrl.stages import GpuMonitor, Stage


def save_checkpoint(path, value):
    temporary = path.with_suffix(".part")
    torch.save(cpu_state(value), temporary)
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main(config="configs/difftod-topdial-fit-v1.json", dataset="TopDial"):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--preflight", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    root = Path.cwd()
    cfg_path = root / config
    cfg = json.loads(cfg_path.read_text())
    if cfg.get("schema_version") != 1:
        raise ValueError("unknown diffusion fitting schema")
    gpu = require_h100_slurm()
    gate = json.loads((args.preflight / "manifest.json").read_text())
    validate_manifest(gate)
    if gate["status"] != "complete" or gate["provenance"]["config"]["sha256"] != file_hash(
        cfg_path
    ):
        raise ValueError("fresh native-budget preflight required")
    for name, sha in gate["outputs"].items():
        if file_hash(args.preflight / name) != sha:
            raise ValueError("preflight output checksum mismatch")
    old = gate["provenance"]["code"]["commit"]
    modules = [
        "diffusion",
        "diffusion_model",
        "diffusion_optimizer",
        "diffusion_training",
        "difftod",
        "backend",
        "evaluation",
        "runtime",
        "judge",
    ]
    for name in modules:
        relative = "tdrl/src/tdrl/" + name + ".py"
        previous = subprocess.check_output(["git", "show", old + ":" + relative], cwd=root)
        if previous != (root / "src/tdrl" / (name + ".py")).read_bytes():
            raise ValueError("preflight controller/inference module changed: " + name)
    projection = json.loads((args.preflight / "projection.json").read_text())
    replay = json.loads((args.preflight / "update-replay.json").read_text())
    if not projection["fits"] or replay["status"] != "passed":
        raise ValueError("native policy update/feasibility gate failed")
    allocation = json.loads((root / "artifacts/two-gpu-58159/allocation.json").read_text())
    remaining = allocation["walltime_seconds"] - (time.time() - allocation["started_unix"])
    if projection["required_seconds_with_safety"] > remaining:
        raise ValueError(
            "measured scope no longer fits allocation clock; explicit budget decision required"
        )
    data = root / cfg["data_preparation"]
    validation = root / cfg["validation_preparation"]
    for directory in [data, validation]:
        manifest = json.loads((directory / "manifest.json").read_text())
        if manifest.get("schema_version") != 1 or manifest["status"] != "complete":
            raise ValueError("unverified fitting input")
        for name, sha in manifest["outputs"].items():
            if file_hash(directory / name) != sha:
                raise ValueError("changed fitting input")
    previous_compute = 0.0
    previous_manifest = args.output / "manifest.json"
    if previous_manifest.exists():
        previous_compute = json.loads(previous_manifest.read_text())["compute"]["h100_seconds"]
    stage = Stage(
        root,
        args.output,
        f"DiffTOD-{dataset}-fitting",
        cfg_path,
        {
            "preflight_manifest_sha256": file_hash(args.preflight / "manifest.json"),
            "training_data_sha256": file_hash(data / "manifest.json"),
            "validation_data_sha256": file_hash(validation / "manifest.json"),
        },
    )
    if stage.reused:
        return
    started = time.monotonic()
    checkpoint = args.output / "checkpoint.pt"
    index = args.output / "checkpoint.json"
    try:
        with GpuMonitor(args.output / "gpu-telemetry.jsonl") as monitor:
            torch.use_deterministic_algorithms(True)
            torch.manual_seed(17)
            torch.cuda.manual_seed_all(17)
            model, provenance = load_initial_policy(root)
            optimizer = DiffusionOptimizer(model, cfg["optimizer"])
            train = np.load(data / "train.npy", mmap_mode="r", allow_pickle=False)
            val = np.load(validation / "validation.npy", mmap_mode="r", allow_pickle=False)
            sampler = ChunkBatches(len(train), cfg["batch_size"], 17)
            generator = torch.Generator(device="cuda").manual_seed(17)
            best, selections, last_validation = float("inf"), [], None
            if not checkpoint.exists() and list(args.output.glob("training-segment-*.jsonl")):
                raise ValueError(
                    "uncommitted fitting history without checkpoint; preserve before fresh namespace"
                )
            if checkpoint.exists():
                saved_index = json.loads(index.read_text())
                if file_hash(checkpoint) != saved_index["sha256"]:
                    raise ValueError("durable fitting checkpoint mismatch")
                saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
                if saved["schema_version"] != 1 or saved["contract"] != stage.contract:
                    raise ValueError("fitting resume contract mismatch")
                model.load_state_dict(saved["model"])
                restore_optimizer(optimizer, saved["optimizer"])
                sampler.load_state_dict(saved["sampler"])
                generator.set_state(saved["generator"])
                torch.set_rng_state(saved["cpu_rng"])
                torch.cuda.set_rng_state_all(saved["cuda_rng"])
                best, selections = saved["best_loss"], saved["selections"]
                sampler_before = sampler.state_dict()
                replay_ids = sampler.next()
                sampler.load_state_dict(sampler_before)
                replay_batch = torch.tensor(
                    train[replay_ids].copy(), device="cuda", dtype=torch.long
                )
                rows = []
                for _ in range(2):
                    model.load_state_dict(saved["model"])
                    restore_optimizer(optimizer, saved["optimizer"])
                    generator.set_state(saved["generator"])
                    torch.set_rng_state(saved["cpu_rng"])
                    torch.cuda.set_rng_state_all(saved["cuda_rng"])
                    model.train()
                    optimizer.zero_grad()
                    replay_loss = policy_loss(model, replay_batch, generator)
                    replay_loss.backward()
                    gradient = fingerprint(
                        {
                            name: parameter.grad
                            for name, parameter in model.named_parameters()
                            if parameter.grad is not None
                        }
                    )
                    optimizer.step()
                    rows.append(
                        {
                            "loss": float(replay_loss.detach()),
                            "gradient": gradient,
                            "optimizer_ema": fingerprint(optimizer.state_dict()),
                        }
                    )
                if rows[0] != rows[1]:
                    raise ValueError("durable DiffTOD checkpoint exact next-update replay failed")
                atomic_json(
                    args.output / f"resume-replay-{optimizer.updates - 1:06d}.json",
                    {
                        "schema_version": 1,
                        "status": "passed",
                        "checkpoint_sha256": saved_index["sha256"],
                        "replay": rows[0],
                    },
                )
                model.load_state_dict(saved["model"])
                restore_optimizer(optimizer, saved["optimizer"])
                generator.set_state(saved["generator"])
                torch.set_rng_state(saved["cpu_rng"])
                torch.cuda.set_rng_state_all(saved["cuda_rng"])
                optimizer.zero_grad()
                del replay_batch, replay_loss, saved

            def durable():
                save_checkpoint(
                    checkpoint,
                    {
                        "schema_version": 1,
                        "contract": stage.contract,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "sampler": sampler.state_dict(),
                        "generator": generator.get_state(),
                        "cpu_rng": torch.get_rng_state(),
                        "cuda_rng": torch.cuda.get_rng_state_all(),
                        "best_loss": best,
                        "selections": selections,
                    },
                )
                atomic_json(
                    index,
                    {
                        "schema_version": 1,
                        "sha256": file_hash(checkpoint),
                        "updates": optimizer.updates,
                    },
                )

            segment = len(list(args.output.glob("training-segment-*.jsonl")))
            atomic_json(
                args.output / f"segment-{segment:03d}.json",
                {
                    "schema_version": 1,
                    "segment": segment,
                    "starting_committed_update": optimizer.updates,
                    "source_checkpoint_sha256": file_hash(checkpoint)
                    if checkpoint.exists()
                    else None,
                    "rule": "prior uncommitted log tails retained; optimizer/RNG resumes only durable checkpoint",
                },
            )
            with (args.output / f"training-segment-{segment:03d}.jsonl").open("a") as log:
                while optimizer.updates < cfg["updates"]:
                    model.train()
                    optimizer.zero_grad()
                    ids = sampler.next()
                    clean = torch.tensor(train[ids].copy(), device="cuda", dtype=torch.long)
                    loss = policy_loss(model, clean, generator)
                    loss.backward()
                    optimizer.step()
                    record = {
                        "update": optimizer.updates,
                        "loss": float(loss.detach()),
                        "epoch": sampler.epoch,
                        "batch_indices_sha256": digest(ids.tolist()),
                    }
                    if (
                        optimizer.updates % cfg["validation_every"] == 0
                        or optimizer.updates == cfg["updates"]
                    ):
                        with ema_policy(model, optimizer):
                            last_validation = validation_loss(model, val, cfg["batch_size"], 17)
                            if last_validation < best:
                                best = last_validation
                                candidate = (
                                    args.output / "candidates" / f"ema-{optimizer.updates:06d}.pt"
                                )
                                candidate.parent.mkdir(exist_ok=True)
                                save_checkpoint(
                                    candidate,
                                    {
                                        "schema_version": 1,
                                        "model": model.state_dict(),
                                        "config": model.config,
                                        "policy_provenance": provenance,
                                        "updates": optimizer.updates,
                                        "validation_loss": last_validation,
                                        "fitting_config_sha256": file_hash(cfg_path),
                                    },
                                )
                                selections.append(
                                    {
                                        "update": optimizer.updates,
                                        "validation_loss": best,
                                        "path": str(candidate.relative_to(args.output)),
                                        "sha256": file_hash(candidate),
                                    }
                                )
                                atomic_json(
                                    args.output / "selected-policy.json",
                                    {
                                        "schema_version": 1,
                                        "status": "partial",
                                        "selected": selections[-1],
                                    },
                                )
                        record["validation_loss"] = last_validation
                    log.write(json.dumps(record) + "\n")
                    log.flush()
                    atomic_json(
                        args.output / "progress.json",
                        {
                            "schema_version": 1,
                            "status": "partial",
                            "updates": optimizer.updates,
                            "expected_updates": cfg["updates"],
                            "best_loss": best if selections else None,
                        },
                    )
                    if (
                        optimizer.updates % cfg["durable_checkpoint_every"] == 0
                        or optimizer.updates == cfg["updates"]
                    ):
                        durable()
                        if optimizer.updates % cfg["source_snapshot_every"] == 0:
                            snapshots = args.output / "source-snapshots"
                            snapshots.mkdir(exist_ok=True)
                            snapshot = snapshots / f"checkpoint-{optimizer.updates:06d}.pt"
                            if snapshot.exists():
                                if file_hash(snapshot) != file_hash(checkpoint):
                                    raise ValueError("source snapshot collision; preserve both")
                            else:
                                shutil.copyfile(checkpoint, snapshot)
                    if (args.output / "stop.requested").exists():
                        durable()
                        raise ValueError(
                            "fitting stopped at durable boundary; preserve all artifacts"
                        )
            if not selections:
                raise ValueError("no finite development-selected EMA policy")
            atomic_json(
                args.output / "selected-policy.json",
                {"schema_version": 1, "status": "complete", "selected": selections[-1]},
            )
            atomic_json(
                args.output / "selection.json",
                {
                    "schema_version": 1,
                    "status": "complete",
                    "selected": selections[-1],
                    "candidates": selections,
                    "rule": "minimum confirmation-family-excluded development score-entropy loss"
                    if dataset == "TopDial"
                    else "minimum DuRecDial development score-entropy loss",
                    "seed": 17,
                },
            )
        atomic_json(args.output / "gpu-summary.json", monitor.summary())
        stage.write(
            "complete",
            provenance_updates={"gpu": gpu, "checkpoint": provenance, "seeds": {"root": 17}},
            compute_updates={
                "h100_seconds": previous_compute + time.monotonic() - started,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            },
        )
    except Exception as error:
        stage.write(
            "failed",
            compute_updates={"h100_seconds": previous_compute + time.monotonic() - started},
            error=error,
        )
        raise


if __name__ == "__main__":
    main()
