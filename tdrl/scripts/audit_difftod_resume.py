#!/usr/bin/env python3
"""Probe immutable nonzero checkpoint restoration on H100; never commit an update."""

import argparse
import json
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
from tdrl.diffusion_resume_execution import execution_only
from tdrl.diffusion_training import ChunkBatches, policy_loss
from tdrl.provenance import atomic_json, file_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path.cwd()
    if subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
        raise ValueError("Committed clean source required")
    execution_only(root, args.source_commit)
    gpu = require_h100_slurm()
    started = time.monotonic()
    sha = file_hash(args.checkpoint)
    cfg = json.loads((root / "configs/difftod-topdial-fit-v1.json").read_text())
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if (
        saved["optimizer"]["updates"] != 82360
        or saved["contract"]["code"]["commit"] != args.source_commit
    ):
        raise ValueError("Wrong migration checkpoint/source")
    for field, path in [
        ("config_sha256", "configs/difftod-topdial-fit-v1.json"),
        ("package_lock_sha256", "runtime/uv.lock"),
    ]:
        if saved["contract"][field] != file_hash(root / path):
            raise ValueError("Changed frozen fitting contract")
    data = root / cfg["data_preparation"]
    manifest = json.loads((data / "manifest.json").read_text())
    if saved["contract"]["training_data_sha256"] != file_hash(data / "manifest.json"):
        raise ValueError("Changed training membership")
    for name, expected in manifest["outputs"].items():
        if file_hash(data / name) != expected:
            raise ValueError("Changed fitting input")
    torch.use_deterministic_algorithms(True)
    model, _ = load_initial_policy(root)
    optimizer = DiffusionOptimizer(model, cfg["optimizer"])
    train = np.load(data / "train.npy", mmap_mode="r", allow_pickle=False)
    sampler = ChunkBatches(len(train), cfg["batch_size"], 17)
    sampler.load_state_dict(saved["sampler"])
    batch = torch.tensor(train[sampler.next()].copy(), device="cuda", dtype=torch.long)
    generator = torch.Generator(device="cuda")
    pristine = cpu_state(saved["optimizer"])
    source_fingerprint = fingerprint(pristine)

    def probe(restore, state):
        model.load_state_dict(saved["model"])
        restore(optimizer, state)
        generator.set_state(saved["generator"])
        torch.set_rng_state(saved["cpu_rng"])
        torch.cuda.set_rng_state_all(saved["cuda_rng"])
        model.train()
        optimizer.zero_grad()
        loss = policy_loss(model, batch, generator)
        loss.backward()
        gradient = fingerprint(
            {n: p.grad for n, p in model.named_parameters() if p.grad is not None}
        )
        optimizer.step()
        return {
            "loss": float(loss.detach()),
            "gradient": gradient,
            "optimizer_ema": fingerprint(optimizer.state_dict()),
        }

    flawed_state = cpu_state(pristine)
    original = [probe(lambda o, s: o.load_state_dict(s), flawed_state) for _ in range(2)]
    alias = fingerprint(flawed_state) != source_fingerprint
    del flawed_state
    fixed = [probe(restore_optimizer, pristine) for _ in range(2)]
    immutable = fingerprint(pristine) == source_fingerprint
    forward = all(
        (r["loss"], r["gradient"]) == (fixed[0]["loss"], fixed[0]["gradient"])
        for r in original + fixed
    )
    passed = (
        alias
        and original[0] != original[1]
        and immutable
        and fixed[0] == fixed[1]
        and forward
        and original[0] == fixed[0]
    )
    unchanged = file_hash(args.checkpoint) == sha
    record = {
        "schema_version": 1,
        "status": "passed" if passed and unchanged else "failed",
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "checkpoint_sha256": sha,
        "checkpoint_unchanged": unchanged,
        "before_updates": 82360,
        "committed_updates": 0,
        "gpu": gpu,
        "source_state_immutable": immutable,
        "original_counter_alias_reproduced": alias,
        "exact_next_update_replay": fixed[0] == fixed[1],
        "unchanged_forward": forward,
        "original": original,
        "corrected": fixed,
        "h100_seconds": time.monotonic() - started,
        "script_sha256": file_hash(Path(__file__)),
    }
    atomic_json(args.output, record)
    if record["status"] != "passed":
        raise ValueError("Restore audit failed; preserve checkpoint")


if __name__ == "__main__":
    main()
