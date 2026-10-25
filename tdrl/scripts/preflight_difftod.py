#!/usr/bin/env python3
"""Real update/interactive replay and measured native-budget feasibility; never fit."""

import argparse
import copy
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer

from tdrl.backend import require_h100_slurm
from tdrl.contracts import Decoding
from tdrl.difftod import DiffTODPlanner
from tdrl.diffusion_model import load_initial_policy
from tdrl.diffusion_optimizer import DiffusionOptimizer
from tdrl.diffusion_training import policy_loss
from tdrl.evaluation import initial_state, run_dialogue
from tdrl.feasibility import full_turn_dialogue_proxy
from tdrl.judge import LocalJudge
from tdrl.provenance import atomic_json, file_hash
from tdrl.runtime import Servers
from tdrl.stages import GpuMonitor, Stage, counters
from tdrl.training import load_prepared


def cpu_state(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_state(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu_state(v) for v in value]
    return copy.deepcopy(value)


def fingerprint(value):
    h = hashlib.sha256()

    def add(v):
        if isinstance(v, torch.Tensor):
            t = v.detach().cpu().contiguous()
            h.update(str(t.dtype).encode())
            h.update(str(tuple(t.shape)).encode())
            h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(v, dict):
            for k in sorted(v, key=str):
                h.update(str(k).encode())
                add(v[k])
        elif isinstance(v, (list, tuple)):
            for item in v:
                add(item)
        else:
            h.update(str(v).encode())

    add(value)
    return h.hexdigest()


def main(config="configs/difftod-topdial-fit-v1.json", dataset="TopDial"):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--slot", type=int, choices=[0, 1], required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    root = Path.cwd()
    config_path = root / config
    cfg = json.loads(config_path.read_text())
    if cfg.get("schema_version") != 1:
        raise ValueError("unknown DiffTOD fitting contract")
    gpu = require_h100_slurm()
    data = root / cfg["data_preparation"]
    validation = root / cfg["validation_preparation"]
    for directory in [data, validation]:
        manifest = json.loads((directory / "manifest.json").read_text())
        if manifest.get("schema_version") != 1 or manifest["status"] != "complete":
            raise ValueError("unverified diffusion data preparation")
        for name, sha in manifest["outputs"].items():
            if file_hash(directory / name) != sha:
                raise ValueError("diffusion data checksum mismatch")
    stage = Stage(
        root,
        args.output,
        "DiffTOD-native-budget-preflight",
        config_path,
        {
            "slot": args.slot,
            "data_sha256": file_hash(data / "manifest.json"),
            "validation_sha256": file_hash(validation / "manifest.json"),
            "comparison_sha256": file_hash(root / cfg["comparison_config"]),
            "nltk_lock_sha256": file_hash(root / "configs/difftod-nltk.lock.json"),
        },
    )
    if stage.reused:
        return
    started = time.monotonic()
    compute = {}
    try:
        with GpuMonitor(args.output / "gpu-telemetry.jsonl") as monitor:
            torch.use_deterministic_algorithms(True)
            torch.manual_seed(17)
            torch.cuda.manual_seed_all(17)
            model, provenance = load_initial_policy(root)
            optimizer = DiffusionOptimizer(model, cfg["optimizer"])
            initial = cpu_state(optimizer.state_dict())
            chunks = np.load(data / "train.npy", mmap_mode="r", allow_pickle=False)
            clean = torch.tensor(chunks[:1].copy(), device="cuda", dtype=torch.long)
            repeats = []
            for _ in range(2):
                optimizer.load_state_dict(initial)
                optimizer.zero_grad()
                model.train()
                torch.manual_seed(17)
                torch.cuda.manual_seed_all(17)
                loss = policy_loss(model, clean, torch.Generator(device="cuda").manual_seed(17))
                loss.backward()
                gradient = fingerprint(
                    {n: p.grad for n, p in model.named_parameters() if p.grad is not None}
                )
                optimizer.step()
                repeats.append(
                    {
                        "loss": float(loss.detach()),
                        "gradient_sha256": gradient,
                        "optimizer_ema_sha256": fingerprint(optimizer.state_dict()),
                    }
                )
            if repeats[0] != repeats[1]:
                raise ValueError("exact diffusion gradient/update/EMA replay failed")
            optimizer.load_state_dict(initial)
            batch = torch.tensor(
                chunks[: cfg["batch_size"]].copy(), device="cuda", dtype=torch.long
            )
            update_times = []
            forward_times = []
            for index in range(3):
                optimizer.zero_grad()
                model.train()
                torch.cuda.synchronize()
                begin = time.monotonic()
                loss = policy_loss(
                    model, batch, torch.Generator(device="cuda").manual_seed(17 + index)
                )
                loss.backward()
                optimizer.step()
                torch.cuda.synchronize()
                update_times.append(time.monotonic() - begin)
                model.eval()
                torch.cuda.synchronize()
                begin = time.monotonic()
                with torch.no_grad():
                    policy_loss(model, batch, torch.Generator(device="cuda").manual_seed(17))
                torch.cuda.synchronize()
                forward_times.append(time.monotonic() - begin)
            optimizer.load_state_dict(initial)
            optimizer.zero_grad()
            model.eval()
            del optimizer, initial, batch, clean, loss
            torch.cuda.empty_cache()
            base = json.loads((root / cfg["comparison_config"]).read_text())
            pin = json.loads((root / cfg["policy_model_lock"]).read_text())
            tokenizer = Tokenizer.from_file(
                str(root / "models/difftod/gpt2" / pin["tokenizer"]["revision"] / "tokenizer.json")
            )
            settings = json.loads((root / "configs/runtime.json").read_text())
            for kind in ["generation", "embedding"]:
                settings[kind]["port"] += args.slot * 100
            scenario = load_prepared(root, dataset, "train")[0]
            dialogues = []
            with Servers(root, args.output / "servers", settings) as servers:
                b = servers.backends()
                judge = LocalJudge(b["generation"], base["judge"])
                for protocol in ["profile", "resistant"]:
                    actor = DiffTODPlanner(
                        model,
                        tokenizer,
                        b["generation"],
                        b["embedding"],
                        scenario,
                        Decoding(**base["assistant_decoding"]),
                        Decoding(**cfg["guidance_decoding"]),
                        args.output / "policy-journal",
                        provenance,
                        root / "artifacts/difftod-nltk-data",
                    )
                    result = run_dialogue(
                        scenario,
                        actor,
                        b["generation"],
                        b["embedding"],
                        protocol,
                        17,
                        args.output / "dialogues" / protocol,
                        {
                            "phase": "DiffTOD-unfitted-budget-proxy",
                            "method": "DiffTOD",
                            "request_namespace": fingerprint(stage.contract),
                        },
                        turn_budget=base["turn_budget"],
                        simulator_decoding=Decoding(**base["simulator_decoding"]),
                        judge=judge,
                    )
                    first_plan = result["turns_trace"][0]["plan"]
                    first_id = first_plan["replay_ids"][0].removesuffix("/live")
                    actor.trace = None
                    actor.seed_context = f"{scenario['id']}/{protocol}/0"
                    executed = len(b["generation"].requests)
                    replayed = actor.act(initial_state(scenario), [], 17, first_id)
                    if (
                        replayed.response != first_plan["response"]
                        or len(b["generation"].requests) != executed
                    ):
                        raise ValueError("DiffTOD exact final plan/journal replay failed")
                    if actor.trace["policy_forwards"] != 5125:
                        raise ValueError("native diffusion budget changed")
                    dialogues.append(
                        {
                            "protocol": protocol,
                            "wall_s": result["wall_s"],
                            "actual_turns": result["actual_turns"],
                            "policy_forwards": actor.trace["policy_forwards"],
                            "policy_sampling_wall_s": actor.trace["policy_sampling_wall_s"],
                        }
                    )
                compute = {
                    **counters(b["generation"], b["embedding"]),
                    **servers.compute_metadata(),
                }
            allocation = json.loads((root / "artifacts/two-gpu-58159/allocation.json").read_text())
            # Allocation metadata already freezes the real original start clock.
            start_epoch = allocation.get("started_unix")
            if start_epoch is None:
                raise ValueError(
                    "allocation start epoch field requires audit; no fabricated remaining time"
                )
            remaining = 20 * 86400 - (time.time() - start_epoch)
            fit = cfg["updates"] * max(update_times)
            valid_batches = (
                len(np.load(validation / "validation.npy", mmap_mode="r")) + cfg["batch_size"] - 1
            ) // cfg["batch_size"]
            val = (cfg["updates"] // cfg["validation_every"]) * valid_batches * max(forward_times)
            # Unsuccessful proxy turns scale to the full common limit, not favorable early stopping.
            dialogue = max(
                full_turn_dialogue_proxy(r["wall_s"], r["actual_turns"], base["turn_budget"])
                for r in dialogues
            )
            evaluation = cfg["evaluation_dialogue_reserve"] * dialogue
            required = (fit + val + evaluation + cfg["checkpoint_io_reserve_seconds"]) * cfg[
                "feasibility_safety_factor"
            ]
            projection = {
                "schema_version": 1,
                "status": "measured_unfitted_proxy_not_scientific_result",
                "native_batch_size": 32,
                "update_seconds": update_times,
                "validation_batch_seconds": forward_times,
                "dialogues": dialogues,
                "fit_seconds": fit,
                "validation_seconds": val,
                "evaluation_seconds": evaluation,
                "required_seconds_with_safety": required,
                "remaining_allocation_seconds": remaining,
                "fits": required <= remaining,
                "scope_limits": "one training case per protocol; conservative common-method proxy, post-fit recheck required",
            }
            atomic_json(args.output / "projection.json", projection)
            atomic_json(
                args.output / "update-replay.json",
                {
                    "schema_version": 1,
                    "status": "passed",
                    "ephemeral_fixture_updates": 5,
                    "committed_fitting_updates": 0,
                    "replay": repeats[0],
                },
            )
            if not projection["fits"]:
                raise ValueError(
                    "scoped matrix exceeds measured remaining allocation; explicit budget/scope decision required"
                )
        atomic_json(args.output / "gpu-summary.json", monitor.summary())
        compute.update(
            h100_seconds=time.monotonic() - started,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        )
        stage.write(
            "complete",
            provenance_updates={"gpu": gpu, "checkpoint": provenance},
            compute_updates=compute,
        )
    except Exception as error:
        compute.update(h100_seconds=time.monotonic() - started)
        stage.write("failed", compute_updates=compute, error=error)
        raise


if __name__ == "__main__":
    main()
