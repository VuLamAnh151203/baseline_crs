#!/usr/bin/env python3
"""Real H100 policy/dual-Qwen smoke; no fitting or comparative success claims."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer

from tdrl.backend import require_h100_slurm
from tdrl.contracts import Decoding, Message
from tdrl.diffusion import (
    forward_mask,
    loglinear_noise,
    sample_analytic,
    score_entropy,
    semantic_constraints,
)
from tdrl.diffusion_model import load_initial_policy
from tdrl.evaluation import initial_state, visible_context
from tdrl.planner import ASSISTANT_POLICY, INSTRUCTION
from tdrl.provenance import atomic_json, digest, file_hash
from tdrl.runtime import Servers
from tdrl.stages import GpuMonitor, Stage, counters
from tdrl.state import SERIALIZATION_HASH
from tdrl.training import load_prepared


def fingerprint(value):
    import hashlib

    tensor = value.detach().cpu().contiguous()
    return hashlib.sha256(tensor.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slot", choices=[0, 1], type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path.cwd()
    config_path = root / "configs/difftod-policy-smoke-v1.json"
    cfg = json.loads(config_path.read_text())
    if cfg.get("schema_version") != 1:
        raise ValueError("unknown diffusion smoke protocol")
    base_path = root / cfg["comparison_config"]
    base = json.loads(base_path.read_text())
    assistant_decoding = Decoding(**base["assistant_decoding"])
    gpu = require_h100_slurm()
    data = root / cfg["data_preparation"]
    prepared = json.loads((data / "manifest.json").read_text())
    if prepared.get("schema_version") != 1 or prepared["status"] != "complete":
        raise ValueError("diffusion data preparation incomplete")
    if (
        file_hash(root / "data/processed/data-manifest.json")
        != prepared["contract"]["data_manifest_sha256"]
    ):
        raise ValueError("diffusion processed dataset changed")
    for name, expected in prepared["outputs"].items():
        if file_hash(data / name) != expected:
            raise ValueError("diffusion prepared data changed: " + name)
    stage = Stage(
        root,
        args.output,
        "DiffTOD-policy-runtime-smoke",
        config_path,
        {
            "slot": args.slot,
            "comparison_config_sha256": file_hash(base_path),
            "serialization_sha256": SERIALIZATION_HASH,
            "policy_model_lock_sha256": file_hash(root / cfg["policy_model_lock"]),
            "prepared_data_manifest_sha256": file_hash(data / "manifest.json"),
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
            chunks = np.load(data / "train.npy", mmap_mode="r", allow_pickle=False)
            clean = torch.tensor(chunks[:1].copy(), device="cuda", dtype=torch.long)
            sigma, rate = loglinear_noise(torch.tensor([[0.5]], device="cuda"))
            perturbed = forward_mask(
                clean, sigma, model.vocabulary, torch.Generator(device="cuda").manual_seed(17)
            )
            rows = []
            model.train()
            for _ in range(2):
                model.zero_grad(set_to_none=True)
                torch.manual_seed(17)
                torch.cuda.manual_seed_all(17)
                scores = model(perturbed, sigma.reshape(-1))
                loss = (
                    (score_entropy(scores, sigma, perturbed, clean, model.vocabulary) * rate)
                    .sum(-1)
                    .mean()
                )
                if not torch.isfinite(loss):
                    raise ValueError("nonfinite real diffusion policy loss")
                loss.backward()
                gradients = {
                    name: fingerprint(p.grad)
                    for name, p in model.named_parameters()
                    if p.grad is not None
                }
                if any(
                    p.grad is not None and not torch.isfinite(p.grad).all()
                    for p in model.parameters()
                ):
                    raise ValueError("nonfinite real diffusion gradients")
                rows.append(
                    {
                        "score_sha256": fingerprint(scores),
                        "loss": float(loss.detach()),
                        "gradient_sha256": gradients,
                    }
                )
            if rows[0] != rows[1]:
                raise ValueError("real BF16 diffusion forward/gradient replay failed")
            model.zero_grad(set_to_none=True)
            del scores, loss
            model.eval()
            lock = json.loads((root / cfg["policy_model_lock"]).read_text())
            tokenizer_dir = root / "models/difftod/gpt2" / lock["tokenizer"]["revision"]
            tokenizer = Tokenizer.from_file(str(tokenizer_dir / "tokenizer.json"))
            scenario = load_prepared(root, "TopDial", "train")[0]
            profile = "user profile: " + "".join(
                f"{k}: {v}\t" for k, v in scenario["profile"].items()
            )
            # Static target phrase is a smoke fixture, not the baseline's five
            # Qwen paraphrases or a substitute for its interactive planning.
            phrase = f"system: Consider {scenario['target'][1]} for {scenario['target'][0]}."
            constraints = semantic_constraints(
                tokenizer.encode(profile).ids, tokenizer.encode(phrase).ids
            )
            torch.cuda.empty_cache()
            settings = json.loads((root / "configs/runtime.json").read_text())
            for kind in ["generation", "embedding"]:
                settings[kind]["port"] += args.slot * 100
            with Servers(root, args.output / "servers", settings) as servers:
                backend = servers.backends()
                first = sample_analytic(
                    model,
                    constraints,
                    cfg["sampling_steps"],
                    1,
                    "cuda",
                    torch.Generator(device="cuda").manual_seed(17),
                )
                second = sample_analytic(
                    model,
                    constraints,
                    cfg["sampling_steps"],
                    1,
                    "cuda",
                    torch.Generator(device="cuda").manual_seed(17),
                )
                if not torch.equal(first, second) or (first >= model.vocabulary).any():
                    raise ValueError("diffusion sampling exact replay/mask-removal failed")
                state = initial_state(scenario)
                rid = stage.name + "/live"
                embedding = backend["embedding"].embed([state.serialize()], INSTRUCTION, rid, rid)
                repeated = backend["embedding"].embed([state.serialize()], INSTRUCTION, rid, rid)
                if embedding.vectors.shape != (1, 4096) or not torch.equal(
                    embedding.vectors, repeated.vectors
                ):
                    raise ValueError("live full-dimensional embedding exact replay failed")
                if not torch.allclose(
                    embedding.vectors.norm(dim=-1), torch.ones(1), atol=1e-5, rtol=0
                ):
                    raise ValueError("live embedding normalization failed")
                messages = [
                    Message("system", ASSISTANT_POLICY + "\n" + visible_context(scenario)),
                    *state.history,
                ]
                reply = backend["generation"].generate(
                    messages, "assistant", assistant_decoding, 17, rid
                )
                replay = backend["generation"].generate(
                    messages, "assistant", assistant_decoding, 17, rid
                )
                if reply != replay or not reply.text.strip():
                    raise ValueError("local generation journal replay failed")
                compute = {
                    **counters(backend["generation"], backend["embedding"]),
                    **servers.compute_metadata(),
                }
            checks = {
                "schema_version": 1,
                "status": "passed",
                "gpu": gpu,
                "policy": provenance,
                "forward_gradient_replay": rows[0],
                "sampling_exact_replay": True,
                "sample_sha256": fingerprint(first),
                "sample_steps": cfg["sampling_steps"],
                "sampling_policy_forwards": 2 * (cfg["sampling_steps"] + 1),
                "gradient_policy_forwards": 2,
                "optimizer_updates": 0,
                "fitted_baseline": False,
                "fitting_gate": "pending update/native-kernel/budget/interactive gates",
            }
            atomic_json(args.output / "checks.json", checks)
        compute.update(
            h100_seconds=time.monotonic() - started,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        )
        atomic_json(args.output / "gpu-summary.json", monitor.summary())
        stage.write(
            "complete",
            provenance_updates={
                "gpu": gpu,
                "checkpoint": provenance,
                "dataset": {
                    "split": "train",
                    "prepared_manifest_sha256": file_hash(data / "manifest.json"),
                },
                "scenarios": {"initial_case": scenario["id"], "gradient_chunks": [0]},
                "seeds": {"root": 17, "gradient": 17, "sampling": 17},
                "prompts": {
                    "assistant_policy_sha256": digest(ASSISTANT_POLICY),
                    "embedding_instruction_sha256": digest(INSTRUCTION),
                    "conditioning_phrase_sha256": digest(phrase),
                },
            },
            compute_updates=compute,
        )
    except Exception as error:
        compute.update(h100_seconds=time.monotonic() - started)
        stage.write("failed", compute_updates=compute, error=error)
        raise


if __name__ == "__main__":
    main()
