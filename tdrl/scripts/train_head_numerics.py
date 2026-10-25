#!/usr/bin/env python3
"""Fresh H100 head fitting after exact numerical replay; immutable BF16 forward."""

import argparse
import copy
import hashlib
import json
from pathlib import Path

import torch

from tdrl.backend import require_h100_slurm
from tdrl.learning import CostConfig, ReachabilityLearner
from tdrl.provenance import atomic_json, file_hash
from tdrl.reachability import IQE, MRNFixed
from tdrl.stages import Stage
from tdrl.state import SERIALIZATION_HASH
from tdrl.training import PairSampler, heldout_temporal_diagnostics, load_embeddings, train_head


def fingerprint(state):
    h = hashlib.sha256()

    def add(value):
        if isinstance(value, torch.Tensor):
            t = value.detach().cpu().contiguous()
            h.update(str(t.dtype).encode())
            h.update(str(tuple(t.shape)).encode())
            h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(value, dict):
            for key in sorted(value, key=str):
                h.update(str(key).encode())
                add(value[key])
        elif isinstance(value, (list, tuple)):
            for item in value:
                add(item)
        else:
            h.update(str(value).encode())

    add(state)
    return h.hexdigest()


p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--head", choices=["MRNFixed", "IQE"], required=True)
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
root = Path.cwd()
config_path = root / "configs/head-numerics-v2.json"
cfg = json.loads(config_path.read_text())
source = root / cfg["source"]
gpu = require_h100_slurm()
records, vectors = load_embeddings(source / "embeddings/TopDial/bank")
manifest = json.loads((source / "embeddings/TopDial/manifest.json").read_text())
if manifest["status"] != "complete":
    raise ValueError("embedding source is not complete")
index = source / "embeddings/TopDial/bank/index.json"
if file_hash(index) != manifest["outputs"]["bank/index.json"]:
    raise ValueError("embedding bank provenance mismatch")
stage = Stage(
    root,
    a.output,
    "TopDial-head-numerics-" + a.head,
    config_path,
    {
        "models": json.loads((root / "configs/models.lock.json").read_text()),
        "source_embedding_manifest_sha256": file_hash(source / "embeddings/TopDial/manifest.json"),
        "serialization_sha256": SERIALIZATION_HASH,
        "numerical_revision": 2,
    },
)
if not stage.reused:
    old_determinism = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        torch.manual_seed(17)
        torch.cuda.manual_seed_all(17)
        head = {"MRNFixed": MRNFixed, "IQE": IQE}[a.head]().to("cuda", torch.bfloat16)
        learner = ReachabilityLearner(
            head, CostConfig(**cfg["head"]["cost"]), lr=cfg["head"]["lr"], numerical_revision=2
        )
        start = copy.deepcopy(learner.state_dict())
        sampler = PairSampler(records, 17)
        indices = sampler.sample(64)
        source_idx, next_idx, goal_idx, is_goal, next_goal, reachable = indices
        pool = vectors.to("cuda", torch.bfloat16)
        inputs = (
            pool[list(source_idx)],
            pool[list(next_idx)],
            pool[list(goal_idx)],
            torch.tensor(is_goal, device="cuda", dtype=torch.bool),
            torch.tensor(next_goal, device="cuda", dtype=torch.bool),
            torch.tensor(reachable, device="cuda", dtype=torch.bool),
        )
        legacy = ReachabilityLearner(
            copy.deepcopy(head), CostConfig(**cfg["head"]["cost"]), lr=cfg["head"]["lr"]
        )
        if not torch.equal(
            learner.head.distance(inputs[0], inputs[2]), legacy.head.distance(inputs[0], inputs[2])
        ) or not torch.equal(
            learner.target.distance(inputs[1], inputs[2]),
            legacy.target.distance(inputs[1], inputs[2]),
        ):
            raise ValueError("numeric state repair changed initial BF16 forward")
        repeats = []
        for _ in range(2):
            learner.load_state_dict(copy.deepcopy(start))
            loss = learner.step(*inputs)
            if any(p.grad is not None for p in learner.target.parameters()):
                raise ValueError("target gradient not detached")
            gradients = {
                n: p.grad for n, p in learner.head.named_parameters() if p.grad is not None
            }
            repeats.append(
                {
                    "loss": loss,
                    "gradient_sha256": fingerprint(gradients),
                    "state_sha256": fingerprint(learner.state_dict()),
                }
            )
        if repeats[0] != repeats[1]:
            raise ValueError("head numeric exact gradient/update/RNG replay failed")
        atomic_json(
            a.output / "numeric-gate.json",
            {
                "status": "passed",
                "gpu": gpu,
                "forward_dtype": "bfloat16",
                "masters_moments_target_ema": "float32",
                "initial_forward_unchanged": True,
                "exact_gradient_update_replay": repeats[0],
                "scientific_fitting_updates": 0,
            },
        )
        del legacy, learner, head, inputs, pool
        torch.cuda.empty_cache()
        fitted = train_head(a.head, records, vectors, cfg["head"], a.output, stage.contract)
        dev_records, dev_vectors = load_embeddings(source / "embeddings/TopDial/dev-bank")
        temporal = heldout_temporal_diagnostics(fitted, dev_records, dev_vectors, a.output, 17)
        atomic_json(a.output / "heldout-temporal.json", temporal)
        stage.write(
            "complete",
            provenance_updates={
                "dataset": {
                    "dataset": "TopDial",
                    "split": "train",
                    "source_index_sha256": file_hash(index),
                },
                "checkpoint": {
                    "sha256": file_hash(a.output / "checkpoint.pt"),
                    "numerical_revision": 2,
                    "head": a.head,
                },
                "seeds": {"root": 17},
                "gpu": gpu,
            },
        )
    except Exception as error:
        stage.write("partial", error=str(error))
        raise
    finally:
        torch.use_deterministic_algorithms(old_determinism)
