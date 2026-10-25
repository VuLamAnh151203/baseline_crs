import os
import random
import tempfile
from pathlib import Path

import numpy as np
import torch

from ...provenance import digest, file_hash


def _atomic_save(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=".pending-", suffix=".pt")
    os.close(descriptor)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def rng_state(generator=None):
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "generator": generator.get_state() if generator is not None else None,
    }


def restore_rng(state, generator=None):
    if set(state) != {"python", "numpy", "torch", "cuda", "generator"}:
        raise ValueError("incomplete TRIP RNG checkpoint")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None:
        if not torch.cuda.is_available():
            raise ValueError("cannot restore a CUDA TRIP checkpoint without CUDA")
        torch.cuda.set_rng_state_all(state["cuda"])
    if generator is not None:
        if state["generator"] is None:
            raise ValueError("TRIP checkpoint has no loader generator state")
        generator.set_state(state["generator"])


def save_checkpoint(
    path,
    model,
    optimizer,
    scheduler,
    step,
    epoch,
    config_sha256,
    generator=None,
    batch_in_epoch=0,
):
    if step < 0 or epoch < 0 or batch_in_epoch < 0 or len(config_sha256) != 64:
        raise ValueError("invalid TRIP checkpoint metadata")
    _atomic_save(
        path,
        {
            "schema_version": 1,
            "step": step,
            "epoch": epoch,
            "batch_in_epoch": batch_in_epoch,
            "config_sha256": config_sha256,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "rng": rng_state(generator),
        },
    )
    return file_hash(Path(path))


def load_checkpoint(
    path,
    model,
    config_sha256,
    optimizer=None,
    scheduler=None,
    generator=None,
    map_location="cpu",
):
    payload = torch.load(path, map_location=map_location, weights_only=False)
    required = {
        "schema_version",
        "step",
        "epoch",
        "batch_in_epoch",
        "config_sha256",
        "model",
        "optimizer",
        "scheduler",
        "rng",
    }
    if set(payload) != required or payload["schema_version"] != 1:
        raise ValueError("unknown/incomplete TRIP checkpoint")
    if payload["config_sha256"] != config_sha256:
        raise ValueError("TRIP checkpoint configuration mismatch")
    model.load_state_dict(payload["model"], strict=True)
    if (optimizer is None) != (scheduler is None):
        raise ValueError("TRIP resume requires both optimizer and scheduler")
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer"])
        scheduler.load_state_dict(payload["scheduler"])
        restore_rng(payload["rng"], generator)
    return payload


def checkpoint_signature(path):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("model")
    if not isinstance(state, dict):
        raise ValueError("TRIP checkpoint has no model state")
    topology = {
        key: {"shape": list(value.shape), "dtype": str(value.dtype)}
        for key, value in sorted(state.items())
        if torch.is_tensor(value)
    }
    return {"sha256": file_hash(Path(path)), "architecture_sha256": digest(topology)}
