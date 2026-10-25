import os
import random
import tempfile
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from ...provenance import digest, file_hash
from .contracts import DPDPCheckpointMetadata


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
        raise ValueError("incomplete DPDP RNG checkpoint")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None:
        if not torch.cuda.is_available():
            raise ValueError("cannot restore CUDA DPDP checkpoint without CUDA")
        torch.cuda.set_rng_state_all(state["cuda"])
    if generator is not None:
        if state["generator"] is None:
            raise ValueError("DPDP checkpoint has no loader-generator state")
        generator.set_state(state["generator"])


def save_checkpoint(
    path,
    model,
    optimizer,
    scheduler,
    step,
    epoch,
    batch_in_epoch,
    metadata: DPDPCheckpointMetadata,
    generator=None,
):
    if step < 0 or epoch < 0 or batch_in_epoch < 0:
        raise ValueError("invalid DPDP checkpoint cursor")
    hashes = (
        metadata.config_sha256,
        metadata.data_sha256,
        metadata.action_taxonomy_sha256,
        metadata.model_lock_sha256,
    )
    if any(len(value) != 64 for value in hashes):
        raise ValueError("incomplete DPDP checkpoint provenance")
    if (
        metadata.parent_checkpoint_sha256 is not None
        and len(metadata.parent_checkpoint_sha256) != 64
    ):
        raise ValueError("invalid DPDP parent-checkpoint provenance")
    _atomic_save(
        path,
        {
            "schema_version": 1,
            "step": step,
            "epoch": epoch,
            "batch_in_epoch": batch_in_epoch,
            "metadata": asdict(metadata),
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "rng": rng_state(generator),
        },
    )
    return file_hash(Path(path))


def load_checkpoint(
    path,
    model,
    expected_metadata: DPDPCheckpointMetadata,
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
        "metadata",
        "model",
        "optimizer",
        "scheduler",
        "rng",
    }
    if set(payload) != required or payload["schema_version"] != 1:
        raise ValueError("unknown/incomplete DPDP checkpoint")
    if payload["metadata"] != asdict(expected_metadata):
        raise ValueError("DPDP checkpoint provenance/configuration mismatch")
    model.load_state_dict(payload["model"], strict=True)
    if (optimizer is None) != (scheduler is None):
        raise ValueError("DPDP resume requires both optimizer and scheduler")
    if optimizer is not None:
        if payload["optimizer"] is None or payload["scheduler"] is None:
            raise ValueError("DPDP checkpoint is not resumable")
        optimizer.load_state_dict(payload["optimizer"])
        scheduler.load_state_dict(payload["scheduler"])
        restore_rng(payload["rng"], generator)
    return payload


def checkpoint_signature(path):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("model")
    if not isinstance(state, dict):
        raise ValueError("DPDP checkpoint has no model state")
    topology = {
        key: {"shape": list(value.shape), "dtype": str(value.dtype)}
        for key, value in sorted(state.items())
        if torch.is_tensor(value)
    }
    return {"sha256": file_hash(Path(path)), "architecture_sha256": digest(topology)}
