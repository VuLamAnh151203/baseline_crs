import os
import tempfile
from copy import deepcopy
from pathlib import Path

import torch

from ...provenance import digest, file_hash
from .model import load_weights


class ExponentialMovingAverage:
    def __init__(self, parameters, decay):
        if not 0 < decay < 1:
            raise ValueError("EMA decay must be in (0, 1)")
        self.decay = decay
        self.shadow = [parameter.detach().clone() for parameter in parameters]
        self.collected = None

    @torch.no_grad()
    def update(self, parameters):
        parameters = list(parameters)
        if len(parameters) != len(self.shadow):
            raise ValueError("EMA parameter topology changed")
        for shadow, parameter in zip(self.shadow, parameters, strict=True):
            shadow.sub_((1 - self.decay) * (shadow - parameter.detach()))

    def state_dict(self):
        return {"decay": self.decay, "shadow": self.shadow}

    def load_state_dict(self, state):
        if set(state) != {"decay", "shadow"} or state["decay"] != self.decay:
            raise ValueError("EMA checkpoint configuration mismatch")
        if len(state["shadow"]) != len(self.shadow):
            raise ValueError("EMA checkpoint parameter topology mismatch")
        for current, stored in zip(self.shadow, state["shadow"], strict=True):
            if current.shape != stored.shape:
                raise ValueError("EMA checkpoint tensor shape mismatch")
        self.shadow = [tensor.detach().clone() for tensor in state["shadow"]]

    def store(self, parameters):
        if self.collected is not None:
            raise RuntimeError("EMA parameters are already active")
        self.collected = [parameter.detach().clone() for parameter in parameters]

    @torch.no_grad()
    def copy_to(self, parameters):
        for parameter, shadow in zip(parameters, self.shadow, strict=True):
            parameter.copy_(shadow.to(device=parameter.device, dtype=parameter.dtype))

    @torch.no_grad()
    def restore(self, parameters):
        if self.collected is None:
            raise RuntimeError("EMA restore called without store")
        for parameter, stored in zip(parameters, self.collected, strict=True):
            parameter.copy_(stored)
        self.collected = None


def atomic_torch_save(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=".pending-", suffix=".pt")
    os.close(descriptor)
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def rng_state(generator=None):
    state = {
        "cpu": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "generator": generator.get_state() if generator is not None else None,
    }
    return state


def restore_rng_state(state, generator=None):
    if set(state) != {"cpu", "cuda", "generator"}:
        raise ValueError("incomplete RNG checkpoint")
    torch.set_rng_state(state["cpu"])
    if state["cuda"] is not None:
        if not torch.cuda.is_available():
            raise ValueError("CUDA RNG checkpoint cannot be restored without CUDA")
        torch.cuda.set_rng_state_all(state["cuda"])
    if generator is not None:
        if state["generator"] is None:
            raise ValueError("checkpoint has no explicit generator state")
        generator.set_state(state["generator"])


def save_training_checkpoint(
    path,
    model,
    optimizer,
    ema,
    scaler,
    step,
    config_sha256,
    generator=None,
):
    if not isinstance(step, int) or step < 0 or len(config_sha256) != 64:
        raise ValueError("invalid DiffTOD checkpoint metadata")
    payload = {
        "schema_version": 2,
        "step": step,
        "config_sha256": config_sha256,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "ema": ema.state_dict(),
        "rng": rng_state(generator),
    }
    atomic_torch_save(Path(path), payload)
    return file_hash(Path(path))


def load_training_checkpoint(
    path,
    model,
    optimizer,
    ema,
    scaler,
    config_sha256,
    *,
    generator=None,
    map_location="cpu",
):
    payload = torch.load(path, map_location=map_location, weights_only=False)
    required = {
        "schema_version",
        "step",
        "config_sha256",
        "model",
        "optimizer",
        "scaler",
        "ema",
        "rng",
    }
    if set(payload) != required or payload["schema_version"] != 2:
        raise ValueError("unknown/incomplete DiffTOD checkpoint")
    if payload["config_sha256"] != config_sha256:
        raise ValueError("DiffTOD resume configuration mismatch")
    load_weights(model, payload["model"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    scaler.load_state_dict(payload["scaler"])
    ema.load_state_dict(payload["ema"])
    restore_rng_state(payload["rng"], generator)
    return payload["step"]


def checkpoint_signature(path: Path):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("model", payload)
    if not isinstance(state, dict):
        raise ValueError("checkpoint does not contain model tensors")
    topology = {
        key.removeprefix("module."): {"shape": list(value.shape), "dtype": str(value.dtype)}
        for key, value in sorted(state.items())
        if torch.is_tensor(value)
    }
    return {
        "sha256": file_hash(path),
        "architecture_sha256": digest(topology),
        "topology": topology,
    }


def cloned_state(module):
    """Test helper used to assert exact resume without aliasing tensors."""
    return deepcopy(module.state_dict())
