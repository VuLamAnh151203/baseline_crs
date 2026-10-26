import os
import tempfile
from dataclasses import asdict
from pathlib import Path

import torch

from ...provenance import file_hash
from .contracts import EPLCheckpointMetadata


def save_checkpoint(path, model, optimizer, epoch, step, metadata, validation_loss):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=".pending-", suffix=".pt")
    os.close(descriptor)
    try:
        torch.save(
            {
                "schema_version": 1,
                "epoch": epoch,
                "step": step,
                "validation_loss": validation_loss,
                "metadata": asdict(metadata),
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict() if optimizer is not None else None,
            },
            temporary,
        )
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return file_hash(path)


def load_checkpoint(path, model, expected_metadata=None, optimizer=None, map_location="cpu"):
    payload = torch.load(path, map_location=map_location, weights_only=False)
    required = {"schema_version", "epoch", "step", "validation_loss", "metadata", "model", "optimizer"}
    if set(payload) != required or payload["schema_version"] != 1:
        raise ValueError("unknown/incomplete EPL checkpoint")
    metadata = EPLCheckpointMetadata(**payload["metadata"])
    if expected_metadata is not None and metadata != expected_metadata:
        raise ValueError("EPL checkpoint provenance mismatch")
    model.load_state_dict(payload["model"], strict=True)
    if optimizer is not None:
        if payload["optimizer"] is None:
            raise ValueError("EPL checkpoint has no optimizer state")
        optimizer.load_state_dict(payload["optimizer"])
    return payload, metadata
