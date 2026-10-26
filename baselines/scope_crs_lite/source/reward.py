"""Goal-conditioned task reward for SCOPE-CRS-lite."""

from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn

from ...provenance import file_hash


class GoalConditionedReward(nn.Module):
    def __init__(self, dimension=1024, hidden_dims=(512, 128)):
        super().__init__()
        layers = []
        width = dimension * 3
        for hidden in hidden_dims:
            layers.extend((nn.Linear(width, hidden), nn.ReLU()))
            width = hidden
        layers.append(nn.Linear(width, 1))
        self.network = nn.Sequential(*layers)

    def logits(self, state, goal):
        state, goal = torch.broadcast_tensors(state, goal)
        features = torch.cat((state, goal, torch.abs(state - goal)), dim=-1)
        output = self.network(features).squeeze(-1)
        if not torch.isfinite(output).all():
            raise FloatingPointError("non-finite SCOPE reward")
        return output

    def forward(self, state, goal):
        return torch.sigmoid(self.logits(state, goal))


@dataclass(frozen=True)
class RewardMetadata:
    schema_version: int
    method: str
    dataset: str
    config_sha256: str
    train_manifest_sha256: str
    dev_manifest_sha256: str
    llama_guard_manifest_sha256: str
    transition_manifest_sha256: str
    architecture: str
    selection_rule: str


def save_reward_checkpoint(path, model, metadata, epoch, development_loss):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": 1,
            "model_state_dict": model.state_dict(),
            "metadata": asdict(metadata),
            "epoch": epoch,
            "development_loss": development_loss,
        },
        path,
    )


def save_training_checkpoint(
    path,
    model,
    optimizer,
    metadata,
    epoch,
    best_loss,
    stale_epochs,
):
    """Save the complete deterministic training state for allocation resume."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "schema_version": 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metadata": asdict(metadata),
            "next_epoch": epoch + 1,
            "best_development_loss": best_loss,
            "stale_epochs": stale_epochs,
            "cpu_rng_state": torch.random.get_rng_state(),
            "cuda_rng_state_all": torch.cuda.get_rng_state_all(),
        },
        temporary,
    )
    temporary.replace(path)


def load_training_checkpoint(path, model, optimizer, metadata):
    """Restore a last-epoch checkpoint, rejecting changed data or configuration."""
    path = Path(path)
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    required = {
        "schema_version",
        "model_state_dict",
        "optimizer_state_dict",
        "metadata",
        "next_epoch",
        "best_development_loss",
        "stale_epochs",
        "cpu_rng_state",
        "cuda_rng_state_all",
    }
    if set(payload) != required or payload["schema_version"] != 1:
        raise ValueError("unknown SCOPE training checkpoint")
    if payload["metadata"] != asdict(metadata):
        raise ValueError("SCOPE resume checkpoint provenance mismatch")
    model.load_state_dict(payload["model_state_dict"], strict=True)
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    torch.random.set_rng_state(payload["cpu_rng_state"])
    torch.cuda.set_rng_state_all(payload["cuda_rng_state_all"])
    return {
        "next_epoch": int(payload["next_epoch"]),
        "best_development_loss": float(payload["best_development_loss"]),
        "stale_epochs": int(payload["stale_epochs"]),
    }


def load_reward_checkpoint(path, config, expected_metadata=None, device="cuda"):
    path = Path(path)
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    required = {
        "schema_version",
        "model_state_dict",
        "metadata",
        "epoch",
        "development_loss",
    }
    if set(payload) != required or payload["schema_version"] != 1:
        raise ValueError("unknown SCOPE reward checkpoint")
    if expected_metadata is not None and payload["metadata"] != asdict(expected_metadata):
        raise ValueError("SCOPE reward checkpoint provenance mismatch")
    if payload["metadata"].get("dataset") != config.dataset:
        raise ValueError("SCOPE reward checkpoint dataset mismatch")
    model = GoalConditionedReward(config.embedding_dim, config.reward_hidden_dims)
    incompatible = model.load_state_dict(payload["model_state_dict"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise ValueError("SCOPE reward checkpoint architecture mismatch")
    return model.to(device).eval(), {
        "sha256": file_hash(path),
        "metadata": payload["metadata"],
        "epoch": payload["epoch"],
        "development_loss": payload["development_loss"],
    }
