"""Strict loader for the released four-seed SCOPE transition ensemble."""

from pathlib import Path

import torch
from torch import nn

from .assets import DIRECTIONS, SEEDS, verify_transition_manifest
from .vendor_moe import HeirarchicalMoE


class RegressionWrapper(nn.Module):
    def __init__(self, dimension=1024):
        super().__init__()
        self.add_module("model", HeirarchicalMoE(dimension))
        self.input_mean = nn.Parameter(torch.zeros(dimension), requires_grad=False)
        self.input_std = nn.Parameter(torch.ones(dimension), requires_grad=False)
        self.output_mean = nn.Parameter(torch.zeros(dimension), requires_grad=False)
        self.output_std = nn.Parameter(torch.ones(dimension), requires_grad=False)
        self.use_residuals = nn.Parameter(torch.tensor(True, dtype=torch.bool), requires_grad=False)

    def forward(self, inputs):
        if inputs.ndim != 2 or inputs.shape[-1] != self.input_mean.numel():
            raise ValueError("invalid SCOPE transition input")
        if not torch.isfinite(inputs).all() or (self.input_std == 0).any():
            raise ValueError("non-finite SCOPE transition state/normalizer")
        scaled = (inputs - self.input_mean) / self.input_std
        output = self.model(scaled[:, None])[0][:, 0, :]
        output = output * self.output_std + self.output_mean
        if bool(self.use_residuals):
            output = inputs + output
        if not torch.isfinite(output).all():
            raise FloatingPointError("non-finite SCOPE transition output")
        return output


def _safe_checkpoint(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


class TransitionEnsemble:
    def __init__(self, root, device="cuda", dimension=1024):
        self.asset = verify_transition_manifest(root)
        self.root = Path(root).resolve()
        self.device = torch.device(device)
        self.dimension = dimension
        self.models = {direction: [] for direction in DIRECTIONS}
        for seed in SEEDS:
            for direction in DIRECTIONS:
                path = (
                    self.root
                    / f"seed_{seed}_batch_2048"
                    / direction
                    / "model_min_val.pth"
                )
                payload = _safe_checkpoint(path)
                if not isinstance(payload, dict) or "model_state_dict" not in payload:
                    raise ValueError(f"invalid SCOPE transition payload: {path}")
                model = RegressionWrapper(dimension).float()
                incompatible = model.load_state_dict(payload["model_state_dict"], strict=True)
                if incompatible.missing_keys or incompatible.unexpected_keys:
                    raise ValueError(f"SCOPE transition state mismatch: {path}")
                model.to(self.device).eval()
                self.models[direction].append(model)
        if any(len(models) != 4 for models in self.models.values()):
            raise ValueError("SCOPE requires exactly four models in each transition direction")

    @torch.inference_mode()
    def _forward(self, states, direction):
        states = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        outputs = torch.stack([model(states) for model in self.models[direction]], dim=0)
        return outputs

    def assistant_states(self, user_states):
        return self._forward(user_states, "human_llm")

    def user_states(self, assistant_states):
        return self._forward(assistant_states, "llm_human")

    def provenance(self):
        return {
            "transition_manifest_sha256": self.asset["sha256"],
            "transition_seeds": list(SEEDS),
            "directions": list(DIRECTIONS),
            "checkpoint_selection": "released model_min_val.pth",
            "architecture": "RegressionWrapper(HeirarchicalMoE(1024))",
        }
