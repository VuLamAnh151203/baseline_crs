import math

import torch
from torch import nn


class MRNFixed(nn.Module):
    """Euclidean symmetric component plus directed max positive residual (sym_p=1).

    Construction follows MRNFixed: https://github.com/quasimetric-learning/torch-quasimetric
    Shared state maps preserve the triangle inequality under composition.
    """

    head_type = "MRNFixed"

    def __init__(self, input_dim=4096, hidden_dim=176, latent_dim=16):
        super().__init__()
        self.sym = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, latent_dim)
        )
        self.asym = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, latent_dim)
        )
        self.checkpoint_hash = "untrained"
        self.encoder_provenance = {}

    def encode(self, states):
        return torch.cat((self.sym(states), self.asym(states)), dim=-1)

    def distance_encoded(self, source, target):
        a, b = (source - target).chunk(2, dim=-1)
        return torch.linalg.vector_norm(a, dim=-1) / math.sqrt(a.shape[-1]) + b.clamp_min(0).amax(
            -1
        )

    def distance(self, source, target):
        return self.distance_encoded(self.encode(source), self.encode(target))

    forward = distance


def interval_union(source, target):
    """Length of union of [source_i,target_i] where source_i < target_i.

    An endpoint sweep implements IQE's directed interval geometry. Arithmetic is
    accumulated in FP32 under BF16; comparisons/sorting retain exact endpoints.
    """
    source, target = torch.broadcast_tensors(source, target)
    n = source.shape[-1]
    endpoints, permutation = torch.cat((source, target), dim=-1).sort(dim=-1)
    valid = source < target
    valid_sorted = torch.gather(valid, -1, permutation % n)
    increments = valid_sorted * torch.where(permutation < n, 1, -1)
    active = increments.cumsum(-1)[..., :-1] > 0
    lengths = endpoints.float().diff(dim=-1)
    return (lengths * active).sum(-1)


class IQE(nn.Module):
    """IQE-maxmean with learned shared state map and convex component reduction."""

    head_type = "IQE"

    def __init__(self, input_dim=4096, hidden_dim=176, latent_dim=128, component_dim=16):
        super().__init__()
        if latent_dim % component_dim or component_dim < 8:
            raise ValueError("IQE components must divide latent size and have >=8 dimensions")
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, latent_dim)
        )
        self.mix = nn.Parameter(torch.zeros(()))
        self.components = latent_dim // component_dim
        self.component_dim = component_dim
        self.checkpoint_hash = "untrained"
        self.encoder_provenance = {}

    def encode(self, states):
        return self.encoder(states)

    def distance_encoded(self, source, target):
        a = source.unflatten(-1, (self.components, self.component_dim))
        b = target.unflatten(-1, (self.components, self.component_dim))
        c = interval_union(a, b)
        alpha = self.mix.sigmoid()
        return alpha * c.amax(-1) + (1 - alpha) * c.mean(-1)

    def distance(self, source, target):
        return self.distance_encoded(self.encode(source), self.encode(target))

    forward = distance


class CosineControl(nn.Module):
    head_type = "cosine_control"
    checkpoint_hash = "no_checkpoint"
    encoder_provenance = {}

    def distance(self, source, target):
        return (1 - nn.functional.cosine_similarity(source, target, dim=-1)).clamp_min(0)


def assert_invariants(head, states, atol=1e-5, rtol=1e-5, require_asymmetry=True):
    with torch.no_grad():
        d = head.distance(states[:, None], states[None, :]).float()
    if not torch.isfinite(d).all() or (d < -atol).any():
        raise ValueError("non-finite or negative distance")
    if d.diag().abs().max() > atol:
        raise ValueError("nonzero self distance")
    for k in range(len(states)):
        via = d[:, k, None] + d[k, None, :]
        if (d > via + atol + rtol * via.abs()).any():
            raise ValueError("triangle inequality failure")
    if require_asymmetry and torch.allclose(d, d.T, atol=atol, rtol=rtol):
        raise ValueError("symmetric/collapsed head")
    return {"max_asymmetry": (d - d.T).abs().max().item(), "states": len(states)}
