"""Absorbing SEDD mechanics for a source-faithful DiffTOD adapter.

Algorithm source: DiffTOD fd46b2887a9c0d9e0e413623e79c72ee32b9b7c1,
sections graph_lib.Absorbing, noise_lib.LogLinearNoise and sampling analytic
predictor. This is policy arithmetic, not a validated/fitted baseline. Public
scientific runners must enforce H100/Slurm and immutable policy provenance.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class Inpainting:
    """Declared token constraints; unobserved conversation tokens remain free."""

    positions: torch.Tensor
    tokens: torch.Tensor
    length: int
    vocabulary: int

    def __post_init__(self):
        if self.length < 1 or self.vocabulary < 2:
            raise ValueError("invalid diffusion context/vocabulary")
        if self.positions.dtype != torch.long or self.tokens.dtype != torch.long:
            raise ValueError("inpainting positions/tokens must be integer IDs")
        if self.positions.ndim != 1 or self.tokens.ndim != 1:
            raise ValueError("inpainting constraints must be one dimensional")
        if len(self.positions) != len(self.tokens) or not len(self.positions):
            raise ValueError("empty/mismatched diffusion constraints")
        if len(torch.unique(self.positions)) != len(self.positions):
            raise ValueError("overlapping diffusion constraints")
        if (self.positions < 0).any() or (self.positions >= self.length).any():
            raise ValueError("guidance exceeds diffusion context; no silent truncation")
        if (self.tokens < 0).any() or (self.tokens >= self.vocabulary).any():
            raise ValueError("conditioning must use real vocabulary tokens, never mask IDs")

    def project(self, batch):
        if batch.ndim != 2 or batch.shape[1] != self.length or batch.dtype != torch.long:
            raise ValueError("diffusion batch contract mismatch")
        batch[:, self.positions.to(batch.device)] = self.tokens.to(batch.device)
        return batch


def semantic_constraints(profile_ids, guidance_ids, length=1024, vocabulary=50257):
    """Native TopDial offset: profile prefix, then target phrase at prefix+192."""
    if not profile_ids or not guidance_ids:
        raise ValueError("profile and semantic guidance cannot be empty")
    return Inpainting(
        torch.tensor(
            [
                *range(len(profile_ids)),
                *range(len(profile_ids) + 192, len(profile_ids) + 192 + len(guidance_ids)),
            ],
            dtype=torch.long,
        ),
        torch.tensor([*profile_ids, *guidance_ids], dtype=torch.long),
        length,
        vocabulary,
    )


def loglinear_noise(time, epsilon=1e-3):
    if (
        not 0 < epsilon < 1
        or not torch.isfinite(time).all()
        or (time < 0).any()
        or (time > 1).any()
    ):
        raise ValueError("invalid log-linear diffusion time")
    survival = 1 - (1 - epsilon) * time
    return -torch.log1p(-(1 - epsilon) * time), (1 - epsilon) / survival


def forward_mask(tokens, sigma, vocabulary, generator=None):
    """Independent absorbing corruption used for score-entropy training."""
    draws = torch.rand(tokens.shape, device=tokens.device, generator=generator)
    return torch.where(draws < 1 - torch.exp(-sigma), vocabulary, tokens)


def score_entropy(log_scores, sigma, corrupted, clean, vocabulary):
    """Per-token score entropy; only absorbing states contribute to loss."""
    masked = corrupted == vocabulary
    denominator = torch.where(sigma < 0.5, torch.expm1(sigma), torch.exp(sigma) - 1)
    ratio = denominator.expand_as(corrupted)[masked].reciprocal()
    values = log_scores[masked]
    target = values.gather(-1, clean[masked, None]).squeeze(-1)
    entropy = values[:, :vocabulary].exp().sum(-1) - ratio * target
    entropy = entropy + ratio * (ratio.log() - 1)
    # Native output accumulation uses FP32 even under BF16 policy execution.
    result = torch.zeros(corrupted.shape, device=corrupted.device, dtype=torch.float32)
    result[masked] = entropy.to(torch.float32)
    return result


def analytic_weights(scores, tokens, delta_sigma, vocabulary):
    """Unnormalized analytic reverse-transition weights, including mask state."""
    adjusted = scores.clone()
    correction = (1 - delta_sigma.exp()) * scores.sum(-1)
    adjusted *= delta_sigma.exp()[:, None]
    adjusted[..., vocabulary] += correction
    sigma = delta_sigma
    while sigma.ndim < tokens.ndim + 1:
        sigma = sigma.unsqueeze(-1)
    transition = (-sigma).exp() * F.one_hot(tokens, vocabulary + 1)
    transition += torch.where(tokens == vocabulary, 1 - (-sigma).squeeze(-1).exp(), 0)[..., None]
    return adjusted * transition


def categorical_race(weights, generator=None):
    """Native exponential-race draw; preserve RNG consumption and tie ordering."""
    draws = torch.rand(
        weights.shape, dtype=weights.dtype, device=weights.device, generator=generator
    )
    race = 1e-10 - (draws + 1e-10).log()
    if not torch.isfinite(weights).all():
        raise ValueError("nonfinite diffusion sampling weights")
    return (weights / race).argmax(-1)


@torch.no_grad()
def sample_analytic(
    log_score, constraints, steps, batch_size, device, generator=None, epsilon=1e-5
):
    """Projected analytic reverse diffusion and final mask-free denoising.

    log_score(tokens, sigma) is the frozen SEDD model's log score, not an LLM
    prompt. A returned plan requires exactly steps+1 policy forwards. The
    caller records policy timing/memory, RNG, weights and conditioning hashes.
    """
    if type(steps) is not int or steps < 1 or type(batch_size) is not int or batch_size < 1:
        raise ValueError("invalid diffusion sampling budget")
    if not 0 < epsilon < 1:
        raise ValueError("invalid diffusion sampling epsilon")
    tokens = torch.full(
        (batch_size, constraints.length), constraints.vocabulary, dtype=torch.long, device=device
    )
    times = torch.linspace(1, epsilon, steps + 1, device=device)
    increment = (1 - epsilon) / steps
    for time in times[:-1]:
        constraints.project(tokens)
        current = time * torch.ones(batch_size, 1, device=device)
        sigma = loglinear_noise(current)[0]
        next_sigma = loglinear_noise(current - increment)[0]
        scores = log_score(tokens, sigma.reshape(-1)).exp()
        weights = analytic_weights(scores, tokens, sigma - next_sigma, constraints.vocabulary)
        tokens = categorical_race(weights, generator)
    constraints.project(tokens)
    sigma = loglinear_noise(times[-1] * torch.ones(batch_size, 1, device=device))[0]
    scores = log_score(tokens, sigma.reshape(-1)).exp()
    weights = analytic_weights(scores, tokens, sigma, constraints.vocabulary)
    tokens = categorical_race(weights[..., : constraints.vocabulary], generator)
    return constraints.project(tokens)
