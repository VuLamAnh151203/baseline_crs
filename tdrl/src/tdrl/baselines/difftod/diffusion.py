from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


def sample_categorical(probabilities, generator=None):
    if probabilities.ndim < 1 or probabilities.shape[-1] < 1:
        raise ValueError("categorical probabilities require a nonempty class dimension")
    if not torch.isfinite(probabilities).all() or (probabilities < 0).any():
        raise ValueError("categorical probabilities must be finite and nonnegative")
    if (probabilities.sum(dim=-1) <= 0).any():
        raise ValueError("categorical probabilities have zero mass")
    uniform = torch.rand(
        probabilities.shape,
        dtype=probabilities.dtype,
        device=probabilities.device,
        generator=generator,
    )
    gumbel_norm = 1e-10 - (uniform + 1e-10).log()
    return (probabilities / gumbel_norm).argmax(dim=-1)


class LogLinearNoise(nn.Module):
    def __init__(self, epsilon=1e-3):
        super().__init__()
        if not 0 < epsilon < 1:
            raise ValueError("noise epsilon must be in (0, 1)")
        self.epsilon = epsilon
        # Retain the released checkpoint/state shape even though this scalar is inert.
        self.empty = nn.Parameter(torch.tensor(0.0))

    def total_noise(self, time):
        return -torch.log1p(-(1 - self.epsilon) * time)

    def rate_noise(self, time):
        return (1 - self.epsilon) / (1 - (1 - self.epsilon) * time)

    def forward(self, time):
        return self.total_noise(time), self.rate_noise(time)


class AbsorbingGraph:
    def __init__(self, vocab_size):
        if not isinstance(vocab_size, int) or vocab_size < 2:
            raise ValueError("vocabulary must contain at least two ordinary tokens")
        self.vocab_size = vocab_size

    @property
    def dim(self):
        return self.vocab_size + 1

    @property
    def mask_id(self):
        return self.dim - 1

    def sample_transition(self, clean, sigma, generator=None):
        move_chance = 1 - (-sigma).exp()
        random = torch.rand(clean.shape, device=clean.device, generator=generator)
        moved = random < move_chance
        return torch.where(moved, self.mask_id, clean)

    def sample_limit(self, *shape, device=None):
        return torch.full(shape, self.mask_id, dtype=torch.long, device=device)

    def transp_transition(self, tokens, sigma):
        sigma = sigma.reshape(len(tokens), *([1] * tokens.ndim))
        edge = (-sigma).exp() * F.one_hot(tokens, num_classes=self.dim)
        addition = torch.where(
            tokens == self.mask_id,
            1 - (-sigma.squeeze(-1)).exp(),
            0,
        )[..., None]
        return edge + addition

    def staggered_score(self, score, delta_sigma):
        score = score.clone()
        multiplier = delta_sigma.exp().reshape(len(score), 1, 1)
        extra = (1 - delta_sigma.exp()).reshape(len(score), 1) * score.sum(dim=-1)
        score *= multiplier
        score[..., -1] += extra
        return score

    def score_entropy(self, log_score, sigma, noisy, clean):
        masked = noisy == self.mask_id
        entropy = torch.zeros_like(noisy, dtype=log_score.dtype)
        if not masked.any():
            return entropy
        expm1 = torch.where(sigma < 0.5, torch.expm1(sigma), torch.exp(sigma) - 1)
        ratio = 1 / expm1.expand_as(noisy)[masked]
        clean_indices = clean[masked]
        selected = torch.gather(log_score[masked], -1, clean_indices[..., None]).squeeze(-1)
        negative = ratio * selected
        positive = log_score[masked][:, :-1].exp().sum(dim=-1)
        constant = ratio * (ratio.log() - 1)
        entropy[masked] = positive - negative + constant
        return entropy


def score_entropy_loss(model, clean, graph, noise, *, time=None, generator=None):
    if time is None:
        time = (
            0.999 * torch.rand(len(clean), device=clean.device, generator=generator) + 0.001
        )
    if time.shape != (len(clean),) or (time <= 0).any() or (time > 1).any():
        raise ValueError("diffusion time must have shape [batch] and lie in (0, 1]")
    sigma, derivative = noise(time)
    noisy = graph.sample_transition(clean, sigma[:, None], generator)
    log_score = model(noisy, sigma)
    token_loss = graph.score_entropy(log_score, sigma[:, None], noisy, clean)
    return (derivative[:, None] * token_loss).sum(dim=-1)


@dataclass(frozen=True)
class TokenProjection:
    mask: torch.Tensor
    values: torch.Tensor

    def __post_init__(self):
        if self.mask.dtype != torch.bool or self.values.dtype != torch.long:
            raise TypeError("projection requires boolean mask and long token values")
        if self.mask.shape != self.values.shape:
            raise ValueError("projection mask/value shapes differ")

    def __call__(self, tokens):
        if tokens.shape != self.mask.shape or tokens.device != self.mask.device:
            raise ValueError("projection and sampled tokens must share shape/device")
        return torch.where(self.mask, self.values, tokens)


@torch.no_grad()
def sample_trajectory(
    model,
    graph,
    noise,
    batch_size,
    length,
    *,
    steps=1_024,
    epsilon=1e-5,
    projection=None,
    generator=None,
    denoise=True,
):
    if steps < 1 or batch_size < 1 or length < 1 or not 0 < epsilon < 1:
        raise ValueError("invalid trajectory sampling configuration")
    device = next(model.parameters()).device
    project = projection or (lambda value: value)
    tokens = graph.sample_limit(batch_size, length, device=device)
    timesteps = torch.linspace(1, epsilon, steps + 1, device=device)
    step_size = (1 - epsilon) / steps
    was_training = model.training
    model.eval()
    try:
        for index in range(steps):
            tokens = project(tokens)
            time = timesteps[index].expand(batch_size)
            current_sigma = noise(time)[0]
            next_time = torch.clamp(time - step_size, min=epsilon)
            next_sigma = noise(next_time)[0]
            delta = current_sigma - next_sigma
            score = model(tokens, current_sigma).exp()
            probabilities = graph.staggered_score(score, delta)
            probabilities *= graph.transp_transition(tokens, delta)
            tokens = sample_categorical(probabilities, generator)
        if denoise:
            tokens = project(tokens)
            time = timesteps[-1].expand(batch_size)
            sigma = noise(time)[0]
            score = model(tokens, sigma).exp()
            probabilities = graph.staggered_score(score, sigma)
            probabilities *= graph.transp_transition(tokens, sigma)
            tokens = sample_categorical(probabilities[..., :-1], generator)
        return project(tokens)
    finally:
        model.train(was_training)
