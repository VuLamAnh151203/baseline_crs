"""Native score-entropy learning and serializable shuffled token-batch membership."""

import copy
from contextlib import contextmanager

import numpy as np
import torch

from .diffusion import forward_mask, loglinear_noise, score_entropy


def policy_loss(model, clean, generator=None):
    time = 0.001 + 0.999 * torch.rand(clean.shape[0], device=clean.device, generator=generator)
    sigma, rate = loglinear_noise(time)
    corrupted = forward_mask(clean, sigma[:, None], model.vocabulary, generator)
    scores = model(corrupted, sigma)
    loss = (
        (score_entropy(scores, sigma[:, None], corrupted, clean, model.vocabulary) * rate[:, None])
        .sum(-1)
        .mean()
    )
    if not torch.isfinite(loss):
        raise ValueError("nonfinite native diffusion policy loss")
    return loss


class ChunkBatches:
    def __init__(self, count, batch_size, seed=17):
        if count < batch_size or batch_size < 1:
            raise ValueError("insufficient diffusion training chunks")
        self.count, self.batch_size = count, batch_size
        self.rng = np.random.default_rng(seed)
        self.order = self.rng.permutation(count)
        self.position, self.epoch = 0, 0

    def next(self):
        # Like a shuffled drop-last DataLoader; no cross-epoch mixed batches.
        if self.position + self.batch_size > self.count:
            self.order = self.rng.permutation(self.count)
            self.position = 0
            self.epoch += 1
        result = self.order[self.position : self.position + self.batch_size].copy()
        self.position += self.batch_size
        return result

    def state_dict(self):
        return {
            "schema_version": 1,
            "count": self.count,
            "batch_size": self.batch_size,
            "rng": copy.deepcopy(self.rng.bit_generator.state),
            "order": self.order.copy(),
            "position": self.position,
            "epoch": self.epoch,
        }

    def load_state_dict(self, state):
        if state.get("schema_version") != 1 or (state["count"], state["batch_size"]) != (
            self.count,
            self.batch_size,
        ):
            raise ValueError("diffusion batch sampler contract mismatch")
        if (
            sorted(state["order"].tolist()) != list(range(self.count))
            or not 0 <= state["position"] <= self.count
        ):
            raise ValueError("invalid diffusion batch permutation/position")
        self.rng.bit_generator.state = copy.deepcopy(state["rng"])
        self.order = state["order"].copy()
        self.position, self.epoch = state["position"], state["epoch"]


@contextmanager
def ema_policy(model, optimizer):
    training = model.training
    with torch.no_grad():
        for (_, parameter), shadow in zip(optimizer.named, optimizer.ema, strict=True):
            parameter.copy_(shadow)
    model.eval()
    try:
        yield model
    finally:
        with torch.no_grad():
            for (_, parameter), master in zip(optimizer.named, optimizer.masters, strict=True):
                parameter.copy_(master)
        model.train(training)


@torch.no_grad()
def validation_loss(model, chunks, batch_size, seed=17):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    total = 0.0
    for offset in range(0, len(chunks), batch_size):
        clean = torch.tensor(
            chunks[offset : offset + batch_size].copy(), device="cuda", dtype=torch.long
        )
        loss = policy_loss(model, clean, generator)
        total += float(loss) * len(clean)
    return total / len(chunks)
