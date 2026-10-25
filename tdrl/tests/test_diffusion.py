"""Pinned upstream arithmetic fixtures; CPU evidence does not satisfy H100 gates."""

import json
from pathlib import Path

import pytest
import torch

from tdrl.diffusion import (
    Inpainting,
    analytic_weights,
    forward_mask,
    loglinear_noise,
    sample_analytic,
    score_entropy,
    semantic_constraints,
)

FIXTURE = json.loads((Path(__file__).parent / "fixtures/difftod-arithmetic-v1.json").read_text())


def test_absorbing_loss_and_gradients_match_upstream_source():
    logits = torch.tensor(FIXTURE["logits"], requires_grad=True)
    sigma = torch.tensor(FIXTURE["sigma"])
    loss = score_entropy(
        logits, sigma, torch.tensor(FIXTURE["corrupted"]), torch.tensor(FIXTURE["clean"]), 4
    )
    gradient = torch.autograd.grad(loss.sum(), logits)[0]
    torch.testing.assert_close(loss, torch.tensor(FIXTURE["entropy"]), rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(
        gradient, torch.tensor(FIXTURE["entropy_gradient"]), rtol=1e-6, atol=1e-6
    )
    assert not gradient[torch.tensor(FIXTURE["corrupted"]) != 4].any()


def test_analytic_transition_matches_upstream_source():
    weights = analytic_weights(
        torch.tensor(FIXTURE["logits"]).exp(),
        torch.tensor(FIXTURE["corrupted"]),
        torch.tensor(FIXTURE["analytic_delta"]),
        4,
    )
    torch.testing.assert_close(weights, torch.tensor(FIXTURE["analytic_weights"]), rtol=0, atol=0)
    sigma, rate = loglinear_noise(torch.tensor([[0.15], [0.75]]))
    torch.testing.assert_close(sigma, torch.tensor(FIXTURE["sigma"]), rtol=0, atol=0)
    torch.testing.assert_close(rate, torch.tensor(FIXTURE["rate"]), rtol=0, atol=0)


def test_full_projected_sampling_matches_native_trajectory_and_counts():
    constraints = semantic_constraints([0, 1], [2, 3], length=200, vocabulary=4)
    calls = []

    def policy(tokens, sigma):
        calls.append(tokens.clone())
        assert torch.equal(tokens[:, constraints.positions], constraints.tokens.expand(2, -1))
        logits = torch.arange(1, 6, dtype=torch.float32).log().repeat(*tokens.shape, 1)
        return logits.scatter(-1, tokens[..., None], 0.0)

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(FIXTURE["sample_seed"])
        tokens = sample_analytic(policy, constraints, 4, 2, "cpu")
    assert torch.equal(tokens, torch.tensor(FIXTURE["sample_tokens"]))
    assert len(calls) == FIXTURE["sample_model_calls"] == 5
    assert (tokens < 4).all()


def test_forward_corruption_uses_explicit_rng_and_preserves_zero_noise():
    tokens = torch.arange(4).repeat(2, 10)
    assert torch.equal(forward_mask(tokens, torch.zeros(2, 1), 4), tokens)
    a = forward_mask(tokens, torch.ones(2, 1), 4, torch.Generator().manual_seed(17))
    b = forward_mask(tokens, torch.ones(2, 1), 4, torch.Generator().manual_seed(17))
    assert torch.equal(a, b) and (a == 4).any() and (a < 4).any()


@pytest.mark.parametrize(
    "time", [torch.tensor([-0.1]), torch.tensor([1.1]), torch.tensor([float("nan")])]
)
def test_unknown_noise_times_rejected(time):
    with pytest.raises(ValueError, match="diffusion time"):
        loglinear_noise(time)


def test_semantic_guidance_overflow_and_overlapping_constraints_fail_closed():
    with pytest.raises(ValueError, match="no silent truncation"):
        semantic_constraints([0, 1], [2, 3], length=195, vocabulary=4)
    with pytest.raises(ValueError, match="overlapping"):
        Inpainting(torch.tensor([0, 0]), torch.tensor([1, 2]), 10, 4)
    with pytest.raises(ValueError, match="never mask"):
        semantic_constraints([0], [4], vocabulary=4)
