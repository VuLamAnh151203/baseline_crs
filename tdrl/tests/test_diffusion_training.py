import copy

import pytest
import torch

from tdrl.diffusion_optimizer import DiffusionOptimizer
from tdrl.diffusion_training import ChunkBatches, ema_policy


def test_shuffled_membership_exact_resume_at_epoch_boundary():
    original = ChunkBatches(7, 3)
    first = original.next()
    original.next()
    state = original.state_dict()
    resumed = ChunkBatches(7, 3, seed=29)
    resumed.load_state_dict(state)
    for _ in range(5):
        assert (original.next() == resumed.next()).all()
    assert len(set(first)) == 3
    broken = copy.deepcopy(state)
    broken["order"][:] = 0
    with pytest.raises(ValueError, match="permutation"):
        resumed.load_state_dict(broken)


def test_ema_validation_restores_online_weights_even_on_exception():
    from test_diffusion_optimizer import CONFIG

    model = torch.nn.Linear(2, 1).bfloat16().train()
    optimizer = DiffusionOptimizer(model, CONFIG)
    original = model.weight.detach().clone()
    optimizer.ema[0].add_(1)
    with pytest.raises(RuntimeError, match="fixture"):
        with ema_policy(model, optimizer):
            assert not model.training and not torch.equal(model.weight, original)
            raise RuntimeError("fixture")
    assert model.training and torch.equal(model.weight, original)
