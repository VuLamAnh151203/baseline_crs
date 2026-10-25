import copy

import torch
from test_diffusion_optimizer import fixture

from tdrl.diffusion_resume import restore_optimizer


def test_nonzero_checkpoint_is_immutable_across_repeated_next_updates():
    model, optimizer, _ = fixture()
    model.weight.grad = torch.tensor([[1.0, -2.0]], dtype=torch.bfloat16)
    optimizer.step()
    saved = copy.deepcopy(optimizer.state_dict())
    original = copy.deepcopy(saved)
    results = []
    for _ in range(2):
        restore_optimizer(optimizer, saved)
        optimizer.zero_grad()
        model.weight.grad = torch.tensor([[2.0, 1.0]], dtype=torch.bfloat16)
        optimizer.step()
        results.append(copy.deepcopy(optimizer.state_dict()))
    for key in saved["optimizer"]["state"]:
        for field, value in saved["optimizer"]["state"][key].items():
            assert torch.equal(value, original["optimizer"]["state"][key][field])
            assert torch.equal(
                results[0]["optimizer"]["state"][key][field],
                results[1]["optimizer"]["state"][key][field],
            )
    for field in ["masters", "ema"]:
        for a, b in zip(results[0][field], results[1][field], strict=True):
            assert torch.equal(a, b)
    assert results[0]["updates"] == results[1]["updates"] == 2
