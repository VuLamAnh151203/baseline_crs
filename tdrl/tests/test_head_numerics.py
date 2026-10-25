import copy

import pytest
import torch

from tdrl.head_optimizer import HeadAdamW
from tdrl.learning import CostConfig, ReachabilityLearner
from tdrl.reachability import MRNFixed


def test_fp32_master_adamw_matches_torch_and_preserves_bf16_forward():
    p = torch.nn.Parameter(torch.tensor([1.0, -0.5], dtype=torch.bfloat16))
    ref = torch.nn.Parameter(p.detach().float().clone())
    actual = HeadAdamW([p], lr=1e-6)
    expected = torch.optim.AdamW([ref], lr=1e-6)
    for _ in range(20):
        grad = torch.tensor([0.25, -0.125], dtype=torch.bfloat16)
        p.grad = grad
        ref.grad = grad.float()
        actual.step()
        expected.step()
        assert torch.equal(actual.masters[0], ref)
        assert torch.equal(p, ref.bfloat16())
        assert p.dtype == torch.bfloat16
    assert not torch.equal(actual.masters[0], torch.tensor([1.0, -0.5]))
    restored_p = torch.nn.Parameter(p.detach().clone())
    restored = HeadAdamW([restored_p], lr=1e-6)
    restored.load_state_dict(copy.deepcopy(actual.state_dict()))
    for opt, param in [(actual, p), (restored, restored_p)]:
        param.grad = torch.tensor([0.125, 0.5], dtype=torch.bfloat16)
        opt.step()
    assert torch.equal(actual.masters[0], restored.masters[0])
    assert torch.equal(p, restored_p)
    assert all(s["exp_avg_sq"].dtype == torch.float32 for s in restored.state.values())


def test_target_ema_accumulates_changes_instead_of_freezing_bf16():
    head = MRNFixed(input_dim=2, hidden_dim=2, latent_dim=2).bfloat16()
    legacy = ReachabilityLearner(copy.deepcopy(head), CostConfig(target_tau=0.01))
    fixed = ReachabilityLearner(
        copy.deepcopy(head), CostConfig(target_tau=0.01), numerical_revision=2
    )
    with torch.no_grad():
        for learner in [legacy, fixed]:
            for p in learner.head.parameters():
                p.fill_(1.0078125)
            for p in learner.target.parameters():
                p.fill_(1.0)
        for master in fixed.target_master:
            master.fill_(1.0)
    for _ in range(1000):
        legacy.update_target()
        fixed.update_target()
    assert all(torch.all(p == 1) for p in legacy.target.parameters())
    assert all(
        torch.all(p == torch.tensor(1.0078125, dtype=torch.bfloat16))
        for p in fixed.target.parameters()
    )
    assert all(p.grad is None for p in fixed.target.parameters())
    resumed = ReachabilityLearner(copy.deepcopy(head), numerical_revision=2)
    resumed.load_state_dict(copy.deepcopy(fixed.state_dict()))
    assert all(
        torch.equal(a, b) for a, b in zip(resumed.target_master, fixed.target_master, strict=True)
    )
    with pytest.raises(ValueError, match="fresh fitting"):
        resumed.load_state_dict(legacy.state_dict())
