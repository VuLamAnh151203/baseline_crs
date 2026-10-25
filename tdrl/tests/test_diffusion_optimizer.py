import copy

import pytest
import torch

from tdrl.diffusion_optimizer import DiffusionOptimizer

CONFIG = {
    "lr": 0.04,
    "weight_decay": 0.0,
    "beta1": 0.9,
    "beta2": 0.999,
    "eps": 1e-8,
    "warmup": 3,
    "grad_clip": 0.7,
    "ema_decay": 0.9999,
}


def fixture():
    model = torch.nn.Linear(2, 1, bias=False).bfloat16()
    initial = torch.tensor([[1.005, -2.012]])
    with torch.no_grad():
        model.weight.copy_(initial)
    model.initial_optimizer_weights = {"weight": initial}
    return model, DiffusionOptimizer(model, CONFIG), initial


def test_native_adamw_warmup_and_adaptive_ema_preserve_fp32_masters():
    model, optimizer, initial = fixture()
    reference = torch.nn.Parameter(initial.clone())
    adamw = torch.optim.AdamW(
        [reference],
        lr=CONFIG["lr"],
        weight_decay=0,
        betas=(0.9, 0.999),
        eps=1e-8,
        foreach=False,
        fused=False,
    )
    shadow = initial.clone()
    assert not hasattr(model, "initial_optimizer_weights")
    for step, gradient in enumerate([[1.0, -2.0], [-0.5, 3.0], [2.0, 0.5]], 1):
        optimizer.zero_grad()
        model.weight.grad = torch.tensor([gradient], dtype=torch.bfloat16)
        reference.grad = model.weight.grad.float().clone()
        torch.nn.utils.clip_grad_norm_([reference], 0.7)
        adamw.param_groups[0]["lr"] = CONFIG["lr"] * min(step / 3, 1)
        adamw.step()
        decay = min(0.9999, (1 + step) / (10 + step))
        with torch.no_grad():
            shadow.sub_((1 - decay) * (shadow - reference))
        optimizer.step()
        assert torch.equal(optimizer.masters[0], reference)
        assert torch.equal(optimizer.ema[0], shadow)
        assert torch.equal(model.weight, reference.bfloat16())
        assert optimizer.ema[0].dtype == torch.float32


def test_durable_optimizer_resume_matches_next_update_exactly():
    model, optimizer, _ = fixture()
    model.weight.grad = torch.tensor([[1.0, -2.0]], dtype=torch.bfloat16)
    optimizer.step()
    saved = copy.deepcopy(optimizer.state_dict())
    resumed_model, resumed, _ = fixture()
    resumed.load_state_dict(saved)
    for actor, policy in [(model, optimizer), (resumed_model, resumed)]:
        policy.zero_grad()
        actor.weight.grad = torch.tensor([[2.0, 1.0]], dtype=torch.bfloat16)
        policy.step()
    assert torch.equal(model.weight, resumed_model.weight)
    assert torch.equal(optimizer.masters[0], resumed.masters[0])
    assert torch.equal(optimizer.ema[0], resumed.ema[0])
    assert optimizer.updates == resumed.updates == 2
    corrupted = copy.deepcopy(saved)
    corrupted["masters"][0] = corrupted["masters"][0].bfloat16()
    with pytest.raises(ValueError, match="precision"):
        resumed.load_state_dict(corrupted)


def test_nonfinite_gradient_is_rejected_before_update():
    model, optimizer, _ = fixture()
    original = model.weight.detach().clone()
    model.weight.grad = torch.tensor([[float("nan"), 1.0]], dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="nonfinite"):
        optimizer.step()
    assert optimizer.updates == 0 and torch.equal(model.weight, original)
