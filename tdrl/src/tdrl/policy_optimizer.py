"""Explicit legacy HF AdamW math; BF16 forward weights with FP32 optimizer masters."""

import math

import torch


class PolicyAdamW(torch.optim.Optimizer):
    def __init__(self, params, lr, eps, weight_decay=0.0):
        super().__init__(
            params, dict(lr=lr, eps=eps, weight_decay=weight_decay, betas=(0.9, 0.999))
        )

    def reserve(self):
        for group in self.param_groups:
            for parameter in group["params"]:
                if not self.state[parameter]:
                    self.state[parameter].update(
                        step=0,
                        master=parameter.detach().float().clone(),
                        exp_avg=torch.zeros_like(parameter, dtype=torch.float32),
                        exp_avg_sq=torch.zeros_like(parameter, dtype=torch.float32),
                    )

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            raise ValueError("policy optimizer does not use closures")
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                grad = parameter.grad.float()
                if grad.is_sparse:
                    raise ValueError("sparse policy gradient unsupported")
                state = self.state[parameter]
                if not state:
                    state.update(
                        step=0,
                        master=parameter.detach().float().clone(),
                        exp_avg=torch.zeros_like(parameter, dtype=torch.float32),
                        exp_avg_sq=torch.zeros_like(parameter, dtype=torch.float32),
                    )
                state["step"] += 1
                state["exp_avg"].mul_(beta1).add_(grad, alpha=1 - beta1)
                state["exp_avg_sq"].mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                denom = state["exp_avg_sq"].sqrt().add_(group["eps"])
                scale = (
                    group["lr"]
                    * math.sqrt(1 - beta2 ** state["step"])
                    / (1 - beta1 ** state["step"])
                )
                state["master"].addcdiv_(state["exp_avg"], denom, value=-scale)
                state["master"].mul_(1 - group["lr"] * group["weight_decay"])
                parameter.copy_(state["master"])

    def load_state_dict(self, saved):
        # torch's generic loader casts floating state to BF16 parameter dtype.
        super().load_state_dict(saved)
        for group, stored_group in zip(self.param_groups, saved["param_groups"], strict=True):
            for parameter, key in zip(group["params"], stored_group["params"], strict=True):
                if key in saved["state"]:
                    self.state[parameter] = {
                        name: value.to(parameter.device, torch.float32)
                        if isinstance(value, torch.Tensor)
                        else value
                        for name, value in saved["state"][key].items()
                    }


def policy_optimizer(policy, cfg, phase):
    if phase == "sft":
        groups = [
            {
                "params": [
                    p
                    for n, p in policy.named_parameters()
                    if not any(k in n for k in ("bias", "LayerNorm.weight"))
                ],
                "weight_decay": 0.01,
            },
            {
                "params": [
                    p
                    for n, p in policy.named_parameters()
                    if any(k in n for k in ("bias", "LayerNorm.weight"))
                ],
                "weight_decay": 0.0,
            },
        ]
        return PolicyAdamW(groups, cfg["sft_lr"], eps=1e-8)
    return PolicyAdamW(policy.parameters(), cfg["rl_lr"], eps=1e-6, weight_decay=0.0)
