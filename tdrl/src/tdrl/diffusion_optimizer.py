"""Native Torch AdamW/warmup/EMA math with explicit FP32 auxiliary states.

BF16 parameters are published after each update. EMA averages FP32 optimizer
masters, preserving the native full-precision optimization trajectory; it is
published in BF16 only for policy evaluation. This numerical execution differs
from upstream FP32 parameters with mixed autocast and must be disclosed.
"""

import torch


class DiffusionOptimizer:
    def __init__(self, model, config, initial_weights=None):
        self.named = list(model.named_parameters())
        if initial_weights is None:
            initial_weights = getattr(model, "initial_optimizer_weights", None)
        self.signature = [(name, list(p.shape)) for name, p in self.named]
        self.config = dict(config)
        self.updates = 0
        self.masters = [
            torch.nn.Parameter(
                (initial_weights[name] if initial_weights is not None else p.detach())
                .to(device=p.device, dtype=torch.float32)
                .clone()
            )
            for name, p in self.named
        ]
        self.optimizer = torch.optim.AdamW(
            self.masters,
            lr=config["lr"],
            weight_decay=config["weight_decay"],
            betas=(config["beta1"], config["beta2"]),
            eps=config["eps"],
            foreach=False,
            fused=False,
        )
        self.ema = [p.detach().clone() for p in self.masters]
        if hasattr(model, "initial_optimizer_weights"):
            del model.initial_optimizer_weights

    def zero_grad(self):
        self.optimizer.zero_grad(set_to_none=True)
        for _, parameter in self.named:
            parameter.grad = None

    @torch.no_grad()
    def step(self):
        for (_, parameter), master in zip(self.named, self.masters, strict=True):
            master.grad = parameter.grad.detach().float() if parameter.grad is not None else None
        if not all(p.grad is None or torch.isfinite(p.grad).all() for p in self.masters):
            raise ValueError("nonfinite diffusion gradients; no update applied")
        if self.config["grad_clip"] >= 0:
            torch.nn.utils.clip_grad_norm_(
                self.masters, self.config["grad_clip"], error_if_nonfinite=True
            )
        self.updates += 1
        scale = min(self.updates / self.config["warmup"], 1) if self.config["warmup"] > 0 else 1
        for group in self.optimizer.param_groups:
            group["lr"] = self.config["lr"] * scale
        self.optimizer.step()
        decay = min(self.config["ema_decay"], (1 + self.updates) / (10 + self.updates))
        for (_, parameter), master, shadow in zip(self.named, self.masters, self.ema, strict=True):
            parameter.copy_(master)
            shadow.sub_((1 - decay) * (shadow - master))

    def state_dict(self):
        return {
            "schema_version": 1,
            "signature": self.signature,
            "config": self.config,
            "updates": self.updates,
            "masters": [p.detach().clone() for p in self.masters],
            "ema": [p.clone() for p in self.ema],
            "optimizer": self.optimizer.state_dict(),
        }

    @torch.no_grad()
    def load_state_dict(self, state):
        if (
            state.get("schema_version") != 1
            or state["signature"] != self.signature
            or state["config"] != self.config
        ):
            raise ValueError("diffusion optimizer provenance/version mismatch")
        for field in ["masters", "ema"]:
            if len(state[field]) != len(self.masters):
                raise ValueError("diffusion optimizer state count mismatch")
            for saved, master in zip(state[field], self.masters, strict=True):
                if (
                    saved.dtype != torch.float32
                    or saved.shape != master.shape
                    or not torch.isfinite(saved).all()
                ):
                    raise ValueError("invalid diffusion optimizer precision/shape/content")
        if type(state["updates"]) is not int or state["updates"] < 0:
            raise ValueError("invalid diffusion update counter")
        for moments in state["optimizer"]["state"].values():
            for name in ["exp_avg", "exp_avg_sq"]:
                value = moments[name]
                if value.dtype != torch.float32 or not torch.isfinite(value).all():
                    raise ValueError("invalid diffusion moment precision/content")
        self.optimizer.load_state_dict(state["optimizer"])
        self.updates = state["updates"]
        for (_, parameter), master, shadow, saved, average in zip(
            self.named, self.masters, self.ema, state["masters"], state["ema"], strict=True
        ):
            master.copy_(saved)
            shadow.copy_(average)
            parameter.copy_(master)
