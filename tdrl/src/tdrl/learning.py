import copy
from dataclasses import dataclass

import torch
from torch import nn

from .head_optimizer import HeadAdamW


@dataclass(frozen=True)
class CostConfig:
    c_step: float = 1.0
    gamma: float = 1.0
    negative_margin: float = 10.0
    negative_weight: float = 1.0
    target_tau: float = 0.01

    def __post_init__(self):
        if self.c_step <= 0 or not 0 < self.gamma <= 1:
            raise ValueError("positive step cost and valid discount required")
        if self.negative_margin <= self.c_step or self.negative_weight <= 0:
            raise ValueError("invalid separation objective")
        if not 0 < self.target_tau <= 1:
            raise ValueError("invalid target update")


@dataclass(frozen=True)
class SuccessReward:
    accepted: float = 1.0
    unsuccessful: float = 0.0


@torch.no_grad()
def bellman_targets(target_head, next_states, goals, is_goal, next_is_goal, reachable, cfg):
    if not reachable.all():
        raise ValueError("failed/censored/unreachable targets require separation loss")
    bootstrap = target_head.distance(next_states, goals)
    if not torch.isfinite(bootstrap).all() or (bootstrap < 0).any():
        raise ValueError("invalid target distance")
    y = cfg.c_step + cfg.gamma * (~next_is_goal).to(bootstrap.dtype) * bootstrap
    return torch.where(is_goal, torch.zeros_like(y), y).detach()


class ReachabilityLearner:
    def __init__(self, head, cfg=None, lr=3e-4, numerical_revision=1):
        if numerical_revision not in {1, 2}:
            raise ValueError("unknown head numerical revision")
        self.numerical_revision = numerical_revision
        cfg = cfg or CostConfig()
        self.head = head
        self.target = copy.deepcopy(head).requires_grad_(False).eval()
        self.cfg = cfg
        self.optimizer = (
            HeadAdamW(head.parameters(), lr=lr)
            if numerical_revision == 2
            else torch.optim.AdamW(head.parameters(), lr=lr)
        )
        self.target_master = (
            [p.detach().float().clone() for p in self.target.parameters()]
            if numerical_revision == 2
            else None
        )
        self.steps = 0

    def step(self, source, successor, goal, is_goal, next_is_goal, reachable):
        self.optimizer.zero_grad(set_to_none=True)
        pred = self.head.distance(source, goal)
        pieces = []
        if reachable.any():
            y = bellman_targets(
                self.target,
                successor[reachable],
                goal[reachable],
                is_goal[reachable],
                next_is_goal[reachable],
                reachable[reachable],
                self.cfg,
            )
            pieces.append(nn.functional.smooth_l1_loss(pred[reachable], y))
        if (~reachable).any():
            pieces.append(
                self.cfg.negative_weight
                * (self.cfg.negative_margin - pred[~reachable]).clamp_min(0).mean()
            )
        if not pieces:
            raise ValueError("empty training batch")
        loss = sum(pieces)
        if not torch.isfinite(loss):
            raise ValueError("non-finite loss")
        loss.backward()
        for p in self.head.parameters():
            if p.grad is not None and not torch.isfinite(p.grad).all():
                raise ValueError("non-finite gradient")
        nn.utils.clip_grad_norm_(self.head.parameters(), 10)
        self.optimizer.step()
        self.update_target()
        self.steps += 1
        return float(loss.detach())

    @torch.no_grad()
    def update_target(self):
        for i, (target, online) in enumerate(
            zip(self.target.parameters(), self.head.parameters(), strict=True)
        ):
            if self.numerical_revision == 2:
                self.target_master[i].lerp_(online.float(), self.cfg.target_tau)
                target.copy_(self.target_master[i])
            else:
                target.lerp_(online, self.cfg.target_tau)

    def state_dict(self):
        return {
            "online": self.head.state_dict(),
            "target": self.target.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "steps": self.steps,
            "rng": torch.get_rng_state(),
            "numerical_revision": self.numerical_revision,
            "target_master": [p.clone() for p in self.target_master]
            if self.target_master is not None
            else None,
        }

    def load_state_dict(self, state):
        if state.get("numerical_revision", 1) != self.numerical_revision:
            raise ValueError("head numerical execution mismatch; fresh fitting required")
        if self.numerical_revision == 2:
            if len(state["target_master"]) != len(self.target_master):
                raise ValueError("target master count mismatch")
            for target, saved in zip(self.target_master, state["target_master"], strict=True):
                if saved.dtype != torch.float32 or saved.shape != target.shape:
                    raise ValueError("target master precision/shape mismatch")
                target.copy_(saved)
        self.head.load_state_dict(state["online"])
        self.target.load_state_dict(state["target"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.steps = state["steps"]
        torch.set_rng_state(state["rng"])
