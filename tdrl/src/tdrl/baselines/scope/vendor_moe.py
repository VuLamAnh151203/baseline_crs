"""Checkpoint-compatible subset of lucidrains/mixture-of-experts.

Derived from https://github.com/lucidrains/mixture-of-experts (MIT,
Copyright 2020 Phil Wang). Names and parameter layout intentionally match the
SCOPE release checkpoints. Only the hierarchical inference architecture is
included here; no external package or mutable dependency is used.
"""

import math
from inspect import isfunction

import torch
import torch.nn.functional as functional
from torch import nn

MIN_EXPERT_CAPACITY = 4


def _default(value, default_value):
    default_value = default_value() if isfunction(default_value) else default_value
    return value if value is not None else default_value


def _cast_tuple(value):
    return value if isinstance(value, tuple) else (value,)


def _top1(tensor):
    values, index = tensor.topk(k=1, dim=-1)
    return values.squeeze(dim=-1), index.squeeze(dim=-1)


def _cumsum_exclusive(tensor, dim=-1):
    padding_dimensions = -dim - 1
    pre_padding = (0, 0) * padding_dimensions
    pre_slice = (slice(None),) * padding_dimensions
    padded = functional.pad(tensor, (*pre_padding, 1, 0)).cumsum(dim=dim)
    return padded[(..., slice(None, -1), *pre_slice)]


def _safe_one_hot(indexes, maximum_length):
    maximum_index = indexes.max() + 1
    return functional.one_hot(indexes, max(maximum_index + 1, maximum_length))[...,
        :maximum_length
    ]


def _initialize(tensor):
    dimension = tensor.shape[-1]
    return tensor.uniform_(-1 / math.sqrt(dimension), 1 / math.sqrt(dimension))


class Experts(nn.Module):
    def __init__(self, dim, num_experts=(4, 4), hidden_dim=None, activation=nn.ReLU):
        super().__init__()
        hidden_dim = _default(hidden_dim, dim * 4)
        num_experts = _cast_tuple(num_experts)
        self.w1 = nn.Parameter(_initialize(torch.zeros(*num_experts, dim, hidden_dim)))
        self.w2 = nn.Parameter(_initialize(torch.zeros(*num_experts, hidden_dim, dim)))
        self.act = activation()

    def forward(self, inputs):
        hidden = torch.einsum("...nd,...dh->...nh", inputs, self.w1)
        hidden = self.act(hidden)
        return torch.einsum("...nh,...hd->...nd", hidden, self.w2)


class Top2Gating(nn.Module):
    def __init__(
        self,
        dim,
        num_gates,
        eps=1e-9,
        outer_expert_dims=tuple(),
        second_policy_train="random",
        second_policy_eval="random",
        second_threshold_train=0.2,
        second_threshold_eval=0.2,
        capacity_factor_train=1.25,
        capacity_factor_eval=2.0,
    ):
        super().__init__()
        self.eps = eps
        self.num_gates = num_gates
        self.w_gating = nn.Parameter(torch.randn(*outer_expert_dims, dim, num_gates))
        self.second_policy_train = second_policy_train
        self.second_policy_eval = second_policy_eval
        self.second_threshold_train = second_threshold_train
        self.second_threshold_eval = second_threshold_eval
        self.capacity_factor_train = capacity_factor_train
        self.capacity_factor_eval = capacity_factor_eval

    def forward(self, inputs, importance=None):
        *_, _batch, group_size, _dim = inputs.shape
        num_gates = self.num_gates
        if self.training:
            policy = self.second_policy_train
            threshold = self.second_threshold_train
            capacity_factor = self.capacity_factor_train
        else:
            policy = self.second_policy_eval
            threshold = self.second_threshold_eval
            capacity_factor = self.capacity_factor_eval
        raw_gates = torch.einsum("...bnd,...de->...bne", inputs, self.w_gating).softmax(
            dim=-1
        )
        gate_1, index_1 = _top1(raw_gates)
        mask_1 = functional.one_hot(index_1, num_gates).float()
        density_1_proxy = raw_gates
        if importance is not None:
            equals_one = (importance == 1.0).float()
            mask_1 *= equals_one[..., None]
            gate_1 *= equals_one
            density_1_proxy *= equals_one[..., None]
        gate_2, index_2 = _top1(raw_gates * (1.0 - mask_1))
        mask_2 = functional.one_hot(index_2, num_gates).float()
        if importance is not None:
            mask_2 *= (importance > 0.0).float()[..., None]
        denominator = gate_1 + gate_2 + self.eps
        gate_1 /= denominator
        gate_2 /= denominator
        density_1 = mask_1.mean(dim=-2)
        density_1_proxy = density_1_proxy.mean(dim=-2)
        loss = (density_1_proxy * density_1).mean() * float(num_gates**2)
        if policy == "none":
            mask_2 = torch.zeros_like(mask_2)
        elif policy == "threshold":
            mask_2 *= (gate_2 > threshold).float()
        elif policy == "random":
            probabilities = torch.zeros_like(gate_2).uniform_(0.0, 1.0)
            mask_2 *= (
                probabilities < (gate_2 / max(threshold, self.eps))
            ).float().unsqueeze(-1)
        elif policy != "all":
            raise ValueError(f"unknown second-expert policy: {policy}")
        capacity = max(
            min(group_size, int(group_size * capacity_factor / num_gates)),
            MIN_EXPERT_CAPACITY,
        )
        position_1 = _cumsum_exclusive(mask_1, dim=-2) * mask_1
        mask_1 *= (position_1 < float(capacity)).float()
        mask_1_count = mask_1.sum(dim=-2, keepdim=True)
        mask_1_flat = mask_1.sum(dim=-1)
        position_1 = position_1.sum(dim=-1)
        gate_1 *= mask_1_flat
        position_2 = (_cumsum_exclusive(mask_2, dim=-2) + mask_1_count) * mask_2
        mask_2 *= (position_2 < float(capacity)).float()
        mask_2_flat = mask_2.sum(dim=-1)
        position_2 = position_2.sum(dim=-1)
        gate_2 *= mask_2_flat
        combine = (
            gate_1[..., None, None]
            * mask_1_flat[..., None, None]
            * functional.one_hot(index_1, num_gates)[..., None]
            * _safe_one_hot(position_1.long(), capacity)[..., None, :]
            + gate_2[..., None, None]
            * mask_2_flat[..., None, None]
            * functional.one_hot(index_2, num_gates)[..., None]
            * _safe_one_hot(position_2.long(), capacity)[..., None, :]
        )
        return combine.bool().to(combine), combine, loss


class HeirarchicalMoE(nn.Module):
    """Misspelling retained because it is the upstream public class name."""

    def __init__(
        self,
        dim,
        num_experts=(4, 4),
        hidden_dim=None,
        activation=nn.ReLU,
        second_policy_train="random",
        second_policy_eval="random",
        second_threshold_train=0.2,
        second_threshold_eval=0.2,
        capacity_factor_train=1.25,
        capacity_factor_eval=2.0,
        loss_coef=1e-2,
        experts=None,
    ):
        super().__init__()
        if len(num_experts) != 2:
            raise ValueError("SCOPE checkpoints require two hierarchical expert levels")
        outer, inner = num_experts
        self.num_experts_outer = outer
        self.num_experts_inner = inner
        gating = {
            "second_policy_train": second_policy_train,
            "second_policy_eval": second_policy_eval,
            "second_threshold_train": second_threshold_train,
            "second_threshold_eval": second_threshold_eval,
            "capacity_factor_train": capacity_factor_train,
            "capacity_factor_eval": capacity_factor_eval,
        }
        self.gate_outer = Top2Gating(dim, num_gates=outer, **gating)
        self.gate_inner = Top2Gating(
            dim, num_gates=inner, outer_expert_dims=(outer,), **gating
        )
        self.experts = _default(
            experts,
            lambda: Experts(
                dim, num_experts=num_experts, hidden_dim=hidden_dim, activation=activation
            ),
        )
        self.loss_coef = loss_coef

    def forward(self, inputs, **_kwargs):
        batch, length, dim = inputs.shape
        outer, inner = self.num_experts_outer, self.num_experts_inner
        dispatch_outer, combine_outer, loss_outer = self.gate_outer(inputs)
        expert_outer = torch.einsum("bnd,bnec->ebcd", inputs, dispatch_outer)
        importance = combine_outer.permute(2, 0, 3, 1).sum(dim=-1)
        importance = 0.5 * ((importance > 0.5).float() + (importance > 0.0).float())
        dispatch_inner, combine_inner, loss_inner = self.gate_inner(
            expert_outer, importance=importance
        )
        expert_inputs = torch.einsum("ebnd,ebnfc->efbcd", expert_outer, dispatch_inner)
        original_shape = expert_inputs.shape
        expert_inputs = expert_inputs.reshape(outer, inner, -1, dim)
        expert_outputs = self.experts(expert_inputs).reshape(*original_shape)
        expert_outputs_outer = torch.einsum(
            "efbcd,ebnfc->ebnd", expert_outputs, combine_inner
        )
        output = torch.einsum("ebcd,bnec->bnd", expert_outputs_outer, combine_outer)
        if output.shape != (batch, length, dim):
            raise RuntimeError("invalid hierarchical MoE output shape")
        return output, (loss_outer + loss_inner) * self.loss_coef
