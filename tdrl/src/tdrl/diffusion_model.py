"""Checkpoint-compatible SEDD diffusion transformer, without remote-code loaders.

DDiT architecture follows the pinned DiffTOD SEDD-medium policy. Attention uses
PyTorch SDPA rather than the upstream FlashAttention entrypoint; this execution
adaptation needs explicit native/H100 fidelity and exact replay gates. Tiny CPU
fixtures only establish architecture/control flow, never pretrained inference.
"""

import math
from contextlib import nullcontext

import torch
from torch import nn
from torch.nn import functional as F


class Norm(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))

    def forward(self, values):
        with torch.autocast(device_type=values.device.type, enabled=False):
            normalized = F.layer_norm(values.float(), (values.shape[-1],))
        return normalized * self.weight


class SigmaEmbedding(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(256, width), nn.SiLU(), nn.Linear(width, width))

    def forward(self, sigma):
        frequencies = torch.exp(
            -math.log(10000) * torch.arange(128, device=sigma.device, dtype=torch.float32) / 128
        )
        phase = sigma.float()[:, None] * frequencies[None]
        return self.mlp(torch.cat((phase.cos(), phase.sin()), dim=-1))


class Rotary(nn.Module):
    def __init__(self, dimension):
        super().__init__()
        if dimension % 2:
            raise ValueError("SEDD rotary head dimension must be even")
        self.register_buffer(
            "inv_freq", 1 / (10000 ** (torch.arange(0, dimension, 2).float() / dimension))
        )

    def _apply(self, fn, recurse=True):
        # Native frequency arithmetic remains FP32 when parameters publish BF16.
        original = self.inv_freq.detach().float().clone()
        super()._apply(fn, recurse=recurse)
        self.inv_freq = original.to(device=self.inv_freq.device, dtype=torch.float32)
        return self

    def forward(self, qkv):
        positions = torch.arange(qkv.shape[1], device=qkv.device, dtype=torch.float32)
        phase = torch.einsum("i,j->ij", positions, self.inv_freq)
        phase = torch.cat((phase, phase), dim=-1)
        cosine = phase.cos()[None, :, None, None, :].repeat(1, 1, 3, 1, 1).to(qkv.dtype)
        sine = phase.sin()[None, :, None, None, :].repeat(1, 1, 3, 1, 1).to(qkv.dtype)
        cosine[:, :, 2] = 1
        sine[:, :, 2] = 0
        left, right = qkv.chunk(2, dim=-1)
        return qkv * cosine + torch.cat((-right, left), dim=-1) * sine


class Block(nn.Module):
    def __init__(self, width, heads, condition, dropout):
        super().__init__()
        self.heads, self.dropout = heads, dropout
        self.norm1, self.norm2 = Norm(width), Norm(width)
        self.attn_qkv = nn.Linear(width, 3 * width, bias=False)
        self.attn_out = nn.Linear(width, width, bias=False)
        self.mlp = nn.Sequential(
            nn.Linear(width, 4 * width), nn.GELU(approximate="tanh"), nn.Linear(4 * width, width)
        )
        self.adaLN_modulation = nn.Linear(condition, 6 * width)
        nn.init.zeros_(self.adaLN_modulation.weight)
        nn.init.zeros_(self.adaLN_modulation.bias)

    def forward(self, values, condition, rotary):
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = self.adaLN_modulation(condition)[
            :, None
        ].chunk(6, -1)
        normalized = self.norm1(values) * (1 + scale_a) + shift_a
        qkv = self.attn_qkv(normalized).reshape(*values.shape[:2], 3, self.heads, -1)
        query, key, value = rotary(qkv).unbind(2)
        attended = (
            F.scaled_dot_product_attention(
                query.transpose(1, 2),
                key.transpose(1, 2),
                value.transpose(1, 2),
                dropout_p=0.0,
                is_causal=False,
            )
            .transpose(1, 2)
            .reshape_as(values)
        )
        values = values + gate_a * F.dropout(self.attn_out(attended), self.dropout, self.training)
        hidden = self.mlp(self.norm2(values) * (1 + scale_m) + shift_m)
        return values + gate_m * F.dropout(hidden, self.dropout, self.training)


class VocabularyEmbedding(nn.Module):
    def __init__(self, vocabulary, width):
        super().__init__()
        self.embedding = nn.Parameter(torch.empty(vocabulary, width))
        nn.init.kaiming_uniform_(self.embedding, a=math.sqrt(5))

    def forward(self, indices):
        return self.embedding[indices]


class Output(nn.Module):
    def __init__(self, width, vocabulary, condition):
        super().__init__()
        self.norm_final = Norm(width)
        self.linear = nn.Linear(width, vocabulary)
        self.adaLN_modulation = nn.Linear(condition, 2 * width)
        for layer in [self.linear, self.adaLN_modulation]:
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, values, condition):
        shift, scale = self.adaLN_modulation(condition)[:, None].chunk(2, -1)
        return self.linear(self.norm_final(values) * (1 + scale) + shift)


class ScoreTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        model = config["model"]
        if (
            config["graph"]["type"] != "absorb"
            or model["type"] != "ddit"
            or model["scale_by_sigma"] is not True
        ):
            raise ValueError("only native absorbing sigma-scaled DDiT is supported")
        width, heads = model["hidden_size"], model["n_heads"]
        if width < 1 or heads < 1 or width % heads or model["n_blocks"] < 1:
            raise ValueError("invalid diffusion transformer architecture")
        if not 0 <= model["dropout"] < 1 or config["tokens"] < 2:
            raise ValueError("invalid diffusion vocabulary/dropout")
        self.config = config
        self.context_length = model["length"]
        self.vocabulary = config["tokens"]
        self.vocab_embed = VocabularyEmbedding(self.vocabulary + 1, width)
        self.sigma_map = SigmaEmbedding(model["cond_dim"])
        self.rotary_emb = Rotary(width // heads)
        self.blocks = nn.ModuleList(
            Block(width, heads, model["cond_dim"], model["dropout"])
            for _ in range(model["n_blocks"])
        )
        self.output_layer = Output(width, self.vocabulary + 1, model["cond_dim"])

    def forward(self, indices, sigma):
        if (
            indices.dtype != torch.long
            or indices.ndim != 2
            or indices.shape[1] > self.context_length
        ):
            raise ValueError("invalid diffusion token input")
        if (
            sigma.shape != (indices.shape[0],)
            or not torch.isfinite(sigma).all()
            or (sigma <= 0).any()
        ):
            raise ValueError("invalid diffusion sigma input")
        precision = (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if indices.device.type == "cuda"
            else nullcontext()
        )
        with precision:
            values = self.vocab_embed(indices)
            condition = F.silu(self.sigma_map(sigma))
            for block in self.blocks:
                values = block(values, condition, self.rotary_emb)
            scores = self.output_layer(values, condition)
        scale = torch.where(sigma < 0.5, torch.expm1(sigma), sigma.exp() - 1).log().to(scores.dtype)
        scores = scores - scale[:, None, None] - math.log(self.vocabulary)
        return scores.scatter(-1, indices[..., None], torch.zeros_like(scores[..., :1]))


def load_initial_policy(root):
    """Verified local SEDD-medium initialization, BF16, one visible Slurm H100."""
    import json

    from .backend import require_h100_slurm
    from .provenance import digest, file_hash

    gpu = require_h100_slurm()
    lock_path = root / "configs/difftod-policy-model.lock.json"
    lock = json.loads(lock_path.read_text())
    if lock.get("schema_version") != 1 or lock["precision"] != "bfloat16":
        raise ValueError("unknown DiffTOD policy pin/precision")
    pin = lock["policy"]
    if (
        pin["model_id"] != "louaaron/sedd-medium"
        or pin["revision"] != "ce71a3c6178b50e899c8be1a1a4c13130308e54f"
    ):
        raise ValueError("native SEDD-medium initialization must remain fixed")
    snapshot = root / "models/difftod/sedd-medium" / pin["revision"]
    for name, expected in pin["files"].items():
        if file_hash(snapshot / name) != expected:
            raise ValueError("DiffTOD policy bootstrap checksum mismatch: " + name)
    config = json.loads((snapshot / "config.json").read_text())
    expected = {
        "hidden_size": 1024,
        "cond_dim": 128,
        "length": 1024,
        "n_blocks": 24,
        "n_heads": 16,
        "dropout": 0.1,
        "type": "ddit",
        "scale_by_sigma": True,
        "name": "medium",
    }
    if config["tokens"] != 50257 or config["model"] != expected:
        raise ValueError("native SEDD-medium architecture mismatch")
    weights = torch.load(snapshot / "pytorch_model.bin", map_location="cpu", weights_only=True)
    # Construct without allocating a duplicate 1.7GB random initialization.
    with torch.device("meta"):
        model = ScoreTransformer(config)
    model.load_state_dict(weights, strict=True, assign=True)
    model = model.to(device="cuda", dtype=torch.bfloat16)
    if any(p.dtype != torch.bfloat16 for p in model.parameters()):
        raise ValueError("diffusion policy must retain BF16 parameters")
    model.initial_optimizer_weights = {name: weights[name] for name, _ in model.named_parameters()}
    provenance = {
        "schema_version": 1,
        "model": pin,
        "model_lock_sha256": file_hash(lock_path),
        "architecture_signature": digest({n: list(v.shape) for n, v in model.state_dict().items()}),
        "precision": "bfloat16",
        "rotary_frequency_precision": "float32",
        "attention_execution": "pytorch_sdpa_v1",
        "gpu": gpu,
        "status": "initialization_loaded_not_fitted",
    }
    return model, provenance
