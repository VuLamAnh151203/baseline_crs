import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig


class LayerNorm(nn.Module):
    """Bias-free LayerNorm matching the released SEDD parameterization."""

    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.dim = dim

    def forward(self, x):
        normalized = F.layer_norm(x.float(), [self.dim])
        return normalized.to(x.dtype) * self.weight.to(x.dtype)


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10_000):
        half = dim // 2
        frequencies = torch.exp(
            -math.log(max_period)
            * torch.arange(half, dtype=torch.float32, device=t.device)
            / half
        )
        arguments = t[:, None].float() * frequencies[None]
        embedding = torch.cat([torch.cos(arguments), torch.sin(arguments)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        return self.mlp(self.timestep_embedding(t, self.frequency_embedding_size))


class Rotary(nn.Module):
    def __init__(self, dim, base=10_000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    @staticmethod
    def _rotate_half(x):
        left, right = x.chunk(2, dim=-1)
        return torch.cat((-right, left), dim=-1)

    def forward(self, qkv):
        length = qkv.shape[1]
        positions = torch.arange(length, device=qkv.device, dtype=self.inv_freq.dtype)
        frequencies = torch.outer(positions, self.inv_freq)
        embedding = torch.cat((frequencies, frequencies), dim=-1).to(qkv.dtype)
        cos = embedding.cos()[None, :, None, :]
        sin = embedding.sin()[None, :, None, :]
        query, key, value = qkv.unbind(dim=2)
        query = query * cos + self._rotate_half(query) * sin
        key = key * cos + self._rotate_half(key) * sin
        return torch.stack((query, key, value), dim=2)


class DDiTBlock(nn.Module):
    def __init__(self, dim, n_heads, cond_dim, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.norm1 = LayerNorm(dim)
        self.attn_qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.attn_out = nn.Linear(dim, dim, bias=False)
        self.dropout1 = nn.Dropout(dropout)
        self.norm2 = LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_ratio * dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_ratio * dim, dim),
        )
        self.dropout2 = nn.Dropout(dropout)
        self.adaLN_modulation = nn.Linear(cond_dim, 6 * dim)
        nn.init.zeros_(self.adaLN_modulation.weight)
        nn.init.zeros_(self.adaLN_modulation.bias)

    @staticmethod
    def _modulate(x, shift, scale):
        return x * (1 + scale[:, None]) + shift[:, None]

    def forward(self, x, rotary, condition):
        shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(condition).chunk(6, dim=-1)
        )
        batch, length, width = x.shape
        qkv = self.attn_qkv(self._modulate(self.norm1(x), shift_attn, scale_attn))
        qkv = qkv.view(batch, length, 3, self.n_heads, width // self.n_heads)
        query, key, value = rotary(qkv).unbind(dim=2)
        query, key, value = (item.transpose(1, 2) for item in (query, key, value))
        attention = F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=0.0,
            is_causal=False,
        )
        attention = attention.transpose(1, 2).contiguous().view(batch, length, width)
        x = x + gate_attn[:, None] * self.dropout1(self.attn_out(attention))
        branch = self.mlp(self._modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x + gate_mlp[:, None] * self.dropout2(branch)


class EmbeddingLayer(nn.Module):
    def __init__(self, dim, vocab_dim):
        super().__init__()
        self.embedding = nn.Parameter(torch.empty(vocab_dim, dim))
        nn.init.kaiming_uniform_(self.embedding, a=math.sqrt(5))

    def forward(self, indices):
        return self.embedding[indices]


class DDitFinalLayer(nn.Module):
    def __init__(self, hidden_size, out_channels, cond_dim):
        super().__init__()
        self.norm_final = LayerNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, out_channels)
        self.adaLN_modulation = nn.Linear(cond_dim, 2 * hidden_size)
        for layer in (self.linear, self.adaLN_modulation):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x, condition):
        shift, scale = self.adaLN_modulation(condition).chunk(2, dim=-1)
        x = self.norm_final(x)
        x = x * (1 + scale[:, None]) + shift[:, None]
        return self.linear(x)


class SEDD(nn.Module):
    """SEDD-medium-compatible absorbing discrete diffusion transformer."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        size = config.model_vocab_size
        self.vocab_embed = EmbeddingLayer(config.hidden_size, size)
        self.sigma_map = TimestepEmbedder(config.cond_dim)
        self.rotary_emb = Rotary(config.hidden_size // config.n_heads)
        self.blocks = nn.ModuleList(
            DDiTBlock(
                config.hidden_size,
                config.n_heads,
                config.cond_dim,
                dropout=config.dropout,
            )
            for _ in range(config.n_blocks)
        )
        self.output_layer = DDitFinalLayer(config.hidden_size, size, config.cond_dim)
        self.scale_by_sigma = config.scale_by_sigma

    def forward(self, indices, sigma):
        if indices.ndim != 2 or indices.shape[1] != self.config.length:
            raise ValueError("SEDD expects [batch, configured_length] token IDs")
        if indices.dtype != torch.long:
            raise TypeError("SEDD token IDs must be torch.long")
        if indices.numel() and (indices.min() < 0 or indices.max() >= self.config.model_vocab_size):
            raise ValueError("token ID outside the absorbing vocabulary")
        sigma = sigma.reshape(-1)
        if len(sigma) != len(indices) or not torch.isfinite(sigma).all() or (sigma <= 0).any():
            raise ValueError("each SEDD sample requires a finite positive sigma")
        x = self.vocab_embed(indices)
        condition = F.silu(self.sigma_map(sigma))
        enabled = x.is_cuda
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=enabled):
            for block in self.blocks:
                x = block(x, self.rotary_emb, condition)
            scores = self.output_layer(x, condition)
        if self.scale_by_sigma:
            expm1_log = torch.where(
                sigma < 0.5,
                torch.expm1(sigma),
                sigma.exp() - 1,
            ).log()
            scores = scores - expm1_log[:, None, None].to(scores.dtype)
            scores = scores - math.log(scores.shape[-1] - 1)
        return torch.scatter(
            scores,
            -1,
            indices[..., None],
            torch.zeros_like(scores[..., :1]),
        )

    @property
    def parameter_count(self):
        return sum(parameter.numel() for parameter in self.parameters())


def normalize_state_dict(value):
    if not isinstance(value, dict):
        raise ValueError("checkpoint must contain a state dictionary")
    if "model" in value and isinstance(value["model"], dict):
        value = value["model"]
    return {key.removeprefix("module."): tensor for key, tensor in value.items()}


def load_weights(model: SEDD, checkpoint, strict=True):
    state = normalize_state_dict(checkpoint)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if strict and (missing or unexpected):
        raise ValueError(
            f"SEDD checkpoint architecture mismatch; missing={missing}, unexpected={unexpected}"
        )
    return {"missing": list(missing), "unexpected": list(unexpected)}
