import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ...provenance import digest


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 50_257
    length: int = 1_024
    hidden_size: int = 1_024
    cond_dim: int = 128
    n_blocks: int = 24
    n_heads: int = 16
    dropout: float = 0.1
    scale_by_sigma: bool = True

    def __post_init__(self):
        values = (
            self.vocab_size,
            self.length,
            self.hidden_size,
            self.cond_dim,
            self.n_blocks,
            self.n_heads,
        )
        if any(not isinstance(v, int) or v < 1 for v in values):
            raise ValueError("DiffTOD model dimensions must be positive integers")
        if self.hidden_size % self.n_heads or (self.hidden_size // self.n_heads) % 2:
            raise ValueError("attention head dimension must be positive and even")
        if not math.isfinite(self.dropout) or not 0 <= self.dropout < 1:
            raise ValueError("invalid dropout")
        if not self.scale_by_sigma:
            raise ValueError("the faithful absorbing DiffTOD adaptation requires sigma scaling")

    @property
    def model_vocab_size(self):
        """GPT-2 vocabulary plus the absorbing mask state."""
        return self.vocab_size + 1

    @classmethod
    def smoke(cls):
        return cls(
            vocab_size=31,
            length=16,
            hidden_size=32,
            cond_dim=16,
            n_blocks=2,
            n_heads=4,
            dropout=0.0,
        )


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int = 32
    gradient_accumulation: int = 1
    steps: int = 100_000
    learning_rate: float = 1e-4
    warmup_steps: int = 2_500
    weight_decay: float = 0.0
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
    gradient_clip: float = 1.0
    ema_decay: float = 0.9999
    checkpoint_every: int = 10_000
    evaluate_every: int = 100
    seed: int = 17

    def __post_init__(self):
        integers = (
            self.batch_size,
            self.gradient_accumulation,
            self.steps,
            self.warmup_steps,
            self.checkpoint_every,
            self.evaluate_every,
        )
        if any(not isinstance(v, int) or v < 1 for v in integers):
            raise ValueError("invalid DiffTOD training step/batch configuration")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning rate must be finite and positive")
        if not 0 <= self.weight_decay or not 0 <= self.gradient_clip:
            raise ValueError("weight decay and gradient clip must be nonnegative")
        if not 0 < self.ema_decay < 1:
            raise ValueError("EMA decay must be in (0, 1)")
        if not 0 <= self.beta1 < 1 or not 0 <= self.beta2 < 1 or self.epsilon <= 0:
            raise ValueError("invalid AdamW configuration")


@dataclass(frozen=True)
class SamplingConfig:
    steps: int = 1_024
    candidates: int = 5
    semantic_offset: int = 192
    epsilon: float = 1e-5
    denoise: bool = True

    def __post_init__(self):
        if any(
            not isinstance(v, int) or v < 1
            for v in (self.steps, self.candidates, self.semantic_offset)
        ):
            raise ValueError("invalid DiffTOD sampling configuration")
        if self.candidates > 5:
            raise ValueError("DiffTOD supports at most five semantic-guidance candidates")
        if not math.isfinite(self.epsilon) or not 0 < self.epsilon < 1:
            raise ValueError("sampling epsilon must be in (0, 1)")
        if self.denoise is not True:
            raise ValueError("faithful DiffTOD sampling requires final absorbing-state denoising")


@dataclass(frozen=True)
class DiffTODConfig:
    schema_version: int = 1
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)

    def __post_init__(self):
        if self.schema_version != 1:
            raise ValueError("unknown DiffTOD configuration schema")
        if self.sampling.semantic_offset >= self.model.length:
            raise ValueError("semantic guidance offset is outside the model sequence")

    @property
    def sha256(self):
        return digest(asdict(self))

    @property
    def training_sha256(self):
        return digest(
            {
                "schema_version": self.schema_version,
                "model": asdict(self.model),
                "training": asdict(self.training),
            }
        )

    @classmethod
    def from_path(cls, path: Path):
        raw = json.loads(path.read_text(encoding="utf-8"))
        if set(raw) != {"schema_version", "model", "training", "sampling"}:
            raise ValueError("unknown/incomplete DiffTOD configuration")
        return cls(
            schema_version=raw["schema_version"],
            model=ModelConfig(**raw["model"]),
            training=TrainingConfig(**raw["training"]),
            sampling=SamplingConfig(**raw["sampling"]),
        )
