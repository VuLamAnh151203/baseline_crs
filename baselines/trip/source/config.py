import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

from ...provenance import digest


@dataclass(frozen=True)
class ModelConfig:
    hidden_size: int = 768
    decoder_layers: int = 6
    attention_heads: int = 8
    feedforward_size: int = 3072
    max_positions: int = 512
    dropout: float = 0.1
    init_std: float = 0.02
    beta: float = 0.1
    gamma: float = 1.0
    tau: float = 0.1

    def __post_init__(self):
        if min(
            self.hidden_size,
            self.decoder_layers,
            self.attention_heads,
            self.feedforward_size,
            self.max_positions,
        ) < 1:
            raise ValueError("TRIP model dimensions must be positive")
        if self.hidden_size % self.attention_heads:
            raise ValueError("TRIP hidden size must be divisible by attention heads")
        if not 0 <= self.dropout < 1:
            raise ValueError("invalid TRIP dropout")
        if not math.isfinite(self.init_std) or self.init_std <= 0:
            raise ValueError("invalid TRIP initialization scale")
        if any(not math.isfinite(x) or x < 0 for x in (self.beta, self.gamma)):
            raise ValueError("invalid TRIP loss weight")
        if not math.isfinite(self.tau) or self.tau <= 0:
            raise ValueError("invalid TRIP contrastive temperature")


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int = 8
    gradient_accumulation: int = 1
    epochs: int = 10
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    gradient_clip: float = 0.5
    evaluate_every: int = 1000
    checkpoint_every: int = 1000
    seed: int = 17
    num_negatives: int = 3

    def __post_init__(self):
        integers = (
            self.batch_size,
            self.gradient_accumulation,
            self.epochs,
            self.evaluate_every,
            self.checkpoint_every,
            self.num_negatives,
        )
        if any(value < 1 for value in integers):
            raise ValueError("TRIP training counts must be positive")
        if not 0 < self.learning_rate < 1 or not 0 <= self.warmup_ratio < 1:
            raise ValueError("invalid TRIP learning-rate schedule")
        if self.weight_decay < 0 or self.gradient_clip <= 0:
            raise ValueError("invalid TRIP optimizer configuration")


@dataclass(frozen=True)
class DecodingConfig:
    beam_size: int = 3
    max_plan_tokens: int = 80
    min_plan_tokens: int = 1
    lambda_scale: float = 3.0
    repetition_penalty: float = 1.0

    def __post_init__(self):
        if self.beam_size < 1 or self.max_plan_tokens < 4:
            raise ValueError("invalid TRIP beam budget")
        if not 0 <= self.min_plan_tokens < self.max_plan_tokens:
            raise ValueError("invalid TRIP minimum plan length")
        if self.lambda_scale < 0 or self.repetition_penalty < 1:
            raise ValueError("invalid TRIP decoding penalty")


@dataclass(frozen=True)
class TripConfig:
    schema_version: int
    model: ModelConfig
    training: TrainingConfig
    decoding: DecodingConfig
    max_context_tokens: int = 512
    max_plan_tokens: int = 128

    def __post_init__(self):
        if self.schema_version != 1:
            raise ValueError("unknown TRIP configuration schema")
        if not 32 <= self.max_context_tokens <= self.model.max_positions:
            raise ValueError("TRIP context budget exceeds model positions")
        if not 8 <= self.max_plan_tokens <= self.model.max_positions:
            raise ValueError("invalid TRIP training plan budget")

    @classmethod
    def from_path(cls, path: Path):
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if set(raw) != {
            "schema_version",
            "model",
            "training",
            "decoding",
            "max_context_tokens",
            "max_plan_tokens",
        }:
            raise ValueError("unknown/incomplete TRIP configuration")
        return cls(
            schema_version=raw["schema_version"],
            model=ModelConfig(**raw["model"]),
            training=TrainingConfig(**raw["training"]),
            decoding=DecodingConfig(**raw["decoding"]),
            max_context_tokens=raw["max_context_tokens"],
            max_plan_tokens=raw["max_plan_tokens"],
        )

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
                "max_context_tokens": self.max_context_tokens,
                "max_plan_tokens": self.max_plan_tokens,
            }
        )

    @classmethod
    def smoke(cls):
        return cls(
            schema_version=1,
            model=ModelConfig(
                hidden_size=32,
                decoder_layers=2,
                attention_heads=4,
                feedforward_size=64,
                max_positions=64,
                dropout=0.0,
            ),
            training=TrainingConfig(
                batch_size=2,
                epochs=1,
                evaluate_every=1,
                checkpoint_every=1,
                num_negatives=2,
            ),
            decoding=DecodingConfig(beam_size=2, max_plan_tokens=24),
            max_context_tokens=64,
            max_plan_tokens=32,
        )
