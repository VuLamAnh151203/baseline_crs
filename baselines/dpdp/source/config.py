import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from ...official_data import ACTION_TAXONOMY
from ...provenance import digest


@dataclass(frozen=True)
class DPDPConfig:
    schema_version: int
    dataset: str
    actions: tuple[str, ...]
    encoder_model: str
    max_sequence_length: int
    header_tokens: int
    gamma: float
    offline_dropout: float
    selfplay_dropout: float
    offline_epochs: int
    offline_batch_size: int
    offline_lr: float
    offline_weight_decay: float
    offline_warmup_steps: int
    offline_critic_weight: float
    offline_checkpoint_every: int
    selfplay_epochs: int
    selfplay_cases_per_epoch: int
    selfplay_lr: float
    selfplay_critic_weight: float
    mcts_simulations: int
    mcts_cpuct: float
    mcts_q0: float
    mcts_max_depth: int
    mcts_max_realizations: int
    critic_samples: int
    gradient_clip: float
    offline_seed: int
    selfplay_seed: int
    gate_threshold: float

    def __post_init__(self):
        if self.schema_version != 2:
            raise ValueError("unknown DPDP configuration schema")
        if self.dataset not in {"TopDial", "DuRecDial2.0"}:
            raise ValueError("unsupported DPDP dataset")
        if tuple(self.actions) != tuple(ACTION_TAXONOMY) or len(set(self.actions)) != len(
            self.actions
        ):
            raise ValueError("DPDP action taxonomy/order mismatch")
        if self.encoder_model != "FacebookAI/roberta-large":
            raise ValueError("DPDP requires the frozen RoBERTa-large architecture")
        counts = (
            self.max_sequence_length,
            self.header_tokens,
            self.offline_epochs,
            self.offline_batch_size,
            self.offline_checkpoint_every,
            self.selfplay_epochs,
            self.selfplay_cases_per_epoch,
            self.mcts_max_depth,
            self.mcts_max_realizations,
            self.critic_samples,
        )
        if any(not isinstance(value, int) or value < 1 for value in counts):
            raise ValueError("DPDP count/budget fields must be positive integers")
        if self.mcts_simulations < 2:
            raise ValueError("DPDP MCTS requires at least two simulations")
        if not 32 <= self.max_sequence_length <= 512 or not 1 <= self.header_tokens < (
            self.max_sequence_length - 2
        ):
            raise ValueError("invalid DPDP policy token budget")
        if not 0 < self.gamma <= 1 or any(
            not 0 <= value < 1
            for value in (self.offline_dropout, self.selfplay_dropout)
        ):
            raise ValueError("invalid DPDP discount/dropout")
        if any(
            not math.isfinite(value) or value <= 0
            for value in (
                self.offline_lr,
                self.selfplay_lr,
                self.offline_critic_weight,
                self.selfplay_critic_weight,
                self.gradient_clip,
                self.mcts_cpuct,
            )
        ):
            raise ValueError("invalid DPDP optimization/search scalar")
        if (
            self.offline_weight_decay < 0
            or not isinstance(self.offline_warmup_steps, int)
            or self.offline_warmup_steps < 0
        ):
            raise ValueError("invalid DPDP optimizer schedule")
        if any(
            not isinstance(value, int) or value < 0
            for value in (self.offline_seed, self.selfplay_seed)
        ):
            raise ValueError("invalid DPDP phase seed")
        if not math.isfinite(self.mcts_q0):
            raise ValueError("invalid DPDP initial Q value")
        if not 0 <= self.gate_threshold <= 1:
            raise ValueError("invalid DPDP confidence-gap threshold")

    @classmethod
    def from_path(cls, path: Path):
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        expected = set(cls.__dataclass_fields__)
        if set(raw) != expected:
            missing, unknown = sorted(expected - set(raw)), sorted(set(raw) - expected)
            raise ValueError(
                "unknown/incomplete DPDP configuration: "
                f"missing={missing}, unknown={unknown}"
            )
        raw["actions"] = tuple(raw["actions"])
        return cls(**raw)

    @property
    def sha256(self):
        return digest(asdict(self))

    @property
    def training_sha256(self):
        values = asdict(self)
        for key in (
            "mcts_simulations",
            "mcts_cpuct",
            "mcts_q0",
            "mcts_max_depth",
            "mcts_max_realizations",
            "gate_threshold",
        ):
            values.pop(key)
        return digest(values)

    @property
    def fitting_sha256(self):
        """Bind offline/self-play fitting, while leaving the dev-selected gate mutable."""
        values = asdict(self)
        values.pop("gate_threshold")
        return digest(values)

    def with_budget(self, simulations: int, depth: int):
        return replace(self, mcts_simulations=simulations, mcts_max_depth=depth)

    @classmethod
    def smoke(cls):
        return cls(
            schema_version=2,
            dataset="TopDial",
            actions=tuple(ACTION_TAXONOMY),
            encoder_model="FacebookAI/roberta-large",
            max_sequence_length=64,
            header_tokens=16,
            gamma=0.999,
            offline_dropout=0.0,
            selfplay_dropout=0.0,
            offline_epochs=1,
            offline_batch_size=2,
            offline_lr=1e-3,
            offline_weight_decay=0.0,
            offline_warmup_steps=0,
            offline_critic_weight=1.0,
            offline_checkpoint_every=1,
            selfplay_epochs=1,
            selfplay_cases_per_epoch=1,
            selfplay_lr=1e-3,
            selfplay_critic_weight=1.0,
            mcts_simulations=3,
            mcts_cpuct=1.0,
            mcts_q0=0.0,
            mcts_max_depth=2,
            mcts_max_realizations=1,
            critic_samples=2,
            gradient_clip=1.0,
            offline_seed=17,
            selfplay_seed=17,
            gate_threshold=0.2,
        )
