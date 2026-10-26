import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

from ...provenance import digest


@dataclass(frozen=True)
class EPLConfig:
    schema_version: int
    dataset: str
    actions: tuple[str, ...]
    encoder_model: str
    retrieval_model: str
    retrieval_dimension: int
    max_sequence_length: int
    header_tokens: int
    hidden_size: int
    dropout: float
    training_epochs: int
    training_batch_size: int
    evaluation_batch_size: int
    training_learning_rate: float
    training_weight_decay: float
    training_warmup_steps: int
    gradient_clip: float | None
    memory_cases: int
    memory_embedding_batch_size: int
    retrieval_k: int
    retrieval_temperature: float
    retrieval_success_threshold: float
    reward_alpha: float
    reward_length_weight: float
    reward_length_temperature: float
    puct_simulations: int
    puct_cpuct: float
    puct_q0: float
    puct_max_depth: int
    puct_max_realizations: int
    policy_top_k: int
    gamma: float
    training_seed: int
    self_simulation_seed: int

    def __post_init__(self):
        if self.schema_version != 1:
            raise ValueError("unknown EPL configuration schema")
        if self.dataset not in {"TopDial", "DuRecDial2.0"}:
            raise ValueError("unknown EPL dataset")
        if len(self.actions) < 2 or len(set(self.actions)) != len(self.actions):
            raise ValueError("invalid EPL action taxonomy")
        if self.encoder_model != "bert-base-cased":
            raise ValueError("EPL requires its released bert-base-cased policy encoder")
        if self.retrieval_model != "sentence-transformers/all-MiniLM-L6-v2":
            raise ValueError("EPL requires its released all-MiniLM-L6-v2 retriever")
        integers = (
            self.max_sequence_length,
            self.header_tokens,
            self.hidden_size,
            self.retrieval_dimension,
            self.training_epochs,
            self.training_batch_size,
            self.evaluation_batch_size,
            self.training_warmup_steps,
            self.memory_cases,
            self.memory_embedding_batch_size,
            self.retrieval_k,
            self.puct_simulations,
            self.puct_max_depth,
            self.puct_max_realizations,
            self.policy_top_k,
        )
        if any(value < 1 for value in integers):
            raise ValueError("EPL count parameters must be positive")
        if self.header_tokens >= self.max_sequence_length - 2:
            raise ValueError("EPL header exceeds the policy context")
        if self.policy_top_k > len(self.actions):
            raise ValueError("EPL policy top-k exceeds the action taxonomy")
        finite = (
            self.dropout,
            self.training_learning_rate,
            self.training_weight_decay,
            self.retrieval_temperature,
            self.retrieval_success_threshold,
            self.reward_alpha,
            self.reward_length_weight,
            self.reward_length_temperature,
            self.puct_cpuct,
            self.puct_q0,
            self.gamma,
        )
        if any(not math.isfinite(value) for value in finite):
            raise ValueError("non-finite EPL configuration")
        if self.gradient_clip is not None:
            if not math.isfinite(self.gradient_clip) or self.gradient_clip <= 0:
                raise ValueError("invalid EPL gradient clipping")
        if not 0 <= self.dropout < 1:
            raise ValueError("invalid EPL dropout")
        if self.training_learning_rate <= 0:
            raise ValueError("invalid EPL optimizer configuration")
        if self.retrieval_temperature <= 0 or self.reward_length_temperature <= 0:
            raise ValueError("invalid EPL retrieval/reward temperature")
        if self.puct_simulations < 2 or self.puct_cpuct < 0 or self.gamma < 0:
            raise ValueError("invalid EPL P-UCT budget")

    @classmethod
    def from_path(cls, path):
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        expected = set(cls.__dataclass_fields__)
        if set(raw) != expected:
            raise ValueError("unknown/incomplete EPL configuration")
        raw["actions"] = tuple(raw["actions"])
        return cls(**raw)

    @property
    def sha256(self):
        return digest(asdict(self))

    @property
    def training_sha256(self):
        payload = asdict(self)
        for key in tuple(payload):
            if key.startswith("memory_") or key.startswith("retrieval_") or key.startswith("puct_"):
                payload.pop(key)
        for key in (
            "reward_alpha",
            "reward_length_weight",
            "reward_length_temperature",
            "policy_top_k",
            "gamma",
        ):
            payload.pop(key)
        return digest(payload)
