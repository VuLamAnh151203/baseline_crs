import json
import math
from dataclasses import dataclass
from pathlib import Path

from ...provenance import digest
from .actions import NativeActionVocabulary


@dataclass(frozen=True)
class EPLNativeConfig:
    schema_version: int
    method: str
    dataset: str
    label_adapter: str
    encoder_model: str
    retrieval_model: str
    retrieval_dimension: int
    max_sequence_length: int
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
    actions: tuple[str, ...]
    vocabulary_sha256: str

    METHOD = "EPL-native-core-Qwen-fixed400"

    def __post_init__(self):
        if self.schema_version != 1 or self.method != self.METHOD:
            raise ValueError("unknown EPL native configuration")
        if self.dataset not in {"TopDial", "DuRecDial2.0"}:
            raise ValueError("unknown EPL native dataset")
        if self.encoder_model != "bert-base-cased":
            raise ValueError("EPL native core requires bert-base-cased")
        if self.retrieval_model != "sentence-transformers/all-MiniLM-L6-v2":
            raise ValueError("EPL native core requires the released MiniLM retriever")
        if len(self.actions) < 2 or len(set(self.actions)) != len(self.actions):
            raise ValueError("invalid EPL native action vocabulary")
        counts = (
            self.retrieval_dimension,
            self.max_sequence_length,
            self.hidden_size,
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
        if any(value < 1 for value in counts) or self.policy_top_k > len(self.actions):
            raise ValueError("invalid EPL native count parameter")
        values = (
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
        if any(not math.isfinite(value) for value in values):
            raise ValueError("non-finite EPL native configuration")
        if self.training_learning_rate <= 0 or self.puct_simulations < 2:
            raise ValueError("invalid EPL native optimizer/search configuration")

    @classmethod
    def from_paths(cls, config_path, vocabulary_path):
        raw = json.loads(Path(config_path).read_text(encoding="utf-8"))
        vocabulary = NativeActionVocabulary.read(vocabulary_path)
        if raw.get("dataset") != vocabulary.dataset:
            raise ValueError("EPL native config/vocabulary dataset mismatch")
        expected = set(cls.__dataclass_fields__) - {"actions", "vocabulary_sha256"}
        if set(raw) != expected:
            raise ValueError("unknown/incomplete EPL native config fields")
        return cls(
            **raw,
            actions=vocabulary.actions,
            vocabulary_sha256=vocabulary.sha256,
        )

    def _payload(self, training=False):
        payload = {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if name != "actions"
        }
        if not training:
            payload["actions"] = list(self.actions)
        else:
            for key in tuple(payload):
                if key.startswith("memory_") or key.startswith("retrieval_") or key.startswith("puct_"):
                    payload.pop(key)
            for key in (
                "reward_alpha",
                "reward_length_weight",
                "reward_length_temperature",
                "policy_top_k",
                "gamma",
                "self_simulation_seed",
            ):
                payload.pop(key)
        return payload

    @property
    def sha256(self):
        return digest(self._payload())

    @property
    def training_sha256(self):
        return digest(self._payload(training=True))

    @property
    def header_tokens(self):
        return 96
