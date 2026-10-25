from dataclasses import dataclass, field
from typing import Literal, Protocol

import torch


@dataclass(frozen=True)
class Message:
    role: Literal["system", "user", "assistant"]
    content: str


@dataclass(frozen=True)
class Decoding:
    temperature: float = 0.7
    top_p: float = 0.8
    top_k: int = 20
    max_tokens: int = 256

    def __post_init__(self):
        if not 0 <= self.temperature <= 2 or not 0 < self.top_p <= 1:
            raise ValueError("invalid decoding")
        if (self.top_k != -1 and self.top_k < 1) or self.max_tokens < 1:
            raise ValueError("invalid decoding budget")


@dataclass(frozen=True)
class ModelProvenance:
    model_id: str
    revision: str
    weight_sha256: dict[str, str]
    template_sha256: str


@dataclass(frozen=True)
class Generation:
    text: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    seed: int
    decoding: Decoding
    latency_s: float
    retries: int
    model: ModelProvenance
    prompt_sha256: str
    request_id: str
    role: str
    finish_reason: str = "unknown"


@dataclass(frozen=True)
class Embeddings:
    vectors: torch.Tensor
    model: ModelProvenance
    instruction_sha256: str
    preprocessing_sha256: str
    batch_size: int
    latency_s: float
    input_tokens: int | None
    peak_memory_bytes: int | None
    batch_id: str
    request_id: str


class GenerationBackend(Protocol):
    def generate(
        self, messages: list[Message], role: str, decoding: Decoding, seed: int, request_id: str
    ) -> Generation: ...


class EmbeddingBackend(Protocol):
    def embed(
        self, texts: list[str], instruction: str, batch_id: str, request_id: str
    ) -> Embeddings: ...


class DirectedReachability(Protocol):
    head_type: str
    checkpoint_hash: str
    encoder_provenance: dict

    def distance(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor: ...


@dataclass
class Plan:
    response: str
    covered: bool
    reference_id: str | None
    directed_distances: dict[str, float]
    path: list[str]
    selected_candidate: int | None
    rollout_scores: list[float | None]
    compute: dict
    replay_ids: list[str] = field(default_factory=list)


class Planner(Protocol):
    def act(self, state, target_nodes: list[str], seed: int, request_id: str) -> Plan: ...
