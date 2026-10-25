"""Stable data contracts for released ConvPlan candidates and the selected bank."""

from dataclasses import asdict, dataclass

ALLOWED_RETRIEVAL_DOMAINS = {
    "movie",
    "music",
    "poi",
    "food",
    "celebrity",
    "news",
    "weather",
    "hobby",
    "general",
}


@dataclass(frozen=True)
class RawRecord:
    conver_id: str
    plans: tuple[str, ...]
    user_profile: dict
    conversation: tuple[str, ...]
    goal: str

    @classmethod
    def parse(cls, value):
        if not isinstance(value, dict) or set(value) != {
            "conver_id",
            "plans",
            "user_profile",
            "conversation",
            "goal",
        }:
            raise ValueError("unknown ConvPlan record schema")
        identifier = value["conver_id"]
        if not isinstance(identifier, (int, str)) or not str(identifier).strip():
            raise ValueError("invalid ConvPlan conversation ID")
        plans, conversation = value["plans"], value["conversation"]
        if (
            not isinstance(plans, list)
            or not all(isinstance(item, str) for item in plans)
            or not isinstance(conversation, list)
            or not all(isinstance(item, str) for item in conversation)
            or not isinstance(value["user_profile"], dict)
            or not isinstance(value["goal"], str)
        ):
            raise ValueError("invalid ConvPlan record types")
        return cls(
            str(identifier),
            tuple(plans),
            value["user_profile"],
            tuple(conversation),
            value["goal"],
        )


@dataclass(frozen=True)
class Keyphrase:
    canonical_text: str
    surface_text: str
    score: float
    first_start: int
    last_end: int


@dataclass(frozen=True)
class PlanBankEntry:
    entry_id: str
    source_dataset: str
    source_split: str
    source_conversation_id: str
    duplicate_group_id: str
    candidate_index: int
    plan_text: str
    user_profile: dict
    goal_raw: str
    target_items: tuple[str, ...]
    demonstration_target: str
    target_resolution: str
    retrieval_domain: str
    retrieval_entities: tuple[str, ...]
    conversation_hash: str
    entity_sequence: tuple[str, ...]
    consistency_score: float
    extraction_config_hash: str
    bank_build_config_hash: str

    def serialize(self):
        return asdict(self)

    @classmethod
    def parse(cls, value):
        required = set(cls.__dataclass_fields__)
        if not isinstance(value, dict) or set(value) != required:
            raise ValueError("unknown EnPL plan-bank schema")
        converted = dict(value)
        for name in ("target_items", "retrieval_entities", "entity_sequence"):
            if not isinstance(converted[name], (list, tuple)):
                raise ValueError(f"invalid EnPL bank field: {name}")
            converted[name] = tuple(converted[name])
        entry = cls(**converted)
        if (
            entry.source_split not in {"train", "dev", "test", "unknown"}
            or entry.retrieval_domain not in ALLOWED_RETRIEVAL_DOMAINS
            or entry.candidate_index < 0
            or not 0.0 <= entry.consistency_score <= 1.0
            or not entry.plan_text.strip()
        ):
            raise ValueError("invalid EnPL plan-bank entry")
        return entry


@dataclass(frozen=True)
class RetrievedPlan:
    entry: PlanBankEntry
    overlap_score: int
    zero_overlap_fallback: bool
