import json
from dataclasses import asdict, dataclass

from .contracts import Message
from .provenance import digest

SERIALIZATION = {"version": 1, "fields": ["history"], "normalization": "none"}
SERIALIZATION_HASH = digest(SERIALIZATION)


@dataclass(frozen=True)
class DialogueState:
    history: tuple[Message, ...]

    def __post_init__(self):
        for m in self.history:
            if m.role not in {"user", "assistant"} or not isinstance(m.content, str):
                raise ValueError("state contains only observed dialogue turns")

    def serialize(self) -> str:
        # Deliberately no target, profile, future turns, labels, or split marker in embeddings.
        return json.dumps(
            {"version": 1, "history": [asdict(m) for m in self.history]},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    def append(self, message: Message):
        return DialogueState(self.history + (message,))


def observed_prefix(messages: list[Message], visible_turns: int) -> DialogueState:
    if not 0 <= visible_turns <= len(messages):
        raise ValueError("invalid visibility boundary")
    return DialogueState(tuple(messages[:visible_turns]))
