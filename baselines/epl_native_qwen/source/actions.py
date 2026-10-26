import json
from dataclasses import asdict, dataclass

from ...provenance import digest

TARGET_TOPIC = "$TARGET"
CONTEXT_TOPIC = "$CONTEXT"
NO_TOPIC = "$NONE"


@dataclass(frozen=True, order=True)
class NativeAction:
    goal: str
    topic_role: str

    def __post_init__(self):
        if not self.goal.strip() or self.topic_role not in {
            TARGET_TOPIC,
            CONTEXT_TOPIC,
            NO_TOPIC,
        }:
            raise ValueError("invalid EPL native goal-topic action")

    @property
    def key(self):
        if "@@" in self.goal:
            raise ValueError("EPL native goal contains reserved delimiter")
        return f"{self.goal}@@{self.topic_role}"

    @classmethod
    def from_key(cls, value):
        parts = value.split("@@")
        if len(parts) != 2:
            raise ValueError("invalid EPL native action key")
        return cls(*parts)

    def resolved_topic(self, scenario):
        if self.topic_role == TARGET_TOPIC:
            return str(scenario["target"][1])
        if self.topic_role == CONTEXT_TOPIC:
            return "the most relevant entity in the visible dialogue context"
        return "none"

    def to_dict(self):
        return asdict(self)


class NativeActionVocabulary:
    SCHEMA_VERSION = 1

    def __init__(self, dataset, goals, label_source):
        goals = tuple(sorted({str(goal).strip() for goal in goals if str(goal).strip()}))
        if dataset not in {"TopDial", "DuRecDial2.0"} or not goals:
            raise ValueError("invalid EPL native vocabulary source")
        self.dataset = dataset
        self.goals = goals
        self.label_source = label_source
        # The released EPL policy predicts a Cartesian goal-topic action. We retain
        # that structure, but use cross-dataset topic roles instead of lexical IDs.
        self.items = tuple(
            NativeAction(goal, role)
            for goal in goals
            for role in (TARGET_TOPIC, CONTEXT_TOPIC, NO_TOPIC)
        )
        self.actions = tuple(item.key for item in self.items)
        self._indices = {value: index for index, value in enumerate(self.actions)}

    def index(self, action):
        try:
            return self._indices[action.key]
        except KeyError as error:
            raise ValueError(f"EPL native action is outside train vocabulary: {action}") from error

    def payload(self):
        return {
            "schema_version": self.SCHEMA_VERSION,
            "dataset": self.dataset,
            "label_source": self.label_source,
            "topic_roles": [TARGET_TOPIC, CONTEXT_TOPIC, NO_TOPIC],
            "goals": list(self.goals),
            "actions": [item.to_dict() for item in self.items],
            "construction": "sorted_train_goal_cartesian_product_with_three_topic_roles",
            "official_test_opened": False,
        }

    @property
    def sha256(self):
        return digest(self.payload())

    def write(self, path):
        from ...provenance import atomic_json

        atomic_json(path, self.payload())

    @classmethod
    def read(cls, path):
        from pathlib import Path

        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        expected = {
            "schema_version",
            "dataset",
            "label_source",
            "topic_roles",
            "goals",
            "actions",
            "construction",
            "official_test_opened",
        }
        if set(raw) != expected or raw["schema_version"] != cls.SCHEMA_VERSION:
            raise ValueError("unknown EPL native action vocabulary")
        result = cls(raw["dataset"], raw["goals"], raw["label_source"])
        if raw != result.payload():
            raise ValueError("EPL native action vocabulary is not canonical")
        return result
