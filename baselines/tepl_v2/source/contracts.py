from dataclasses import dataclass


@dataclass(frozen=True)
class EPLCheckpointMetadata:
    schema_version: int
    dataset: str
    config_sha256: str
    training_sha256: str
    train_sha256: str
    dev_sha256: str
    action_taxonomy_sha256: str
    model_lock_sha256: str
    source_revision: str


@dataclass(frozen=True)
class Experience:
    experience_id: str
    scenario_id: str
    state_text: str
    continuation: tuple[dict, ...]
    score: float
    target: tuple[str, str]

    def to_dict(self):
        return {
            "experience_id": self.experience_id,
            "scenario_id": self.scenario_id,
            "state_text": self.state_text,
            "continuation": list(self.continuation),
            "score": self.score,
            "target": list(self.target),
        }
