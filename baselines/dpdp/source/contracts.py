from dataclasses import asdict, dataclass, field
from typing import Literal

from ...contracts import Message

RewardLabel = Literal["rejected", "no_progress", "interested", "accepted"]


@dataclass(frozen=True)
class DPDPTransition:
    transition_id: str
    scenario_id: str
    dataset: str
    split: str
    turn_index: int
    state: tuple[Message, ...]
    next_state: tuple[Message, ...]
    action_index: int
    action_name: str
    reward: float | None
    discounted_return: float | None
    done: bool
    profile: object
    target: tuple[str, str]
    provenance: dict[str, str]

    def to_dict(self):
        row = asdict(self)
        row["target"] = list(self.target)
        return row


@dataclass(frozen=True)
class DPDPEpisode:
    scenario_id: str
    transitions: tuple[DPDPTransition, ...]


@dataclass(frozen=True)
class PolicyOutput:
    probabilities: tuple[float, ...]
    q_values: tuple[float, ...]
    confidence_gap: float


@dataclass
class TreeNodeStats:
    path: tuple[str, ...]
    priors: list[float]
    visits: list[int]
    q_values: list[float]
    realizations: dict[int, list["RewardSample"]] = field(default_factory=dict)


@dataclass(frozen=True)
class RewardSample:
    assistant_response: str
    user_response: str
    reward: float
    label: RewardLabel
    critic_labels: tuple[RewardLabel, ...]
    deterministic_accepted: bool
    acceptance_disagreement: bool
    terminal: bool
    terminal_reasons: tuple[str, ...]
    refused: bool
    next_state: tuple[Message, ...]
    request_ids: tuple[str, ...]


@dataclass(frozen=True)
class MCTSResult:
    action_index: int
    action_name: str
    response: str
    user_response: str
    reward: float
    reward_label: RewardLabel
    critic_labels: tuple[RewardLabel, ...]
    deterministic_accepted: bool
    acceptance_disagreement: bool
    terminal: bool
    terminal_reasons: tuple[str, ...]
    refused: bool
    next_state: tuple[Message, ...]
    root_visits: tuple[int, ...]
    root_q_values: tuple[float, ...]
    root_priors: tuple[float, ...]
    simulations: int
    expanded_nodes: int
    max_depth_reached: int
    request_ids: tuple[str, ...]
    tree: dict


@dataclass(frozen=True)
class DPDPCheckpointMetadata:
    phase: str
    config_sha256: str
    data_sha256: str
    action_taxonomy_sha256: str
    model_lock_sha256: str
    source_revision: str
    parent_checkpoint_sha256: str | None = None


@dataclass(frozen=True)
class DPDPPlanTrace:
    system: Literal["system1", "system2", "refusal_guard"]
    selected_action: str | None
    policy: PolicyOutput | None
    mcts: MCTSResult | None
