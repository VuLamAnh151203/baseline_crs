"""Strict configuration for the SCOPE-CRS-lite adaptation."""

import json
import math
from dataclasses import dataclass
from pathlib import Path

from ...provenance import file_hash


@dataclass(frozen=True)
class ScopeConfig:
    dataset: str
    embedding_dim: int
    projection_seed: int
    candidates: int
    search_depth: int
    mcts_seconds: float
    gamma: float
    uct_coefficient: float
    transition_samples: int
    q_learning_rate: float
    q_update_steps: int
    reward_hidden_dims: tuple[int, ...]
    reward_epochs: int
    reward_batch_size: int
    reward_learning_rate: float
    reward_weight_decay: float
    reward_patience: int
    training_seed: int
    counterfactual_negatives: int
    terminal_semantic_judge: bool
    max_train_dialogues: int | None
    max_dev_dialogues: int | None
    sha256: str

    @classmethod
    def from_path(cls, path):
        path = Path(path)
        raw = json.loads(path.read_text(encoding="utf-8"))
        required = {
            "schema_version",
            "method",
            "dataset",
            "semantic_space",
            "planning",
            "reward_training",
        }
        if set(raw) != required or raw["schema_version"] != 1:
            raise ValueError("unknown/incomplete SCOPE configuration")
        if raw["method"] != "SCOPE-CRS-lite" or raw["dataset"] not in {
            "TopDial",
            "DuRecDial2.0",
        }:
            raise ValueError("invalid SCOPE method or dataset")
        semantic = raw["semantic_space"]
        if semantic != {
            "encoder": "meta-llama/Meta-Llama-Guard-2-8B",
            "revision": "7d257f3c1a0ec6ed99b2cb715027149dfb9784ef",
            "embedding_dim": 1024,
            "projection_seed": 42,
            "transition": "official-lmsys-chat-1m-four-seed-ensemble",
        }:
            raise ValueError("SCOPE semantic-space contract changed")
        planning = raw["planning"]
        if set(planning) != {
            "candidates",
            "search_depth",
            "mcts_seconds",
            "gamma",
            "uct_coefficient",
            "transition_samples",
            "q_learning_rate",
            "q_update_steps",
        }:
            raise ValueError("invalid SCOPE planning configuration")
        reward = raw["reward_training"]
        if set(reward) != {
            "hidden_dims",
            "epochs",
            "batch_size",
            "learning_rate",
            "weight_decay",
            "patience",
            "seed",
            "counterfactual_negatives",
            "terminal_semantic_judge",
            "max_train_dialogues",
            "max_dev_dialogues",
        }:
            raise ValueError("invalid SCOPE reward-training configuration")
        integer_positive = (
            planning["candidates"],
            planning["search_depth"],
            planning["transition_samples"],
            planning["q_update_steps"],
            reward["epochs"],
            reward["batch_size"],
            reward["patience"],
            reward["counterfactual_negatives"],
        )
        if any(type(value) is not int or value < 1 for value in integer_positive):
            raise ValueError("SCOPE count settings must be positive integers")
        if planning["candidates"] != 5 or planning["search_depth"] != 6:
            raise ValueError("SCOPE native five-candidate/depth-six contract changed")
        if planning["transition_samples"] != 4:
            raise ValueError("SCOPE transition ensemble must contain four seeds")
        for value in (
            planning["mcts_seconds"],
            planning["gamma"],
            planning["uct_coefficient"],
            planning["q_learning_rate"],
            reward["learning_rate"],
            reward["weight_decay"],
        ):
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError("invalid finite SCOPE scalar")
        if not 0 < planning["gamma"] <= 1 or planning["mcts_seconds"] <= 0:
            raise ValueError("invalid SCOPE search horizon")
        hidden = reward["hidden_dims"]
        if not isinstance(hidden, list) or not hidden or any(
            type(value) is not int or value < 1 for value in hidden
        ):
            raise ValueError("invalid SCOPE reward architecture")
        for key in ("max_train_dialogues", "max_dev_dialogues"):
            if reward[key] is not None and (
                type(reward[key]) is not int or reward[key] < 1
            ):
                raise ValueError(f"invalid SCOPE diagnostic limit: {key}")
        if type(reward["terminal_semantic_judge"]) is not bool:
            raise ValueError("terminal_semantic_judge must be boolean")
        return cls(
            dataset=raw["dataset"],
            embedding_dim=semantic["embedding_dim"],
            projection_seed=semantic["projection_seed"],
            candidates=planning["candidates"],
            search_depth=planning["search_depth"],
            mcts_seconds=float(planning["mcts_seconds"]),
            gamma=float(planning["gamma"]),
            uct_coefficient=float(planning["uct_coefficient"]),
            transition_samples=planning["transition_samples"],
            q_learning_rate=float(planning["q_learning_rate"]),
            q_update_steps=planning["q_update_steps"],
            reward_hidden_dims=tuple(hidden),
            reward_epochs=reward["epochs"],
            reward_batch_size=reward["batch_size"],
            reward_learning_rate=float(reward["learning_rate"]),
            reward_weight_decay=float(reward["weight_decay"]),
            reward_patience=reward["patience"],
            training_seed=reward["seed"],
            counterfactual_negatives=reward["counterfactual_negatives"],
            terminal_semantic_judge=reward["terminal_semantic_judge"],
            max_train_dialogues=reward["max_train_dialogues"],
            max_dev_dialogues=reward["max_dev_dialogues"],
            sha256=file_hash(path),
        )
