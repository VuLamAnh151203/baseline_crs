"""Train/dev-only weak supervision for the SCOPE task reward."""

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

from ...acceptance import normalize
from ...contracts import Message
from ...planner import derived_seed
from ...provenance import atomic_json, digest, file_hash
from ...state import DialogueState
from .semantic import target_messages

ACTION_POTENTIAL = {
    "ask_preference": 0.15,
    "clarify": 0.25,
    "acknowledge": 0.25,
    "explain": 0.45,
    "recommend": 0.65,
    "farewell": 0.10,
}
TERMINAL_POTENTIAL = {
    "worse": 0.0,
    "unchanged": 0.25,
    "progress": 0.80,
    "achieved": 1.0,
}
LABEL_PROTOCOL = "user-state-weak-action-progress-plus-final-shared-semantic-judge-v2"


def _messages(row):
    messages = tuple(Message(**item) for item in row["messages"])
    if not messages or any(
        message.role not in {"user", "assistant"} or not message.content
        for message in messages
    ):
        raise ValueError(f"invalid training dialogue: {row.get('id')}")
    return messages


def _latest_user_prefix(messages):
    index = max((i for i, message in enumerate(messages) if message.role == "user"), default=-1)
    if index < 0:
        raise ValueError("SCOPE reward labels require an observed user turn")
    return DialogueState(messages[: index + 1])


def terminal_judgment(row, messages, judge, seed, request_id):
    if judge is None:
        return None
    return judge.assess(
        _latest_user_prefix(messages),
        {**row, "latest_user_reply_finish_reason": "stop"},
        derived_seed(seed, row["id"], "scope-reward-terminal"),
        request_id,
    )


def prefix_potentials(row, messages, judgment):
    actions = row.get("assistant_actions")
    if not isinstance(actions, list) or len(actions) != len(messages):
        raise ValueError(f"missing/alignment-broken assistant actions: {row['id']}")
    target = normalize(row["target"][1])
    best = 0.05
    labels = []
    mentioned = False
    latest_user = max(i for i, message in enumerate(messages) if message.role == "user")
    for index, message in enumerate(messages):
        action = actions[index]
        if action is not None:
            if action not in ACTION_POTENTIAL:
                raise ValueError(f"unknown weak action label: {action}")
            best = max(best, ACTION_POTENTIAL[action])
        if message.role == "assistant" and target and target in normalize(message.content):
            mentioned = True
            best = max(best, 0.70)
        if message.role == "user" and mentioned:
            best = max(best, 0.75)
        label = best
        if judgment is not None and index >= latest_user:
            label = TERMINAL_POTENTIAL[judgment.progress]
        labels.append(float(label))
    return labels


def read_rows(path, limit=None):
    path = Path(path)
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows or len({row["id"] for row in rows}) != len(rows):
        raise ValueError(f"empty or duplicate SCOPE rows: {path}")
    rows.sort(key=lambda row: digest(row["id"]))
    return rows if limit is None else rows[:limit]


def prepare_split(
    source,
    output,
    dataset,
    split,
    encoder,
    judge,
    seed,
    counterfactual_negatives,
    limit=None,
    stop_requested=lambda: False,
):
    source, output = Path(source), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows = read_rows(source, limit)
    goal_dir, shard_dir = output / "goals", output / "shards"
    goal_dir.mkdir(exist_ok=True)
    shard_dir.mkdir(exist_ok=True)
    goal_vectors = {}
    for row in rows:
        key = digest(row["id"])
        path = goal_dir / f"{key}.pt"
        if path.exists():
            payload = torch.load(path, map_location="cpu", weights_only=True)
            if payload["scenario_id"] != row["id"]:
                raise ValueError("SCOPE goal cache collision")
            vector = payload["vector"].float()
        else:
            vector = encoder.embed(
                target_messages(row), f"reward-data/{dataset}/{split}/{row['id']}/goal"
            )
            torch.save({"scenario_id": row["id"], "vector": vector.half()}, path)
        goal_vectors[row["id"]] = vector
    row_ids = [row["id"] for row in rows]
    for row_index, row in enumerate(rows):
        if stop_requested():
            raise InterruptedError("SCOPE reward preparation stopped")
        shard = shard_dir / f"{digest(row['id'])}.pt"
        if shard.exists():
            payload = torch.load(shard, map_location="cpu", weights_only=True)
            if payload.get("scenario_id") != row["id"]:
                raise ValueError("SCOPE reward shard collision")
            continue
        messages = _messages(row)
        judgment = terminal_judgment(
            row,
            messages,
            judge,
            seed,
            f"reward-data/{dataset}/{split}/{row['id']}/terminal-judge",
        )
        labels = prefix_potentials(row, messages, judgment)
        state_indices = [
            index for index, message in enumerate(messages) if message.role == "user"
        ]
        if not state_indices:
            raise ValueError("SCOPE reward training requires user-state prefixes")
        states = torch.stack(
            [
                encoder.embed(
                    messages[: index + 1],
                    f"reward-data/{dataset}/{split}/{row['id']}/state/{index}",
                )
                for index in state_indices
            ]
        ).half()
        goal = goal_vectors[row["id"]].half().repeat(len(state_indices), 1)
        all_states, all_goals = [states], [goal]
        all_labels = [
            torch.tensor([labels[index] for index in state_indices], dtype=torch.float32)
        ]
        for offset in range(counterfactual_negatives):
            negative_id = row_ids[(row_index + offset + 1) % len(row_ids)]
            if negative_id == row["id"]:
                raise ValueError("counterfactual reward target requires at least two dialogues")
            all_states.append(states)
            all_goals.append(
                goal_vectors[negative_id].half().repeat(len(state_indices), 1)
            )
            all_labels.append(torch.zeros(len(state_indices), dtype=torch.float32))
        torch.save(
            {
                "schema_version": 1,
                "scenario_id": row["id"],
                "states": torch.cat(all_states),
                "goals": torch.cat(all_goals),
                "labels": torch.cat(all_labels),
                "positive_prefixes": len(state_indices),
                "state_role": "user",
                "counterfactual_negatives": counterfactual_negatives,
                "terminal_judgment": None if judgment is None else judgment.__dict__,
                "label_protocol": LABEL_PROTOCOL,
            },
            shard,
        )
    files = sorted(shard_dir.glob("*.pt"))
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "dataset": dataset,
        "split": split,
        "source": str(source),
        "source_sha256": file_hash(source),
        "dialogues": len(rows),
        "scenario_ids_sha256": digest(row_ids),
        "label_protocol": LABEL_PROTOCOL,
        "terminal_semantic_judge": judge is not None,
        "counterfactual_negatives": counterfactual_negatives,
        "diagnostic_subset": limit is not None,
        "shards": [
            {"path": path.name, "sha256": file_hash(path), "bytes": path.stat().st_size}
            for path in files
        ],
    }
    atomic_json(output / "manifest.json", manifest)
    return manifest


class RewardDataset(Dataset):
    def __init__(self, root):
        self.root = Path(root)
        manifest_path = self.root / "manifest.json"
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if self.manifest.get("status") != "complete":
            raise ValueError("complete SCOPE reward-data manifest required")
        states, goals, labels = [], [], []
        for row in self.manifest["shards"]:
            path = self.root / "shards" / row["path"]
            if file_hash(path) != row["sha256"]:
                raise ValueError(f"SCOPE reward shard changed: {path}")
            payload = torch.load(path, map_location="cpu", weights_only=True)
            states.append(payload["states"])
            goals.append(payload["goals"])
            labels.append(payload["labels"])
        self.states = torch.cat(states)
        self.goals = torch.cat(goals)
        self.labels = torch.cat(labels)
        if (
            self.states.shape != self.goals.shape
            or self.states.shape[0] != self.labels.shape[0]
            or self.states.shape[1] != 1024
        ):
            raise ValueError("invalid SCOPE reward dataset tensors")
        self.manifest_sha256 = file_hash(manifest_path)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.states[index], self.goals[index], self.labels[index]
