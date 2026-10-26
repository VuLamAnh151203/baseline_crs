import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

from ...contracts import Message
from ...provenance import digest, file_hash
from .serialization import encode_policy_input


def read_jsonl(path):
    rows = [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows or len({row["id"] for row in rows}) != len(rows):
        raise ValueError("empty or duplicate EPL source scenarios")
    return rows


def policy_examples(path, dataset, actions):
    action_to_index = {action: index for index, action in enumerate(actions)}
    examples = []
    for row in read_jsonl(path):
        messages = tuple(Message(**message) for message in row["messages"])
        labels = row["assistant_actions"]
        if len(labels) != len(messages):
            raise ValueError("EPL action labels are not turn aligned")
        scenario = {"profile": row["profile"], "target": row["target"]}
        for turn, message in enumerate(messages):
            if message.role != "assistant" or not messages[:turn]:
                continue
            action = labels[turn]
            if action not in action_to_index:
                raise ValueError(f"unknown EPL action label: {action}")
            examples.append(
                {
                    "id": f"{row['id']}:{turn}",
                    "dataset": dataset,
                    "scenario": scenario,
                    "history": messages[:turn],
                    "label": action_to_index[action],
                    "action": action,
                }
            )
    if not examples or len({row["id"] for row in examples}) != len(examples):
        raise ValueError("empty or duplicate EPL policy examples")
    return examples


class EPLDataset(Dataset):
    def __init__(self, path, dataset, tokenizer, config):
        self.examples = policy_examples(path, dataset, config.actions)
        self.tokenizer = tokenizer
        self.config = config
        self.source_sha256 = file_hash(Path(path))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        row = self.examples[index]
        ids, mask = encode_policy_input(
            self.tokenizer,
            row["scenario"],
            row["history"],
            self.config.max_sequence_length,
            self.config.header_tokens,
        )
        return {"id": row["id"], "input_ids": ids, "attention_mask": mask, "label": row["label"]}


class EPLCollator:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, rows):
        if not rows:
            raise ValueError("cannot collate an empty EPL batch")
        length = max(len(row["input_ids"]) for row in rows)
        ids, masks = [], []
        for row in rows:
            padding = length - len(row["input_ids"])
            ids.append(torch.nn.functional.pad(row["input_ids"], (0, padding), value=self.pad_token_id))
            masks.append(torch.nn.functional.pad(row["attention_mask"], (0, padding), value=0))
        return {
            "ids": [row["id"] for row in rows],
            "input_ids": torch.stack(ids),
            "attention_mask": torch.stack(masks),
            "labels": torch.tensor([row["label"] for row in rows], dtype=torch.long),
        }


def dataset_signature(path, dataset, actions):
    examples = policy_examples(path, dataset, actions)
    return {"source_sha256": file_hash(Path(path)), "examples_sha256": digest([row["id"] for row in examples]), "count": len(examples)}
