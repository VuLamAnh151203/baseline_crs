import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

from ...acceptance import classify_user
from ...contracts import Message
from ...official_data import ACTION_TAXONOMY
from ...provenance import atomic_json, digest, file_hash
from ...state import DialogueState
from .contracts import DPDPTransition
from .serialization import SERIALIZATION_SHA256, encode_policy_input


def _read_jsonl(path):
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def build_transitions(record, dataset, split, source_sha256):
    required = {
        "id",
        "messages",
        "profile",
        "personality",
        "knowledge",
        "target",
        "assistant_actions",
        "action_label_source",
    }
    if set(record) != required:
        raise ValueError("unexpected canonical DPDP record fields")
    messages = tuple(Message(**row) for row in record["messages"])
    actions = record["assistant_actions"]
    if len(actions) != len(messages):
        raise ValueError("assistant action alignment mismatch")
    action_to_index = {name: index for index, name in enumerate(ACTION_TAXONOMY)}
    transitions = []
    for index, message in enumerate(messages[:-1]):
        if message.role != "assistant" or messages[index + 1].role != "user":
            continue
        action = actions[index]
        if action not in action_to_index:
            raise ValueError(f"unknown DPDP action: {action}")
        state = messages[:index]
        next_state = messages[: index + 2]
        if not state:
            # A first system greeting contains no observed conversational state and is not
            # a valid policy transition for this adaptation.
            continue
        label = classify_user(DialogueState(next_state), record["target"])
        done = bool(label.accepted or index + 2 == len(messages))
        transition_id = f"{record['id']}:{index}"
        transitions.append(
            DPDPTransition(
                transition_id=transition_id,
                scenario_id=record["id"],
                dataset=dataset,
                split=split,
                turn_index=index,
                state=state,
                next_state=next_state,
                action_index=action_to_index[action],
                action_name=action,
                reward=None,
                discounted_return=None,
                done=done,
                profile=record["profile"],
                target=tuple(record["target"]),
                provenance={
                    "source_sha256": source_sha256,
                    "serialization_sha256": SERIALIZATION_SHA256,
                    "action_taxonomy_sha256": digest(ACTION_TAXONOMY),
                    "action_label_source": record["action_label_source"],
                },
            )
        )
    return transitions


def prepare_transitions(input_path, output_path, dataset, split):
    if dataset not in {"TopDial", "DuRecDial2.0"} or split not in {"train", "dev"}:
        raise ValueError("DPDP preparation accepts only canonical train/dev splits")
    input_path, output_path = Path(input_path), Path(output_path)
    manifest_path = output_path.with_suffix(output_path.suffix + ".manifest.json")
    if output_path.exists() or manifest_path.exists():
        raise FileExistsError("DPDP prepared artifacts are immutable")
    source_sha256 = file_hash(input_path)
    transitions = []
    scenario_ids = []
    for row in _read_jsonl(input_path):
        if row["id"] in scenario_ids:
            raise ValueError("duplicate DPDP scenario ID")
        scenario_ids.append(row["id"])
        transitions.extend(build_transitions(row, dataset, split, source_sha256))
    ids = [row.transition_id for row in transitions]
    if not transitions or len(ids) != len(set(ids)):
        raise ValueError("empty or duplicate DPDP transitions")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".pending")
    temporary.write_text(
        "".join(
            json.dumps(row.to_dict(), ensure_ascii=False, sort_keys=True) + "\n"
            for row in transitions
        ),
        encoding="utf-8",
    )
    temporary.replace(output_path)
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "dataset": dataset,
        "split": split,
        "input_sha256": source_sha256,
        "output_sha256": file_hash(output_path),
        "action_taxonomy_sha256": digest(ACTION_TAXONOMY),
        "serialization_sha256": SERIALIZATION_SHA256,
        "scenario_ids": scenario_ids,
        "transition_ids_sha256": digest(ids),
        "transition_count": len(transitions),
        "test_bodies": "never_opened",
    }
    atomic_json(manifest_path, manifest)
    return manifest


class DPDPDataset(Dataset):
    def __init__(self, path, tokenizer, config):
        self.rows = _read_jsonl(path)
        self.tokenizer, self.config = tokenizer, config
        if not self.rows:
            raise ValueError("empty DPDP dataset")
        for row in self.rows:
            if row["dataset"] != config.dataset or row["split"] not in {"train", "dev"}:
                raise ValueError("DPDP dataset/config mismatch")
            if row["action_name"] != config.actions[row["action_index"]]:
                raise ValueError("DPDP action index mismatch")
            if row["reward"] is None or row["discounted_return"] is None:
                raise ValueError("DPDP training requires reward-annotated transitions")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        scenario = {"profile": row["profile"], "target": row["target"]}
        ids, mask = encode_policy_input(
            self.tokenizer,
            scenario,
            row["state"],
            self.config.max_sequence_length,
            self.config.header_tokens,
        )
        return {
            "transition_id": row["transition_id"],
            "input_ids": ids,
            "attention_mask": mask,
            "action": row["action_index"],
            "return": row["discounted_return"],
        }


class DPDPCollator:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, rows):
        if not rows:
            raise ValueError("cannot collate an empty DPDP batch")
        length = max(len(row["input_ids"]) for row in rows)
        ids, masks = [], []
        for row in rows:
            padding = length - len(row["input_ids"])
            ids.append(
                torch.nn.functional.pad(
                    row["input_ids"], (0, padding), value=self.pad_token_id
                )
            )
            masks.append(torch.nn.functional.pad(row["attention_mask"], (0, padding), value=0))
        return {
            "transition_ids": [row["transition_id"] for row in rows],
            "input_ids": torch.stack(ids),
            "attention_mask": torch.stack(masks),
            "actions": torch.tensor([row["action"] for row in rows], dtype=torch.long),
            "returns": torch.tensor([row["return"] for row in rows], dtype=torch.float32),
        }
