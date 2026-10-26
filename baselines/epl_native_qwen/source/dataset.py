import json
import re
from pathlib import Path

from torch.utils.data import Dataset

from ...contracts import Message
from ...provenance import digest, file_hash
from .actions import (
    CONTEXT_TOPIC,
    NO_TOPIC,
    TARGET_TOPIC,
    NativeAction,
    NativeActionVocabulary,
)
from .serialization import encode_policy_input


def read_jsonl(path):
    rows = [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError(f"empty EPL native source: {path}")
    return rows


def _normalize(value):
    return " ".join(re.findall(r"\w+", str(value).casefold(), flags=re.UNICODE))


def _topic_role(topic, target):
    topic, target = _normalize(topic), _normalize(target)
    if not topic or topic in {"none", "null"}:
        return NO_TOPIC
    if target and (topic == target or topic in target or target in topic):
        return TARGET_TOPIC
    return CONTEXT_TOPIC


def _topdial_action(message, scenario, assistant_index, user_seen, final_assistant):
    text = message.content.strip()
    lowered = text.casefold()
    target_goal, target_topic = scenario["target"]
    if not user_seen and assistant_index == 0:
        return NativeAction("Greetings", NO_TOPIC)
    if any(token in lowered for token in ("goodbye", "bye", "see you")) and final_assistant:
        return NativeAction("Say goodbye", NO_TOPIC)
    if _normalize(target_topic) in _normalize(text):
        return NativeAction(str(target_goal), TARGET_TOPIC)
    if "?" in text:
        return NativeAction("Ask questions", TARGET_TOPIC)
    return NativeAction("Chat about target", CONTEXT_TOPIC)


def topdial_examples(processed_path):
    examples, goals = [], set()
    for row in read_jsonl(processed_path):
        messages = tuple(Message(**message) for message in row["messages"])
        scenario = {
            "profile": row["profile"],
            "personality": row.get("personality", {}),
            "knowledge": row.get("knowledge", []),
            "target": row["target"],
        }
        action_history = []
        assistant_positions = [i for i, item in enumerate(messages) if item.role == "assistant"]
        final_position = assistant_positions[-1] if assistant_positions else -1
        user_seen = False
        assistant_index = 0
        for turn, message in enumerate(messages):
            if message.role == "user":
                user_seen = True
                continue
            if message.role != "assistant":
                raise ValueError("invalid TopDial role in EPL native data")
            action = _topdial_action(
                message,
                scenario,
                assistant_index,
                user_seen,
                turn == final_position,
            )
            goals.add(action.goal)
            if messages[:turn]:
                examples.append(
                    {
                        "id": f"{row['id']}:{turn}",
                        "scenario": scenario,
                        "history": messages[:turn],
                        "action_history": tuple(action_history),
                        "action": action,
                    }
                )
            action_history.append(action)
            assistant_index += 1
    return examples, goals


def durecdial_examples(processed_path, raw_path):
    processed = read_jsonl(processed_path)
    raw = read_jsonl(raw_path)
    if len(processed) != len(raw):
        raise ValueError("DuRecDial raw/processed row count mismatch")
    examples, goals = [], set()
    for index, (row, source) in enumerate(zip(processed, raw, strict=True)):
        expected_id = f"{'train' if 'train' in Path(processed_path).name else 'dev'}:en:{index:06d}"
        if row["id"] != expected_id:
            raise ValueError("DuRecDial normalized order/id changed")
        messages = tuple(Message(**message) for message in row["messages"])
        goal_labels, topic_labels = source["goal_type_list"], source["goal_topic_list"]
        if not (len(messages) == len(goal_labels) == len(topic_labels)):
            raise ValueError("DuRecDial native goal/topic labels are not turn aligned")
        scenario = {
            "profile": row["profile"],
            "personality": row.get("personality", {}),
            "knowledge": row.get("knowledge", []),
            "target": row["target"],
        }
        action_history = []
        for turn, message in enumerate(messages):
            action = NativeAction(
                str(goal_labels[turn]).strip(),
                _topic_role(topic_labels[turn], row["target"][1]),
            )
            goals.add(action.goal)
            if message.role == "assistant":
                if messages[:turn]:
                    examples.append(
                        {
                            "id": f"{row['id']}:{turn}",
                            "scenario": scenario,
                            "history": messages[:turn],
                            "action_history": tuple(action_history),
                            "action": action,
                        }
                    )
                action_history.append(action)
            elif message.role != "user":
                raise ValueError("invalid DuRecDial role in EPL native data")
    return examples, goals


def load_native_examples(root, dataset, split):
    root = Path(root)
    processed = root / "data/processed" / dataset / f"{split}.jsonl"
    if dataset == "TopDial":
        return topdial_examples(processed)
    if dataset == "DuRecDial2.0":
        return durecdial_examples(
            processed, root / "data/raw/DuRecDial2.0" / f"en_{split}.txt"
        )
    raise ValueError("unknown EPL native dataset")


def build_vocabulary(root, dataset):
    _, goals = load_native_examples(root, dataset, "train")
    source = (
        "released_DuRecDial_goal_type_list_with_target_relative_topic_v1"
        if dataset == "DuRecDial2.0"
        else "declared_TopDial_target_with_deterministic_goal_adapter_v1"
    )
    return NativeActionVocabulary(dataset, goals, source)


class EPLNativeDataset(Dataset):
    def __init__(self, root, dataset, split, tokenizer, config, vocabulary):
        self.examples, _ = load_native_examples(root, dataset, split)
        self.tokenizer = tokenizer
        self.config = config
        self.vocabulary = vocabulary
        self.source_sha256 = file_hash(
            Path(root) / "data/processed" / dataset / f"{split}.jsonl"
        )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        row = self.examples[index]
        ids, mask = encode_policy_input(
            self.tokenizer,
            row["scenario"],
            row["history"],
            row["action_history"],
            self.config.max_sequence_length,
        )
        return {
            "id": row["id"],
            "input_ids": ids,
            "attention_mask": mask,
            "label": self.vocabulary.index(row["action"]),
        }


def dataset_signature(dataset):
    return {
        "source_sha256": dataset.source_sha256,
        "examples_sha256": digest([row["id"] for row in dataset.examples]),
        "count": len(dataset),
    }


__all__ = [
    "EPLNativeDataset",
    "build_vocabulary",
    "dataset_signature",
    "load_native_examples",
]
