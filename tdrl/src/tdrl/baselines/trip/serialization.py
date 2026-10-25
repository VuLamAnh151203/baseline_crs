import json
import os
import re
import tempfile
import zipfile
from pathlib import Path

from ...official_data import SPLITS, strategy_label
from ...provenance import digest, file_hash

DATASETS = {"TopDial", "DuRecDial2.0"}
SPECIAL_TOKENS = ("[A]", "[T]", "[BOS]", "[EOS]")
NULL_TOPIC = "NULL"
SERIALIZATION_VERSION = 1


def atomic_text(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=".pending-")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _validate_record(record, dataset):
    required = {"id", "profile", "knowledge", "target", "messages"}
    if dataset not in DATASETS or not required.issubset(record):
        raise ValueError("invalid canonical TRIP record")
    if not isinstance(record["messages"], list) or not record["messages"]:
        raise ValueError("TRIP requires a nonempty dialogue")
    if not isinstance(record["target"], list) or len(record["target"]) != 2:
        raise ValueError("TRIP target must be [action, topic]")
    for message in record["messages"]:
        if (
            not isinstance(message, dict)
            or message.get("role") not in {"user", "assistant"}
            or not isinstance(message.get("content"), str)
            or not message["content"].strip()
        ):
            raise ValueError("invalid canonical dialogue message")


def _goal_nodes(goal):
    nodes = []
    for expression in goal.split("-->"):
        expression = re.sub(r"^\s*\[\d+\]\s*", "", expression).strip()
        if not expression or expression.casefold().replace(" ", "") == "saygoodbye":
            continue
        match = re.fullmatch(r"([^()]+)\((.*)\)", expression)
        action, topic = (
            (match.group(1).strip(), match.group(2).strip())
            if match
            else (expression, NULL_TOPIC)
        )
        nodes.append((action, topic or NULL_TOPIC))
    if not nodes:
        raise ValueError("DuRecDial goal has no usable nodes")
    return nodes


def _turn_stage(text, previous):
    match = re.match(r"^\s*\[(\d+)\]", text)
    return max(0, int(match.group(1)) - 1) if match else previous


def load_durecdial_source(archive: Path, split: str):
    if split not in {"train", "dev"}:
        raise ValueError("TRIP source labels are restricted to train/dev")
    member = SPLITS["DuRecDial2.0"][split]
    with zipfile.ZipFile(archive) as source:
        rows = [json.loads(line) for line in source.read(member).splitlines() if line.strip()]
    return rows


def _entities(knowledge):
    values = set()
    for triple in knowledge:
        if not isinstance(triple, list) or len(triple) != 3:
            raise ValueError("TRIP knowledge must contain triples")
        for value in (triple[0], triple[2]):
            if isinstance(value, str) and value.strip():
                values.add(value.strip())
    return sorted(values, key=lambda value: (-len(value), value.casefold(), value))


def _topic_for_text(text, knowledge, target_topic):
    folded = text.casefold()
    if target_topic.casefold() in folded:
        return target_topic
    for entity in _entities(knowledge):
        if entity.casefold() in folded:
            return entity
    return NULL_TOPIC


def _deduplicate_path(path):
    result = []
    for action, topic in path:
        pair = (action.strip(), topic.strip() or NULL_TOPIC)
        if not result or result[-1] != pair:
            result.append(pair)
    return result


def _topdial_paths(record):
    actions = record.get("assistant_actions")
    if actions is not None and len(actions) != len(record["messages"]):
        raise ValueError("TopDial assistant action alignment mismatch")
    path = []
    positions = []
    for index, message in enumerate(record["messages"]):
        if message["role"] != "assistant":
            continue
        action = (
            actions[index]
            if actions and actions[index]
            else strategy_label(message["content"])
        )
        topic = _topic_for_text(message["content"], record["knowledge"], record["target"][1])
        path.append((action, topic))
        positions.append(index)
    return positions, path, "weak_path_v1"


def _durecdial_paths(record, source):
    nodes = _goal_nodes(source["goal"])
    raw_turns = source["conversation"]
    if len(raw_turns) != len(record["messages"]):
        raise ValueError("DuRecDial source/canonical turn alignment mismatch")
    stages, stage = [], 0
    for text in raw_turns:
        stage = _turn_stage(text, stage)
        stages.append(min(stage, len(nodes) - 1))
    positions = [
        index
        for index, message in enumerate(record["messages"])
        if message["role"] == "assistant"
    ]
    paths = [nodes[stages[position] :] for position in positions]
    return positions, paths, "official_goal_path_v1"


def _examples(record, dataset, source=None):
    _validate_record(record, dataset)
    target = tuple(value.strip() for value in record["target"])
    if dataset == "TopDial":
        positions, full_path, label_source = _topdial_paths(record)
        paths = [full_path[offset:] for offset in range(len(full_path))]
    else:
        if source is None:
            raise ValueError("DuRecDial TRIP preparation requires its pinned raw source row")
        positions, paths, label_source = _durecdial_paths(record, source)
    rows = []
    for position, path in zip(positions, paths, strict=True):
        path = _deduplicate_path(path)
        if not path or path[-1] != target:
            path.append(target)
        rows.append(
            {
                "id": f"{record['id']}:{position}",
                "dialogue_id": record["id"],
                "turn": position,
                "profile": record["profile"],
                "personality": record.get("personality", {}),
                "knowledge": record["knowledge"],
                "target": list(target),
                "messages": record["messages"][:position],
                "action_path": [pair[0] for pair in path],
                "topic_path": [pair[1] for pair in path],
                "label_source": label_source,
                "source_sha256": digest(record),
            }
        )
    return rows


def prepare_corpus(input_path, output_path, dataset, split, source_archive=None, num_negatives=3):
    if dataset not in DATASETS or split not in {"train", "dev"}:
        raise ValueError("TRIP fitting accepts only canonical train/dev data")
    records = [
        json.loads(line)
        for line in Path(input_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not records or len({row["id"] for row in records}) != len(records):
        raise ValueError("TRIP canonical input must have unique, nonempty records")
    source_rows = None
    if dataset == "DuRecDial2.0":
        if source_archive is None:
            raise ValueError("DuRecDial TRIP preparation requires --source-archive")
        source_rows = load_durecdial_source(Path(source_archive), split)
    examples = []
    for record in records:
        source = None
        if source_rows is not None:
            match = re.fullmatch(rf"{split}:en:(\d{{6}})", record["id"])
            if not match:
                raise ValueError("unexpected canonical DuRecDial ID")
            source = source_rows[int(match.group(1))]
        examples.extend(_examples(record, dataset, source))
    topics = sorted(
        {
            topic
            for row in examples
            for topic in [*row["topic_path"], *_entities(row["knowledge"])]
            if topic != NULL_TOPIC
        },
        key=lambda value: (value.casefold(), value),
    )
    if len(topics) <= num_negatives:
        raise ValueError("TRIP corpus does not contain enough negative topics")
    for row in examples:
        candidates = [topic for topic in topics if topic not in set(row["topic_path"])]
        candidates.sort(key=lambda topic: digest([row["id"], topic]))
        if len(candidates) < num_negatives:
            raise ValueError("TRIP sample does not have enough deterministic negative topics")
        row["negative_topics"] = candidates[:num_negatives]
    content = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in examples
    )
    atomic_text(Path(output_path), content)
    manifest = {
        "schema_version": SERIALIZATION_VERSION,
        "status": "complete",
        "dataset": dataset,
        "split": split,
        "records": len(examples),
        "dialogues": len(records),
        "label_source": "weak_path_v1" if dataset == "TopDial" else "official_goal_path_v1",
        "source_sha256": file_hash(Path(input_path)),
        "source_archive_sha256": file_hash(Path(source_archive)) if source_archive else None,
        "output_sha256": file_hash(Path(output_path)),
        "ids_sha256": digest([row["id"] for row in examples]),
        "serialization_sha256": digest(
            {
                "version": SERIALIZATION_VERSION,
                "special_tokens": SPECIAL_TOKENS,
                "future_test_turns": "forbidden",
                "topdial_labels": "weak_path_v1",
                "durecdial_labels": "official_goal_path_v1",
            }
        ),
    }
    return manifest


def visible_input(record, state):
    """Build a deterministic planner input from only the currently observed state."""
    supplied = {
        "profile": record["profile"],
        "personality": record.get("personality", {}),
        "knowledge": record["knowledge"],
        "target": record["target"],
        "messages": [
            {"role": message.role, "content": message.content} for message in state.history
        ],
    }
    return supplied
