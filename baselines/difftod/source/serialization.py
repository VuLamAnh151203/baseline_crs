import json
import os
import re
import tempfile
from pathlib import Path

from ...provenance import digest, file_hash

DATASETS = {"TopDial", "DuRecDial2.0"}
ROLE_MAP = {"assistant": "system", "user": "user"}
SERIALIZATION_VERSION = 1


def _scalar(value):
    if isinstance(value, str):
        return value.strip()
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _profile_text(profile):
    if not isinstance(profile, dict):
        raise ValueError("canonical DiffTOD profile must be an object")
    fields = [f"{key}: {_scalar(profile[key])}" for key in sorted(profile)]
    return "user profile: " + "\t".join(fields)


def _target_text(target):
    if (
        not isinstance(target, list)
        or len(target) != 2
        or any(not isinstance(value, str) or not value.strip() for value in target)
    ):
        raise ValueError("canonical DiffTOD target must contain action and item")
    return f"target: {target[0].strip()}, {target[1].strip()}"


def _knowledge_text(knowledge):
    if not isinstance(knowledge, list):
        raise ValueError("canonical permitted knowledge must be a list")
    if not knowledge:
        return ""
    return "permitted knowledge: " + json.dumps(
        knowledge, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def validate_canonical(record, dataset):
    if dataset not in DATASETS:
        raise ValueError("DiffTOD supports only the two frozen comparison datasets")
    required = {"id", "profile", "personality", "knowledge", "target", "messages"}
    if not required.issubset(record):
        raise ValueError("canonical DiffTOD record is missing required fields")
    if not isinstance(record["id"], str) or not record["id"].strip():
        raise ValueError("canonical record requires a stable scenario ID")
    _profile_text(record["profile"])
    _target_text(record["target"])
    _knowledge_text(record["knowledge"])
    if not isinstance(record["messages"], list) or not record["messages"]:
        raise ValueError("training record requires a nonempty observed trajectory")
    for message in record["messages"]:
        if (
            not isinstance(message, dict)
            or set(message) != {"role", "content"}
            or message["role"] not in ROLE_MAP
            or not isinstance(message["content"], str)
            or not message["content"].strip()
        ):
            raise ValueError("unknown canonical dialogue message")


def condition_header(record, dataset):
    """Visible planner condition; deliberately excludes every reference dialogue turn."""
    validate_canonical(record, dataset)
    lines = [_profile_text(record["profile"])]
    if dataset == "DuRecDial2.0":
        knowledge = _knowledge_text(record["knowledge"])
        if knowledge:
            lines.append(knowledge)
    return "\n".join(lines) + "\n"


def training_trajectory(record, dataset):
    validate_canonical(record, dataset)
    lines = [condition_header(record, dataset).rstrip("\n"), _target_text(record["target"])]
    lines.extend(
        f"{ROLE_MAP[message['role']]}: {message['content'].strip()}"
        for message in record["messages"]
    )
    return "\n".join(lines) + "\n"


def semantic_target(record):
    return {"action": record["target"][0], "item": record["target"][1]}


def initial_observation(record):
    """Shared initial state, exposed to the Qwen realizer but not diffusion plan generation."""
    prefix = []
    for message in record["messages"]:
        prefix.append(message)
        if message["role"] == "user":
            return prefix
    raise ValueError("scenario has no legitimate initial user observation")


def atomic_text(path, content):
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


def prepare_corpus(input_path: Path, output_path: Path, dataset, split):
    if dataset not in DATASETS or split not in {"train", "dev"}:
        raise ValueError("DiffTOD fitting accepts only canonical train/dev records")
    rows = []
    ids = set()
    with input_path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            validate_canonical(record, dataset)
            if record["id"] in ids:
                raise ValueError(f"duplicate canonical ID at line {line_number}")
            ids.add(record["id"])
            rows.append(
                {
                    "id": record["id"],
                    "source_sha256": digest(record),
                    "text": training_trajectory(record, dataset),
                }
            )
    if not rows:
        raise ValueError("cannot prepare an empty DiffTOD corpus")
    content = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    )
    atomic_text(output_path, content)
    return {
        "schema_version": SERIALIZATION_VERSION,
        "dataset": dataset,
        "split": split,
        "records": len(rows),
        "source_sha256": file_hash(input_path),
        "output_sha256": file_hash(output_path),
        "ids_sha256": digest([row["id"] for row in rows]),
        "serialization_sha256": digest(
            {
                "version": SERIALIZATION_VERSION,
                "roles": ROLE_MAP,
                "durecdial_knowledge": "canonical_permitted_only",
                "test_reference_turns": "excluded_from_condition",
            }
        ),
    }


def extract_dialogue_plan(decoded, condition):
    if not isinstance(decoded, str) or not decoded:
        raise ValueError("decoded DiffTOD trajectory is empty")
    text = decoded.replace("<|endoftext|>", "\n").replace("\x00", " ")
    if text.startswith(condition):
        text = text[len(condition) :]
    match = re.search(r"(?:^|\n)(?:system|user):\s*", text, flags=re.IGNORECASE)
    if match:
        text = text[match.start() :].lstrip()
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if not text:
        raise ValueError("DiffTOD trajectory contains no dialogue plan")
    return text
