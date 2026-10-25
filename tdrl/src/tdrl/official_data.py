"""Adapters for the pinned official releases; test bodies remain sealed until selection."""

import json
import re
import zipfile
from pathlib import Path

from .contracts import Message
from .near_duplicates import audit_train_dev
from .provenance import atomic_json, digest, file_hash
from .state import DialogueState

SPLITS = {
    "TopDial": {
        "train": "topdial/dialogue_train.jsonl",
        "dev": "topdial/dialogue_dev.jsonl",
        "test_seen": "topdial/dialogue_test_seen.jsonl",
        "test_unseen": "topdial/dialogue_test_unseen.jsonl",
    },
    "DuRecDial2.0": {
        "train": "data/en_train.txt",
        "dev": "data/en_dev.txt",
        "test": "data/en_test.txt",
    },
}
LICENSES = {
    "TopDial": "Apache-2.0 (official repository; linked release has no separate license file)",
    "DuRecDial2.0": "CC-BY-NC-SA-4.0 (dataset); Apache-2.0 (code)",
}
ADAPTER_VERSION = 1
ACTION_TAXONOMY = ["acknowledge", "ask_preference", "clarify", "explain", "recommend", "farewell"]


def strategy_label(text):
    """Declared weak supervision for this recommendation-task adaptation, never gold labels."""
    t = text.casefold()
    if re.search(r"\b(?:goodbye|bye|see you)\b", t):
        return "farewell"
    if re.search(
        r"\b(?:recommend|suggest|you (?:can|could|should) (?:watch|listen|try|visit))\b", t
    ):
        return "recommend"
    if "?" in text:
        return (
            "ask_preference"
            if re.search(r"\b(?:like|prefer|favorite|interest|enjoy)\b", t)
            else "clarify"
        )
    if re.search(r"\b(?:because|stars|rating|directed|genre|cost|price|features)\b", t):
        return "explain"
    return "acknowledge"


def adapt(dataset, row, index):
    if dataset == "TopDial":
        history = []
        for turn in row["conversation"]:
            if len(turn) != 1 or next(iter(turn)) not in {"system", "user"}:
                raise ValueError("unknown official TopDial role schema")
            role, content = next(iter(turn.items()))
            history.append(Message("assistant" if role == "system" else "user", content))
        target = row["target"]
        scenario_id = row["id"]
        personality = row["user_personality"]
        knowledge = row["knowledge"]
    else:
        expressions = [re.sub(r"^\s*\[\d+\]\s*", "", x).strip() for x in row["goal"].split("-->")]
        meaningful = [e for e in expressions if e.casefold().replace(" ", "") != "saygoodbye"]
        if not meaningful:
            raise ValueError("official goal declaration contains no target")
        goal = meaningful[-1]
        match = re.fullmatch(r"([^()]+)\((.*)\)", goal)
        target = [match[1].strip(), match[2].strip()] if match else [goal, goal]
        # Official documentation: seeker starts only when the first goal is Greetings.
        assistant_first = expressions[0].casefold().startswith("greetings")
        history = [
            Message(
                "assistant" if (i % 2 == 0) == assistant_first else "user",
                re.sub(r"^\s*\[\d+\]\s*", "", text),
            )
            for i, text in enumerate(row["conversation"])
        ]
        scenario_id = (
            f"en:{index:06d}"  # Split prefix is attached below; source row order is immutable.
        )
        personality = {}
        # Turn-aligned knowledge at future turns is not legitimate initial information.
        initial_user = 1 if assistant_first else 0
        knowledge = row["knowledge"][: initial_user + 1]
    if len(target) != 2 or not all(isinstance(x, str) and x.strip() for x in target):
        raise ValueError("invalid declared target")
    if not history:
        raise ValueError("empty official dialogue")
    DialogueState(tuple(history))
    return {
        "id": scenario_id,
        "profile": row["user_profile"],
        "personality": personality,
        "knowledge": knowledge,
        "target": target,
        "messages": [{"role": m.role, "content": m.content} for m in history],
        "assistant_actions": [
            strategy_label(m.content) if m.role == "assistant" else None for m in history
        ],
        "action_label_source": "weak_rule_v1",
    }


def read_release(root, dataset, split, selection_verified=False):
    if split.startswith("test") and not selection_verified:
        raise ValueError("official test bodies are sealed until selection verification")
    archive = root / "data/raw" / dataset / "data.zip"
    with zipfile.ZipFile(archive) as z:
        raw = z.read(SPLITS[dataset][split])
    rows = []
    for index, line in enumerate(raw.splitlines()):
        if not line.strip():
            continue
        record = adapt(dataset, json.loads(line), index)
        if dataset == "DuRecDial2.0":
            record["id"] = split + ":" + record["id"]
        rows.append(record)
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("duplicate official scenario ID")
    return rows


def prepare_releases(root: Path):
    sources = json.loads((root / "configs/sources.lock.json").read_text())["sources"]
    result = {
        "schema_version": 1,
        "status": "partial",
        "datasets": {},
        "test_bodies": "sealed",
        "adapter_sha256": file_hash(Path(__file__)),
    }
    for dataset, mapping in SPLITS.items():
        archive = root / "data/raw" / dataset / "data.zip"
        with zipfile.ZipFile(archive) as z:
            if not set(mapping.values()).issubset(z.namelist()):
                raise ValueError("official split topology differs from pinned adapter")
        train, dev = read_release(root, dataset, "train"), read_release(root, dataset, "dev")
        train_hashes = {digest(r["messages"]) for r in train}
        if train_hashes & {digest(r["messages"]) for r in dev}:
            raise ValueError("exact train/development dialogue leakage")

        # Audit source-family IDs separately from exact text. Preserve official membership;
        # shared provenance families need a recorded, pre-fitting exclusion decision.
        def family(r, name=dataset):
            return r["id"].rsplit("_", 1)[0] if name == "TopDial" else r["id"]

        family_overlap = sorted({family(r) for r in train} & {family(r) for r in dev})
        near = audit_train_dev(train, dev)
        # Frozen preprocessing rule: remove contaminated training examples, never development.
        excluded = {r["train_id"] for r in near["collisions"]}
        excluded |= {r["id"] for r in train if family(r) in family_overlap}
        original_train_count = len(train)
        train = [r for r in train if r["id"] not in excluded]
        output = root / "data/processed" / dataset
        output.mkdir(parents=True, exist_ok=True)
        hashes = {}
        for split, rows in (("train", train), ("dev", dev)):
            path = output / (split + ".jsonl")
            content = "".join(
                json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
            )
            if path.exists() and path.read_text() != content:
                raise ValueError("prepared corpus provenance mismatch")
            path.write_text(content)
            hashes[split] = file_hash(path)
        result["datasets"][dataset] = {
            "source": sources[dataset],
            "license": LICENSES[dataset],
            "archive_sha256": file_hash(archive),
            "language": "English official release",
            "count": {"train": len(train), "dev": len(dev)},
            "prepared_sha256": hashes,
            "ids_sha256": {
                "train": digest([r["id"] for r in train]),
                "dev": digest([r["id"] for r in dev]),
            },
            "official_test_members": {k: v for k, v in mapping.items() if k.startswith("test")},
            "exact_train_dev_duplicates": 0,
            "source_family_overlap": family_overlap,
            "near_duplicate_audit": near,
            "training_exclusion_rule": "exclude_dev_family_and_jaccard_ge_0.9_v1",
            "original_train_count": original_train_count,
            "excluded_train_ids": sorted(excluded),
            "test_duplicate_audit": "post_selection_only",
        }
    result["status"] = "complete"
    atomic_json(root / "data/processed/data-manifest.json", result)
    return result
