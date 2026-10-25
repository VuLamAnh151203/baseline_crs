"""Train/dev-only native-style DiffTOD dialogue text and fixed token packing."""

from pathlib import Path

import numpy as np

from .provenance import digest, file_hash
from .training import load_prepared

SERIALIZATION_VERSION = "difftod_topdial_profile_target_dialogue_v1"


def native_dialogue_text(scenario):
    target = scenario["target"]
    if len(target) != 2 or not all(isinstance(x, str) and x.strip() for x in target):
        raise ValueError("invalid diffusion task target")
    # Prepared JSON fixes profile insertion order. Native upstream uses Python
    # string rendering of each value and tabs between fields, without JSON quoting.
    profile = "user profile: " + "".join(f"{k}: {v}\t" for k, v in scenario["profile"].items())
    text = profile + f"target: {target[0]}, {target[1]}\n"
    for turn in scenario["messages"]:
        if turn["role"] not in {"user", "assistant"}:
            raise ValueError("unsupported native dialogue role")
        text += (
            ("system" if turn["role"] == "assistant" else "user") + ": " + turn["content"] + "\n"
        )
    return text


def pack_dialogues(rows, encode, eos_token, length=1024, group_size=1000):
    """Native concatenate/EOS/drop-remainder packing within frozen map batches.

    Keep train and development calls separate. Entire observed training dialogues
    are fitting input; no complete dialogue is ever used by the online planner.
    """
    if type(length) is not int or length < 1 or type(group_size) is not int or group_size < 1:
        raise ValueError("invalid diffusion packing budget")
    if type(eos_token) is not int or eos_token < 0:
        raise ValueError("invalid diffusion EOS")
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("duplicate diffusion training/development IDs")
    chunks, groups = [], []
    for offset in range(0, len(rows), group_size):
        group = rows[offset : offset + group_size]
        tokens = []
        for row in group:
            ids = list(encode(native_dialogue_text(row)))
            if not ids or any(type(t) is not int or t < 0 for t in ids):
                raise ValueError("invalid encoded diffusion dialogue")
            tokens.extend([*ids, eos_token])
        end = len(tokens) // length * length
        chunks.extend(tokens[i : i + length] for i in range(0, end, length))
        groups.append(
            {
                "source_ids": [r["id"] for r in group],
                "input_tokens": len(tokens),
                "chunks": end // length,
                "dropped_remainder_tokens": len(tokens) - end,
            }
        )
    if not chunks:
        raise ValueError("no complete diffusion training chunks")
    return np.asarray(chunks, dtype=np.int64), groups


def load_diffusion_split(root: Path, split, data_manifest):
    """Reject test access and mutated normalized data before tokenization."""
    if split not in {"train", "dev"}:
        raise ValueError("diffusion fitting/selection may only open train/dev")
    if data_manifest.get("schema_version") != 1:
        raise ValueError("unknown diffusion input manifest schema")
    path = root / "data/processed/TopDial" / (split + ".jsonl")
    expected = data_manifest["datasets"]["TopDial"]
    if file_hash(path) != expected["prepared_sha256"][split]:
        raise ValueError("diffusion input content hash mismatch")
    rows = load_prepared(root, "TopDial", split)
    if (
        len(rows) != expected["count"][split]
        or digest([r["id"] for r in rows]) != expected["ids_sha256"][split]
    ):
        raise ValueError("diffusion input membership mismatch")
    return rows
