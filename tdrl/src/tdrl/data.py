import json
from pathlib import Path

from .contracts import Message
from .provenance import atomic_json, digest, file_hash
from .selection import verify_selection
from .state import observed_prefix

DATASETS = {"TopDial", "DuRecDial2.0"}


def prepare_canonical(
    input_path: Path,
    output_dir: Path,
    dataset: str,
    split: str,
    source_manifest: dict,
    selection_path=None,
    selection_sha256=None,
):
    """Audit already adapted official JSONL; does not guess source-field semantics.

    Each record: id, messages[{role,content}], profile, knowledge, target.
    A source adapter must attest license, immutable input/revision and official membership.
    """
    if dataset not in DATASETS or split not in {"train", "dev", "test_seen", "test_unseen", "test"}:
        raise ValueError("unsupported dataset/split")
    if split.startswith("test"):
        if selection_path is None or not selection_sha256:
            raise ValueError("test input is locked until selection")
        verify_selection(selection_path, selection_sha256)
    required = {
        "schema_version",
        "source_url",
        "source_revision",
        "raw_sha256",
        "license",
        "split_membership_sha256",
        "adapter_sha256",
        "official_split",
        "transformations",
    }
    if set(source_manifest) != required or source_manifest["schema_version"] != 1:
        raise ValueError("source provenance/license/membership required")
    if not all(source_manifest[k] for k in required - {"transformations"}):
        raise ValueError("unresolved source metadata")
    if source_manifest["official_split"] != split:
        raise ValueError("split membership mismatch")
    if output_dir.exists():
        raise FileExistsError("prepared splits are immutable; use a new versioned directory")
    examples, states, ids, duplicate_keys = [], [], set(), set()
    with input_path.open() as f:
        for line in f:
            row = json.loads(line)
            if set(row) != {"id", "messages", "profile", "knowledge", "target"}:
                raise ValueError("unexpected canonical record fields")
            if not row["id"] or row["id"] in ids:
                raise ValueError("duplicate/empty scenario ID")
            ids.add(row["id"])
            messages = [Message(**x) for x in row["messages"]]
            observed_prefix(messages, len(messages))  # Role/type validation.
            key = digest(row["messages"])
            if key in duplicate_keys:
                raise ValueError("duplicate dialogue within split")
            duplicate_keys.add(key)
            examples.append(row)
            for t, message in enumerate(messages):
                if message.role == "assistant":
                    states.append(
                        {
                            "id": f"{row['id']}:{t}",
                            "dialogue_id": row["id"],
                            "turn": t,
                            "split": split,
                            "state": observed_prefix(messages, t).serialize(),
                            "guidance": message.content,
                        }
                    )
    if not examples:
        raise ValueError("empty split")
    membership = digest([r["id"] for r in examples])
    if source_manifest["split_membership_sha256"] != membership:
        raise ValueError("canonical membership differs from audited official split")
    output_dir.mkdir(parents=True)
    for filename, records in (("scenarios.jsonl", examples), ("states.jsonl", states)):
        (output_dir / filename).write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records)
        )
    manifest = {
        "schema_version": 1,
        "status": "partial",
        "dataset": dataset,
        "split": split,
        "input_sha256": file_hash(input_path),
        "source": source_manifest,
        "scenario_ids": [r["id"] for r in examples],
        "count": len(examples),
        "state_count": len(states),
        "cross_split_audit": "pending",
        "outputs": {
            name: file_hash(output_dir / name) for name in ("scenarios.jsonl", "states.jsonl")
        },
    }
    atomic_json(output_dir / "manifest.json", manifest)
    return manifest


def duplicate_audit(splits: dict[str, list[dict]], near_threshold=0.9):
    """Exact hashes plus exhaustive normalized token 3-gram Jaccard audit.

    For large corpora, replace the candidate enumerator with a validated exact-recall
    index; this reference is intentionally exhaustive rather than silently approximate.
    """
    import re

    rows = []
    for split, examples in splits.items():
        for row in examples:
            text = " ".join(m["content"] for m in row["messages"])
            tokens = re.findall(r"\w+", text.casefold())
            shingles = {tuple(tokens[i : i + 3]) for i in range(max(0, len(tokens) - 2))}
            rows.append((split, row["id"], digest(row["messages"]), shingles))
    collisions = []
    for i, a in enumerate(rows):
        for b in rows[i + 1 :]:
            if a[0] == b[0]:
                continue
            exact = a[2] == b[2]
            union = a[3] | b[3]
            similarity = len(a[3] & b[3]) / len(union) if union else 0.0
            if exact or similarity >= near_threshold:
                collisions.append(
                    {"left": a[:2], "right": b[:2], "exact": exact, "jaccard": similarity}
                )
    return collisions
