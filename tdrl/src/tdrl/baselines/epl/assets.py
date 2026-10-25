import json
from pathlib import Path

from ...provenance import file_hash


def directory_hashes(path):
    root = Path(path)
    if not root.is_dir():
        raise FileNotFoundError(root)
    return {
        file.relative_to(root).as_posix(): file_hash(file)
        for file in sorted(root.rglob("*"))
        if file.is_file() and ".cache" not in file.relative_to(root).parts
    }


def verify_assets(lock_path, policy_path, retrieval_path):
    lock = json.loads(Path(lock_path).read_text(encoding="utf-8"))
    required = {"schema_version", "status", "policy_encoder", "retrieval_encoder"}
    if set(lock) != required or lock["schema_version"] != 1:
        raise ValueError("unknown/incomplete EPL asset lock")
    if lock["status"] != "pinned_and_verified":
        raise ValueError("EPL model assets are not pinned")
    expected = {
        "policy_encoder": ("google-bert/bert-base-cased", Path(policy_path)),
        "retrieval_encoder": (
            "sentence-transformers/all-MiniLM-L6-v2",
            Path(retrieval_path),
        ),
    }
    for name, (model_id, path) in expected.items():
        record = lock[name]
        if (
            set(record) != {"model_id", "revision", "files"}
            or record["model_id"] != model_id
            or not isinstance(record["revision"], str)
            or len(record["revision"]) != 40
            or directory_hashes(path) != record["files"]
        ):
            raise ValueError(f"EPL {name} differs from its immutable lock")
    return lock
