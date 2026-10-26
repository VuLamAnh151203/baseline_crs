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
        if file.is_file()
    }


def verify_assets(lock_path, encoder_path, tokenizer_path):
    lock = json.loads(Path(lock_path).read_text(encoding="utf-8"))
    if set(lock) != {"schema_version", "status", "model_id", "revision", "encoder", "tokenizer"}:
        raise ValueError("unknown/incomplete TRIP asset lock")
    if lock["schema_version"] != 1 or lock["status"] != "pinned_and_verified":
        raise ValueError("TRIP assets are not pinned and verified")
    actual_encoder = directory_hashes(encoder_path)
    actual_tokenizer = directory_hashes(tokenizer_path)
    if actual_encoder != lock["encoder"] or actual_tokenizer != lock["tokenizer"]:
        raise ValueError("TRIP asset bytes differ from the immutable lock")
    revision = lock["revision"]
    if not isinstance(revision, str) or len(revision) != 40:
        raise ValueError("TRIP encoder revision must be a full commit SHA")
    return lock
