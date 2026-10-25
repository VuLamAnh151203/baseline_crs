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
    required = {"schema_version", "status", "model_id", "revision", "encoder", "tokenizer"}
    if set(lock) != required or lock["schema_version"] != 1:
        raise ValueError("unknown/incomplete DPDP asset lock")
    if lock["status"] != "pinned_and_verified" or lock["model_id"] != "FacebookAI/roberta-large":
        raise ValueError("DPDP RoBERTa-large assets are not pinned and verified")
    if not isinstance(lock["revision"], str) or len(lock["revision"]) != 40:
        raise ValueError("DPDP encoder revision must be a full commit SHA")
    if directory_hashes(encoder_path) != lock["encoder"]:
        raise ValueError("DPDP encoder bytes differ from immutable lock")
    if directory_hashes(tokenizer_path) != lock["tokenizer"]:
        raise ValueError("DPDP tokenizer bytes differ from immutable lock")
    return lock
