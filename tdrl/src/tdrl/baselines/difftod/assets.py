import json
import re
from pathlib import Path

from ...provenance import file_hash


def verify_assets(lock_path: Path, checkpoint: Path, tokenizer: Path):
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("schema_version") != 1 or lock.get("status") != "pinned_and_verified":
        raise ValueError("DiffTOD assets must be pinned and byte-verified before use")
    if set(lock) != {"schema_version", "status", "sedd", "tokenizer"}:
        raise ValueError("unknown/incomplete DiffTOD asset lock")
    sedd, token = lock["sedd"], lock["tokenizer"]
    for item in (sedd, token):
        if not re.fullmatch(r"[0-9a-f]{40}", item.get("revision", "")):
            raise ValueError("DiffTOD asset revision must be an immutable commit")
    if file_hash(checkpoint) != sedd.get("checkpoint_sha256"):
        raise ValueError("DiffTOD SEDD checkpoint hash mismatch")
    expected = token.get("files")
    if not isinstance(expected, dict) or not expected:
        raise ValueError("DiffTOD tokenizer lock has no files")
    actual_names = {path.name for path in tokenizer.iterdir() if path.is_file()}
    if set(expected) != actual_names:
        raise ValueError("DiffTOD tokenizer file set differs from its lock")
    for name, sha256 in expected.items():
        if file_hash(tokenizer / name) != sha256:
            raise ValueError(f"DiffTOD tokenizer hash mismatch: {name}")
    return lock
