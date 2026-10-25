"""Selection for user-requested prefix evaluation; no outcome-based candidate changes."""

import json
from pathlib import Path

from .confirmation_lock import verify_candidate
from .provenance import digest, file_hash


def verify_prefix_selection(root: Path, path: Path, commit: str):
    envelope = json.loads(path.read_text())
    payload = envelope["payload"]
    if set(envelope) != {"payload", "sha256"} or digest(payload) != envelope["sha256"]:
        raise ValueError("prefix selection checksum mismatch")
    if payload.get("schema_version") != 1 or payload.get("scope") != "fixed1000-prefix-test":
        raise ValueError("unknown prefix selection schema")
    if payload["code_commit"] != commit:
        raise ValueError("prefix test code changed after selection")
    for ref in payload["files"].values():
        rel = Path(ref["path"])
        if rel.is_absolute() or ".." in rel.parts or file_hash(root / rel) != ref["sha256"]:
            raise ValueError("prefix selection input changed")
    candidate = verify_candidate(root / payload["files"]["candidate_lock"]["path"], root)
    if payload["candidate_payload_sha256"] != digest(candidate):
        raise ValueError("frozen planner candidate changed")
    return payload, candidate, envelope["sha256"]
