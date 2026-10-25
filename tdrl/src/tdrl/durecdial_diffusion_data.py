"""DuRecDial train/dev adapter; native text/packing code remains shared."""

from pathlib import Path

from .diffusion_data import native_dialogue_text as native_dialogue_text
from .diffusion_data import pack_dialogues as pack_dialogues
from .provenance import digest, file_hash
from .training import load_prepared

SERIALIZATION_VERSION = "difftod_durecdial_profile_target_dialogue_v1"


def load_diffusion_split(root: Path, split, data_manifest):
    """Reject test access and mutated normalized data before tokenization."""
    if split not in {"train", "dev"}:
        raise ValueError("diffusion fitting/selection may only open train/dev")
    if data_manifest.get("schema_version") != 1:
        raise ValueError("unknown diffusion input manifest schema")
    path = root / "data/processed/DuRecDial2.0" / (split + ".jsonl")
    expected = data_manifest["datasets"]["DuRecDial2.0"]
    if file_hash(path) != expected["prepared_sha256"][split]:
        raise ValueError("diffusion input content hash mismatch")
    rows = load_prepared(root, "DuRecDial2.0", split)
    if (
        len(rows) != expected["count"][split]
        or digest([r["id"] for r in rows]) != expected["ids_sha256"][split]
    ):
        raise ValueError("diffusion input membership mismatch")
    return rows
