"""Immutable asset manifests for the released SCOPE semantic space."""

import json
from pathlib import Path

from ...provenance import atomic_json, file_hash

TRANSITION_FOLDER_ID = "1NLK8f8aV476frtIuMwC8IgwTVPxbOB6S"
LLAMA_GUARD_ID = "meta-llama/Meta-Llama-Guard-2-8B"
LLAMA_GUARD_REVISION = "7d257f3c1a0ec6ed99b2cb715027149dfb9784ef"
SEEDS = tuple(range(4))
DIRECTIONS = ("human_llm", "llm_human")


def transition_files(root):
    root = Path(root)
    return {
        f"seed_{seed}_batch_2048/{direction}/model_min_val.pth": (
            root / f"seed_{seed}_batch_2048" / direction / "model_min_val.pth"
        )
        for seed in SEEDS
        for direction in DIRECTIONS
    }


def build_transition_manifest(root):
    root = Path(root).resolve()
    files = transition_files(root)
    missing = [str(path) for path in files.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing official SCOPE transitions: " + ", ".join(missing))
    payload = {
        "schema_version": 1,
        "asset": "SCOPE deterministic transition ensemble",
        "source": {
            "repository": "https://github.com/chenzhiliang94/convo-plan-SCOPE.git",
            "revision": "b35dbc64808afce25e8f8e5d6838ecf271ec4d0b",
            "google_drive_folder_id": TRANSITION_FOLDER_ID,
        },
        "selection": "released model_min_val.pth",
        "seeds": list(SEEDS),
        "directions": list(DIRECTIONS),
        "files": {
            relative: {"bytes": path.stat().st_size, "sha256": file_hash(path)}
            for relative, path in sorted(files.items())
        },
    }
    if any(row["bytes"] < 500_000_000 for row in payload["files"].values()):
        raise ValueError("truncated SCOPE transition checkpoint")
    atomic_json(root / "manifest.json", payload)
    return payload


def verify_transition_manifest(root):
    root = Path(root).resolve()
    path = root / "manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected_keys = {
        "schema_version",
        "asset",
        "source",
        "selection",
        "seeds",
        "directions",
        "files",
    }
    if set(payload) != expected_keys or payload["schema_version"] != 1:
        raise ValueError("unknown SCOPE transition manifest")
    if payload["source"] != {
        "repository": "https://github.com/chenzhiliang94/convo-plan-SCOPE.git",
        "revision": "b35dbc64808afce25e8f8e5d6838ecf271ec4d0b",
        "google_drive_folder_id": TRANSITION_FOLDER_ID,
    }:
        raise ValueError("SCOPE transition source changed")
    if (
        payload["selection"] != "released model_min_val.pth"
        or payload["seeds"] != list(SEEDS)
        or payload["directions"] != list(DIRECTIONS)
    ):
        raise ValueError("SCOPE transition ensemble contract changed")
    expected = transition_files(root)
    if set(payload["files"]) != set(expected):
        raise ValueError("SCOPE transition manifest has missing/extra files")
    for relative, checkpoint in expected.items():
        recorded = payload["files"][relative]
        if (
            not checkpoint.is_file()
            or checkpoint.stat().st_size != recorded["bytes"]
            or file_hash(checkpoint) != recorded["sha256"]
        ):
            raise ValueError(f"SCOPE transition checkpoint mismatch: {checkpoint}")
    return {"path": str(root), "manifest": payload, "sha256": file_hash(path)}


def verify_llama_guard_snapshot(snapshot, manifest_path):
    snapshot, manifest_path = Path(snapshot).resolve(), Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    required = {"schema_version", "model_id", "revision", "files"}
    if set(manifest) != required or manifest["schema_version"] != 1:
        raise ValueError("unknown Llama Guard asset manifest")
    if (
        manifest["model_id"] != LLAMA_GUARD_ID
        or manifest["revision"] != LLAMA_GUARD_REVISION
    ):
        raise ValueError("wrong Llama Guard model or mutable revision")
    mandatory = {
        "config.json",
        "model.safetensors.index.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        *(f"model-{index:05d}-of-00004.safetensors" for index in range(1, 5)),
    }
    if set(manifest["files"]) != mandatory:
        raise ValueError("Llama Guard manifest has missing/extra runtime files")
    for relative, recorded in manifest["files"].items():
        path = snapshot / relative
        if (
            not path.is_file()
            or path.stat().st_size != recorded["bytes"]
            or file_hash(path) != recorded["sha256"]
        ):
            raise ValueError(f"Llama Guard file mismatch: {path}")
    return {
        "path": str(snapshot),
        "manifest_path": str(manifest_path),
        "manifest_sha256": file_hash(manifest_path),
        **manifest,
    }
