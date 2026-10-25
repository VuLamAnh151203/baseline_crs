"""Fail-closed loading of the complete, development-selected DiffTOD EMA policy."""

import json
import math
from pathlib import Path

import torch

from .diffusion_model import load_initial_policy
from .provenance import digest, file_hash, validate_manifest


def verify_selected_policy(source: Path):
    manifest_path = source / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    validate_manifest(manifest)
    if manifest["status"] != "complete" or manifest["stage"] != "DiffTOD-DuRecDial2.0-fitting":
        raise ValueError("DiffTOD baseline requires completed native-budget fitting")
    for name in ["selection.json", "checkpoint.json"]:
        if name not in manifest["outputs"] or file_hash(source / name) != manifest["outputs"][name]:
            raise ValueError("DiffTOD fitting output checksum mismatch")
    selection = json.loads((source / "selection.json").read_text())
    checkpoint = json.loads((source / "checkpoint.json").read_text())
    if selection.get("schema_version") != 1 or selection["status"] != "complete":
        raise ValueError("unknown/incomplete DiffTOD selection")
    if checkpoint.get("schema_version") != 1 or checkpoint["updates"] != 100001:
        raise ValueError("DiffTOD native fitting budget incomplete")
    candidates = selection["candidates"]
    if not candidates or any(not math.isfinite(c["validation_loss"]) for c in candidates):
        raise ValueError("invalid DiffTOD development selection metrics")
    expected = min(candidates, key=lambda c: (c["validation_loss"], c["update"]))
    chosen = selection["selected"]
    if chosen != expected or not 0 < chosen["update"] <= checkpoint["updates"]:
        raise ValueError("DiffTOD checkpoint selection rule mismatch")
    relative = Path(chosen["path"])
    if (
        not relative.parts
        or relative.is_absolute()
        or ".." in relative.parts
        or relative.parts[0] != "candidates"
    ):
        raise ValueError("unsafe DiffTOD selected checkpoint path")
    path = source / relative
    if (
        manifest["outputs"].get(str(relative)) != chosen["sha256"]
        or file_hash(path) != chosen["sha256"]
    ):
        raise ValueError("DiffTOD selected EMA checkpoint checksum mismatch")
    return path, chosen, manifest


def load_selected_policy(root: Path, source: Path):
    path, chosen, manifest = verify_selected_policy(source)
    model, initialization = load_initial_policy(root)
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if saved.get("schema_version") != 1 or saved["config"] != model.config:
        raise ValueError("DiffTOD selected architecture/version mismatch")
    if (
        saved["updates"] != chosen["update"]
        or saved["validation_loss"] != chosen["validation_loss"]
        or saved["fitting_config_sha256"] != manifest["provenance"]["config"]["sha256"]
    ):
        raise ValueError("DiffTOD selected checkpoint metadata mismatch")
    model.load_state_dict(saved["model"], strict=True)
    model.eval()
    model.requires_grad_(False)
    if hasattr(model, "initial_optimizer_weights"):
        del model.initial_optimizer_weights
    if any(p.dtype != torch.bfloat16 for p in model.parameters()):
        raise ValueError("DiffTOD selected policy must retain BF16")
    provenance = {
        **initialization,
        "status": "fitted_ema_selected_by_development_loss",
        "checkpoint_sha256": chosen["sha256"],
        "selected_update": chosen["update"],
        "completed_updates": 100001,
        "fitting_manifest_sha256": file_hash(source / "manifest.json"),
        "selection_sha256": file_hash(source / "selection.json"),
        "fitting_contract_sha256": digest(manifest["provenance"]["config"]),
    }
    return model, provenance
