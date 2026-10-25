import pytest

from tdrl.difftod_evaluation import verify_selected_policy
from tdrl.provenance import COMPUTE_FIELDS, PROVENANCE_FIELDS, atomic_json, file_hash


def completed_fit(root, *, updates=100001, selected_loss=2.0):
    candidate = root / "candidates/ema-000100.pt"
    candidate.parent.mkdir()
    candidate.write_bytes(b"control-flow fixture, not model weights")
    chosen = {
        "update": 100,
        "validation_loss": selected_loss,
        "path": "candidates/ema-000100.pt",
        "sha256": file_hash(candidate),
    }
    atomic_json(
        root / "selection.json",
        {"schema_version": 1, "status": "complete", "selected": chosen, "candidates": [chosen]},
    )
    atomic_json(root / "checkpoint.json", {"schema_version": 1, "updates": updates})
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "run_id": "fixture",
        "stage": "DiffTOD-TopDial-fitting",
        "started_at": "start",
        "ended_at": "end",
        "provenance": {k: {} for k in PROVENANCE_FIELDS},
        "compute": {k: 0 for k in COMPUTE_FIELDS},
        "outputs": {
            name: file_hash(root / name)
            for name in ["selection.json", "checkpoint.json", chosen["path"]]
        },
    }
    atomic_json(root / "manifest.json", manifest)
    return chosen, manifest


def test_partial_or_unknown_fitting_cannot_be_used_as_diffusion_baseline(tmp_path):
    atomic_json(tmp_path / "manifest.json", {"schema_version": 99, "status": "complete"})
    with pytest.raises(ValueError, match="unknown run manifest"):
        verify_selected_policy(tmp_path)


def test_no_selected_baseline_without_a_manifest(tmp_path):
    with pytest.raises(FileNotFoundError):
        verify_selected_policy(tmp_path)


def test_complete_budget_and_verified_selection_are_required(tmp_path):
    chosen, _ = completed_fit(tmp_path)
    path, observed, _ = verify_selected_policy(tmp_path)
    assert path == tmp_path / chosen["path"]
    assert observed == chosen
    path.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="checksum mismatch"):
        verify_selected_policy(tmp_path)


def test_partial_budget_rejected_even_with_complete_manifest(tmp_path):
    completed_fit(tmp_path, updates=100000)
    with pytest.raises(ValueError, match="budget incomplete"):
        verify_selected_policy(tmp_path)


def test_selection_cannot_ignore_better_development_checkpoint(tmp_path):
    chosen, manifest = completed_fit(tmp_path)
    better = {**chosen, "update": 200, "validation_loss": 1.0}
    atomic_json(
        tmp_path / "selection.json",
        {
            "schema_version": 1,
            "status": "complete",
            "selected": chosen,
            "candidates": [chosen, better],
        },
    )
    manifest["outputs"]["selection.json"] = file_hash(tmp_path / "selection.json")
    atomic_json(tmp_path / "manifest.json", manifest)
    with pytest.raises(ValueError, match="selection rule mismatch"):
        verify_selected_policy(tmp_path)
