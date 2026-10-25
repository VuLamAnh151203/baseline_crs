import importlib.util
from pathlib import Path

import pytest

from tdrl.provenance import COMPUTE_FIELDS, PROVENANCE_FIELDS, atomic_json, file_hash

spec = importlib.util.spec_from_file_location(
    "difftod_development",
    Path(__file__).resolve().parents[1] / "scripts/evaluate_difftod_development.py",
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def smoke(root, uuid, contract):
    root.mkdir()
    atomic_json(
        root / "checks.json",
        {
            "status": "passed",
            "contract": contract,
            "gpu": {"uuid": uuid},
            "required_seconds_with_safety": 100,
        },
    )
    atomic_json(
        root / "manifest.json",
        {
            "schema_version": 1,
            "status": "complete",
            "stage": "DiffTOD-development-smoke",
            "run_id": uuid,
            "started_at": "start",
            "ended_at": "end",
            "provenance": {k: {} for k in PROVENANCE_FIELDS},
            "compute": {k: 0 for k in COMPUTE_FIELDS},
            "outputs": {"checks.json": file_hash(root / "checks.json")},
        },
    )
    return root


def test_gates_require_both_gpu_runtime_same_contract_and_current_budget(tmp_path):
    contract = {"fitting_manifest_sha256": "selected", "config_sha256": "frozen"}
    paths = [smoke(tmp_path / str(i), str(i), contract) for i in range(2)]
    assert len(module.verify_gates(paths, contract, 101)) == 2
    with pytest.raises(ValueError, match="allocation time"):
        module.verify_gates(paths, contract, 99)
    with pytest.raises(ValueError, match="gate mismatch"):
        module.verify_gates(paths, {**contract, "config_sha256": "changed"}, 101)
    with pytest.raises(ValueError, match="distinct H100"):
        module.verify_gates([paths[0], paths[0]], contract, 101)
    with pytest.raises(ValueError, match="two independent"):
        module.verify_gates(paths[:1], contract, 101)
    (paths[0] / "checks.json").write_text("{}")
    with pytest.raises(ValueError, match="checksum mismatch"):
        module.verify_gates(paths, contract, 101)
