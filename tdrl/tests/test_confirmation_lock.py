import json

import pytest

from tdrl.confirmation_lock import freeze_candidate, verify_candidate
from tdrl.provenance import file_hash


def candidate(root):
    source = root / "frozen.json"
    source.write_text("{}")
    return {
        "schema_version": 1,
        "scope": "TopDial disjoint development confirmation",
        "root_seed": 17,
        "methods": ["TDRL-R", "TDRL", "DiffTOD"],
        "discovery_macro_differences": {"TDRL-R": 0.1, "TDRL": 0.2},
        "thresholds": {"H_edge": 1, "H_cover": 2},
        "head_sha256": "a" * 64,
        "head_type": "MRNFixed",
        "code_commit": "c" * 40,
        "package_lock_sha256": "d" * 64,
        "difftod_policy_sha256": "b" * 64,
        "files": {
            name: {"path": "frozen.json", "sha256": file_hash(source)}
            for name in [
                "candidate_config",
                "comparison_config",
                "confirmation_membership",
                "strategy_overlay_manifest",
                "discovery_report",
            ]
        },
    }


def test_confirmation_contract_is_immutable_and_rejects_changed_evidence(tmp_path):
    payload = candidate(tmp_path)
    lock = tmp_path / "confirmation-candidate.lock"
    freeze_candidate(lock, payload)
    assert verify_candidate(lock, tmp_path) == payload
    with pytest.raises(FileExistsError):
        freeze_candidate(lock, payload)
    (tmp_path / "frozen.json").write_text('{"modified":true}')
    with pytest.raises(ValueError, match="artifact changed"):
        verify_candidate(lock, tmp_path)


def test_confirmation_lock_cannot_authorize_other_scope_or_weaker_candidate(tmp_path):
    payload = candidate(tmp_path)
    payload["scope"] = "official test"
    with pytest.raises(ValueError, match="scope/seed"):
        freeze_candidate(tmp_path / "lock", payload)
    payload = candidate(tmp_path)
    payload["discovery_macro_differences"]["TDRL-R"] = 0
    with pytest.raises(ValueError, match="positive discovery"):
        freeze_candidate(tmp_path / "lock", payload)


def test_confirmation_lock_rejects_checksum_tamper(tmp_path):
    payload = candidate(tmp_path)
    lock = tmp_path / "lock"
    freeze_candidate(lock, payload)
    value = json.loads(lock.read_text())
    value["payload"]["thresholds"]["H_cover"] = 100
    lock.chmod(0o644)
    lock.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="checksum mismatch"):
        verify_candidate(lock, tmp_path)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), True])
def test_lock_rejects_nonfinite_or_untyped_frozen_values(tmp_path, value):
    payload = candidate(tmp_path)
    payload["thresholds"]["H_cover"] = value
    with pytest.raises(ValueError, match="thresholds"):
        freeze_candidate(tmp_path / "bad-threshold", payload)
    payload = candidate(tmp_path)
    payload["discovery_macro_differences"]["TDRL"] = value
    with pytest.raises(ValueError, match="positive discovery"):
        freeze_candidate(tmp_path / "bad-difference", payload)
