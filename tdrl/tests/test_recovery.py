import json
import subprocess

import pytest

from tdrl.backend import LocalBackend
from tdrl.provenance import digest
from tdrl.recovery import verified_file, verify_source_changes


def test_recovery_payload_change_fails_before_marking_request(tmp_path):
    from dataclasses import replace

    from test_core import PIN

    b = LocalBackend(
        "http://127.0.0.1:18001/v1",
        replace(PIN, model_id="Qwen/Qwen3-Embedding-8B"),
        "embedding",
        ledger=tmp_path / "embedding",
    )
    b.authorized_reissue = {"request_id": "approved", "request_sha256": digest({"text": "old"})}
    with pytest.raises(ValueError, match="payload mismatch"):
        b._begin("approved", {"text": "changed"})
    assert not list(tmp_path.rglob("pending-*.json"))
    b._begin("approved", {"text": "old"})
    assert (
        json.loads((tmp_path / "authorized-reissue.json").read_text())["request_id"] == "approved"
    )
    with pytest.raises(ValueError, match="unresolved"):
        b._begin("approved", {"text": "old"})


def test_recovery_rejects_changed_evidence(tmp_path):
    from tdrl.provenance import file_hash

    p = tmp_path / "checkpoint"
    p.write_bytes(b"original")
    sha = file_hash(p)
    verified_file(p, sha)
    p.write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum mismatch"):
        verified_file(p, sha)


def test_recovery_rejects_unreviewed_scientific_source(tmp_path, monkeypatch):
    def output(args, **kwargs):
        assert args[:3] == ["git", "diff", "--name-only"]
        return "tdrl/src/tdrl/evaluation.py\n"

    monkeypatch.setattr(subprocess, "check_output", output)
    with pytest.raises(ValueError, match="unreviewed scientific"):
        verify_source_changes(tmp_path)


def test_strict_backward_restores_flags_after_exception():
    import torch

    from tdrl.policy_training import deterministic_backward

    previous = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(False, warn_only=True)
        with pytest.raises(RuntimeError, match="fixture"):
            with deterministic_backward(True):
                assert torch.are_deterministic_algorithms_enabled()
                assert not torch.is_deterministic_algorithms_warn_only_enabled()
                raise RuntimeError("fixture")
        assert not torch.are_deterministic_algorithms_enabled()
        assert torch.is_deterministic_algorithms_warn_only_enabled()
    finally:
        torch.use_deterministic_algorithms(previous, warn_only=warn_only)


def test_policy_backward_scopes_recovery_only():
    from types import SimpleNamespace

    import torch

    from tdrl.policy_training import policy_backward

    observed = []
    loss = SimpleNamespace(
        backward=lambda: observed.append(torch.are_deterministic_algorithms_enabled())
    )
    previous = torch.are_deterministic_algorithms_enabled()
    try:
        torch.use_deterministic_algorithms(False)
        policy_backward(SimpleNamespace(), loss)
        policy_backward(SimpleNamespace(_recovery_execution={"deterministic_backward": True}), loss)
        assert observed == [False, True]
        assert not torch.are_deterministic_algorithms_enabled()
    finally:
        torch.use_deterministic_algorithms(previous)


def test_gradient_diagnostic_identifies_parameter_and_magnitude():
    import torch

    from tdrl.recovery import gradient_differences, gradient_fingerprint

    left = {"weight": torch.tensor([1, 2], dtype=torch.bfloat16)}
    right = {"weight": torch.tensor([1, 3], dtype=torch.bfloat16)}
    assert gradient_fingerprint(left) != gradient_fingerprint(right)
    assert gradient_differences(left, right) == [
        {
            "parameter": "weight",
            "changed_elements": 1,
            "elements": 2,
            "max_abs": 1.0,
        }
    ]
    with pytest.raises(ValueError, match="membership"):
        gradient_differences(left, {})


def test_failed_replay_retains_diagnostics_and_restores_rng(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import torch

    from tdrl.recovery import replay_check

    monkeypatch.setattr("tdrl.backend.require_h100_slurm", lambda: None)
    monkeypatch.setenv("SLURM_JOB_ID", "fixture")
    monkeypatch.setenv("TDRL_POLICY_DETERMINISM", "strict-backward-v1")
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    monkeypatch.setattr(torch.cuda, "set_rng_state_all", lambda states: None)
    output = tmp_path / "episodes/000243"
    output.mkdir(parents=True)
    (output / "dialogue.json").write_text(json.dumps({"turns_trace": [{}, {}, {}, {}]}))
    state = torch.get_rng_state()
    rng_path = output / "pre-episode-rng.pt"
    torch.save({"cpu_rng": state, "cuda_rng": [], "episode": 243}, rng_path)

    def pair(*args):
        torch.rand(4)
        strict = args[-1]
        return {
            "exact": not strict,
            "log_probabilities": [[1], [1]],
            "forward_rng": [{}, {}],
            "gradient_sha256": ["first", "second"],
        }

    monkeypatch.setattr("tdrl.recovery.replay_pair", pair)
    policy = SimpleNamespace(
        policy=SimpleNamespace(config=SimpleNamespace(_attn_implementation="fixture")),
        zero_grad=lambda **kwargs: None,
    )
    with pytest.raises(ValueError, match="not exact"):
        replay_check(policy, None, None, None, output, rng_path)
    report = json.loads((tmp_path / "h100-replay-audit.json").read_text())
    assert report["status"] == "failed"
    assert report["optimizer_updates_during_audit"] == 0
    assert report["default_backward"]["exact"]
    assert not report["strict_backward"]["exact"]
    assert torch.equal(torch.get_rng_state(), state)


def test_controller_determinism_settings_do_not_reach_inference_workers():
    from tdrl.runtime import inference_environment

    controller = {
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "TDRL_POLICY_DETERMINISM": "strict-backward-v1",
        "VLLM_BATCH_INVARIANT": "1",
    }
    assert inference_environment(controller) == {"VLLM_BATCH_INVARIANT": "1"}
    assert "CUBLAS_WORKSPACE_CONFIG" in controller
    other = {"CUBLAS_WORKSPACE_CONFIG": ":16:8"}
    assert inference_environment(other) == other
