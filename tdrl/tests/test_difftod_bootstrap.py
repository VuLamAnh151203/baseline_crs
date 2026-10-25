import importlib.util
import json
from pathlib import Path

import pytest

from tdrl.provenance import file_hash

spec = importlib.util.spec_from_file_location(
    "difftod_bootstrap", Path(__file__).parents[1] / "scripts/bootstrap_difftod_models.py"
)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def local_fixture(tmp_path, monkeypatch):
    root = tmp_path / "cache"
    lock = {"schema_version": 1}
    for key, name in [("policy", "louaaron/sedd-medium"), ("tokenizer", "openai-community/gpt2")]:
        directory = root / name.split("/")[-1] / ("a" * 40)
        directory.mkdir(parents=True)
        path = directory / "fixture.bin"
        path.write_bytes(b"complete-checksummed-fixture")
        lock[key] = {"model_id": name, "revision": "a" * 40, "files": {path.name: file_hash(path)}}
    path = tmp_path / "lock.json"
    path.write_text(json.dumps(lock))

    def no_network(*args, **kwargs):
        raise AssertionError("bootstrap fixture must never download or execute inference")

    monkeypatch.setattr(runner.subprocess, "run", no_network)
    return path, root


def test_verified_existing_bootstrap_reuses_without_network_or_inference(tmp_path, monkeypatch):
    lock, root = local_fixture(tmp_path, monkeypatch)
    runner.bootstrap(lock, root)
    runner.bootstrap(lock, root)
    status = json.loads((root / "bootstrap-status.json").read_text())
    assert status["status"] == "complete" and status["inference_performed"] is False
    assert len(status["verified_files"]) == 2


def test_verified_complete_partial_promotes_without_resampling_or_download(tmp_path, monkeypatch):
    lock, root = local_fixture(tmp_path, monkeypatch)
    path = root / "sedd-medium" / ("a" * 40) / "fixture.bin"
    partial = path.with_name("fixture.bin.partial")
    path.rename(partial)
    runner.bootstrap(lock, root)
    assert path.exists() and not partial.exists()


def test_corrupt_cache_is_retained_and_marked_partial(tmp_path, monkeypatch):
    lock, root = local_fixture(tmp_path, monkeypatch)
    path = root / "sedd-medium" / ("a" * 40) / "fixture.bin"
    path.write_bytes(b"mutated")
    with pytest.raises(ValueError, match="existing bootstrap checksum"):
        runner.bootstrap(lock, root)
    assert path.read_bytes() == b"mutated"
    assert json.loads((root / "bootstrap-status.json").read_text())["status"] == "partial"


def test_unknown_bootstrap_manifest_is_rejected_before_cache_reuse(tmp_path, monkeypatch):
    lock, root = local_fixture(tmp_path, monkeypatch)
    runner.bootstrap(lock, root)
    path = root / "bootstrap-status.json"
    status = json.loads(path.read_text())
    status["schema_version"] = 99
    path.write_text(json.dumps(status))
    with pytest.raises(ValueError, match="provenance mismatch"):
        runner.bootstrap(lock, root)
