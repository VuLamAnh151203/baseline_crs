from __future__ import annotations

import json
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[2]
METHODS = {
    "direct_llm_v2",
    "tepl_v2",
    "enpl_domain3",
    "procot",
    "dpdp",
    "scope_crs_lite",
    "epl_native_qwen",
}


def test_public_registry_and_method_docs_are_complete():
    registry = json.loads((REPOSITORY / "configs" / "baseline_registry.json").read_text())
    assert registry["schema_version"] == 1
    assert set(registry["methods"]) == METHODS
    for method in METHODS | {"difftod", "trip"}:
        readme = REPOSITORY / "baselines" / method / "README.md"
        assert readme.is_file()
        assert readme.read_text().startswith("# ")


def test_portable_launchers_do_not_embed_cluster_user_paths():
    for relative in (
        "scripts/submit_baseline.sh",
        "scripts/slurm/baseline_e2e.sbatch",
        ".env.example",
    ):
        text = (REPOSITORY / relative).read_text()
        assert "/home/u26anhvl" not in text
        assert "/mnt/data/u26anhvl" not in text
        assert "/mnt/data/duclm3" not in text


def test_fixed_400_resource_and_runtime_lock_are_packaged():
    assert (
        REPOSITORY
        / "tdrl"
        / "src"
        / "tdrl"
        / "resources"
        / "test_scenarios_1000_td500_dr500_v1.json"
    ).is_file()
    assert (REPOSITORY / "tdrl" / "runtime" / "uv.lock").is_file()
