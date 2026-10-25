"""MRN reporting must work without IQE and reject incompatible paired evidence."""

import json
import runpy
import sys
from pathlib import Path

import pytest

from tdrl.provenance import file_hash


def fixture_tree(tmp_path):
    output = tmp_path / "mrn"
    baseline = tmp_path / "topdial-debug-v2-58159" / "eval-0"
    current = output / "eval-0"
    ids = [f"s_{i}" for i in range(64)]
    provenance = {
        key: {"fixed": True} for key in ["models", "dataset", "seeds", "prompts", "preprocessing"]
    }
    provenance.update(checkpoint={"PPDPP": "fixed-hash"}, scenarios={"ids": ids})
    for directory, variants in [
        (baseline, ["PPDPP"]),
        (current, ["repair-MRNFixed-R", "repair-MRNFixed-Full"]),
    ]:
        directory.mkdir(parents=True)
        rows = []
        for variant in variants:
            for protocol in ["profile", "resistant"]:
                for i, scenario_id in enumerate(ids):
                    # Two discordant scenarios: one won, one lost by Full.
                    accepted = i == (1 if variant.endswith("-Full") else 0)
                    rows.append(
                        {
                            "variant": variant,
                            "protocol": protocol,
                            "record": {
                                "scenario_id": scenario_id,
                                "status": "complete",
                                "seed": 17,
                                "infrastructure_errors": 0,
                                "accepted": accepted,
                                "turns": 1 if accepted else 10,
                                "turn_budget": 10,
                                "h100_seconds": 1,
                                "protocol": protocol,
                                "judge_protocol_sha256": "judge-v2",
                                "acceptance_rule_version": 2,
                            },
                        }
                    )
        records = directory / "records.json"
        records.write_text(json.dumps(rows))
        (directory / "manifest.json").write_text(
            json.dumps(
                {
                    "status": "complete",
                    "provenance": provenance,
                    "outputs": {"records.json": file_hash(records)},
                }
            )
        )
    return output


def run_report(monkeypatch, output, flag="--mrn-only"):
    monkeypatch.setattr(sys, "argv", ["summarize", str(output), flag])
    runpy.run_path(str(Path(__file__).parents[1] / "scripts/summarize_topdial_debug.py"))


def test_mrn_report_without_iqe_and_with_paired_discordances(tmp_path, monkeypatch):
    output = fixture_tree(tmp_path)
    run_report(monkeypatch, output)
    report = json.loads((output / "report.json").read_text())
    assert report["dialogs"] == 384 and report["head_scope"] == "MRNFixed only"
    for result in report["paired_Full_vs_R"].values():
        assert result["won"] == result["lost"] == 1
        assert result["difference"] == 0 and result["resamples"] == 10000


@pytest.mark.parametrize("mutation", ["partial", "checkpoint", "checksum"])
def test_mrn_report_rejects_invalid_evidence(tmp_path, monkeypatch, mutation):
    output = fixture_tree(tmp_path)
    path = output / "eval-0/manifest.json"
    manifest = json.loads(path.read_text())
    if mutation == "partial":
        manifest["status"] = "partial"
    elif mutation == "checkpoint":
        manifest["provenance"]["checkpoint"]["PPDPP"] = "different"
    else:
        manifest["outputs"]["records.json"] = "changed"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        run_report(monkeypatch, output)
    assert not (output / "report.json").exists()


def test_mrn_report_pairs_retrieval_and_full_on_separate_gpu_shards(tmp_path, monkeypatch):
    output = fixture_tree(tmp_path)
    original = json.loads((output / "eval-0/records.json").read_text())
    manifest = json.loads((output / "eval-0/manifest.json").read_text())
    for slot, variant in [(0, "repair-MRNFixed-R"), (1, "repair-MRNFixed-Full")]:
        directory = output / f"eval-{slot}"
        directory.mkdir(exist_ok=True)
        records = directory / "records.json"
        records.write_text(json.dumps([x for x in original if x["variant"] == variant]))
        shard_manifest = {**manifest, "outputs": {"records.json": file_hash(records)}}
        (directory / "manifest.json").write_text(json.dumps(shard_manifest))
    run_report(monkeypatch, output, "--mrn-split")
    report = json.loads((output / "report.json").read_text())
    assert report["dialogs"] == 384
    assert "eval-1/records.json" in report["source_sha256"]
    assert report["paired_Full_vs_R"]["repair-MRNFixed-Full/resistant"]["lost"] == 1
