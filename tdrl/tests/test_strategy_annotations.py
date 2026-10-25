import json

import pytest

from tdrl.provenance import COMPUTE_FIELDS, PROVENANCE_FIELDS, atomic_json, file_hash
from tdrl.strategy_annotations import apply_strategy_overlay, strategy_payload


def record(node="training:1", split="train"):
    return {
        "node_id": node,
        "split": split,
        "guidance": "Would you like The Film?",
        "action": "ask_preference",
        "state": json.dumps({"history": [{"role": "user", "content": "A comedy please."}]}),
        "target": ["Movie recommendation", "The Film"],
        "terminal": False,
    }


def overlay(root, labels, status="complete"):
    atomic_json(root / "labels.json", labels)
    atomic_json(
        root / "manifest.json",
        {
            "schema_version": 1,
            "status": status,
            "run_id": "fixture",
            "stage": "TopDial-semantic-strategy-annotate",
            "started_at": "start",
            "ended_at": "end",
            "provenance": {
                **{k: {} for k in PROVENANCE_FIELDS},
                "dataset": {"bank_sha256": "bank"},
            },
            "compute": {k: 0 for k in COMPUTE_FIELDS},
            "outputs": {"labels.json": file_hash(root / "labels.json")},
        },
    )


def test_annotation_payload_has_observed_training_move_without_weak_label_or_future():
    source = record()
    payload = strategy_payload(source)
    assert set(payload) == {"target", "preceding_dialogue", "observed_assistant_move"}
    assert payload["preceding_dialogue"] == [{"role": "user", "content": "A comedy please."}]
    with pytest.raises(ValueError, match="training move"):
        strategy_payload(record(split="dev"))


def test_overlay_changes_only_training_guidance_action_and_retains_source(tmp_path):
    source = [
        record(),
        record("training:2"),
        {**record("terminal"), "guidance": "", "action": None, "terminal": True},
    ]
    overlay(
        tmp_path,
        [
            {
                "node_id": "training:1",
                "weak_label": "ask_preference",
                "semantic_label": "recommend",
            },
            {
                "node_id": "training:2",
                "weak_label": "ask_preference",
                "semantic_label": "ambiguous",
            },
        ],
    )
    changed = apply_strategy_overlay(source, tmp_path, "bank")
    assert changed[0]["action"] == "recommend"
    assert changed[0]["state"] == source[0]["state"]
    assert changed[1]["guidance"] == "" and changed[1]["action"] is None
    assert changed[2] == source[2]
    assert source[0]["action"] == "ask_preference" and source[1]["guidance"]


def test_overlay_rejects_partial_changed_bank_labels_and_checksum(tmp_path):
    labels = [
        {"node_id": "training:1", "weak_label": "ask_preference", "semantic_label": "recommend"}
    ]
    overlay(tmp_path, labels, status="partial")
    with pytest.raises(ValueError, match="complete semantic"):
        apply_strategy_overlay([record()], tmp_path, "bank")
    overlay(tmp_path, labels)
    with pytest.raises(ValueError, match="bank mismatch"):
        apply_strategy_overlay([record()], tmp_path, "other")
    with pytest.raises(ValueError, match="partial semantic"):
        apply_strategy_overlay([record(), record("training:2")], tmp_path, "bank")
    overlay(tmp_path, labels + labels)
    with pytest.raises(ValueError, match="duplicate"):
        apply_strategy_overlay([record()], tmp_path, "bank")
    overlay(tmp_path, labels)
    (tmp_path / "labels.json").write_text("[]")
    with pytest.raises(ValueError, match="checksum"):
        apply_strategy_overlay([record()], tmp_path, "bank")


def test_closing_guidance_exclusion_is_opt_in_and_preserves_geometry(tmp_path):
    source = [record()]
    overlay(
        tmp_path,
        [{"node_id": "training:1", "weak_label": "ask_preference", "semantic_label": "farewell"}],
    )
    default = apply_strategy_overlay(source, tmp_path, "bank")
    excluded = apply_strategy_overlay(source, tmp_path, "bank", exclude_farewell=True)
    assert default[0]["action"] == "farewell" and default[0]["guidance"] == source[0]["guidance"]
    assert excluded[0]["action"] is None and excluded[0]["guidance"] == ""
    for field in ["node_id", "state", "target", "terminal"]:
        assert excluded[0][field] == source[0][field]
    assert source[0]["action"] == "ask_preference"
    with pytest.raises(ValueError, match="explicitly boolean"):
        apply_strategy_overlay(source, tmp_path, "bank", exclude_farewell="true")


def test_non_target_recommendations_exclusion_preserves_intermediate_states(tmp_path):
    source = [
        {**record("train:target"), "goal_reference": True},
        {**record("train:other"), "goal_reference": False},
        {**record("train:question"), "goal_reference": False},
    ]
    overlay(
        tmp_path,
        [
            {"node_id": r["node_id"], "weak_label": "ask_preference", "semantic_label": label}
            for r, label in zip(source, ["recommend", "recommend", "ask_preference"], strict=True)
        ],
    )
    old = apply_strategy_overlay(source, tmp_path, "bank")
    changed = apply_strategy_overlay(
        source, tmp_path, "bank", exclude_non_target_recommendations=True
    )
    assert old[1]["guidance"] == source[1]["guidance"]
    assert changed[0] == old[0] and changed[2] == old[2]
    assert changed[1]["guidance"] == "" and changed[1]["action"] is None
    for before, after in zip(source, changed, strict=True):
        for key in before.keys() - {"action", "guidance"}:
            assert before[key] == after[key]
    with pytest.raises(ValueError, match="explicitly boolean"):
        apply_strategy_overlay(source, tmp_path, "bank", exclude_non_target_recommendations="true")
