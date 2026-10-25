import json

import pytest

from tdrl.diffusion_data import load_diffusion_split, native_dialogue_text, pack_dialogues
from tdrl.provenance import digest, file_hash


def scenario(idx="train1"):
    return {
        "id": idx,
        "profile": {"Name": "Alex"},
        "target": ["Ask a question", "Jazz"],
        "messages": [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Do you like jazz?"},
        ],
        "future_annotation": "must not enter the text",
        "success": True,
    }


def test_native_format_preserves_non_recommendation_tasks_without_outcome_labels():
    assert native_dialogue_text(scenario()) == (
        "user profile: Name: Alex\ttarget: Ask a question, Jazz\n"
        "user: Hi\nsystem: Do you like jazz?\n"
    )


def test_packing_keeps_eos_and_drops_each_native_map_batch_remainder():
    chunks, groups = pack_dialogues(
        [scenario(str(i)) for i in range(3)], lambda text: [1, 2, 3], 9, length=3, group_size=2
    )
    assert chunks.tolist() == [[1, 2, 3], [9, 1, 2], [1, 2, 3]]
    assert [g["dropped_remainder_tokens"] for g in groups] == [2, 1]
    assert groups[1]["source_ids"] == ["2"]
    with pytest.raises(ValueError, match="duplicate"):
        pack_dialogues([scenario(), scenario()], lambda text: [1, 2, 3], 9, length=3)


def test_fitting_loader_rejects_test_before_opening_any_resource(tmp_path):
    with pytest.raises(ValueError, match="only open train/dev"):
        load_diffusion_split(tmp_path, "test", {})


def test_fitting_loader_requires_normalized_content_and_ordered_membership(tmp_path):
    path = tmp_path / "data/processed/TopDial/train.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(scenario()) + "\n")
    manifest = {
        "schema_version": 1,
        "datasets": {
            "TopDial": {
                "prepared_sha256": {"train": file_hash(path)},
                "count": {"train": 1},
                "ids_sha256": {"train": digest(["train1"])},
            }
        },
    }
    assert load_diffusion_split(tmp_path, "train", manifest) == [scenario()]
    manifest["datasets"]["TopDial"]["ids_sha256"]["train"] = digest(["other"])
    with pytest.raises(ValueError, match="membership"):
        load_diffusion_split(tmp_path, "train", manifest)
    path.write_text(json.dumps(scenario("changed")) + "\n")
    with pytest.raises(ValueError, match="content hash"):
        load_diffusion_split(tmp_path, "train", manifest)
