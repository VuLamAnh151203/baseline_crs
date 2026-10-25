import pytest

from tdrl.test_prefix import amended_resource, dialogue_contract, prefix_membership, seal_sample


def test_extension_keeps_exact_original_order_and_dialogue_namespace():
    first = prefix_membership(200)
    full = prefix_membership(1000)
    assert full["prefix"][:200] == first["prefix"]
    assert first["full_order_sha256"] == full["full_order_sha256"]
    assert {d for d, _ in first["prefix"]} == {"TopDial"}
    assert [d for d, _ in full["prefix"]].count("TopDial") == 500
    assert [d for d, _ in full["prefix"]].count("DuRecDial2.0") == 500
    d, sid = first["prefix"][0]
    contract = dialogue_contract("a" * 64, "TDRL", d, sid)
    assert "requested_prefix_count" not in contract
    assert contract != dialogue_contract("b" * 64, "TDRL", d, sid)
    assert contract != dialogue_contract("a" * 64, "DiffTOD", d, sid)


@pytest.mark.parametrize("count", [0, 1001, True, 200.0])
def test_invalid_counts_fail_closed(count):
    with pytest.raises(ValueError, match="prefix count"):
        prefix_membership(count)


def test_initial_400_is_exactly_200_per_dataset_and_original_is_preserved():
    resource = amended_resource()
    prefix = prefix_membership(400)["prefix"]
    assert [d for d, _ in prefix].count("TopDial") == 200
    assert [d for d, _ in prefix].count("DuRecDial2.0") == 200
    assert len(set(resource["added_ids"]["DuRecDial2.0"])) == 330
    assert len(resource["known_content_sha256"]["DuRecDial2.0"]) == 170
    assert len(resource["known_content_sha256"]["TopDial"]) == 330
    assert resource["original_resource_sha256"] == (
        "a56f7a0c909bd1ac460f41904da12f99582e98d4ef0b602f2593f7ab21870cb0"
    )


def test_content_seal_rejects_tail_changes_across_prefix_extension(tmp_path):
    path = tmp_path / "sealed.json"
    content = {"content_sha256": {"extra-tail": "before"}, "full_count": 1000}
    seal_sample(path, content)
    seal_sample(path, content)
    with pytest.raises(ValueError, match="content/provenance changed"):
        seal_sample(path, {**content, "content_sha256": {"extra-tail": "after"}})
    assert not list(tmp_path.glob("*.part"))


def test_official_prefix_report_checks_exact_ids_lock_and_pairing():
    from test_paired_development import matrix

    from tdrl.test_prefix import compare_prefix_records

    left, right = matrix(True), matrix(False)
    for row in left + right:
        row["selection_sha256"] = "official-lock"
    report = compare_prefix_records(left, right, ["0", "1"], "official-lock")
    assert report["paired_macro_protocol_sr"]["difference"] == 1
    assert report["paired_macro_protocol_sr"]["resamples"] == 10000
    with pytest.raises(ValueError, match="lock mismatch"):
        compare_prefix_records(left, right, ["0", "1"], "different-lock")
    with pytest.raises(ValueError, match="duplicate/unexpected"):
        compare_prefix_records(left, right, ["0", "other"], "official-lock")
