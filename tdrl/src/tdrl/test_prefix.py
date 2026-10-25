"""Replay the user-amended fixed 1000; no runtime random sampling."""

import hashlib
import json
import os
from importlib.resources import files

from .provenance import canonical, digest
from .test_sampling import _fixed_subset, sample_test_scenarios

RESOURCE = "resources/test_scenarios_1000_td500_dr500_v1.json"
RESOURCE_SHA256 = "c0bb70f77d305616a54cfc546bb75a15ad6651c42b7ef8fa7ae98508a103af97"


def amended_resource():
    # Git may check out JSON with CRLF on Windows; frozen hashes use LF bytes.
    raw = files("tdrl").joinpath(RESOURCE).read_bytes().replace(b"\r\n", b"\n")
    if hashlib.sha256(raw).hexdigest() != RESOURCE_SHA256:
        raise ValueError("amended fixed test resource checksum mismatch")
    resource = json.loads(raw)
    order = resource["execution_order"]
    if (
        resource["schema_version"] != 1
        or len(order) != 1000
        or len({tuple(x) for x in order}) != 1000
        or digest(order) != resource["execution_order_sha256"]
    ):
        raise ValueError("amended fixed test membership/order mismatch")
    return resource


def prefix_membership(count):
    if type(count) is not int or not 1 <= count <= 1000:
        raise ValueError("prefix count must be an integer in [1,1000]")
    resource = amended_resource()
    order = resource["execution_order"]
    return {
        "schema_version": 1,
        "resource_sha256": RESOURCE_SHA256,
        "execution_order": "first_200_TopDial_then_200_DuRecDial_then_300_TopDial_then_300_DuRecDial",
        "full_order_sha256": digest(order),
        "total_frozen_scenarios": 1000,
        "requested_prefix_count": count,
        "prefix": order[:count],
        "prefix_sha256": digest(order[:count]),
        "claim": "prefix_of_amended_stratified_sampled_official_test_subset",
    }


def prefix_cases(populations, count):
    original, _ = _fixed_subset()
    # Audit the complete original population/membership as well as the new subset.
    _, original_sample = sample_test_scenarios(populations, original["protocol"])
    resource = amended_resource()
    lookup = {(d, r["id"]): r for d, rows in populations.items() for r in rows}
    chosen, hashes = [], {}
    for dataset, scenario_id in resource["execution_order"]:
        key = (dataset, scenario_id)
        if key not in lookup:
            raise ValueError("amended fixed scenario missing")
        row = lookup[key]
        content_hash = digest(row)
        expected = resource["known_content_sha256"][dataset].get(scenario_id)
        if expected and expected != content_hash:
            raise ValueError("amended known scenario content mismatch")
        ordered = sorted(populations[dataset], key=lambda r: r["id"])
        position = resource["selected_ids"][dataset].index(scenario_id)
        index = resource["selected_indices"][dataset][position]
        if ordered[index]["id"] != scenario_id:
            raise ValueError("amended fixed scenario index mismatch")
        hashes.setdefault(dataset, {})[scenario_id] = content_hash
        chosen.append((dataset, row))
    sample = {
        "schema_version": 1,
        "resource_sha256": RESOURCE_SHA256,
        "original_sample": original_sample,
        "content_sha256": hashes,
        "full_order_sha256": resource["execution_order_sha256"],
    }
    return chosen[:count], sample, prefix_membership(count)


def dialogue_contract(selection_sha256, method, dataset, scenario_id, goals=None):
    if method not in {"TDRL", "DiffTOD"}:
        raise ValueError("unknown prefix method")
    return {
        "phase": "amended-fixed1000-prefix-official-test-v1",
        "method": method,
        "dataset": dataset,
        "selection_sha256": selection_sha256,
        "target_nodes": goals or [],
        "request_namespace": digest(
            [RESOURCE_SHA256, selection_sha256, method, dataset, scenario_id]
        ),
    }


def seal_sample(path, sample):
    """First post-selection content freeze is immutable across processes/devices."""
    raw = canonical(sample) + b"\n"
    # Atomic hard-link publication prevents a concurrent reader seeing a partial file.
    pending = path.with_name(path.name + f".{os.getpid()}.part")
    pending.write_bytes(raw)
    try:
        try:
            os.link(pending, path)
        except FileExistsError:
            pass
        if path.read_bytes() != raw:
            raise ValueError("sealed amended scenario content/provenance changed")
    finally:
        pending.unlink()


def compare_prefix_records(left, right, expected_ids, selection_sha):
    from .metrics import aggregate, paired_bootstrap, validate_records
    from .paired_development import generation_contract

    protocols = ("profile", "resistant")
    groups = []
    for records in (left, right):
        if len(records) != 2 * len(expected_ids):
            raise ValueError("incomplete official prefix matrix")
        cells = {}
        for protocol in protocols:
            rows = [r for r in records if r["protocol"] == protocol]
            validate_records(rows, expected_ids, [17], selection_sha)
            if any(r["status"] != "complete" for r in rows):
                raise ValueError("partial official prefix dialogue")
            cells[protocol] = {r["scenario_id"]: r for r in rows}
        groups.append(cells)
    a, b = groups
    for protocol in protocols:
        for sid in expected_ids:
            x, y = a[protocol][sid], b[protocol][sid]
            if generation_contract(x) != generation_contract(y) or any(
                x[field] != y[field]
                for field in ("judge_protocol_sha256", "acceptance_rule_version", "turn_budget")
            ):
                raise ValueError("paired official outcome/model/decoding mismatch")
    return {
        "scope": "single-seed sampled official-test prefix; not full split or human evidence",
        "candidate_metrics": {p: aggregate(list(a[p].values())) for p in protocols},
        "baseline_metrics": {p: aggregate(list(b[p].values())) for p in protocols},
        "paired_by_protocol": {
            p: paired_bootstrap(
                [a[p][i]["accepted"] for i in expected_ids],
                [b[p][i]["accepted"] for i in expected_ids],
            )
            for p in protocols
        },
        "paired_macro_protocol_sr": paired_bootstrap(
            [[a[p][i]["accepted"] for p in protocols] for i in expected_ids],
            [[b[p][i]["accepted"] for p in protocols] for i in expected_ids],
        ),
    }
