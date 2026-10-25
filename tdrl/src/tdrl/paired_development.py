"""Strict paired development comparison; never select repairs using official tests."""

from .metrics import aggregate, paired_bootstrap, validate_records
from .planner import derived_seed
from .provenance import digest


def cells(records, expected_count=64):
    groups = {}
    for row in records:
        if row["status"] != "complete" or row["infrastructure_errors"] != 0 or row["seed"] != 17:
            raise ValueError("incomplete/infrastructure/seed mismatch")
        if not isinstance(row["accepted"], bool):
            raise ValueError("invalid acceptance label")
        protocol = row["protocol"]
        if protocol not in {"profile", "resistant"}:
            raise ValueError("unknown protocol")
        by_id = groups.setdefault(protocol, {})
        if row["scenario_id"] in by_id:
            raise ValueError("duplicate scenario")
        by_id[row["scenario_id"]] = row
    if set(groups) != {"profile", "resistant"} or any(
        len(v) != expected_count for v in groups.values()
    ):
        raise ValueError("incomplete development matrix")
    if groups["profile"].keys() != groups["resistant"].keys():
        raise ValueError("unpaired protocol populations")
    for group in groups.values():
        validate_records(list(group.values()), list(group), [17], "development_or_smoke")
    return groups


def generation_contract(record):
    roles = {r: set() for r in ["assistant", "simulator", "judge"]}
    for turn in record["turns_trace"]:
        expected_seed = derived_seed(
            17,
            f"{record['scenario_id']}/{record['protocol']}/{turn['turn']}",
            "simulator",
        )
        if turn["simulator_seed"] != expected_seed:
            raise ValueError("shared simulator seed mismatch")
        for item in turn["generation"]:
            if item["role"] in roles:
                roles[item["role"]].add(
                    digest({"model": item["model"], "decoding": item["decoding"]})
                )
    if any(len(v) != 1 for v in roles.values()):
        raise ValueError("missing/inconsistent shared role model or decoding")
    return {r: next(iter(v)) for r, v in roles.items()}


def compare_records(candidate, baseline, expected_count=64):
    left, right = cells(candidate, expected_count), cells(baseline, expected_count)
    ids = sorted(left["profile"])
    paired = {}
    for protocol in ["profile", "resistant"]:
        if left[protocol].keys() != right[protocol].keys():
            raise ValueError("unpaired scenario membership")
        for sid in ids:
            a, b = left[protocol][sid], right[protocol][sid]
            for field in [
                "judge_protocol_sha256",
                "acceptance_rule_version",
                "turn_budget",
                "protocol",
                "seed",
            ]:
                if a[field] != b[field]:
                    raise ValueError("shared outcome/simulator contract mismatch: " + field)
            if generation_contract(a) != generation_contract(b):
                raise ValueError("shared model/decoding contract mismatch")
        paired[protocol] = paired_bootstrap(
            [left[protocol][i]["accepted"] for i in ids],
            [right[protocol][i]["accepted"] for i in ids],
        )
    macro = paired_bootstrap(
        [[left[p][i]["accepted"] for p in ["profile", "resistant"]] for i in ids],
        [[right[p][i]["accepted"] for p in ["profile", "resistant"]] for i in ids],
    )
    return {
        "candidate_metrics": {p: aggregate(list(v.values())) for p, v in left.items()},
        "baseline_metrics": {p: aggregate(list(v.values())) for p, v in right.items()},
        "paired_by_protocol": paired,
        "paired_macro_protocol_sr": macro,
        "candidate_macro_ci_excludes_zero_in_positive_direction": macro["ci95"][0] > 0,
        "scope_limits": "Single-seed paired simulated development evidence; discovery cannot replace disjoint confirmation.",
    }
