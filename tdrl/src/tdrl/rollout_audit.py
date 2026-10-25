"""Frozen MRN rollout-prefix audit, separate from planner/evaluation outcomes."""

import math
from collections import defaultdict

from .contracts import Message
from .provenance import digest
from .state import DialogueState


def mrn_branches(payload):
    if payload.get("schema_version") != 1 or payload.get("status") != "awaiting_semantic_judgments":
        raise ValueError("unknown frozen rollout audit input")
    branches = [b for b in payload["branches"] if b["variant"] == "repair-MRNFixed-Full"]
    groups = defaultdict(list)
    for branch in branches:
        key = (branch["protocol"], branch["scenario_id"], branch["turn"])
        if branch["protocol"] not in {"profile", "resistant"}:
            raise ValueError("unknown user protocol")
        if not math.isfinite(branch["leaf_cost"]) or branch["leaf_cost"] < 0:
            raise ValueError("invalid leaf cost")
        points = branch["points"]
        if not points or [p["depth"] for p in points] != list(range(len(points))):
            raise ValueError("missing/out-of-order rollout prefix")
        if len(points) > 2:
            raise ValueError("unexpected lookahead budget")
        for point in points:
            audit_state(branch, point)
        groups[key].append(branch)
    if len(groups) != 16 or any(
        sum(k[0] == p for k in groups) != 8 for p in ["profile", "resistant"]
    ):
        raise ValueError("audit requires eight frozen turns per protocol")
    for group in groups.values():
        if (
            sorted(b["candidate"] for b in group) != [0, 1, 2]
            or sum(b["selected"] for b in group) != 1
        ):
            raise ValueError("audit requires three unique candidates and one chosen response")
    return branches


def audit_state(branch, point):
    # Explicit allowlist: the judge never receives branch identity, scores or selection.
    state = DialogueState(tuple(Message(**m) for m in point["history"]))
    if not state.history or state.history[-1].role != "user":
        raise ValueError("audit judge requires observed latest user reply")
    scenario = {k: branch["scenario_context"][k] for k in ["target", "profile", "knowledge"]}
    scenario["latest_user_reply_finish_reason"] = point["latest_user_reply_finish_reason"]
    return state, scenario


def point_id(input_hash, branch, point):
    return digest(
        [
            "mrn-rollout-audit-v1",
            input_hash,
            branch["protocol"],
            branch["scenario_id"],
            branch["turn"],
            branch["candidate"],
            point["depth"],
        ]
    )


def summarize(branches, judgments, input_hash):
    expected = {point_id(input_hash, b, p) for b in branches for p in b["points"]}
    if set(judgments) != expected:
        raise ValueError("missing or unexpected semantic judgments")
    groups = defaultdict(list)
    for b in branches:
        labels = [judgments[point_id(input_hash, b, p)] for p in b["points"]]
        groups[(b["protocol"], b["scenario_id"], b["turn"])].append(
            {
                "candidate": b["candidate"],
                "selected": b["selected"],
                "leaf_cost": b["leaf_cost"],
                "ever_accepted": any(x["accepted"] for x in labels),
                "leaf_accepted": labels[-1]["accepted"],
                "continued_after_acceptance": any(x["accepted"] for x in labels[:-1]),
                "continued_after_refusal": any(x["refused"] for x in labels[:-1]),
            }
        )
    totals = {}
    rows = []
    for (protocol, scenario_id, turn), candidates in sorted(groups.items()):
        selected = next(x for x in candidates if x["selected"])
        missed = any(x["ever_accepted"] for x in candidates) and not selected["ever_accepted"]
        totals.setdefault(
            protocol,
            {
                "turns": 0,
                "turns_with_acceptance": 0,
                "turns_discarding_all_accepted_branches": 0,
                "branches_continued_after_acceptance": 0,
                "branches_continued_after_refusal": 0,
            },
        )
        counter = totals[protocol]
        counter["turns"] += 1
        counter["turns_with_acceptance"] += int(any(x["ever_accepted"] for x in candidates))
        counter["turns_discarding_all_accepted_branches"] += int(missed)
        for name in ["acceptance", "refusal"]:
            counter["branches_continued_after_" + name] += sum(
                x["continued_after_" + name] for x in candidates
            )
        rows.append(
            {
                "protocol": protocol,
                "scenario_id": scenario_id,
                "turn": turn,
                "candidates": candidates,
                "discarded_all_accepted_branches": missed,
            }
        )
    return {
        "schema_version": 1,
        "scope": "MRN-only frozen development rollout-prefix audit",
        "branches": len(branches),
        "observed_user_prefixes": len(expected),
        "results": totals,
        "turns": rows,
        "limitation": "Hypothetical branch judgments are diagnostics, not fresh independent real-user continuations or a causal SR estimate.",
    }
