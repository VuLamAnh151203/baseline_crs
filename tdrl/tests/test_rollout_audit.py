"""Audit isolation and terminal-prefix accounting; synthetic labels are CPU tests only."""

import copy

import pytest

from tdrl.judge import LocalJudge
from tdrl.rollout_audit import audit_state, mrn_branches, point_id, summarize


def payload():
    branches = []
    for protocol in ["profile", "resistant"]:
        for turn in range(8):
            for candidate in range(3):
                history = [
                    {"role": "user", "content": "Suggest a movie."},
                    {"role": "assistant", "content": "Consider Example."},
                    {"role": "user", "content": "Tell me more."},
                ]
                points = [
                    {"depth": 0, "history": history, "latest_user_reply_finish_reason": "stop"}
                ]
                if len(branches) < 22:
                    points.append(
                        {
                            "depth": 1,
                            "history": history
                            + [
                                {"role": "assistant", "content": "Here is some information."},
                                {"role": "user", "content": "I will think about it."},
                            ],
                            "latest_user_reply_finish_reason": "stop",
                        }
                    )
                branches.append(
                    {
                        "variant": "repair-MRNFixed-Full",
                        "protocol": protocol,
                        "scenario_id": f"s_{turn}",
                        "turn": turn,
                        "candidate": candidate,
                        "selected": candidate == 0,
                        "leaf_cost": float(candidate + 1),
                        "scenario_context": {
                            "target": ["Movie recommendation", "Example"],
                            "profile": {},
                            "knowledge": [],
                        },
                        "points": points,
                    }
                )
    return {"schema_version": 1, "status": "awaiting_semantic_judgments", "branches": branches}


def test_frozen_mrn_membership_and_blind_judge_input():
    branches = mrn_branches(payload())
    assert len(branches) == 48 and sum(len(b["points"]) for b in branches) == 70
    branch = copy.deepcopy(branches[0])
    branch.update(candidate="HIDDEN_CANDIDATE", leaf_cost="HIDDEN_COST", selected="HIDDEN_SELECTED")
    state, scenario = audit_state(branch, branch["points"][0])
    judge = LocalJudge(None, {"schema_version": 2, "critic_samples": 10})
    text = judge.messages(state, scenario)[1].content
    assert all(
        marker not in text
        for marker in ["HIDDEN_CANDIDATE", "HIDDEN_COST", "HIDDEN_SELECTED", "repair-MRNFixed-Full"]
    )


def test_audit_counts_discarded_acceptance_and_continuation_separately():
    branches = mrn_branches(payload())
    labels = {
        point_id("input", b, p): {"accepted": False, "refused": False}
        for b in branches
        for p in b["points"]
    }
    branch = next(b for b in branches if not b["selected"] and len(b["points"]) == 2)
    labels[point_id("input", branch, branch["points"][0])]["accepted"] = True
    report = summarize(branches, labels, "input")
    result = report["results"][branch["protocol"]]
    assert result["turns_discarding_all_accepted_branches"] == 1
    assert result["branches_continued_after_acceptance"] == 1
    candidate = next(c for t in report["turns"] for c in t["candidates"] if c["ever_accepted"])
    assert not candidate["leaf_accepted"]
    labels.pop(next(iter(labels)))
    with pytest.raises(ValueError, match="missing or unexpected"):
        summarize(branches, labels, "input")


def test_audit_rejects_duplicate_candidates_and_unobserved_user_prefix():
    data = payload()
    branches = mrn_branches(data)
    branches[1]["candidate"] = branches[0]["candidate"]
    with pytest.raises(ValueError, match="three unique"):
        mrn_branches(data)
    data = payload()
    branch = mrn_branches(data)[0]
    branch["points"][0]["history"][-1]["role"] = "assistant"
    with pytest.raises(ValueError, match="observed latest"):
        mrn_branches(data)
