import copy
import json
from dataclasses import asdict

import pytest

from tdrl.contracts import Message
from tdrl.judge import validate_judgment
from tdrl.state import DialogueState
from tdrl.training_goal_audit import audited_terminal


def fixture(accepted=True):
    scenario = {
        "id": "train-only",
        "target": ["Movie recommendation", "Example"],
        "messages": [{"role": "user", "content": "Suggest a movie."}],
    }
    history = [
        *scenario["messages"],
        {"role": "assistant", "content": "Consider Example."},
        {"role": "user", "content": "I choose Example." if accepted else "No thanks."},
    ]
    payload = {
        "new_choice": accepted,
        "selected_topic": "Example" if accepted else "NO_NEW_CHOICE",
        "label": "acceptance" if accepted else "rejection",
        "progress": "achieved" if accepted else "worse",
        "refused": not accepted,
        "pressured": False,
        "evidence_turn": 2,
    }
    label = asdict(
        validate_judgment(
            json.dumps(payload),
            DialogueState(tuple(Message(**m) for m in history)),
            scenario["target"],
        )
    )
    record = {
        "schema_version": 1,
        "status": "complete",
        "scenario_id": scenario["id"],
        "judge_protocol_sha256": "frozen-judge",
        "infrastructure_errors": 0,
        "method": "PPDPP",
        "turns_trace": [
            {
                "turn": 0,
                "acceptance": label,
                "generation": [
                    {"role": "simulator", "text": history[-1]["content"], "finish_reason": "stop"}
                ],
            }
        ],
        "history": history,
        "actual_turns": 1,
        "accepted": accepted,
        "outcome_label": payload["label"],
    }
    return scenario, record


def test_goal_audit_uses_validated_user_acceptance_not_assistant_mention():
    scenario, record = fixture()
    goal = audited_terminal(scenario, record, "frozen-judge")
    assert goal["evidence"] == "I choose Example."
    assert "target" not in json.loads(goal["state"])
    scenario, rejection = fixture(False)
    assert audited_terminal(scenario, rejection, "frozen-judge") is None


@pytest.mark.parametrize(
    "field,value",
    [("rule_version", 1), ("evidence", "fabricated"), ("selected_topic", "OTHER_TOPIC")],
)
def test_goal_audit_rejects_legacy_wrong_target_or_unobserved_evidence(field, value):
    scenario, record = fixture()
    record["turns_trace"][0]["acceptance"][field] = value
    with pytest.raises(ValueError):
        audited_terminal(scenario, record, "frozen-judge")


def test_goal_audit_rejects_continuation_after_success_and_mismatched_scenario():
    scenario, record = fixture()
    record["history"] += [
        {"role": "assistant", "content": "Keep discussing."},
        {"role": "user", "content": "Enough."},
    ]
    record["turns_trace"].append(
        {"turn": 1, "acceptance": copy.deepcopy(record["turns_trace"][0]["acceptance"])}
    )
    record["actual_turns"] = 2
    with pytest.raises(ValueError, match="continued after terminal"):
        audited_terminal(scenario, record, "frozen-judge")
    scenario, record = fixture()
    record["scenario_id"] = "dev-case"
    with pytest.raises(ValueError, match="provenance mismatch"):
        audited_terminal(scenario, record, "frozen-judge")


def test_goal_audit_rejects_truncated_acceptance():
    scenario, record = fixture()
    record["turns_trace"][0]["generation"][0]["finish_reason"] = "length"
    with pytest.raises(ValueError, match="truncated"):
        audited_terminal(scenario, record, "frozen-judge")
