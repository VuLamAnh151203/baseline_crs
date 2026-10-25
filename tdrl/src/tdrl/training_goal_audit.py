"""Validate existing training outcome evidence; never infer acceptance from text."""

import json
from dataclasses import asdict

from .contracts import Message
from .evaluation import initial_state
from .judge import validate_judgment
from .state import DialogueState


def audited_terminal(scenario, record, judge_hash):
    if record.get("schema_version") != 1 or record.get("status") != "complete":
        raise ValueError("training dialogue is incomplete or unknown")
    if record["scenario_id"] != scenario["id"] or record["judge_protocol_sha256"] != judge_hash:
        raise ValueError("training scenario/judge provenance mismatch")
    if record["infrastructure_errors"] or record["method"] != "PPDPP":
        raise ValueError("invalid training source")
    trace = record["turns_trace"]
    history = tuple(Message(**m) for m in record["history"])
    prefix = initial_state(scenario).history
    if (
        history[: len(prefix)] != prefix
        or len(history) != len(prefix) + 2 * len(trace)
        or not trace
    ):
        raise ValueError("observed training history differs from scenario/turn count")
    if record["actual_turns"] != len(trace):
        raise ValueError("training turn count mismatch")
    last = None
    for index, turn in enumerate(trace):
        if turn["turn"] != index or last is not None and (last.accepted or last.refused):
            raise ValueError("training dialogue continued after terminal judgment")
        label = turn["acceptance"]
        if label["rule_version"] != 2:
            raise ValueError("historical parser labels cannot establish semantic goals")
        state = DialogueState(history[: len(prefix) + 2 * (index + 1)])
        payload = {
            k: label[k]
            for k in [
                "label",
                "progress",
                "new_choice",
                "selected_topic",
                "refused",
                "pressured",
                "evidence_turn",
            ]
        }
        last = validate_judgment(json.dumps(payload), state, scenario["target"])
        if asdict(last) != label:
            raise ValueError("stored judgment differs from validated observed evidence")
        users = [r for r in turn["generation"] if r["role"] == "simulator"]
        if len(users) != 1 or users[0]["text"] != state.history[-1].content:
            raise ValueError("user evidence differs from recorded simulator response")
        if last.accepted and users[0]["finish_reason"] == "length":
            raise ValueError("truncated user response cannot establish a goal")
    if record["accepted"] != last.accepted or record["outcome_label"] != last.label:
        raise ValueError("training outcome differs from terminal judgment")
    if not last.accepted:
        return None
    return {
        "scenario_id": scenario["id"],
        "target": scenario["target"],
        "state": DialogueState(history).serialize(),
        "evidence": last.evidence,
        "outcome_label": last.label,
        "judge_protocol_sha256": judge_hash,
        "source": "existing training-only local semantic outcome; not human annotation",
    }
