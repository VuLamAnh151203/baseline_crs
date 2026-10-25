import json

import pytest

from tdrl.planner import PlannerConfig, target_fact_guidance


def context(knowledge):
    return json.dumps(
        {"target": ["Movie recommendation", "Current Film"], "profile": {}, "knowledge": knowledge}
    )


def facts(guidance):
    line = next(x for x in guidance.splitlines() if x.startswith("Current target facts"))
    return json.loads(line.split(": ", 1)[1])["facts"]


def test_grounding_keeps_subject_binding_and_never_adds_other_item_plot():
    rows = [
        ["Other Film", "Plot", "Other plot"],
        ["Current Film", "Rating", "7.6"],
        ["Current Film 2", "Plot", "Sequel plot"],
        ["Other Film", "Related film", "Current Film"],
    ]
    assert facts(target_fact_guidance(context(rows))) == [rows[1]]
    assert "Other plot" not in target_fact_guidance(context(rows))
    assert facts(target_fact_guidance(context([[" current  FILM ", "Type", "Comedy"]]))) == [
        [" current  FILM ", "Type", "Comedy"]
    ]
    assert facts(target_fact_guidance(context([]))) == []


def test_grounding_rejects_future_context_and_malformed_triples():
    payload = json.loads(context([]))
    payload["future_turns"] = ["Unknown future"]
    with pytest.raises(ValueError, match="shared visible context"):
        target_fact_guidance(json.dumps(payload))
    with pytest.raises(ValueError, match="triples"):
        target_fact_guidance(context([["Current Film", "Missing value"]]))
    with pytest.raises(ValueError, match="revision 2"):
        PlannerConfig(1, ground_target_facts=True)
    with pytest.raises(ValueError, match="revision 2"):
        PlannerConfig(1, revision=2, ground_target_facts="true")
