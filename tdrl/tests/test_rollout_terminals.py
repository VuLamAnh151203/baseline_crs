"""Control-flow evidence only: terminal judging still requires real H100 gates."""

from dataclasses import replace

import pytest
from test_core import FakeEmbedding, FakeGeneration, graph_fixture

from tdrl.judge import Judgment
from tdrl.planner import ConditionalPlanner, PlannerConfig


class FixtureJudge:
    protocol_hash = "fixture-protocol"

    def __init__(self, outcomes):
        self.outcomes, self.calls = outcomes, []

    def assess(self, state, scenario, seed, request_id):
        candidate = int(request_id.split("/rollout/")[1].split("/")[0])
        accepted, refused = self.outcomes[candidate]
        self.calls.append((candidate, state, scenario, seed, request_id))
        return Judgment(
            "acceptance" if accepted else "rejection" if refused else "unresolved",
            accepted,
            2,
            state.history[-1].content,
            len(state.history) - 1,
            refused,
            False,
            "achieved" if accepted else "worse" if refused else "unchanged",
            1.0 if accepted else -1.0 if refused else -0.5,
            "Film",
            accepted,
        )


def planner_fixture(outcomes, **kwargs):
    graph = graph_fixture()
    graph.references = {k: replace(v, action="recommend") for k, v in graph.references.items()}
    # Nonterminal learned cost is zero: semantic achievement must still win the tie.
    graph.retrieve = lambda live, goals: [(0.0, "2", ["2"])]
    judge = FixtureJudge(outcomes)
    planner = ConditionalPlanner(
        FakeGeneration(),
        FakeEmbedding(),
        graph,
        PlannerConfig(0, routing="always", revision=2, semantic_rollout_terminals=True, **kwargs),
        simulator_prompt="fixture",
        rollout_judge=judge,
        rollout_scenario={"target": ["Movie recommendation", "Film"], "knowledge": []},
    )
    return planner, judge, graph.references["0"].state


@pytest.mark.parametrize("workers", ["1", "3"])
def test_terminal_achievement_wins_zero_cost_tie_and_stops_lookahead(monkeypatch, workers):
    monkeypatch.setenv("TDRL_CANDIDATE_WORKERS", workers)
    planner, judge, state = planner_fixture([(False, False), (True, False), (False, True)])
    result = planner.act(state, ["2"], 17, "terminal")
    assert result.selected_candidate == 1
    assert result.rollout_scores == [0.0, 0.0, None]
    assert sorted(c for c, *_ in judge.calls) == [0, 0, 1, 2]
    assert not any(
        r.request_id.endswith("/assistant") and "/rollout/1/" in r.request_id
        for r in planner.generation.requests
    )
    assert [len(t) for t in result.compute["rollout_judgments"]] == [2, 1, 1]
    assert all(row[2]["latest_user_reply_finish_reason"] == "unknown" for row in judge.calls)
    assert all(row[-1] in result.replay_ids for row in judge.calls)
    assert result.compute["rollout_judge_protocol_sha256"] == judge.protocol_hash


def test_judge_transport_failure_does_not_fall_back_to_distance():
    planner, judge, state = planner_fixture([(False, False)] * 3)

    def fail(*args):
        raise ValueError("invalid judge JSON")

    judge.assess = fail
    with pytest.raises(ValueError, match="invalid judge JSON"):
        planner.act(state, ["2"], 17, "failure")


def test_terminal_budget_reserves_extra_judge_calls():
    planner, judge, state = planner_fixture([(True, False)] * 3, generation_call_cap=12)
    result = planner.act(state, ["2"], 17, "budget")
    assert result.compute["recovery_budget_exhausted"]
    assert not judge.calls
    assert result.compute["generation_requests"] == 1


def test_terminal_mode_requires_explicit_revision_and_frozen_judge():
    with pytest.raises(ValueError, match="revision 2"):
        PlannerConfig(0, semantic_rollout_terminals=True)
    with pytest.raises(ValueError, match="frozen local judge"):
        ConditionalPlanner(
            FakeGeneration(),
            FakeEmbedding(),
            None,
            PlannerConfig(0, revision=2, semantic_rollout_terminals=True),
        )
