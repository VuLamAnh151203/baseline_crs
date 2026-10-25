from dataclasses import replace

import pytest
import torch
from test_core import FakeEmbedding, FakeGeneration, OrderedHead, graph_fixture

from tdrl.planner import ConditionalPlanner, PlannerConfig
from tdrl.strategies import ACTION_INSTRUCTIONS
from tdrl.training import make_graph


def test_actionable_guidance_uses_next_move_not_three_future_moves():
    graph = graph_fixture()
    graph.references = {k: replace(v, action="explain") for k, v in graph.references.items()}
    graph.add_edge("0", "1", 1)
    graph.add_edge("1", "2", 1)
    captured = []

    class Generation(FakeGeneration):
        def generate(self, messages, *args):
            captured.append(messages[0].content)
            return super().generate(messages, *args)

    planner = ConditionalPlanner(
        Generation(),
        FakeEmbedding(),
        graph,
        PlannerConfig(10, routing="retrieval_only", revision=2),
    )
    result = planner.act(graph.references["0"].state, ["2"], 17, "actionable")
    assert ACTION_INSTRUCTIONS["explain"] in captured[0]
    assert "next move from this directed offline guidance path" not in captured[0]
    assert result.compute["generation_requests"] == 1
    assert result.compute["leaf_scoring"] == "directed_entry_plus_path_cost"


def test_remaining_cost_uses_recorded_forward_edges():
    graph = graph_fixture()
    graph.add_edge("0", "1", 2)
    graph.add_edge("1", "2", 3)
    assert graph.remaining_cost(["0", "1", "2"]) == 5
    assert graph.remaining_cost(["2"]) == 0
    with pytest.raises(KeyError):
        graph.remaining_cost(["2", "1"])


def test_observed_move_guidance_is_opt_in_and_keeps_retrieval_and_call_budget():
    graph = graph_fixture()
    graph.references = {
        k: replace(v, action="recommend", guidance="Training example " + k)
        for k, v in graph.references.items()
    }
    captured = []

    class Generation(FakeGeneration):
        def generate(self, messages, *args):
            captured.append(messages[0].content)
            return super().generate(messages, *args)

    results = []
    for enabled in [False, True]:
        planner = ConditionalPlanner(
            Generation(),
            FakeEmbedding(),
            graph,
            PlannerConfig(10, revision=2, routing="retrieval_only", use_observed_guidance=enabled),
        )
        results.append(planner.act(graph.references["0"].state, ["0"], 17, str(enabled)))
    assert "Training example" not in captured[0]
    assert "Training example " + results[1].reference_id in captured[1]
    assert "never copy its names, preferences" in captured[1]
    assert results[0].reference_id == results[1].reference_id
    assert results[0].directed_distances == results[1].directed_distances
    assert all(r.compute["generation_requests"] == 1 for r in results)
    with pytest.raises(ValueError, match="revision 2"):
        PlannerConfig(10, use_observed_guidance=True)


def test_goal_anchor_conditioning_reserves_known_target_and_declares_transfer():
    state = graph_fixture().references["0"].state.serialize()
    records = [
        {
            "split": "train",
            "guidance": "observed target mention",
            "target": ["movie", topic],
            "node_id": str(i),
            "state": state,
            "action": "recommend",
            "dialogue_id": "train" + str(i),
            "goal_reference": True,
        }
        for i, topic in enumerate(["Wrong Film", "Current Film"])
    ]
    graph, goals = make_graph(
        records,
        torch.tensor([[0.0], [1.0]]),
        OrderedHead(),
        10,
        1,
        17,
        "movie",
        "Current Film",
        True,
    )
    assert goals == ["1"] and set(graph.references) == {"1"}
    assert graph.references["1"].action == "recommend"
    assert graph.provenance["goal_conditioning"] == "exact_target"
    transferred, goals = make_graph(
        records,
        torch.tensor([[0.0], [1.0]]),
        OrderedHead(),
        10,
        2,
        17,
        "movie",
        "Unseen Film",
        True,
    )
    assert set(goals) == {"0", "1"}
    assert transferred.provenance["goal_conditioning"] == "goal_type_transfer"


def test_planner_leaf_score_includes_remaining_goal_cost():
    graph = graph_fixture()
    graph.references = {k: replace(v, action="recommend") for k, v in graph.references.items()}
    graph.add_edge("0", "2", 10)
    graph.retrieve = lambda live, goals: [(0.1, "0", ["0", "2"]), (2.0, "2", ["2"])]
    legacy = ConditionalPlanner(
        FakeGeneration(), FakeEmbedding(), graph, PlannerConfig(0), simulator_prompt="fixture"
    )
    repaired = ConditionalPlanner(
        FakeGeneration(),
        FakeEmbedding(),
        graph,
        PlannerConfig(0, revision=2),
        simulator_prompt="fixture",
    )
    before = legacy.act(graph.references["0"].state, ["2"], 17, "legacy")
    after = repaired.act(graph.references["0"].state, ["2"], 17, "repair")
    assert before.rollout_scores == [0.1] * 3
    assert after.rollout_scores == [2.0] * 3
    assert after.reference_id == "2"


def test_target_reference_pool_keeps_intermediate_moves_and_unseen_goal_transfer():
    state = graph_fixture().references["0"].state.serialize()
    records = [
        {
            "split": "train",
            "guidance": "training next move",
            "target": ["movie", topic],
            "node_id": str(i),
            "state": state,
            "action": "explain",
            "dialogue_id": "train" + str(i),
            "goal_reference": goal,
        }
        for i, (topic, goal) in enumerate(
            [("Wrong Film", True), ("Current Film", False), ("Current Film", True)]
        )
    ]
    vectors = torch.tensor([[0.0], [1.0], [2.0]])
    graph, goals = make_graph(
        records,
        vectors,
        OrderedHead(),
        10,
        3,
        17,
        "movie",
        "Current Film",
        True,
        reference_scope="exact_target_when_supported",
    )
    assert set(graph.references) == {"1", "2"}
    assert goals == ["2"]
    assert graph.provenance["reference_scope"] == "exact_target"
    transferred, goals = make_graph(
        records,
        vectors,
        OrderedHead(),
        10,
        3,
        17,
        "movie",
        "Unseen Film",
        True,
        reference_scope="exact_target_when_supported",
    )
    assert set(transferred.references) == {"0", "1", "2"}
    assert set(goals) == {"0", "2"}
    assert transferred.provenance["reference_scope"] == "goal_type"
    assert transferred.provenance["goal_conditioning"] == "goal_type_transfer"
    with pytest.raises(ValueError, match="unknown training reference scope"):
        make_graph(records, vectors, OrderedHead(), 10, 3, 17, "movie", reference_scope="unknown")


def test_nearest_entry_retains_local_next_move_instead_of_skipping_to_goal():
    graph = graph_fixture()
    graph.references = {
        k: replace(v, action="recommend" if k == "2" else "ask_preference")
        for k, v in graph.references.items()
    }
    graph.add_edge("0", "2", 2)
    # Triangle consistency permits the goal's direct distance to beat total path
    # cost, while the intermediate reference still best matches the live state.
    graph.retrieve = lambda live, goals: [(0.1, "0", ["0", "2"]), (2.0, "2", ["2"])]
    results = {}
    for selection in ["entry_plus_path", "nearest_entry"]:
        planner = ConditionalPlanner(
            FakeGeneration(),
            FakeEmbedding(),
            graph,
            PlannerConfig(10, routing="retrieval_only", revision=2, reference_selection=selection),
        )
        results[selection] = planner.act(graph.references["0"].state, ["2"], 17, selection)
    assert results["entry_plus_path"].reference_id == "2"
    assert results["nearest_entry"].reference_id == "0"
    assert results["nearest_entry"].path == ["0", "2"]
    assert all(p.covered and p.compute["generation_requests"] == 1 for p in results.values())
    with pytest.raises(ValueError, match="unknown reference selection"):
        PlannerConfig(1, reference_selection="unknown")
