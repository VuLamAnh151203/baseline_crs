"""Baseline candidate keeps R's input/seed without extra candidate or rollout calls."""

import json
from dataclasses import replace

import pytest
from test_core import FakeEmbedding, FakeGeneration, graph_fixture

from tdrl.planner import ConditionalPlanner, PlannerConfig, derived_seed


@pytest.mark.parametrize("workers", [1, 3])
@pytest.mark.parametrize("with_graph", [False, True])
@pytest.mark.parametrize("observed_guidance", [False, True])
@pytest.mark.parametrize("grounded", [False, True])
def test_candidate_zero_matches_retrieval_contract_and_preserves_other_streams(
    monkeypatch, workers, with_graph, observed_guidance, grounded
):
    monkeypatch.setenv("TDRL_CANDIDATE_WORKERS", str(workers))
    fixture = graph_fixture()
    graph = fixture if with_graph else None
    if graph:
        graph.references = {k: replace(v, action="explain") for k, v in graph.references.items()}
        graph.add_edge("0", "2", 2)
    generation = FakeGeneration()
    state = fixture.references["0"].state
    common = dict(
        simulator_prompt="frozen simulator",
        visible_context=json.dumps(
            {"target": ["movie", "Film"], "profile": {}, "knowledge": [["Film", "Type", "Comedy"]]}
        )
        if grounded
        else "same visible context",
    )
    cfg = PlannerConfig(
        0,
        routing="always",
        revision=2,
        reference_selection="nearest_entry",
        preserve_retrieval_candidate=True,
        use_observed_guidance=observed_guidance,
        ground_target_facts=grounded,
    )
    retrieval = ConditionalPlanner(
        generation,
        FakeEmbedding(),
        graph,
        cfg.retrieval_control(),
        **common,
    )
    full = ConditionalPlanner(
        generation,
        FakeEmbedding(),
        graph,
        cfg,
        **common,
    )
    for planner in [retrieval, full]:
        planner.seed_context = "scenario/profile/turn1"
        planner.semantic_judging = True
    goals = ["2"] if graph else []
    retrieval.act(state, goals, 17, "R")
    plan = full.act(state, goals, 17, "Full")
    control = next(r for r in generation.requests if r.request_id == "R/final")
    zero = next(r for r in generation.requests if r.request_id == "Full/candidate/0")
    assert (zero.seed, zero.prompt_sha256, zero.decoding) == (
        control.seed,
        control.prompt_sha256,
        control.decoding,
    )
    assert plan.compute["generation_requests"] == 12
    assert plan.compute["embedding_requests"] == 4
    for candidate in [1, 2]:
        record = next(
            r for r in generation.requests if r.request_id == f"Full/candidate/{candidate}"
        )
        assert record.seed == derived_seed(17, full.seed_context, "candidate", candidate)
    for candidate in [0, 1, 2]:
        for depth in [0, 1]:
            record = next(
                r
                for r in generation.requests
                if r.request_id == f"Full/rollout/{candidate}/{depth}/user"
            )
            assert record.seed == derived_seed(
                17, full.seed_context, "rollout_user", candidate, depth
            )


def test_retrieval_candidate_is_opt_in_and_requires_correct_revision():
    assert not PlannerConfig(1).preserve_retrieval_candidate
    with pytest.raises(ValueError, match="revision 2"):
        PlannerConfig(1, preserve_retrieval_candidate=True)
