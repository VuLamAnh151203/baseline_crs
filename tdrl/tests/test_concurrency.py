import json
import threading

import pytest
from test_core import FakeEmbedding, FakeGeneration, graph_fixture

from tdrl.concurrency import ActiveShare, run_method_pair
from tdrl.contracts import Decoding, Message, Plan
from tdrl.evaluation import run_dialogue
from tdrl.planner import ConditionalPlanner, PlannerConfig
from tdrl.state import DialogueState


def test_shared_busy_time_is_conserved_when_workers_overlap():
    now = [0.0]
    meter = ActiveShare(lambda: now[0])
    meter.enter("PPDPP")
    now[0] = 2.0
    meter.enter("TDRL")
    now[0] = 6.0
    assert meter.leave("PPDPP") == 4.0
    now[0] = 9.0
    assert meter.leave("TDRL") == 5.0
    assert sum(meter.shares.values()) == 9.0


def test_concurrent_dialogue_counters_and_resume_are_isolated(tmp_path):
    barrier = threading.Barrier(2)
    generation, embedding = FakeGeneration(), FakeEmbedding()

    class Actor:
        def act(self, state, goals, seed, request_id):
            embedding.embed([state.serialize()], "fixture", request_id, request_id + "/live")
            reply = generation.generate(
                [Message("user", "hello")], "assistant", Decoding(), seed, request_id + "/final"
            )
            # Both workers have appended requests before either builds its trace.
            barrier.wait(timeout=5)
            return Plan(reply.text, False, None, {}, [], None, [], {})

    scenario = {
        "id": "same-scenario",
        "profile": {},
        "personality": {},
        "knowledge": [],
        "target": ["movie", "The Star"],
        "messages": [{"role": "user", "content": "I like films."}],
    }

    def callback(method):
        path = tmp_path / method
        return run_dialogue(
            scenario,
            Actor(),
            generation,
            embedding,
            "profile",
            17,
            path,
            {"phase": "fixture", "method": method},
            turn_budget=1,
        ), path

    results = run_method_pair(callback, tmp_path / "timing")
    for result in results:
        assert result["generation_requests"] == 2
        assert result["embedding_requests"] == 1
        assert result["total_tokens"] == 10
        assert result["compute_accounting"] == "equal_active_dialogue_share_v1"
        for request in result["turns_trace"][0]["generation"]:
            assert "/" + result["method"] + "/" in request["request_id"]
    assert sum(r["generation_requests"] for r in results) == len(generation.requests)
    counts = len(generation.requests), len(embedding.requests)
    assert run_method_pair(callback, tmp_path / "timing") == results
    assert counts == (len(generation.requests), len(embedding.requests))
    timing = json.loads((tmp_path / "timing/pair-timing.json").read_text())
    assert sum(r["h100_seconds"] for r in results) == pytest.approx(timing["h100_seconds"])
    assert timing["h100_seconds"] <= timing["wall_s"]


def test_pair_failure_drains_other_worker_and_retains_its_record(tmp_path):
    finished = threading.Event()

    def callback(method):
        if method == "PPDPP":
            raise ValueError("inference failed")
        finished.set()
        return {"method": method}, tmp_path / method

    with pytest.raises(ValueError, match="inference failed"):
        run_method_pair(callback, tmp_path / "timing")
    assert finished.is_set()
    assert (tmp_path / "TDRL/dialogue.json").exists()
    assert not (tmp_path / "timing/pair-timing.json").exists()


def test_nested_planner_counters_exclude_interleaved_foreign_requests():
    class InterleavedGeneration(FakeGeneration):
        def generate(self, messages, role, decoding, seed, request_id):
            own = super().generate(messages, role, decoding, seed, request_id)
            super().generate(messages, role, decoding, seed, "foreign/" + request_id)
            return own

    class InterleavedEmbedding(FakeEmbedding):
        def embed(self, texts, instruction, batch_id, request_id):
            own = super().embed(texts, instruction, batch_id, request_id)
            super().embed(texts, instruction, batch_id, "foreign/" + request_id)
            return own

    graph = graph_fixture()
    graph.add_edge("0", "2", 2)
    generation, embedding = InterleavedGeneration(), InterleavedEmbedding()
    planner = ConditionalPlanner(
        generation,
        embedding,
        graph,
        PlannerConfig(h_cover=0.1, candidates=2, lookahead=2),
        simulator_prompt="frozen-user",
    )
    plan = planner.act(graph.references["0"].state, ["2"], 17, "dialogue/turn/0")
    assert len(generation.requests) == 16
    assert len(embedding.requests) == 6
    assert plan.compute["generation_requests"] == 8
    assert plan.compute["embedding_requests"] == 3
    assert plan.compute["total_tokens"] == 40


def test_candidate_branches_overlap_keep_serial_semantics_and_drain_errors(monkeypatch):
    graph = graph_fixture()
    graph.add_edge("0", "2", 2)
    cfg = PlannerConfig(h_cover=0.1, candidates=3, lookahead=2)
    generation, embedding = FakeGeneration(), FakeEmbedding()
    serial = ConditionalPlanner(generation, embedding, graph, cfg, simulator_prompt="frozen-user")
    expected = serial.act(graph.references["0"].state, ["2"], 17, "turn")
    expected_requests = {r.request_id: (r.seed, r.prompt_sha256) for r in generation.requests}
    barrier = threading.Barrier(3)
    finished = []
    failed = [False]

    class ConcurrentGeneration(FakeGeneration):
        def generate(self, messages, role, decoding, seed, request_id):
            if "/candidate/" in request_id:
                barrier.wait(timeout=5)
                if failed[0] and request_id.endswith("/0"):
                    raise ValueError("failed branch")
            result = super().generate(messages, role, decoding, seed, request_id)
            if request_id.endswith("/1/user"):
                finished.append(request_id.split("/rollout/")[1].split("/")[0])
            return result

    monkeypatch.setenv("TDRL_CANDIDATE_WORKERS", "3")
    generation, embedding = ConcurrentGeneration(), FakeEmbedding()
    parallel = ConditionalPlanner(generation, embedding, graph, cfg, simulator_prompt="frozen-user")
    result = parallel.act(graph.references["0"].state, ["2"], 17, "turn")
    assert result.response == expected.response
    assert result.selected_candidate == expected.selected_candidate == 0
    assert result.rollout_scores == expected.rollout_scores
    assert result.replay_ids == expected.replay_ids
    assert result.compute["generation_requests"] == 12
    assert result.compute["embedding_requests"] == 4
    assert {
        r.request_id: (r.seed, r.prompt_sha256) for r in generation.requests
    } == expected_requests
    finished.clear()
    failed[0] = True
    with pytest.raises(ValueError, match="failed branch"):
        parallel.act(graph.references["0"].state, ["2"], 17, "fail")
    assert set(finished) == {"1", "2"}


def test_parallel_candidates_preserve_matched_budget_and_refusal(monkeypatch):
    monkeypatch.setenv("TDRL_CANDIDATE_WORKERS", "3")
    graph = graph_fixture()
    graph.add_edge("0", "2", 2)
    generation, embedding = FakeGeneration(), FakeEmbedding()
    planner = ConditionalPlanner(
        generation,
        embedding,
        graph,
        PlannerConfig(0.1, generation_call_cap=1),
        simulator_prompt="fixture",
    )
    plan = planner.act(graph.references["0"].state, ["2"], 17, "matched")
    assert plan.compute["recovery_budget_exhausted"]
    assert plan.compute["generation_requests"] == 1
    assert plan.selected_candidate is None
    refused = planner.act(
        DialogueState((Message("user", "No thanks, I'm not interested."),)), ["2"], 17, "refused"
    )
    assert refused.compute["generation_requests"] == 0
