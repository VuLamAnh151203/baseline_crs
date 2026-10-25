import json
from dataclasses import replace

import pytest
import torch
from test_core import FakeEmbedding, FakeGeneration, graph_fixture

from tdrl.backend import EMBEDDING_PREPROCESSING
from tdrl.contracts import Decoding, Message
from tdrl.difftod import DiffTODPlanner, native_plan_text
from tdrl.planner import INSTRUCTION
from tdrl.provenance import digest


def test_native_header_and_eos_extraction_retains_upstream_cases():
    profile = "user profile: Name: A\t"
    target = ["Movie recommendation", "Film"]
    header = profile + "target: Movie recommendation, Film\n"
    assert native_plan_text(header + "first", profile, target) == "first"
    assert (
        native_plan_text(header + "first<|endoftext|>second<|endoftext|>third", profile, target)
        == "firstsecond"
    )
    assert (
        native_plan_text(header + "first<|endoftext|>user profile: B", profile, target) == "first"
    )


def actor_fixture(tmp_path, monkeypatch):
    import tdrl.difftod as adapter

    class Generator:
        def __init__(self, **kwargs):
            pass

        def manual_seed(self, seed):
            return self

    class Policy:
        def eval(self):
            return self

    class Tokenizer:
        def encode(self, text):
            return type("Encoded", (), {"ids": [1, 2]})()

        def decode(self, ids, skip_special_tokens=False):
            return "user profile: Name: A\ttarget: Movie recommendation, Film\nplanned response"

    class Generation(FakeGeneration):
        def generate(self, messages, role, *args):
            result = super().generate(messages, role, *args)
            if role == "planner_guidance":
                result = replace(
                    result,
                    text="One target phrase\nTwo target phrase\nThree target phrase\nFour target phrase\nFive target phrase",
                    finish_reason="stop",
                )
            return result

    calls = []

    def sample(*args):
        calls.append(True)
        return torch.tensor([[1, 2, 3]])

    monkeypatch.setattr(adapter.torch, "Generator", Generator)
    monkeypatch.setattr(adapter, "sample_analytic", sample)
    scenario = {
        "id": "case1",
        "profile": {"Name": "A"},
        "target": ["Movie recommendation", "Film"],
        "knowledge": [],
    }
    generation = Generation()

    class FullEmbedding(FakeEmbedding):
        def embed(self, *args):
            value = super().embed(*args)
            vectors = torch.zeros(1, 4096)
            vectors[0, 0] = -1
            value = replace(
                value,
                vectors=vectors,
                instruction_sha256=digest(INSTRUCTION),
                preprocessing_sha256=digest(EMBEDDING_PREPROCESSING),
            )
            self.requests[-1] = value
            return value

    actor = DiffTODPlanner(
        Policy(),
        Tokenizer(),
        generation,
        FullEmbedding(),
        scenario,
        Decoding(),
        Decoding(),
        tmp_path,
        {"checkpoint": "fixed"},
        tmp_path,
        consensus=lambda plans, path: (2, [1.0] * 5),
    )
    actor.seed_context = "paired-case/profile/0"
    return actor, generation, calls


def test_diffusion_plan_reuses_five_native_branches_for_later_turns_and_resume(
    tmp_path, monkeypatch
):
    actor, generation, calls = actor_fixture(tmp_path, monkeypatch)
    state = graph_fixture().references["0"].state
    first = actor.act(state, [], 17, "dialogue/turn/0")
    actor.seed_context = "paired-case/profile/1"
    again = actor.act(state, [], 17, "dialogue/turn/1")
    assert len(calls) == 5 and first.selected_candidate == again.selected_candidate == 2
    assert first.compute["diffusion"]["policy_forwards"] == 5 * 1025
    assert len([r for r in generation.requests if r.role == "planner_guidance"]) == 1
    actor.trace = None
    recovered = actor.act(state, [], 17, "dialogue/turn/2")
    assert len(calls) == 5 and recovered.compute["diffusion"] == first.compute["diffusion"]
    assert not list(tmp_path.glob("*.pending.json"))


def test_pending_policy_outcome_and_corrupt_journal_fail_closed(tmp_path, monkeypatch):
    actor, _, _ = actor_fixture(tmp_path, monkeypatch)
    state = graph_fixture().references["0"].state
    actor.act(state, [], 17, "dialogue/turn/0")
    path = next(
        p
        for p in tmp_path.glob("*.json")
        if "chosen" in json.loads(p.read_text()).get("outputs", {})
    )
    record = json.loads(path.read_text())
    record["outputs"]["chosen"] = 4
    path.write_text(json.dumps(record))
    actor.trace = None
    with pytest.raises(ValueError, match="checksum"):
        actor.act(state, [], 17, "dialogue/turn/1")


def test_completed_live_embedding_replay_does_not_issue_duplicate_request(tmp_path, monkeypatch):
    actor, generation, calls = actor_fixture(tmp_path, monkeypatch)
    state = graph_fixture().references["0"].state
    actor.act(state, [], 17, "dialogue/turn/0")
    requests = len(actor.embedding.requests)
    cached = actor._live_embedding(state, "dialogue/turn/0/live")
    assert len(actor.embedding.requests) == requests
    assert torch.equal(cached.vectors, actor.embedding.requests[-1].vectors)
    live = next(tmp_path.glob("live-*.json"))
    value = json.loads(live.read_text())
    value["outputs"]["vectors"][0][0] = 0
    live.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="checksum"):
        actor._live_embedding(state, "dialogue/turn/0/live")


def test_live_embedding_cache_rejects_reused_id_with_changed_state_or_unknown_outcome(
    tmp_path, monkeypatch
):
    actor, _, _ = actor_fixture(tmp_path, monkeypatch)
    state = graph_fixture().references["0"].state
    actor._live_embedding(state, "live-id")
    changed = state.append(Message("user", "changed"))
    with pytest.raises(ValueError, match="provenance"):
        actor._live_embedding(changed, "live-id")
    pending = tmp_path / ("live-" + digest("unknown") + ".pending.json")
    pending.write_text("{}")
    with pytest.raises(ValueError, match="unknown live embedding"):
        actor._live_embedding(state, "unknown")
