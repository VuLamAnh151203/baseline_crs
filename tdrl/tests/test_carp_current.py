from dataclasses import dataclass

from tdrl.carp_current import NativePlannerCarpAdapter, history_to_native, scenario_to_native
from tdrl.contracts import Decoding, Generation, ModelProvenance, Plan


@dataclass
class FakeCarpMessage:
    role: str
    text: str


@dataclass
class FakeScenario:
    id: str = "scenario-1"
    target_action: str = "recommend movie"
    target_item: str = "Arrival"
    user_profile: dict = None
    user_personality: dict = None
    knowledge: list = None
    dataset_name: str = "topdial"
    domain: str = "movie"

    def __post_init__(self):
        self.user_profile = self.user_profile or {"Accepted movies": "Dune"}
        self.user_personality = self.user_personality or {"style": "brief"}
        self.knowledge = self.knowledge or [("Arrival", "genre", "science fiction")]


class FakeGeneration:
    def __init__(self):
        self.consumed_requests = []


class FakePlanner:
    def __init__(self, generation, scenario):
        self.generation = generation
        self.scenario = scenario
        self.seed_context = scenario["id"]

    def act(self, state, target_nodes, seed, request_id):
        assert not target_nodes
        assert seed == 17
        roles = [message.role for message in state.history]
        assert self.seed_context.endswith(f"/carp-current/{roles.count('assistant')}")
        assert roles in (["assistant", "user"], [])
        self.generation.consumed_requests.append(
            Generation(
                "answer",
                10,
                2,
                12,
                1,
                Decoding(max_tokens=128),
                0.25,
                0,
                ModelProvenance("Qwen/Qwen3.5-9B", "a" * 40, {"w": "h"}, "t"),
                "p",
                request_id + "/native",
                "assistant",
                "stop",
            )
        )
        return Plan("answer", False, None, {}, [], None, [], {"generation_requests": 1}, [request_id])


def test_scenario_and_history_translation_exposes_no_released_messages():
    scenario = FakeScenario()
    native = scenario_to_native(scenario)
    assert native["target"] == ["recommend movie", "Arrival"]
    assert native["messages"] == []
    assert native["knowledge"] == [["Arrival", "genre", "science fiction"]]
    state = history_to_native([FakeCarpMessage("assistant", "a"), FakeCarpMessage("user", "u")])
    assert [(row.role, row.content) for row in state.history] == [("assistant", "a"), ("user", "u")]


def test_adapter_preserves_native_planner_and_reports_per_dialogue_compute():
    scenario = FakeScenario()
    generation = FakeGeneration()
    adapter = NativePlannerCarpAdapter(
        "procot",
        generation,
        lambda native: FakePlanner(generation, native),
        root_seed=17,
        request_namespace="contract",
        method_provenance={"policy_version": "test"},
    )
    result = adapter.generate_reply(
        scenario,
        [FakeCarpMessage("assistant", "a"), FakeCarpMessage("user", "u")],
    )
    assert result == "answer"
    meta = adapter.get_assistant_meta()
    assert meta["coverage_available"] is False
    assert meta["compute"] == {
        "assistant_generation_requests": 1,
        "assistant_input_tokens": 10,
        "assistant_output_tokens": 2,
        "assistant_total_tokens": 12,
        "assistant_generation_latency_s": 0.25,
    }
    adapter.generate_reply(FakeScenario(id="scenario-2"), [])
    assert len(adapter.get_assistant_meta()["turn_traces"]) == 1
