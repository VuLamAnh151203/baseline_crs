import json
from dataclasses import asdict
from pathlib import Path

from tdrl.baselines.direct_llm.planner import (
    DIRECT_LLM_POLICY_VERSION,
    DIRECT_LLM_V2_POLICY_VERSION,
    DirectLLMPlanner,
    DirectLLMV2Planner,
    direct_system_prompt,
    direct_v2_system_prompt,
)
from tdrl.comparison_direct_llm import validate_config
from tdrl.contracts import Decoding, Generation, Message, ModelProvenance
from tdrl.evaluation import visible_context
from tdrl.planner import (
    ASSISTANT_POLICY,
    derived_seed,
    target_fact_guidance,
)
from tdrl.state import DialogueState


class FakeGeneration:
    def __init__(self):
        self.requests = []

    def generate(self, messages, role, decoding, seed, request_id):
        result = Generation(
            text="A grounded direct response.",
            input_tokens=10,
            output_tokens=5,
            total_tokens=15,
            seed=seed,
            decoding=decoding,
            latency_s=0.1,
            retries=0,
            model=ModelProvenance(
                "Qwen/Qwen3.5-9B",
                "a" * 40,
                {"weights.safetensors": "b" * 64},
                "c" * 64,
            ),
            prompt_sha256="d" * 64,
            request_id=request_id,
            role=role,
            finish_reason="stop",
        )
        self.requests.append((messages, result))
        return result


def scenario():
    return {
        "id": "direct-test",
        "target": ["Movie recommendation", "Target Film"],
        "profile": {"genre": "drama"},
        "knowledge": [["Target Film", "genre", "drama"]],
        "personality": "HIDDEN_PERSONALITY_SENTINEL",
        "messages": [
            {"role": "user", "content": "I want a film."},
            {"role": "assistant", "content": "HIDDEN_FUTURE_SENTINEL"},
        ],
    }


def test_direct_prompt_exposes_only_shared_visible_context():
    prompt = direct_system_prompt(scenario())
    assert prompt.startswith(ASSISTANT_POLICY)
    assert "Target Film" in prompt
    assert "drama" in prompt
    assert "HIDDEN_PERSONALITY_SENTINEL" not in prompt
    assert "HIDDEN_FUTURE_SENTINEL" not in prompt


def test_direct_planner_makes_exactly_one_final_generation_call():
    backend = FakeGeneration()
    row = scenario()
    planner = DirectLLMPlanner(backend, row, Decoding(max_tokens=128))
    planner.seed_context = f"{row['id']}/profile/0"
    state = DialogueState((Message("user", "I want a film."),))
    plan = planner.act(state, [], 17, "test/direct/turn/0")
    assert len(backend.requests) == 1
    messages, generation = backend.requests[0]
    assert messages[0] == Message("system", direct_system_prompt(row))
    assert messages[1:] == list(state.history)
    assert generation.role == "assistant"
    assert generation.request_id == "test/direct/turn/0/final"
    assert generation.seed == derived_seed(17, planner.seed_context, "final")
    assert plan.response == generation.text
    assert plan.compute == {
        "strategy": "direct-llm",
        "policy_version": DIRECT_LLM_POLICY_VERSION,
        "generation_requests": 1,
        "embedding_requests": 0,
        "retrieval_requests": 0,
        "search_simulations": 0,
        "recovery_triggered": False,
        "planner_latency_s": plan.compute["planner_latency_s"],
    }
    assert plan.replay_ids == [generation.request_id]


def test_v2_prompt_is_exact_tdrl_scaffold_with_empty_guidance():
    row = scenario()
    context = visible_context(row)
    assert direct_v2_system_prompt(row, "TopDial") == (
        ASSISTANT_POLICY + "\n" + context + "\nGuidance: " + target_fact_guidance(context)
    )
    assert direct_v2_system_prompt(row, "DuRecDial2.0") == (
        ASSISTANT_POLICY + "\n" + context + "\nGuidance: "
    )
    assert "Current target facts" in direct_v2_system_prompt(row, "TopDial")
    assert "Current target facts" not in direct_v2_system_prompt(row, "DuRecDial2.0")


def test_v2_planner_is_one_call_and_carries_no_planner_guidance():
    backend = FakeGeneration()
    row = scenario()
    planner = DirectLLMV2Planner(backend, row, "TopDial", Decoding(max_tokens=128))
    state = DialogueState((Message("user", "I want a film."),))
    plan = planner.act(state, [], 17, "test/direct-v2/turn/0")
    assert len(backend.requests) == 1
    assert backend.requests[0][0][0] == Message(
        "system", direct_v2_system_prompt(row, "TopDial")
    )
    assert "\nGuidance: \nCurrent target facts" in planner.system_prompt
    assert plan.compute["policy_version"] == DIRECT_LLM_V2_POLICY_VERSION
    assert plan.compute["generation_requests"] == 1
    assert plan.compute["embedding_requests"] == 0
    assert plan.compute["retrieval_requests"] == 0
    assert plan.compute["search_simulations"] == 0


def test_direct_configs_keep_the_shared_evaluation_axes():
    root = Path(__file__).parents[1]
    full = json.loads((root / "configs/comparison.direct_llm.json").read_text())
    mini = json.loads((root / "configs/comparison.direct_llm-mini.json").read_text())
    full_v2 = json.loads((root / "configs/comparison.direct_llm-v2.json").read_text())
    mini_v2 = json.loads(
        (root / "configs/comparison.direct_llm-v2-mini.json").read_text()
    )
    for config in (full, mini, full_v2, mini_v2):
        validate_config(config)
        assert config["seeds"] == [17]
        assert config["protocols"] == ["profile", "resistant"]
        assert config["turn_budget"] == 10
        assert config["budgets"] == ["natural"]
        assert config["assistant_decoding"] == config["simulator_decoding"]
    assert full["development_scenarios"] == 16
    assert full_v2["development_scenarios"] == 16
    assert mini["development_scenarios"] == 1
    assert mini_v2["development_scenarios"] == 1


def test_plan_serialization_contains_no_hidden_scenario_fields():
    backend = FakeGeneration()
    row = scenario()
    plan = DirectLLMPlanner(backend, row).act(
        DialogueState((Message("user", "I want a film."),)), [], 17, "serialization/turn/0"
    )
    serialized = json.dumps(asdict(plan), sort_keys=True)
    assert "HIDDEN_PERSONALITY_SENTINEL" not in serialized
    assert "HIDDEN_FUTURE_SENTINEL" not in serialized
