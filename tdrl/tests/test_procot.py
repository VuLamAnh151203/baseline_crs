import json
from dataclasses import asdict
from pathlib import Path

from tdrl.baselines.procot.parser import parse_procot
from tdrl.baselines.procot.planner import ProCoTPlanner
from tdrl.baselines.procot.prompts import (
    PROCOT_FINAL_MARKER,
    PROCOT_POLICY,
    PROCOT_POLICY_VERSION,
    procot_system_prompt,
)
from tdrl.baselines.procot.selection import prompt_contract, verify_reference_lock
from tdrl.comparison_procot import validate_config
from tdrl.contracts import Decoding, Generation, Message, ModelProvenance
from tdrl.planner import derived_seed
from tdrl.provenance import digest
from tdrl.state import DialogueState


class FakeGeneration:
    def __init__(self, text):
        self.text = text
        self.requests = []

    def generate(self, messages, role, decoding, seed, request_id):
        result = Generation(
            text=self.text,
            input_tokens=30,
            output_tokens=20,
            total_tokens=50,
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
        "id": "procot-test",
        "target": ["Movie recommendation", "Target Film"],
        "profile": {"genre": "drama"},
        "knowledge": [["Target Film", "genre", "drama"]],
        "personality": "HIDDEN_PERSONALITY_SENTINEL",
        "messages": [
            {"role": "user", "content": "I want a film."},
            {"role": "assistant", "content": "HIDDEN_FUTURE_SENTINEL"},
        ],
    }


def valid_output():
    return (
        'The current topics are ["films"]. To bridge the current topics with the '
        'target goal, the next topics are ["drama"]. Based on the predicted next '
        'topics, the response is "Do you enjoy thoughtful drama films?"'
    )


def test_procot_parser_extracts_only_the_final_response():
    parsed = parse_procot(valid_output())
    assert parsed.response == "Do you enjoy thoughtful drama films?"
    assert parsed.current_topics == '["films"]'
    assert parsed.next_topics == '["drama"]'
    assert parsed.format_valid is True
    assert PROCOT_FINAL_MARKER not in parsed.response
    assert digest(parsed.planning_text)


def test_procot_parser_preserves_malformed_output_as_observable_failure():
    raw = "A direct answer without the required ProCoT fields."
    parsed = parse_procot(raw)
    assert parsed.response == raw
    assert parsed.planning_text == ""
    assert parsed.current_topics is None
    assert parsed.next_topics is None
    assert parsed.format_valid is False


def test_procot_prompt_exposes_only_shared_visible_context():
    prompt = procot_system_prompt(scenario())
    assert prompt.startswith(PROCOT_POLICY)
    assert "Return only your final conversational response" not in prompt
    assert "Target Film" in prompt
    assert "drama" in prompt
    assert "HIDDEN_PERSONALITY_SENTINEL" not in prompt
    assert "HIDDEN_FUTURE_SENTINEL" not in prompt


def test_procot_planner_makes_one_call_and_executes_only_final_response():
    backend = FakeGeneration(valid_output())
    row = scenario()
    planner = ProCoTPlanner(backend, row, Decoding(max_tokens=128))
    planner.seed_context = f"{row['id']}/profile/0"
    state = DialogueState((Message("user", "I want a film."),))
    plan = planner.act(state, [], 17, "test/procot/turn/0")
    assert len(backend.requests) == 1
    messages, generation = backend.requests[0]
    assert messages[0] == Message("system", procot_system_prompt(row))
    assert messages[1:] == list(state.history)
    assert generation.role == "assistant"
    assert generation.request_id == "test/procot/turn/0/procot"
    assert generation.seed == derived_seed(17, planner.seed_context, "final")
    assert plan.response == "Do you enjoy thoughtful drama films?"
    assert plan.compute["strategy"] == "procot-one-shot"
    assert plan.compute["policy_version"] == PROCOT_POLICY_VERSION
    assert plan.compute["format_valid"] is True
    assert plan.compute["generation_requests"] == 1
    assert plan.compute["embedding_requests"] == 0
    assert plan.compute["retrieval_requests"] == 0
    assert plan.compute["search_simulations"] == 0
    assert plan.replay_ids == [generation.request_id]


def test_procot_planner_records_malformed_output_without_hidden_recovery():
    backend = FakeGeneration("Only a direct response.")
    planner = ProCoTPlanner(backend, scenario())
    plan = planner.act(
        DialogueState((Message("user", "I want a film."),)),
        [],
        17,
        "test/procot/malformed/turn/0",
    )
    assert plan.response == "Only a direct response."
    assert plan.compute["format_valid"] is False
    assert plan.compute["format_violation"] is True
    assert plan.compute["generation_requests"] == 1
    assert plan.compute["recovery_triggered"] is False
    assert len(backend.requests) == 1


def test_procot_contract_discloses_observable_malformed_output_handling():
    contract = prompt_contract()
    assert contract["policy_version"] == "adapted-target-goal-one-shot-v3"
    assert contract["response_parser"] == "final-marker-deterministic-observable-v3"
    assert (
        contract["malformed_output"]
        == "execute_raw_and_record_violation_without_retry"
    )


def test_procot_configs_keep_the_shared_evaluation_axes():
    root = Path(__file__).parents[1]
    full = json.loads((root / "configs/comparison.procot.json").read_text())
    mini = json.loads((root / "configs/comparison.procot-mini.json").read_text())
    validate_config(full)
    validate_config(mini)
    for config in (full, mini):
        assert config["seeds"] == [17]
        assert config["protocols"] == ["profile", "resistant"]
        assert config["turn_budget"] == 10
        assert config["budgets"] == ["natural"]
        assert config["assistant_decoding"] == config["simulator_decoding"]
    assert full["development_scenarios"] == 16
    assert mini["development_scenarios"] == 1


def test_procot_reference_lock_is_exactly_pinned():
    root = Path(__file__).parents[1]
    lock = verify_reference_lock(root / "configs/procot-reference.lock.json")
    assert lock["revision"] == "77ff4ed7dfda11afd719db3bd40886a1a51edf4e"
    assert len(lock["files"]) == 3


def test_plan_serialization_contains_no_hidden_scenario_fields():
    backend = FakeGeneration(valid_output())
    row = scenario()
    plan = ProCoTPlanner(backend, row).act(
        DialogueState((Message("user", "I want a film."),)), [], 17, "serialization/turn/0"
    )
    serialized = json.dumps(asdict(plan), sort_keys=True)
    assert "HIDDEN_PERSONALITY_SENTINEL" not in serialized
    assert "HIDDEN_FUTURE_SENTINEL" not in serialized
