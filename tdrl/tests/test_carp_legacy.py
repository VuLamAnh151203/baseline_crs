import json
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

from tdrl.backend import LocalBackend
from tdrl.carp_legacy import (
    build_assistant_turn_prompt,
    build_moderator_prompt,
    build_moderator_repair_prompt,
    build_user_role_description,
    build_user_turn_prompt,
    load_carp_scenarios,
    normalize_role_output,
    parse_yes_no,
    run_carp_dialogue,
)
from tdrl.contracts import Message, ModelProvenance


ROOT = Path(__file__).resolve().parents[1]
CARP_ROOT = ROOT.parent.parent / "carp_www"
DATA_ROOT = CARP_ROOT / "data/eval"


def test_carp_runtime_is_isolated_and_matches_reference_server():
    runtime = json.loads(
        (ROOT / "configs/carp-legacy-runtime.json").read_text(encoding="utf-8")
    )
    assert runtime["mode"] == "carp_legacy_v1"
    assert runtime["generation"] == {
        "port": 18000,
        "gpu_memory_utilization": 0.85,
        "max_model_len": 16384,
        "max_num_seqs": 64,
        "request_concurrency": 2,
    }
    assert runtime["embedding"]["gpu_memory_utilization"] == 0.4
    assert runtime["embedding"]["max_model_len"] == 8192
    assert runtime["embedding"]["max_num_seqs"] == 64


def test_scope_carp_runtime_only_reduces_server_capacity():
    runner = runpy.run_path(str(ROOT / "scripts/carp_legacy/run.py"))
    base = json.loads(
        (ROOT / "configs/carp-legacy-runtime.json").read_text(encoding="utf-8")
    )
    scope = runner["runtime_for_method"](base, "scope_crs_lite")
    assert "scope_crs_lite" in runner["METHODS"]
    assert base["generation"]["gpu_memory_utilization"] == 0.85
    assert base["generation"]["max_num_seqs"] == 64
    assert scope["generation"] == {
        **base["generation"],
        "gpu_memory_utilization": 0.48,
        "max_num_seqs": 4,
    }
    assert scope["embedding"] == base["embedding"]


def _upstream():
    sys.path.insert(0, str(CARP_ROOT / "src"))
    from carp.harness.dataset import Message as CarpMessage
    from carp.harness.dataset import load_scenarios
    from carp.harness.prompts import (
        build_moderator_prompt as upstream_moderator,
        build_moderator_repair_prompt as upstream_repair,
        build_assistant_turn_prompt as upstream_assistant,
        build_user_role_description as upstream_role,
        build_user_turn_prompt as upstream_user,
    )

    return (
        CarpMessage,
        load_scenarios,
        upstream_assistant,
        upstream_role,
        upstream_user,
        upstream_moderator,
        upstream_repair,
    )


def test_prompts_and_normalization_match_carp_source():
    (
        CarpMessage,
        load_scenarios,
        upstream_assistant,
        upstream_role,
        upstream_user,
        upstream_moderator,
        upstream_repair,
    ) = _upstream()
    ours = load_carp_scenarios(DATA_ROOT, "TopDial", 42)[0]
    upstream_rows = load_scenarios(
        DATA_ROOT / "topdial_test200_seed2911.jsonl", dataset_kind="topdial"
    )
    upstream_by_id = {row.id: row for row in upstream_rows}
    theirs = upstream_by_id[ours.id]
    our_history = [Message("assistant", "Hello."), Message("user", "Hi.")]
    their_history = [CarpMessage("assistant", "Hello."), CarpMessage("user", "Hi.")]
    assert build_assistant_turn_prompt(ours, []) == upstream_assistant(theirs, [])
    assert build_user_role_description(ours) == upstream_role(theirs)
    assert build_user_turn_prompt(ours, our_history) == upstream_user(theirs, their_history)
    assert build_moderator_prompt(ours, our_history) == upstream_moderator(theirs, their_history)
    assert build_moderator_repair_prompt(ours, our_history, "maybe") == upstream_repair(
        theirs, their_history, "maybe"
    )
    assert ours.method_scenario()["messages"] == []


def test_assistant_opening_matches_carp_source_for_both_datasets():
    (
        _,
        load_scenarios,
        upstream_assistant,
        _,
        _,
        _,
        _,
    ) = _upstream()
    resources = (
        ("TopDial", "topdial_test200_seed2911.jsonl", "topdial"),
        ("DuRecDial2.0", "durecdial_test200_seed2911.jsonl", "durecdial"),
    )
    for dataset, filename, dataset_kind in resources:
        ours = load_carp_scenarios(DATA_ROOT, dataset, 42)[0]
        upstream_rows = load_scenarios(DATA_ROOT / filename, dataset_kind=dataset_kind)
        theirs = {row.id: row for row in upstream_rows}[ours.id]
        assert build_assistant_turn_prompt(ours, []) == upstream_assistant(theirs, [])


def test_durecdial_target_is_hidden_and_minimal_parser_matches():
    scenario = load_carp_scenarios(DATA_ROOT, "DuRecDial2.0", 42)[0]
    target = scenario.target_item.casefold()
    for key, value in scenario.user_profile.items():
        if key not in {"Age Range", "Name", "Gender", "Residence", "Occupation"}:
            assert target not in {item.strip().casefold() for item in value.split(";")}
    assert normalize_role_output("Assistant: user: Fine") == "Fine"
    assert parse_yes_no("Yes, clearly.") == "yes"
    assert parse_yes_no("yes or no") is None
    assert parse_yes_no("") is None


def test_backend_carp_request_omits_seed_and_sampling_extras(monkeypatch):
    model = ModelProvenance("Qwen/Qwen3.5-9B", "a" * 40, {"w": "b" * 64}, "c" * 64)
    backend = LocalBackend("http://127.0.0.1:18000/v1", model, "generation")
    observed = {}

    def fake_post(endpoint, payload):
        observed.update({"endpoint": endpoint, "payload": payload})
        return (
            {
                "model": model.model_id,
                "choices": [{"message": {"content": "yes"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
            },
            0.1,
        )

    monkeypatch.setattr(backend, "_post", fake_post)
    result = backend.generate_carp_legacy("Judge this", "carp_moderator", 3072, "request")
    assert result.text == "yes"
    assert observed["endpoint"] == "/chat/completions"
    assert observed["payload"] == {
        "model": "Qwen/Qwen3.5-9B",
        "messages": [{"role": "user", "content": "Judge this"}],
        "temperature": 0.0,
        "max_tokens": 3072,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    assistant = backend.generate_carp_legacy(
        "Generate the opening greeting", "carp_assistant", 3072, "assistant-request"
    )
    assert assistant.text == "yes"


def test_empty_history_uses_carp_opening_before_method_core(tmp_path):
    scenario = load_carp_scenarios(DATA_ROOT, "TopDial", 42)[0]

    class Planner:
        calls = 0

        def act(self, *args, **kwargs):
            self.calls += 1
            raise AssertionError("method core must not run before CARP's opening turn")

    class Generation:
        def __init__(self):
            self.requests = []
            self.calls = []

        def generate_carp_legacy(self, prompt, role, max_tokens, request_id):
            self.calls.append((prompt, role, max_tokens, request_id))
            if role == "carp_assistant":
                return SimpleNamespace(text="Assistant: Hello!")
            if role == "carp_user_simulator":
                return SimpleNamespace(text="User: That sounds good.")
            if role == "carp_moderator":
                return SimpleNamespace(text="yes")
            raise AssertionError(f"unexpected role: {role}")

    planner, generation = Planner(), Generation()
    result = run_carp_dialogue(
        scenario,
        planner,
        generation,
        tmp_path / "dialogue",
        {
            "method": "DirectLLM-v2",
            "dataset": "TopDial",
            "request_namespace": "opening-regression",
        },
    )
    assert result["status"] == "complete"
    assert result["transcript"][0] == {"role": "assistant", "text": "Hello!"}
    assert result["turns_trace"][0]["plan"]["compute"]["strategy"] == "carp-legacy-shared-opening"
    assert planner.calls == 0
    assert generation.calls[0][0] == build_assistant_turn_prompt(scenario, [])
    assert [row[1] for row in generation.calls] == [
        "carp_assistant",
        "carp_user_simulator",
        "carp_moderator",
    ]
