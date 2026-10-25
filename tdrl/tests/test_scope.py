import json
import sys
from pathlib import Path

import pytest
import torch

from scripts.scope.run_e2e import ScopeE2E
from tdrl.baselines.scope import semantic
from tdrl.baselines.scope.config import ScopeConfig
from tdrl.baselines.scope.dataset import RewardDataset, prefix_potentials, prepare_split
from tdrl.baselines.scope.reward import GoalConditionedReward
from tdrl.baselines.scope.selection import prompt_contract, verify_reference_lock
from tdrl.baselines.scope.semantic import alternating_chat, target_messages
from tdrl.comparison_scope import validate_config
from tdrl.contracts import Message


def scenario():
    return {
        "id": "scope-test",
        "target": ["Movie recommendation", "Target Film"],
        "profile": {"genre": "drama"},
        "knowledge": [["Target Film", "genre", "drama"]],
        "personality": "HIDDEN_PERSONALITY_SENTINEL",
        "messages": [],
        "assistant_actions": [],
    }


def test_scope_configs_keep_native_planning_and_shared_axes():
    root = Path(__file__).parents[1]
    comparison = json.loads((root / "configs/comparison.scope-lite.json").read_text())
    mini = json.loads(
        (root / "configs/comparison.scope-lite-mini.json").read_text()
    )
    validate_config(comparison)
    validate_config(mini)
    for name in ("topdial", "durecdial"):
        config = ScopeConfig.from_path(root / f"configs/scope/{name}.json")
        assert config.candidates == 5
        assert config.search_depth == 6
        assert config.mcts_seconds == 3.0
        assert config.embedding_dim == 1024
        assert config.transition_samples == 4
    for config in (comparison, mini):
        assert config["seeds"] == [17]
        assert config["protocols"] == ["profile", "resistant"]
        assert config["budgets"] == ["natural"]


def test_scope_e2e_passes_only_supported_stage_arguments(tmp_path):
    root = Path(__file__).parents[1]
    pipeline = ScopeE2E(
        root,
        tmp_path / "run",
        Path(sys.executable),
        "mini",
        tmp_path / "storage",
    )
    smoke = pipeline.smoke_asset_args("TopDial")
    assert "--transitions" in smoke
    assert "--run-root" not in smoke
    assert "--comparison-config" not in smoke
    preparation = pipeline.reward_preparation_args("TopDial")
    assert "--run-root" in preparation
    assert "--comparison-config" in preparation
    assert "--transitions" not in preparation


def test_scope_reference_and_prompt_contract_are_pinned():
    root = Path(__file__).parents[1]
    lock = verify_reference_lock(root / "configs/scope-reference.lock.json")
    assert lock["revision"] == "b35dbc64808afce25e8f8e5d6838ecf271ec4d0b"
    contract = prompt_contract()
    assert contract["candidate_calls_per_executed_turn"] == 5
    assert contract["shared_qwen_embedding_calls_per_turn"] == 0
    assert "future_messages" in contract["hidden_fields"]


def test_llama_guard_chat_format_does_not_mutate_input():
    original = [Message("assistant", "Hello"), Message("user", "Hi")]
    formatted = alternating_chat(original)
    assert original == [Message("assistant", "Hello"), Message("user", "Hi")]
    assert formatted == [
        {"role": "user", "content": ""},
        {"role": "assistant", "content": "Hello"},
        {"role": "user", "content": "Hi"},
    ]
    with pytest.raises(ValueError, match="alternating"):
        alternating_chat([Message("user", "one"), Message("user", "two")])


def test_chat_template_accepts_batch_encoding_and_tensor_variants():
    expected = torch.tensor([[1, 2, 3]])

    class BatchTokenizer:
        @staticmethod
        def apply_chat_template(_chat, return_tensors):
            assert return_tensors == "pt"
            return {"input_ids": expected.clone()}

    class TensorTokenizer:
        @staticmethod
        def apply_chat_template(_chat, return_tensors):
            assert return_tensors == "pt"
            return expected.clone()

    for tokenizer in (BatchTokenizer(), TensorTokenizer()):
        actual = semantic._chat_template_input_ids(
            tokenizer, [{"role": "user", "content": "hello"}], torch.device("cpu")
        )
        assert torch.equal(actual, expected)

    class InvalidTokenizer:
        @staticmethod
        def apply_chat_template(_chat, return_tensors):
            return {"attention_mask": expected}

    with pytest.raises(ValueError, match="input_ids"):
        semantic._chat_template_input_ids(
            InvalidTokenizer(),
            [{"role": "user", "content": "hello"}],
            torch.device("cpu"),
        )


def test_llama_guard_load_does_not_require_accelerate(monkeypatch):
    calls = {}

    class Model:
        def to(self, device):
            calls["device"] = device
            return self

        def eval(self):
            calls["eval"] = True
            return self

    class Factory:
        @staticmethod
        def from_pretrained(snapshot, **kwargs):
            calls["snapshot"] = snapshot
            calls["kwargs"] = kwargs
            return Model()

    monkeypatch.setattr(semantic, "AutoModelForCausalLM", Factory)
    result = semantic._load_local_causal_lm("local-snapshot", torch.device("cuda"))
    assert isinstance(result, Model)
    assert calls["snapshot"] == "local-snapshot"
    assert calls["kwargs"] == {
        "local_files_only": True,
        "dtype": torch.bfloat16,
    }
    assert calls["device"] == torch.device("cuda")
    assert calls["eval"] is True


def test_target_embedding_exposes_only_allowed_fields():
    row = scenario()
    serialized = target_messages(row)[0].content
    assert "Target Film" in serialized
    assert "drama" in serialized
    assert "HIDDEN_PERSONALITY_SENTINEL" not in serialized


def test_task_reward_is_goal_conditioned_and_finite():
    torch.manual_seed(17)
    reward = GoalConditionedReward(8, (4,))
    state = torch.randn(3, 8)
    goal = torch.randn(3, 8)
    result = reward(state, goal)
    assert result.shape == (3,)
    assert torch.isfinite(result).all()
    assert ((0 <= result) & (result <= 1)).all()


def test_prefix_labels_use_action_progress_and_terminal_judgment():
    row = scenario()
    row["target"] = ["Movie recommendation", "Target Film"]
    row["assistant_actions"] = [None, "recommend", None]
    messages = (
        Message("user", "I want a film"),
        Message("assistant", "Try Target Film"),
        Message("user", "That sounds good"),
    )

    class Judgment:
        progress = "achieved"

    labels = prefix_potentials(row, messages, Judgment())
    assert labels[0] == 0.05
    assert labels[1] == 0.70
    assert labels[2] == 1.0


def test_reward_data_contains_only_user_states_and_counterfactuals(tmp_path):
    rows = []
    for index in range(2):
        rows.append(
            {
                **scenario(),
                "id": f"row-{index}",
                "target": ["Movie recommendation", f"Film {index}"],
                "messages": [
                    {"role": "assistant", "content": "What do you like?"},
                    {"role": "user", "content": "Drama."},
                    {"role": "assistant", "content": f"Try Film {index}."},
                    {"role": "user", "content": "Sounds good."},
                ],
                "assistant_actions": ["ask_preference", None, "recommend", None],
            }
        )
    source = tmp_path / "train.jsonl"
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))

    class Encoder:
        def embed(self, _messages, request_id):
            offset = int(request_id[-1]) if request_id[-1].isdigit() else 0
            return torch.arange(1024, dtype=torch.float32) + offset

    output = tmp_path / "prepared"
    manifest = prepare_split(
        source,
        output,
        "TopDial",
        "train",
        Encoder(),
        None,
        17,
        counterfactual_negatives=1,
    )
    dataset = RewardDataset(output)
    assert manifest["dialogues"] == 2
    assert len(dataset) == 8  # 2 rows x 2 user states x (positive + negative).
    for shard in (output / "shards").glob("*.pt"):
        payload = torch.load(shard, map_location="cpu", weights_only=True)
        assert payload["state_role"] == "user"
        assert payload["positive_prefixes"] == 2
