import json
from dataclasses import asdict
from pathlib import Path

import pytest
import torch

from tdrl.backend import LocalBackend
from tdrl.baselines.dpdp.checkpoint import load_checkpoint, save_checkpoint
from tdrl.baselines.dpdp.cli import _TinyEncoder, _TinyRollout, _TinyTokenizer, smoke
from tdrl.baselines.dpdp.config import DPDPConfig
from tdrl.baselines.dpdp.contracts import DPDPCheckpointMetadata
from tdrl.baselines.dpdp.dataset import DPDPCollator, build_transitions, prepare_transitions
from tdrl.baselines.dpdp.mcts import OpenLoopMCTS, puct_score
from tdrl.baselines.dpdp.model import DPDPPolicy
from tdrl.baselines.dpdp.planner import DPDPPlanner
from tdrl.baselines.dpdp.rewards import (
    REWARD_VALUES,
    annotate_rewards,
    critic_messages,
    discounted_returns,
    parse_reward_label,
)
from tdrl.baselines.dpdp.rollout import QwenRolloutProvider
from tdrl.baselines.dpdp.selection import validate_selection
from tdrl.baselines.dpdp.serialization import encode_policy_input
from tdrl.baselines.dpdp.training import selfplay_update
from tdrl.comparison_dpdp import validate_config as validate_comparison_config
from tdrl.contracts import Decoding, Generation, Message, ModelProvenance
from tdrl.official_data import ACTION_TAXONOMY
from tdrl.provenance import digest
from tdrl.state import DialogueState
from tdrl.strategies import ACTION_INSTRUCTIONS


def scenario(identifier="sample-1"):
    return {
        "id": identifier,
        "profile": {"likes": "mystery"},
        "personality": {"openness": "curious"},
        "knowledge": [["Moon", "type", "mystery film"]],
        "target": ["Movie recommendation", "Moon"],
        "messages": [
            {"role": "assistant", "content": "What films do you enjoy?"},
            {"role": "user", "content": "Mysteries."},
            {"role": "assistant", "content": "I recommend Moon."},
            {"role": "user", "content": "I will watch Moon."},
        ],
        "assistant_actions": ["ask_preference", None, "recommend", None],
        "action_label_source": "weak_rule_v1",
    }


def metadata(phase="offline"):
    return DPDPCheckpointMetadata(
        phase=phase,
        config_sha256="a" * 64,
        data_sha256="b" * 64,
        action_taxonomy_sha256="c" * 64,
        model_lock_sha256="d" * 64,
        source_revision="4d1e22e35db452d51f0f39388456d310ec1c971b",
    )


def test_config_is_strict_and_matched_budget_keeps_training_signature(tmp_path):
    config = DPDPConfig.smoke()
    matched = config.with_budget(2, 1)
    assert config.sha256 != matched.sha256
    assert config.training_sha256 == matched.training_sha256
    assert config.fitting_sha256 != matched.fitting_sha256
    raw = asdict(config)
    raw["actions"] = list(raw["actions"])
    raw["unknown"] = True
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown/incomplete"):
        DPDPConfig.from_path(path)


def test_comparison_config_matches_shared_seed_judge_and_test_subset():
    root = Path(__file__).resolve().parents[1]
    dpdp = json.loads((root / "configs/comparison.dpdp.json").read_text())
    shared = json.loads((root / "configs/comparison-judge-v2-sample500-seed17.json").read_text())
    validate_comparison_config(dpdp)
    for field in (
        "datasets",
        "seeds",
        "protocols",
        "turn_budget",
        "assistant_decoding",
        "simulator_decoding",
        "judge",
        "test_sampling",
    ):
        assert dpdp[field] == shared[field]


def test_policy_serialization_keeps_complete_recent_turns_and_header():
    tokenizer = _TinyTokenizer()
    state = [Message("user", "old" * 20), Message("assistant", "new"), Message("user", "latest")]
    ids, mask = encode_policy_input(
        tokenizer,
        {"profile": {"likes": "mystery"}, "target": ["recommend", "Moon"]},
        state,
        max_tokens=64,
        header_tokens=16,
    )
    assert ids.shape == mask.shape
    assert len(ids) <= 64
    assert ids[0] == tokenizer.cls_token_id and ids[-1] == tokenizer.sep_token_id


def test_data_adapter_uses_only_observed_prefix_and_stable_action_index(tmp_path):
    rows = build_transitions(scenario(), "TopDial", "train", "a" * 64)
    assert len(rows) == 1
    row = rows[0]
    assert row.action_name == "recommend"
    assert row.action_index == ACTION_TAXONOMY.index("recommend")
    assert [message.content for message in row.state] == [
        "What films do you enjoy?",
        "Mysteries.",
    ]
    assert row.next_state[-1].content == "I will watch Moon."
    assert row.done
    source = tmp_path / "train.jsonl"
    source.write_text(json.dumps(scenario()) + "\n", encoding="utf-8")
    output = tmp_path / "prepared.jsonl"
    manifest = prepare_transitions(source, output, "TopDial", "train")
    assert manifest["transition_count"] == 1
    assert manifest["test_bodies"] == "never_opened"
    with pytest.raises(FileExistsError):
        prepare_transitions(source, output, "TopDial", "train")


def test_reward_parser_and_discounted_returns_fail_closed():
    assert parse_reward_label(" accepted\n") == "accepted"
    assert REWARD_VALUES["rejected"] == -1.0
    with pytest.raises(ValueError, match="malformed"):
        parse_reward_label("probably interested")
    with pytest.raises(ValueError, match="malformed"):
        parse_reward_label("ACCEPTED")
    assert discounted_returns([0.1, 1.0], [False, True], 0.5) == pytest.approx([0.6, 1.0])
    assert discounted_returns([1.0, -1.0], [True, True], 0.9) == pytest.approx([1.0, -1.0])


def test_critic_prompt_wraps_dialogue_as_classification_data():
    messages = critic_messages(
        {"target": ["Movie recommendation", "Moon"], "profile": {}},
        [Message("user", "What do you suggest?")],
    )
    assert [message.role for message in messages] == ["system", "user"]
    case = json.loads(messages[-1].content)
    assert case["dialogue"] == [{"role": "user", "content": "What do you suggest?"}]
    assert "Classify" in case["task"]


def test_dpdp_critic_uses_vllm_choice_constraint(monkeypatch):
    model = ModelProvenance("Qwen/Qwen3.5-9B", "a" * 40, {"weights": "b" * 64}, "c" * 64)
    backend = LocalBackend("http://127.0.0.1:18000/v1", model, "generation")
    observed = {}

    def post(endpoint, payload):
        observed.update(payload)
        return (
            {
                "model": model.model_id,
                "choices": [{"message": {"content": "interested"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 4, "completion_tokens": 1, "total_tokens": 5},
            },
            0.01,
        )

    monkeypatch.setattr(backend, "_post", post)
    backend.generate(
        [Message("system", "classify"), Message("user", "{}")],
        "dpdp_rollout_critic",
        Decoding(max_tokens=8),
        17,
        "critic/request",
    )
    assert observed["structured_outputs"] == {"choice": list(REWARD_VALUES)}


class _RevisingRolloutBackend:
    def __init__(self):
        self.critic_messages = []
        self.assistant_messages = []

    def generate(self, messages, role, decoding, seed, request_id):
        if role == "dpdp_rollout_assistant":
            self.assistant_messages.append(messages)
            text = "I recommend Moon."
        elif role == "dpdp_rollout_user":
            text = "That sounds interesting."
        else:
            self.critic_messages.append(messages)
            text = "invalid label" if request_id.endswith("attempt/0") else "interested"
        return Generation(
            text,
            4,
            1,
            5,
            seed,
            decoding,
            0.01,
            0,
            ModelProvenance("fake", "a" * 40, {"w": "b" * 64}, "c" * 64),
            "d" * 64,
            request_id,
            role,
        )


def test_rollout_critic_revises_invalid_label_with_bounded_request_id():
    backend = _RevisingRolloutBackend()
    provider = QwenRolloutProvider(backend, scenario(), "profile", 17, "rollout", 1)
    sample = provider.expand(
        DialogueState((Message("user", "I like mysteries."),)),
        4,
        "recommend",
        ("recommend",),
        0,
        0,
    )
    assert sample.label == "interested"
    assert ACTION_INSTRUCTIONS["recommend"] in backend.assistant_messages[0][0].content
    assert "hard constraint" in backend.assistant_messages[0][0].content
    assert sample.critic_labels == ("interested",)
    assert not sample.deterministic_accepted
    assert not sample.acceptance_disagreement
    assert sample.request_ids[-2:] == (
        "rollout/sim/0/depth/0/path/recommend/critic/0/attempt/0",
        "rollout/sim/0/depth/0/path/recommend/critic/0/attempt/1",
    )
    revised = json.loads(backend.critic_messages[-1][-1].content)
    assert "revision" in revised


class _AcceptedCriticBackend(_RevisingRolloutBackend):
    def generate(self, messages, role, decoding, seed, request_id):
        generated = super().generate(messages, role, decoding, seed, request_id)
        if role != "dpdp_rollout_critic":
            return generated
        return Generation(
            "accepted",
            generated.input_tokens,
            generated.output_tokens,
            generated.total_tokens,
            generated.seed,
            generated.decoding,
            generated.latency_s,
            generated.retries,
            generated.model,
            generated.prompt_sha256,
            generated.request_id,
            generated.role,
        )


def test_rollout_critic_acceptance_terminates_and_records_classifier_disagreement():
    provider = QwenRolloutProvider(
        _AcceptedCriticBackend(), scenario(), "profile", 17, "rollout", 1
    )
    sample = provider.expand(
        DialogueState((Message("user", "I like mysteries."),)),
        4,
        "recommend",
        ("recommend",),
        0,
        0,
    )
    assert sample.label == "accepted"
    assert sample.terminal
    assert not sample.deterministic_accepted
    assert sample.acceptance_disagreement
    assert sample.terminal_reasons == ("critic_accepted",)


class _CriticBackend:
    def __init__(self):
        self.requests = []

    def generate(self, messages, role, decoding, seed, request_id):
        del messages
        result = Generation(
            text="interested",
            input_tokens=4,
            output_tokens=1,
            total_tokens=5,
            seed=seed,
            decoding=decoding,
            latency_s=0.01,
            retries=0,
            model=ModelProvenance("fake", "a" * 40, {"w": "b" * 64}, "c" * 64),
            prompt_sha256="d" * 64,
            request_id=request_id,
            role=role,
        )
        self.requests.append(result)
        return result


class _ConcurrentCriticBackend(_CriticBackend):
    request_concurrency = 2

    def __init__(self):
        super().__init__()
        self.batches = []

    def generate_many(self, requests, max_workers):
        requests = list(requests)
        self.batches.append(([request[4] for request in requests], max_workers))
        return [self.generate(*request) for request in requests]


def test_reward_annotation_is_resumable_and_does_not_cross_dialogues(tmp_path):
    prepared = tmp_path / "prepared.jsonl"
    rows = []
    for name in ("a", "b"):
        transition = build_transitions(scenario(name), "TopDial", "train", "a" * 64)[0]
        rows.append(transition.to_dict())
    prepared.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    output = tmp_path / "annotated.jsonl"
    backend = _CriticBackend()
    report = annotate_rewards(prepared, output, backend, 2, 0.999, 17, tmp_path / "journal")
    annotated = [json.loads(line) for line in output.read_text().splitlines()]
    assert report["transition_count"] == 2
    assert len(backend.requests) == 4
    assert [row["discounted_return"] for row in annotated] == pytest.approx([0.1, 0.1])


def test_reward_annotation_uses_ordered_bounded_critic_batches(tmp_path):
    prepared = tmp_path / "prepared.jsonl"
    transition = build_transitions(scenario(), "TopDial", "train", "a" * 64)[0]
    prepared.write_text(json.dumps(transition.to_dict()) + "\n", encoding="utf-8")
    backend = _ConcurrentCriticBackend()
    report = annotate_rewards(
        prepared,
        tmp_path / "annotated.jsonl",
        backend,
        4,
        0.999,
        17,
        tmp_path / "journal",
    )
    assert report["request_concurrency"] == 2
    assert len(backend.batches) == 1
    request_ids, workers = backend.batches[0]
    assert workers == 2
    assert request_ids == [f"dpdp/reward/train/sample-1:2/{index}/attempt/0" for index in range(4)]


def test_local_generation_batch_records_source_order(monkeypatch):
    backend = object.__new__(LocalBackend)
    backend.request_concurrency = 2
    backend.requests = []
    backend.consumed_requests = []
    monkeypatch.setattr(
        backend,
        "_generate_unrecorded",
        lambda messages, role, decoding, seed, request_id: (request_id, False),
    )
    requests = [(None, None, None, index, f"request-{index}") for index in range(4)]
    assert backend.generate_many(requests) == [f"request-{index}" for index in range(4)]
    assert backend.requests == [f"request-{index}" for index in range(4)]
    assert backend.consumed_requests == [f"request-{index}" for index in range(4)]


def _model(dropout=0.0):
    return DPDPPolicy(_TinyEncoder(), 32, tuple(ACTION_TAXONOMY), dropout)


def test_actor_q_forward_loss_and_detached_advantage():
    torch.manual_seed(17)
    model = _model()
    ids = torch.tensor([[0, 3, 1], [0, 4, 1]])
    mask = torch.ones_like(ids)
    actions = torch.tensor([1, 4])
    returns = torch.tensor([0.2, 1.0])
    output = model(ids, mask, actions, returns, critic_weight=0.0)
    assert output["policy_logits"].shape == (2, 6)
    assert output["q_values"].shape == (2, 6)
    assert torch.allclose(output["action_probs"].sum(-1), torch.ones(2))
    expected_actor = (
        -torch.log_softmax(output["policy_logits"], -1).gather(1, actions[:, None]).squeeze(1)
        * (returns - output["selected_q"]).detach()
    ).mean()
    assert torch.allclose(output["actor_loss"], expected_actor)
    output["total_loss"].backward()
    critic_grad = [parameter.grad for parameter in model.critic.parameters()]
    assert all(gradient is None or torch.count_nonzero(gradient) == 0 for gradient in critic_grad)
    assert any(parameter.grad is not None for parameter in model.encoder.parameters())


def test_critic_and_encoder_receive_finite_gradients():
    model = _model()
    ids = torch.tensor([[0, 3, 1], [0, 4, 1]])
    output = model(ids, torch.ones_like(ids), torch.tensor([0, 1]), torch.tensor([0.1, 1.0]))
    output["total_loss"].backward()
    assert any(parameter.grad is not None for parameter in model.critic.parameters())
    assert all(
        torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
        if parameter.grad is not None
    )


def test_checkpoint_restores_exact_cursor_and_weights(tmp_path):
    model = _model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    generator = torch.Generator().manual_seed(29)
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, model, optimizer, scheduler, 7, 2, 4, metadata(), generator)
    expected = next(model.parameters()).detach().clone()
    with torch.no_grad():
        next(model.parameters()).add_(1)
    payload = load_checkpoint(
        path, model, metadata(), optimizer, scheduler, generator, map_location="cpu"
    )
    assert (payload["step"], payload["epoch"], payload["batch_in_epoch"]) == (7, 2, 4)
    assert torch.equal(next(model.parameters()), expected)
    with pytest.raises(ValueError, match="mismatch"):
        load_checkpoint(path, model, metadata("selfplay"))


def test_puct_and_incremental_backup_are_deterministic():
    assert puct_score(0.2, 0.5, 4, 1, 1.0) == pytest.approx(0.7)
    state = DialogueState((Message("user", "hello"),))
    priors = [0.01] * 6
    priors[4] = 0.95
    search = OpenLoopMCTS(
        tuple(ACTION_TAXONOMY),
        lambda _: (priors, [0.0] * 6),
        _TinyRollout(),
        simulations=4,
        cpuct=1.0,
        q0=0.0,
        max_depth=2,
        max_realizations=1,
    )
    result = search.search(state)
    assert sum(result.root_visits) == 4
    assert result.action_index == 4
    assert result.action_name == "recommend"
    assert result.reward_label == "interested"
    assert result.critic_labels == ("interested",)
    assert result.expanded_nodes >= 2
    assert result.root_q_values[4] == pytest.approx(0.1075)


def test_trees_do_not_share_state_and_minimum_simulations_is_enforced():
    args = (tuple(ACTION_TAXONOMY), lambda _: ([1 / 6] * 6, [0.0] * 6), _TinyRollout())
    with pytest.raises(ValueError, match="budget"):
        OpenLoopMCTS(*args, 1, 1.0, 0.0, 2, 1)
    left = OpenLoopMCTS(*args, 2, 1.0, 0.0, 1, 1)
    right = OpenLoopMCTS(*args, 2, 1.0, 0.0, 1, 1)
    state = DialogueState((Message("user", "hello"),))
    left.search(state)
    assert right.nodes == {}


class _PlannerBackend:
    def __init__(self):
        self.requests = []

    def generate(self, messages, role, decoding, seed, request_id):
        assert "DPDP next action" in messages[0].content
        assert ACTION_INSTRUCTIONS["recommend"] in messages[0].content
        assert "hard constraint" in messages[0].content
        result = Generation(
            "Would you like to try Moon?",
            10,
            7,
            17,
            seed,
            decoding,
            0.01,
            0,
            ModelProvenance("fake", "a" * 40, {"w": "b" * 64}, "c" * 64),
            "d" * 64,
            request_id,
            role,
        )
        self.requests.append(result)
        return result


def test_system1_planner_uses_shared_generation_and_no_embedding():
    config = DPDPConfig.smoke()
    config = type(config)(
        **{**asdict(config), "actions": tuple(config.actions), "gate_threshold": 0.0}
    )
    tokenizer = _TinyTokenizer()
    model = _model()
    with torch.no_grad():
        model.actor.weight.zero_()
        model.actor.bias.copy_(torch.tensor([0.0, 0.0, 0.0, 0.0, 5.0, 0.0]))
    backend = _PlannerBackend()
    planner = DPDPPlanner(
        backend,
        model,
        tokenizer,
        config,
        scenario(),
        "e" * 64,
        Decoding(max_tokens=64),
    )
    plan = planner.act(DialogueState((Message("user", "I like mysteries."),)), [], 17, "req")
    assert plan.response == "Would you like to try Moon?"
    assert plan.compute["selected_system"] == "system1"
    assert plan.compute["selected_action"] == "recommend"
    assert plan.compute["embedding_requests"] == 0


def test_selfplay_update_uses_only_executed_action_batch():
    model = _model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    batch = DPDPCollator(2)(
        [
            {
                "transition_id": "executed",
                "input_ids": torch.tensor([0, 3, 1]),
                "attention_mask": torch.ones(3, dtype=torch.long),
                "action": 4,
                "return": 1.0,
            }
        ]
    )
    result = selfplay_update(model, optimizer, batch, 1.0, 1.0)
    assert set(result) == {"total_loss", "actor_loss", "critic_loss"}
    assert all(torch.isfinite(torch.tensor(value)) for value in result.values())


def test_cpu_smoke_covers_policy_mcts_and_selfplay(tmp_path):
    report = smoke(tmp_path)
    assert report["status"] == "smoke_non_scientific"
    assert sum(report["mcts_root_visits"]) == DPDPConfig.smoke().mcts_simulations
    assert (tmp_path / "smoke.json").is_file()


def test_selection_lock_validation_fails_closed():
    with pytest.raises(ValueError, match="unknown/incomplete"):
        validate_selection({"schema_version": 1, "config_sha256": digest("x")})
