import json
import zipfile
from dataclasses import asdict
from types import SimpleNamespace

import pytest
import torch

from tdrl.baselines.trip.checkpoint import load_checkpoint, save_checkpoint
from tdrl.baselines.trip.cli import _TinyEncoder, _TinyTokenizer
from tdrl.baselines.trip.config import DecodingConfig, TripConfig
from tdrl.baselines.trip.dataset import TripCollator
from tdrl.baselines.trip.decoding import DecodedPlan, _parse_pairs, generate_plan
from tdrl.baselines.trip.model import TripModel
from tdrl.baselines.trip.planner import TripPlanner
from tdrl.baselines.trip.serialization import _examples, prepare_corpus
from tdrl.baselines.trip.selection import validate_selection
from tdrl.contracts import Decoding, Generation, Message, ModelProvenance
from tdrl.provenance import digest
from tdrl.state import DialogueState


def topdial_record(identifier="sample-1"):
    return {
        "id": identifier,
        "profile": {"likes": "mystery"},
        "personality": {"openness": "curious"},
        "knowledge": [
            ["Moon", "type", "mystery film"],
            ["Sun", "type", "drama"],
            ["Stars", "type", "comedy"],
            ["Cloud", "type", "thriller"],
        ],
        "target": ["Movie recommendation", "Moon"],
        "messages": [
            {"role": "assistant", "content": "What films do you enjoy?"},
            {"role": "user", "content": "Mysteries."},
            {"role": "assistant", "content": "I recommend Moon, a mystery film."},
            {"role": "user", "content": "Thanks."},
        ],
        "assistant_actions": ["ask_preference", None, "recommend", None],
    }


def test_topdial_weak_paths_use_only_prefix_and_end_at_target():
    rows = _examples(topdial_record(), "TopDial")
    assert len(rows) == 2
    assert rows[0]["messages"] == []
    assert rows[1]["messages"][-1]["content"] == "Mysteries."
    assert rows[1]["action_path"][-1] == "Movie recommendation"
    assert rows[1]["topic_path"][-1] == "Moon"
    assert rows[0]["label_source"] == "weak_path_v1"


def test_durecdial_preparation_recovers_goal_markers_without_test_data(tmp_path):
    record = {
        "id": "train:en:000000",
        "profile": {},
        "personality": {},
        "knowledge": [
            ["Moon", "type", "mystery"],
            ["Sun", "type", "drama"],
            ["Star", "type", "comedy"],
            ["Cloud", "type", "thriller"],
        ],
        "target": ["Movie recommendation", "Moon"],
        "messages": [
            {"role": "assistant", "content": "Hello"},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Try Moon"},
        ],
        "assistant_actions": ["acknowledge", None, "recommend"],
    }
    source = {
        "goal": "[1]Greetings-->[2]Q&A(Movie)-->[3]Movie recommendation(Moon)-->[4]Say goodbye",
        "conversation": ["[1] Hello", "[2] Hi", "[3] Try Moon"],
    }
    canonical = tmp_path / "train.jsonl"
    canonical.write_text(json.dumps(record) + "\n", encoding="utf-8")
    archive = tmp_path / "data.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("data/en_train.txt", json.dumps(source) + "\n")
    output = tmp_path / "prepared.jsonl"
    report = prepare_corpus(canonical, output, "DuRecDial2.0", "train", archive, 3)
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert report["label_source"] == "official_goal_path_v1"
    assert rows[-1]["action_path"] == ["Movie recommendation"]
    assert rows[-1]["topic_path"] == ["Moon"]
    assert all("Say goodbye" not in row["action_path"] for row in rows)


def _batch():
    target_mask = [0, 1, 1, 1, 1, 0]
    row = {
        "id": "x",
        "context_ids": [1, 2, 3],
        "target_ids": [1, 4, 8, 5, 9],
        "forward_ids": [6, 4, 8, 5, 9, 7],
        "forward_target_mask": target_mask,
        "backward_ids": [6, 4, 8, 5, 9, 7],
        "backward_target_mask": target_mask,
        "negative_forward": [
            ([6, 4, 8, 5, 10, 7], target_mask),
            ([6, 4, 8, 5, 11, 7], target_mask),
        ],
        "negative_backward": [
            ([6, 4, 8, 5, 10, 7], target_mask),
            ([6, 4, 8, 5, 11, 7], target_mask),
        ],
    }
    return TripCollator(0)([row, {**row, "id": "y"}])


def test_trip_three_part_loss_is_finite_and_backpropagates():
    config = TripConfig.smoke()
    model = TripModel(
        _TinyEncoder(32, 32), _TinyEncoder(32, 32), 32, 0, config.model
    )
    output = model(_batch())
    assert set(("lm_loss", "distance_loss", "contrastive_loss")) <= set(output)
    assert all(torch.isfinite(output[key]) for key in ("loss", "lm_loss", "distance_loss"))
    output["loss"].backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_bidirectional_decoding_returns_target_pair():
    torch.manual_seed(17)
    config = TripConfig.smoke()
    tokenizer = _TinyTokenizer()
    model = TripModel(
        _TinyEncoder(32, 32), _TinyEncoder(32, 32), 32, 0, config.model
    )
    plan = generate_plan(
        model,
        tokenizer,
        tokenizer.ids,
        [1, 2, 3],
        [1, 4, 8, 5, 9],
        [4, 8, 5, 9],
        config.decoding,
    )
    assert ("tok8", "tok9") in plan.pairs
    assert plan.forward_candidates and plan.backward_candidates
    assert plan.sequence_forwards > 0


def test_backward_parser_reverses_action_topic_path():
    tokenizer = _TinyTokenizer()
    ids = tokenizer.ids
    sequence = [ids["[BOS]"], ids["[A]"], 8, ids["[T]"], 9, ids["[A]"], 10, ids["[T]"], 11]
    assert _parse_pairs(sequence, tokenizer, ids) == (("tok10", "tok11"), ("tok8", "tok9"))


def test_trip_config_rejects_mismatched_hidden_heads(tmp_path):
    raw = asdict(TripConfig.smoke())
    raw["model"]["hidden_size"] = 31
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="divisible"):
        TripConfig.from_path(path)


def test_natural_and_matched_decoding_share_training_signature():
    natural = TripConfig.smoke()
    matched = TripConfig(
        schema_version=natural.schema_version,
        model=natural.model,
        training=natural.training,
        decoding=DecodingConfig(beam_size=1, max_plan_tokens=16),
        max_context_tokens=natural.max_context_tokens,
        max_plan_tokens=natural.max_plan_tokens,
    )
    assert natural.sha256 != matched.sha256
    assert natural.training_sha256 == matched.training_sha256


def test_checkpoint_restores_cursor_model_optimizer_and_scheduler(tmp_path):
    config = TripConfig.smoke()
    model = TripModel(
        _TinyEncoder(32, 32), _TinyEncoder(32, 32), 32, 0, config.model
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    generator = torch.Generator().manual_seed(29)
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(
        path,
        model,
        optimizer,
        scheduler,
        7,
        2,
        config.training_sha256,
        generator,
        batch_in_epoch=4,
    )
    expected = next(model.parameters()).detach().clone()
    with torch.no_grad():
        next(model.parameters()).add_(1)
    payload = load_checkpoint(
        path,
        model,
        config.training_sha256,
        optimizer,
        scheduler,
        generator,
    )
    assert payload["step"] == 7 and payload["epoch"] == 2
    assert payload["batch_in_epoch"] == 4
    assert torch.equal(next(model.parameters()), expected)


class _FakeGeneration:
    def __init__(self):
        self.requests = []

    def generate(self, messages, role, decoding, seed, request_id):
        assert "TRIP plan" in messages[0].content
        result = Generation(
            text="Would you like to try Moon?",
            input_tokens=10,
            output_tokens=7,
            total_tokens=17,
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


def test_trip_planner_uses_shared_qwen_contract(monkeypatch):
    generated = DecodedPlan(
        pairs=(("recommend", "Moon"),),
        token_ids=(6, 4, 8, 5, 9, 7),
        beam_score=-1.0,
        agreement_distance=0.2,
        adjusted_score=-1.6,
        forward_candidates=2,
        backward_candidates=2,
        sequence_forwards=12,
    )
    monkeypatch.setattr("tdrl.baselines.trip.planner.generate_plan", lambda *args: generated)
    monkeypatch.setattr(
        _TinyTokenizer,
        "__call__",
        lambda self, *args, **kwargs: {"input_ids": [1, 2, 3]},
        raising=False,
    )
    monkeypatch.setattr(
        _TinyTokenizer,
        "encode",
        lambda self, *args, **kwargs: [4, 8, 5, 9],
        raising=False,
    )
    tokenizer = _TinyTokenizer()
    backend = _FakeGeneration()
    planner = TripPlanner(
        backend,
        SimpleNamespace(parameters=lambda: iter([torch.zeros(1)])),
        tokenizer,
        tokenizer.ids,
        TripConfig.smoke(),
        topdial_record(),
        "e" * 64,
        Decoding(max_tokens=128),
    )
    state = DialogueState((Message("user", "I like mysteries."),))
    plan = planner.act(state, [], 17, "request")
    assert plan.response == "Would you like to try Moon?"
    assert plan.compute["next_action"] == "recommend"
    assert plan.compute["next_topic"] == "Moon"
    assert plan.compute["embedding_requests"] == 0


def test_trip_selection_fails_closed_on_incomplete_payload():
    with pytest.raises(ValueError, match="unknown/incomplete"):
        validate_selection({"schema_version": 1, "sha256": digest("x")})
