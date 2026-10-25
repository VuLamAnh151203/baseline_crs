import json

import pytest

from tdrl.baselines.epl_native.actions import (
    CONTEXT_TOPIC,
    NO_TOPIC,
    TARGET_TOPIC,
    NativeAction,
    NativeActionVocabulary,
)
from tdrl.baselines.epl_native.dataset import topdial_examples
from tdrl.baselines.epl_native.serialization import encode_policy_input, policy_text
from tdrl.contracts import Message
from scripts.epl_native.full_worker import restore_committed_actions


def test_native_action_round_trip_and_resolution():
    action = NativeAction("Movie recommendation", TARGET_TOPIC)
    assert NativeAction.from_key(action.key) == action
    assert action.resolved_topic({"target": ["Movie recommendation", "Arrival"]}) == "Arrival"
    assert NativeAction("Ask questions", CONTEXT_TOPIC).resolved_topic(
        {"target": ["Movie recommendation", "Arrival"]}
    ).startswith("the most relevant")
    assert NativeAction("Say goodbye", NO_TOPIC).resolved_topic(
        {"target": ["Movie recommendation", "Arrival"]}
    ) == "none"


def test_vocabulary_is_sorted_cartesian_and_canonical(tmp_path):
    vocabulary = NativeActionVocabulary("DuRecDial2.0", ["Q&A", "Greetings", "Q&A"], "unit")
    assert vocabulary.goals == ("Greetings", "Q&A")
    assert len(vocabulary.actions) == 6
    assert [NativeAction.from_key(key) for key in vocabulary.actions] == list(
        vocabulary.items
    )
    path = tmp_path / "vocabulary.json"
    vocabulary.write(path)
    assert NativeActionVocabulary.read(path).sha256 == vocabulary.sha256


def test_topdial_adapter_uses_goal_topic_pairs(tmp_path):
    path = tmp_path / "train.jsonl"
    row = {
        "id": "sample",
        "profile": {},
        "personality": {},
        "knowledge": [],
        "target": ["Movie recommendation", "Arrival"],
        "messages": [
            {"role": "assistant", "content": "Hello!"},
            {"role": "user", "content": "I like science fiction."},
            {"role": "assistant", "content": "Would you like a thoughtful film?"},
            {"role": "user", "content": "Yes."},
            {"role": "assistant", "content": "I recommend Arrival."},
        ],
    }
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    examples, goals = topdial_examples(path)
    assert goals == {"Greetings", "Ask questions", "Movie recommendation"}
    assert [item["action"] for item in examples] == [
        NativeAction("Ask questions", TARGET_TOPIC),
        NativeAction("Movie recommendation", TARGET_TOPIC),
    ]


def test_serialization_contains_native_path_without_future_text():
    scenario = {"target": ["Movie recommendation", "Arrival"]}
    text = policy_text(
        scenario,
        [Message("user", "I like science fiction")],
        [NativeAction("Ask questions", TARGET_TOPIC)],
    )
    assert "[GOAL]Ask questions[TOPIC]Arrival[SEP]" in text
    assert "[USER]I like science fiction" in text
    assert "future response" not in text


def test_invalid_topic_role_fails_closed():
    with pytest.raises(ValueError):
        NativeAction("Q&A", "literal movie title")


def test_native_serialization_left_truncates_without_tokenizer_warning_path():
    class Tokenizer:
        cls_token_id = 101
        sep_token_id = 102
        truncation_side = "right"

        def __call__(self, text, **options):
            assert text.startswith("[PATH]:")
            assert self.truncation_side == "left"
            assert options == {
                "add_special_tokens": False,
                "truncation": True,
                "max_length": 4,
                "return_attention_mask": False,
                "return_token_type_ids": False,
            }
            return {"input_ids": [7, 8, 9, 10]}

    tokenizer = Tokenizer()
    ids, mask = encode_policy_input(
        tokenizer,
        {"target": ["Movie recommendation", "Arrival"]},
        [Message("user", "Please recommend something")],
        (),
        max_tokens=6,
    )
    assert tokenizer.truncation_side == "right"
    assert ids.tolist() == [101, 7, 8, 9, 10, 102]
    assert mask.tolist() == [1, 1, 1, 1, 1, 1]


def test_resume_restores_only_committed_native_actions(tmp_path):
    destination = tmp_path / "dialogue"
    destination.mkdir()
    (destination / "dialogue.json").write_text(
        json.dumps(
            {
                "turns_trace": [
                    {
                        "plan": {
                            "path": [
                                NativeAction("Ask questions", TARGET_TOPIC).key
                            ]
                        }
                    },
                    {"plan": {"path": []}},
                    {
                        "plan": {
                            "path": [
                                NativeAction("Movie recommendation", CONTEXT_TOPIC).key
                            ]
                        }
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    class Planner:
        committed_actions = []

    planner = Planner()
    restore_committed_actions(planner, destination)
    assert planner.committed_actions == [
        NativeAction("Ask questions", TARGET_TOPIC),
        NativeAction("Movie recommendation", CONTEXT_TOPIC),
    ]
