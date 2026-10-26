from dataclasses import asdict

import torch

from ...contracts import Message
from ...provenance import digest

SERIALIZATION = {
    "version": 1,
    "source": "EPL convert_example_to_feature_for_goal_prediction",
    "layout": "[PATH]: repeated [GOAL]/[TOPIC] pairs [TARGET]: item [CONTEXT]: observed history",
    "topic_adapter": "target-relative TARGET/CONTEXT/NONE roles",
    "history_truncation": "left; retain most recent tokens",
    "future_fields": "forbidden",
}
SERIALIZATION_SHA256 = digest(SERIALIZATION)


def _messages(history):
    messages = tuple(Message(**row) if isinstance(row, dict) else row for row in history)
    if any(message.role not in {"user", "assistant"} for message in messages):
        raise ValueError("EPL native history contains a non-observed role")
    return messages


def policy_text(scenario, history, action_history=()):
    target = scenario["target"]
    if not isinstance(target, (list, tuple)) or len(target) != 2:
        raise ValueError("EPL native target must be a goal/item pair")
    path = "".join(
        f"[GOAL]{action.goal}[TOPIC]{action.resolved_topic(scenario)}[SEP]"
        for action in action_history
    )
    dialogue = "".join(
        ("[USER]" if message.role == "user" else "[SYSTEM]") + message.content
        for message in _messages(history)
    )
    return f"[PATH]: {path} [TARGET]: {target[1]} [CONTEXT]: {dialogue}"


def encode_policy_input(tokenizer, scenario, history, action_history, max_tokens=512):
    if max_tokens < 3:
        raise ValueError("invalid EPL native serialization token budget")
    # The released EPL feature converter keeps the most recent tokens.  Ask the
    # tokenizer to perform that same truncation itself so Transformers never
    # constructs (and warns about) an over-length intermediate sequence.
    original_side = tokenizer.truncation_side
    tokenizer.truncation_side = "left"
    try:
        encoded = tokenizer(
            policy_text(scenario, history, action_history),
            add_special_tokens=False,
            truncation=True,
            max_length=max_tokens - 2,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
    finally:
        tokenizer.truncation_side = original_side
    tokens = encoded["input_ids"]
    ids = torch.tensor(
        [tokenizer.cls_token_id, *tokens, tokenizer.sep_token_id], dtype=torch.long
    )
    return ids, torch.ones_like(ids)


def visible_state_record(scenario, history, action_history=()):
    return {
        "target": scenario["target"],
        "history": [asdict(row) if isinstance(row, Message) else row for row in history],
        "action_history": [row.to_dict() for row in action_history],
        "serialization_sha256": SERIALIZATION_SHA256,
    }
