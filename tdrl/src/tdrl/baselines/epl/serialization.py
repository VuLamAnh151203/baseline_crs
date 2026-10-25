from dataclasses import asdict

import torch

from ...contracts import Message
from ...provenance import digest

SERIALIZATION = {
    "version": 2,
    "source": "EPL convert_example_to_feature_for_goal_prediction",
    "layout": "[PATH]: <empty> [TARGET]: target-item [CONTEXT]: role-tagged observed history",
    "history_truncation": "left; retain most recent tokens",
    "future_fields": "forbidden",
    "adaptation": "goal-topic labels mapped to shared six-action taxonomy",
}
SERIALIZATION_SHA256 = digest(SERIALIZATION)


def _messages(history):
    messages = tuple(Message(**row) if isinstance(row, dict) else row for row in history)
    if any(message.role not in {"user", "assistant"} for message in messages):
        raise ValueError("EPL history contains a non-observed role")
    return messages


def policy_text(scenario, history):
    target = scenario["target"]
    if not isinstance(target, (list, tuple)) or len(target) != 2:
        raise ValueError("EPL scenario target must be a goal/item pair")
    dialogue = "".join(
        ("[USER]" if message.role == "user" else "[SYSTEM]") + message.content
        for message in _messages(history)
    )
    return f"[PATH]:  [TARGET]: {target[1]} [CONTEXT]: {dialogue}"


def encode_policy_input(tokenizer, scenario, history, max_tokens=512, header_tokens=96):
    del header_tokens  # retained in the frozen config for adapter compatibility
    if max_tokens < 3:
        raise ValueError("invalid EPL serialization token budget")
    tokens = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(policy_text(scenario, history)))
    tokens = tokens[-(max_tokens - 2) :]
    ids = torch.tensor(
        [tokenizer.cls_token_id, *tokens, tokenizer.sep_token_id], dtype=torch.long
    )
    return ids, torch.ones_like(ids)


def visible_state_record(scenario, history):
    return {
        "target": scenario["target"],
        "history": [asdict(row) if isinstance(row, Message) else row for row in history],
        "serialization_sha256": SERIALIZATION_SHA256,
    }

__all__ = [
    "SERIALIZATION",
    "SERIALIZATION_SHA256",
    "encode_policy_input",
    "policy_text",
    "visible_state_record",
]
