import json
from dataclasses import asdict

import torch

from ...contracts import Message
from ...provenance import digest

SERIALIZATION = {
    "version": 1,
    "header_fields": ["target", "profile"],
    "history": "recent_complete_observed_turns",
    "future_fields": "forbidden",
}
SERIALIZATION_SHA256 = digest(SERIALIZATION)


def _header(scenario):
    return json.dumps(
        {"target": scenario["target"], "profile": scenario["profile"]},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def policy_text(scenario, history):
    messages = [Message(**row) if isinstance(row, dict) else row for row in history]
    return _header(scenario) + "\n" + "\n".join(
        f"{message.role}: {message.content}" for message in messages
    )


def encode_policy_input(tokenizer, scenario, history, max_tokens=512, header_tokens=96):
    if not 1 <= header_tokens < max_tokens - 2:
        raise ValueError("invalid DPDP serialization token budget")
    messages = [Message(**row) if isinstance(row, dict) else row for row in history]
    if any(message.role not in {"user", "assistant"} for message in messages):
        raise ValueError("DPDP history contains a non-observed role")
    header = tokenizer.encode(_header(scenario), add_special_tokens=False)[:header_tokens]
    remaining = max_tokens - len(header) - 2
    selected = []
    for message in reversed(messages):
        tokens = tokenizer.encode(
            f"{message.role}: {message.content}", add_special_tokens=False
        )
        if len(tokens) > remaining:
            break
        selected = tokens + selected
        remaining -= len(tokens)
    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id
    if cls_id is None or sep_id is None:
        raise ValueError("DPDP tokenizer must define CLS and SEP tokens")
    ids = torch.tensor([cls_id, *header, *selected, sep_id], dtype=torch.long)
    return ids, torch.ones_like(ids)


def visible_state_record(scenario, history):
    return {
        "profile": scenario["profile"],
        "target": scenario["target"],
        "history": [asdict(m) if isinstance(m, Message) else m for m in history],
        "serialization_sha256": SERIALIZATION_SHA256,
    }
