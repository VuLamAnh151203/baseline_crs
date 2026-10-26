import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

from .serialization import SPECIAL_TOKENS

ACT, TOPIC, BOS, EOS = SPECIAL_TOKENS


def add_special_tokens(tokenizer):
    tokenizer.add_special_tokens({"additional_special_tokens": list(SPECIAL_TOKENS)})
    ids = {token: tokenizer.convert_tokens_to_ids(token) for token in SPECIAL_TOKENS}
    invalid = len(set(ids.values())) != len(ids) or any(
        value is None or value < 0 for value in ids.values()
    )
    if invalid:
        raise ValueError("TRIP tokenizer did not register its special tokens")
    if tokenizer.pad_token_id is None:
        raise ValueError("TRIP tokenizer requires a pad token")
    return ids


def _encode(tokenizer, text, max_length):
    encoded = tokenizer(
        text,
        add_special_tokens=True,
        truncation=True,
        max_length=max_length,
        return_attention_mask=False,
    )
    ids = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
    if not ids or any(not isinstance(token, int) or token < 0 for token in ids):
        raise ValueError("TRIP tokenizer returned invalid IDs")
    return ids


def serialize_context(row):
    header = json.dumps(
        {
            "profile": row["profile"],
            "personality": row.get("personality", {}),
            "knowledge": row["knowledge"],
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    turns = "\n".join(
        f"{message['role']}: {message['content'].strip()}" for message in row["messages"]
    )
    return header + ("\n" + turns if turns else "")


def encode_context(row, tokenizer, max_length):
    """Keep a bounded scenario header plus the most recent observed turns."""
    header = json.dumps(
        {
            "profile": row["profile"],
            "personality": row.get("personality", {}),
            "knowledge": row["knowledge"],
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    special_count = (
        tokenizer.num_special_tokens_to_add(pair=False)
        if hasattr(tokenizer, "num_special_tokens_to_add")
        else 0
    )
    budget = max_length - special_count
    if budget < 1:
        raise ValueError("TRIP context budget cannot hold tokenizer special tokens")
    header_budget = min(256, max(1, budget // 2))
    header_ids = tokenizer.encode(header, add_special_tokens=False)[:header_budget]
    remaining = budget - len(header_ids)
    selected = []
    for message in reversed(row["messages"]):
        turn = tokenizer.encode(
            f"{message['role']}: {message['content'].strip()}\n", add_special_tokens=False
        )
        if len(turn) > remaining:
            if not selected and remaining:
                selected = turn[-remaining:]
            break
        selected = turn + selected
        remaining -= len(turn)
    body = header_ids + selected
    if hasattr(tokenizer, "build_inputs_with_special_tokens"):
        body = tokenizer.build_inputs_with_special_tokens(body)
    if not body or len(body) > max_length:
        raise ValueError("TRIP context serialization exceeded its declared budget")
    return body


def path_text(actions, topics):
    if len(actions) != len(topics) or not actions:
        raise ValueError("TRIP action/topic paths must be aligned and nonempty")
    return "".join(
        f"{ACT}{action}{TOPIC}{topic}"
        for action, topic in zip(actions, topics, strict=True)
    )


def _plan_ids(tokenizer, actions, topics, token_ids, max_length):
    body = tokenizer.encode(path_text(actions, topics), add_special_tokens=False)
    ids = [token_ids[BOS], *body, token_ids[EOS]]
    if len(ids) > max_length:
        raise ValueError("TRIP plan exceeds configured training length; refusing silent truncation")
    return ids


def _target_token_mask(sequence, target_ids):
    mask = [0] * len(sequence)
    found = False
    for start in range(len(sequence) - len(target_ids) + 1):
        if sequence[start : start + len(target_ids)] == target_ids:
            mask[start : start + len(target_ids)] = [1] * len(target_ids)
            found = True
    if not found:
        raise ValueError("TRIP target span is absent from a positive/negative plan")
    return mask


class TripDataset(Dataset):
    def __init__(self, path, tokenizer, max_context_tokens, max_plan_tokens):
        self.tokenizer = tokenizer
        self.token_ids = add_special_tokens(tokenizer)
        self.max_context_tokens = max_context_tokens
        self.max_plan_tokens = max_plan_tokens
        self.rows = [
            json.loads(line)
            for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not self.rows or len({row["id"] for row in self.rows}) != len(self.rows):
            raise ValueError("TRIP dataset must contain unique, nonempty examples")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        target_text = f"{ACT}{row['target'][0]}{TOPIC}{row['target'][1]}"
        target = _encode(self.tokenizer, target_text, self.max_plan_tokens)
        target_body = self.tokenizer.encode(target_text, add_special_tokens=False)
        forward = _plan_ids(
            self.tokenizer,
            row["action_path"],
            row["topic_path"],
            self.token_ids,
            self.max_plan_tokens,
        )
        backward = _plan_ids(
            self.tokenizer,
            list(reversed(row["action_path"])),
            list(reversed(row["topic_path"])),
            self.token_ids,
            self.max_plan_tokens,
        )
        negatives_forward, negatives_backward = [], []
        for topic in row["negative_topics"]:
            topics = [topic if value == row["target"][1] else value for value in row["topic_path"]]
            negative_target = self.tokenizer.encode(
                f"{ACT}{row['target'][0]}{TOPIC}{topic}", add_special_tokens=False
            )
            negative_forward = _plan_ids(
                self.tokenizer,
                row["action_path"],
                topics,
                self.token_ids,
                self.max_plan_tokens,
            )
            negative_backward = _plan_ids(
                self.tokenizer,
                list(reversed(row["action_path"])),
                list(reversed(topics)),
                self.token_ids,
                self.max_plan_tokens,
            )
            negatives_forward.append(
                (negative_forward, _target_token_mask(negative_forward, negative_target))
            )
            negatives_backward.append(
                (negative_backward, _target_token_mask(negative_backward, negative_target))
            )
        return {
            "id": row["id"],
            "context_ids": encode_context(row, self.tokenizer, self.max_context_tokens),
            "target_ids": target,
            "forward_ids": forward,
            "forward_target_mask": _target_token_mask(forward, target_body),
            "backward_ids": backward,
            "backward_target_mask": _target_token_mask(backward, target_body),
            "negative_forward": negatives_forward,
            "negative_backward": negatives_backward,
        }


def _pad(sequences, value):
    length = max(map(len, sequences))
    result = torch.full((len(sequences), length), value, dtype=torch.long)
    mask = torch.zeros_like(result)
    for index, sequence in enumerate(sequences):
        result[index, : len(sequence)] = torch.tensor(sequence)
        mask[index, : len(sequence)] = 1
    return result, mask


class TripCollator:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, rows):
        batch = {"ids": [row["id"] for row in rows]}
        for name in ("context_ids", "target_ids", "forward_ids", "backward_ids"):
            batch[name], batch[name.replace("ids", "attention_mask")] = _pad(
                [row[name] for row in rows], self.pad_token_id
            )
        for name in ("forward_target_mask", "backward_target_mask"):
            batch[name], _ = _pad([row[name] for row in rows], 0)
        for direction in ("forward", "backward"):
            values = [row[f"negative_{direction}"] for row in rows]
            count = len(values[0])
            if not count or any(len(item) != count for item in values):
                raise ValueError("TRIP negative count must be constant within a batch")
            flat_ids = [pair[0] for item in values for pair in item]
            flat_targets = [pair[1] for item in values for pair in item]
            ids, masks = _pad(flat_ids, self.pad_token_id)
            targets, _ = _pad(flat_targets, 0)
            batch[f"negative_{direction}_ids"] = ids.view(len(rows), count, -1)
            batch[f"negative_{direction}_attention_mask"] = masks.view(len(rows), count, -1)
            batch[f"negative_{direction}_target_mask"] = targets.view(len(rows), count, -1)
        return batch


def inference_inputs(record, state, tokenizer, max_context_tokens, max_plan_tokens):
    from .serialization import visible_input

    supplied = visible_input(record, state)
    row = {**supplied, "messages": supplied["messages"]}
    context = encode_context(row, tokenizer, max_context_tokens)
    target_text = f"{ACT}{record['target'][0]}{TOPIC}{record['target'][1]}"
    target = _encode(tokenizer, target_text, max_plan_tokens)
    constraint = tokenizer.encode(target_text, add_special_tokens=False)
    return context, target, constraint
