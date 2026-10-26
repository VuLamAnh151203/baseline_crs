from dataclasses import dataclass

import torch
import torch.nn.functional as functional

from .dataset import ACT, BOS, EOS, TOPIC


@dataclass(frozen=True)
class Beam:
    tokens: tuple[int, ...]
    score: float
    constraint_progress: int = 0
    constraint_done: bool = False


@dataclass(frozen=True)
class DecodedPlan:
    pairs: tuple[tuple[str, str], ...]
    token_ids: tuple[int, ...]
    beam_score: float
    agreement_distance: float
    adjusted_score: float
    forward_candidates: int
    backward_candidates: int
    sequence_forwards: int


def _constraint_progress(tokens, constraint):
    maximum = min(len(tokens), len(constraint))
    for size in range(maximum, 0, -1):
        if list(tokens[-size:]) == list(constraint[:size]):
            return size
    return 0


def _contains(tokens, constraint):
    return any(
        list(tokens[start : start + len(constraint)]) == list(constraint)
        for start in range(len(tokens) - len(constraint) + 1)
    )


@torch.no_grad()
def _beam_search(
    model,
    direction,
    memory,
    memory_mask,
    prefix,
    eos_id,
    beam_size,
    max_length,
    min_length,
    repetition_penalty,
    constraint=(),
):
    if len(prefix) >= max_length:
        raise ValueError("TRIP decoding prefix exceeds maximum plan length")
    beams = [Beam(tuple(prefix), 0.0, 0, not constraint)]
    completed = []
    forwards = 0
    while beams and len(completed) < beam_size:
        candidates = []
        for beam in beams:
            if len(beam.tokens) >= max_length:
                if beam.constraint_done:
                    completed.append(beam)
                continue
            ids = torch.tensor([beam.tokens], dtype=torch.long, device=memory.device)
            mask = torch.ones_like(ids)
            logits, _ = model.decode_direction(direction, ids, mask, memory, memory_mask)
            forwards += 1
            scores = logits[0, -1].float()
            if repetition_penalty > 1:
                for token in set(beam.tokens):
                    scores[token] = (
                        scores[token] * repetition_penalty
                        if scores[token] < 0
                        else scores[token] / repetition_penalty
                    )
            log_probs = functional.log_softmax(scores, dim=-1)
            token_candidates = torch.topk(
                log_probs, min(beam_size * 2, len(log_probs))
            ).indices.tolist()
            progress = beam.constraint_progress
            remaining = max_length - len(beam.tokens)
            if constraint and not beam.constraint_done:
                forced = constraint[progress] if progress < len(constraint) else constraint[0]
                token_candidates.append(forced)
                required = len(constraint) - progress
                if remaining <= required + 1:
                    token_candidates = [forced]
            for token in dict.fromkeys(token_candidates):
                if token == eos_id:
                    if len(beam.tokens) < min_length or not beam.constraint_done:
                        continue
                    completed.append(
                        Beam(
                            (*beam.tokens, token),
                            beam.score + float(log_probs[token]),
                            beam.constraint_progress,
                            beam.constraint_done,
                        )
                    )
                    continue
                new_tokens = (*beam.tokens, token)
                done = beam.constraint_done or (constraint and _contains(new_tokens, constraint))
                new_progress = (
                    0 if done or not constraint else _constraint_progress(new_tokens, constraint)
                )
                candidates.append(
                    Beam(
                        new_tokens,
                        beam.score + float(log_probs[token]),
                        new_progress,
                        bool(done),
                    )
                )
        if not candidates:
            break
        candidates.sort(key=lambda item: item.score / max(1, len(item.tokens)), reverse=True)
        beams = candidates[:beam_size]
        if len(beams[0].tokens) >= max_length:
            completed.extend(beam for beam in beams if beam.constraint_done)
            break
    valid = [beam for beam in completed if beam.constraint_done]
    if len(valid) < beam_size:
        valid.extend(beam for beam in beams if beam.constraint_done)
    unique = {}
    for beam in valid:
        unique.setdefault(beam.tokens, beam)
    result = sorted(
        unique.values(), key=lambda item: item.score / max(1, len(item.tokens)), reverse=True
    )[:beam_size]
    if not result:
        raise RuntimeError("TRIP constrained beam search produced no valid plan")
    return result, forwards


@torch.no_grad()
def _hidden(model, direction, beams, memory, memory_mask):
    result = []
    for beam in beams:
        ids = torch.tensor([beam.tokens], dtype=torch.long, device=memory.device)
        mask = torch.ones_like(ids)
        _, hidden = model.decode_direction(direction, ids, mask, memory, memory_mask)
        result.append((hidden, mask))
    return result


def _parse_pairs(tokens, tokenizer, token_ids, backward=True):
    act_id, topic_id = token_ids[ACT], token_ids[TOPIC]
    bos_id, eos_id = token_ids[BOS], token_ids[EOS]
    clean = [token for token in tokens if token not in {bos_id, eos_id, tokenizer.pad_token_id}]
    pairs = []
    index = 0
    while index < len(clean):
        if clean[index] != act_id:
            index += 1
            continue
        topic_at = index + 1
        while topic_at < len(clean) and clean[topic_at] not in {act_id, topic_id}:
            topic_at += 1
        if topic_at >= len(clean) or clean[topic_at] != topic_id:
            break
        end = topic_at + 1
        while end < len(clean) and clean[end] != act_id:
            end += 1
        action = tokenizer.decode(clean[index + 1 : topic_at], skip_special_tokens=True).strip()
        topic = tokenizer.decode(clean[topic_at + 1 : end], skip_special_tokens=True).strip()
        if action and topic:
            pairs.append((action, topic))
        index = end
    if backward:
        pairs.reverse()
    if not pairs:
        raise RuntimeError("TRIP generated plan has no parseable [A]/[T] pair")
    return tuple(pairs)


@torch.no_grad()
def generate_plan(model, tokenizer, token_ids, context_ids, target_ids, constraint_ids, config):
    if len(constraint_ids) + 2 > config.max_plan_tokens:
        raise ValueError("TRIP target constraint cannot fit inside the decoding budget")
    device = next(model.parameters()).device
    context = torch.tensor([context_ids], dtype=torch.long, device=device)
    target = torch.tensor([target_ids], dtype=torch.long, device=device)
    context_mask, target_mask = torch.ones_like(context), torch.ones_like(target)
    memory, memory_mask, _ = model.encode(context, context_mask, target, target_mask)
    bos, eos = token_ids[BOS], token_ids[EOS]
    forward, forward_calls = _beam_search(
        model,
        "forward",
        memory,
        memory_mask,
        [bos],
        eos,
        config.beam_size,
        config.max_plan_tokens,
        config.min_plan_tokens,
        config.repetition_penalty,
        tuple(constraint_ids),
    )
    backward_prefix = [bos, *constraint_ids]
    backward, backward_calls = _beam_search(
        model,
        "backward",
        memory,
        memory_mask,
        backward_prefix,
        eos,
        config.beam_size,
        config.max_plan_tokens,
        max(config.min_plan_tokens, len(backward_prefix)),
        config.repetition_penalty,
    )
    forward_hidden = _hidden(model, "forward", forward, memory, memory_mask)
    backward_hidden = _hidden(model, "backward", backward, memory, memory_mask)
    scored = []
    for index, beam in enumerate(backward):
        distances = [
            float(
                model.distance(
                    backward_hidden[index][0], hidden, backward_hidden[index][1], mask
                )
            )
            for hidden, mask in forward_hidden
        ]
        average_distance = sum(distances) / len(distances)
        normalized_score = beam.score / max(1, len(beam.tokens))
        adjusted = normalized_score - config.lambda_scale * average_distance
        scored.append((adjusted, index, average_distance, normalized_score))
    adjusted, selected, agreement, normalized_score = max(
        scored, key=lambda value: (value[0], -value[1])
    )
    beam = backward[selected]
    target_pairs = _parse_pairs(constraint_ids, tokenizer, token_ids, backward=False)
    suffix = beam.tokens[len(backward_prefix) :]
    try:
        preceding_pairs = _parse_pairs(suffix, tokenizer, token_ids, backward=False)
    except RuntimeError:
        preceding_pairs = ()
    pairs = tuple(reversed((*target_pairs, *preceding_pairs)))
    return DecodedPlan(
        pairs=pairs,
        token_ids=beam.tokens,
        beam_score=normalized_score,
        agreement_distance=agreement,
        adjusted_score=adjusted,
        forward_candidates=len(forward),
        backward_candidates=len(backward),
        sequence_forwards=forward_calls + backward_calls + len(forward) + len(backward),
    )
