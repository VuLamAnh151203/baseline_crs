"""Paper-specified entity-sequence consistency with explicit empty handling."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ConsistencyResult:
    score: float
    valid: bool
    reason: str | None = None


def levenshtein_tokens(left, right):
    previous = list(range(len(right) + 1))
    for index, source in enumerate(left, start=1):
        current = [index]
        for offset, target in enumerate(right, start=1):
            current.append(
                min(
                    previous[offset] + 1,
                    current[offset - 1] + 1,
                    previous[offset - 1] + int(source != target),
                )
            )
        previous = current
    return previous[-1]


def consistency(plan_entities, conversation_entities):
    if not plan_entities or not conversation_entities:
        return ConsistencyResult(0.0, False, "empty_entity_sequence")
    distance = levenshtein_tokens(plan_entities, conversation_entities)
    denominator = max(len(plan_entities), len(conversation_entities))
    return ConsistencyResult(1.0 - distance / denominator, True)
