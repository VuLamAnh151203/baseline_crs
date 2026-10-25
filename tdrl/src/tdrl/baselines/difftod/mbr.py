import math
import re
from collections import Counter


def tokenize(text):
    if not isinstance(text, str):
        raise TypeError("MBR input must be text")
    return re.findall(r"\w+|[^\w\s]", text.casefold(), flags=re.UNICODE)


def _ngrams(tokens, order):
    return Counter(tuple(tokens[index : index + order]) for index in range(len(tokens) - order + 1))


def sentence_bleu(candidate, reference, max_order=4, smoothing=0.1):
    candidate = tokenize(candidate) if isinstance(candidate, str) else list(candidate)
    reference = tokenize(reference) if isinstance(reference, str) else list(reference)
    if not candidate or not reference:
        return 0.0
    precisions = []
    for order in range(1, max_order + 1):
        proposed = _ngrams(candidate, order)
        expected = _ngrams(reference, order)
        matches = sum((proposed & expected).values())
        total = max(1, sum(proposed.values()))
        precisions.append(matches / total if matches else smoothing / total)
    geometric = math.exp(sum(math.log(value) for value in precisions) / max_order)
    brevity = (
        1.0
        if len(candidate) > len(reference)
        else math.exp(1 - len(reference) / len(candidate))
    )
    return brevity * geometric


def select_mbr(plans):
    if not isinstance(plans, list) or not plans:
        raise ValueError("MBR requires nonempty candidate plans")
    if any(not isinstance(plan, str) or not plan for plan in plans):
        raise ValueError("MBR requires nonempty candidate plans")
    matrix = [[0.0 for _ in plans] for _ in plans]
    totals = [0.0 for _ in plans]
    for left, candidate in enumerate(plans):
        for right, reference in enumerate(plans):
            if left == right:
                continue
            score = sentence_bleu(candidate, reference)
            matrix[left][right] = score
            totals[left] += score
    selected = max(range(len(plans)), key=lambda index: (totals[index], -index))
    return {"selected": selected, "scores": totals, "matrix": matrix}
