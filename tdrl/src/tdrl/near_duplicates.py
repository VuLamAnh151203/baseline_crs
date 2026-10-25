"""Exact Jaccard verification with a lossless prefix-filter candidate index."""

import math
import re
from collections import Counter, defaultdict


def shingles(messages):
    tokens = re.findall(r"\w+", " ".join(m["content"] for m in messages).casefold())
    return {tuple(tokens[i : i + 3]) for i in range(len(tokens) - 2)}


def audit_train_dev(train, dev, threshold=0.9):
    if not 0 < threshold <= 1:
        raise ValueError("invalid near-duplicate threshold")
    sets = [shingles(r["messages"]) for r in [*train, *dev]]
    counts = Counter(s for tokens in sets for s in tokens)

    def order(tokens):
        return sorted(tokens, key=lambda s: (counts[s], s))

    index = defaultdict(list)
    for i, tokens in enumerate(sets[: len(train)]):
        prefix = len(tokens) - math.ceil(threshold * len(tokens)) + 1
        for token in order(tokens)[:prefix]:
            index[token].append(i)
    collisions, candidate_count = [], 0
    for j, tokens in enumerate(sets[len(train) :]):
        prefix = len(tokens) - math.ceil(threshold * len(tokens)) + 1
        candidates = {i for token in order(tokens)[:prefix] for i in index[token]}
        for i in sorted(candidates):
            other = sets[i]
            if len(other) < threshold * len(tokens) or len(tokens) < threshold * len(other):
                continue
            candidate_count += 1
            union = len(other | tokens)
            score = len(other & tokens) / union if union else 0.0
            if score >= threshold:
                collisions.append(
                    {"train_id": train[i]["id"], "dev_id": dev[j]["id"], "jaccard": score}
                )
    return {
        "schema_version": 1,
        "threshold": threshold,
        "tokenization": "casefold_word_trigrams",
        "candidate_filter": "lossless_global_frequency_prefix",
        "verified_candidates": candidate_count,
        "collisions": collisions,
    }
