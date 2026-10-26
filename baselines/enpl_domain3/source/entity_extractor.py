"""Deterministic TextRank reconstruction with phrase-preserving output."""

import math
import re
from collections import Counter, defaultdict

from ...provenance import digest
from .normalize import normalized_text
from .schema import Keyphrase

TOKEN = re.compile(r"[^\W_]+(?:['’-][^\W_]+)*", re.UNICODE)
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "but", "by", "for", "from",
    "had", "has", "have", "he", "her", "his", "i", "in", "is", "it", "its", "me",
    "my", "of", "on", "or", "our", "she", "that", "the", "their", "them", "they",
    "this", "to", "was", "we", "were", "will", "with", "you", "your",
}


class DeterministicTextRankExtractor:
    """TextRank [D]: token graph, window 4, damping .85, merged adjacent phrases."""

    def __init__(self, keyword_budget=20, window_size=4, damping=0.85, iterations=50):
        if keyword_budget < 1 or window_size < 2 or iterations < 1:
            raise ValueError("invalid TextRank reconstruction settings")
        self.keyword_budget = keyword_budget
        self.window_size = window_size
        self.damping = damping
        self.iterations = iterations
        self.config = {
            "implementation": "independent_token_textrank_v1",
            "keyword_budget": keyword_budget,
            "window_size": window_size,
            "damping": damping,
            "iterations": iterations,
            "phrase_order": "first_occurrence",
            "duplicate_policy": "keep_first",
        }
        self.config_hash = digest(self.config)

    @staticmethod
    def _tokens(text):
        result = []
        for match in TOKEN.finditer(text):
            canonical = normalized_text(match.group())
            if len(canonical) > 1 and canonical not in STOPWORDS:
                result.append((canonical, match.group(), match.start(), match.end()))
        return result

    def extract(self, text):
        tokens = self._tokens(text)
        if not tokens:
            return []
        graph = defaultdict(Counter)
        for index, (word, *_unused) in enumerate(tokens):
            stop = min(len(tokens), index + self.window_size)
            for other_index in range(index + 1, stop):
                other = tokens[other_index][0]
                if word != other:
                    graph[word][other] += 1
                    graph[other][word] += 1
        nodes = sorted({item[0] for item in tokens})
        scores = dict.fromkeys(nodes, 1.0)
        for _iteration in range(self.iterations):
            updated = {}
            for node in nodes:
                incoming = 0.0
                for neighbor, weight in graph[node].items():
                    denominator = sum(graph[neighbor].values())
                    if denominator:
                        incoming += scores[neighbor] * weight / denominator
                updated[node] = (1.0 - self.damping) + self.damping * incoming
            if max(abs(updated[node] - scores[node]) for node in nodes) < 1e-8:
                scores = updated
                break
            scores = updated
        ranked = sorted(nodes, key=lambda node: (-scores[node], node))
        chosen = set(ranked[: self.keyword_budget])
        phrases = []
        start = 0
        while start < len(tokens):
            if tokens[start][0] not in chosen:
                start += 1
                continue
            end = start + 1
            while (
                end < len(tokens)
                and tokens[end][0] in chosen
                and not text[tokens[end - 1][3] : tokens[end][2]].strip()
            ):
                end += 1
            canonical = " ".join(item[0] for item in tokens[start:end])
            surface = text[tokens[start][2] : tokens[end - 1][3]]
            score = sum(scores[item[0]] for item in tokens[start:end]) / math.sqrt(end - start)
            phrases.append(
                Keyphrase(canonical, surface, score, tokens[start][2], tokens[end - 1][3])
            )
            start = end
        best = {}
        for phrase in phrases:
            old = best.get(phrase.canonical_text)
            if old is None or (-phrase.score, phrase.first_start) < (
                -old.score,
                old.first_start,
            ):
                best[phrase.canonical_text] = phrase
        return sorted(best.values(), key=lambda item: (item.first_start, item.canonical_text))
