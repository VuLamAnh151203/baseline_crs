"""Conservative explicit-acceptance protocol; mention is never acceptance.

Rule v1 requires an explicit prospective choice and an exact normalized target mention
in the user utterance. Unrecognized/paraphrased/pronominal acceptance is ambiguous and
may be examined by a separate Qwen diagnostic, never promoted to primary success.
"""

import re
import unicodedata
from dataclasses import dataclass

from .autonomy import explicit_refusal

ACCEPTANCE_VERSION = 1
COMMITMENTS = (
    r"\bi (?:will|would like to|want to|am going to|have decided to) (?:watch|listen|try|visit|buy|book|eat|read|see|accept|check out)\b",
    r"\bi['’]ll (?:watch|listen|try|visit|buy|book|eat|read|see|accept|check out)\b",
    r"\b(?:let['’]s|please) (?:book|try|watch|visit|buy)\b",
    r"\bi accept\b",
)


def normalize(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


@dataclass(frozen=True)
class Acceptance:
    label: str
    accepted: bool
    rule_version: int
    evidence: str | None


def classify_user(state, target):
    if not state.history or state.history[-1].role != "user":
        raise ValueError("acceptance is evaluated on an observed user response")
    text = normalize(state.history[-1].content)
    topic = normalize(target[1])
    if explicit_refusal(state):
        return Acceptance("rejection", False, ACCEPTANCE_VERSION, None)
    topic_pattern = re.compile(r"(?<!\w)" + re.escape(topic) + r"(?!\w)")
    if not topic or not topic_pattern.search(text):
        return Acceptance("unresolved", False, ACCEPTANCE_VERSION, None)
    # Future/negation/conditional markers invalidate an apparent commitment.
    if re.search(r"\b(?:not|never|won't|wouldn't|don't|maybe|might|if|perhaps)\b", text):
        return Acceptance("mention", False, ACCEPTANCE_VERSION, None)
    for pattern in COMMITMENTS:
        m = re.search(pattern, text)
        if m:
            # Tie the exact target mention to the same sentence as the prospective choice.
            sentence = next(
                (
                    s
                    for s in re.split(r"[.!?;]", text)
                    if topic_pattern.search(s) and re.search(pattern, s)
                ),
                None,
            )
            if sentence:
                return Acceptance("acceptance", True, ACCEPTANCE_VERSION, sentence)
    return Acceptance("mention", False, ACCEPTANCE_VERSION, None)
