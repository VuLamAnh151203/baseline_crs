"""Deterministic extraction of the user-visible response from one-call ProCoT output."""

import re
from dataclasses import dataclass

from .prompts import PROCOT_FINAL_MARKER

CURRENT_RE = re.compile(r"the current topics are\s*(\[[^\n]*?\])", re.IGNORECASE)
NEXT_RE = re.compile(r"the next topics are\s*(\[[^\n]*?\])", re.IGNORECASE)


@dataclass(frozen=True)
class ParsedProCoT:
    response: str
    planning_text: str
    current_topics: str | None
    next_topics: str | None
    format_valid: bool


def _trim_response(value):
    text = value.strip().lstrip(":").strip()
    if len(text) >= 2 and text[0] in {'"', "'"}:
        quote = text[0]
        if text.endswith(quote):
            text = text[1:-1].strip()
        elif text.endswith(quote + "."):
            text = text[1:-2].strip()
    return text


def parse_procot(text):
    """Use the final declared marker; malformed output remains an observable method failure."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("ProCoT generation must be non-empty text")
    raw = text.strip()
    index = raw.casefold().rfind(PROCOT_FINAL_MARKER.casefold())
    current = CURRENT_RE.search(raw)
    next_topics = NEXT_RE.search(raw)
    if index < 0:
        return ParsedProCoT(
            response=raw,
            planning_text="",
            current_topics=current.group(1) if current else None,
            next_topics=next_topics.group(1) if next_topics else None,
            format_valid=False,
        )
    planning = raw[:index].strip()
    response = _trim_response(raw[index + len(PROCOT_FINAL_MARKER) :])
    valid = bool(response and current and next_topics)
    return ParsedProCoT(
        response=response or raw,
        planning_text=planning,
        current_topics=current.group(1) if current else None,
        next_topics=next_topics.group(1) if next_topics else None,
        format_valid=valid,
    )
