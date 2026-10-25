"""Deterministic normalization for ConvPlan mapping and domain retrieval."""

import re
import unicodedata

from ...provenance import digest

TURN_MARKER = re.compile(r"^\s*\[\d+\]\s*")
SPACE = re.compile(r"\s+")
MOVIE_FIELDS = {
    "accepted movie",
    "accepted movies",
    "rejected movie",
    "rejected movies",
    "favorite movie",
    "favorite movies",
}

DOMAIN_TERMS = {
    "movie": ("movie", "film"),
    "music": ("music", "song", "play music"),
    "poi": ("poi", "restaurant", "hotel", "attraction", "place"),
    "food": ("food", "dish", "cuisine", "meal", "eat"),
    "celebrity": ("celebrity", "celebrities", "actor", "actress", "singer"),
    "news": ("news",),
    "weather": ("weather",),
    "hobby": ("hobby", "hobbies", "interest"),
}
PREFERENCE_TERMS = ("accept", "reject", "favorite", "favourite", "prefer", "poi")


def normalized_text(value):
    if not isinstance(value, str):
        raise TypeError("normalization requires text")
    return SPACE.sub(" ", unicodedata.normalize("NFKC", value)).strip().casefold()


def normalized_utterance(value):
    return normalized_text(TURN_MARKER.sub("", value))


def conversation_hash(conversation):
    return digest([normalized_utterance(item) for item in conversation])


def _values(value):
    if isinstance(value, str):
        return [value]
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return value
    return []


def _atomic_values(value):
    result = []
    for item in _values(value):
        result.extend(part for part in re.split(r"[;|]", item) if part.strip())
    return result


def target_domain(target_items):
    """Map a target type to a frozen coarse domain without using outcome data."""
    target_items = tuple(target_items)
    goal = normalized_text(target_items[0]) if target_items else ""
    for domain, terms in DOMAIN_TERMS.items():
        if any(term in goal for term in terms):
            return domain
    return "general"


def _field_domain(key):
    normalized = normalized_text(str(key))
    for domain, terms in DOMAIN_TERMS.items():
        if any(term in normalized for term in terms):
            return domain
    return None


def extract_domain_entities(profile, target_items=()):
    """Generalize EnPL's exact movie overlap to same-domain preference entities."""
    if not isinstance(profile, dict):
        raise TypeError("EnPL profile must be a mapping")
    target_items = tuple(target_items)
    domain = target_domain(target_items)
    entities = set()
    if len(target_items) >= 2 and str(target_items[1]).strip():
        entities.add(normalized_text(str(target_items[1])))
    for key, value in profile.items():
        normalized_key = normalized_text(str(key))
        field_domain = _field_domain(key)
        if domain == "general":
            eligible = any(term in normalized_key for term in PREFERENCE_TERMS)
        elif domain == "hobby":
            eligible = field_domain == "hobby" or any(
                term in normalized_key for term in PREFERENCE_TERMS
            )
        else:
            eligible = field_domain == domain
        if eligible:
            entities.update(
                normalized_text(item) for item in _atomic_values(value) if item.strip()
            )
    return domain, tuple(sorted(entities))


def extract_retrieval_movies(profile, target_items=()):
    """Paper movie-overlap retrieval with an explicit profile-field allowlist."""
    result = set()
    for key, value in profile.items():
        if normalized_text(str(key)) in MOVIE_FIELDS:
            result.update(normalized_text(item) for item in _values(value) if item.strip())
    target_items = tuple(target_items)
    if len(target_items) >= 2 and "movie" in normalized_text(target_items[0]):
        if target_items[1].strip():
            result.add(normalized_text(target_items[1]))
    return tuple(sorted(result))
