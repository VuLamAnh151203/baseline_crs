"""Isolated reproduction of the CARP legacy interactive evaluation protocol.

This module intentionally does not modify :mod:`tdrl.evaluation` or
:mod:`tdrl.judge`.  It reproduces the public CARP harness contract for a
separate, explicitly labelled comparison: fixed seed-2911 resources, empty
initial history, one assistant/user/moderator cycle, the original user and
moderator prompts, exact yes/no parsing with one repair, and CARP stopping.
"""

from __future__ import annotations

import json
import random
import re
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

from .contracts import Message, Plan
from .provenance import atomic_json, digest, file_hash
from .state import DialogueState

PROTOCOL = "carp_legacy_v1"
DATASETS = ("TopDial", "DuRecDial2.0")
RESOURCE_CONTRACT = {
    "TopDial": {
        "filename": "topdial_test200_seed2911.jsonl",
        "sha256": "bcb3a8e56f2d767eb0f544b4dbeed23f57e051f4d99f0048e7fe6e717b693d4c",
        "count": 200,
        "dataset_name": "topdial",
        "min_success_turn": 1,
    },
    "DuRecDial2.0": {
        "filename": "durecdial_test200_seed2911.jsonl",
        "sha256": "6d8fdc3e9e6d68581224e59789946e5de4b75717276a1c6d330cbe0afaff572e",
        "count": 200,
        "dataset_name": "durecdial",
        "min_success_turn": 2,
    },
}
DEMOGRAPHIC_PROFILE_KEYS = {"Age Range", "Name", "Gender", "Residence", "Occupation"}
MOVIE_OR_MUSIC_KEYS = (
    ("Accepted movies", "preferred movies"),
    ("Accepted music", "preferred music"),
    ("Accepted celebrities", "preferred celebrities"),
    ("Rejected movies", "disliked movies"),
    ("Rejected music", "disliked music"),
)
FOOD_OR_POI_KEYS = (("Accepted food", "preferred food"), ("Accepted POI", "preferred POI"))
ROLE_PREFIX_RE = re.compile(r"^(assistant|system|user)\s*:\s*", flags=re.IGNORECASE)


@dataclass(frozen=True)
class CarpScenario:
    id: str
    domain: str
    target_action: str
    target_item: str
    user_profile: dict[str, str]
    user_personality: dict[str, str]
    knowledge: list[tuple[str, str, str]]
    source_path: str
    dataset_name: str

    def method_scenario(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "target": [self.target_action, self.target_item],
            "profile": dict(self.user_profile),
            "personality": dict(self.user_personality),
            "knowledge": [list(item) for item in self.knowledge],
            # CARP evaluates from an empty history and ignores released conversations.
            "messages": [],
            "dataset_name": self.dataset_name,
            "domain": self.domain,
        }


def _parse_mapping(value: Any) -> dict[str, str]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return {
            str(key).strip(): str(item).strip()
            for key, item in value.items()
            if item is not None and str(item).strip()
        }
    if isinstance(value, str):
        text = value.strip()
        if text.lower().startswith("user profile:"):
            text = text.split(":", 1)[1].strip()
        parsed = {}
        for field in text.split("\t"):
            if ":" not in field:
                continue
            key, item = (part.strip() for part in field.split(":", 1))
            if key and item:
                parsed[key] = item
        return parsed
    raise TypeError(f"unsupported mapping value: {type(value).__name__}")


def _parse_target(value: Any) -> tuple[str, str]:
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return str(value[0]).strip(), str(value[1]).strip()
    if isinstance(value, str):
        text = value.strip()
        if text.lower().startswith("target:"):
            text = text.split(":", 1)[1].strip()
        parts = text.split(",", 1)
        if len(parts) == 2:
            return parts[0].strip(), parts[1].strip()
    raise ValueError(f"could not parse CARP target: {value!r}")


def _domain(action: str, row: dict[str, Any]) -> str:
    lowered = action.lower()
    for value in ("movie", "music", "food", "poi"):
        if value in lowered:
            return value
    supplied = str(row.get("target_domain", "")).strip().lower()
    if supplied in {"movie", "music", "food", "poi"}:
        return supplied
    raise ValueError(f"unsupported CARP target action: {action!r}")


def _parse_knowledge(value: Any) -> list[tuple[str, str, str]]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise TypeError("CARP knowledge must be a list")
    result = []
    for item in value:
        if isinstance(item, (list, tuple)) and len(item) >= 3:
            result.append(tuple(str(part).strip() for part in item[:3]))
        elif isinstance(item, dict):
            values = (item.get("subject") or item.get("s"), item.get("relation") or item.get("r"), item.get("object") or item.get("o"))
            if any(part is None for part in values):
                raise ValueError(f"invalid CARP knowledge item: {item!r}")
            result.append(tuple(str(part).strip() for part in values))
        else:
            raise ValueError(f"invalid CARP knowledge item: {item!r}")
    return result


def _normalized_item(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").replace("\u00a0", " ")).strip()


def _split_items(value: Any) -> list[str]:
    return [item for part in re.split(r"\s*;\s*", _normalized_item(value)) if (item := _normalized_item(part))]


def _remove_target(profile: dict[str, str], target: str) -> dict[str, str]:
    target_key = _normalized_item(target).casefold()
    if not target_key:
        return dict(profile)
    result = {}
    for key, value in profile.items():
        if key in DEMOGRAPHIC_PROFILE_KEYS:
            result[key] = value
            continue
        kept, seen = [], set()
        for item in _split_items(value):
            item_key = item.casefold()
            if item_key != target_key and item_key not in seen:
                kept.append(item)
                seen.add(item_key)
        if kept:
            result[key] = "; ".join(kept)
    return result


def load_carp_scenarios(data_root: Path, dataset: str, shuffle_seed: int = 42):
    if dataset not in RESOURCE_CONTRACT:
        raise ValueError(f"unknown CARP dataset: {dataset}")
    contract = RESOURCE_CONTRACT[dataset]
    path = Path(data_root) / contract["filename"]
    if not path.is_file() or file_hash(path) != contract["sha256"]:
        raise ValueError(f"missing or changed canonical CARP resource: {path}")
    scenarios = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        action, target = _parse_target(row.get("target"))
        profile = _parse_mapping(row.get("user_profile"))
        if dataset == "DuRecDial2.0":
            profile = _remove_target(profile, target)
        scenarios.append(
            CarpScenario(
                id=str(row["id"]),
                domain=_domain(action, row),
                target_action=action,
                target_item=target,
                user_profile=profile,
                user_personality=_parse_mapping(row.get("user_personality")),
                knowledge=_parse_knowledge(row.get("knowledge")),
                source_path=str(path.resolve()),
                dataset_name=contract["dataset_name"],
            )
        )
    if len(scenarios) != contract["count"] or len({row.id for row in scenarios}) != len(scenarios):
        raise ValueError(f"invalid CARP scenario population for {dataset}")
    random.Random(shuffle_seed).shuffle(scenarios)
    return scenarios


# The following prompt builders reproduce carp.harness.prompts for the live paths
# used by its runner.  Dead DuRecDial-specific simulator branches are intentionally
# not introduced here because the upstream runner does not call them.
def build_demographic_description(profile: dict[str, str]) -> str:
    age_range = profile.get("Age Range", "unknown age range").lower()
    residence = profile.get("Residence", "an unknown city")
    occupation = profile.get("Occupation", "").strip().lower()
    gender = profile.get("Gender", "").strip().lower()
    if occupation == "student":
        if gender == "male":
            return f"a male student in the age range of {age_range}, living in {residence}"
        return f"a female student in the age range of {age_range}, living in {residence}"
    if occupation == "employed":
        if gender == "male":
            return f"a man in the age range of {age_range}, working in a company and living in {residence}"
        return f"a woman in the age range of {age_range}, working in a company and living in {residence}"
    if gender == "male":
        return f"a retired man in the age range of {age_range}, living in {residence}"
    return f"a retired woman in the age range of {age_range}, living in {residence}"


def _preference_lines(scenario: CarpScenario) -> list[str]:
    pairs = MOVIE_OR_MUSIC_KEYS if scenario.domain in {"movie", "music"} else FOOD_OR_POI_KEYS
    lines = []
    for key, label in pairs:
        value = scenario.user_profile.get(key, "").strip()
        if value:
            target_key = _normalized_item(scenario.target_item).casefold()
            values, seen = [], set()
            for item in _split_items(value):
                item_key = item.casefold()
                if item_key != target_key and item_key not in seen:
                    values.append(item)
                    seen.add(item_key)
            if values:
                lines.append(f"Your {label}: {'; '.join(values)}.")
    return lines


def build_user_role_description(scenario: CarpScenario) -> str:
    name = scenario.user_profile.get("Name", "the user")
    lines = [
        f"Now enter the role-playing mode. In the following conversation, you will play as a User in a recommendation game. You are looking for a {scenario.domain}. You are {name}, {build_demographic_description(scenario.user_profile)}.",
        "",
        "Based on your known past experiences, you have the following preferences:",
    ]
    preferences = _preference_lines(scenario)
    lines.extend(preferences or ["You have no additional preferences listed beyond what is already implied by your profile."])
    lines.extend(
        [
            "",
            "Respond naturally to the latest assistant message and the ongoing conversation.",
            "Vary your phrasing and avoid repeating yourself.",
            "Your response should match your profile and personality.",
            "Feel free to explore your preferences in a way that fits your profile and the context.",
            "You don't need to recommend anything, but feel free to express your personal interests.",
            "Reply with only one short sentence of no more than 30 words.",
        ]
    )
    return "\n".join(lines)


def _format_history(history: Sequence[Message]) -> str:
    if not history:
        return "(empty)"
    return "\n".join(f"{message.role}: {message.content}" for message in history)


def _assistant_profile_summary(scenario: CarpScenario, *, demographic_only: bool = False) -> str:
    profile = build_demographic_description(scenario.user_profile)
    if demographic_only:
        return profile + "."
    pronoun = "his" if scenario.user_profile.get("Gender", "").strip().lower() == "male" else "her"
    pairs = MOVIE_OR_MUSIC_KEYS if scenario.domain in {"movie", "music"} else FOOD_OR_POI_KEYS
    suffixes = []
    for key, label in pairs:
        values = _split_items(scenario.user_profile.get(key, ""))
        if values:
            suffixes.append(f"{pronoun} {label}: {'; '.join(values)}")
    return profile + ("; " + "; ".join(suffixes) if suffixes else "") + "."


def _format_knowledge(knowledge: Sequence[tuple[str, str, str]]) -> str:
    if not knowledge:
        return "(No scenario knowledge provided.)"
    return "\n".join(f"- {subject} | {relation} | {obj}" for subject, relation, obj in knowledge)


def _durecdial_goal_label(scenario: CarpScenario) -> str:
    action = scenario.target_action.strip()
    if action:
        return "Restaurant recommendation" if "poi" in action.lower() else action
    return {
        "movie": "Movie recommendation",
        "music": "Music recommendation",
        "food": "Food recommendation",
    }.get(scenario.domain, "Restaurant recommendation")


def _durecdial_item_label(scenario: CarpScenario) -> str:
    return {
        "movie": "movie",
        "music": "song or music item",
        "food": "food item",
        "poi": "restaurant or place",
    }.get(scenario.domain, "item")


def build_assistant_role_description(scenario: CarpScenario) -> str:
    """Reproduce the CARP assistant role prompt, including its DuRecDial branch."""
    name = scenario.user_profile.get("Name", "the user")
    descriptions = {
        "movie": "You are a conversational recommender system that recommends films.",
        "music": "You are a conversational recommender system that recommends music.",
        "food": "You are a conversational recommender system that recommends food.",
    }
    has_target = bool(scenario.target_item.strip())
    if scenario.dataset_name == "durecdial" and not has_target:
        item_label = _durecdial_item_label(scenario)
        lines = [
            descriptions.get(
                scenario.domain,
                "You are a conversational recommender system that recommends restaurants or places.",
            ),
            f"You are conversing with {name}, whose profile is below:",
            f"## {_assistant_profile_summary(scenario, demographic_only=True)}",
            "",
            f"The current task is: {_durecdial_goal_label(scenario)}.",
            f"Your goal is to recommend one concrete {item_label} that fits the user and guide them toward accepting it.",
            "A successful recommendation needs a named item and clear user acceptance.",
            "Use the conversation context, user profile, and scenario knowledge to choose the recommendation.",
            "Each reply should be concise (no longer than 30 words).",
            "You may access the following domain knowledge for conversation:",
            "",
            "Scenario Knowledge:",
            _format_knowledge(scenario.knowledge),
        ]
        return "\n".join(lines)

    lines = [
        descriptions.get(
            scenario.domain,
            "You are a conversational recommender system that recommends restaurants.",
        ),
        f"You are conversing with {name}, whose profile is below:",
        f"## {_assistant_profile_summary(scenario)}",
        "",
        f'Your goal is to proactively lead the conversation with {name} towards the target {scenario.domain} "{scenario.target_item}".',
        f"To start the conversation, begin with a greeting and avoid mentioning the target {scenario.domain} or target item.",
        f"As the conversation progresses, use your domain knowledge to steer the discussion towards the target {scenario.domain} step by step.",
        f"Be informative and engaging while providing insights to arouse {name}'s interest.",
        f'Remember to ultimately recommend "{scenario.target_item}" only when it fits the dialogue context.',
        "Each reply should be concise (no longer than 30 words).",
        "You may access the following domain knowledge for conversation:",
        "",
        "Scenario Knowledge:",
        _format_knowledge(scenario.knowledge),
    ]
    return "\n".join(lines)


def build_assistant_turn_prompt(scenario: CarpScenario, history: Sequence[Message]) -> str:
    lines = [
        build_assistant_role_description(scenario),
        "",
        "Here is the conversation history:",
        _format_history(history),
        "",
    ]
    if not history:
        lines.append("If the dialogue history is empty, please generate a response to start the conversation.")
    lines.append("Now generate a succinct response for the next turn:")
    return "\n".join(lines)


def build_user_turn_prompt(scenario: CarpScenario, history: Sequence[Message]) -> str:
    return (
        build_user_role_description(scenario)
        + "\n\nHere is the conversation history:\n"
        + _format_history(history)
        + "\n\nNow generate a succinct response for the next turn."
    )


def build_moderator_prompt(scenario: CarpScenario, history: Sequence[Message]) -> str:
    user_name = scenario.user_profile.get("Name", "the user")
    if scenario.dataset_name == "durecdial" and not scenario.target_item.strip():
        item_label = {"movie": "movie", "music": "song or music item", "food": "food item", "poi": "restaurant or place"}.get(scenario.domain, "item")
        instruction = (
            f"Please decide whether the user {user_name} has accepted any concrete {item_label} that the assistant recommended in the conversation. "
            "Answer yes only if the assistant clearly recommended or named a concrete item and the user clearly accepts, agrees to try, listen to, watch, visit, eat, or check it out. "
            "Answer no if no concrete assistant recommendation exists, or if the user is only curious, asks for more information, stays neutral, changes the subject, or rejects it. "
            "Please only respond with yes or no."
        )
    else:
        instruction = (
            f'Please decide whether the user {user_name} has accepted the target {scenario.domain} "{scenario.target_item}" in the conversation. '
            "Answer yes if the user clearly accepts, agrees to try, listen to, watch, visit, or check out the target item. "
            "Answer no if the user is only curious, asks for more information, stays neutral, changes the subject, or explicitly rejects the target item. "
            "Please only respond with yes or no."
        )
    return "\n".join([instruction, "Here is the conversation history:", _format_history(history)])


def build_moderator_repair_prompt(scenario: CarpScenario, history: Sequence[Message], raw: str) -> str:
    return "\n".join(
        [build_moderator_prompt(scenario, history), "", f"Your previous answer was: {raw.strip()}", "Repair it now. Return exactly one token: yes or no."]
    )


def normalize_role_output(text: str) -> str:
    normalized = text.strip()
    while True:
        updated = ROLE_PREFIX_RE.sub("", normalized, count=1).strip()
        if updated == normalized:
            return normalized
        normalized = updated


def parse_yes_no(text: str) -> str | None:
    normalized = text.strip().lower()
    if not normalized:
        return None
    if normalized in {"yes", "no"}:
        return normalized
    matches = re.findall(r"\b(yes|no)\b", normalized)
    unique = list(dict.fromkeys(matches))
    return unique[0] if len(unique) == 1 else None


def _generation_records(values):
    return [asdict(item) for item in values]


def run_carp_dialogue(
    scenario: CarpScenario,
    planner,
    generation,
    output: Path,
    contract: dict[str, Any],
    *,
    root_seed: int = 17,
    max_turns: int = 10,
    max_output_tokens: int = 3072,
):
    """Run one resumable dialogue under the legacy CARP harness."""
    output.mkdir(parents=True, exist_ok=True)
    snapshot = output / "dialogue.json"
    contract = {**contract, "protocol": PROTOCOL, "evaluator_prompt_sha256": prompt_contract_hash()}
    contract_sha256 = digest(contract)
    if snapshot.exists():
        prior = json.loads(snapshot.read_text(encoding="utf-8"))
        if prior.get("contract_sha256") != contract_sha256:
            raise ValueError("CARP dialogue resume contract mismatch")
        if prior.get("status") == "complete":
            return prior
        history = [Message(**item) for item in prior.get("history", [])]
        turns_trace = list(prior.get("turns_trace", []))
        moderator_trace = list(prior.get("moderator_trace", []))
        start_turn = int(prior.get("actual_turns", 0))
    else:
        history, turns_trace, moderator_trace, start_turn = [], [], [], 0
    if len(history) != start_turn * 2 or len(turns_trace) != start_turn:
        raise ValueError("invalid CARP partial snapshot")
    planner.semantic_judging = True
    planner.simulator_prompt = build_user_role_description(scenario)
    base = f"carp-legacy/{contract['method']}/{scenario.dataset_name}/{scenario.id}"
    namespace = contract.get("request_namespace")
    if namespace:
        base += "/namespace/" + digest(namespace)
    consumed = getattr(generation, "consumed_requests", generation.requests)
    started = time.monotonic()
    success, stop_reason = False, "max_turns"
    for turn_index in range(start_turn, max_turns):
        planner.seed_context = f"{scenario.id}/carp/{turn_index}"
        request_base = f"{base}/turn/{turn_index}"
        generation_start = len(consumed)
        turn_started = time.monotonic()
        if not history:
            # CARP owns the opening turn: its native adapters all receive the
            # assistant-turn prompt as one user message, while method-specific
            # planning starts only after the simulated user has replied.  This
            # preserves CARP's empty-history protocol and avoids fabricating a
            # user utterance in the dialogue state.
            opening_id = request_base + "/assistant-opening"
            opening = generation.generate_carp_legacy(
                build_assistant_turn_prompt(scenario, history),
                "carp_assistant",
                max_output_tokens,
                opening_id,
            )
            plan = Plan(
                response=normalize_role_output(opening.text),
                covered=False,
                reference_id=None,
                directed_distances={},
                path=[],
                selected_candidate=None,
                rollout_scores=[],
                compute={
                    "strategy": "carp-legacy-shared-opening",
                    "method_core_invoked": False,
                    "generation_requests": 1,
                    "embedding_requests": 0,
                    "retrieval_requests": 0,
                    "search_simulations": 0,
                    "recovery_triggered": False,
                    "planner_latency_s": time.monotonic() - turn_started,
                },
                replay_ids=[opening_id],
            )
        else:
            plan = planner.act(DialogueState(tuple(history)), [], root_seed, request_base)
        history.append(Message("assistant", plan.response))
        user_prompt = build_user_turn_prompt(scenario, history)
        user_result = generation.generate_carp_legacy(
            user_prompt,
            "carp_user_simulator",
            max_output_tokens,
            request_base + "/user",
        )
        user_text = normalize_role_output(user_result.text)
        history.append(Message("user", user_text))
        turn_number = turn_index + 1
        if turn_number < RESOURCE_CONTRACT[contract["dataset"]]["min_success_turn"]:
            decision = {
                "accepted": False,
                "raw_response": f"forced_no_before_min_success_turn:{turn_number}",
                "parsed_response": "no",
                "repair_raw_response": None,
                "error": None,
            }
        else:
            moderator_prompt = build_moderator_prompt(scenario, history)
            moderator = generation.generate_carp_legacy(
                moderator_prompt,
                "carp_moderator",
                max_output_tokens,
                request_base + "/moderator",
            )
            parsed = parse_yes_no(moderator.text)
            repair_text = None
            if parsed is None:
                repair = generation.generate_carp_legacy(
                    build_moderator_repair_prompt(scenario, history, moderator.text),
                    "carp_moderator",
                    max_output_tokens,
                    request_base + "/moderator-repair",
                )
                repair_text = repair.text
                parsed = parse_yes_no(repair.text)
            decision = {
                "accepted": None if parsed is None else parsed == "yes",
                "raw_response": moderator.text,
                "parsed_response": parsed,
                "repair_raw_response": repair_text,
                "error": None if parsed is not None else "Could not parse moderator response as yes/no after one repair attempt.",
            }
        moderator_trace.append(decision)
        turns_trace.append(
            {
                "turn": turn_number,
                "plan": asdict(plan),
                "user_prompt_sha256": digest(user_prompt),
                "user_response": user_text,
                "moderator": decision,
                "generation": _generation_records(consumed[generation_start:]),
                "wall_s": time.monotonic() - turn_started,
            }
        )
        if decision["accepted"] is None:
            stop_reason = "moderator_parse_error"
        elif decision["accepted"]:
            success, stop_reason = True, "accepted"
        complete = success or decision["accepted"] is None or turn_number == max_turns
        result = {
            "schema_version": 1,
            "status": "complete" if complete else "partial",
            "protocol": PROTOCOL,
            "contract": contract,
            "contract_sha256": contract_sha256,
            "scenario_id": scenario.id,
            "dataset": contract["dataset"],
            "dataset_name": scenario.dataset_name,
            "domain": scenario.domain,
            "target_action": scenario.target_action,
            "target_item": scenario.target_item,
            "success": success,
            "accepted": success,
            "turns": turn_number,
            "actual_turns": turn_number,
            "min_success_turn": RESOURCE_CONTRACT[contract["dataset"]]["min_success_turn"],
            "stop_reason": stop_reason if complete else "in_progress",
            "history": [asdict(item) for item in history],
            "transcript": [{"role": item.role, "text": item.content} for item in history],
            "turns_trace": turns_trace,
            "moderator_trace": moderator_trace,
            "error": decision["error"],
            "wall_s": time.monotonic() - started,
        }
        atomic_json(snapshot, result)
        if complete:
            if decision["accepted"] is None:
                raise ValueError(decision["error"])
            return result
    raise AssertionError("unreachable CARP dialogue exit")


def prompt_contract_hash() -> str:
    fixture = CarpScenario(
        "fixture", "movie", "Movie recommendation", "Target", {"Name": "User", "Age Range": "18-25", "Gender": "Male", "Residence": "City", "Occupation": "Employed", "Accepted movies": "Known"}, {}, [], "fixture", "topdial"
    )
    history = [Message("assistant", "Hello."), Message("user", "Hi.")]
    return digest(
        {
            "user_role": build_user_role_description(fixture),
            "assistant_opening": build_assistant_turn_prompt(fixture, []),
            "user_turn": build_user_turn_prompt(fixture, history[:1]),
            "moderator": build_moderator_prompt(fixture, history),
            "repair": build_moderator_repair_prompt(fixture, history, "maybe"),
        }
    )


def summarize_carp_results(results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    domains = defaultdict(list)
    for row in results:
        domains[row["domain"]].append(row)
    successes = sum(bool(row["success"]) for row in results)
    return {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "num_scenarios": len(results),
        "num_success": successes,
        "success_rate": successes / len(results) if results else 0.0,
        "average_turns": sum(int(row["turns"]) for row in results) / len(results) if results else 0.0,
        "error_count": sum(bool(row.get("error")) for row in results),
        "stop_reason_counts": dict(Counter(row["stop_reason"] for row in results)),
        "per_domain": {
            domain: {
                "num_scenarios": len(rows),
                "num_success": sum(bool(row["success"]) for row in rows),
                "success_rate": sum(bool(row["success"]) for row in rows) / len(rows),
                "average_turns": sum(int(row["turns"]) for row in rows) / len(rows),
            }
            for domain, rows in sorted(domains.items())
        },
    }
