"""Canonical paper and adapted EnPL prompt renderers."""

import json
from pathlib import Path

from ...planner import ASSISTANT_POLICY
from ...provenance import digest

TEMPLATE_ROOT = Path(__file__).with_name("templates")
ENPL_PLAN_POLICY = (
    "You are the planning component of a helpful conversational recommender. Use only the "
    "supplied conversation and permitted guidance. Training examples specify conversational "
    "strategy, not facts about this user. Do not copy reference people, items, preferences, or "
    "factual claims. Respect explicit refusal and do not plan further persuasion after refusal. "
    "Return a future multi-step conversation plan, not a conversational reply to the user."
)


def template_text(name):
    value = (TEMPLATE_ROOT / name).read_text(encoding="utf-8")
    if not value.endswith("\n"):
        raise ValueError(f"EnPL template must end in newline: {name}")
    return value


def template_hashes():
    return {
        name: digest(template_text(name))
        for name in (
            "paper_distill.txt",
            "paper_generate_plan.txt",
            "paper_generate_response.txt",
            "adapted_generate_plan.txt",
            "adapted_generate_response.txt",
        )
    }


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def render_examples(examples):
    blocks = []
    for index, retrieved in enumerate(examples, start=1):
        entry = retrieved.entry
        blocks.append(
            "\n".join(
                (
                    f"Example {index}:",
                    "Target: " + _json(entry.demonstration_target),
                    "User Setting: " + _json(entry.user_profile),
                    "Plan: " + _json(entry.plan_text),
                )
            )
        )
    return "\n\n".join(blocks)


def render_plan_prompt(scenario, examples):
    template = template_text("adapted_generate_plan.txt")
    return ENPL_PLAN_POLICY + "\n" + template.format(
        examples=render_examples(examples),
        target=_json(scenario["target"]),
        user_setting=_json(scenario["profile"]),
        knowledge=_json(scenario["knowledge"]),
    )


def render_response_prompt(scenario, plan):
    template = template_text("adapted_generate_response.txt")
    return ASSISTANT_POLICY + "\n" + template.format(
        target=_json(scenario["target"]),
        user_setting=_json(scenario["profile"]),
        knowledge=_json(scenario["knowledge"]),
        plan=_json(plan),
    )
