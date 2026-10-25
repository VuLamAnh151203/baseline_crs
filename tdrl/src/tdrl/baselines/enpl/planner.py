"""Two-call, turn-wise EnPL planner and response generator."""

import re
import time

from ...contracts import Decoding, Message, Plan
from ...planner import derived_seed
from ...provenance import digest
from .normalize import target_domain
from .prompts import render_plan_prompt, render_response_prompt, template_hashes

ENPL_POLICY_VERSION = "independent-enpl-domain3-history-conditioned-v4"
ROLE_BOUNDARY = re.compile(r"(?:^|\n)\s*(?:user|assistant|ai)\s*:\s*", re.IGNORECASE)
NUMBERED_PLAN_STEP = re.compile(r"(?m)^\s*\d+[.)]\s*\S")
BULLETED_PLAN_STEP = re.compile(r"(?m)^\s*[-*\u2022]\s+\S")


def _single_response(text):
    if not isinstance(text, str) or not text.strip():
        raise ValueError("EnPL response generation returned empty text")
    raw = text.strip()
    cleaned = re.sub(r"^\s*(?:assistant|ai)\s*:\s*", "", raw, flags=re.IGNORECASE)
    boundary = ROLE_BOUNDARY.search(cleaned)
    if boundary:
        cleaned = cleaned[: boundary.start()].strip()
    return (cleaned or raw), boundary is None


def _parsed_plan(text):
    if not isinstance(text, str) or not text.strip():
        raise ValueError("EnPL planning returned an empty plan")
    plan = text.strip()
    numbered_steps = len(NUMBERED_PLAN_STEP.findall(plan))
    bulleted_steps = len(BULLETED_PLAN_STEP.findall(plan))
    steps = numbered_steps + bulleted_steps
    if numbered_steps and not bulleted_steps:
        plan_format = "numbered"
    elif bulleted_steps and not numbered_steps:
        plan_format = "bulleted"
    elif steps:
        plan_format = "mixed_list"
    else:
        plan_format = "free_text"
    return plan, steps, plan_format, steps >= 2


class EnPLPlanner:
    def __init__(
        self,
        generation,
        scenario,
        retriever,
        plan_decoding=None,
        response_decoding=None,
        max_input_characters=12000,
    ):
        self.generation = generation
        self.scenario = scenario
        self.retriever = retriever
        self.plan_decoding = plan_decoding or Decoding(max_tokens=256)
        self.response_decoding = response_decoding or Decoding(max_tokens=128)
        if not isinstance(max_input_characters, int) or max_input_characters < 1000:
            raise ValueError("invalid EnPL context character budget")
        self.max_input_characters = max_input_characters
        self.seed_context = scenario["id"]

    def _bounded_plan_prompt(self, state, retrieved):
        examples = list(retrieved)
        history_characters = sum(len(message.content) for message in state.history)
        while True:
            prompt = render_plan_prompt(self.scenario, examples)
            if len(prompt) + history_characters <= self.max_input_characters:
                return prompt, tuple(examples)
            if not examples:
                raise ValueError("EnPL plan request exceeds the frozen context budget")
            examples.pop()

    def _check_response_budget(self, state, prompt):
        characters = len(prompt) + sum(len(message.content) for message in state.history)
        if characters > self.max_input_characters:
            raise ValueError("EnPL response request exceeds the frozen context budget")
        return characters

    def act(self, state, target_nodes, seed, request_id):
        del target_nodes
        started = time.monotonic()
        retrieved = self.retriever.select(
            self.scenario["profile"], self.scenario["target"]
        )
        plan_prompt, examples = self._bounded_plan_prompt(state, retrieved)
        plan_input_characters = len(plan_prompt) + sum(
            len(message.content) for message in state.history
        )
        plan_id = request_id + "/enpl-plan"
        plan_result = self.generation.generate(
            [Message("system", plan_prompt), *state.history],
            "planner",
            self.plan_decoding,
            derived_seed(seed, self.seed_context, "plan"),
            plan_id,
        )
        generated_plan, plan_steps, plan_format, plan_format_valid = _parsed_plan(
            plan_result.text
        )
        response_prompt = render_response_prompt(self.scenario, generated_plan)
        response_input_characters = self._check_response_budget(state, response_prompt)
        response_id = request_id + "/enpl-response"
        response_result = self.generation.generate(
            [Message("system", response_prompt), *state.history],
            "assistant",
            self.response_decoding,
            derived_seed(seed, self.seed_context, "final"),
            response_id,
        )
        response, format_valid = _single_response(response_result.text)
        return Plan(
            response=response,
            covered=False,
            reference_id=None,
            directed_distances={},
            path=[],
            selected_candidate=None,
            rollout_scores=[],
            compute={
                "strategy": "enpl-domain-entity-3-history-conditioned",
                "policy_version": ENPL_POLICY_VERSION,
                "planning_mode": "history_conditioned",
                "retrieval_domain": target_domain(self.scenario["target"]),
                "plan_format": plan_format,
                "plan_format_valid": plan_format_valid,
                "plan_format_violation": not plan_format_valid,
                "plan_finish_reason": plan_result.finish_reason,
                "plan_truncated": plan_result.finish_reason == "length",
                "plan_steps": plan_steps,
                "generated_plan": generated_plan,
                "generated_plan_sha256": digest(generated_plan),
                "plan_prompt_sha256": digest(plan_prompt),
                "response_prompt_sha256": digest(response_prompt),
                "template_sha256": template_hashes(),
                "retrieved_entry_ids": [item.entry.entry_id for item in examples],
                "retrieved_group_ids": [
                    item.entry.duplicate_group_id for item in examples
                ],
                "retrieval_overlap_scores": [item.overlap_score for item in examples],
                "zero_overlap_fallbacks": [
                    item.zero_overlap_fallback for item in examples
                ],
                "examples_requested": 3,
                "examples_used": len(examples),
                "examples_dropped_for_context": len(retrieved) - len(examples),
                "max_input_characters": self.max_input_characters,
                "plan_input_characters": plan_input_characters,
                "response_input_characters": response_input_characters,
                "response_format_valid": format_valid,
                "generation_requests": 2,
                "planner_generation_requests": 1,
                "response_generation_requests": 1,
                "embedding_requests": 0,
                "retrieval_requests": 1,
                "search_simulations": 0,
                "recovery_triggered": False,
                "planner_latency_s": time.monotonic() - started,
            },
            replay_ids=[plan_id, response_id],
        )
