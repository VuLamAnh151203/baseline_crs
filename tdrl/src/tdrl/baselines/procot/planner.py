"""Faithful one-call ProCoT planner with deterministic response extraction."""

import time

from ...contracts import Decoding, Message, Plan
from ...planner import derived_seed
from ...provenance import digest
from .parser import parse_procot
from .prompts import PROCOT_POLICY_VERSION, procot_system_prompt


class ProCoTPlanner:
    def __init__(self, generation, scenario, decoding=None):
        self.generation = generation
        self.scenario = scenario
        self.decoding = decoding or Decoding(max_tokens=128)
        self.system_prompt = procot_system_prompt(scenario)
        self.seed_context = scenario["id"]

    def act(self, state, target_nodes, seed, request_id):
        del target_nodes
        started = time.monotonic()
        generation_id = request_id + "/procot"
        generated = self.generation.generate(
            [Message("system", self.system_prompt), *state.history],
            "assistant",
            self.decoding,
            derived_seed(seed, self.seed_context, "final"),
            generation_id,
        )
        parsed = parse_procot(generated.text)
        return Plan(
            response=parsed.response,
            covered=False,
            reference_id=None,
            directed_distances={},
            path=[],
            selected_candidate=None,
            rollout_scores=[],
            compute={
                "strategy": "procot-one-shot",
                "policy_version": PROCOT_POLICY_VERSION,
                "format_valid": parsed.format_valid,
                "format_violation": not parsed.format_valid,
                "current_topics": parsed.current_topics,
                "next_topics": parsed.next_topics,
                "planning_sha256": digest(parsed.planning_text),
                "raw_output_sha256": digest(generated.text),
                "final_response_sha256": digest(parsed.response),
                "generation_requests": 1,
                "embedding_requests": 0,
                "retrieval_requests": 0,
                "search_simulations": 0,
                "recovery_triggered": False,
                "planner_latency_s": time.monotonic() - started,
            },
            replay_ids=[generation_id],
        )
