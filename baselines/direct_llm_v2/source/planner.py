"""One-call, target-conditioned local-LLM policy for the shared evaluator."""

import time

from ...contracts import Decoding, Message, Plan
from ...evaluation import visible_context
from ...planner import (
    ASSISTANT_POLICY,
    derived_seed,
    target_fact_guidance,
)

DIRECT_LLM_POLICY_VERSION = "target-conditioned-one-call-v1"
DIRECT_LLM_V2_POLICY_VERSION = "tdrl-final-scaffold-no-guidance-v2"
DIRECT_LLM_METHOD = "DirectLLM"
DIRECT_LLM_V2_METHOD = "DirectLLM-v2"
DIRECT_LLM_METHODS = (DIRECT_LLM_METHOD, DIRECT_LLM_V2_METHOD)


def direct_system_prompt(scenario):
    """Expose exactly the target/profile/knowledge context used by the shared planner."""
    return ASSISTANT_POLICY + "\nVisible scenario context: " + visible_context(scenario)


def direct_v2_system_prompt(scenario, dataset):
    """Use TDRL's final-response scaffold with no planner-produced guidance."""
    context = visible_context(scenario)
    if dataset == "TopDial":
        grounding = target_fact_guidance(context)
    elif dataset == "DuRecDial2.0":
        grounding = ""
    else:
        raise ValueError(f"unsupported DirectLLM-v2 dataset: {dataset}")
    return ASSISTANT_POLICY + "\n" + context + "\nGuidance: " + grounding


def policy_version(method):
    if method == DIRECT_LLM_METHOD:
        return DIRECT_LLM_POLICY_VERSION
    if method == DIRECT_LLM_V2_METHOD:
        return DIRECT_LLM_V2_POLICY_VERSION
    raise ValueError(f"unsupported DirectLLM method: {method}")


def prompt_hashes(method):
    """Return immutable prompt hashes in the v1-compatible report representation."""
    from ...provenance import digest

    fixture = {
        "id": "prompt-contract",
        "target": ["Movie recommendation", "Example"],
        "profile": {"preference": "example"},
        "knowledge": [["Example", "type", "movie"]],
    }
    if method == DIRECT_LLM_METHOD:
        return digest(direct_system_prompt(fixture))
    if method == DIRECT_LLM_V2_METHOD:
        return {
            dataset: digest(direct_v2_system_prompt(fixture, dataset))
            for dataset in ("TopDial", "DuRecDial2.0")
        }
    raise ValueError(f"unsupported DirectLLM method: {method}")


class DirectLLMPlanner:
    """Generate one final assistant response without retrieval, search, or training."""

    def __init__(self, generation, scenario, decoding=None):
        self.generation = generation
        self.scenario = scenario
        self.decoding = decoding or Decoding(max_tokens=128)
        self.system_prompt = direct_system_prompt(scenario)
        self.seed_context = scenario["id"]
        self.policy_version = DIRECT_LLM_POLICY_VERSION
        self.strategy = "direct-llm"

    def act(self, state, target_nodes, seed, request_id):
        del target_nodes
        started = time.monotonic()
        generation_id = request_id + "/final"
        response = self.generation.generate(
            [Message("system", self.system_prompt), *state.history],
            "assistant",
            self.decoding,
            derived_seed(seed, self.seed_context, "final"),
            generation_id,
        )
        return Plan(
            response=response.text,
            covered=False,
            reference_id=None,
            directed_distances={},
            path=[],
            selected_candidate=None,
            rollout_scores=[],
            compute={
                "strategy": self.strategy,
                "policy_version": self.policy_version,
                "generation_requests": 1,
                "embedding_requests": 0,
                "retrieval_requests": 0,
                "search_simulations": 0,
                "recovery_triggered": False,
                "planner_latency_s": time.monotonic() - started,
            },
            replay_ids=[generation_id],
        )


class DirectLLMV2Planner(DirectLLMPlanner):
    """One-call control using TDRL's response scaffold and no planner guidance."""

    def __init__(self, generation, scenario, dataset, decoding=None):
        super().__init__(generation, scenario, decoding)
        self.system_prompt = direct_v2_system_prompt(scenario, dataset)
        self.policy_version = DIRECT_LLM_V2_POLICY_VERSION
        self.strategy = "direct-llm-v2-tdrl-scaffold-no-guidance"


def planner_for_method(method, generation, scenario, dataset, decoding=None):
    if method == DIRECT_LLM_METHOD:
        return DirectLLMPlanner(generation, scenario, decoding)
    if method == DIRECT_LLM_V2_METHOD:
        return DirectLLMV2Planner(generation, scenario, dataset, decoding)
    raise ValueError(f"unsupported DirectLLM method: {method}")
