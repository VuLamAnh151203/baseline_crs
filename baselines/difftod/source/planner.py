import json
import time

from ...autonomy import REFUSAL_RESPONSE, explicit_refusal
from ...contracts import Decoding, Message, Plan
from ...planner import ASSISTANT_POLICY, derived_seed
from ...provenance import digest
from .plan_store import PlanStore

DIFFTOD_REALIZER_POLICY = (
    "A frozen DiffTOD trajectory plan is supplied below. Follow its ordering and intent when it is "
    "compatible with the observed conversation, but respond only to the current user turn. Never "
    "quote the plan, invent inaccessible facts, or continue persuasion after refusal."
)


class DiffTODPlanner:
    """Thin Qwen realizer over a precomputed, immutable DiffTOD trajectory plan."""

    def __init__(
        self,
        generation,
        plan_store: PlanStore,
        dataset,
        split,
        scenario_id,
        plan_seed,
        budget,
        visible_context,
        decoding=None,
    ):
        self.generation = generation
        self.plan_store = plan_store
        self.dataset = dataset
        self.split = split
        self.scenario_id = scenario_id
        self.plan_seed = plan_seed
        self.budget = budget
        self.visible_context = visible_context
        self.decoding = decoding or Decoding(max_tokens=128)
        self.artifact = plan_store.load(dataset, split, scenario_id, plan_seed, budget)
        self.plan_file_sha256 = plan_store.hash(
            dataset, split, scenario_id, plan_seed, budget
        )
        self.seed_context = scenario_id
        self.simulator_prompt = ""

    def _system_prompt(self):
        return "\n".join(
            (
                ASSISTANT_POLICY,
                DIFFTOD_REALIZER_POLICY,
                "Visible scenario context:",
                self.visible_context,
                "Frozen DiffTOD trajectory plan:",
                self.artifact.selected_plan,
            )
        )

    def act(self, state, target_nodes, seed, request_id):
        del target_nodes
        started = time.monotonic()
        if explicit_refusal(state):
            return Plan(
                response=REFUSAL_RESPONSE,
                covered=False,
                reference_id=None,
                directed_distances={},
                path=[],
                selected_candidate=self.artifact.selected_candidate,
                rollout_scores=self.artifact.self_bleu_scores,
                compute={
                    "strategy": "difftod",
                    "generation_requests": 0,
                    "embedding_requests": 0,
                    "recovery_triggered": False,
                    "refusal_guard": True,
                    "planner_latency_s": time.monotonic() - started,
                    "plan_artifact_sha256": self.plan_file_sha256,
                },
                replay_ids=[],
            )
        generation_id = request_id + "/final"
        response = self.generation.generate(
            [Message("system", self._system_prompt()), *state.history],
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
            selected_candidate=self.artifact.selected_candidate,
            rollout_scores=self.artifact.self_bleu_scores,
            compute={
                "strategy": "difftod",
                "generation_requests": 1,
                "embedding_requests": 0,
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
                "total_tokens": response.total_tokens,
                "recovery_triggered": False,
                "refusal_guard": False,
                "planner_latency_s": time.monotonic() - started,
                "plan_artifact_sha256": self.plan_file_sha256,
                "plan_contract_sha256": digest(
                    {
                        "dataset": self.dataset,
                        "split": self.split,
                        "scenario_id": self.scenario_id,
                        "plan_seed": self.plan_seed,
                        "budget": self.budget,
                        "checkpoint": self.artifact.checkpoint_sha256,
                        "config": self.artifact.config_sha256,
                    }
                ),
                "diffusion_steps": self.artifact.diffusion_steps,
                "diffusion_model_calls": self.artifact.diffusion_model_calls,
                "diffusion_sequence_forwards": self.artifact.diffusion_sequence_forwards,
                "offline_plan_wall_s": self.artifact.wall_s,
                "offline_plan_peak_memory_bytes": self.artifact.peak_memory_bytes,
                "visible_context_sha256": digest(json.loads(self.visible_context)),
            },
            replay_ids=[generation_id],
        )
