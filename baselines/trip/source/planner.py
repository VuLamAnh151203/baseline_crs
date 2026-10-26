import json
import time

from ...autonomy import REFUSAL_RESPONSE, explicit_refusal
from ...contracts import Decoding, Message, Plan
from ...planner import ASSISTANT_POLICY, derived_seed
from ...provenance import digest
from .dataset import inference_inputs
from .decoding import generate_plan

TRIP_REALIZER_POLICY = (
    "A TRIP target-constrained action/topic plan is supplied below. Follow only its next move when "
    "it fits the observed conversation. Use only visible scenario knowledge, respond naturally to "
    "the current user, never quote the plan, and stop persuasion after an explicit refusal."
)


class TripPlanner:
    def __init__(
        self,
        generation,
        model,
        tokenizer,
        token_ids,
        trip_config,
        scenario,
        checkpoint_sha256,
        decoding=None,
    ):
        self.generation = generation
        self.model = model
        self.tokenizer = tokenizer
        self.token_ids = token_ids
        self.trip_config = trip_config
        self.scenario = scenario
        self.checkpoint_sha256 = checkpoint_sha256
        self.decoding = decoding or Decoding(max_tokens=128)
        self.seed_context = scenario["id"]
        self.simulator_prompt = ""

    def _prompt(self, plan):
        pairs = "\n".join(
            f"{index + 1}. action={action}; topic={topic}"
            for index, (action, topic) in enumerate(plan.pairs)
        )
        visible = json.dumps(
            {
                "profile": self.scenario["profile"],
                "personality": self.scenario.get("personality", {}),
                "knowledge": self.scenario["knowledge"],
                "target": self.scenario["target"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        return "\n".join(
            (
                ASSISTANT_POLICY,
                TRIP_REALIZER_POLICY,
                "Visible scenario context:",
                visible,
                "TRIP plan:",
                pairs,
            )
        )

    def act(self, state, target_nodes, seed, request_id):
        del target_nodes
        started = time.monotonic()
        if explicit_refusal(state):
            return Plan(
                REFUSAL_RESPONSE,
                False,
                None,
                {},
                [],
                None,
                [],
                {
                    "strategy": "trip",
                    "generation_requests": 0,
                    "embedding_requests": 0,
                    "refusal_guard": True,
                    "recovery_triggered": False,
                    "planner_latency_s": time.monotonic() - started,
                },
                [],
            )
        context, target, constraint = inference_inputs(
            self.scenario,
            state,
            self.tokenizer,
            self.trip_config.max_context_tokens,
            self.trip_config.max_plan_tokens,
        )
        generated = generate_plan(
            self.model,
            self.tokenizer,
            self.token_ids,
            context,
            target,
            constraint,
            self.trip_config.decoding,
        )
        generation_id = request_id + "/final"
        response = self.generation.generate(
            [Message("system", self._prompt(generated)), *state.history],
            "assistant",
            self.decoding,
            derived_seed(seed, self.seed_context, "final"),
            generation_id,
        )
        path = [f"{action} :: {topic}" for action, topic in generated.pairs]
        return Plan(
            response=response.text,
            covered=False,
            reference_id=None,
            directed_distances={},
            path=path,
            selected_candidate=0,
            rollout_scores=[generated.adjusted_score],
            compute={
                "strategy": "trip",
                "next_action": generated.pairs[0][0],
                "next_topic": generated.pairs[0][1],
                "generation_requests": 1,
                "embedding_requests": 0,
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
                "total_tokens": response.total_tokens,
                "forward_candidates": generated.forward_candidates,
                "backward_candidates": generated.backward_candidates,
                "trip_sequence_forwards": generated.sequence_forwards,
                "bidirectional_distance": generated.agreement_distance,
                "beam_score": generated.beam_score,
                "adjusted_score": generated.adjusted_score,
                "checkpoint_sha256": self.checkpoint_sha256,
                "trip_config_sha256": self.trip_config.sha256,
                "visible_state_sha256": digest(json.loads(state.serialize())),
                "prompt_sha256": digest(TRIP_REALIZER_POLICY),
                "refusal_guard": False,
                "recovery_triggered": False,
                "planner_latency_s": time.monotonic() - started,
            },
            replay_ids=[generation_id],
        )
