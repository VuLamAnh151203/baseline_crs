import json
import time
from dataclasses import asdict

import torch

from ...autonomy import REFUSAL_RESPONSE, explicit_refusal
from ...contracts import Decoding, Message, Plan
from ...planner import ASSISTANT_POLICY, derived_seed
from ...provenance import digest
from ...strategies import ACTION_INSTRUCTIONS
from .contracts import PolicyOutput
from .mcts import OpenLoopMCTS
from .rollout import QwenRolloutProvider
from .serialization import encode_policy_input

DPDP_REALIZER_POLICY = (
    "Follow the supplied DPDP dialogue action as a hard constraint for the next conversational "
    "move. Realize exactly that one move; do not add a question, recommendation, explanation, or "
    "farewell unless the action instruction explicitly calls for it. The target never overrides "
    "the selected action. Use only visible scenario context, never expose the action label, and "
    "return only the final response."
)


class DPDPPlanner:
    def __init__(
        self,
        generation,
        model,
        tokenizer,
        config,
        scenario,
        checkpoint_sha256,
        decoding=None,
        protocol="profile",
    ):
        self.generation = generation
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.scenario = scenario
        self.checkpoint_sha256 = checkpoint_sha256
        self.decoding = decoding or Decoding(max_tokens=128)
        self.protocol = protocol
        self.seed_context = scenario["id"]
        self.simulator_prompt = ""

    def _history(self, state):
        return state.history if hasattr(state, "history") else tuple(state)

    def _policy(self, state):
        ids, mask = encode_policy_input(
            self.tokenizer,
            self.scenario,
            self._history(state),
            self.config.max_sequence_length,
            self.config.header_tokens,
        )
        parameter = next(self.model.parameters())
        ids, mask = ids[None].to(parameter.device), mask[None].to(parameter.device)
        with torch.inference_mode():
            output = self.model(ids, mask)
        probabilities = output["action_probs"][0].float().cpu().tolist()
        q_values = output["q_values"][0].float().cpu().tolist()
        top = sorted(probabilities, reverse=True)
        trace = PolicyOutput(tuple(probabilities), tuple(q_values), top[0] - top[1])
        return probabilities, q_values, trace

    def _prompt(self, action):
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
                DPDP_REALIZER_POLICY,
                "Visible scenario context: " + visible,
                "DPDP next action: " + action,
                "Action instruction: " + ACTION_INSTRUCTIONS[action],
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
                    "strategy": "dpdp",
                    "selected_system": "refusal_guard",
                    "generation_requests": 0,
                    "embedding_requests": 0,
                    "refusal_guard": True,
                    "recovery_triggered": False,
                    "planner_latency_s": time.monotonic() - started,
                },
                [],
            )
        probabilities, q_values, policy_trace = self._policy(state)
        if policy_trace.confidence_gap >= self.config.gate_threshold:
            action_index = max(
                range(len(probabilities)), key=lambda index: (probabilities[index], -index)
            )
            action = self.config.actions[action_index]
            generation_id = request_id + "/system1/final"
            response = self.generation.generate(
                [Message("system", self._prompt(action)), *state.history],
                "assistant",
                self.decoding,
                derived_seed(seed, self.seed_context, "dpdp_system1"),
                generation_id,
            )
            text, replay_ids, mcts_trace, system = response.text, [generation_id], None, "system1"
        else:
            rollout = QwenRolloutProvider(
                self.generation,
                self.scenario,
                self.protocol,
                seed,
                request_id + "/system2",
                self.config.critic_samples,
                self.decoding,
                Decoding(max_tokens=128),
            )
            initial_assistants = 0
            for message in self.scenario["messages"]:
                if message["role"] == "assistant":
                    initial_assistants += 1
                if message["role"] == "user":
                    break
            completed_turns = (
                sum(message.role == "assistant" for message in state.history)
                - initial_assistants
            )
            remaining_turns = max(1, 10 - completed_turns)
            search = OpenLoopMCTS(
                self.config.actions,
                lambda candidate_state: self._policy(candidate_state)[:2],
                rollout,
                self.config.mcts_simulations,
                self.config.mcts_cpuct,
                self.config.mcts_q0,
                min(self.config.mcts_max_depth, remaining_turns),
                self.config.mcts_max_realizations,
            )
            mcts_trace = search.search(state)
            action_index, action = mcts_trace.action_index, mcts_trace.action_name
            text, replay_ids, system = mcts_trace.response, list(mcts_trace.request_ids), "system2"
        compute = {
            "strategy": "dpdp",
            "selected_system": system,
            "selected_action": action,
            "action_index": action_index,
            "policy_probabilities": probabilities,
            "q_values": q_values,
            "confidence_gap": policy_trace.confidence_gap,
            "gate_threshold": self.config.gate_threshold,
            "mcts_simulations": mcts_trace.simulations if mcts_trace else 0,
            "mcts_expanded_nodes": mcts_trace.expanded_nodes if mcts_trace else 0,
            "mcts_depth": mcts_trace.max_depth_reached if mcts_trace else 0,
            "mcts_root_visits": list(mcts_trace.root_visits) if mcts_trace else [],
            "mcts_root_q_values": list(mcts_trace.root_q_values) if mcts_trace else [],
            "mcts_root_priors": list(mcts_trace.root_priors) if mcts_trace else [],
            "mcts_tree_sha256": digest(mcts_trace.tree) if mcts_trace else None,
            "mcts_tree": mcts_trace.tree if mcts_trace else None,
            "generation_requests": len(replay_ids),
            "embedding_requests": 0,
            "checkpoint_sha256": self.checkpoint_sha256,
            "dpdp_config_sha256": self.config.sha256,
            "policy_trace": asdict(policy_trace),
            "refusal_guard": False,
            "recovery_triggered": system == "system2",
            "planner_latency_s": time.monotonic() - started,
        }
        return Plan(
            response=text,
            covered=False,
            reference_id=None,
            directed_distances={},
            path=[action],
            selected_candidate=action_index,
            rollout_scores=list(mcts_trace.root_q_values) if mcts_trace else list(q_values),
            compute=compute,
            replay_ids=replay_ids,
        )
