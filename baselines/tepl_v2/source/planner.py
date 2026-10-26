import json
import time

import torch

from ...autonomy import REFUSAL_RESPONSE, explicit_refusal
from ...contracts import Decoding, Message, Plan
from ...planner import ASSISTANT_POLICY, derived_seed
from ...provenance import digest
from ...strategies import ACTION_INSTRUCTIONS
from .rollout import EPL_ROLLOUT_TERMINAL_REVISION, EPLRolloutProvider
from .search import EPL_SEARCH_REVISION, ExperiencePUCT
from .serialization import encode_policy_input

EPL_POLICY_REALIZER = (
    "Follow the supplied EPL policy action as the only next conversational move. "
    "Use visible profile and knowledge, remain natural and truthful, never reveal the action "
    "label, and return only the assistant reply."
)

EPL_METHOD_REVISION = "source-corrected-v2"
EPL_METHOD_CONTRACT = {
    "revision": EPL_METHOD_REVISION,
    "search": EPL_SEARCH_REVISION,
    "final_action": "maximum_backed_up_mean_return",
    "rollout_terminal": EPL_ROLLOUT_TERMINAL_REVISION,
    "final_realization": "fresh_generation_after_action_selection",
}


class EPLPlanner:
    def __init__(
        self,
        generation,
        model,
        tokenizer,
        memory,
        config,
        scenario,
        checkpoint_sha256,
        protocol="profile",
        assistant_decoding=None,
        simulator_decoding=None,
    ):
        self.generation = generation
        self.model = model
        self.tokenizer = tokenizer
        self.memory = memory
        self.config = config
        self.scenario = scenario
        self.checkpoint_sha256 = checkpoint_sha256
        self.protocol = protocol
        self.assistant_decoding = assistant_decoding or Decoding(max_tokens=128)
        self.simulator_decoding = simulator_decoding or Decoding(max_tokens=128)
        self.seed_context = scenario["id"]
        self.simulator_prompt = ""

    def _realizer_prompt(self, action):
        visible = json.dumps(
            {
                "target": self.scenario["target"],
                "profile": self.scenario["profile"],
                "personality": self.scenario.get("personality", {}),
                "knowledge": self.scenario["knowledge"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        return "\n".join(
            (
                ASSISTANT_POLICY,
                EPL_POLICY_REALIZER,
                "Visible scenario context: " + visible,
                "EPL next action: " + action,
                "Action instruction: " + ACTION_INSTRUCTIONS[action],
            )
        )

    def _policy(self, state):
        ids, mask = encode_policy_input(
            self.tokenizer,
            self.scenario,
            state.history,
            self.config.max_sequence_length,
            self.config.header_tokens,
        )
        parameter = next(self.model.parameters())
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = self.model(ids[None].to(parameter.device), mask[None].to(parameter.device))
        probabilities = output["probabilities"][0].float().cpu()
        top = torch.topk(probabilities, self.config.policy_top_k).indices.tolist()
        masked = torch.zeros_like(probabilities)
        masked[top] = probabilities[top]
        masked /= masked.sum()
        return masked.tolist()

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
                    "strategy": "epl",
                    "refusal_guard": True,
                    "recovery_triggered": False,
                    "generation_requests": 0,
                    "embedding_requests": 0,
                    "planner_latency_s": time.monotonic() - started,
                },
                [],
            )
        initial_assistants = 0
        for message in self.scenario["messages"]:
            if message["role"] == "assistant":
                initial_assistants += 1
            if message["role"] == "user":
                break
        completed_turns = sum(message.role == "assistant" for message in state.history) - initial_assistants
        max_depth = min(self.config.puct_max_depth, max(1, 10 - completed_turns))
        rollout = EPLRolloutProvider(
            self.generation,
            self.memory,
            self.scenario,
            self.protocol,
            seed,
            request_id,
            self.assistant_decoding,
            self.simulator_decoding,
        )
        search = ExperiencePUCT(
            self.config.actions, self._policy, rollout, self.config, max_depth
        ).search(state)
        # The source pipeline searches for an action, then realizes that action
        # separately; a simulated tree response is not emitted to the user.
        generation_id = request_id + "/final-realization"
        response = self.generation.generate(
            [Message("system", self._realizer_prompt(search.action_name)), *state.history],
            "assistant",
            self.assistant_decoding,
            derived_seed(seed, self.seed_context, "epl-final-realization"),
            generation_id,
        )
        compute = {
            "strategy": "epl",
            "method_revision": EPL_METHOD_REVISION,
            "search_revision": EPL_METHOD_CONTRACT["search"],
            "final_action_rule": EPL_METHOD_CONTRACT["final_action"],
            "rollout_terminal_rule": EPL_METHOD_CONTRACT["rollout_terminal"],
            "selected_action": search.action_name,
            "action_index": search.action_index,
            "puct_simulations": search.simulations,
            "puct_expanded_nodes": search.expanded_nodes,
            "puct_depth": search.max_depth_reached,
            "puct_root_visits": list(search.root_visits),
            "puct_root_q_values": list(search.root_q_values),
            "puct_root_priors": list(search.root_priors),
            "puct_tree_sha256": digest(search.tree),
            "puct_tree": search.tree,
            "generation_requests": 1
            + sum("/experience" not in item for item in search.request_ids),
            "embedding_requests": sum(item.endswith("/experience") for item in search.request_ids),
            "checkpoint_sha256": self.checkpoint_sha256,
            "epl_config_sha256": self.config.sha256,
            "refusal_guard": False,
            "recovery_triggered": False,
            "planner_latency_s": time.monotonic() - started,
        }
        return Plan(
            response.text,
            False,
            None,
            {},
            [search.action_name],
            search.action_index,
            list(search.root_q_values),
            compute,
            [*search.request_ids, generation_id],
        )


class EPLPolicyPlanner(EPLPlanner):
    """Policy-only actor used to create train-only self-simulation experience."""

    def _prompt(self, action):
        return self._realizer_prompt(action)

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
                    "strategy": "epl-policy-self-simulation",
                    "refusal_guard": True,
                    "recovery_triggered": False,
                    "generation_requests": 0,
                    "embedding_requests": 0,
                    "planner_latency_s": time.monotonic() - started,
                },
                [],
            )
        probabilities = self._policy(state)
        action_index = max(
            range(len(probabilities)), key=lambda index: (probabilities[index], -index)
        )
        action = self.config.actions[action_index]
        generation_id = request_id + "/policy-realization"
        response = self.generation.generate(
            [Message("system", self._prompt(action)), *state.history],
            "epl_self_simulation_assistant",
            self.assistant_decoding,
            derived_seed(seed, self.seed_context, "epl-policy-realization"),
            generation_id,
        )
        return Plan(
            response.text,
            False,
            None,
            {},
            [action],
            action_index,
            probabilities,
            {
                "strategy": "epl-policy-self-simulation",
                "selected_action": action,
                "policy_probabilities": probabilities,
                "generation_requests": 1,
                "embedding_requests": 0,
                "checkpoint_sha256": self.checkpoint_sha256,
                "epl_config_sha256": self.config.sha256,
                "refusal_guard": False,
                "recovery_triggered": False,
                "planner_latency_s": time.monotonic() - started,
            },
            [generation_id],
        )
