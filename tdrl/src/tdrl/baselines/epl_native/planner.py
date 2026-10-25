import json
import time
from dataclasses import dataclass

import torch

from ...autonomy import REFUSAL_RESPONSE, explicit_refusal
from ...contracts import Message, Plan
from ...evaluation import simulator_prompt
from ...planner import ASSISTANT_POLICY, derived_seed
from ...provenance import digest
from ...state import DialogueState
from .actions import NativeAction
from .search import NATIVE_SEARCH_REVISION, NativeExperiencePUCT
from .serialization import encode_policy_input

METHOD_REVISION = "epl-native-core-qwen-fixed400-v1"
METHOD_CONTRACT = {
    "revision": METHOD_REVISION,
    "policy": "released_BERT_goal_topic_classifier_with_target_relative_topics",
    "search": NATIVE_SEARCH_REVISION,
    "transition": "shared_local_Qwen3.5-9B_realizer_and_user_simulator",
    "value": "released_MiniLM_experience_retrieval_shape",
    "final_action": "maximum_backed_up_mean_return",
}

NATIVE_REALIZER = (
    "Execute exactly the supplied EPL goal-topic action as the next conversational move. "
    "Use only visible profile, knowledge and dialogue. Be natural and truthful, do not reveal "
    "the action or planning process, and return only the assistant reply."
)


def action_instruction(action, scenario):
    topic = action.resolved_topic(scenario)
    goal = action.goal.casefold()
    if "goodbye" in goal:
        behavior = "End the conversation politely without introducing another recommendation."
    elif "greeting" in goal:
        behavior = "Give a concise greeting and begin the conversation naturally."
    elif "recommendation" in goal:
        behavior = "Recommend or explain the specified topic, grounding claims in visible knowledge."
    elif goal.startswith("ask") or goal == "q&a":
        behavior = "Ask or answer the question needed to advance this goal."
    elif "play music" in goal or "music on demand" in goal:
        behavior = "Help with the requested music task without claiming unavailable device control."
    else:
        behavior = "Advance this dialogue goal directly and concisely."
    return f"Goal: {action.goal}. Topic: {topic}. {behavior}"


@dataclass(frozen=True)
class NativeExperienceSample:
    assistant_response: str
    user_response: str
    reward: float
    terminal: bool
    refused: bool
    next_state: tuple
    retrieval_evidence: tuple[dict, ...]
    request_ids: tuple[str, ...]


class NativeRolloutProvider:
    def __init__(
        self,
        generation,
        memory,
        scenario,
        protocol,
        seed,
        request_base,
        assistant_decoding,
        simulator_decoding,
    ):
        self.generation, self.memory, self.scenario = generation, memory, scenario
        self.protocol, self.seed, self.request_base = protocol, seed, request_base
        self.assistant_decoding, self.simulator_decoding = (
            assistant_decoding,
            simulator_decoding,
        )

    def _prompt(self, action):
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
                NATIVE_REALIZER,
                "Visible scenario context: " + visible,
                "Required EPL action: " + action_instruction(action, self.scenario),
            )
        )

    def expand(self, state, action_index, action_name, path, simulation, depth):
        del action_index
        action = NativeAction.from_key(action_name)
        if not isinstance(state, DialogueState):
            state = DialogueState(tuple(state))
        branch = digest(path)[:16]
        prefix = (
            f"{self.request_base}/simulation/{simulation}/depth/{depth}/path/{branch}"
        )
        assistant_id = prefix + "/assistant"
        assistant = self.generation.generate(
            [Message("system", self._prompt(action)), *state.history],
            "epl_native_rollout_assistant",
            self.assistant_decoding,
            derived_seed(self.seed, prefix, "assistant"),
            assistant_id,
        )
        after_assistant = state.append(Message("assistant", assistant.text))
        flipped = [
            Message("assistant" if item.role == "user" else "user", item.content)
            for item in after_assistant.history
        ]
        user_id = prefix + "/user"
        user = self.generation.generate(
            [Message("system", simulator_prompt(self.scenario, self.protocol)), *flipped],
            "epl_native_rollout_user",
            self.simulator_decoding,
            derived_seed(self.seed, prefix, "user"),
            user_id,
        )
        next_state = after_assistant.append(Message("user", user.text))
        embedding_id = prefix + "/experience"
        reward, evidence, _ = self.memory.lookup(self.scenario, next_state, embedding_id)
        return NativeExperienceSample(
            assistant.text,
            user.text,
            reward,
            action.goal == "Say goodbye",
            explicit_refusal(next_state),
            next_state.history,
            tuple(evidence),
            (assistant_id, user_id, embedding_id),
        )


class EPLNativePlanner:
    def __init__(
        self,
        generation,
        model,
        tokenizer,
        memory,
        config,
        vocabulary,
        scenario,
        checkpoint_sha256,
        protocol,
        assistant_decoding,
        simulator_decoding,
    ):
        self.generation, self.model, self.tokenizer = generation, model, tokenizer
        self.memory, self.config, self.vocabulary = memory, config, vocabulary
        self.scenario, self.checkpoint_sha256 = scenario, checkpoint_sha256
        self.protocol = protocol
        self.assistant_decoding, self.simulator_decoding = (
            assistant_decoding,
            simulator_decoding,
        )
        self.committed_actions = []

    def _prompt(self, action):
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
                NATIVE_REALIZER,
                "Visible scenario context: " + visible,
                "Selected EPL action: " + action_instruction(action, self.scenario),
            )
        )

    def _policy(self, state, search_path=()):
        simulated_actions = tuple(NativeAction.from_key(value) for value in search_path)
        ids, mask = encode_policy_input(
            self.tokenizer,
            self.scenario,
            state.history,
            (*self.committed_actions, *simulated_actions),
            self.config.max_sequence_length,
        )
        parameter = next(self.model.parameters())
        with torch.inference_mode(), torch.autocast(
            device_type="cuda", dtype=torch.bfloat16
        ):
            output = self.model(
                ids[None].to(parameter.device), mask[None].to(parameter.device)
            )
        probabilities = output["probabilities"][0].float().cpu()
        top = torch.topk(probabilities, self.config.policy_top_k).indices.tolist()
        masked = torch.zeros_like(probabilities)
        masked[top] = probabilities[top]
        masked /= masked.sum()
        return masked.tolist()

    def _policy_plan(self, state, seed, request_id):
        probabilities = self._policy(state)
        index = max(range(len(probabilities)), key=lambda item: (probabilities[item], -item))
        action = self.vocabulary.items[index]
        generation_id = request_id + "/policy-realization"
        response = self.generation.generate(
            [Message("system", self._prompt(action)), *state.history],
            "epl_native_self_simulation_assistant",
            self.assistant_decoding,
            derived_seed(seed, self.scenario["id"], "native-policy-realization"),
            generation_id,
        )
        self.committed_actions.append(action)
        return Plan(
            response.text,
            False,
            None,
            {},
            [action.key],
            index,
            probabilities,
            {
                "strategy": "epl-native-policy-self-simulation",
                "selected_goal": action.goal,
                "selected_topic_role": action.topic_role,
                "generation_requests": 1,
                "embedding_requests": 0,
                "checkpoint_sha256": self.checkpoint_sha256,
                "config_sha256": self.config.sha256,
                "planner_latency_s": 0.0,
            },
            [generation_id],
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
                    "strategy": "epl-native-core",
                    "refusal_guard": True,
                    "generation_requests": 0,
                    "embedding_requests": 0,
                    "planner_latency_s": time.monotonic() - started,
                },
                [],
            )
        rollout = NativeRolloutProvider(
            self.generation,
            self.memory,
            self.scenario,
            self.protocol,
            seed,
            request_id,
            self.assistant_decoding,
            self.simulator_decoding,
        )
        search = NativeExperiencePUCT(
            self.config.actions,
            self._policy,
            rollout,
            self.config,
            self.config.puct_max_depth,
        ).search(state)
        action = self.vocabulary.items[search.action_index]
        generation_id = request_id + "/final-realization"
        response = self.generation.generate(
            [Message("system", self._prompt(action)), *state.history],
            "assistant",
            self.assistant_decoding,
            derived_seed(seed, self.scenario["id"], "native-final-realization"),
            generation_id,
        )
        self.committed_actions.append(action)
        compute = {
            "strategy": "epl-native-core",
            "method_revision": METHOD_REVISION,
            "method_contract": METHOD_CONTRACT,
            "selected_goal": action.goal,
            "selected_topic_role": action.topic_role,
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
            "embedding_requests": sum(
                item.endswith("/experience") for item in search.request_ids
            ),
            "checkpoint_sha256": self.checkpoint_sha256,
            "config_sha256": self.config.sha256,
            "vocabulary_sha256": self.vocabulary.sha256,
            "refusal_guard": False,
            "recovery_triggered": False,
            "planner_latency_s": time.monotonic() - started,
        }
        return Plan(
            response.text,
            False,
            None,
            {},
            [action.key],
            search.action_index,
            list(search.root_q_values),
            compute,
            [*search.request_ids, generation_id],
        )


class EPLNativePolicyPlanner(EPLNativePlanner):
    def act(self, state, target_nodes, seed, request_id):
        del target_nodes
        return self._policy_plan(state, seed, request_id)
