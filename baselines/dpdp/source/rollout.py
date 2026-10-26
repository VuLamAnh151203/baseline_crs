import json

from ...acceptance import classify_user
from ...autonomy import explicit_refusal
from ...contracts import Decoding, Message
from ...evaluation import simulator_prompt
from ...planner import ASSISTANT_POLICY, derived_seed
from ...state import DialogueState
from ...strategies import ACTION_INSTRUCTIONS
from .concurrency import ordered_generate
from .contracts import RewardSample
from .rewards import REWARD_VALUES, critic_messages, parse_reward_label

ROLLOUT_ASSISTANT_POLICY = (
    "This is an internal DPDP tree rollout. Follow the supplied dialogue action as a hard "
    "constraint. Realize exactly that one move; do not add a question, recommendation, "
    "explanation, or farewell unless the action instruction explicitly calls for it. The target "
    "never overrides the selected action. Remain truthful, natural, and grounded in visible "
    "scenario context. Return only the assistant reply."
)
MAX_CRITIC_RETRIES = 2


def _as_state(state):
    return state if isinstance(state, DialogueState) else DialogueState(tuple(state))


class QwenRolloutProvider:
    """Local-Qwen realization, user simulation and critic for internal MCTS branches."""

    def __init__(
        self,
        backend,
        scenario,
        protocol,
        seed,
        request_base,
        critic_samples,
        assistant_decoding=None,
        simulator_decoding=None,
    ):
        if critic_samples < 1:
            raise ValueError("DPDP rollout critic requires at least one sample")
        self.backend = backend
        self.scenario = scenario
        self.protocol = protocol
        self.seed = seed
        self.request_base = request_base
        self.critic_samples = critic_samples
        self.assistant_decoding = assistant_decoding or Decoding(max_tokens=128)
        self.simulator_decoding = simulator_decoding or Decoding(max_tokens=128)

    def _visible_context(self):
        return json.dumps(
            {
                "target": self.scenario["target"],
                "profile": self.scenario["profile"],
                "personality": self.scenario.get("personality", {}),
                "knowledge": self.scenario["knowledge"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )

    def expand(self, state, action_index, action_name, path, simulation, depth):
        del action_index
        state = _as_state(state)
        branch = "/".join(path)
        prefix = f"{self.request_base}/sim/{simulation}/depth/{depth}/path/{branch}"
        assistant_id = prefix + "/assistant"
        assistant = self.backend.generate(
            [
                Message(
                    "system",
                    "\n".join(
                        (
                            ASSISTANT_POLICY,
                            ROLLOUT_ASSISTANT_POLICY,
                            "Visible context: " + self._visible_context(),
                            "Required next dialogue action: " + action_name,
                            "Action instruction: " + ACTION_INSTRUCTIONS[action_name],
                        )
                    ),
                ),
                *state.history,
            ],
            "dpdp_rollout_assistant",
            self.assistant_decoding,
            derived_seed(self.seed, prefix, "assistant"),
            assistant_id,
        )
        after_assistant = state.append(Message("assistant", assistant.text))
        # The internal simulated user is deliberately separate from run_dialogue's actual user.
        flipped = [
            Message("assistant" if message.role == "user" else "user", message.content)
            for message in after_assistant.history
        ]
        user_id = prefix + "/user"
        user = self.backend.generate(
            [Message("system", simulator_prompt(self.scenario, self.protocol)), *flipped],
            "dpdp_rollout_user",
            self.simulator_decoding,
            derived_seed(self.seed, prefix, "user"),
            user_id,
        )
        next_state = after_assistant.append(Message("user", user.text))
        labels = [None] * self.critic_samples
        request_ids = [assistant_id, user_id]
        pending = list(range(self.critic_samples))
        last_errors = {}
        context = json.loads(self._visible_context())
        for attempt in range(MAX_CRITIC_RETRIES + 1):
            if not pending:
                break
            critic_requests = []
            for sample_index in pending:
                critic_id = prefix + f"/critic/{sample_index}/attempt/{attempt}"
                critic_requests.append(
                    (
                        critic_messages(context, next_state.history, revise=attempt > 0),
                        "dpdp_rollout_critic",
                        Decoding(temperature=0.7, top_p=0.8, top_k=20, max_tokens=8),
                        derived_seed(self.seed, prefix, "critic", sample_index, attempt),
                        critic_id,
                    )
                )
                request_ids.append(critic_id)
            retry = []
            for sample_index, critic in zip(
                pending, ordered_generate(self.backend, critic_requests), strict=True
            ):
                try:
                    labels[sample_index] = parse_reward_label(critic.text)
                except ValueError as error:
                    last_errors[sample_index] = error
                    retry.append(sample_index)
            pending = retry
        if pending:
            sample_index = pending[0]
            raise ValueError(
                f"DPDP rollout critic failed closed for sample {sample_index}"
            ) from last_errors[sample_index]
        reward = sum(REWARD_VALUES[label] for label in labels) / len(labels)
        # Stable order breaks equal vote counts and is recorded independently of mean reward.
        order = tuple(REWARD_VALUES)
        label = max(order, key=lambda item: (labels.count(item), -order.index(item)))
        accepted = classify_user(next_state, self.scenario["target"]).accepted
        refused = explicit_refusal(next_state)
        critic_accepted = label == "accepted"
        disagreement = critic_accepted != accepted
        terminal_reasons = tuple(
            reason
            for reason, active in (
                ("critic_accepted", critic_accepted),
                ("deterministic_accepted", accepted),
                ("explicit_refusal", refused),
            )
            if active
        )
        return RewardSample(
            assistant_response=assistant.text,
            user_response=user.text,
            reward=reward,
            label=label,
            critic_labels=tuple(labels),
            deterministic_accepted=accepted,
            acceptance_disagreement=disagreement,
            # A majority "accepted" reward must terminate the same simulated
            # trajectory; otherwise repeated +1 rewards inflate its return.
            terminal=bool(critic_accepted or accepted or refused),
            terminal_reasons=terminal_reasons,
            refused=refused,
            next_state=next_state.history,
            request_ids=tuple(request_ids),
        )
