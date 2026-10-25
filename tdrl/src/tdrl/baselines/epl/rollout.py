import json
from dataclasses import dataclass

from ...autonomy import explicit_refusal
from ...contracts import Decoding, Message
from ...evaluation import simulator_prompt
from ...planner import ASSISTANT_POLICY, derived_seed
from ...state import DialogueState
from ...strategies import ACTION_INSTRUCTIONS

EPL_ROLLOUT_POLICY = (
    "This is an internal T-EPL planning transition. Follow exactly the supplied dialogue action. "
    "Use only visible profile and knowledge, remain natural and truthful, never reveal the action "
    "label or planning process, and return only the assistant reply."
)

EPL_ROLLOUT_TERMINAL_REVISION = "source-farewell-or-search-horizon-v2"


def source_rollout_terminal(action_name):
    """Match the released dialogue environment's action-based termination."""
    return action_name == "farewell"


@dataclass(frozen=True)
class ExperienceSample:
    assistant_response: str
    user_response: str
    reward: float
    terminal: bool
    refused: bool
    next_state: tuple
    retrieval_evidence: tuple[dict, ...]
    request_ids: tuple[str, ...]


class EPLRolloutProvider:
    def __init__(
        self,
        generation,
        memory,
        scenario,
        protocol,
        seed,
        request_base,
        assistant_decoding=None,
        simulator_decoding=None,
    ):
        self.generation = generation
        self.memory = memory
        self.scenario = scenario
        self.protocol = protocol
        self.seed = seed
        self.request_base = request_base
        self.assistant_decoding = assistant_decoding or Decoding(max_tokens=128)
        self.simulator_decoding = simulator_decoding or Decoding(max_tokens=128)

    def _context(self):
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
        if not isinstance(state, DialogueState):
            state = DialogueState(tuple(state))
        branch = "/".join(path)
        prefix = f"{self.request_base}/simulation/{simulation}/depth/{depth}/path/{branch}"
        assistant_id = prefix + "/assistant"
        assistant = self.generation.generate(
            [
                Message(
                    "system",
                    "\n".join(
                        (
                            ASSISTANT_POLICY,
                            EPL_ROLLOUT_POLICY,
                            "Visible scenario context: " + self._context(),
                            "Required next dialogue action: " + action_name,
                            "Action instruction: " + ACTION_INSTRUCTIONS[action_name],
                        )
                    ),
                ),
                *state.history,
            ],
            "epl_rollout_assistant",
            self.assistant_decoding,
            derived_seed(self.seed, prefix, "assistant"),
            assistant_id,
        )
        after_assistant = state.append(Message("assistant", assistant.text))
        flipped = [
            Message("assistant" if message.role == "user" else "user", message.content)
            for message in after_assistant.history
        ]
        user_id = prefix + "/user"
        user = self.generation.generate(
            [Message("system", simulator_prompt(self.scenario, self.protocol)), *flipped],
            "epl_rollout_user",
            self.simulator_decoding,
            derived_seed(self.seed, prefix, "user"),
            user_id,
        )
        next_state = after_assistant.append(Message("user", user.text))
        embedding_id = prefix + "/experience"
        reward, evidence, _ = self.memory.lookup(self.scenario, next_state, embedding_id)
        refused = explicit_refusal(next_state)
        return ExperienceSample(
            assistant.text,
            user.text,
            reward,
            source_rollout_terminal(action_name),
            refused,
            next_state.history,
            tuple(evidence),
            (assistant_id, user_id, embedding_id),
        )
