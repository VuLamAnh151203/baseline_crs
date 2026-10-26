"""Frozen prompt contract for the target-goal adaptation of ProCoT."""

from ...evaluation import visible_context

PROCOT_POLICY_VERSION = "adapted-target-goal-one-shot-v3"
PROCOT_FINAL_MARKER = "Based on the predicted next topics, the response is"

PROCOT_POLICY = (
    "You are a helpful target-guided conversational recommender. Use only the supplied "
    "conversation, visible target, user profile, and permitted knowledge. The demonstration "
    "specifies a planning strategy, not facts about this user. Do not copy its people, items, "
    "preferences, or factual claims. Be natural and honest, and respect explicit refusal. "
    "You must expose the requested current-topic and next-topic bridge before the final response."
)

PROCOT_INSTRUCTION = (
    "Given the conversation history and the target topics, consider the relationship between "
    "the current conversation topics and the target topics, and then predict the appropriate "
    "next topics that can bridge the current conversation topics to approach the target topics "
    "smoothly. Then based on the predicted next topics, generate an appropriate response. "
    'Please reply by completing the output template "The current topics are []. To bridge the '
    "current topics with the target topics, the next topics are []. Based on the predicted next "
    'topics, the response is "<one or two sentence response>"." Use this exact three-part '
    "template and do not omit any part."
)

# The upstream fs-pcot demonstration, with only its model-specific </s> separators omitted.
# It is unrelated to TopDial/DuRecDial and is fixed identically for both datasets.
PROCOT_ONE_SHOT_DEMONSTRATION = (
    'The target topics is "chicken". The conversation history is ["user": "i also remodel '
    'homes when i am not out bow hunting .", "system": "that\'s neat . when i was in high '
    'school i placed 6th in 100m dash !", "user": "that\'s awesome . do you have a favorite '
    'season or time of year ?"] Please reply: The current topics are ["season", "time", '
    '"year"]. To bridge the current topics with the target topics, the next topics are ["eat", '
    '"meat"]. Based on the predicted next topics, the response is "i do not . but i do have a '
    'favorite meat since that is all i eat exclusively".'
)


def procot_system_prompt(scenario):
    """Build a one-shot prompt using only the shared visible scenario fields."""
    return "\n".join(
        (
            PROCOT_POLICY,
            PROCOT_INSTRUCTION,
            PROCOT_ONE_SHOT_DEMONSTRATION,
            "Visible scenario context: " + visible_context(scenario),
            "The actual conversation history follows as chat messages. Produce the templated "
            "analysis and response for this actual conversation only. Your entire output must "
            "follow the required three-part template.",
        )
    )
