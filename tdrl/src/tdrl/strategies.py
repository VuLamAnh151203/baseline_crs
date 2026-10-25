"""Shared strategy instructions; independent of planner selection mechanism."""

ACTION_INSTRUCTIONS = {
    "acknowledge": "Acknowledge the user's stated needs without inventing preferences.",
    "ask_preference": "Ask one relevant question about the user's preferences.",
    "clarify": "Clarify missing or ambiguous information needed for the current goal.",
    "explain": "Explain relevant facts or reasons using only permitted knowledge.",
    "recommend": "Suggest the target if appropriate, explain its fit, and invite a free choice.",
    "farewell": "Close politely and stop persuasion; respect any expressed refusal.",
}
