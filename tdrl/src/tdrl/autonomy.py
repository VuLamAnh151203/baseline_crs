"""Frozen conservative refusal rule, distinct from acceptance/success labeling.

Coverage is limited to explicit forms; annotation must audit other refusal forms.
This rule does not claim to recognize all natural-language refusals.
"""

import re

REFUSAL_VERSION = 2
REFUSAL_FORMS = (
    r"\bno thanks\b",
    r"\bnot (?:really |particularly |at all )?interested\b",
    r"\bi (?:do not|don't|don’t) need (?:any )?(?:other |more )?(?:recommendations?|suggestions?)\b",
    r"\bstop (?:recommending|suggesting|pushing)\b",
    r"\bi (?:do not|don't) want (?:that|this|it|(?:any )?(?:more )?(?:recommendations?|suggestions?))\b",
    "不感兴趣",
    "不要再推荐",
    "不用了",
    "不想要",
)
REFUSAL_RESPONSE = (
    "Understood. We can leave that recommendation aside. What would you like to discuss instead?"
)


def explicit_refusal(state):
    # Sticky for the rest of this dialogue; a new topic does not revoke a prior refusal.
    return any(
        re.search(pattern, message.content.casefold())
        for message in state.history
        if message.role == "user"
        for pattern in REFUSAL_FORMS
    )
