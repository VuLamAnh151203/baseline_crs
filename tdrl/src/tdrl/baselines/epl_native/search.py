import math

from ..epl.search import ExperiencePUCT, SearchNode

NATIVE_SEARCH_REVISION = "source-puct-mean-return-native-path-v1"


class NativeExperiencePUCT(ExperiencePUCT):
    """Released P-UCT accounting with the native goal-topic path in policy state."""

    def _expand(self, path, state):
        probabilities = [float(value) for value in self.policy(state, path)]
        if len(probabilities) != len(self.actions) or any(
            not math.isfinite(value) or value < 0 for value in probabilities
        ):
            raise ValueError("invalid EPL native policy prior")
        total = sum(probabilities)
        if total <= 0:
            raise ValueError("zero EPL native policy prior")
        self.nodes[path] = SearchNode(
            path,
            [value / total for value in probabilities],
            [0] * len(self.actions),
            [self.config.puct_q0] * len(self.actions),
            [self.config.puct_q0] * len(self.actions),
        )
