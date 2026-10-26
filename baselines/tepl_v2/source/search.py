import math
from dataclasses import asdict, dataclass, field

from ...state import DialogueState

EPL_SEARCH_REVISION = "source-puct-mean-return-v2"


def puct_score(q_value, prior, parent_visits, edge_visits, cpuct):
    if any(not math.isfinite(value) for value in (q_value, prior, cpuct)):
        raise ValueError("non-finite EPL P-UCT score")
    if prior < 0 or cpuct < 0 or min(parent_visits, edge_visits) < 0:
        raise ValueError("invalid EPL P-UCT inputs")
    # The released EPL tree starts every chance node with sampled_returns=[0],
    # initializes the parent visit count at one, and uses sqrt(log(N)).
    return q_value + cpuct * prior * math.sqrt(math.log(max(1, parent_visits))) / (
        2 + edge_visits
    )


@dataclass
class SearchNode:
    path: tuple[str, ...]
    priors: list[float]
    visits: list[int]
    q_values: list[float]
    return_sums: list[float]
    realizations: dict[int, list] = field(default_factory=dict)


@dataclass(frozen=True)
class EPLSearchResult:
    action_index: int
    action_name: str
    response: str
    root_visits: tuple[int, ...]
    root_q_values: tuple[float, ...]
    root_priors: tuple[float, ...]
    simulations: int
    expanded_nodes: int
    max_depth_reached: int
    request_ids: tuple[str, ...]
    tree: dict


class ExperiencePUCT:
    """Open-loop P-UCT with an experience-memory value on every realized edge."""

    def __init__(self, actions, policy, rollout, config, max_depth):
        self.actions = tuple(actions)
        self.policy = policy
        self.rollout = rollout
        self.config = config
        self.max_depth = max_depth
        self.nodes = {}
        self.request_ids = []
        self.max_depth_reached = 0

    def _expand(self, path, state):
        probabilities = [float(value) for value in self.policy(state)]
        if len(probabilities) != len(self.actions) or any(
            not math.isfinite(value) or value < 0 for value in probabilities
        ):
            raise ValueError("invalid EPL policy prior")
        total = sum(probabilities)
        if total <= 0:
            raise ValueError("zero EPL policy prior")
        self.nodes[path] = SearchNode(
            path,
            [value / total for value in probabilities],
            [0] * len(self.actions),
            [self.config.puct_q0] * len(self.actions),
            [self.config.puct_q0] * len(self.actions),
        )

    def _select(self, node):
        # EPL initializes DecisionNode.visits to one.
        parent_visits = 1 + sum(node.visits)
        scores = [
            puct_score(
                node.q_values[index],
                node.priors[index],
                parent_visits,
                node.visits[index],
                self.config.puct_cpuct,
            )
            for index in range(len(self.actions))
        ]
        return max(range(len(scores)), key=lambda index: (scores[index], -index))

    def _realize(self, path, node, action_index, state, simulation, depth):
        samples = node.realizations.setdefault(action_index, [])
        if len(samples) < self.config.puct_max_realizations:
            sample = self.rollout.expand(
                state,
                action_index,
                self.actions[action_index],
                path + (self.actions[action_index],),
                simulation,
                depth,
            )
            samples.append(sample)
            self.request_ids.extend(sample.request_ids)
        return samples[simulation % len(samples)]

    def _simulate(self, root_state, simulation):
        state, path, traversed = root_state, (), []
        for depth in range(self.max_depth):
            self.max_depth_reached = max(self.max_depth_reached, depth + 1)
            if path not in self.nodes:
                self._expand(path, state)
            node = self.nodes[path]
            action_index = self._select(node)
            sample = self._realize(path, node, action_index, state, simulation, depth)
            traversed.append((node, action_index, sample.reward))
            path = path + (self.actions[action_index],)
            if sample.terminal or depth + 1 >= self.max_depth:
                break
            state = DialogueState(tuple(sample.next_state))
            if path not in self.nodes:
                self._expand(path, state)
                break
        value = 0.0
        for node, action_index, reward in reversed(traversed):
            value = reward + self.config.gamma * value
            visits = node.visits[action_index] + 1
            node.return_sums[action_index] += value
            # Match chance_node.sampled_returns=[0, *observed_returns].
            node.q_values[action_index] = node.return_sums[action_index] / (visits + 1)
            node.visits[action_index] = visits

    def search(self, root_state):
        self._expand((), root_state)
        for simulation in range(self.config.puct_simulations):
            self._simulate(root_state, simulation)
        root = self.nodes[()]
        if sum(root.visits) != self.config.puct_simulations:
            raise RuntimeError("EPL root visit accounting mismatch")
        # The released EPL implementation returns the root action with the
        # largest sampled return estimate. Selecting by visits made five
        # rollouts over five candidates tie and collapse to action index zero.
        action_index = max(
            range(len(self.actions)), key=lambda index: (root.q_values[index], -index)
        )
        samples = root.realizations.get(action_index, [])
        selected = (
            max(enumerate(samples), key=lambda item: (item[1].reward, -item[0]))[1]
            if samples
            else None
        )
        tree = {
            "/".join(path) if path else "<root>": {
                "priors": node.priors,
                "visits": node.visits,
                "q_values": node.q_values,
                "realizations": {
                    self.actions[index]: [asdict(sample) for sample in values]
                    for index, values in node.realizations.items()
                },
            }
            for path, node in self.nodes.items()
        }
        return EPLSearchResult(
            action_index,
            self.actions[action_index],
            selected.assistant_response if selected is not None else "",
            tuple(root.visits),
            tuple(root.q_values),
            tuple(root.priors),
            self.config.puct_simulations,
            len(self.nodes),
            self.max_depth_reached,
            tuple(self.request_ids),
            tree,
        )
