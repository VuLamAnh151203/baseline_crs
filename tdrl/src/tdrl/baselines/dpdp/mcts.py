import math
from dataclasses import asdict

from .contracts import MCTSResult, RewardSample, TreeNodeStats


def puct_score(q_value, prior, parent_visits, edge_visits, cpuct):
    values = (q_value, prior, cpuct)
    if any(not math.isfinite(value) for value in values) or min(prior, cpuct) < 0:
        raise ValueError("invalid PUCT inputs")
    if parent_visits < 0 or edge_visits < 0:
        raise ValueError("invalid PUCT visit count")
    return q_value + cpuct * prior * math.sqrt(max(1, parent_visits)) / (1 + edge_visits)


class OpenLoopMCTS:
    """Open-loop PUCT tree keyed only by action sequences from the current root."""

    def __init__(
        self,
        actions,
        policy,
        rollout,
        simulations,
        cpuct,
        q0,
        max_depth,
        max_realizations,
    ):
        if simulations < 2 or max_depth < 1 or max_realizations < 1:
            raise ValueError("invalid DPDP MCTS budget")
        if len(actions) < 2 or len(set(actions)) != len(actions):
            raise ValueError("invalid DPDP action taxonomy")
        self.actions = tuple(actions)
        self.policy = policy
        self.rollout = rollout
        self.simulations = simulations
        self.cpuct = cpuct
        self.q0 = q0
        self.max_depth = max_depth
        self.max_realizations = max_realizations
        self.nodes = {}
        self.request_ids = []
        self.max_depth_reached = 0

    def _expand_node(self, path, state):
        probabilities, q_values = self.policy(state)
        probabilities = [float(value) for value in probabilities]
        q_values = [float(value) for value in q_values]
        if len(probabilities) != len(self.actions) or len(q_values) != len(self.actions):
            raise ValueError("DPDP policy/MCTS action dimension mismatch")
        if any(not math.isfinite(value) or value < 0 for value in probabilities):
            raise ValueError("invalid DPDP policy prior")
        total = sum(probabilities)
        if total <= 0:
            raise ValueError("zero DPDP policy prior")
        priors = [value / total for value in probabilities]
        self.nodes[path] = TreeNodeStats(
            path=path,
            priors=priors,
            visits=[0] * len(self.actions),
            q_values=[self.q0] * len(self.actions),
        )

    def _select(self, node):
        parent_visits = sum(node.visits)
        scores = [
            puct_score(
                node.q_values[index],
                node.priors[index],
                parent_visits,
                node.visits[index],
                self.cpuct,
            )
            for index in range(len(self.actions))
        ]
        # Stable action order is the deterministic final tie-break.
        return max(range(len(scores)), key=lambda index: (scores[index], -index))

    def _realize(self, path, node, action_index, state, simulation, depth):
        samples = node.realizations.setdefault(action_index, [])
        if len(samples) < self.max_realizations:
            sample = self.rollout.expand(
                state=state,
                action_index=action_index,
                action_name=self.actions[action_index],
                path=path + (self.actions[action_index],),
                simulation=simulation,
                depth=depth,
            )
            if not isinstance(sample, RewardSample):
                raise TypeError("DPDP rollout must return RewardSample")
            samples.append(sample)
            self.request_ids.extend(sample.request_ids)
        return samples[simulation % len(samples)]

    def _simulate(self, root_state, simulation):
        path = ()
        state = root_state
        traversed = []
        value = 0.0
        for depth in range(self.max_depth):
            self.max_depth_reached = max(self.max_depth_reached, depth + 1)
            if path not in self.nodes:
                self._expand_node(path, state)
            node = self.nodes[path]
            action_index = self._select(node)
            sample = self._realize(path, node, action_index, state, simulation, depth)
            traversed.append((node, action_index))
            value = sample.reward
            path = path + (self.actions[action_index],)
            state = sample.next_state
            if sample.terminal or sample.refused or depth + 1 >= self.max_depth:
                break
            if path not in self.nodes:
                self._expand_node(path, state)
                break
        for node, action_index in reversed(traversed):
            visits = node.visits[action_index] + 1
            node.q_values[action_index] += (value - node.q_values[action_index]) / visits
            node.visits[action_index] = visits

    def search(self, root_state):
        self._expand_node((), root_state)
        for simulation in range(self.simulations):
            self._simulate(root_state, simulation)
        root = self.nodes[()]
        if sum(root.visits) != self.simulations:
            raise RuntimeError("DPDP MCTS root was not visited once per simulation")
        action_index = max(range(len(self.actions)), key=lambda index: (root.visits[index], -index))
        samples = root.realizations.get(action_index, [])
        if not samples:
            raise RuntimeError("DPDP MCTS selected an unrealized root action")
        best = max(enumerate(samples), key=lambda item: (item[1].reward, -item[0]))[1]
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
        return MCTSResult(
            action_index=action_index,
            action_name=self.actions[action_index],
            response=best.assistant_response,
            user_response=best.user_response,
            reward=best.reward,
            reward_label=best.label,
            critic_labels=best.critic_labels,
            deterministic_accepted=best.deterministic_accepted,
            acceptance_disagreement=best.acceptance_disagreement,
            terminal=best.terminal,
            terminal_reasons=best.terminal_reasons,
            refused=best.refused,
            next_state=best.next_state,
            root_visits=tuple(root.visits),
            root_q_values=tuple(root.q_values),
            root_priors=tuple(root.priors),
            simulations=self.simulations,
            expanded_nodes=len(self.nodes),
            max_depth_reached=self.max_depth_reached,
            request_ids=tuple(self.request_ids),
            tree=tree,
        )
