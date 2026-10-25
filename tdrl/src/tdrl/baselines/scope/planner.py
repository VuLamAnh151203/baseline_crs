"""Source-corrected semantic-space MCTS for SCOPE-CRS-lite."""

import json
import math
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
from torch import nn

from ...contracts import Decoding, Message, Plan
from ...planner import derived_seed
from ...provenance import atomic_json, digest
from ..direct_llm.planner import direct_v2_system_prompt
from .semantic import target_messages

POLICY_VERSION = "scope-crs-lite-llamaguard1024-mcts-v1"


class OnlineSemanticQ:
    """Upstream SCOPE online Q approximator with per-search state."""

    def __init__(self, dimension, learning_rate, update_steps, device, seed):
        self.dimension = dimension
        self.learning_rate = learning_rate
        self.update_steps = update_steps
        self.device = torch.device(device)
        with torch.random.fork_rng(devices=[self.device.index or 0]):
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            self.network = nn.Sequential(
                nn.Linear(dimension * 2, dimension),
                nn.ReLU(),
                nn.Linear(dimension, dimension // 4),
                nn.ReLU(),
                nn.Linear(dimension // 4, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
            ).to(self.device)
        self.replay_inputs = None
        self.replay_rewards = None

    def merged(self, state, action):
        return torch.cat((state, action), dim=-1).reshape(1, -1).to(self.device)

    @torch.inference_mode()
    def value(self, state, action):
        return float(self.network(self.merged(state, action))[0, 0].cpu())

    def update(self, state, action, reward, visits):
        item = self.merged(state, action)
        target = torch.tensor([reward], dtype=torch.float32, device=self.device)
        optimizer = torch.optim.Adam(
            self.network.parameters(), lr=self.learning_rate * (1 / visits) ** 2
        )
        for _ in range(self.update_steps):
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.mse_loss(self.network(item).squeeze(1), target)
            loss.backward()
            optimizer.step()
        self.replay_inputs = (
            item.detach()
            if self.replay_inputs is None
            else torch.cat((self.replay_inputs, item.detach()), dim=0)
        )
        self.replay_rewards = (
            target.detach()
            if self.replay_rewards is None
            else torch.cat((self.replay_rewards, target.detach()), dim=0)
        )
        for _ in range(self.update_steps):
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.mse_loss(
                self.network(self.replay_inputs).squeeze(1), self.replay_rewards
            )
            loss.backward()
            optimizer.step()


@dataclass
class SearchNode:
    state: torch.Tensor
    depth: int
    parent: "SearchNode | None" = None
    parent_action: int | None = None
    reward: float = 0.0
    actions: list[torch.Tensor] | None = None
    children: dict[int, list["SearchNode"]] = field(default_factory=dict)
    action_visits: dict[int, int] = field(default_factory=dict)
    visits: int = 0


class SemanticMCTS:
    def __init__(self, transitions, reward, goal, config, device, seed):
        self.transitions, self.reward, self.goal = transitions, reward, goal
        self.config, self.device = config, torch.device(device)
        self.seed = seed
        self.rng = random.Random(seed)
        self.q = OnlineSemanticQ(
            config.embedding_dim,
            config.q_learning_rate,
            config.q_update_steps,
            self.device,
            seed,
        )
        self.simulations = 0

    @torch.inference_mode()
    def potential(self, state):
        state = state.to(self.device, dtype=torch.float32)
        goal = self.goal.to(self.device, dtype=torch.float32)
        return float(self.reward(state, goal).cpu())

    def actions(self, node):
        if node.actions is None:
            assistant = self.transitions.assistant_states(node.state[None].to(self.device))
            node.actions = [row[0].cpu() - node.state.cpu() for row in assistant]
        return node.actions

    def outcome(self, node, action_index):
        action = self.actions(node)[action_index]
        assistant = node.state.to(self.device) + action.to(self.device)
        outcomes = self.transitions.user_states(assistant[None])[:, 0, :].cpu()
        selected = self.rng.randrange(len(outcomes))
        next_state = outcomes[selected]
        reward = self.potential(next_state) - self.potential(node.state)
        child = SearchNode(next_state, node.depth + 1, node, action_index, reward)
        node.children.setdefault(action_index, []).append(child)
        return child

    def select(self, root):
        node = root
        while node.depth < self.config.search_depth // 2:
            actions = self.actions(node)
            unvisited = [i for i in range(len(actions)) if i not in node.action_visits]
            if unvisited:
                return node, unvisited[0]
            log_total = math.log(max(1, sum(node.action_visits.values())))
            scores = []
            for index, action in enumerate(actions):
                count = node.action_visits[index]
                score = self.config.uct_coefficient * self.q.value(node.state, action)
                score += math.sqrt(2 * log_total / count)
                scores.append(score)
            action_index = max(range(len(scores)), key=lambda index: (scores[index], -index))
            outcomes = node.children[action_index]
            node = outcomes[self.rng.randrange(len(outcomes))]
        return node, None

    def rollout(self, node):
        state, depth = node.state, node.depth
        cumulative, discount = 0.0, 1.0
        while depth < self.config.search_depth // 2:
            assistant_states = self.transitions.assistant_states(state[None].to(self.device))
            action_index = self.rng.randrange(len(assistant_states))
            assistant = assistant_states[action_index, 0]
            user_states = self.transitions.user_states(assistant[None])[:, 0, :]
            next_state = user_states[self.rng.randrange(len(user_states))].cpu()
            reward = self.potential(next_state) - self.potential(state)
            cumulative += discount * reward
            discount *= self.config.gamma
            state, depth = next_state, depth + 1
        return cumulative

    def backpropagate(self, node, rollout_reward):
        value = rollout_reward
        while node.parent is not None:
            parent, action_index = node.parent, node.parent_action
            value = node.reward + self.config.gamma * value
            parent.visits += 1
            parent.action_visits[action_index] = parent.action_visits.get(action_index, 0) + 1
            self.q.update(
                parent.state,
                self.actions(parent)[action_index],
                value,
                parent.action_visits[action_index],
            )
            node = parent

    def run(self, initial_state, candidate_actions):
        devices = [self.device.index or torch.cuda.current_device()]
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
            root = SearchNode(
                initial_state.cpu(),
                0,
                actions=[row.cpu() for row in candidate_actions],
            )
            started = time.monotonic()
            # Each released candidate receives one expansion before timed UCT comparisons.
            for index in range(len(candidate_actions)):
                child = self.outcome(root, index)
                self.backpropagate(child, self.rollout(child))
            self.simulations += 1
            while time.monotonic() - started < self.config.mcts_seconds:
                node, action_index = self.select(root)
                if action_index is None:
                    self.backpropagate(node, 0.0)
                else:
                    child = self.outcome(node, action_index)
                    self.backpropagate(child, self.rollout(child))
                self.simulations += 1
            values = [self.q.value(root.state, action) for action in candidate_actions]
        return values, {
            "simulations": self.simulations,
            "root_action_visits": [root.action_visits.get(i, 0) for i in range(len(values))],
            "search_wall_s": time.monotonic() - started,
        }


class ScopeLitePlanner:
    def __init__(
        self,
        generation,
        semantic_encoder,
        transitions,
        reward,
        scenario,
        dataset,
        config,
        decoding=None,
        plan_directory=None,
    ):
        self.generation = generation
        self.encoder = semantic_encoder
        self.transitions = transitions
        self.reward = reward
        self.scenario, self.dataset, self.config = scenario, dataset, config
        self.decoding = decoding or Decoding(max_tokens=128)
        self.system_prompt = direct_v2_system_prompt(scenario, dataset)
        self.plan_directory = Path(plan_directory) if plan_directory else None
        self.goal = self.encoder.embed(
            target_messages(scenario), f"scope/{dataset}/{scenario['id']}/goal"
        )
        self.seed_context = scenario["id"]

    def act(self, state, target_nodes, seed, request_id):
        del target_nodes
        started = time.monotonic()
        generation_start, semantic_start = len(self.generation.requests), len(self.encoder.requests)
        requests = [
            (
                [Message("system", self.system_prompt), *state.history],
                "assistant",
                self.decoding,
                derived_seed(seed, self.seed_context, "scope-candidate", index),
                f"{request_id}/candidate/{index}",
            )
            for index in range(self.config.candidates)
        ]
        generated = self.generation.generate_many(requests)
        candidates = [row.text.strip() for row in generated]
        if any(not candidate for candidate in candidates):
            raise ValueError("SCOPE candidate generation returned empty text")
        plan_request = {
            "request_id": request_id,
            "history": [asdict(message) for message in state.history],
            "candidate_sha256": [digest(candidate) for candidate in candidates],
            "config_sha256": self.config.sha256,
            "policy_version": POLICY_VERSION,
        }
        plan_path = (
            self.plan_directory / (digest(request_id) + ".json")
            if self.plan_directory is not None
            else None
        )
        if plan_path is not None and plan_path.exists():
            cached = json.loads(plan_path.read_text(encoding="utf-8"))
            if cached.get("request_sha256") != digest(plan_request):
                raise ValueError("SCOPE plan replay contract mismatch")
            return Plan(**cached["plan"])
        current = self.encoder.embed(state.history, request_id + "/semantic/current")
        candidate_states = torch.stack(
            [
                self.encoder.embed(
                    (*state.history, Message("assistant", candidate)),
                    f"{request_id}/semantic/candidate/{index}",
                )
                for index, candidate in enumerate(candidates)
            ]
        )
        actions = candidate_states - current
        search_seed = derived_seed(seed, self.seed_context, "scope-mcts")
        search = SemanticMCTS(
            self.transitions,
            self.reward,
            self.goal,
            self.config,
            "cuda",
            search_seed,
        )
        values, trace = search.run(current, actions)
        selected = max(range(len(values)), key=lambda index: (values[index], -index))
        semantic_records = self.encoder.requests[semantic_start:]
        plan = Plan(
            response=candidates[selected],
            covered=False,
            reference_id=None,
            directed_distances={},
            path=[],
            selected_candidate=selected,
            rollout_scores=values,
            compute={
                "strategy": "scope-semantic-mcts-task-reward",
                "policy_version": POLICY_VERSION,
                "candidate_count": len(candidates),
                "candidate_sha256": [digest(item) for item in candidates],
                "generation_requests": len(self.generation.requests) - generation_start,
                "embedding_requests": 0,
                "intrinsic_semantic_requests": len(semantic_records),
                "intrinsic_semantic_tokens": sum(row.input_tokens for row in semantic_records),
                "search_simulations": trace["simulations"],
                "root_action_visits": trace["root_action_visits"],
                "search_wall_s": trace["search_wall_s"],
                "search_depth": self.config.search_depth,
                "mcts_seconds": self.config.mcts_seconds,
                "recovery_triggered": False,
                "planner_latency_s": time.monotonic() - started,
            },
            replay_ids=[row.request_id for row in generated],
        )
        if plan_path is not None:
            atomic_json(
                plan_path,
                {"request_sha256": digest(plan_request), "plan": asdict(plan)},
            )
        return plan
