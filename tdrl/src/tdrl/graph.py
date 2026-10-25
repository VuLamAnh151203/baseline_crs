import heapq
import math
from dataclasses import dataclass

import torch

from .state import DialogueState


@dataclass(frozen=True)
class Reference:
    node_id: str
    state: DialogueState
    embedding: torch.Tensor
    guidance: str  # Observed next assistant move, derived only from training.
    dialogue_id: str
    split: str = "train"
    action: str | None = None


class DirectedGraph:
    def __init__(self, references: list[Reference], head, h_edge: float, provenance: dict):
        if not references or h_edge < 0 or not math.isfinite(h_edge):
            raise ValueError("invalid graph")
        if any(r.split != "train" for r in references):
            raise ValueError("graphs accept training resources only")
        if len({r.node_id for r in references}) != len(references):
            raise ValueError("duplicate node IDs")
        self.references = {r.node_id: r for r in references}
        self.head, self.h_edge, self.provenance = head, h_edge, provenance
        self.edges = {r.node_id: {} for r in references}
        self._head_signature = self.signature()
        self._goal_paths = {}
        self._encoded = None
        self._node_indices = {node: i for i, node in enumerate(self.references)}

    def signature(self):
        # Content hash catches in-place training/weight changes after graph construction.
        import hashlib

        h = hashlib.sha256()
        for name, value in self.head.state_dict().items():
            h.update(name.encode())
            h.update(
                value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
            )
        return h.hexdigest()

    def ensure_frozen(self):
        if self.signature() != self._head_signature:
            raise ValueError("graph/retrieval head changed")

    def add_edge(self, source: str, target: str, weight: float):
        if source not in self.edges or target not in self.edges:
            raise ValueError("unknown edge endpoint")
        if not math.isfinite(weight) or weight < 0:
            raise ValueError("invalid directed edge cost")
        self.edges[source][target] = weight
        self._goal_paths.clear()

    def build(self, block_size=128):
        self.ensure_frozen()
        refs = list(self.references.values())
        parameter = next(self.head.parameters(), torch.empty(0))
        targets = torch.stack([r.embedding for r in refs]).to(
            device=parameter.device, dtype=parameter.dtype
        )
        with torch.no_grad():
            if hasattr(self.head, "encode"):
                self._encoded = self.head.encode(targets).detach()
            for start in range(0, len(refs), block_size):
                d = (
                    self.head.distance_encoded(
                        self._encoded[start : start + block_size, None], self._encoded[None, :]
                    )
                    if self._encoded is not None
                    else self.head.distance(
                        targets[start : start + block_size, None], targets[None, :]
                    )
                )
                if not torch.isfinite(d).all() or (d < 0).any():
                    raise ValueError("invalid graph scoring")
                for i, row in enumerate(d.cpu().tolist()):
                    for j, weight in enumerate(row):
                        if start + i != j and weight <= self.h_edge:
                            self.add_edge(refs[start + i].node_id, refs[j].node_id, weight)
        return self

    def shortest_path(self, source: str, goals: list[str]) -> tuple[float, list[str]]:
        if source not in self.edges or not set(goals).issubset(self.edges):
            raise ValueError("unknown graph endpoint")
        queue = [(0.0, source)]
        best, previous = {source: 0.0}, {}
        while queue:
            cost, node = heapq.heappop(queue)
            if cost != best[node]:
                continue
            if node in goals:
                path = [node]
                while path[-1] in previous:
                    path.append(previous[path[-1]])
                return cost, path[::-1]
            for target, weight in self.edges[node].items():
                new = cost + weight
                if new < best.get(target, math.inf):
                    best[target], previous[target] = new, node
                    heapq.heappush(queue, (new, target))
        return math.inf, []

    def retrieve(self, live: torch.Tensor, target_nodes: list[str]):
        self.ensure_frozen()
        paths = self.paths_to_goals(target_nodes)
        eligible = [ref for node, ref in self.references.items() if node in paths and ref.guidance]
        if not eligible:
            return []
        parameter = next(self.head.parameters(), torch.empty(0))
        with torch.no_grad():
            live = live.to(device=parameter.device, dtype=parameter.dtype)
            if self._encoded is not None:
                encoded = self._encoded[[self._node_indices[r.node_id] for r in eligible]]
                distances = self.head.distance_encoded(self.head.encode(live), encoded).float()
            else:
                targets = torch.stack([r.embedding for r in eligible]).to(
                    device=parameter.device, dtype=parameter.dtype
                )
                distances = self.head.distance(live, targets).float()
        if not torch.isfinite(distances).all() or (distances < 0).any():
            raise ValueError("invalid live-to-reference distance")
        candidates = [
            (d, ref.node_id, paths[ref.node_id])
            for ref, d in zip(eligible, distances.cpu().tolist(), strict=True)
        ]
        return sorted(candidates, key=lambda x: (x[0], x[1]))

    def remaining_cost(self, path):
        """Cost along recorded forward edges; reference goal costs zero."""
        cost = sum(self.edges[u][v] for u, v in zip(path, path[1:], strict=False))
        if not math.isfinite(cost) or cost < 0:
            raise ValueError("invalid remaining directed path cost")
        return cost

    def paths_to_goals(self, goals):
        """Reverse *traversal* of recorded edges; never infer a reverse edge/weight."""
        if not goals or not set(goals).issubset(self.edges):
            raise ValueError("unknown/empty graph goals")
        key = tuple(sorted(set(goals)))
        if key in self._goal_paths:
            return self._goal_paths[key]
        predecessors = {node: [] for node in self.edges}
        for source, outgoing in self.edges.items():
            for target, weight in outgoing.items():
                predecessors[target].append((source, weight))
        best = dict.fromkeys(goals, 0.0)
        queue, successor = [(0.0, goal) for goal in goals], {}
        heapq.heapify(queue)
        while queue:
            cost, node = heapq.heappop(queue)
            if cost != best[node]:
                continue
            for source, weight in predecessors[node]:
                new = cost + weight  # The recorded cost is d(source,node), not d(node,source).
                if new < best.get(source, math.inf):
                    best[source], successor[source] = new, node
                    heapq.heappush(queue, (new, source))
        paths = {}
        for source in best:
            path = [source]
            while path[-1] in successor:
                path.append(successor[path[-1]])
            paths[source] = path
        self._goal_paths[key] = paths
        return paths
