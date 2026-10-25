import json
import math
from pathlib import Path

import torch

from ...provenance import file_hash
from .serialization import policy_text


class NativeExperienceMemory:
    """EPL MiniLM experience value with native goal-topic state serialization."""

    def __init__(self, memory_path, vector_path, embedding, config):
        self.rows = [
            json.loads(line)
            for line in Path(memory_path).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        payload = torch.load(vector_path, map_location="cpu", weights_only=False)
        if payload.get("memory_sha256") != file_hash(Path(memory_path)):
            raise ValueError("EPL native memory/vector mismatch")
        if payload.get("experience_ids") != [row["experience_id"] for row in self.rows]:
            raise ValueError("EPL native memory vector order mismatch")
        self.vectors = payload["vectors"].float()
        if self.vectors.shape != (len(self.rows), config.retrieval_dimension):
            raise ValueError("invalid EPL native memory vectors")
        if not torch.isfinite(self.vectors).all():
            raise ValueError("non-finite EPL native memory vectors")
        self.embedding, self.config = embedding, config

    def lookup(self, scenario, state, request_id):
        query = policy_text(scenario, state.history)
        result = self.embedding.embed(
            [query], "", request_id + "/batch", request_id
        )
        similarities = self.vectors @ result.vectors[0].cpu()
        count = min(self.config.retrieval_k, len(self.rows))
        values, indices = torch.topk(similarities, count, sorted=True)
        weights = torch.softmax(values / self.config.retrieval_temperature, dim=0)
        target = str(scenario["target"][1]).casefold().strip()
        rewards, evidence = [], []
        for similarity, index in zip(values.tolist(), indices.tolist(), strict=True):
            row = self.rows[index]
            mentioned = any(
                item["role"] == "assistant" and target in item["content"].casefold()
                for item in row["continuation"]
            )
            score = float(row["score"])
            positive = mentioned and score >= self.config.retrieval_success_threshold
            outcome = 1.0 if positive else -1.0
            confidence = score if positive else 1.0 - score
            reward = (
                outcome * self.config.reward_alpha * confidence
                + self.config.reward_length_weight
                * math.exp(
                    -len(row["continuation"]) / self.config.reward_length_temperature
                )
            )
            rewards.append(reward)
            evidence.append(
                {
                    "experience_id": row["experience_id"],
                    "similarity": similarity,
                    "score": score,
                    "target_mentioned": mentioned,
                    "reward": reward,
                }
            )
        return float((weights * torch.tensor(rewards)).sum()), evidence, result.request_id
