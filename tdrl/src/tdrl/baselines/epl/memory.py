import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import torch

from ...acceptance import classify_user
from ...contracts import Embeddings, Message, ModelProvenance
from ...provenance import Journal, atomic_json, digest, file_hash
from ...state import DialogueState
from .assets import verify_assets
from .contracts import Experience
from .dataset import read_jsonl
from .serialization import policy_text

# SentenceTransformer.encode in the released EPL code embeds raw state text.
MEMORY_INSTRUCTION = ""


class NativeMiniLMEmbedding:
    """Original EPL MiniLM mean-pooling retriever with durable exact replay."""

    def __init__(
        self,
        policy_path,
        retrieval_path,
        model_lock,
        directory,
        device="cuda",
    ):
        from transformers import AutoModel, AutoTokenizer

        assets = verify_assets(model_lock, policy_path, retrieval_path)
        record = assets["retrieval_encoder"]
        self.model = ModelProvenance(
            model_id=record["model_id"],
            revision=record["revision"],
            weight_sha256=record["files"],
            template_sha256=digest("sentence-transformers mean pooling; no normalization"),
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            retrieval_path, local_files_only=True
        )
        self.encoder = AutoModel.from_pretrained(
            retrieval_path, local_files_only=True
        ).to(device)
        if getattr(self.encoder.config, "hidden_size", None) != 384:
            raise ValueError("EPL MiniLM retriever must have hidden size 384")
        self.encoder.eval()
        self.device = torch.device(device)
        self.requests = []
        self.preprocessing_sha256 = digest(
            {
                "pooling": "attention-mask mean",
                "normalize_embeddings": False,
                "max_length": 256,
                "instruction": "ignored; original EPL embeds raw state text",
            }
        )
        self.journal = Journal(Path(directory), asdict(self.model))

    def embed(self, texts, instruction, batch_id, request_id):
        if not texts or any(not isinstance(text, str) for text in texts):
            raise ValueError("EPL MiniLM embedding requires non-empty strings")
        request = {
            "texts": texts,
            "instruction": instruction,
            "batch_id": batch_id,
            "model": asdict(self.model),
            "preprocessing_sha256": self.preprocessing_sha256,
        }
        cached = self.journal.read(request_id, request)
        if cached is not None:
            return Embeddings(
                vectors=torch.tensor(cached["vectors"], dtype=torch.float32),
                model=ModelProvenance(**cached["model"]),
                instruction_sha256=cached["instruction_sha256"],
                preprocessing_sha256=cached["preprocessing_sha256"],
                batch_size=cached["batch_size"],
                latency_s=cached["latency_s"],
                input_tokens=cached["input_tokens"],
                peak_memory_bytes=cached["peak_memory_bytes"],
                batch_id=cached["batch_id"],
                request_id=cached["request_id"],
            )
        started = time.monotonic()
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        with torch.inference_mode():
            hidden = self.encoder(**encoded, return_dict=True).last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            vectors = ((hidden * mask).sum(1) / mask.sum(1).clamp_min(1)).float().cpu()
        result = Embeddings(
            vectors=vectors,
            model=self.model,
            instruction_sha256=digest(""),
            preprocessing_sha256=self.preprocessing_sha256,
            batch_size=len(texts),
            latency_s=time.monotonic() - started,
            input_tokens=int(encoded["attention_mask"].sum()),
            peak_memory_bytes=(
                torch.cuda.max_memory_allocated(self.device)
                if self.device.type == "cuda"
                else None
            ),
            batch_id=batch_id,
            request_id=request_id,
        )
        payload = asdict(result)
        payload["vectors"] = vectors.tolist()
        self.journal.write(request_id, request, payload)
        self.requests.append(result)
        return result


def build_experiences(input_path, output_path, dataset, config):
    rows = read_jsonl(input_path)
    selected = sorted(
        rows,
        key=lambda row: digest([dataset, row["id"], config.self_simulation_seed]),
    )[
        : config.memory_cases
    ]
    experiences = []
    for row in selected:
        messages = tuple(Message(**message) for message in row["messages"])
        label = classify_user(DialogueState(messages), row["target"])
        score = 1.0 if label.accepted else 0.0
        scenario = {"profile": row["profile"], "target": row["target"]}
        for turn, message in enumerate(messages):
            if message.role != "assistant" or not messages[:turn]:
                continue
            experiences.append(
                Experience(
                    experience_id=f"{row['id']}:{turn}",
                    scenario_id=row["id"],
                    state_text=policy_text(scenario, messages[:turn]),
                    continuation=tuple(asdict(item) for item in messages[turn:]),
                    score=score,
                    target=tuple(row["target"]),
                )
            )
    if not experiences or len({row.experience_id for row in experiences}) != len(experiences):
        raise ValueError("empty or duplicate EPL experience memory")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    raw = "".join(
        json.dumps(row.to_dict(), ensure_ascii=False, sort_keys=True) + "\n"
        for row in experiences
    )
    if output_path.exists() and output_path.read_text(encoding="utf-8") != raw:
        raise ValueError("immutable EPL experience memory changed")
    output_path.write_text(raw, encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "dataset": dataset,
        "source_split": "train",
        "source_sha256": file_hash(Path(input_path)),
        "selected_scenario_ids": [row["id"] for row in selected],
        "selected_scenario_ids_sha256": digest([row["id"] for row in selected]),
        "experience_count": len(experiences),
        "memory_sha256": file_hash(output_path),
        "score": "deterministic observed-training acceptance; no development/test outcomes",
        "test_bodies": "never_opened",
    }
    atomic_json(output_path.with_suffix(".manifest.json"), manifest)
    return manifest


class JournaledEmbedding:
    """Durable exact replay wrapper for EPL's intrinsic embedding requests."""

    def __init__(self, backend, directory):
        self.backend = backend
        self.journal = Journal(Path(directory), asdict(backend.model))

    @property
    def requests(self):
        return self.backend.requests

    @property
    def model(self):
        return self.backend.model

    def embed(self, texts, instruction, batch_id, request_id):
        request = {
            "texts": texts,
            "instruction": instruction,
            "batch_id": batch_id,
            "model": asdict(self.backend.model),
        }
        cached = self.journal.read(request_id, request)
        if cached is not None:
            return Embeddings(
                vectors=torch.tensor(cached["vectors"], dtype=torch.float32),
                model=ModelProvenance(**cached["model"]),
                instruction_sha256=cached["instruction_sha256"],
                preprocessing_sha256=cached["preprocessing_sha256"],
                batch_size=cached["batch_size"],
                latency_s=cached["latency_s"],
                input_tokens=cached["input_tokens"],
                peak_memory_bytes=cached["peak_memory_bytes"],
                batch_id=cached["batch_id"],
                request_id=cached["request_id"],
            )
        output = self.backend.embed(texts, instruction, batch_id, request_id)
        payload = asdict(output)
        payload["vectors"] = output.vectors.tolist()
        self.journal.write(request_id, request, payload)
        return output


def embed_experiences(memory_path, vector_path, embedding, config, request_namespace):
    memory_path, vector_path = Path(memory_path), Path(vector_path)
    experiences = [
        json.loads(line)
        for line in memory_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    memory_sha = file_hash(memory_path)
    if vector_path.exists():
        payload = torch.load(vector_path, map_location="cpu", weights_only=False)
        if (
            payload.get("schema_version") != 1
            or payload.get("memory_sha256") != memory_sha
            or payload.get("experience_ids") != [row["experience_id"] for row in experiences]
        ):
            raise ValueError("EPL memory vector cache provenance changed")
        return payload
    vectors = []
    batch_size = config.memory_embedding_batch_size
    for start in range(0, len(experiences), batch_size):
        rows = experiences[start : start + batch_size]
        result = embedding.embed(
            [row["state_text"] for row in rows],
            MEMORY_INSTRUCTION,
            f"epl-memory/{config.dataset}/{start}",
            f"{request_namespace}/memory/{start}",
        )
        vectors.append(result.vectors.cpu())
    payload = {
        "schema_version": 1,
        "memory_sha256": memory_sha,
        "experience_ids": [row["experience_id"] for row in experiences],
        "vectors": torch.cat(vectors),
        "model": asdict(embedding.model),
        "instruction_sha256": digest(""),
    }
    vector_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = vector_path.with_suffix(vector_path.suffix + ".pending")
    torch.save(payload, temporary)
    temporary.replace(vector_path)
    atomic_json(
        vector_path.with_suffix(".manifest.json"),
        {
            "schema_version": 1,
            "status": "complete",
            "memory_sha256": memory_sha,
            "vector_sha256": file_hash(vector_path),
            "experience_count": len(experiences),
            "embedding_model": payload["model"],
            "instruction_sha256": payload["instruction_sha256"],
        },
    )
    return payload


class ExperienceMemory:
    def __init__(self, memory_path, vector_path, embedding, config):
        self.rows = [
            json.loads(line)
            for line in Path(memory_path).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        payload = torch.load(vector_path, map_location="cpu", weights_only=False)
        if payload.get("memory_sha256") != file_hash(Path(memory_path)):
            raise ValueError("EPL memory/vector mismatch")
        if payload.get("experience_ids") != [row["experience_id"] for row in self.rows]:
            raise ValueError("EPL memory vector order mismatch")
        self.vectors = payload["vectors"].float()
        if self.vectors.shape != (
            len(self.rows),
            config.retrieval_dimension,
        ) or not torch.isfinite(self.vectors).all():
            raise ValueError("invalid EPL memory vectors")
        self.embedding = embedding
        self.config = config

    def lookup(self, scenario, state, request_id):
        query = policy_text(scenario, state.history)
        result = self.embedding.embed(
            [query], MEMORY_INSTRUCTION, request_id + "/batch", request_id
        )
        similarities = self.vectors @ result.vectors[0].cpu()
        count = min(self.config.retrieval_k, len(self.rows))
        values, indices = torch.topk(similarities, count, sorted=True)
        weights = torch.softmax(values / self.config.retrieval_temperature, dim=0)
        target = scenario["target"][1].casefold().strip()
        rewards = []
        evidence = []
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
        value = float((weights * torch.tensor(rewards)).sum())
        return value, evidence, result.request_id
