"""Exact Llama Guard 2 semantic encoder used by released SCOPE transitions."""

import json
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ...contracts import Message
from ...provenance import digest
from .assets import verify_llama_guard_snapshot


@dataclass(frozen=True)
class SemanticRequest:
    request_id: str
    batch_size: int
    input_tokens: int
    latency_s: float
    vectors_sha256: str


def _load_local_causal_lm(snapshot, device):
    """Load pinned BF16 weights locally, then place them without Accelerate."""
    model = AutoModelForCausalLM.from_pretrained(
        snapshot,
        local_files_only=True,
        dtype=torch.bfloat16,
    )
    return model.to(device).eval()


def alternating_chat(messages):
    """Match upstream Conversation formatting without its list-mutation bug."""
    rows = [asdict(row) if isinstance(row, Message) else dict(row) for row in messages]
    if not rows:
        raise ValueError("SCOPE cannot embed an empty conversation")
    if any(set(row) != {"role", "content"} for row in rows):
        raise ValueError("invalid SCOPE chat row")
    if rows[0]["role"] == "assistant":
        rows.insert(0, {"role": "user", "content": ""})
    for index, row in enumerate(rows):
        expected = "user" if index % 2 == 0 else "assistant"
        if row["role"] != expected or not isinstance(row["content"], str):
            raise ValueError("Llama Guard requires alternating user/assistant messages")
    return rows


def _chat_template_input_ids(tokenizer, chat, device):
    """Normalize Transformers tensor/BatchEncoding return variants."""
    encoded = tokenizer.apply_chat_template(chat, return_tensors="pt")
    if isinstance(encoded, Mapping):
        if "input_ids" not in encoded:
            raise ValueError("Llama Guard chat template did not return input_ids")
        encoded = encoded["input_ids"]
    if not torch.is_tensor(encoded) or encoded.ndim != 2 or encoded.shape[0] != 1:
        raise ValueError("invalid Llama Guard chat-template input_ids")
    return encoded.to(device)


class LlamaGuardSemanticEncoder:
    def __init__(
        self,
        snapshot,
        manifest,
        device="cuda",
        output_dim=1024,
        projection_seed=42,
    ):
        self.asset = verify_llama_guard_snapshot(snapshot, manifest)
        self.device = torch.device(device)
        self.output_dim = output_dim
        self.projection_seed = projection_seed
        self.requests = []
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.asset["path"], local_files_only=True
        )
        self.model = _load_local_causal_lm(self.asset["path"], self.device)
        safe_unsafe = self.tokenizer(
            "safeunsafe", return_attention_mask=False
        )["input_ids"]
        categories = self.tokenizer(
            list(map(str, range(1, 12))), return_attention_mask=False
        )["input_ids"]
        category_indices = [row[0] for row in categories]
        # Selecting before float conversion is algebraically identical to upstream
        # and avoids materializing the full vocabulary head in FP32.
        head = self.model.lm_head.weight.detach()
        safe_matrix = head[safe_unsafe, :].float().T.cpu()
        category_matrix = head[category_indices, :].float().T.cpu()
        zeros_safe = torch.zeros((head.shape[1], len(safe_unsafe)))
        zeros_category = torch.zeros((head.shape[1], len(category_indices)))
        projection_a = torch.cat((safe_matrix, zeros_safe), dim=0)
        projection_b = torch.cat((zeros_category, category_matrix), dim=0)
        basis = torch.cat((projection_a, projection_b), dim=1)
        generator = torch.Generator(device="cpu").manual_seed(projection_seed)
        random_basis = torch.randn(output_dim, basis.shape[1], generator=generator)
        self.projection = torch.linalg.lstsq(
            random_basis.T, basis.T, driver="gelsd"
        ).solution.T.to(self.device)
        if self.projection.shape != (head.shape[1] * 2, output_dim):
            raise ValueError("SCOPE random projection shape mismatch")
        self.unsafe_prompt = self.tokenizer(
            "unsafe\nS", return_tensors="pt", return_attention_mask=False
        )["input_ids"].to(self.device)

    @torch.inference_mode()
    def embed(self, messages, request_id):
        if not request_id:
            raise ValueError("SCOPE semantic requests require stable IDs")
        started = time.monotonic()
        chat = alternating_chat(messages)
        input_ids = _chat_template_input_ids(self.tokenizer, chat, self.device)
        input_ids = torch.cat((input_ids, self.unsafe_prompt), dim=1)
        maximum = int(getattr(self.model.config, "max_position_embeddings", 8192))
        if input_ids.shape[1] > maximum:
            raise ValueError("SCOPE semantic input exceeds pinned Llama Guard context")
        hidden = self.model.model(input_ids=input_ids, use_cache=False).last_hidden_state
        vector = torch.cat((hidden[:, -4, :], hidden[:, -1, :]), dim=-1).float()
        vector = (vector @ self.projection).cpu()[0]
        if vector.shape != (self.output_dim,) or not torch.isfinite(vector).all():
            raise FloatingPointError("invalid SCOPE semantic embedding")
        record = SemanticRequest(
            request_id=request_id,
            batch_size=1,
            input_tokens=int(input_ids.shape[1]),
            latency_s=time.monotonic() - started,
            vectors_sha256=digest(vector.tolist()),
        )
        self.requests.append(record)
        return vector

    def provenance(self):
        return {
            "model_id": self.asset["model_id"],
            "revision": self.asset["revision"],
            "manifest_sha256": self.asset["manifest_sha256"],
            "projection": {
                "output_dim": self.output_dim,
                "seed": self.projection_seed,
                "algorithm": "upstream-lstsq-random-projection",
            },
        }


class NoSharedEmbedding:
    """Shared evaluator placeholder; SCOPE records its intrinsic embeddings itself."""

    def __init__(self):
        self.requests = []

    def embed(self, *_args, **_kwargs):
        raise RuntimeError("SCOPE-CRS-lite must not call the shared Qwen embedding backend")


def target_messages(scenario):
    visible = {
        "target": scenario["target"],
        "profile": scenario["profile"],
        "knowledge": scenario["knowledge"],
    }
    return [
        Message(
            "user",
            "Recommendation goal and permitted context:\n"
            + json.dumps(visible, ensure_ascii=False, sort_keys=True),
        )
    ]
