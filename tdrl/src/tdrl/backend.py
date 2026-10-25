import ipaddress
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from urllib.parse import urlsplit

import httpx
import torch

from .contracts import Decoding, Embeddings, Generation, Message, ModelProvenance
from .provenance import atomic_json, digest, file_hash

MODEL_IDS = {"generation": "Qwen/Qwen3.5-9B", "embedding": "Qwen/Qwen3-Embedding-8B"}
DPDP_CRITIC_ROLES = {"dpdp_critic", "dpdp_rollout_critic"}
DPDP_REWARD_LABELS = ["rejected", "no_progress", "interested", "accepted"]
EMBEDDING_PREPROCESSING = {
    "version": 1,
    "format": "Instruct: {instruction}\nQuery:{text}",
    "normalization": "l2",
    "dimensions": 4096,
    "truncation": "reject",
}


def final_generation_text(text, role, request_id, repair_request_id=None):
    """Validate server text; permit one explicitly audited EnPL tag repair."""
    if isinstance(text, str) and "<think>" not in text and "</think>" not in text:
        return text, False
    if (
        isinstance(text, str)
        and role == "planner"
        and request_id == repair_request_id
        and ("<think>" in text or "</think>" in text)
    ):
        repaired = text.replace("<think>", "").replace("</think>", "").strip()
        if repaired and "<think>" not in repaired and "</think>" not in repaired:
            return repaired, True
    raise ValueError("expected final text only")


def loopback_url(url: str) -> str:
    p = urlsplit(url)
    try:
        local = ipaddress.ip_address(p.hostname or "").is_loopback
    except ValueError:
        local = False
    if p.scheme != "http" or not local or p.username or p.password or p.query or p.fragment:
        raise ValueError("inference transport requires numeric loopback HTTP; redirects forbidden")
    if p.path.rstrip("/") != "/v1":
        raise ValueError("expected /v1 adapter root")
    return url.rstrip("/")


def allocated_gpu_uuid():
    """Canonical NVML UUID for the CUDA-visible allocated device, never a physical index."""
    from uuid import UUID

    value = str(torch.cuda.get_device_properties(0).uuid)
    return "GPU-" + str(UUID(value.removeprefix("GPU-")))


def require_h100_slurm():
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("inference requires a Slurm allocation")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("exactly one visible H100 required")
    p = torch.cuda.get_device_properties(0)
    if "H100" not in p.name or p.total_memory < 75 * 1024**3:
        raise RuntimeError("one full H100 80 GB required; MIG is forbidden")
    return {
        "model": p.name,
        "count": 1,
        "total_memory_bytes": p.total_memory,
        "uuid": allocated_gpu_uuid(),
    }


def verify_model_cache(lock_path: Path, model_root: Path) -> dict:
    lock = json.loads(lock_path.read_text())
    if lock.get("schema_version") != 1 or len(lock.get("models", [])) != 2:
        raise ValueError("invalid model lock")
    snapshots = {}
    for entry in lock["models"]:
        if entry["model_id"] not in MODEL_IDS.values() or not re.fullmatch(
            r"[0-9a-f]{40}", entry["revision"]
        ):
            raise ValueError("wrong model or mutable revision")
        weights = entry["weight_sha256"]
        if not weights:
            raise ValueError("missing model weight hashes")
        snapshot = model_root / entry["model_id"].split("/")[-1] / entry["revision"]
        for filename, expected in weights.items():
            path = snapshot / filename
            if not path.is_file() or file_hash(path) != expected:
                raise ValueError(f"missing or mismatched weight: {path}")
        if set(weights) != {p.name for p in snapshot.glob("*.safetensors")}:
            raise ValueError("unexpected weight files")
        template_path = snapshot / "chat_template.jinja"
        if template_path.exists():
            template_hash = file_hash(template_path)
        else:
            tokenizer = json.loads((snapshot / "tokenizer_config.json").read_text())
            template_hash = digest(tokenizer["chat_template"])
        snapshots[entry["model_id"]] = {
            "path": str(snapshot.resolve()),
            "template_sha256": template_hash,
            **entry,
            "verified_weights_locally": True,
        }
    if set(snapshots) != set(MODEL_IDS.values()):
        raise ValueError("missing required model")
    return snapshots


class LocalBackend:
    def __init__(
        self,
        url: str,
        model: ModelProvenance,
        kind: str,
        journal=None,
        ledger=None,
        request_concurrency=1,
    ):
        self.url = loopback_url(url)
        if kind not in MODEL_IDS or model.model_id != MODEL_IDS[kind]:
            raise ValueError("model substitution forbidden")
        if not re.fullmatch(r"[0-9a-f]{40}", model.revision) or not model.weight_sha256:
            raise ValueError("immutable weight provenance required")
        if not isinstance(request_concurrency, int) or request_concurrency < 1:
            raise ValueError("request concurrency must be a positive integer")
        self.model, self.kind, self.journal = model, kind, journal
        self.ledger = ledger
        self.request_concurrency = request_concurrency
        self.requests = []
        self.consumed_requests = []
        self.client = httpx.Client(
            trust_env=False,
            follow_redirects=False,
            timeout=180,
            limits=httpx.Limits(max_keepalive_connections=0),
        )

    def _begin(self, request_id, request):
        if self.ledger is None:
            return None
        path = self.ledger / ("pending-" + digest(request_id) + ".json")
        if path.exists():
            raise ValueError(
                "in-flight request outcome is unresolved; retain evidence before rerunning paired methods"
            )
        authorized = getattr(self, "authorized_reissue", None)
        if authorized and request_id == authorized["request_id"]:
            if digest(request) != authorized["request_sha256"]:
                raise ValueError("audited recovery request payload mismatch")
            atomic_json(self.ledger.parent / "authorized-reissue.json", authorized)
            self.authorized_reissue = None
        atomic_json(path, {"request_id": request_id, "request_sha256": digest(request)})
        return path

    def _finish(self, pending, record):
        if pending is None:
            return
        import uuid

        atomic_json(self.ledger / ("completed-" + uuid.uuid4().hex + ".json"), record)
        pending.unlink()

    def _post(self, endpoint, payload):
        require_h100_slurm()
        start = time.monotonic()
        r = self.client.post(self.url + endpoint, json=payload)
        if r.is_redirect:
            raise ValueError("inference redirect forbidden")
        r.raise_for_status()
        return r.json(), time.monotonic() - start

    def _generate_unrecorded(
        self, messages: list[Message], role: str, decoding: Decoding, seed: int, request_id: str
    ) -> tuple[Generation, bool]:
        if self.kind != "generation" or seed < 0 or not request_id:
            raise ValueError("invalid generation request")
        payload = {
            "model": self.model.model_id,
            "messages": [asdict(m) for m in messages],
            **asdict(decoding),
            "seed": seed,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        if role in {"judge", "critic"}:
            from .judge import judgment_schema

            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "goal_judgment",
                    "schema": judgment_schema(json.loads(messages[-1].content)),
                },
            }
        elif role in DPDP_CRITIC_ROLES:
            payload["structured_outputs"] = {"choice": DPDP_REWARD_LABELS}
        request = {"payload": payload, "role": role, "model": asdict(self.model)}
        cached = self.journal.read(request_id, request) if self.journal else None
        if cached is not None:
            output = Generation(
                **{
                    **cached,
                    "model": ModelProvenance(**cached["model"]),
                    "decoding": Decoding(**cached["decoding"]),
                }
            )
            return output, True
        pending = self._begin(request_id, request)
        result, latency = self._post("/chat/completions", payload)
        if result["model"] != self.model.model_id:
            raise ValueError("server model mismatch")
        raw_text = result["choices"][0]["message"]["content"]
        text, repaired = final_generation_text(
            raw_text,
            role,
            request_id,
            getattr(self, "audited_final_text_repair_request_id", None),
        )
        if repaired:
            if self.ledger is None:
                raise ValueError("audited final-text repair requires request ledger")
            atomic_json(
                self.ledger.parent / "format-repairs" / (digest(request_id) + ".json"),
                {
                    "schema_version": 1,
                    "request_id": request_id,
                    "role": role,
                    "repair": "remove_literal_think_transport_tags_only",
                    "raw_text_sha256": digest(raw_text),
                    "repaired_text_sha256": digest(text),
                },
            )
        usage = result["usage"]
        counts = [usage[k] for k in ("prompt_tokens", "completion_tokens", "total_tokens")]
        if any(not isinstance(v, int) or v < 0 for v in counts) or sum(counts[:2]) != counts[2]:
            raise ValueError("invalid token counts")
        finish_reason = result["choices"][0].get("finish_reason") or "unknown"
        if finish_reason not in {"stop", "length", "unknown"}:
            raise ValueError("unexpected generation finish reason")
        output = Generation(
            text,
            *counts,
            seed,
            decoding,
            latency,
            0,
            self.model,
            digest(request),
            request_id,
            role,
            finish_reason,
        )
        if self.journal:
            self.journal.write(request_id, request, asdict(output))
        self._finish(pending, asdict(output))
        return output, False

    def generate(
        self, messages: list[Message], role: str, decoding: Decoding, seed: int, request_id: str
    ) -> Generation:
        output, replayed = self._generate_unrecorded(
            messages, role, decoding, seed, request_id
        )
        if not replayed:
            self.requests.append(output)
        self.consumed_requests.append(output)
        return output

    def generate_carp_legacy(self, prompt: str, role: str, max_tokens: int, request_id: str):
        """Issue the legacy CARP evaluator request without CRS decoding extras.

        CARP sends one plain user message, temperature zero, ``stream=false`` and
        optionally ``max_tokens``.  In particular it does not send a seed,
        top-p/top-k controls, or a structured-output schema.  Keeping this as a
        separate entry point prevents the legacy evaluator from weakening the
        typed contract used by every existing CRS experiment.
        """
        if (
            self.kind != "generation"
            or not isinstance(prompt, str)
            or not prompt
            or role not in {"carp_assistant", "carp_user_simulator", "carp_moderator"}
            or not isinstance(max_tokens, int)
            or max_tokens < 1
            or not request_id
        ):
            raise ValueError("invalid CARP legacy generation request")
        payload = {
            "model": self.model.model_id,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": max_tokens,
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        request = {
            "payload": payload,
            "role": role,
            "model": asdict(self.model),
            "protocol": "carp_legacy_v1",
        }
        cached = self.journal.read(request_id, request) if self.journal else None
        decoding = Decoding(temperature=0.0, top_p=1.0, top_k=-1, max_tokens=max_tokens)
        if cached is not None:
            output = Generation(
                **{
                    **cached,
                    "model": ModelProvenance(**cached["model"]),
                    "decoding": Decoding(**cached["decoding"]),
                }
            )
            self.consumed_requests.append(output)
            return output
        pending = self._begin(request_id, request)
        result, latency = self._post("/chat/completions", payload)
        if result["model"] != self.model.model_id:
            raise ValueError("server model mismatch")
        choices = result.get("choices") or []
        raw_text = choices[0].get("message", {}).get("content", "") if choices else ""
        text, _ = final_generation_text(raw_text, role, request_id)
        usage = result.get("usage") or {}
        counts = [usage.get(key, 0) for key in ("prompt_tokens", "completion_tokens")]
        total = usage.get("total_tokens", sum(counts))
        if (
            any(not isinstance(value, int) or value < 0 for value in (*counts, total))
            or sum(counts) != total
        ):
            raise ValueError("invalid token counts")
        finish_reason = choices[0].get("finish_reason") if choices else "unknown"
        finish_reason = finish_reason or "unknown"
        if finish_reason not in {"stop", "length", "unknown"}:
            raise ValueError("unexpected generation finish reason")
        # The sentinel is provenance only: no seed is present in the HTTP payload.
        output = Generation(
            text,
            counts[0],
            counts[1],
            total,
            -1,
            decoding,
            latency,
            0,
            self.model,
            digest(request),
            request_id,
            role,
            finish_reason,
        )
        if self.journal:
            self.journal.write(request_id, request, asdict(output))
        self._finish(pending, asdict(output))
        self.requests.append(output)
        self.consumed_requests.append(output)
        return output

    def generate_many(self, requests, max_workers=None):
        """Execute independent requests concurrently and record them in source order."""
        requests = list(requests)
        if not requests:
            return []
        workers = self.request_concurrency if max_workers is None else max_workers
        if not isinstance(workers, int) or not 1 <= workers <= self.request_concurrency:
            raise ValueError("invalid generation worker count")
        request_ids = [request[4] for request in requests]
        if len(request_ids) != len(set(request_ids)):
            raise ValueError("concurrent generation request IDs must be unique")
        if workers == 1:
            return [self.generate(*request) for request in requests]

        def execute(request):
            return self._generate_unrecorded(*request)

        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="local-generation") as pool:
            completed = list(pool.map(execute, requests))
        outputs = [output for output, _ in completed]
        self.requests.extend(output for output, replayed in completed if not replayed)
        self.consumed_requests.extend(outputs)
        return outputs

    def embed(
        self, texts: list[str], instruction: str, batch_id: str, request_id: str
    ) -> Embeddings:
        if self.kind != "embedding" or not texts or not instruction:
            raise ValueError("invalid embedding request")
        inputs = [f"Instruct: {instruction}\nQuery:{t}" for t in texts]
        pending = self._begin(request_id, {"inputs": inputs, "model": asdict(self.model)})
        result, latency = self._post(
            "/embeddings",
            {"model": self.model.model_id, "input": inputs, "encoding_format": "float"},
        )
        if result["model"] != self.model.model_id:
            raise ValueError("embedding server model mismatch")
        rows = sorted(result["data"], key=lambda row: row["index"])
        if [r["index"] for r in rows] != list(range(len(texts))):
            raise ValueError("unstable/missing embedding source order")
        vectors = torch.tensor([r["embedding"] for r in rows], dtype=torch.float32)
        if vectors.shape != (len(texts), 4096) or not torch.isfinite(vectors).all():
            raise ValueError("expected finite full 4096-D embeddings")
        norms = vectors.norm(dim=-1, keepdim=True)
        if (norms <= 1e-8).any():
            raise ValueError("zero embedding")
        vectors = vectors / norms
        output = Embeddings(
            vectors,
            self.model,
            digest(instruction),
            digest(EMBEDDING_PREPROCESSING),
            len(texts),
            latency,
            result.get("usage", {}).get("prompt_tokens"),
            None,
            batch_id,
            request_id,
        )
        self._finish(
            pending,
            {
                "request_id": request_id,
                "batch_size": len(texts),
                "input_tokens": output.input_tokens,
                "latency_s": latency,
                "model": asdict(self.model),
                "instruction_sha256": output.instruction_sha256,
                "preprocessing_sha256": output.preprocessing_sha256,
                "vectors_sha256": digest(vectors.tolist()),
            },
        )
        self.requests.append(output)
        return output
