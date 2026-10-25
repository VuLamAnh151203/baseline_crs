import json
import math
import re
from dataclasses import asdict, dataclass

from ...contracts import Decoding, Message
from ...planner import derived_seed
from ...provenance import digest

GUIDANCE_POLICY = (
    "Create exactly five concise and meaningfully different recommendation utterances that can "
    "guide a target-oriented dialogue toward the supplied action and item. Use only the supplied "
    "profile and permitted knowledge. Do not number the lines. Return exactly five nonempty lines "
    "and no commentary."
)


@dataclass(frozen=True)
class GuidanceArtifact:
    schema_version: int
    dataset: str
    split: str
    scenario_id: str
    seed: int
    request_id: str
    prompt_sha256: str
    target_sha256: str
    guidance: list[str]
    generation_request_id: str
    latency_s: float
    input_tokens: int
    output_tokens: int
    total_tokens: int

    def __post_init__(self):
        if self.schema_version != 1 or self.dataset not in {"TopDial", "DuRecDial2.0"}:
            raise ValueError("unknown DiffTOD guidance schema or dataset")
        if self.split not in {"dev", "test"}:
            raise ValueError("unknown DiffTOD guidance split")
        if not self.scenario_id or self.seed < 0 or not self.request_id:
            raise ValueError("invalid DiffTOD guidance identity")
        if not self.generation_request_id:
            raise ValueError("DiffTOD guidance requires a generation request ID")
        if any(
            not re.fullmatch(r"[0-9a-f]{64}", value)
            for value in (self.prompt_sha256, self.target_sha256)
        ):
            raise ValueError("DiffTOD guidance provenance requires SHA-256 hashes")
        if len(self.guidance) != 5:
            raise ValueError("DiffTOD requires exactly five semantic guidance sentences")
        if any(not item.strip() for item in self.guidance) or len(set(self.guidance)) != 5:
            raise ValueError("DiffTOD semantic guidance must be nonempty and distinct")
        if not math.isfinite(self.latency_s) or self.latency_s < 0:
            raise ValueError("DiffTOD guidance latency must be finite and nonnegative")
        if any(
            not isinstance(value, int) or value < 0
            for value in (self.input_tokens, self.output_tokens, self.total_tokens)
        ):
            raise ValueError("DiffTOD guidance compute counters must be nonnegative")
        if self.total_tokens != self.input_tokens + self.output_tokens:
            raise ValueError("DiffTOD guidance token accounting mismatch")

    @property
    def sha256(self):
        return digest(asdict(self))


def guidance_prompt(record):
    payload = {
        "target": record["target"],
        "profile": record["profile"],
        "permitted_knowledge": record["knowledge"],
    }
    return [
        Message("system", GUIDANCE_POLICY),
        Message("user", json.dumps(payload, ensure_ascii=False, sort_keys=True)),
    ]


def parse_guidance(text):
    lines = []
    for raw in text.splitlines():
        line = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", raw).strip()
        if line and line not in lines:
            lines.append(line)
    if len(lines) != 5:
        raise ValueError(
            f"Qwen guidance response must contain exactly five distinct lines, got {len(lines)}"
        )
    return lines


def generate_guidance(backend, record, dataset, split, seed, request_id):
    messages = guidance_prompt(record)
    generation_id = request_id + "/semantic-guidance"
    result = backend.generate(
        messages,
        "difftod_guidance",
        Decoding(temperature=0.7, top_p=0.8, top_k=20, max_tokens=384),
        derived_seed(seed, record["id"], "difftod_guidance"),
        generation_id,
    )
    return GuidanceArtifact(
        schema_version=1,
        dataset=dataset,
        split=split,
        scenario_id=record["id"],
        seed=seed,
        request_id=request_id,
        prompt_sha256=digest([{"role": m.role, "content": m.content} for m in messages]),
        target_sha256=digest(record["target"]),
        guidance=parse_guidance(result.text),
        generation_request_id=result.request_id,
        latency_s=result.latency_s,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        total_tokens=result.total_tokens,
    )
