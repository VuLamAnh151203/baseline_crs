import json
import time
from pathlib import Path

import torch

from ...planner import derived_seed
from .checkpoint import ExponentialMovingAverage
from .config import DiffTODConfig
from .diffusion import AbsorbingGraph, LogLinearNoise, TokenProjection, sample_trajectory
from .guidance import GuidanceArtifact
from .mbr import select_mbr
from .model import SEDD, load_weights
from .plan_store import PlanArtifact, PlanStore
from .serialization import condition_header, extract_dialogue_plan


def _token_ids(tokenizer, text):
    encoded = tokenizer(text, add_special_tokens=False, return_attention_mask=False)
    ids = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
    if not isinstance(ids, list) or any(
        not isinstance(value, int) or value < 0 for value in ids
    ):
        raise ValueError("tokenizer returned invalid token IDs")
    return ids


def build_projection(tokenizer, header, guidance, config: DiffTODConfig, device):
    if len(guidance) != config.sampling.candidates:
        raise ValueError("guidance count differs from configured DiffTOD candidates")
    header_ids = _token_ids(tokenizer, header)
    semantic_ids = [_token_ids(tokenizer, "system: " + sentence) for sentence in guidance]
    length = config.model.length
    longest_semantic = max(map(len, semantic_ids))
    maximum_header = length - config.sampling.semantic_offset - longest_semantic
    if maximum_header < 1:
        raise ValueError("DiffTOD sequence leaves no room for its condition header")
    header_ids = header_ids[:maximum_header]
    start = len(header_ids) + config.sampling.semantic_offset
    mask = torch.zeros((len(guidance), length), dtype=torch.bool, device=device)
    values = torch.zeros((len(guidance), length), dtype=torch.long, device=device)
    header_tensor = torch.tensor(header_ids, dtype=torch.long, device=device)
    mask[:, : len(header_ids)] = True
    values[:, : len(header_ids)] = header_tensor
    for row, tokens in enumerate(semantic_ids):
        end = start + len(tokens)
        if start <= len(header_ids) or end > length:
            raise ValueError("semantic guidance collides with header or exceeds model length")
        mask[row, start:end] = True
        values[row, start:end] = torch.tensor(tokens, dtype=torch.long, device=device)
    if values[mask].numel() and values[mask].max() >= config.model.vocab_size:
        raise ValueError("condition token is outside the configured GPT-2 vocabulary")
    return TokenProjection(mask, values)


def load_guidance(path: Path):
    return GuidanceArtifact(**json.loads(path.read_text(encoding="utf-8")))


def load_sampling_model(checkpoint_path, config: DiffTODConfig, device):
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != 2
        or "model" not in payload
        or "ema" not in payload
    ):
        raise ValueError("sampling requires a complete DiffTOD training checkpoint")
    if payload.get("config_sha256") != config.training_sha256:
        raise ValueError("DiffTOD checkpoint/model-training configuration mismatch")
    model = SEDD(config.model).to(device)
    load_weights(model, payload["model"], strict=True)
    ema = ExponentialMovingAverage(model.parameters(), config.training.ema_decay)
    ema.load_state_dict(payload["ema"])
    ema.copy_to(model.parameters())
    model.eval()
    return model


@torch.no_grad()
def generate_plan(
    model,
    tokenizer,
    record,
    dataset,
    split,
    guidance: GuidanceArtifact,
    config: DiffTODConfig,
    checkpoint_sha256,
    seed,
    budget,
    plan_store: PlanStore,
):
    if (guidance.dataset, guidance.split, guidance.scenario_id, guidance.seed) != (
        dataset,
        split,
        record["id"],
        seed,
    ):
        raise ValueError("DiffTOD guidance identity does not match plan request")
    device = next(model.parameters()).device
    header = condition_header(record, dataset)
    used_guidance = guidance.guidance[: config.sampling.candidates]
    projection = build_projection(tokenizer, header, used_guidance, config, device)
    sampling_seed = derived_seed(seed, record["id"], f"difftod_{budget}")
    generator = torch.Generator(device=device).manual_seed(sampling_seed)
    graph = AbsorbingGraph(config.model.vocab_size)
    noise = LogLinearNoise().to(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = time.monotonic()
    samples = sample_trajectory(
        model,
        graph,
        noise,
        config.sampling.candidates,
        config.model.length,
        steps=config.sampling.steps,
        epsilon=config.sampling.epsilon,
        projection=projection,
        generator=generator,
        denoise=config.sampling.denoise,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    wall_s = time.monotonic() - started
    peak_memory_bytes = (
        torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    )
    decoded = tokenizer.batch_decode(samples.detach().cpu().tolist(), skip_special_tokens=False)
    plans = [extract_dialogue_plan(text, header) for text in decoded]
    selection = select_mbr(plans)
    model_calls = config.sampling.steps + int(config.sampling.denoise)
    artifact = PlanArtifact(
        schema_version=1,
        dataset=dataset,
        split=split,
        scenario_id=record["id"],
        seed=seed,
        budget=budget,
        checkpoint_sha256=checkpoint_sha256,
        config_sha256=config.sha256,
        guidance_sha256=guidance.sha256,
        guidance=used_guidance,
        candidate_plans=plans,
        self_bleu_scores=selection["scores"],
        selected_candidate=selection["selected"],
        selected_plan=plans[selection["selected"]],
        diffusion_steps=config.sampling.steps,
        diffusion_model_calls=model_calls,
        diffusion_sequence_forwards=model_calls * config.sampling.candidates,
        wall_s=wall_s,
        guidance_wall_s=guidance.latency_s,
        peak_memory_bytes=peak_memory_bytes,
    )
    plan_store.write(artifact)
    return artifact
