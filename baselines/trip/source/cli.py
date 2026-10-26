import argparse
import json
import random
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from ...backend import require_h100_slurm
from ...provenance import atomic_json, file_hash
from .assets import verify_assets
from .config import TripConfig
from .dataset import TripCollator, TripDataset, _target_token_mask
from .decoding import generate_plan
from .model import TripModel, load_pretrained_model
from .serialization import SPECIAL_TOKENS, prepare_corpus
from .training import TripTrainer


class _TinyEncoder(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)

    def resize_token_embeddings(self, size):
        if size != self.embedding.num_embeddings:
            self.embedding = nn.Embedding(size, self.embedding.embedding_dim)

    def forward(self, input_ids, attention_mask=None, return_dict=True):
        del attention_mask, return_dict
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))


class _TinyTokenizer:
    pad_token_id = 0

    def __init__(self):
        self.ids = {token: index + 4 for index, token in enumerate(SPECIAL_TOKENS)}

    def convert_tokens_to_ids(self, token):
        return self.ids[token]

    def decode(self, ids, skip_special_tokens=True):
        reverse = {value: key for key, value in self.ids.items()}
        values = []
        for token in ids:
            text = reverse.get(int(token), f"tok{int(token)}")
            if skip_special_tokens and text in self.ids:
                continue
            values.append(text)
        return " ".join(values)


def smoke(output, require_h100=False):
    output.mkdir(parents=True, exist_ok=True)
    gpu = require_h100_slurm() if require_h100 else None
    device = torch.device("cuda" if require_h100 else "cpu")
    torch.manual_seed(17)
    config = TripConfig.smoke()
    tokenizer = _TinyTokenizer()
    token_ids = tokenizer.ids
    vocab_size = 32
    model = TripModel(
        _TinyEncoder(vocab_size, config.model.hidden_size),
        _TinyEncoder(vocab_size, config.model.hidden_size),
        vocab_size,
        tokenizer.pad_token_id,
        config.model,
    ).to(device)
    context = torch.tensor([[1, 2, 3], [2, 3, 1]])
    target = torch.tensor([[1, 4, 8, 5, 9], [1, 4, 10, 5, 11]])
    forward = torch.tensor([[6, 4, 8, 5, 9, 7], [6, 4, 10, 5, 11, 7]])
    backward = forward.clone()
    negatives = torch.tensor(
        [
            [[6, 4, 8, 5, 12, 7], [6, 4, 8, 5, 13, 7]],
            [[6, 4, 10, 5, 12, 7], [6, 4, 10, 5, 13, 7]],
        ]
    )
    target_mask = torch.tensor([[0, 1, 1, 1, 1, 0], [0, 1, 1, 1, 1, 0]])
    negative_target_mask = target_mask[:, None, :].expand(-1, 2, -1).clone()
    batch = {
        "context_ids": context,
        "context_attention_mask": torch.ones_like(context),
        "target_ids": target,
        "target_attention_mask": torch.ones_like(target),
        "forward_ids": forward,
        "forward_attention_mask": torch.ones_like(forward),
        "forward_target_mask": target_mask,
        "backward_ids": backward,
        "backward_attention_mask": torch.ones_like(backward),
        "backward_target_mask": target_mask,
        "negative_forward_ids": negatives,
        "negative_forward_attention_mask": torch.ones_like(negatives),
        "negative_forward_target_mask": negative_target_mask,
        "negative_backward_ids": negatives,
        "negative_backward_attention_mask": torch.ones_like(negatives),
        "negative_backward_target_mask": negative_target_mask,
    }
    batch = {key: value.to(device) for key, value in batch.items()}
    result = model(batch)
    if not torch.isfinite(result["loss"]):
        raise FloatingPointError("TRIP smoke loss is non-finite")
    result["loss"].backward()
    plan = generate_plan(
        model,
        tokenizer,
        token_ids,
        [1, 2, 3],
        [1, 4, 8, 5, 9],
        [4, 8, 5, 9],
        config.decoding,
    )
    report = {
        "schema_version": 1,
        "status": "smoke_non_scientific",
        "device": str(device),
        "h100": gpu,
        "finite_loss": True,
        "loss": float(result["loss"].detach()),
        "generated_pairs": [list(pair) for pair in plan.pairs],
        "sequence_forwards": plan.sequence_forwards,
        "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
    }
    atomic_json(output / "smoke.json", report)
    return report


def smoke_model(args):
    gpu = require_h100_slurm()
    assets = verify_assets(args.model_lock, args.encoder, args.tokenizer)
    config = TripConfig.from_path(args.config)
    random.seed(config.training.seed)
    np.random.seed(config.training.seed)
    torch.manual_seed(config.training.seed)
    torch.cuda.manual_seed_all(config.training.seed)
    from .runtime import load_tokenizer

    tokenizer, token_ids = load_tokenizer(args.tokenizer)
    model = load_pretrained_model(args.encoder, tokenizer, config.model, "cuda")
    action, topic = "recommend", "moon"
    target_text = f"[A]{action}[T]{topic}"
    constraint = tokenizer.encode(target_text, add_special_tokens=False)
    target = tokenizer.encode(target_text, add_special_tokens=True)
    plan = [token_ids["[BOS]"], *constraint, token_ids["[EOS]"]]
    target_mask = _target_token_mask(plan, constraint)
    negatives = []
    for negative_topic in ("sun", "stars", "cloud"):
        negative_constraint = tokenizer.encode(
            f"[A]{action}[T]{negative_topic}", add_special_tokens=False
        )
        negative_plan = [
            token_ids["[BOS]"],
            *negative_constraint,
            token_ids["[EOS]"],
        ]
        negatives.append(
            (negative_plan, _target_token_mask(negative_plan, negative_constraint))
        )
    row = {
        "id": "h100-smoke",
        "context_ids": tokenizer.encode(
            "profile: likes mystery films user: I enjoy mysteries", add_special_tokens=True
        ),
        "target_ids": target,
        "forward_ids": plan,
        "forward_target_mask": target_mask,
        "backward_ids": plan,
        "backward_target_mask": target_mask,
        "negative_forward": negatives,
        "negative_backward": negatives,
    }
    batch = TripCollator(tokenizer.pad_token_id)([row])
    batch = {
        key: value.cuda(non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }
    torch.cuda.reset_peak_memory_stats()
    model.train()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        result = model(batch)
    if not torch.isfinite(result["loss"]):
        raise FloatingPointError("full TRIP H100 smoke loss is non-finite")
    result["loss"].backward()
    model.zero_grad(set_to_none=True)
    model.to(dtype=torch.bfloat16).eval()
    smoke_decoding = replace(
        config.decoding,
        beam_size=1,
        max_plan_tokens=max(16, len(constraint) + 4),
    )
    generated = generate_plan(
        model,
        tokenizer,
        token_ids,
        row["context_ids"],
        target,
        constraint,
        smoke_decoding,
    )
    report = {
        "schema_version": 1,
        "status": "smoke_non_scientific",
        "device": "cuda",
        "precision": "bfloat16",
        "h100": gpu,
        "model_revision": assets["revision"],
        "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "finite_loss": True,
        "loss": float(result["loss"].detach()),
        "generated_pairs": [list(pair) for pair in generated.pairs],
        "sequence_forwards": generated.sequence_forwards,
        "peak_memory_bytes": torch.cuda.max_memory_allocated(),
        "config_sha256": config.sha256,
        "model_lock_sha256": file_hash(args.model_lock),
    }
    atomic_json(args.output / "smoke-model.json", report)
    return report


def smoke_shared(args):
    from ...contracts import Decoding
    from ...evaluation import initial_state, run_dialogue
    from ...planner import INSTRUCTION
    from ...runtime import Servers
    from .planner import TripPlanner
    from .runtime import load_tokenizer

    gpu = require_h100_slurm()
    assets = verify_assets(args.model_lock, args.encoder, args.tokenizer)
    config = TripConfig.from_path(args.config)
    tokenizer, token_ids = load_tokenizer(args.tokenizer)
    model = load_pretrained_model(args.encoder, tokenizer, config.model, "cuda")
    model.to(dtype=torch.bfloat16).eval()
    scenario = {
        "id": "trip-shared-h100-smoke",
        "profile": {"preference": "mystery films"},
        "personality": {},
        "knowledge": [["Moon", "type", "mystery film"]],
        "target": ["Movie recommendation", "Moon"],
        "messages": [{"role": "user", "content": "I enjoy mystery films."}],
    }
    runtime = json.loads(args.runtime_config.read_text(encoding="utf-8"))
    server_output = args.output / "servers"
    with Servers(args.root, server_output, runtime) as servers:
        backends = servers.backends()
        state = initial_state(scenario)
        embedding = backends["embedding"].embed(
            [state.serialize()],
            INSTRUCTION,
            "trip-smoke/live",
            "trip-smoke/live",
        )
        if embedding.vectors.shape != (1, 4096):
            raise ValueError("TRIP shared smoke did not receive the locked 4096-D embedding")
        planner = TripPlanner(
            backends["generation"],
            model,
            tokenizer,
            token_ids,
            config,
            scenario,
            "untrained-smoke-only",
            Decoding(max_tokens=64),
        )
        result = run_dialogue(
            scenario,
            planner,
            backends["generation"],
            backends["embedding"],
            "profile",
            17,
            args.output / "dialogue",
            {
                "phase": "smoke",
                "method": "TRIP",
                "selection_sha256": "smoke",
                "budget": "smoke",
                "target_nodes": [],
                "request_namespace": config.sha256,
            },
            turn_budget=1,
            simulator_decoding=Decoding(max_tokens=64),
        )
        compute = servers.compute_metadata()
    report = {
        "schema_version": 1,
        "status": "smoke_non_scientific",
        "device": "cuda",
        "precision": "bfloat16",
        "h100": gpu,
        "model_revision": assets["revision"],
        "live_embedding_shape": list(embedding.vectors.shape),
        "dialogue_status": result["status"],
        "dialogue_turns": result["actual_turns"],
        "server_compute": compute,
        "config_sha256": config.sha256,
        "model_lock_sha256": file_hash(args.model_lock),
    }
    atomic_json(args.output / "smoke-shared.json", report)
    return report


def train(args):
    gpu = require_h100_slurm()
    assets = verify_assets(args.model_lock, args.encoder, args.tokenizer)
    config = TripConfig.from_path(args.config)
    random.seed(config.training.seed)
    np.random.seed(config.training.seed)
    torch.manual_seed(config.training.seed)
    torch.cuda.manual_seed_all(config.training.seed)
    torch.backends.cudnn.benchmark = False
    from .runtime import load_tokenizer

    tokenizer, _ = load_tokenizer(args.tokenizer)
    training = TripDataset(
        args.train, tokenizer, config.max_context_tokens, config.max_plan_tokens
    )
    validation = TripDataset(
        args.dev, tokenizer, config.max_context_tokens, config.max_plan_tokens
    )
    model = load_pretrained_model(args.encoder, tokenizer, config.model, "cuda")
    trainer = TripTrainer(model, config, "cuda", tokenizer.pad_token_id)
    started = time.monotonic()
    result = trainer.fit(
        training,
        validation,
        args.output,
        resume=args.resume,
        stop_requested=lambda: (args.output / "stop.requested").exists(),
    )
    result.update(
        status="complete",
        gpu=gpu,
        wall_s=time.monotonic() - started,
        config_sha256=config.sha256,
        training_config_sha256=config.training_sha256,
        train_sha256=file_hash(args.train),
        dev_sha256=file_hash(args.dev),
        model_revision=assets["revision"],
        model_lock_sha256=file_hash(args.model_lock),
    )
    atomic_json(args.output / "training-summary.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description="TRIP shared-setting baseline")
    sub = parser.add_subparsers(dest="command", required=True)
    command = sub.add_parser("prepare")
    command.add_argument("--input", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--dataset", choices=("TopDial", "DuRecDial2.0"), required=True)
    command.add_argument("--split", choices=("train", "dev"), required=True)
    command.add_argument("--source-archive", type=Path)
    command.add_argument("--num-negatives", type=int, default=3)
    command = sub.add_parser("smoke")
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--require-h100", action="store_true")
    command = sub.add_parser("smoke-model")
    command.add_argument("--config", type=Path, required=True)
    command.add_argument("--encoder", type=Path, required=True)
    command.add_argument("--tokenizer", type=Path, required=True)
    command.add_argument("--model-lock", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command = sub.add_parser("smoke-shared")
    command.add_argument("--root", type=Path, required=True)
    command.add_argument("--runtime-config", type=Path, required=True)
    command.add_argument("--config", type=Path, required=True)
    command.add_argument("--encoder", type=Path, required=True)
    command.add_argument("--tokenizer", type=Path, required=True)
    command.add_argument("--model-lock", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command = sub.add_parser("train")
    command.add_argument("--config", type=Path, required=True)
    command.add_argument("--train", type=Path, required=True)
    command.add_argument("--dev", type=Path, required=True)
    command.add_argument("--encoder", type=Path, required=True)
    command.add_argument("--tokenizer", type=Path, required=True)
    command.add_argument("--model-lock", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--resume", type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare_corpus(
            args.input,
            args.output,
            args.dataset,
            args.split,
            args.source_archive,
            args.num_negatives,
        )
        atomic_json(args.output.with_suffix(args.output.suffix + ".manifest.json"), result)
    elif args.command == "smoke":
        result = smoke(args.output, args.require_h100)
    elif args.command == "smoke-model":
        args.output.mkdir(parents=True, exist_ok=True)
        result = smoke_model(args)
    elif args.command == "smoke-shared":
        args.output.mkdir(parents=True, exist_ok=True)
        result = smoke_shared(args)
    else:
        result = train(args)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
