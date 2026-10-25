import argparse
import json
import time
from pathlib import Path

import torch

from ...backend import require_h100_slurm
from ...official_data import SPLITS, read_release
from ...provenance import atomic_json, digest, file_hash
from ...runtime import Servers
from .assets import verify_assets
from .checkpoint import checkpoint_signature
from .config import DiffTODConfig, ModelConfig
from .diffusion import AbsorbingGraph, LogLinearNoise, TokenProjection, sample_trajectory
from .guidance import generate_guidance
from .guidance_store import GuidanceStore
from .model import SEDD, load_weights
from .plan_store import PlanStore
from .sampling import generate_plan, load_sampling_model
from .selection import verify_selection
from .serialization import atomic_text, initial_observation, prepare_corpus
from .training import DiffTODTrainer, token_blocks


def _records(path):
    with path.open(encoding="utf-8") as source:
        rows = [json.loads(line) for line in source if line.strip()]
    if not rows or len({row.get("id") for row in rows}) != len(rows):
        raise ValueError("scenario file must contain unique, nonempty IDs")
    return rows


def _verified_scenarios(path, dataset, split, selection_sha256):
    rows = _records(path)
    manifest_path = path.with_suffix(path.suffix + ".manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {
        "schema_version": 1,
        "status": "complete",
        "dataset": dataset,
        "split": split,
        "selection_sha256": selection_sha256,
        "records": len(rows),
        "ids_sha256": digest([row["id"] for row in rows]),
        "output_sha256": file_hash(path),
    }
    if manifest != expected:
        raise ValueError("DiffTOD scenario manifest does not match its data or stage")
    return rows


def _tokenizer(path):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, use_fast=True)
    if tokenizer.eos_token_id is None or tokenizer.vocab_size != 50_257:
        raise ValueError("faithful DiffTOD requires the pinned GPT-2 tokenizer")
    return tokenizer


def _verified_lock(path):
    return verify_selection(path)["sha256"]


def _test_gate(split, selection_lock):
    if split.startswith("test"):
        if selection_lock is None:
            raise ValueError("test guidance/plans require a verified selection lock")
        return _verified_lock(selection_lock)
    if selection_lock is not None:
        raise ValueError("selection lock is accepted only for a test stage")
    return None


def export_scenarios(args):
    selection_sha256 = _test_gate(args.split, args.selection_lock)
    if args.split == "dev":
        source = args.root / "data/processed" / args.dataset / "dev.jsonl"
        rows = _records(source)
        comparison = json.loads(
            (args.root / "configs/comparison.difftod.json").read_text(encoding="utf-8")
        )
        count = comparison.get("development_scenarios")
        if not isinstance(count, int) or count < 1:
            raise ValueError("invalid DiffTOD development scenario count")
        rows = sorted(rows, key=lambda row: digest([args.dataset, row["id"]]))[:count]
    else:
        members = [name for name in SPLITS[args.dataset] if name.startswith("test")]
        rows = [
            record
            for member in members
            for record in read_release(args.root, args.dataset, member, selection_verified=True)
        ]
    rows = [{**row, "messages": initial_observation(row)} for row in rows]
    content = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    )
    atomic_text(args.output, content)
    report = {
        "schema_version": 1,
        "status": "complete",
        "dataset": args.dataset,
        "split": args.split,
        "selection_sha256": selection_sha256,
        "records": len(rows),
        "ids_sha256": digest([row["id"] for row in rows]),
        "output_sha256": file_hash(args.output),
    }
    atomic_json(args.output.with_suffix(args.output.suffix + ".manifest.json"), report)
    return report


def smoke(output, require_h100=False):
    output.mkdir(parents=True, exist_ok=True)
    gpu = require_h100_slurm() if require_h100 else None
    device = torch.device("cuda" if require_h100 else "cpu")
    torch.manual_seed(17)
    config = ModelConfig.smoke()
    model = SEDD(config).to(device)
    graph, noise = AbsorbingGraph(config.vocab_size), LogLinearNoise().to(device)
    clean = torch.randint(0, config.vocab_size, (2, config.length), device=device)
    time_values = torch.tensor([0.2, 0.8], device=device)
    sigma = noise(time_values)[0]
    noisy = graph.sample_transition(clean, sigma[:, None])
    scores = model(noisy, sigma)
    loss = graph.score_entropy(scores, sigma[:, None], noisy, clean).mean()
    if not torch.isfinite(loss):
        raise FloatingPointError("DiffTOD smoke loss is non-finite")
    loss.backward()
    mask = torch.zeros((2, config.length), dtype=torch.bool, device=device)
    values = torch.zeros((2, config.length), dtype=torch.long, device=device)
    mask[:, :2] = True
    values[:, :2] = clean[:, :2]
    sampled = sample_trajectory(
        model,
        graph,
        noise,
        2,
        config.length,
        steps=4,
        projection=TokenProjection(mask, values),
        generator=torch.Generator(device=device).manual_seed(29),
    )
    if sampled.shape != clean.shape or not torch.equal(sampled[:, :2], clean[:, :2]):
        raise ValueError("DiffTOD smoke projection/sampling failed")
    report = {
        "schema_version": 1,
        "status": "smoke_non_scientific",
        "device": str(device),
        "h100": gpu,
        "model_parameters": model.parameter_count,
        "score_shape": list(scores.shape),
        "sample_shape": list(sampled.shape),
        "finite_loss": True,
    }
    atomic_json(output / "smoke.json", report)
    return report


def train(args):
    gpu = require_h100_slurm()
    assets = verify_assets(args.model_lock, args.base_checkpoint, args.tokenizer)
    config = DiffTODConfig.from_path(args.config)
    tokenizer = _tokenizer(args.tokenizer)
    training = token_blocks(args.train, tokenizer, config.model.length)
    validation = token_blocks(args.dev, tokenizer, config.model.length)
    model = SEDD(config.model)
    base = torch.load(args.base_checkpoint, map_location="cpu", weights_only=False)
    load_weights(model, base, strict=True)
    trainer = DiffTODTrainer(model, config, "cuda")
    if args.resume:
        trainer.resume(args.resume)
    started = time.monotonic()
    result = trainer.fit(
        training,
        validation,
        args.output,
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
        base_checkpoint=checkpoint_signature(args.base_checkpoint),
        asset_lock_sha256=file_hash(args.model_lock),
        asset_revisions={
            "sedd": assets["sedd"]["revision"],
            "tokenizer": assets["tokenizer"]["revision"],
        },
    )
    atomic_json(args.output / "training-summary.json", result)
    return result


def guidance(args):
    selection_sha256 = _test_gate(args.split, args.selection_lock)
    settings = json.loads(args.runtime_config.read_text(encoding="utf-8"))
    store = GuidanceStore(args.output)
    rows = _verified_scenarios(
        args.scenarios, args.dataset, args.split, selection_sha256
    )
    started = time.monotonic()
    generated = 0
    server_output = args.output / "_servers" / args.dataset / args.split
    with Servers(args.root, server_output, settings, kinds=("generation",)) as servers:
        backend = servers.backends()["generation"]
        for seed in args.seeds:
            for record in rows:
                target = store.path(args.dataset, args.split, record["id"], seed)
                if target.exists():
                    store.load(args.dataset, args.split, record["id"], seed)
                    generated += 1
                    continue
                artifact = generate_guidance(
                    backend,
                    record,
                    args.dataset,
                    args.split,
                    seed,
                    f"difftod/{args.split}/{args.dataset}/{seed}/{record['id']}",
                )
                store.write(artifact)
                generated += 1
        requests = list(backend.consumed_requests)
        compute = {
            "generation_requests": len(requests),
            "input_tokens": sum(request.input_tokens for request in requests),
            "output_tokens": sum(request.output_tokens for request in requests),
            "total_tokens": sum(request.total_tokens for request in requests),
            **servers.compute_metadata(),
        }
    report = {
        "schema_version": 1,
        "status": "complete",
        "dataset": args.dataset,
        "split": args.split,
        "selection_sha256": selection_sha256,
        "scenario_sha256": file_hash(args.scenarios),
        "seeds": args.seeds,
        "guidance_artifacts": generated,
        "wall_s": time.monotonic() - started,
        "compute": compute,
    }
    atomic_json(args.output / f"guidance-{args.dataset}-{args.split}.json", report)
    return report


def sample(args):
    gpu = require_h100_slurm()
    selection_sha256 = _test_gate(args.split, args.selection_lock)
    config = DiffTODConfig.from_path(args.config)
    verify_assets(args.model_lock, args.base_checkpoint, args.tokenizer)
    tokenizer = _tokenizer(args.tokenizer)
    checkpoint_sha256 = file_hash(args.checkpoint)
    if args.split == "test":
        selected = verify_selection(args.selection_lock)["payload"]
        if (
            selected["checkpoint_sha256"][args.dataset] != checkpoint_sha256
            or selected["plan_config_sha256"][args.dataset][args.budget] != config.sha256
        ):
            raise ValueError("DiffTOD test sampling differs from frozen selection")
    model = load_sampling_model(args.checkpoint, config, "cuda")
    guidance_store = GuidanceStore(args.guidance)
    plan_store = PlanStore(args.output)
    rows = _verified_scenarios(
        args.scenarios, args.dataset, args.split, selection_sha256
    )
    completed = 0
    started = time.monotonic()
    for seed in args.seeds:
        for record in rows:
            target = plan_store.path(
                args.dataset, args.split, record["id"], seed, args.budget
            )
            if target.exists():
                artifact = plan_store.load(
                    args.dataset, args.split, record["id"], seed, args.budget
                )
                if (
                    artifact.config_sha256 != config.sha256
                    or artifact.checkpoint_sha256 != checkpoint_sha256
                ):
                    raise ValueError("resumed DiffTOD plan has different provenance")
                completed += 1
                continue
            guidance = guidance_store.load(args.dataset, args.split, record["id"], seed)
            generate_plan(
                model,
                tokenizer,
                record,
                args.dataset,
                args.split,
                guidance,
                config,
                checkpoint_sha256,
                seed,
                args.budget,
                plan_store,
            )
            completed += 1
    report = {
        "schema_version": 1,
        "status": "complete",
        "dataset": args.dataset,
        "split": args.split,
        "budget": args.budget,
        "selection_sha256": selection_sha256,
        "scenario_sha256": file_hash(args.scenarios),
        "checkpoint_sha256": checkpoint_sha256,
        "config_sha256": config.sha256,
        "seeds": args.seeds,
        "plans": completed,
        "wall_s": time.monotonic() - started,
        "gpu": gpu,
    }
    atomic_json(
        args.output / f"sample-{args.dataset}-{args.split}-{args.budget}.json", report
    )
    return report


def main():
    parser = argparse.ArgumentParser(description="Isolated faithful DiffTOD baseline runtime")
    sub = parser.add_subparsers(dest="command", required=True)
    command = sub.add_parser("prepare")
    command.add_argument("--input", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--dataset", choices=("TopDial", "DuRecDial2.0"), required=True)
    command.add_argument("--split", choices=("train", "dev"), required=True)
    command = sub.add_parser("smoke")
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--require-h100", action="store_true")
    command = sub.add_parser("export-scenarios")
    command.add_argument("--root", type=Path, required=True)
    command.add_argument("--dataset", choices=("TopDial", "DuRecDial2.0"), required=True)
    command.add_argument("--split", choices=("dev", "test"), required=True)
    command.add_argument("--selection-lock", type=Path)
    command.add_argument("--output", type=Path, required=True)
    command = sub.add_parser("train")
    command.add_argument("--config", type=Path, required=True)
    command.add_argument("--train", type=Path, required=True)
    command.add_argument("--dev", type=Path, required=True)
    command.add_argument("--tokenizer", type=Path, required=True)
    command.add_argument("--base-checkpoint", type=Path, required=True)
    command.add_argument("--model-lock", type=Path, required=True)
    command.add_argument("--resume", type=Path)
    command.add_argument("--output", type=Path, required=True)
    command = sub.add_parser("sample")
    command.add_argument("--config", type=Path, required=True)
    command.add_argument("--checkpoint", type=Path, required=True)
    command.add_argument("--base-checkpoint", type=Path, required=True)
    command.add_argument("--model-lock", type=Path, required=True)
    command.add_argument("--tokenizer", type=Path, required=True)
    command.add_argument("--scenarios", type=Path, required=True)
    command.add_argument("--guidance", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--dataset", choices=("TopDial", "DuRecDial2.0"), required=True)
    command.add_argument("--split", choices=("dev", "test"), required=True)
    command.add_argument("--budget", choices=("natural", "matched"), required=True)
    command.add_argument("--selection-lock", type=Path)
    command.add_argument("--seeds", nargs="+", type=int, default=[17, 29, 43])
    command = sub.add_parser("guidance")
    command.add_argument("--root", type=Path, required=True)
    command.add_argument("--runtime-config", type=Path, required=True)
    command.add_argument("--scenarios", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--dataset", choices=("TopDial", "DuRecDial2.0"), required=True)
    command.add_argument("--split", choices=("dev", "test"), required=True)
    command.add_argument("--selection-lock", type=Path)
    command.add_argument("--seeds", nargs="+", type=int, default=[17, 29, 43])
    args = parser.parse_args()
    if args.command == "prepare":
        report = prepare_corpus(args.input, args.output, args.dataset, args.split)
        atomic_json(args.output.with_suffix(args.output.suffix + ".manifest.json"), report)
    elif args.command == "smoke":
        report = smoke(args.output, args.require_h100)
    elif args.command == "export-scenarios":
        report = export_scenarios(args)
    elif args.command == "train":
        report = train(args)
    elif args.command == "guidance":
        report = guidance(args)
    else:
        report = sample(args)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
