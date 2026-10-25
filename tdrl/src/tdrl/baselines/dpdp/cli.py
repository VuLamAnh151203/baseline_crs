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
from ...contracts import Decoding, Message
from ...evaluation import initial_state, run_dialogue
from ...judge import LocalJudge
from ...planner import INSTRUCTION
from ...provenance import atomic_json, digest, file_hash
from ...runtime import Servers
from ...state import DialogueState
from .assets import verify_assets
from .checkpoint import load_checkpoint, save_checkpoint
from .config import DPDPConfig
from .contracts import DPDPCheckpointMetadata, RewardSample
from .dataset import DPDPCollator, DPDPDataset, prepare_transitions
from .mcts import OpenLoopMCTS
from .model import DPDPPolicy, load_pretrained_policy
from .planner import DPDPPlanner
from .rewards import annotate_rewards
from .rollout import QwenRolloutProvider
from .runtime import load_tokenizer
from .serialization import encode_policy_input
from .training import DPDPTrainer, episode_returns, selfplay_update

SOURCE_REVISION = "4d1e22e35db452d51f0f39388456d310ec1c971b"
# A self-play case is expensive (MCTS plus local-Qwen rollout calls).  Five cases
# bounds lost optimizer work while avoiding a multi-GB checkpoint after every case.
SELFPLAY_CHECKPOINT_EVERY = 5


class _TinyEncoder(nn.Module):
    def __init__(self, vocab_size=64, hidden_size=32):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.projection = nn.Linear(hidden_size, hidden_size)

    def forward(self, input_ids, attention_mask=None, return_dict=True):
        del attention_mask, return_dict
        hidden = self.embedding(input_ids)
        return SimpleNamespace(
            last_hidden_state=hidden,
            pooler_output=torch.tanh(self.projection(hidden[:, 0])),
        )


class _TinyTokenizer:
    cls_token_id = 0
    sep_token_id = 1
    pad_token_id = 2

    def encode(self, text, add_special_tokens=False):
        values = [3 + (ord(character) % 61) for character in text][:256]
        return [self.cls_token_id, *values, self.sep_token_id] if add_special_tokens else values


class _TinyRollout:
    def expand(self, state, action_index, action_name, path, simulation, depth):
        del action_index, path
        history = state.history if isinstance(state, DialogueState) else tuple(state)
        assistant = f"{action_name}-{simulation}-{depth}"
        user = "That sounds interesting."
        next_state = (*history, Message("assistant", assistant), Message("user", user))
        return RewardSample(
            assistant_response=assistant,
            user_response=user,
            reward=0.1 + depth * 0.01,
            label="interested",
            critic_labels=("interested",),
            deterministic_accepted=False,
            acceptance_disagreement=False,
            terminal=depth >= 1,
            terminal_reasons=("tiny_depth_limit",) if depth >= 1 else (),
            refused=False,
            next_state=next_state,
            request_ids=(f"tiny/{simulation}/{depth}",),
        )


def _seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def smoke(output, require_h100=False):
    output.mkdir(parents=True, exist_ok=True)
    gpu = require_h100_slurm() if require_h100 else None
    device = torch.device("cuda" if require_h100 else "cpu")
    _seed_all(17)
    config = DPDPConfig.smoke()
    model = DPDPPolicy(_TinyEncoder(), 32, config.actions, config.offline_dropout).to(device)
    ids = torch.tensor([[0, 5, 6, 1], [0, 7, 8, 1]], device=device)
    mask = torch.ones_like(ids)
    actions = torch.tensor([1, 4], device=device)
    returns = torch.tensor([0.2, 1.0], device=device)
    output_values = model(ids, mask, actions, returns, config.offline_critic_weight)
    output_values["total_loss"].backward()
    if not torch.isfinite(output_values["total_loss"]):
        raise FloatingPointError("DPDP smoke loss is non-finite")
    state = DialogueState((Message("user", "I enjoy mystery films."),))
    search = OpenLoopMCTS(
        config.actions,
        lambda _: ([1 / len(config.actions)] * len(config.actions), [0.0] * len(config.actions)),
        _TinyRollout(),
        config.mcts_simulations,
        config.mcts_cpuct,
        config.mcts_q0,
        config.mcts_max_depth,
        config.mcts_max_realizations,
    )
    mcts = search.search(state)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    update = selfplay_update(
        model,
        optimizer,
        {"input_ids": ids, "attention_mask": mask, "actions": actions, "returns": returns},
        1.0,
        1.0,
    )
    report = {
        "schema_version": 1,
        "status": "smoke_non_scientific",
        "device": str(device),
        "h100": gpu,
        "finite_loss": True,
        "loss": float(output_values["total_loss"].detach()),
        "mcts_action": mcts.action_name,
        "mcts_root_visits": list(mcts.root_visits),
        "mcts_nodes": mcts.expanded_nodes,
        "selfplay_update": update,
        "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
    }
    atomic_json(output / "smoke.json", report)
    return report


def _metadata(phase, config, data_sha256, model_lock, parent_checkpoint_sha256=None):
    return DPDPCheckpointMetadata(
        phase=phase,
        config_sha256=config.fitting_sha256,
        data_sha256=data_sha256,
        action_taxonomy_sha256=digest(config.actions),
        model_lock_sha256=file_hash(model_lock),
        source_revision=SOURCE_REVISION,
        parent_checkpoint_sha256=parent_checkpoint_sha256,
    )


def train(args):
    gpu = require_h100_slurm()
    assets = verify_assets(args.model_lock, args.encoder, args.tokenizer)
    config = DPDPConfig.from_path(args.config)
    _seed_all(config.offline_seed)
    tokenizer = load_tokenizer(args.tokenizer)
    training = DPDPDataset(args.train, tokenizer, config)
    validation = DPDPDataset(args.dev, tokenizer, config)
    model = load_pretrained_policy(args.encoder, config.actions, config.offline_dropout, "cuda")
    data_sha256 = digest({"train": file_hash(args.train), "dev": file_hash(args.dev)})
    metadata = _metadata("offline", config, data_sha256, args.model_lock)
    trainer = DPDPTrainer(model, config, "cuda", tokenizer.pad_token_id, metadata)
    started = time.monotonic()
    result = trainer.fit(
        training,
        validation,
        args.output,
        resume=args.resume,
        stop_requested=lambda: (args.output / "stop.requested").exists(),
    )
    result.update(
        gpu=gpu,
        wall_s=time.monotonic() - started,
        config_sha256=config.sha256,
        training_config_sha256=config.training_sha256,
        data_sha256=data_sha256,
        model_revision=assets["revision"],
        model_lock_sha256=file_hash(args.model_lock),
    )
    atomic_json(args.output / "training-summary.json", result)
    return result


def annotate(args):
    require_h100_slurm()
    config = DPDPConfig.from_path(args.config)
    runtime = json.loads(args.runtime_config.read_text(encoding="utf-8"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with Servers(
        args.root, args.output.parent / "servers", runtime, kinds=("generation",)
    ) as servers:
        backend = servers.backends()["generation"]
        result = annotate_rewards(
            args.input,
            args.output,
            backend,
            config.critic_samples,
            config.gamma,
            config.offline_seed,
            args.output.parent / f"{args.output.name}.critic-journal",
        )
        result["server_compute"] = servers.compute_metadata()
    return result


def _policy_function(model, tokenizer, config, scenario):
    def evaluate(state):
        history = state.history if isinstance(state, DialogueState) else tuple(state)
        ids, mask = encode_policy_input(
            tokenizer,
            scenario,
            history,
            config.max_sequence_length,
            config.header_tokens,
        )
        device = next(model.parameters()).device
        with torch.inference_mode():
            output = model(ids[None].to(device), mask[None].to(device))
        return (
            output["action_probs"][0].float().cpu().tolist(),
            output["q_values"][0].float().cpu().tolist(),
        )

    return evaluate


def _selfplay_rows(model, tokenizer, config, scenario, backend, epoch, case_index):
    state = initial_state(scenario)
    rows, rewards, traces = [], [], []
    for turn in range(10):
        ids, mask = encode_policy_input(
            tokenizer,
            scenario,
            state.history,
            config.max_sequence_length,
            config.header_tokens,
        )
        request_base = f"dpdp/selfplay/{epoch}/{case_index}/{scenario['id']}/turn/{turn}"
        rollout = QwenRolloutProvider(
            backend,
            scenario,
            "profile",
            config.selfplay_seed + epoch,
            request_base,
            config.critic_samples,
        )
        search = OpenLoopMCTS(
            config.actions,
            _policy_function(model, tokenizer, config, scenario),
            rollout,
            config.mcts_simulations,
            config.mcts_cpuct,
            config.mcts_q0,
            min(config.mcts_max_depth, 10 - turn),
            config.mcts_max_realizations,
        )
        result = search.search(state)
        rows.append(
            {
                "transition_id": request_base,
                "input_ids": ids,
                "attention_mask": mask,
                "action": result.action_index,
                "return": None,
            }
        )
        rewards.append(result.reward)
        rollout_samples = [
            sample
            for node in result.tree.values()
            for samples in node["realizations"].values()
            for sample in samples
        ]
        traces.append(
            {
                "turn": turn,
                "transition_id": request_base,
                "selected_action_index": result.action_index,
                "selected_action": result.action_name,
                "assistant_response": result.response,
                "user_response": result.user_response,
                "reward": result.reward,
                "reward_label": result.reward_label,
                "critic_labels": list(result.critic_labels),
                "deterministic_accepted": result.deterministic_accepted,
                "acceptance_disagreement": result.acceptance_disagreement,
                "rollout_samples": len(rollout_samples),
                "rollout_acceptance_disagreements": sum(
                    int(sample["acceptance_disagreement"]) for sample in rollout_samples
                ),
                "terminal": result.terminal,
                "terminal_reasons": list(result.terminal_reasons),
                "refused": result.refused,
                "root_visits": list(result.root_visits),
                "root_q_values": list(result.root_q_values),
                "root_priors": list(result.root_priors),
                "simulations": result.simulations,
                "expanded_nodes": result.expanded_nodes,
                "max_depth_reached": result.max_depth_reached,
                "request_ids": list(result.request_ids),
                "mcts_tree_sha256": digest(result.tree),
                "mcts_tree": result.tree,
            }
        )
        state = DialogueState(result.next_state)
        if result.terminal:
            break
    returns = episode_returns(rewards, config.gamma)
    for row, trace, value in zip(rows, traces, returns, strict=True):
        row["return"] = value
        trace["discounted_return"] = value
    return rows, rewards, traces


def selfplay(args):
    gpu = require_h100_slurm()
    assets = verify_assets(args.model_lock, args.encoder, args.tokenizer)
    config = DPDPConfig.from_path(args.config)
    _seed_all(config.selfplay_seed)
    tokenizer = load_tokenizer(args.tokenizer)
    model = load_pretrained_policy(args.encoder, config.actions, config.selfplay_dropout, "cuda")
    offline_payload = torch.load(args.offline_checkpoint, map_location="cpu", weights_only=False)
    offline_metadata = DPDPCheckpointMetadata(**offline_payload.get("metadata", {}))
    if (
        offline_metadata.phase != "offline"
        or offline_metadata.config_sha256 != config.fitting_sha256
        or offline_metadata.action_taxonomy_sha256 != digest(config.actions)
        or offline_metadata.model_lock_sha256 != file_hash(args.model_lock)
        or offline_metadata.source_revision != SOURCE_REVISION
        or offline_metadata.parent_checkpoint_sha256 is not None
    ):
        raise ValueError("DPDP self-play requires a matching offline checkpoint")
    load_checkpoint(args.offline_checkpoint, model, offline_metadata, map_location="cuda")
    scenarios = [
        json.loads(line)
        for line in args.scenarios.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not scenarios or len({row["id"] for row in scenarios}) != len(scenarios):
        raise ValueError("empty or duplicate DPDP self-play scenarios")
    data_sha256 = file_hash(args.scenarios)
    metadata = _metadata(
        "selfplay",
        config,
        data_sha256,
        args.model_lock,
        file_hash(args.offline_checkpoint),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.selfplay_lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    loader_generator = torch.Generator().manual_seed(config.selfplay_seed)
    start_epoch = start_case = step = 0
    if args.resume:
        payload = load_checkpoint(
            args.resume,
            model,
            metadata,
            optimizer,
            scheduler,
            loader_generator,
            "cuda",
        )
        step, start_epoch, start_case = (
            payload["step"],
            payload["epoch"],
            payload["batch_in_epoch"],
        )
    args.output.mkdir(parents=True, exist_ok=True)
    runtime = json.loads(args.runtime_config.read_text(encoding="utf-8"))
    collator = DPDPCollator(tokenizer.pad_token_id)
    metrics = args.output / "metrics.jsonl"
    trace_path = args.output / "selfplay-trace.jsonl"
    if args.resume and metrics.exists():
        retained = [
            line
            for line in metrics.read_text(encoding="utf-8").splitlines()
            if line.strip() and json.loads(line)["step"] <= step
        ]
        metrics.write_text("".join(line + "\n" for line in retained), encoding="utf-8")
    if args.resume and trace_path.exists():
        retained = [
            line
            for line in trace_path.read_text(encoding="utf-8").splitlines()
            if line.strip() and json.loads(line)["step"] <= step
        ]
        trace_path.write_text("".join(line + "\n" for line in retained), encoding="utf-8")
    started = time.monotonic()
    with Servers(args.root, args.output / "servers", runtime, kinds=("generation",)) as servers:
        backend = servers.backends()["generation"]
        for epoch in range(start_epoch, config.selfplay_epochs):
            order = sorted(
                scenarios,
                key=lambda row: digest([config.selfplay_seed + epoch, row["id"]]),
            )
            count = min(config.selfplay_cases_per_epoch, len(order))
            offset = start_case if epoch == start_epoch else 0
            for case_index in range(offset, count):
                episode, rewards, traces = _selfplay_rows(
                    model, tokenizer, config, order[case_index], backend, epoch, case_index
                )
                batch = collator(episode)
                batch = {
                    key: value.cuda(non_blocking=True) if torch.is_tensor(value) else value
                    for key, value in batch.items()
                }
                values = selfplay_update(
                    model,
                    optimizer,
                    batch,
                    config.selfplay_critic_weight,
                    config.gradient_clip,
                )
                scheduler.step()
                step += 1
                disagreement_count = sum(int(turn["acceptance_disagreement"]) for turn in traces)
                rollout_disagreement_count = sum(
                    turn["rollout_acceptance_disagreements"] for turn in traces
                )
                with trace_path.open("a", encoding="utf-8") as handle:
                    handle.write(
                        json.dumps(
                            {
                                "schema_version": 1,
                                "step": step,
                                "epoch": epoch,
                                "case": case_index,
                                "scenario_id": order[case_index]["id"],
                                "turns": traces,
                                "selected_acceptance_disagreements": disagreement_count,
                                "rollout_acceptance_disagreements": rollout_disagreement_count,
                            },
                            ensure_ascii=False,
                            sort_keys=True,
                        )
                        + "\n"
                    )
                    handle.flush()
                next_epoch = epoch + 1 if case_index + 1 == count else epoch
                next_case = 0 if next_epoch != epoch else case_index + 1
                with metrics.open("a", encoding="utf-8") as handle:
                    handle.write(
                        json.dumps(
                            {
                                "step": step,
                                "epoch": epoch,
                                "case": case_index,
                                "scenario_id": order[case_index]["id"],
                                "turns": len(episode),
                                "rewards": rewards,
                                "selected_acceptance_disagreements": disagreement_count,
                                "rollout_acceptance_disagreements": rollout_disagreement_count,
                                **values,
                            },
                            sort_keys=True,
                        )
                        + "\n"
                    )
                    handle.flush()
                stopping = (args.output / "stop.requested").exists()
                if (
                    (case_index + 1) % SELFPLAY_CHECKPOINT_EVERY == 0
                    or next_epoch != epoch
                    or stopping
                ):
                    save_checkpoint(
                        args.output / "checkpoint-last.pt",
                        model,
                        optimizer,
                        scheduler,
                        step,
                        next_epoch,
                        next_case,
                        metadata,
                        loader_generator,
                    )
                if stopping:
                    return {"status": "partial", "step": step}
        save_checkpoint(
            args.output / "checkpoint-final.pt",
            model,
            optimizer,
            scheduler,
            step,
            config.selfplay_epochs,
            0,
            metadata,
            loader_generator,
        )
        compute = servers.compute_metadata()
    trace_rows = [
        json.loads(line)
        for line in trace_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    result = {
        "schema_version": 1,
        "status": "complete",
        "step": step,
        "wall_s": time.monotonic() - started,
        "gpu": gpu,
        "server_compute": compute,
        "request_concurrency": runtime["generation"].get("request_concurrency", 1),
        "checkpoint_every_cases": SELFPLAY_CHECKPOINT_EVERY,
        "selfplay_trace": str(trace_path),
        "selected_acceptance_disagreements": sum(
            row["selected_acceptance_disagreements"] for row in trace_rows
        ),
        "rollout_acceptance_disagreements": sum(
            row["rollout_acceptance_disagreements"] for row in trace_rows
        ),
        "config_sha256": config.sha256,
        "data_sha256": data_sha256,
        "model_revision": assets["revision"],
    }
    atomic_json(args.output / "selfplay-summary.json", result)
    return result


def smoke_model(args):
    gpu = require_h100_slurm()
    assets = verify_assets(args.model_lock, args.encoder, args.tokenizer)
    config = DPDPConfig.from_path(args.config)
    tokenizer = load_tokenizer(args.tokenizer)
    model = load_pretrained_policy(args.encoder, config.actions, config.offline_dropout, "cuda")
    scenario = {"profile": {"likes": "mystery"}, "target": ["Movie recommendation", "Moon"]}
    ids, mask = encode_policy_input(
        tokenizer,
        scenario,
        [Message("user", "I like mysteries.")],
        config.max_sequence_length,
        config.header_tokens,
    )
    ids, mask = ids[None].cuda(), mask[None].cuda()
    actions, returns = torch.tensor([4], device="cuda"), torch.tensor([1.0], device="cuda")
    torch.cuda.reset_peak_memory_stats()
    model.train()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        values = model(ids, mask, actions, returns, config.offline_critic_weight)
    values["total_loss"].backward()
    report = {
        "schema_version": 1,
        "status": "smoke_non_scientific",
        "gpu": gpu,
        "precision": "bfloat16",
        "finite_loss": bool(torch.isfinite(values["total_loss"])),
        "loss": float(values["total_loss"].detach()),
        "peak_memory_bytes": torch.cuda.max_memory_allocated(),
        "model_revision": assets["revision"],
        "config_sha256": config.sha256,
    }
    if not report["finite_loss"]:
        raise FloatingPointError("full DPDP H100 smoke loss is non-finite")
    atomic_json(args.output / "smoke-model.json", report)
    return report


def smoke_shared(args):
    gpu = require_h100_slurm()
    assets = verify_assets(args.model_lock, args.encoder, args.tokenizer)
    config = replace(
        DPDPConfig.from_path(args.config),
        gate_threshold=1.0,
        mcts_simulations=2,
        mcts_max_depth=1,
        critic_samples=2,
    )
    tokenizer = load_tokenizer(args.tokenizer)
    model = load_pretrained_policy(args.encoder, config.actions, config.selfplay_dropout, "cuda")
    model.to(dtype=torch.bfloat16).eval()
    scenario = {
        "id": "dpdp-shared-h100-smoke",
        "profile": {"preference": "mystery films"},
        "personality": {},
        "knowledge": [["Moon", "type", "mystery film"]],
        "target": ["Movie recommendation", "Moon"],
        "messages": [{"role": "user", "content": "I enjoy mystery films."}],
    }
    runtime = json.loads(args.runtime_config.read_text(encoding="utf-8"))
    comparison = json.loads(args.comparison_config.read_text(encoding="utf-8"))
    if comparison.get("schema_version") != 2 or comparison.get("seeds") != [17]:
        raise ValueError("DPDP shared smoke requires the single-seed shared comparison config")
    with Servers(args.root, args.output / "servers", runtime) as servers:
        backends = servers.backends()
        state = initial_state(scenario)
        embedding = backends["embedding"].embed(
            [state.serialize()], INSTRUCTION, "dpdp-smoke/live", "dpdp-smoke/live"
        )
        if embedding.vectors.shape != (1, 4096):
            raise ValueError("DPDP shared smoke did not receive the locked 4096-D embedding")
        planner = DPDPPlanner(
            backends["generation"],
            model,
            tokenizer,
            config,
            scenario,
            "untrained-smoke-only",
            Decoding(**comparison["assistant_decoding"]),
            "profile",
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
                "method": "DPDP",
                "selection_sha256": "smoke",
                "budget": "smoke",
                "target_nodes": [],
                "request_namespace": config.sha256,
            },
            turn_budget=1,
            simulator_decoding=Decoding(**comparison["simulator_decoding"]),
            judge=LocalJudge(backends["generation"], comparison["judge"]),
        )
        compute = servers.compute_metadata()
    selected_system = result["turns_trace"][0]["plan"]["compute"]["selected_system"]
    if selected_system != "system2":
        raise RuntimeError("DPDP shared smoke did not exercise System 2 MCTS")
    report = {
        "schema_version": 1,
        "status": "smoke_non_scientific",
        "gpu": gpu,
        "precision": "bfloat16",
        "model_revision": assets["revision"],
        "live_embedding_shape": list(embedding.vectors.shape),
        "dialogue_status": result["status"],
        "judge_protocol_sha256": result["judge_protocol_sha256"],
        "outcome_label": result["outcome_label"],
        "selected_system": selected_system,
        "mcts_simulations": result["turns_trace"][0]["plan"]["compute"]["mcts_simulations"],
        "request_concurrency": runtime["generation"].get("request_concurrency", 1),
        "server_compute": compute,
        "config_sha256": config.sha256,
    }
    atomic_json(args.output / "smoke-shared.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description="DPDP shared-setting baseline")
    sub = parser.add_subparsers(dest="command", required=True)
    command = sub.add_parser("prepare")
    command.add_argument("--input", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--dataset", choices=("TopDial", "DuRecDial2.0"), required=True)
    command.add_argument("--split", choices=("train", "dev"), required=True)
    command = sub.add_parser("smoke")
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--require-h100", action="store_true")
    for name in ("train", "smoke-model", "smoke-shared"):
        command = sub.add_parser(name)
        command.add_argument("--config", type=Path, required=True)
        command.add_argument("--encoder", type=Path, required=True)
        command.add_argument("--tokenizer", type=Path, required=True)
        command.add_argument("--model-lock", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
        if name == "smoke-shared":
            command.add_argument("--root", type=Path, required=True)
            command.add_argument("--runtime-config", type=Path, required=True)
            command.add_argument("--comparison-config", type=Path, required=True)
        if name == "train":
            command.add_argument("--train", type=Path, required=True)
            command.add_argument("--dev", type=Path, required=True)
            command.add_argument("--resume", type=Path)
    command = sub.add_parser("annotate-rewards")
    command.add_argument("--root", type=Path, required=True)
    command.add_argument("--runtime-config", type=Path, required=True)
    command.add_argument("--config", type=Path, required=True)
    command.add_argument("--input", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command = sub.add_parser("selfplay")
    command.add_argument("--root", type=Path, required=True)
    command.add_argument("--runtime-config", type=Path, required=True)
    command.add_argument("--config", type=Path, required=True)
    command.add_argument("--scenarios", type=Path, required=True)
    command.add_argument("--offline-checkpoint", type=Path, required=True)
    command.add_argument("--encoder", type=Path, required=True)
    command.add_argument("--tokenizer", type=Path, required=True)
    command.add_argument("--model-lock", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--resume", type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare_transitions(args.input, args.output, args.dataset, args.split)
    elif args.command == "smoke":
        result = smoke(args.output, args.require_h100)
    elif args.command == "smoke-model":
        args.output.mkdir(parents=True, exist_ok=True)
        result = smoke_model(args)
    elif args.command == "smoke-shared":
        args.output.mkdir(parents=True, exist_ok=True)
        result = smoke_shared(args)
    elif args.command == "train":
        result = train(args)
    elif args.command == "annotate-rewards":
        result = annotate(args)
    else:
        result = selfplay(args)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
