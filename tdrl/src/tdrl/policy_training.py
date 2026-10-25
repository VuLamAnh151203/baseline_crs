"""PPDPP supervised initialization followed by recommendation-task on-policy RL."""

import json
import math
import time
from contextlib import contextmanager

import numpy as np
import torch

from .backend import require_h100_slurm
from .contracts import Decoding, Message
from .evaluation import PPDPPPlanner, policy_input, run_dialogue, visible_context
from .judge import LocalJudge
from .policy_optimizer import policy_optimizer
from .ppdpp import PPDPPPolicy, reinforce_loss
from .provenance import atomic_json, digest, file_hash, stop_if_requested
from .state import DialogueState
from .training import atomic_torch


def policy_snapshot(root):
    pin = json.loads((root / "configs/policy-model.lock.json").read_text())["models"][0]
    if pin["model_id"] != "FacebookAI/roberta-large" or not pin["weight_sha256"]:
        raise ValueError("intrinsic policy model substitution forbidden")
    directory = root / "models/roberta-large" / pin["revision"]
    for filename, sha in pin["weight_sha256"].items():
        if file_hash(directory / filename) != sha:
            raise ValueError("intrinsic policy weight checksum mismatch")
    return pin, directory


def load_policy(root, checkpoint=None):
    require_h100_slurm()
    from transformers import RobertaModel, RobertaTokenizer

    pin, directory = policy_snapshot(root)
    tokenizer = RobertaTokenizer.from_pretrained(str(directory), local_files_only=True)
    backbone = RobertaModel.from_pretrained(
        str(directory), local_files_only=True, dtype=torch.bfloat16
    )
    policy = PPDPPPolicy(backbone, backbone.config.hidden_size).to("cuda", torch.bfloat16)
    if checkpoint:
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if (
            saved["schema_version"] != 1
            or saved["policy_model"] != pin
            or saved["actions"] != policy.actions
        ):
            raise ValueError("policy checkpoint architecture/provenance mismatch")
        policy.load_state_dict(saved["policy"])
    return policy, tokenizer, pin


def supervised_policy(root, scenarios, cfg, directory, provenance, validation_scenarios=None):
    torch.manual_seed(cfg["seed"])
    torch.cuda.manual_seed_all(cfg["seed"])
    policy, tokenizer, pin = load_policy(root)
    policy.train()
    policy.policy.gradient_checkpointing_enable()
    faithful = cfg.get("implementation_version") == 2
    if faithful and not validation_scenarios:
        raise ValueError("PPDPP SFT requires frozen development strategy validation")
    optimizer = (
        policy_optimizer(policy, cfg, "sft")
        if faithful
        else torch.optim.AdamW(policy.parameters(), lr=cfg["sft_lr"])
    )
    generator = torch.Generator().manual_seed(cfg["seed"])
    examples = []
    for scenario in scenarios:
        messages = [Message(**m) for m in scenario["messages"]]
        context = visible_context(scenario)
        for t, message in enumerate(messages):
            if message.role == "assistant" and t > 0 and messages[t - 1].role == "user":
                examples.append(
                    (
                        DialogueState(tuple(messages[:t])),
                        context,
                        policy.actions.index(scenario["assistant_actions"][t]),
                    )
                )
    if not examples:
        raise ValueError("no supervised strategy training examples")
    total_steps = math.ceil(len(examples) / cfg["sft_batch_size"]) * cfg["sft_epochs"]
    scheduler = (
        torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda step: (
                (step + 1) / cfg["sft_warmup_steps"]
                if step + 1 < cfg["sft_warmup_steps"]
                else max(
                    0.0, (total_steps - step - 1) / max(1, total_steps - cfg["sft_warmup_steps"])
                )
            ),
        )
        if faithful
        else None
    )
    checkpoint = directory / ("resume.pt" if faithful else "checkpoint.pt")
    best_precision = -1.0
    epoch, cursor, order, steps = 0, 0, None, 0
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if saved["schema_version"] != 1 or saved["provenance"] != provenance:
            raise ValueError("SFT resume provenance mismatch")
        policy.load_state_dict(saved["policy"])
        optimizer.load_state_dict(saved["optimizer"])
        epoch, cursor, order, steps = (
            saved["epoch"],
            saved["cursor"],
            saved["order"],
            saved["steps"],
        )
        generator.set_state(saved["shuffle_rng"])
        if faithful:
            scheduler.load_state_dict(saved["scheduler"])
            best_precision = saved["best_precision"]
        torch.set_rng_state(saved["cpu_rng"])
        torch.cuda.set_rng_state_all(saved["cuda_rng"])
    else:
        torch.manual_seed(cfg["seed"])
        torch.cuda.manual_seed_all(cfg["seed"])
    started = time.monotonic()
    log_path = directory / "loss.jsonl"
    while epoch < cfg["sft_epochs"]:
        if order is None:
            order = torch.randperm(len(examples), generator=generator).tolist()
        for offset in range(cursor, len(order), cfg["sft_batch_size"]):
            subset = [examples[i] for i in order[offset : offset + cfg["sft_batch_size"]]]
            ids = [policy_input(tokenizer, state, context)[0][0] for state, context, _ in subset]
            mask = [torch.ones_like(i) for i in ids]
            padded = torch.nn.utils.rnn.pad_sequence(
                ids, batch_first=True, padding_value=tokenizer.pad_token_id
            )
            attention = torch.nn.utils.rnn.pad_sequence(mask, batch_first=True, padding_value=0)
            labels = torch.tensor([label for _, _, label in subset], device="cuda")
            optimizer.zero_grad(set_to_none=True)
            loss = policy.supervised_loss(padded, attention, labels)
            if not torch.isfinite(loss):
                raise ValueError("nonfinite PPDPP supervised loss")
            loss.backward()
            if any(
                p.grad is not None and not torch.isfinite(p.grad).all() for p in policy.parameters()
            ):
                raise ValueError("nonfinite PPDPP supervised gradients")
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()
            if scheduler:
                scheduler.step()
            steps += 1
            cursor = min(len(order), offset + cfg["sft_batch_size"])
            if steps % 100 == 0:
                with log_path.open("a") as log:
                    log.write(
                        json.dumps(
                            {
                                "epoch": epoch,
                                "cursor": cursor,
                                "step": steps,
                                "loss": float(loss.detach()),
                            }
                        )
                        + "\n"
                    )
            if steps % 500 == 0 or cursor == len(order):
                atomic_torch(
                    checkpoint,
                    {
                        "schema_version": 1,
                        "provenance": provenance,
                        "policy_model": pin,
                        "actions": policy.actions,
                        "policy": policy.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "epoch": epoch,
                        "cursor": cursor,
                        "steps": steps,
                        "order": order,
                        "shuffle_rng": generator.get_state(),
                        "cpu_rng": torch.get_rng_state(),
                        "cuda_rng": torch.cuda.get_rng_state_all(),
                        "scheduler": scheduler.state_dict() if scheduler else None,
                        "best_precision": best_precision,
                    },
                )
                atomic_json(
                    directory / "progress.json",
                    {
                        "stage": "PPDPP-SFT",
                        "epoch": epoch,
                        "epochs": cfg["sft_epochs"],
                        "examples": len(examples),
                        "cursor": cursor,
                        "elapsed_s": time.monotonic() - started,
                        "status": "partial",
                    },
                )
        if faithful:
            validation = validate_strategy_policy(
                policy, tokenizer, validation_scenarios, cfg["sft_batch_size"]
            )
            atomic_json(directory / f"validation-{epoch:02d}.json", validation)
            if validation["macro_precision"] > best_precision:
                best_precision = validation["macro_precision"]
                saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
                saved["best_precision"] = best_precision
                saved["selected_epoch"] = epoch
                atomic_torch(directory / "checkpoint.pt", saved)
            saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
            saved["best_precision"] = best_precision
            atomic_torch(checkpoint, saved)
        stop_if_requested(directory)
        epoch, cursor, order = epoch + 1, 0, None
    atomic_json(
        directory / "fidelity.json",
        {
            "schema_version": 1,
            "mechanism": "pooled_roberta/dropout_0.5/classifier/cross_entropy_then_on_policy_REINFORCE",
            "upstream_revision": "f44dccdd43f335e5e5f372ab556cfd9f182f9c55",
            "actions": policy.actions,
            "supervision": "declared weak recommendation labels, not gold",
            "target_conditioning": "96-token visible header + recent complete turns within 512",
            "evaluation_mode": "dropout_disabled_argmax",
            "checkpoint_sha256": file_hash(directory / "checkpoint.pt"),
            "selection": "development macro precision" if faithful else "final epoch",
            "status": "sft_complete_rl_required",
        },
    )
    return policy.eval(), tokenizer, pin


@torch.no_grad()
def validate_strategy_policy(policy, tokenizer, scenarios, batch_size):
    policy.eval()
    examples = []
    for scenario in scenarios:
        history = [Message(**m) for m in scenario["messages"]]
        for t, message in enumerate(history):
            if message.role == "assistant" and t and history[t - 1].role == "user":
                ids = policy_input(
                    tokenizer, DialogueState(tuple(history[:t])), visible_context(scenario)
                )[0][0]
                examples.append((ids, policy.actions.index(scenario["assistant_actions"][t])))
    confusion = torch.zeros(len(policy.actions), len(policy.actions), dtype=torch.long)
    for offset in range(0, len(examples), batch_size):
        rows = examples[offset : offset + batch_size]
        ids = torch.nn.utils.rnn.pad_sequence(
            [r[0] for r in rows], batch_first=True, padding_value=tokenizer.pad_token_id
        )
        predicted = policy.logits(ids, ids.ne(tokenizer.pad_token_id)).argmax(-1).cpu()
        for (_, gold), guess in zip(rows, predicted.tolist(), strict=True):
            confusion[gold, guess] += 1
    if not examples:
        raise ValueError("empty development strategy validation")
    precision = confusion.diag() / confusion.sum(0).clamp_min(1)
    policy.train()
    return {
        "macro_precision": float(precision.mean()),
        "examples": len(examples),
        "confusion": confusion.tolist(),
        "actions": policy.actions,
        "scenario_ids_sha256": digest([s["id"] for s in scenarios]),
        "label_source": "weak_rule_v1; not human strategy quality",
    }


def replay_policy_actions(actor, stored, scenario):
    """Rebuild differentiable sampled actions with the saved pre-episode RNG.

    Generates no dialogue and imports no old checkpoint/results. Recorded strategies
    must match the newly sampled actions exactly before gradients may be reused.
    """
    from .evaluation import initial_state

    prefix = len(initial_state(scenario).history)
    history = [Message(**m) for m in stored["history"]]
    for turn, row in enumerate(stored["turns_trace"]):
        state = DialogueState(tuple(history[: prefix + 2 * turn]))
        ids, mask = policy_input(actor.tokenizer, state, actor.context)
        action, log_prob = actor.policy.select(ids, mask, actor.sample_generator, evaluate=False)
        expected = row["plan"]["compute"].get("strategy")
        if actor.policy.actions[int(action.item())] != expected:
            raise ValueError("on-policy action replay mismatch; gradients cannot be reconstructed")
        actor.log_probabilities.append(log_prob.squeeze())


@contextmanager
def deterministic_backward(enabled):
    """Scope strict CUDA determinism to backward; restore both global flags."""
    previous = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        if enabled:
            torch.use_deterministic_algorithms(True, warn_only=False)
        yield
    finally:
        torch.use_deterministic_algorithms(previous, warn_only=warn_only)


def policy_backward(policy, loss):
    amendment = getattr(policy, "_recovery_execution", {})
    with deterministic_backward(amendment.get("deterministic_backward", False)):
        loss.backward()


def reinforce_policy(
    policy,
    tokenizer,
    pin,
    scenarios,
    generation,
    embedding,
    cfg,
    directory,
    provenance,
    recovery_check=None,
):
    checkpoint = directory / "checkpoint.pt"
    faithful = cfg.get("implementation_version") == 2
    optimizer = (
        policy_optimizer(policy, cfg, "rl")
        if faithful
        else torch.optim.AdamW(policy.parameters(), lr=cfg["rl_lr"])
    )
    next_episode = 0
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if saved["schema_version"] != 1 or saved["provenance"] != provenance:
            raise ValueError("PPDPP RL checkpoint provenance mismatch")
        policy.load_state_dict(saved["policy"])
        optimizer.load_state_dict(saved["optimizer"])
        torch.set_rng_state(saved["cpu_rng"])
        torch.cuda.set_rng_state_all(saved["cuda_rng"])
        next_episode = saved["next_episode"]
    else:
        torch.manual_seed(cfg["seed"])
        torch.cuda.manual_seed_all(cfg["seed"])
    from .autonomy import explicit_refusal
    from .evaluation import initial_state

    ordered = sorted(
        scenarios if faithful else [s for s in scenarios if not explicit_refusal(initial_state(s))],
        key=lambda s: s["id"],
    )
    # Precompute a seeded native-style random sample schedule; resume needs no mutable RNG.
    if not ordered:
        raise ValueError("no non-refusal on-policy training scenarios")
    schedule = np.random.RandomState(cfg["seed"]).choice(len(ordered), cfg["rl_episodes"])
    if faithful:
        atomic_json(
            directory / "sampling.json",
            {
                "mechanism": "seeded uniform sampling with replacement; all training scenarios",
                "seed": cfg["seed"],
                "training_ids_sha256": digest([s["id"] for s in ordered]),
                "episode_scenario_ids": [ordered[int(i)]["id"] for i in schedule],
            },
        )
    policy.policy.gradient_checkpointing_enable()
    for episode in range(next_episode, cfg["rl_episodes"]):
        stop_if_requested(directory)
        scenario = ordered[int(schedule[episode])] if faithful else ordered[episode % len(ordered)]
        policy.train()
        output = directory / "episodes" / f"{episode:06d}"
        output.mkdir(parents=True, exist_ok=True)
        rng_path = output / "pre-episode-rng.pt"
        if rng_path.exists():
            rng = torch.load(rng_path, map_location="cpu", weights_only=False)
            if rng["provenance"] != provenance or rng["episode"] != episode:
                raise ValueError("on-policy pre-episode RNG provenance mismatch")
            torch.set_rng_state(rng["cpu_rng"])
            torch.cuda.set_rng_state_all(rng["cuda_rng"])
        else:
            atomic_torch(
                rng_path,
                {
                    "provenance": provenance,
                    "episode": episode,
                    "cpu_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state_all(),
                },
            )
        if recovery_check is not None:
            recovery_check(policy, tokenizer, scenario, cfg, output, rng_path)
            recovery_check = None
        actor = PPDPPPlanner(
            policy,
            tokenizer,
            generation,
            embedding,
            visible_context(scenario),
            Decoding(**cfg["assistant_decoding"]),
            training=True,
            policy_seed=cfg["seed"] + episode,
        )
        if (output / "dialogue.json").exists():
            replay_policy_actions(
                actor, json.loads((output / "dialogue.json").read_text()), scenario
            )
        result = run_dialogue(
            scenario,
            actor,
            generation,
            embedding,
            "profile" if episode % 2 == 0 else "resistant",
            cfg["seed"] + episode,
            output,
            {"phase": "PPDPP-RL", "method": "PPDPP", "provenance": provenance},
            cfg["turn_budget"],
            Decoding(**cfg["simulator_decoding"]),
            judge=LocalJudge(generation, cfg["judge"]) if faithful else None,
            training_critic=faithful,
        )
        rewards = [
            t["critic"]["reward"]
            if faithful
            else cfg["accepted_reward"]
            if t["acceptance"]["accepted"]
            else cfg["step_reward"]
            for t in result["turns_trace"]
        ]
        if len(actor.log_probabilities) != len(rewards):
            raise ValueError("PPDPP action/reward trajectory mismatch")
        loss = reinforce_loss(actor.log_probabilities, rewards, cfg["gamma"])
        optimizer.zero_grad(set_to_none=True)
        policy_backward(policy, loss)
        if not torch.isfinite(loss) or any(
            p.grad is not None and not torch.isfinite(p.grad).all() for p in policy.parameters()
        ):
            raise ValueError("nonfinite PPDPP REINFORCE update")
        if not faithful:
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()
        atomic_torch(
            checkpoint,
            {
                "schema_version": 1,
                "provenance": provenance,
                "policy_model": pin,
                "actions": policy.actions,
                "policy": policy.state_dict(),
                "optimizer": optimizer.state_dict(),
                "next_episode": episode + 1,
                **(
                    {"execution_amendment": policy._recovery_execution}
                    if hasattr(policy, "_recovery_execution")
                    else {}
                ),
                "cpu_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all(),
            },
        )
        atomic_json(
            directory / "progress.json",
            {
                "episode": episode + 1,
                "episodes": cfg["rl_episodes"],
                "accepted": result["accepted"],
                "loss": float(loss.detach()),
                "status": "partial",
            },
        )
    policy.eval()
    return policy
