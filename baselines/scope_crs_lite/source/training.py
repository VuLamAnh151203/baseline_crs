"""Deterministic train/dev reward fitting with minimum-dev-loss selection."""

import json
import math
from pathlib import Path

import torch
from torch.nn import functional
from torch.utils.data import DataLoader

from ...provenance import atomic_json
from .reward import (
    GoalConditionedReward,
    load_training_checkpoint,
    save_reward_checkpoint,
    save_training_checkpoint,
)


class RewardTrainer:
    def __init__(self, config, metadata, device="cuda"):
        self.config, self.metadata = config, metadata
        self.device = torch.device(device)
        torch.manual_seed(config.training_seed)
        torch.cuda.manual_seed_all(config.training_seed)
        self.model = GoalConditionedReward(
            config.embedding_dim, config.reward_hidden_dims
        ).to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.reward_learning_rate,
            weight_decay=config.reward_weight_decay,
        )

    def loader(self, dataset, epoch, shuffle):
        generator = torch.Generator().manual_seed(self.config.training_seed + epoch)
        return DataLoader(
            dataset,
            batch_size=self.config.reward_batch_size,
            shuffle=shuffle,
            generator=generator if shuffle else None,
            num_workers=0,
            pin_memory=True,
        )

    @torch.inference_mode()
    def evaluate(self, dataset):
        self.model.eval()
        loss_sum = absolute_sum = count = 0
        for states, goals, labels in self.loader(dataset, 0, False):
            states = states.to(self.device, dtype=torch.float32, non_blocking=True)
            goals = goals.to(self.device, dtype=torch.float32, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            logits = self.model.logits(states, goals)
            loss_sum += float(functional.binary_cross_entropy_with_logits(logits, labels, reduction="sum"))
            absolute_sum += float((torch.sigmoid(logits) - labels).abs().sum())
            count += len(labels)
        if not count:
            raise ValueError("empty SCOPE reward development dataset")
        return {"loss": loss_sum / count, "mean_absolute_error": absolute_sum / count}

    def fit(self, training, development, output, stop_requested=lambda: False):
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        best_loss, stale, start_epoch = math.inf, 0, 0
        last_checkpoint = output / "checkpoint-last.pt"
        if last_checkpoint.exists():
            resume = load_training_checkpoint(
                last_checkpoint, self.model, self.optimizer, self.metadata
            )
            start_epoch = resume["next_epoch"]
            best_loss = resume["best_development_loss"]
            stale = resume["stale_epochs"]
        metrics_path = output / "metrics.jsonl"
        if start_epoch >= self.config.reward_epochs or stale >= self.config.reward_patience:
            if not (output / "checkpoint-best.pt").is_file():
                raise FileNotFoundError("SCOPE resume is complete but best checkpoint is absent")
        epochs_completed = start_epoch
        for epoch in range(start_epoch, self.config.reward_epochs):
            self.model.train()
            total_loss = count = 0
            for states, goals, labels in self.loader(training, epoch, True):
                if stop_requested():
                    raise InterruptedError("SCOPE reward training stopped")
                states = states.to(self.device, dtype=torch.float32, non_blocking=True)
                goals = goals.to(self.device, dtype=torch.float32, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                self.optimizer.zero_grad(set_to_none=True)
                logits = self.model.logits(states, goals)
                loss = functional.binary_cross_entropy_with_logits(logits, labels)
                loss.backward()
                for name, parameter in self.model.named_parameters():
                    if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                        raise FloatingPointError(f"non-finite SCOPE reward gradient: {name}")
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                total_loss += float(loss.detach()) * len(labels)
                count += len(labels)
            dev = self.evaluate(development)
            record = {
                "epoch": epoch,
                "training_loss": total_loss / count,
                "development_loss": dev["loss"],
                "development_mean_absolute_error": dev["mean_absolute_error"],
            }
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
                handle.flush()
            if dev["loss"] < best_loss:
                best_loss, stale = dev["loss"], 0
                save_reward_checkpoint(
                    output / "checkpoint-best.pt",
                    self.model,
                    self.metadata,
                    epoch,
                    best_loss,
                )
            else:
                stale += 1
            save_training_checkpoint(
                last_checkpoint,
                self.model,
                self.optimizer,
                self.metadata,
                epoch,
                best_loss,
                stale,
            )
            epochs_completed = epoch + 1
            if stale >= self.config.reward_patience:
                break
        result = {
            "schema_version": 1,
            "status": "complete",
            "epochs_completed": epochs_completed,
            "resumed_from_epoch": start_epoch,
            "selected_checkpoint": str(output / "checkpoint-best.pt"),
            "selection_rule": "minimum development BCE; train/dev only",
            "best_development_loss": best_loss,
            "training_examples": len(training),
            "development_examples": len(development),
        }
        atomic_json(output / "training-result.json", result)
        return result
