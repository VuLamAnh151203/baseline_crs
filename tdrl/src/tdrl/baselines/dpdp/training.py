import json
import math
from pathlib import Path

import torch

from ...provenance import atomic_json
from .checkpoint import load_checkpoint, save_checkpoint
from .dataset import DPDPCollator


def _to_device(batch, device):
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def assert_finite_gradients(model):
    for name, parameter in model.named_parameters():
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
            raise FloatingPointError(f"non-finite DPDP gradient: {name}")


class DPDPTrainer:
    def __init__(self, model, config, device, pad_token_id, metadata, phase="offline"):
        if phase not in {"offline", "selfplay"} or metadata.phase != phase:
            raise ValueError("DPDP trainer/checkpoint phase mismatch")
        self.model = model.to(device)
        self.config = config
        self.device = torch.device(device)
        self.metadata = metadata
        self.phase = phase
        self.collate = DPDPCollator(pad_token_id)
        lr = config.offline_lr if phase == "offline" else config.selfplay_lr
        self.critic_weight = (
            config.offline_critic_weight if phase == "offline" else config.selfplay_critic_weight
        )
        self.epochs = config.offline_epochs if phase == "offline" else config.selfplay_epochs
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=config.offline_weight_decay
        )
        self.scheduler = None
        self.generator = torch.Generator().manual_seed(config.offline_seed)
        self.step = 0

    def _make_scheduler(self, total_steps):
        warmup = self.config.offline_warmup_steps

        def scale(step):
            if warmup and step < warmup:
                return step / max(1, warmup)
            return max(0.0, (total_steps - step) / max(1, total_steps - warmup))

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, scale)

    def _batches(self, dataset, epoch, offset=0):
        size = self.config.offline_batch_size
        order = torch.randperm(
            len(dataset),
            generator=torch.Generator().manual_seed(self.config.offline_seed + epoch),
        )
        for batch_index, start in enumerate(range(0, len(order), size)):
            if batch_index < offset:
                continue
            indices = order[start : start + size].tolist()
            yield batch_index, self.collate([dataset[index] for index in indices])

    @torch.no_grad()
    def evaluate(self, dataset):
        self.model.eval()
        totals = []
        for start in range(0, len(dataset), self.config.offline_batch_size):
            rows = [
                dataset[index]
                for index in range(
                    start, min(start + self.config.offline_batch_size, len(dataset))
                )
            ]
            batch = _to_device(self.collate(rows), self.device)
            output = self.model(
                batch["input_ids"],
                batch["attention_mask"],
                batch["actions"],
                batch["returns"],
                self.critic_weight,
            )
            totals.append(
                {
                    "total_loss": float(output["total_loss"]),
                    "actor_loss": float(output["actor_loss"]),
                    "critic_loss": float(output["critic_loss"]),
                }
            )
        if not totals:
            raise ValueError("empty DPDP validation dataset")
        return {key: sum(row[key] for row in totals) / len(totals) for key in totals[0]}

    def fit(self, training, validation, output: Path, resume=None, stop_requested=lambda: False):
        output.mkdir(parents=True, exist_ok=True)
        batches_per_epoch = math.ceil(len(training) / self.config.offline_batch_size)
        self._make_scheduler(batches_per_epoch * self.epochs)
        start_epoch = start_batch = 0
        if resume is not None:
            payload = load_checkpoint(
                resume,
                self.model,
                self.metadata,
                self.optimizer,
                self.scheduler,
                self.generator,
                self.device,
            )
            self.step, start_epoch = payload["step"], payload["epoch"]
            start_batch = payload["batch_in_epoch"]
        log_path = output / "metrics.jsonl"
        if resume is not None and log_path.exists():
            lines = [
                line
                for line in log_path.read_text(encoding="utf-8").splitlines()
                if line.strip() and json.loads(line)["step"] <= self.step
            ]
            log_path.write_text("".join(line + "\n" for line in lines), encoding="utf-8")
        best_loss = math.inf
        if log_path.exists():
            prior = [json.loads(line) for line in log_path.read_text().splitlines() if line]
            best_loss = min((row["validation_total_loss"] for row in prior), default=math.inf)
        for epoch in range(start_epoch, self.epochs):
            offset = start_batch if epoch == start_epoch else 0
            for batch_index, raw_batch in self._batches(training, epoch, offset):
                self.model.train()
                batch = _to_device(raw_batch, self.device)
                self.optimizer.zero_grad(set_to_none=True)
                output_values = self.model(
                    batch["input_ids"],
                    batch["attention_mask"],
                    batch["actions"],
                    batch["returns"],
                    self.critic_weight,
                )
                output_values["total_loss"].backward()
                assert_finite_gradients(self.model)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
                self.optimizer.step()
                self.scheduler.step()
                self.step += 1
                next_epoch = epoch + 1 if batch_index + 1 == batches_per_epoch else epoch
                next_batch = 0 if next_epoch != epoch else batch_index + 1
                stopping = stop_requested()
                checkpoint_due = (
                    self.step % self.config.offline_checkpoint_every == 0
                    or next_epoch != epoch
                    or stopping
                )
                if checkpoint_due:
                    save_checkpoint(
                        output / "checkpoint-last.pt",
                        self.model,
                        self.optimizer,
                        self.scheduler,
                        self.step,
                        next_epoch,
                        next_batch,
                        self.metadata,
                        self.generator,
                    )
                if self.step % self.config.offline_checkpoint_every == 0 or next_epoch != epoch:
                    validation_metrics = self.evaluate(validation)
                    row = {
                        "step": self.step,
                        "epoch": epoch,
                        "batch": batch_index,
                        "training_total_loss": float(output_values["total_loss"].detach()),
                        **{f"validation_{key}": value for key, value in validation_metrics.items()},
                    }
                    with log_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(row, sort_keys=True) + "\n")
                    if validation_metrics["total_loss"] < best_loss:
                        best_loss = validation_metrics["total_loss"]
                        save_checkpoint(
                            output / "checkpoint-best.pt",
                            self.model,
                            self.optimizer,
                            self.scheduler,
                            self.step,
                            next_epoch,
                            next_batch,
                            self.metadata,
                            self.generator,
                        )
                if stopping:
                    return {
                        "status": "partial",
                        "step": self.step,
                        "best_validation_loss": best_loss,
                    }
        save_checkpoint(
            output / "checkpoint-final.pt",
            self.model,
            self.optimizer,
            self.scheduler,
            self.step,
            self.epochs,
            0,
            self.metadata,
            self.generator,
        )
        result = {"status": "complete", "step": self.step, "best_validation_loss": best_loss}
        atomic_json(output / "training-result.json", result)
        return result


def episode_returns(rewards, gamma):
    values, running = [], 0.0
    for reward in reversed(rewards):
        running = float(reward) + gamma * running
        values.append(running)
    return list(reversed(values))


def selfplay_update(model, optimizer, batch, critic_weight, gradient_clip):
    """One released-code-style actor/Q update from actions actually executed by MCTS."""
    model.train()
    optimizer.zero_grad(set_to_none=True)
    output = model(
        batch["input_ids"],
        batch["attention_mask"],
        batch["actions"],
        batch["returns"],
        critic_weight,
    )
    output["total_loss"].backward()
    assert_finite_gradients(model)
    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
    optimizer.step()
    return {key: float(output[key].detach()) for key in ("total_loss", "actor_loss", "critic_loss")}
