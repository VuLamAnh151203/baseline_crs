import json
import math
from pathlib import Path

import torch

from .checkpoint import load_checkpoint, save_checkpoint
from .dataset import TripCollator


def _to_device(batch, device):
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


class TripTrainer:
    def __init__(self, model, config, device, pad_token_id):
        self.model = model.to(device)
        self.config = config
        self.device = torch.device(device)
        self.collate = TripCollator(pad_token_id)
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        )
        self.scheduler = None
        self.generator = torch.Generator().manual_seed(config.training.seed)
        self.step = 0

    def _make_scheduler(self, total_steps):
        warmup = round(total_steps * self.config.training.warmup_ratio)

        def scale(step):
            if warmup and step < warmup:
                return step / max(1, warmup)
            return max(0.0, (total_steps - step) / max(1, total_steps - warmup))

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, scale)

    def _batches(self, dataset, epoch, offset=0):
        size = self.config.training.batch_size
        count = len(dataset) // size
        order = torch.randperm(
            len(dataset), generator=torch.Generator().manual_seed(self.config.training.seed + epoch)
        )[: count * size]
        for batch_index in range(offset, count):
            indices = order[batch_index * size : (batch_index + 1) * size].tolist()
            yield self.collate([dataset[index] for index in indices])

    @torch.no_grad()
    def evaluate(self, dataset):
        self.model.eval()
        size = self.config.training.batch_size
        values = []
        for start in range(0, len(dataset), size):
            batch = self.collate(
                [dataset[index] for index in range(start, min(start + size, len(dataset)))]
            )
            output = self.model(_to_device(batch, self.device))
            values.append({key: float(value) for key, value in output.items()})
        if not values:
            raise ValueError("TRIP validation dataset is empty")
        return {key: sum(row[key] for row in values) / len(values) for key in values[0]}

    def fit(self, training, validation, output: Path, resume=None, stop_requested=lambda: False):
        output.mkdir(parents=True, exist_ok=True)
        batch_count = len(training) // self.config.training.batch_size
        accumulation = self.config.training.gradient_accumulation
        if batch_count < accumulation or batch_count % accumulation:
            raise ValueError(
                "TRIP full batches per epoch must be a positive multiple of gradient_accumulation"
            )
        optimizer_steps_per_epoch = batch_count // accumulation
        total_steps = optimizer_steps_per_epoch * self.config.training.epochs
        self._make_scheduler(total_steps)
        start_epoch = 0
        start_batch = 0
        if resume is not None:
            payload = load_checkpoint(
                resume,
                self.model,
                self.config.training_sha256,
                self.optimizer,
                self.scheduler,
                self.generator,
                self.device,
            )
            self.step, start_epoch = payload["step"], payload["epoch"]
            start_batch = payload["batch_in_epoch"]
        log_path = output / "metrics.jsonl"
        if resume is not None and log_path.exists():
            retained = [
                line
                for line in log_path.read_text(encoding="utf-8").splitlines()
                if line.strip() and json.loads(line)["step"] <= self.step
            ]
            log_path.write_text("".join(line + "\n" for line in retained), encoding="utf-8")
        best_accuracy = -math.inf
        if log_path.exists():
            previous = [
                json.loads(line)
                for line in log_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            best_accuracy = max(
                (row["validation_accuracy"] for row in previous if "validation_accuracy" in row),
                default=-math.inf,
            )
        self.optimizer.zero_grad(set_to_none=True)
        for epoch in range(start_epoch, self.config.training.epochs):
            losses = []
            offset = start_batch if epoch == start_epoch else 0
            for batch_index, batch in enumerate(
                self._batches(training, epoch, offset), start=offset + 1
            ):
                self.model.train()
                with torch.autocast(
                    device_type=self.device.type,
                    dtype=torch.bfloat16,
                    enabled=self.device.type == "cuda",
                ):
                    model_output = self.model(_to_device(batch, self.device))
                    loss = model_output["loss"] / accumulation
                if not torch.isfinite(loss):
                    raise FloatingPointError("non-finite TRIP training loss")
                loss.backward()
                losses.append(float(model_output["loss"].detach()))
                if batch_index % accumulation:
                    continue
                norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.training.gradient_clip
                )
                if not torch.isfinite(norm):
                    raise FloatingPointError("non-finite TRIP gradient norm")
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)
                self.step += 1
                record = {
                    "step": self.step,
                    "epoch": epoch,
                    "train_loss": sum(losses[-accumulation:]) / accumulation,
                    "gradient_norm": float(norm),
                    "learning_rate": self.optimizer.param_groups[0]["lr"],
                }
                if self.step % self.config.training.evaluate_every == 0:
                    validation_metrics = self.evaluate(validation)
                    record.update(
                        {
                            f"validation_{key}": value
                            for key, value in validation_metrics.items()
                        }
                    )
                    if validation_metrics["accuracy"] > best_accuracy:
                        best_accuracy = validation_metrics["accuracy"]
                        record["best_checkpoint_sha256"] = save_checkpoint(
                            output / "checkpoint-best.pt",
                            self.model,
                            self.optimizer,
                            self.scheduler,
                            self.step,
                            epoch,
                            self.config.training_sha256,
                            self.generator,
                            batch_in_epoch=batch_index,
                        )
                if self.step % self.config.training.checkpoint_every == 0:
                    record["checkpoint_sha256"] = save_checkpoint(
                        output / f"checkpoint-{self.step:08d}.pt",
                        self.model,
                        self.optimizer,
                        self.scheduler,
                        self.step,
                        epoch,
                        self.config.training_sha256,
                        self.generator,
                        batch_in_epoch=batch_index,
                    )
                    save_checkpoint(
                        output / "checkpoint-last.pt",
                        self.model,
                        self.optimizer,
                        self.scheduler,
                        self.step,
                        epoch,
                        self.config.training_sha256,
                        self.generator,
                        batch_in_epoch=batch_index,
                    )
                with log_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, sort_keys=True) + "\n")
                if stop_requested():
                    save_checkpoint(
                        output / "checkpoint-last.pt",
                        self.model,
                        self.optimizer,
                        self.scheduler,
                        self.step,
                        epoch,
                        self.config.training_sha256,
                        self.generator,
                        batch_in_epoch=batch_index,
                    )
                    raise InterruptedError("TRIP training stop requested after durable checkpoint")
            # Epoch boundaries are exact deterministic resume points.
            save_checkpoint(
                output / "checkpoint-last.pt",
                self.model,
                self.optimizer,
                self.scheduler,
                self.step,
                epoch + 1,
                self.config.training_sha256,
                self.generator,
            )
        final_hash = save_checkpoint(
            output / "checkpoint-final.pt",
            self.model,
            self.optimizer,
            self.scheduler,
            self.step,
            self.config.training.epochs,
            self.config.training_sha256,
            self.generator,
        )
        return {
            "step": self.step,
            "epochs": self.config.training.epochs,
            "checkpoint_sha256": final_hash,
        }
