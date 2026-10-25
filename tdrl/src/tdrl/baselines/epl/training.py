import json
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ...provenance import atomic_json
from .checkpoint import load_checkpoint, save_checkpoint
from .dataset import EPLCollator


def _device_batch(batch, device):
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


class EPLTrainer:
    def __init__(self, model, config, metadata, pad_token_id, device="cuda"):
        self.model = model.to(device)
        self.config = config
        self.metadata = metadata
        self.device = torch.device(device)
        self.collator = EPLCollator(pad_token_id)
        no_decay = ("bias", "LayerNorm.weight")
        groups = [
            {
                "params": [
                    parameter
                    for name, parameter in self.model.named_parameters()
                    if not any(item in name for item in no_decay)
                ],
                "weight_decay": config.training_weight_decay,
            },
            {
                "params": [
                    parameter
                    for name, parameter in self.model.named_parameters()
                    if any(item in name for item in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        self.optimizer = torch.optim.AdamW(
            groups,
            lr=config.training_learning_rate,
        )

    def _set_learning_rate(self, step, total_steps):
        warmup = self.config.training_warmup_steps
        if warmup and step < warmup:
            scale = step / warmup
        else:
            scale = max(0.0, (total_steps - step) / max(1, total_steps - warmup))
        learning_rate = self.config.training_learning_rate * scale
        for group in self.optimizer.param_groups:
            group["lr"] = learning_rate
        return learning_rate

    def _loader(self, dataset, epoch, shuffle):
        generator = torch.Generator().manual_seed(self.config.training_seed + epoch)
        return DataLoader(
            dataset,
            batch_size=(
                self.config.training_batch_size
                if shuffle
                else self.config.evaluation_batch_size
            ),
            shuffle=shuffle,
            generator=generator if shuffle else None,
            collate_fn=self.collator,
            num_workers=0,
            pin_memory=True,
        )

    @torch.no_grad()
    def evaluate(self, dataset):
        self.model.eval()
        total_loss = total_correct = total_examples = 0
        for batch in self._loader(dataset, 0, False):
            batch = _device_batch(batch, self.device)
            output = self.model(
                batch["input_ids"], batch["attention_mask"], batch["labels"]
            )
            size = len(batch["labels"])
            total_loss += float(output["loss"]) * size
            total_correct += int((output["logits"].argmax(-1) == batch["labels"]).sum())
            total_examples += size
        if not total_examples:
            raise ValueError("empty EPL development set")
        return {"loss": total_loss / total_examples, "accuracy": total_correct / total_examples}

    def fit(self, training, validation, output, stop_requested=lambda: False):
        output_dir = Path(output)
        output_dir.mkdir(parents=True, exist_ok=True)
        last_path = output_dir / "checkpoint-last.pt"
        best_path = output_dir / "checkpoint-best.pt"
        metrics_path = output_dir / "metrics.jsonl"
        start_epoch = step = 0
        best_loss = math.inf
        steps_per_epoch = math.ceil(len(training) / self.config.training_batch_size)
        total_steps = steps_per_epoch * self.config.training_epochs
        if last_path.exists():
            payload, _ = load_checkpoint(
                last_path,
                self.model,
                self.metadata,
                self.optimizer,
                self.device,
            )
            start_epoch = payload["epoch"] + 1
            step = payload["step"]
        if best_path.exists():
            payload, _ = load_checkpoint(
                best_path, self.model, self.metadata, map_location=self.device
            )
            best_loss = float(payload["validation_loss"])
            if last_path.exists():
                load_checkpoint(
                    last_path,
                    self.model,
                    self.metadata,
                    self.optimizer,
                    self.device,
                )
        for epoch in range(start_epoch, self.config.training_epochs):
            self.model.train()
            train_loss = examples = 0
            for batch in self._loader(training, epoch, True):
                if stop_requested():
                    save_checkpoint(
                        last_path,
                        self.model,
                        self.optimizer,
                        epoch - 1,
                        step,
                        self.metadata,
                        best_loss,
                    )
                    return {"status": "partial", "epoch": epoch, "step": step}
                batch = _device_batch(batch, self.device)
                self.optimizer.zero_grad(set_to_none=True)
                model_output = self.model(
                    batch["input_ids"], batch["attention_mask"], batch["labels"]
                )
                model_output["loss"].backward()
                for name, parameter in self.model.named_parameters():
                    if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                        raise FloatingPointError(f"non-finite EPL gradient: {name}")
                if self.config.gradient_clip is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.gradient_clip
                    )
                learning_rate = self._set_learning_rate(step, total_steps)
                self.optimizer.step()
                size = len(batch["labels"])
                train_loss += float(model_output["loss"].detach()) * size
                examples += size
                step += 1
            development = self.evaluate(validation)
            record = {
                "epoch": epoch,
                "step": step,
                "training_loss": train_loss / examples,
                "development_loss": development["loss"],
                "development_accuracy": development["accuracy"],
                "learning_rate": learning_rate,
            }
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
                handle.flush()
            save_checkpoint(
                last_path,
                self.model,
                self.optimizer,
                epoch,
                step,
                self.metadata,
                development["loss"],
            )
            if development["loss"] < best_loss:
                best_loss = development["loss"]
                save_checkpoint(
                    best_path,
                    self.model,
                    None,
                    epoch,
                    step,
                    self.metadata,
                    development["loss"],
                )
        result = {
            "status": "complete",
            "epochs": self.config.training_epochs,
            "steps": step,
            "selected_checkpoint": str(best_path),
            "selection_rule": "minimum development cross-entropy; train/dev only",
            "best_development_loss": best_loss,
        }
        atomic_json(output_dir / "training-result.json", result)
        return result
