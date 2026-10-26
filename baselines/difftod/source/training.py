import json
import math
from pathlib import Path

import torch
from torch.utils.data import TensorDataset

from .checkpoint import ExponentialMovingAverage, load_training_checkpoint, save_training_checkpoint
from .config import DiffTODConfig
from .diffusion import AbsorbingGraph, LogLinearNoise, score_entropy_loss


def _encode(tokenizer, text):
    encoded = tokenizer(text, add_special_tokens=False, return_attention_mask=False)
    ids = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
    if not isinstance(ids, list) or any(
        not isinstance(token, int) or token < 0 for token in ids
    ):
        raise ValueError("tokenizer returned invalid input IDs")
    return ids


def token_blocks(path: Path, tokenizer, length):
    if not path.is_file() or length < 1:
        raise ValueError("DiffTOD corpus and positive block length are required")
    eos = tokenizer.eos_token_id
    if not isinstance(eos, int) or eos < 0:
        raise ValueError("tokenizer requires an EOS token ID")
    stream = []
    seen_ids = set()
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if set(row) != {"id", "source_sha256", "text"}:
                raise ValueError(f"invalid prepared DiffTOD row at line {line_number}")
            if row["id"] in seen_ids or not isinstance(row["text"], str) or not row["text"]:
                raise ValueError("duplicate ID or empty DiffTOD training trajectory")
            seen_ids.add(row["id"])
            stream.extend(_encode(tokenizer, row["text"]))
            stream.append(eos)
    usable = len(stream) // length * length
    if usable == 0:
        raise ValueError("prepared DiffTOD corpus is shorter than one model block")
    tokens = torch.tensor(stream[:usable], dtype=torch.long).view(-1, length)
    return TensorDataset(tokens)


def make_optimizer(model, noise, config):
    parameters = [*model.parameters(), *noise.parameters()]
    return torch.optim.AdamW(
        parameters,
        lr=config.learning_rate,
        betas=(config.beta1, config.beta2),
        eps=config.epsilon,
        weight_decay=config.weight_decay,
    )


def set_learning_rate(optimizer, step, config):
    scale = min(step / config.warmup_steps, 1.0)
    for group in optimizer.param_groups:
        group["lr"] = config.learning_rate * scale


class DiffTODTrainer:
    def __init__(self, model, config: DiffTODConfig, device):
        self.model = model.to(device)
        self.config = config
        self.device = torch.device(device)
        self.graph = AbsorbingGraph(config.model.vocab_size)
        self.noise = LogLinearNoise().to(device)
        self.optimizer = make_optimizer(model, self.noise, config.training)
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.device.type == "cuda")
        self.ema = ExponentialMovingAverage(model.parameters(), config.training.ema_decay)
        self.generator = torch.Generator(device=self.device).manual_seed(config.training.seed)
        self.step = 0

    def train_batch(self, batch):
        batch = batch.to(self.device, non_blocking=True)
        self.model.train()
        loss = score_entropy_loss(
            self.model,
            batch,
            self.graph,
            self.noise,
            generator=self.generator,
        ).mean()
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite DiffTOD training loss")
        self.scaler.scale(loss / self.config.training.gradient_accumulation).backward()
        return float(loss.detach())

    def optimizer_step(self):
        next_step = self.step + 1
        set_learning_rate(self.optimizer, next_step, self.config.training)
        self.scaler.unscale_(self.optimizer)
        norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.config.training.gradient_clip
        )
        if not torch.isfinite(norm):
            raise FloatingPointError("non-finite DiffTOD gradient norm")
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)
        self.ema.update(self.model.parameters())
        self.step = next_step
        return float(norm)

    @torch.no_grad()
    def evaluate(self, batch):
        batch = batch.to(self.device, non_blocking=True)
        self.model.eval()
        self.ema.store(self.model.parameters())
        self.ema.copy_to(self.model.parameters())
        try:
            loss = score_entropy_loss(
                self.model,
                batch,
                self.graph,
                self.noise,
                generator=self.generator,
            ).mean()
        finally:
            self.ema.restore(self.model.parameters())
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite DiffTOD validation loss")
        return float(loss)

    def save(self, path):
        return save_training_checkpoint(
            path,
            self.model,
            self.optimizer,
            self.ema,
            self.scaler,
            self.step,
            self.config.training_sha256,
            self.generator,
        )

    def resume(self, path):
        self.step = load_training_checkpoint(
            path,
            self.model,
            self.optimizer,
            self.ema,
            self.scaler,
            self.config.training_sha256,
            generator=self.generator,
            map_location=self.device,
        )
        return self.step

    def fit(self, training, validation, output: Path, *, stop_requested=lambda: False):
        output.mkdir(parents=True, exist_ok=True)
        train_tokens, valid_tokens = training.tensors[0], validation.tensors[0]
        batch_size = self.config.training.batch_size
        train_batches = len(train_tokens) // batch_size
        valid_batches = math.ceil(len(valid_tokens) / batch_size)
        if not train_batches or not valid_batches:
            raise ValueError("DiffTOD train/validation loaders must be nonempty")

        def training_batches(start_microstep):
            microstep = start_microstep
            while True:
                epoch, offset = divmod(microstep, train_batches)
                order = torch.randperm(
                    len(train_tokens),
                    generator=torch.Generator().manual_seed(self.config.training.seed + epoch),
                )[: train_batches * batch_size]
                for batch_index in range(offset, train_batches):
                    indices = order[batch_index * batch_size : (batch_index + 1) * batch_size]
                    yield train_tokens[indices]
                    microstep += 1
                # The next loop starts a fresh deterministic epoch.
                microstep = (epoch + 1) * train_batches

        start_microstep = self.step * self.config.training.gradient_accumulation
        batches = training_batches(start_microstep)
        log_path = output / "metrics.jsonl"
        while self.step < self.config.training.steps:
            if stop_requested():
                self.save(output / "checkpoint-last.pt")
                raise InterruptedError("DiffTOD training stop requested after durable checkpoint")
            losses = []
            for _ in range(self.config.training.gradient_accumulation):
                losses.append(self.train_batch(next(batches)))
            gradient_norm = self.optimizer_step()
            record = {
                "step": self.step,
                "train_loss": sum(losses) / len(losses),
                "gradient_norm": gradient_norm,
                "learning_rate": self.optimizer.param_groups[0]["lr"],
            }
            if self.step % self.config.training.evaluate_every == 0:
                batch_index = (self.step // self.config.training.evaluate_every) % valid_batches
                start = batch_index * batch_size
                record["validation_loss"] = self.evaluate(valid_tokens[start : start + batch_size])
            if self.step % self.config.training.checkpoint_every == 0:
                record["checkpoint_sha256"] = self.save(
                    output / f"checkpoint-{self.step:08d}.pt"
                )
                self.save(output / "checkpoint-last.pt")
            finite = all(
                math.isfinite(value) for value in record.values() if isinstance(value, float)
            )
            if not finite:
                raise FloatingPointError("non-finite DiffTOD metric")
            with log_path.open("a", encoding="utf-8") as log:
                log.write(json.dumps(record, sort_keys=True) + "\n")
        final_hash = self.save(output / "checkpoint-final.pt")
        self.save(output / "checkpoint-last.pt")
        return {"step": self.step, "checkpoint_sha256": final_hash}
