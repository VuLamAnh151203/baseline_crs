import math

import torch
import torch.nn.functional as functional
from torch import nn


class TripDecoder(nn.Module):
    def __init__(self, vocab_size, padding_idx, config):
        super().__init__()
        self.padding_idx = padding_idx
        self.token_embedding = nn.Embedding(vocab_size, config.hidden_size, padding_idx=padding_idx)
        self.position_embedding = nn.Embedding(config.max_positions, config.hidden_size)
        layer = nn.TransformerDecoderLayer(
            d_model=config.hidden_size,
            nhead=config.attention_heads,
            dim_feedforward=config.feedforward_size,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.decoder = nn.TransformerDecoder(layer, config.decoder_layers)
        self.dropout = nn.Dropout(config.dropout)
        self.apply(lambda module: self._initialize(module, config.init_std))

    @staticmethod
    def _initialize(module, std):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].zero_()

    def forward(self, input_ids, attention_mask, memory, memory_mask):
        length = input_ids.shape[1]
        if length > self.position_embedding.num_embeddings:
            raise ValueError("TRIP decoder input exceeds positional capacity")
        positions = torch.arange(length, device=input_ids.device).unsqueeze(0)
        hidden = self.dropout(
            self.token_embedding(input_ids) * math.sqrt(self.token_embedding.embedding_dim)
        )
        hidden = hidden + self.position_embedding(positions)
        causal = torch.triu(
            torch.ones(length, length, dtype=torch.bool, device=input_ids.device), diagonal=1
        )
        return self.decoder(
            hidden,
            memory,
            tgt_mask=causal,
            tgt_key_padding_mask=attention_mask == 0,
            memory_key_padding_mask=memory_mask == 0,
        )


class TripModel(nn.Module):
    """Modern, checkpoint-safe implementation of TRIP's dual-encoder/dual-decoder core."""

    def __init__(self, context_encoder, target_encoder, vocab_size, pad_token_id, config):
        super().__init__()
        self.context_encoder = context_encoder
        self.target_encoder = target_encoder
        self.config = config
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id
        self.context_encoder.resize_token_embeddings(vocab_size)
        self.target_encoder.resize_token_embeddings(vocab_size)
        self.forward_decoder = TripDecoder(vocab_size, pad_token_id, config)
        self.backward_decoder = TripDecoder(vocab_size, pad_token_id, config)
        self.forward_lm = nn.Linear(config.hidden_size, vocab_size)
        self.backward_lm = nn.Linear(config.hidden_size, vocab_size)
        self.projection = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size), nn.ReLU()
        )
        for head in (self.forward_lm, self.backward_lm):
            nn.init.normal_(head.weight, mean=0.0, std=config.init_std)
            nn.init.zeros_(head.bias)

    def encode(self, context_ids, context_mask, target_ids, target_mask):
        context = self.context_encoder(
            input_ids=context_ids, attention_mask=context_mask, return_dict=True
        ).last_hidden_state
        target = self.target_encoder(
            input_ids=target_ids, attention_mask=target_mask, return_dict=True
        ).last_hidden_state
        return torch.cat((context, target), dim=1), torch.cat(
            (context_mask, target_mask), dim=1
        ), target

    @staticmethod
    def average(hidden, mask):
        mask = mask.to(hidden.dtype).unsqueeze(-1)
        return (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)

    def distance(self, backward_hidden, forward_hidden, backward_mask, forward_mask):
        backward = self.average(self.projection(backward_hidden), backward_mask)
        forward = self.average(self.projection(forward_hidden), forward_mask)
        # This intentionally preserves upstream TRIP's batch-level torch.dist objective.
        return torch.linalg.vector_norm(backward - forward)

    def decode_direction(self, direction, ids, mask, memory, memory_mask):
        decoder = self.forward_decoder if direction == "forward" else self.backward_decoder
        head = self.forward_lm if direction == "forward" else self.backward_lm
        hidden = decoder(ids, mask, memory, memory_mask)
        return head(hidden), hidden

    def _negative_hidden(self, direction, ids, mask, memory, memory_mask):
        batch, count, length = ids.shape
        repeated_memory = memory[:, None].expand(-1, count, -1, -1).reshape(
            batch * count, memory.shape[1], memory.shape[2]
        )
        repeated_mask = memory_mask[:, None].expand(-1, count, -1).reshape(
            batch * count, memory_mask.shape[1]
        )
        _, hidden = self.decode_direction(
            direction,
            ids.reshape(batch * count, length),
            mask.reshape(batch * count, length),
            repeated_memory,
            repeated_mask,
        )
        return hidden.reshape(batch, count, length, -1)

    def forward(self, batch):
        memory, memory_mask, target_hidden = self.encode(
            batch["context_ids"],
            batch["context_attention_mask"],
            batch["target_ids"],
            batch["target_attention_mask"],
        )
        outputs = {}
        hidden_by_direction = {}
        lm_loss = 0.0
        accuracy = 0.0
        for direction in ("forward", "backward"):
            plan = batch[f"{direction}_ids"]
            plan_mask = batch[f"{direction}_attention_mask"]
            inputs, input_mask = plan[:, :-1], plan_mask[:, :-1]
            labels = plan[:, 1:].masked_fill(plan_mask[:, 1:] == 0, -100)
            logits, hidden = self.decode_direction(
                direction, inputs, input_mask, memory, memory_mask
            )
            loss = functional.cross_entropy(logits.reshape(-1, self.vocab_size), labels.reshape(-1))
            outputs[f"{direction}_loss"] = loss
            lm_loss = lm_loss + loss
            valid = labels != -100
            accuracy = accuracy + (
                ((logits.argmax(-1) == labels) & valid).sum() / valid.sum().clamp_min(1)
            )
            hidden_by_direction[direction] = hidden
        forward_mask = batch["forward_attention_mask"][:, :-1]
        backward_mask = batch["backward_attention_mask"][:, :-1]
        distance_loss = self.distance(
            hidden_by_direction["backward"],
            hidden_by_direction["forward"],
            backward_mask,
            forward_mask,
        )
        target_vector = self.average(
            self.projection(target_hidden), batch["target_attention_mask"]
        )
        contrastive_losses = []
        for direction in ("forward", "backward"):
            positive_mask = batch[f"{direction}_target_mask"][:, :-1]
            positive = self.average(
                self.projection(hidden_by_direction[direction]), positive_mask
            )
            negative_ids = batch[f"negative_{direction}_ids"][:, :, :-1]
            negative_masks = batch[f"negative_{direction}_attention_mask"][:, :, :-1]
            negative_target_masks = batch[f"negative_{direction}_target_mask"][:, :, :-1]
            negative_hidden = self._negative_hidden(
                direction, negative_ids, negative_masks, memory, memory_mask
            )
            batch_size, count, length, width = negative_hidden.shape
            negative = self.average(
                self.projection(negative_hidden.reshape(batch_size * count, length, width)),
                negative_target_masks.reshape(batch_size * count, length),
            ).reshape(batch_size, count, width)
            positive_similarity = functional.cosine_similarity(target_vector, positive).unsqueeze(1)
            negative_similarity = functional.cosine_similarity(
                target_vector[:, None, :], negative, dim=-1
            )
            logits = torch.cat((positive_similarity, negative_similarity), dim=1) / self.config.tau
            contrastive_losses.append(
                functional.cross_entropy(
                    logits, torch.zeros(batch_size, dtype=torch.long, device=logits.device)
                )
            )
        contrastive_loss = sum(contrastive_losses) / len(contrastive_losses)
        total = lm_loss + self.config.beta * distance_loss + self.config.gamma * contrastive_loss
        return {
            "loss": total,
            "lm_loss": lm_loss,
            "distance_loss": distance_loss,
            "contrastive_loss": contrastive_loss,
            "accuracy": accuracy / 2,
            **outputs,
        }


def load_pretrained_model(encoder_path, tokenizer, config, device="cpu"):
    from transformers import AutoModel

    context = AutoModel.from_pretrained(encoder_path, local_files_only=True)
    target = AutoModel.from_pretrained(encoder_path, local_files_only=True)
    hidden = getattr(context.config, "hidden_size", None)
    if hidden != config.hidden_size or getattr(target.config, "hidden_size", None) != hidden:
        raise ValueError("TRIP encoder hidden size does not match its configuration")
    return TripModel(context, target, len(tokenizer), tokenizer.pad_token_id, config).to(device)
