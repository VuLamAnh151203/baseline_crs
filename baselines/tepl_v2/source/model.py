import torch
from torch import nn
from torch.nn import functional


class EPLPolicy(nn.Module):
    """Released EPL-style encoder, projection layer, and action classifier."""

    def __init__(self, encoder, encoder_size, hidden_size, actions, dropout=0.5):
        super().__init__()
        if min(encoder_size, hidden_size) < 1 or len(actions) < 2:
            raise ValueError("invalid EPL policy architecture")
        self.encoder = encoder
        self.actions = tuple(actions)
        # Kept for architecture provenance: the released EPL module declares this
        # layer but does not apply it in forward().
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Linear(encoder_size, hidden_size)
        self.output = nn.Linear(hidden_size, len(actions))

    def forward(self, input_ids, attention_mask, labels=None):
        encoded = self.encoder(
            input_ids=input_ids, attention_mask=attention_mask, return_dict=True
        )
        pooled = getattr(encoded, "pooler_output", None)
        if pooled is None:
            pooled = encoded.last_hidden_state[:, 0]
        hidden = functional.relu(self.projection(pooled))
        logits = self.output(hidden)
        if not torch.isfinite(logits).all():
            raise FloatingPointError("non-finite EPL policy output")
        result = {"logits": logits, "probabilities": functional.softmax(logits, dim=-1)}
        if labels is not None:
            if labels.ndim != 1 or len(labels) != len(input_ids):
                raise ValueError("invalid EPL policy labels")
            result["loss"] = functional.cross_entropy(logits, labels)
        return result


def load_pretrained_policy(encoder_path, config, vocabulary_size, device="cpu"):
    from transformers import AutoModel

    encoder = AutoModel.from_pretrained(encoder_path, local_files_only=True)
    encoder_size = getattr(encoder.config, "hidden_size", None)
    if encoder_size != 768:
        raise ValueError("EPL requires bert-base-cased with hidden size 768")
    encoder.resize_token_embeddings(vocabulary_size)
    return EPLPolicy(
        encoder, encoder_size, config.hidden_size, config.actions, config.dropout
    ).to(device)
