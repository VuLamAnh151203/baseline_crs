import torch
from torch import nn
from torch.nn import functional


class DPDPPolicy(nn.Module):
    """Released-code-compatible shared encoder with actor and action-value heads."""

    def __init__(self, encoder, hidden_size, actions, dropout=0.1):
        super().__init__()
        if hidden_size < 1 or len(actions) < 2 or len(set(actions)) != len(actions):
            raise ValueError("invalid DPDP policy architecture")
        self.encoder = encoder
        self.actions = tuple(actions)
        self.dropout = nn.Dropout(dropout)
        self.actor = nn.Linear(hidden_size, len(actions))
        self.critic = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, len(actions)),
        )

    def _pooled(self, input_ids, attention_mask):
        output = self.encoder(
            input_ids=input_ids, attention_mask=attention_mask, return_dict=True
        )
        pooled = getattr(output, "pooler_output", None)
        if pooled is None:
            pooled = output.last_hidden_state[:, 0]
        return self.dropout(pooled)

    def forward(
        self,
        input_ids,
        attention_mask,
        actions=None,
        returns=None,
        critic_weight=1.0,
    ):
        pooled = self._pooled(input_ids, attention_mask)
        logits = self.actor(pooled)
        q_values = self.critic(pooled)
        probabilities = functional.softmax(logits, dim=-1)
        if not torch.isfinite(logits).all() or not torch.isfinite(q_values).all():
            raise FloatingPointError("non-finite DPDP policy output")
        result = {
            "policy_logits": logits,
            "action_probs": probabilities,
            "q_values": q_values,
        }
        if actions is None and returns is None:
            return result
        if actions is None or returns is None or actions.ndim != 1 or returns.ndim != 1:
            raise ValueError("DPDP training requires aligned 1-D actions and returns")
        if len(actions) != len(input_ids) or len(returns) != len(input_ids):
            raise ValueError("DPDP loss batch mismatch")
        selected_q = q_values.gather(1, actions[:, None]).squeeze(1)
        selected_log_probability = functional.log_softmax(logits, dim=-1).gather(
            1, actions[:, None]
        ).squeeze(1)
        advantage = returns.to(selected_q.dtype) - selected_q
        actor_loss = (-selected_log_probability * advantage.detach()).mean()
        critic_loss = functional.mse_loss(selected_q, returns.to(selected_q.dtype))
        total_loss = actor_loss + critic_weight * critic_loss
        if not all(torch.isfinite(value) for value in (actor_loss, critic_loss, total_loss)):
            raise FloatingPointError("non-finite DPDP loss")
        result.update(
            selected_q=selected_q,
            advantage=advantage,
            actor_loss=actor_loss,
            critic_loss=critic_loss,
            total_loss=total_loss,
        )
        return result


def load_pretrained_policy(encoder_path, actions, dropout=0.1, device="cpu"):
    from transformers import AutoModel

    encoder = AutoModel.from_pretrained(encoder_path, local_files_only=True)
    hidden_size = getattr(encoder.config, "hidden_size", None)
    if hidden_size != 1024:
        raise ValueError("DPDP requires a RoBERTa-large encoder with hidden size 1024")
    return DPDPPolicy(encoder, hidden_size, actions, dropout).to(device)
