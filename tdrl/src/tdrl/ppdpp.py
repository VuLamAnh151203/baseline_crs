"""Independent implementation of the pinned PPDPP policy mechanism.

Reference: dengyang17/PPDPP agent.py at f44dccdd43f335e5e5f372ab556cfd9f182f9c55.
Retains pooled pretrained encoder, dropout=.5, categorical strategy classifier,
supervised cross entropy then on-policy REINFORCE with normalized discounted returns.
Recommendation action labels are a disclosed weak-label dataset adaptation.
No source environment/hosted client is imported.
"""

import torch
from torch import nn

from .official_data import ACTION_TAXONOMY


class PPDPPPolicy(nn.Module):
    def __init__(self, backbone, hidden_size, actions=ACTION_TAXONOMY):
        super().__init__()
        self.policy = backbone
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(hidden_size, len(actions))
        self.actions = sorted(actions)

    def logits(self, input_ids, attention_mask):
        outputs = self.policy(input_ids=input_ids, attention_mask=attention_mask)
        return self.classifier(self.dropout(outputs[1]))

    def supervised_loss(self, input_ids, attention_mask, labels):
        return nn.functional.cross_entropy(self.logits(input_ids, attention_mask).float(), labels)

    def select(self, input_ids, attention_mask, generator=None, evaluate=False):
        probs = self.logits(input_ids, attention_mask).float().softmax(-1)
        indices = (
            probs.argmax(-1)
            if evaluate
            else torch.multinomial(probs, 1, generator=generator).squeeze(-1)
        )
        logs = probs.gather(-1, indices[:, None]).squeeze(-1).log()
        return indices, logs


def reinforce_loss(log_probabilities, rewards, gamma=0.99):
    if len(log_probabilities) != len(rewards) or not rewards:
        raise ValueError("each sampled action needs an on-policy reward")
    accumulated, returns = 0.0, []
    for reward in reversed(rewards):
        accumulated = float(reward) + gamma * accumulated
        returns.insert(0, accumulated)
    returns = torch.tensor(returns, dtype=torch.float32, device=log_probabilities[0].device)
    if len(returns) > 1:
        returns = (returns - returns.mean()) / (returns.std() + torch.finfo(torch.float32).eps)
    return -(torch.stack(log_probabilities).reshape(-1) * returns.detach()).sum()
