"""Torch AdamW with FP32 numerical state and unchanged BF16 forward parameters."""

import torch


class HeadAdamW:
    def __init__(self, parameters, lr):
        self.parameters = list(parameters)
        self.masters = [torch.nn.Parameter(p.detach().float().clone()) for p in self.parameters]
        self.optimizer = torch.optim.AdamW(self.masters, lr=lr)

    @property
    def state(self):
        return self.optimizer.state

    def zero_grad(self, set_to_none=True):
        self.optimizer.zero_grad(set_to_none=set_to_none)
        for p in self.parameters:
            if set_to_none:
                p.grad = None
            elif p.grad is not None:
                p.grad.zero_()

    @torch.no_grad()
    def step(self):
        for p, master in zip(self.parameters, self.masters, strict=True):
            master.grad = p.grad.detach().float() if p.grad is not None else None
        self.optimizer.step()
        for p, master in zip(self.parameters, self.masters, strict=True):
            p.copy_(master)

    def state_dict(self):
        return {
            "schema_version": 1,
            "torch_adamw": self.optimizer.state_dict(),
            "masters": [p.detach().clone() for p in self.masters],
        }

    def load_state_dict(self, state):
        if state.get("schema_version") != 1 or len(state["masters"]) != len(self.masters):
            raise ValueError("unknown head optimizer schema/master count")
        self.optimizer.load_state_dict(state["torch_adamw"])
        with torch.no_grad():
            for p, master, saved in zip(
                self.parameters, self.masters, state["masters"], strict=True
            ):
                if saved.dtype != torch.float32 or saved.shape != master.shape:
                    raise ValueError("head optimizer master precision/shape mismatch")
                master.copy_(saved)
                p.copy_(master)
