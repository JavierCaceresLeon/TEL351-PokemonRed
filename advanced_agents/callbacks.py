"""Stable-Baselines callbacks used by the advanced agents."""

from __future__ import annotations

from typing import Callable, Optional

import torch
from stable_baselines3.common.callbacks import BaseCallback


class AuxiliaryDynamicsCallback(BaseCallback):
    """Optimizes a provided loss alongside PPO updates."""

    def __init__(self, optimizer: torch.optim.Optimizer, loss_fn: Callable[[dict, torch.nn.Module], Optional[torch.Tensor]]):
        super().__init__()
        self.optimizer = optimizer
        self.loss_fn = loss_fn

    def _on_step(self) -> bool:
        loss = self.loss_fn(self.locals, self.model)
        if loss is None:
            return True
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.policy.parameters(), 1.0)
        self.optimizer.step()
        return True
