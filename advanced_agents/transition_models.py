"""Lightweight differentiable transition models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
from torch import nn


@dataclass
class TransitionOutputs:
    prediction: torch.Tensor
    auxiliary: torch.Tensor


class CombatDynamicsModel(nn.Module):
    """Predicts HP deltas and battle win probabilities."""

    def __init__(self, obs_dim: int, action_dim: int):
        super().__init__()
        self.encoder = nn.GRU(obs_dim + action_dim, 128, batch_first=True)
        self.head = nn.Linear(128, 2)

    def forward(self, battle_features: torch.Tensor, actions: torch.Tensor) -> TransitionOutputs:
        seq = torch.cat([battle_features, actions], dim=-1).unsqueeze(1)
        hidden, _ = self.encoder(seq)
        logits = self.head(hidden.squeeze(1))
        hp_delta = torch.tanh(logits[:, :1])
        win_logit = logits[:, 1:]
        return TransitionOutputs(prediction=hp_delta, auxiliary=win_logit)


class PuzzleGraphTransitionModel(nn.Module):
    """Value-iteration style planner implemented with convolutions."""

    def __init__(self, grid_dim: int = 16, iterations: int = 5):
        super().__init__()
        self.grid_dim = grid_dim
        self.iterations = iterations
        self.reward_conv = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
        self.transition_conv = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)

    def forward(self, grid: torch.Tensor) -> torch.Tensor:
        value = torch.zeros_like(grid)
        reward = self.reward_conv(grid)
        for _ in range(self.iterations):
            q = reward + self.transition_conv(value)
            value = torch.max(value, q)
        return value


class HybridLatentPlanner(nn.Module):
    """Shared latent model with combat and puzzle heads."""

    def __init__(self, latent_dim: int = 128):
        super().__init__()
        self.encoder = nn.Linear(latent_dim, latent_dim)
        self.combat_head = nn.Linear(latent_dim, latent_dim // 2)
        self.puzzle_head = nn.Linear(latent_dim, latent_dim // 2)
        self.decoder = nn.Linear(latent_dim, latent_dim)

    def forward(self, latent: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.tanh(self.encoder(latent))
        combat = self.combat_head(hidden)
        puzzle = self.puzzle_head(hidden)
        recon = self.decoder(hidden)
        return combat, puzzle, recon
