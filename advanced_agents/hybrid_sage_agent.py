"""Hybrid agent that balances combat and puzzle solving."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Iterable, Optional

import torch
import torch.nn.functional as F

from .base import AdvancedAgent, BaseAgentConfig
from .callbacks import AuxiliaryDynamicsCallback
from .features import HybridFeatureExtractor
from .transition_models import HybridLatentPlanner
from .wrappers import (
    CombatObservationWrapper,
    PuzzleObservationWrapper,
    HybridRewardWrapper,
)


@dataclass
class HybridAgentConfig(BaseAgentConfig):
    latent_dim: int = 128
    aux_lr: float = 1e-4
    item_penalty_beta: tuple[float, float, float] = (0.6, 0.2, 0.4)


class HybridSageAgent(AdvancedAgent):
    name = "hybrid_sage"

    def __init__(self, config: HybridAgentConfig):
        super().__init__(config)
        self.planner = HybridLatentPlanner(latent_dim=config.latent_dim)
        self.planner_optimizer = torch.optim.Adam(self.planner.parameters(), lr=config.aux_lr)

    def build_wrappers(self) -> Iterable:
        return [
            CombatObservationWrapper,
            PuzzleObservationWrapper,
            partial(HybridRewardWrapper, beta=self.config.item_penalty_beta),
        ]

    def policy_kwargs(self):
        return {
            "features_extractor_class": HybridFeatureExtractor,
            "features_extractor_kwargs": {"embed_dim": self.config.latent_dim},
            "net_arch": dict(pi=[512, 256], vf=[512, 256]),
        }

    def extra_callbacks(self) -> Iterable:
        return [AuxiliaryDynamicsCallback(self.planner_optimizer, self._hybrid_loss)]

    def _hybrid_loss(self, locals_, model) -> Optional[torch.Tensor]:
        buffer = getattr(model, "rollout_buffer", None)
        if buffer is None or buffer.pos < 2:
            return None
        self.planner.to(model.device)
        battle = torch.as_tensor(buffer.observations["battle_features"][: buffer.pos, 0], device=model.device, dtype=torch.float32)
        puzzle = torch.as_tensor(buffer.observations["puzzle_features"][: buffer.pos, 0], device=model.device, dtype=torch.float32)
        half = self.config.latent_dim // 2
        battle_latent = F.pad(battle, (0, max(0, half - battle.shape[-1])))[:, :half]
        puzzle_latent = F.pad(puzzle, (0, max(0, half - puzzle.shape[-1])))[:, :half]
        latent = torch.cat([battle_latent, puzzle_latent], dim=-1)
        combat_head, puzzle_head, recon = self.planner(latent)
        combat_loss = F.mse_loss(combat_head, battle_latent)
        puzzle_loss = F.mse_loss(puzzle_head, puzzle_latent)
        recon_loss = F.mse_loss(recon, latent)
        return combat_loss + puzzle_loss + 0.1 * recon_loss
