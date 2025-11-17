"""Puzzle-speed specialist agent."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Iterable, Optional

import torch
import torch.nn.functional as F

from .base import AdvancedAgent, BaseAgentConfig
from .callbacks import AuxiliaryDynamicsCallback
from .features import PuzzleFeatureExtractor
from .transition_models import PuzzleGraphTransitionModel
from .wrappers import PuzzleObservationWrapper, PuzzleRewardWrapper


@dataclass
class PuzzleAgentConfig(BaseAgentConfig):
    patch_radius: int = 16
    graph_iterations: int = 5
    aux_lr: float = 5e-5
    step_penalty: float = -0.05


class PuzzleSpeedAgent(AdvancedAgent):
    name = "puzzle_speed"

    def __init__(self, config: PuzzleAgentConfig):
        super().__init__(config)
        self.graph_model = PuzzleGraphTransitionModel(grid_dim=config.patch_radius, iterations=config.graph_iterations)
        self.graph_optimizer = torch.optim.Adam(self.graph_model.parameters(), lr=config.aux_lr)

    def build_wrappers(self) -> Iterable:
        return [
            partial(PuzzleObservationWrapper, patch_radius=self.config.patch_radius),
            partial(PuzzleRewardWrapper, step_penalty=self.config.step_penalty),
        ]

    def policy_kwargs(self):
        return {
            "features_extractor_class": PuzzleFeatureExtractor,
            "features_extractor_kwargs": {"embed_dim": 192},
            "net_arch": dict(pi=[256, 256], vf=[256, 256]),
        }

    def extra_callbacks(self) -> Iterable:
        return [AuxiliaryDynamicsCallback(self.graph_optimizer, self._puzzle_loss)]

    def _puzzle_loss(self, locals_, model) -> Optional[torch.Tensor]:
        buffer = getattr(model, "rollout_buffer", None)
        if buffer is None or buffer.pos < 2:
            return None
        self.graph_model.to(model.device)
        puzzles = buffer.observations["puzzle_features"]
        patch_size = self.config.patch_radius * self.config.patch_radius
        obs_tensor = torch.as_tensor(puzzles[: buffer.pos, 0, :patch_size], device=model.device, dtype=torch.float32)
        grids = obs_tensor.view(-1, 1, self.config.patch_radius, self.config.patch_radius)
        values = self.graph_model(grids).view(grids.shape[0], -1).max(dim=-1).values
        deltas = values[:-1] - values[1:]
        loss = F.relu(-deltas).mean()
        return loss
