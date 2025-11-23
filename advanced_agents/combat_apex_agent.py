"""Combat-first agent implementation."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Iterable, Optional

import torch
import torch.nn.functional as F

from gymnasium import Env

from .base import AdvancedAgent, BaseAgentConfig
from .callbacks import AuxiliaryDynamicsCallback
from .features import CombatFeatureExtractor
from .transition_models import CombatDynamicsModel
from .wrappers import CombatObservationWrapper, CombatRewardWrapper


@dataclass
class CombatAgentConfig(BaseAgentConfig):
    risk_penalty: float = 0.2
    history_len: int = 6
    aux_lr: float = 1e-4


class CombatApexAgent(AdvancedAgent):
    name = "combat_apex"

    def __init__(self, config: CombatAgentConfig):
        super().__init__(config)
        obs_dim = 64
        action_dim = self.config.env_config.get("action_dim", 7)
        self.dynamics = CombatDynamicsModel(obs_dim, action_dim)
        self.dynamics_optimizer = torch.optim.Adam(self.dynamics.parameters(), lr=config.aux_lr)

    def build_wrappers(self) -> Iterable:
        return [
            partial(CombatObservationWrapper, history_len=self.config.history_len),
            partial(CombatRewardWrapper, risk_penalty=self.config.risk_penalty),
        ]

    def policy_name(self) -> str:
        return "MlpPolicy"

    def policy_kwargs(self):
        return {
            "features_extractor_class": CombatFeatureExtractor,
            "features_extractor_kwargs": {"embed_dim": 160},
            "net_arch": dict(pi=[256, 256], vf=[256, 256]),
        }

    def extra_callbacks(self) -> Iterable:
        return [AuxiliaryDynamicsCallback(self.dynamics_optimizer, self._combat_loss)]

    def _combat_loss(self, locals_, model) -> Optional[torch.Tensor]:
        buffer = getattr(model, "rollout_buffer", None)
        if buffer is None or buffer.pos < 2:
            return None
        self.dynamics.to(model.device)
        obs = buffer.observations["battle_features"]
        obs_tensor = torch.as_tensor(obs[: buffer.pos, 0], device=model.device, dtype=torch.float32)
        actions = torch.as_tensor(buffer.actions[: buffer.pos, 0], device=model.device)
        
        # DEBUG: Print shapes
        # print(f"DEBUG: obs_tensor shape: {obs_tensor.shape}")
        # print(f"DEBUG: actions raw shape: {actions.shape}")

        # Squeeze the last dimension if it exists (N, 1) -> (N,)
        if actions.dim() > 1:
            actions = actions.squeeze(-1)
        
        action_oh = F.one_hot(actions.long(), num_classes=model.action_space.n).float()
        
        # DEBUG: Print shapes after processing
        # print(f"DEBUG: action_oh shape: {action_oh.shape}")
        
        # Ensure obs_tensor is 2D (N, F)
        if obs_tensor.dim() > 2:
             obs_tensor = obs_tensor.flatten(start_dim=1)
        
        outputs = self.dynamics(obs_tensor, action_oh)
        target = torch.roll(obs_tensor[:, 0], shifts=-1)
        target = target[:-1]
        pred = outputs.prediction.squeeze(-1)[:-1]
        mse = F.mse_loss(pred, target)
        win_targets = torch.zeros_like(pred)
        win_loss = F.binary_cross_entropy_with_logits(outputs.auxiliary[:-1], win_targets.unsqueeze(-1))
        return mse + 0.2 * win_loss
