"""Custom Stable-Baselines3 feature extractors."""

from __future__ import annotations

from typing import Dict

import torch
from torch import nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


def _make_cnn(in_channels: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, 32, kernel_size=3, stride=2),
        nn.ReLU(),
        nn.Conv2d(32, 64, kernel_size=3, stride=2),
        nn.ReLU(),
        nn.Conv2d(64, 64, kernel_size=3, stride=2),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d((4, 4)),
        nn.Flatten(),
        nn.Linear(64 * 4 * 4, out_dim),
        nn.ReLU(),
    )


class CombatFeatureExtractor(BaseFeaturesExtractor):
    """Transformer-style encoder for combat stats."""

    def __init__(self, observation_space: spaces.Dict, embed_dim: int = 128):
        super().__init__(observation_space, features_dim=embed_dim)
        screen_shape = observation_space["screens"].shape
        
        # Robust channel detection: assume channels is the smallest dimension (usually 1, 3, or 4)
        # This handles both (C, H, W) and (H, W, C)
        self.in_channels = min(screen_shape)
        
        self.cnn = _make_cnn(self.in_channels, embed_dim)
        self.battle_proj = nn.Linear(observation_space["battle_features"].shape[0], embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=4, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.out = nn.Linear(embed_dim * 2, embed_dim)

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        screens = observations["screens"].float()
        
        # Robust permutation: ensure (N, C, H, W)
        # If the second dimension (dim 1) matches in_channels, it's likely already CHW
        if screens.shape[1] != self.in_channels and screens.shape[-1] == self.in_channels:
            # HWC -> CHW
            screens = screens.permute(0, 3, 1, 2)
            
        screens = screens / 255.0
        z_screens = self.cnn(screens)
        battle = self.battle_proj(observations["battle_features"].float())
        seq = torch.stack([z_screens, battle], dim=1)
        fused = self.transformer(seq)[:, 0, :]
        feat = torch.cat([fused, battle], dim=-1)
        return self.out(feat)


class PuzzleFeatureExtractor(BaseFeaturesExtractor):
    """Graph-inspired encoder for puzzle state."""

    def __init__(self, observation_space: spaces.Dict, embed_dim: int = 128):
        super().__init__(observation_space, features_dim=embed_dim)
        patch_dim = observation_space["puzzle_features"].shape[0]
        self.patch_proj = nn.Sequential(
            nn.Linear(patch_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )
        self.coord_proj = nn.Linear(3, embed_dim)
        self.out = nn.Linear(embed_dim * 2, embed_dim)

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        patch = self.patch_proj(observations["puzzle_features"].float())
        coords = torch.stack(
            [observations["map"].float().mean(dim=[1, 2, 3]), observations["health"].float().squeeze(-1), observations["level"].float().mean(dim=-1)],
            dim=-1,
        )
        coord_embed = self.coord_proj(coords)
        return self.out(torch.cat([patch, coord_embed], dim=-1))


class HybridFeatureExtractor(BaseFeaturesExtractor):
    """Cross-attention fusion between combat and puzzle features."""

    def __init__(self, observation_space: spaces.Dict, embed_dim: int = 192):
        super().__init__(observation_space, features_dim=embed_dim)
        self.combat = CombatFeatureExtractor(observation_space, embed_dim=embed_dim // 2)
        self.puzzle = PuzzleFeatureExtractor(observation_space, embed_dim=embed_dim // 2)
        self.cross_attn = nn.MultiheadAttention(embed_dim=embed_dim // 2, num_heads=2, batch_first=True)
        self.out = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        combat_feat = self.combat(observations)
        puzzle_feat = self.puzzle(observations)
        query = combat_feat.unsqueeze(1)
        key_value = puzzle_feat.unsqueeze(1)
        attn, _ = self.cross_attn(query, key_value, key_value)
        fused = torch.cat([attn.squeeze(1), combat_feat, puzzle_feat], dim=-1)
        return self.out(fused)
