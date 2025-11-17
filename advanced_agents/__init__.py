"""Factory helpers for the advanced gym agents."""

from .combat_apex_agent import CombatApexAgent, CombatAgentConfig
from .puzzle_speed_agent import PuzzleSpeedAgent, PuzzleAgentConfig
from .hybrid_sage_agent import HybridSageAgent, HybridAgentConfig
from .train_agents import (
    train_combat_apex,
    train_puzzle_speed,
    train_hybrid_sage,
)

__all__ = [
    "CombatApexAgent",
    "CombatAgentConfig",
    "PuzzleSpeedAgent",
    "PuzzleAgentConfig",
    "HybridSageAgent",
    "HybridAgentConfig",
    "train_combat_apex",
    "train_puzzle_speed",
    "train_hybrid_sage",
]
