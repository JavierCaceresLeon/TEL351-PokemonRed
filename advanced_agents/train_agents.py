"""Convenience helpers to train each advanced agent."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

from .combat_apex_agent import CombatApexAgent, CombatAgentConfig
from .puzzle_speed_agent import PuzzleSpeedAgent, PuzzleAgentConfig
from .hybrid_sage_agent import HybridSageAgent, HybridAgentConfig


def _base_env_config(overrides: Optional[Dict] = None) -> Dict:
    cfg = {
        "session_path": Path("advanced_agents/sessions"),
        "save_final_state": False,
        "print_rewards": False,
        "headless": True,
        "init_state": "init.state",
        "action_freq": 24,
        "max_steps": 2048,
        "save_video": False,
        "fast_video": True,
        "gb_path": "PokemonRed.gb",
        "reward_scale": 1.0,
        "explore_weight": 0.25,
        "save_final_state": False,
        "save_video": False,
    }
    overrides = overrides or {}
    cfg.update(overrides)
    session_path = Path(cfg["session_path"])
    session_path.mkdir(parents=True, exist_ok=True)
    cfg["session_path"] = session_path
    return cfg


def train_combat_apex(total_timesteps: int = 2_000_000, env_overrides: Optional[Dict] = None, **agent_kwargs):
    config = CombatAgentConfig(env_config=_base_env_config(env_overrides), total_timesteps=total_timesteps, **agent_kwargs)
    agent = CombatApexAgent(config)
    return agent.train()


def train_puzzle_speed(total_timesteps: int = 2_000_000, env_overrides: Optional[Dict] = None, **agent_kwargs):
    config = PuzzleAgentConfig(env_config=_base_env_config(env_overrides), total_timesteps=total_timesteps, **agent_kwargs)
    agent = PuzzleSpeedAgent(config)
    return agent.train()


def train_hybrid_sage(total_timesteps: int = 2_000_000, env_overrides: Optional[Dict] = None, **agent_kwargs):
    config = HybridAgentConfig(env_config=_base_env_config(env_overrides), total_timesteps=total_timesteps, **agent_kwargs)
    agent = HybridSageAgent(config)
    return agent.train()
