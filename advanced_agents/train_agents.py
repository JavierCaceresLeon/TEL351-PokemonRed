"""Convenience helpers to train each advanced agent."""

from __future__ import annotations

import argparse
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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an advanced Pokemon Red agent")
    parser.add_argument(
        "--agent",
        choices=["combat", "puzzle", "hybrid"],
        default="combat",
        help="Which advanced agent to train",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=2_000_000,
        help="Total timesteps to run PPO training for",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional custom name for the checkpoint directory",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Torch device passed to Stable Baselines (auto, cpu, cuda, etc.)",
    )
    headless_group = parser.add_mutually_exclusive_group()
    headless_group.add_argument(
        "--headless",
        dest="headless",
        action="store_true",
        default=True,
        help="Run PyBoy headless (default)",
    )
    headless_group.add_argument(
        "--no-headless",
        dest="headless",
        action="store_false",
        help="Display the SDL2 Game Boy window while training",
    )
    parser.add_argument(
        "--session-dir",
        type=Path,
        default=Path("advanced_agents/sessions"),
        help="Directory where emulator sessions and logs are stored",
    )
    return parser.parse_args()


def _run_from_cli(args: argparse.Namespace) -> None:
    env_overrides: Dict[str, object] = {
        "headless": args.headless,
        "session_path": args.session_dir,
    }
    agent_kwargs = {"device": args.device}
    agent_map = {
        "combat": train_combat_apex,
        "puzzle": train_puzzle_speed,
        "hybrid": train_hybrid_sage,
    }
    train_fn = agent_map[args.agent]
    runtime = train_fn(total_timesteps=args.timesteps, env_overrides=env_overrides, run_name=args.run_name, **agent_kwargs)
    print(
        f"\n✅ Training finished for {args.agent} agent."
        f"\n   Checkpoints saved under: {runtime.save_dir}"
    )


if __name__ == "__main__":
    _run_from_cli(_parse_args())
