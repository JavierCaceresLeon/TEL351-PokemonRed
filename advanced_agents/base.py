"""Shared utilities for advanced agents."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional

import numpy as np
from gymnasium import Env
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from v2.red_gym_env_v2 import RedGymEnv


@dataclass
class AgentRuntime:
    """Keeps references to the env factory, policy, and checkpoints."""

    env_factory: Callable[[], Env]
    model: PPO
    save_dir: Path


@dataclass
class BaseAgentConfig:
    """Configuration shared across the concrete agents."""

    env_config: Dict[str, Any]
    total_timesteps: int = 2_000_000
    learning_rate: float = 2.5e-4
    gamma: float = 0.999
    gae_lambda: float = 0.95
    clip_range: float = 0.15
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    max_grad_norm: float = 0.5  # Limitar gradientes explosivos
    n_steps: int = 1024
    batch_size: int = 256
    seed: Optional[int] = None
    policy_kwargs: Optional[Dict[str, Any]] = None
    save_path: Path = field(default_factory=lambda: Path("advanced_agents/runs"))
    device: str = "auto"
    tensorboard_log: Optional[str] = None

    def resolve_save_dir(self, run_name: str) -> Path:
        target = self.save_path / run_name
        target.mkdir(parents=True, exist_ok=True)
        return target


class AdvancedAgent:
    """Base class with orchestration utilities."""

    name: str = "advanced_agent"

    def __init__(self, config: BaseAgentConfig):
        self.config = config
        self._model: Optional[PPO] = None

    # ---- hooks that subclasses override ---------------------------------
    def build_wrappers(self) -> Iterable[Callable[[Env], Env]]:
        return []

    def policy_name(self) -> str:
        return "MlpPolicy"

    def policy_kwargs(self) -> Optional[Dict[str, Any]]:
        return self.config.policy_kwargs

    def extra_callbacks(self) -> Iterable[Any]:
        return []

    # ---- high level API --------------------------------------------------
    def make_env(self) -> DummyVecEnv:
        wrappers = list(self.build_wrappers())

        def _factory() -> Env:
            env = RedGymEnv(self.config.env_config)
            for wrapper in wrappers:
                env = wrapper(env)
            return env

        return DummyVecEnv([_factory])

    def make_model(self) -> PPO:
        env = self.make_env()
        model = PPO(
            policy=self.policy_name(),
            env=env,
            learning_rate=self.config.learning_rate,
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
            clip_range=self.config.clip_range,
            vf_coef=self.config.vf_coef,
            ent_coef=self.config.ent_coef,
            max_grad_norm=self.config.max_grad_norm,  # Limitar gradientes
            n_steps=self.config.n_steps,
            batch_size=self.config.batch_size,
            seed=self.config.seed,
            policy_kwargs=self.policy_kwargs(),
            device=self.config.device,
            tensorboard_log=self.config.tensorboard_log,
            verbose=1,
        )
        self._model = model
        return model

    def train(self, run_name: Optional[str] = None) -> AgentRuntime:
        run_name = run_name or self.name
        save_dir = self.config.resolve_save_dir(run_name)
        model = self.make_model()
        callbacks = list(self.extra_callbacks())
        
        # Activar barra de progreso solo si tqdm/rich están disponibles
        try:
            import tqdm
            import rich
            use_progress_bar = True
        except ImportError:
            use_progress_bar = False
            
        model.learn(total_timesteps=self.config.total_timesteps, callback=callbacks, progress_bar=use_progress_bar)
        model.save(str(save_dir / "model"))
        # Only save the env if it has a save method (e.g. VecNormalize)
        env = model.get_env()
        if hasattr(env, "save"):
            env.save(str(save_dir / "vec_env"))
        runtime = AgentRuntime(env_factory=self.make_env, model=model, save_dir=save_dir)
        self._model = model
        return runtime

    def evaluate(self, episodes: int = 5) -> Dict[str, float]:
        env = self.make_env()
        model = self.make_model()
        stats: Dict[str, list] = {}
        for _ in range(episodes):
            obs, _ = env.reset()
            done = False
            truncated = False
            total_reward = 0.0
            while not (done or truncated):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, truncated, _ = env.step(action)
                total_reward += reward
            stats.setdefault("return", []).append(total_reward)
        return {
            "return_mean": float(np.mean(stats["return"])),
            "return_std": float(np.std(stats["return"]))
        }
