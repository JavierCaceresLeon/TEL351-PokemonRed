"""
Comprehensive Comparison System: PPO vs Epsilon Greedy
======================================================

This module provides a comprehensive comparison framework between the PPO agent
(from v2) and the new Epsilon Greedy agent with advanced heuristics.
"""

import sys
import time
import json
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import concurrent.futures
import threading

# Add paths for imports
sys.path.append('../v2')
sys.path.append('../gym_scenarios')

try:
    from red_gym_env_v2 import RedGymEnv
    from stream_agent_wrapper import StreamWrapper
    from stable_baselines3 import PPO
except ImportError as e:
    print(f"Warning: Could not import v2 dependencies: {e}")
    print("Make sure you have the v2 environment and stable-baselines3 installed")

from epsilon_greedy_agent import EpsilonGreedyAgent, GameScenario
from gym_metrics import GymMetricsTracker
from gym_memory_addresses import (
    BADGE_COUNT_ADDRESS,
    BAG_ITEM_COUNT,
    BAG_ITEMS_START,
    HP_ADDRESSES,
    MAX_HP_ADDRESSES,
)


@dataclass
class ComparisonMetrics:
    """Structure to hold comparison metrics"""

    agent_name: str
    episode_rewards: List[float]
    episode_lengths: List[int]
    total_steps: int
    total_time: float
    convergence_episodes: int
    exploration_efficiency: float
    scenario_distribution: Dict[str, float]
    action_distribution: Dict[str, float]
    performance_stability: float
    learning_rate: float
    pokemon_lost: float = 0.0
    badge_delta: float = 0.0
    avg_remaining_hp: float = 0.0
    avg_items_used: float = 0.0
    avg_episode_time: float = 0.0


@dataclass
class AgentSpec:
    """Declarative description for a comparable agent."""

    name: str
    runner: Callable[["AgentComparator", "AgentSpec"], ComparisonMetrics]
    params: Dict[str, object] = field(default_factory=dict)


class AgentComparator:
    """
    Comprehensive comparison system for different agents
    """
    
    def __init__(self, 
                 env_config: Dict,
                 comparison_config: Dict = None,
                 save_dir: str = "comparison_results"):
        
        self.env_config = env_config
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        # Default comparison configuration
        default_config = {
            'num_episodes': 10,
            'max_steps_per_episode': 50000,
            'parallel_execution': False,
            'save_detailed_logs': True,
            'create_visualizations': True,
            'metrics_to_compare': [
                'episode_rewards', 'episode_lengths', 'exploration_efficiency',
                'convergence_rate', 'stability', 'scenario_adaptation'
            ]
        }
        
        if comparison_config:
            default_config.update(comparison_config)
        
        self.config = default_config
        
        # Results storage
        self.results = {}
        self.detailed_logs = {}
        
    def run_ppo_agent(self, model_path: Optional[str] = None) -> ComparisonMetrics:
        """
        Run PPO agent and collect metrics
        """
        print("Running PPO Agent...")
        
        if model_path is None:
            print("Warning: No pre-trained model provided. Using random PPO agent.")
        
        # Create environment for PPO
        env = StreamWrapper(
            RedGymEnv(self.env_config),
            stream_metadata={
                "user": "ppo-comparison",
                "env_id": 0,
                "color": "#4477aa",
                "extra": "PPO Comparison Agent",
            }
        )
        
        # Load or create PPO model
        if model_path and Path(model_path).exists():
            print(f"Loading PPO model from {model_path}")
            model = PPO.load(model_path, env=env)
        else:
            print("Creating new PPO model (will be random)")
            model = PPO("MultiInputPolicy", env, verbose=0)
        
        # Run episodes
        episode_rewards = []
        episode_lengths = []
        episode_times = []
        total_steps = 0
        
        scenario_counts = {scenario.value: 0 for scenario in GameScenario}
        action_counts = {f'action_{i}': 0 for i in range(7)}
        
        start_time = time.time()
        
        for episode in range(self.config['num_episodes']):
            print(f"PPO Episode {episode + 1}/{self.config['num_episodes']}")
            
            obs, info = env.reset()
            episode_reward = 0
            episode_length = 0
            episode_start = time.time()
            
            done = False
            while not done and episode_length < self.config['max_steps_per_episode']:
                # PPO action selection
                action, _states = model.predict(obs, deterministic=False)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                episode_reward += reward
                episode_length += 1
                total_steps += 1
                
                # Track actions (simplified)
                action_counts[f'action_{action}'] += 1
            
            episode_time = time.time() - episode_start
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            episode_times.append(episode_time)
            
            print(f"  Reward: {episode_reward:.2f}, Length: {episode_length}")
        
        total_time = time.time() - start_time
        
        # Calculate metrics
        convergence_episodes = self._calculate_convergence(episode_rewards)
        exploration_efficiency = self._estimate_exploration_efficiency(episode_rewards, episode_lengths)
        performance_stability = np.std(episode_rewards) / np.mean(episode_rewards) if np.mean(episode_rewards) > 0 else float('inf')
        learning_rate = self._calculate_learning_rate(episode_rewards)
        
        env.close()
        
        return ComparisonMetrics(
            agent_name="PPO",
            episode_rewards=episode_rewards,
            episode_lengths=episode_lengths,
            total_steps=total_steps,
            total_time=total_time,
            convergence_episodes=convergence_episodes,
            exploration_efficiency=exploration_efficiency,
            scenario_distribution=self._normalize_dict(scenario_counts, total_steps),
            action_distribution=self._normalize_dict(action_counts, total_steps),
            performance_stability=performance_stability,
            learning_rate=learning_rate
        )
    
    def run_epsilon_greedy_agent(self, agent_config: Dict = None) -> ComparisonMetrics:
        """
        Run Epsilon Greedy agent and collect metrics
        """
        print("Running Epsilon Greedy Agent...")
        
        # Default configuration for epsilon greedy
        default_agent_config = {
            'epsilon_start': 0.5,
            'epsilon_min': 0.05,
            'epsilon_decay': 0.995,
            'scenario_detection_enabled': True
        }
        
        if agent_config:
            default_agent_config.update(agent_config)
        
        # Create environment
        env = StreamWrapper(
            RedGymEnv(self.env_config),
            stream_metadata={
                "user": "epsilon-greedy-comparison",
                "env_id": 1,
                "color": "#44aa77",
                "extra": "Epsilon Greedy Comparison Agent",
            }
        )
        
        # Create agent
        agent = EpsilonGreedyAgent(**default_agent_config)
        
        # Run episodes
        episode_rewards = []
        episode_lengths = []
        episode_times = []
        total_steps = 0
        
        scenario_counts = {scenario.value: 0 for scenario in GameScenario}
        action_counts = {f'action_{i}': 0 for i in range(7)}
        
        start_time = time.time()
        
        for episode in range(self.config['num_episodes']):
            print(f"Epsilon Greedy Episode {episode + 1}/{self.config['num_episodes']}")
            
            obs, info = env.reset()
            agent.reset()
            
            episode_reward = 0
            episode_length = 0
            episode_start = time.time()
            
            done = False
            while not done and episode_length < self.config['max_steps_per_episode']:
                # Epsilon Greedy action selection
                action = agent.select_action(obs)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                episode_reward += reward
                episode_length += 1
                total_steps += 1
                
                # Track scenarios and actions
                scenario = agent.detect_scenario(obs)
                scenario_counts[scenario.value] += 1
                action_counts[f'action_{action}'] += 1
                
                # Update agent position tracking
                agent.update_position(obs)
            
            episode_time = time.time() - episode_start
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            episode_times.append(episode_time)
            
            print(f"  Reward: {episode_reward:.2f}, Length: {episode_length}, Epsilon: {agent.epsilon:.3f}")
        
        total_time = time.time() - start_time
        
        # Get final agent metrics
        agent_metrics = agent.get_performance_metrics()
        
        # Calculate metrics
        convergence_episodes = self._calculate_convergence(episode_rewards)
        performance_stability = np.std(episode_rewards) / np.mean(episode_rewards) if np.mean(episode_rewards) > 0 else float('inf')
        learning_rate = self._calculate_learning_rate(episode_rewards)
        
        env.close()
        
        return ComparisonMetrics(
            agent_name="Epsilon_Greedy",
            episode_rewards=episode_rewards,
            episode_lengths=episode_lengths,
            total_steps=total_steps,
            total_time=total_time,
            convergence_episodes=convergence_episodes,
            exploration_efficiency=agent_metrics['exploration_efficiency'],
            scenario_distribution=self._normalize_dict(scenario_counts, total_steps),
            action_distribution=self._normalize_dict(action_counts, total_steps),
            performance_stability=performance_stability,
            learning_rate=learning_rate
        )
    
    def run_comparison(
        self,
        ppo_model_path: Optional[str] = None,
        epsilon_config: Optional[Dict] = None,
        agent_specs: Optional[List[AgentSpec]] = None,
    ) -> Dict[str, ComparisonMetrics]:
        """Run the registered agents and collect metrics."""

        print("Starting Agent Comparison...")
        print(f"Configuration: {self.config}")

        specs = agent_specs or self._build_default_agent_specs(ppo_model_path, epsilon_config)
        if not specs:
            raise ValueError("No agent specifications were provided to AgentComparator")

        results: Dict[str, ComparisonMetrics] = {}

        if self.config['parallel_execution'] and len(specs) > 1:
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(specs)) as executor:
                future_map = {
                    executor.submit(spec.runner, self, spec): spec for spec in specs
                }
                for future in concurrent.futures.as_completed(future_map):
                    spec = future_map[future]
                    results[spec.name] = future.result()
        else:
            for spec in specs:
                print(f"\n▶️  Running agent: {spec.name}")
                results[spec.name] = spec.runner(self, spec)

        self.results = results

        self.generate_comparison_report()
        if self.config['create_visualizations']:
            self.create_visualizations()

        return results

    def _build_default_agent_specs(
        self,
        ppo_model_path: Optional[str],
        epsilon_config: Optional[Dict],
    ) -> List[AgentSpec]:
        """Fallback specs when the caller does not provide custom agents."""

        specs: List[AgentSpec] = []
        specs.append(
            AgentSpec(
                name="PPO",
                runner=lambda comp, spec: comp.run_ppo_agent(ppo_model_path),
            )
        )
        specs.append(
            AgentSpec(
                name="Epsilon_Greedy",
                runner=lambda comp, spec: comp.run_epsilon_greedy_agent(epsilon_config),
            )
        )
        return specs

    # ------------------------------------------------------------------
    # Gym scenario evaluation helpers
    # ------------------------------------------------------------------
    def run_agent_on_gym_scenarios(
        self,
        *,
        agent_name: str,
        scenario_path: Path,
        env_builder: Callable[[Dict[str, Any]], object],
        model_loader: Callable[[object], object],
        headless: bool = True,
        episodes: int = 1,
        deterministic: bool = True,
        metrics_subdir: Optional[Path] = None,
    ) -> ComparisonMetrics:
        """Evaluate an agent across the eight gym scenarios."""

        scenario_path = Path(scenario_path)
        scenarios = self._load_gym_scenarios(scenario_path)
        if not scenarios:
            raise ValueError(f"No scenarios found in {scenario_path}")

        metrics_subdir = metrics_subdir or self.save_dir / "gym_metrics" / agent_name
        metrics_subdir = self._ensure_dir(metrics_subdir)

        per_episode_records: List[Dict[str, float]] = []
        action_counter: Dict[str, int] = defaultdict(int)
        scenario_counter: Dict[str, int] = defaultdict(int)

        for scenario in scenarios:
            for phase in scenario.get("phases", []):
                state_file: Path = phase.get("_state_file")
                if state_file is None:
                    continue
                if not state_file.exists():
                    print(f"⚠️  Missing state file for {scenario['id']}::{phase['name']} -> {state_file}")
                    continue

                phase_key = f"{scenario['id']}::{phase['name']}"
                scenario_counter[phase_key] += 1

                session_dir = self._ensure_dir(
                    self.save_dir / "gym_sessions" / agent_name / scenario['id'] / phase['name']
                )
                env_config = self._phase_env_config(
                    init_state=state_file,
                    session_dir=session_dir,
                    max_steps=phase.get("_max_steps", self.env_config.get("max_steps", 2000)),
                    headless=headless,
                )

                vec_env = env_builder(env_config)
                model = model_loader(vec_env)

                phase_metrics = self._rollout_scenario_phase(
                    vec_env=vec_env,
                    model=model,
                    scenario=scenario,
                    phase=phase,
                    agent_name=agent_name,
                    deterministic=deterministic,
                    episodes=episodes,
                    metrics_dir=self._ensure_dir(metrics_subdir / scenario['id'] / phase['name']),
                )

                vec_env.close() if hasattr(vec_env, "close") else None

                per_episode_records.extend(phase_metrics["episodes"])
                for action_id, count in phase_metrics["action_counts"].items():
                    action_counter[str(action_id)] += count

        if not per_episode_records:
            raise RuntimeError("Gym scenario evaluation did not produce any episodes")

        episode_rewards = [record["reward"] for record in per_episode_records]
        episode_lengths = [record["length"] for record in per_episode_records]
        total_steps = int(sum(episode_lengths))
        total_time = float(sum(record["duration"] for record in per_episode_records))

        pokemon_lost = float(np.mean([record["pokemon_lost"] for record in per_episode_records]))
        badge_delta = float(np.mean([record["badge_delta"] for record in per_episode_records]))
        avg_hp = float(np.mean([record["avg_hp"] for record in per_episode_records]))
        avg_items = float(np.mean([record["items_used"] for record in per_episode_records]))
        avg_episode_time = float(np.mean([record["duration"] for record in per_episode_records]))

        total_unique_tiles = [record.get("unique_tiles", 0) for record in per_episode_records]
        exploration_eff = float(np.mean([
            tiles / max(length, 1)
            for tiles, length in zip(total_unique_tiles, episode_lengths)
        ]))

        convergence = self._calculate_convergence(episode_rewards)
        stability = (
            np.std(episode_rewards) / np.mean(episode_rewards)
            if np.mean(episode_rewards) > 0 else float('inf')
        )
        learning_rate = self._calculate_learning_rate(episode_rewards)

        scenario_distribution = self._normalize_dict(
            scenario_counter,
            max(sum(scenario_counter.values()), 1)
        )
        action_distribution = self._normalize_dict(action_counter, max(total_steps, 1))

        return ComparisonMetrics(
            agent_name=agent_name,
            episode_rewards=episode_rewards,
            episode_lengths=episode_lengths,
            total_steps=total_steps,
            total_time=total_time,
            convergence_episodes=convergence,
            exploration_efficiency=exploration_eff,
            scenario_distribution=scenario_distribution,
            action_distribution=action_distribution,
            performance_stability=stability,
            learning_rate=learning_rate,
            pokemon_lost=pokemon_lost,
            badge_delta=badge_delta,
            avg_remaining_hp=avg_hp,
            avg_items_used=avg_items,
            avg_episode_time=avg_episode_time,
        )

    def _load_gym_scenarios(self, scenario_path: Path) -> List[Dict[str, Any]]:
        with scenario_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        default_steps = payload.get("default_max_steps", self.env_config.get("max_steps", 2000))
        scenarios = payload.get("scenarios", [])
        for scenario in scenarios:
            for phase in scenario.get("phases", []):
                state_file = Path(phase.get("state_file", ""))
                if not state_file.is_absolute():
                    state_file = scenario_path.parent / state_file
                phase["_state_file"] = state_file
                phase["_max_steps"] = phase.get("max_steps", default_steps)
        return scenarios

    def _phase_env_config(self, init_state: Path, session_dir: Path, max_steps: int, headless: bool) -> Dict[str, Any]:
        cfg = dict(self.env_config)
        cfg["init_state"] = str(init_state)
        cfg["session_path"] = session_dir
        cfg["max_steps"] = max_steps
        cfg["headless"] = headless
        cfg.setdefault("save_final_state", False)
        cfg.setdefault("print_rewards", False)
        return cfg

    def _ensure_dir(self, path: Path) -> Path:
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _rollout_scenario_phase(
        self,
        *,
        vec_env,
        model,
        scenario: Dict[str, Any],
        phase: Dict[str, Any],
        agent_name: str,
        deterministic: bool,
        episodes: int,
        metrics_dir: Path,
    ) -> Dict[str, Any]:
        phase_key = f"{scenario['id']}::{phase['name']}"
        episode_summaries: List[Dict[str, float]] = []
        action_counts: Dict[int, int] = defaultdict(int)

        for episode in range(episodes):
            tracker = GymMetricsTracker(
                gym_number=scenario.get("badge_bit", episode + 1),
                agent_name=agent_name,
                gym_name=scenario.get("leader", f"Scenario {scenario['id']}")
            )
            tracker.metadata["scenario_id"] = scenario["id"]
            tracker.metadata["phase"] = phase["name"]
            tracker.start()

            obs = vec_env.reset()
            done = np.array([False])
            steps = 0
            start_time = time.time()

            core_env = self._extract_core_env(vec_env)
            start_badges = self._read_badges(core_env)
            start_inventory = self._read_bag_inventory(core_env)
            tracker.metadata["badge_start"] = start_badges
            tracker.record_badges(start_badges)

            tracker.record_team_state(self._read_team_hp(core_env))

            max_steps = phase.get("_max_steps", self.env_config.get("max_steps", 2000))

            while steps < max_steps and not bool(done[0]):
                action, _states = model.predict(obs, deterministic=deterministic)
                obs, reward, done, info = vec_env.step(action)
                core_env = self._extract_core_env(vec_env)
                game_state = self._capture_game_state(core_env)
                scalar_action = self._scalar_action(action)
                tracker.record_step(scalar_action, float(self._scalar_value(reward)), game_state)
                action_counts[scalar_action] += 1
                steps += 1

            duration = time.time() - start_time
            end_badges = self._read_badges(core_env)
            tracker.metadata["badge_end"] = end_badges
            tracker.record_team_state(self._read_team_hp(core_env), is_final=True)
            tracker.record_badges(end_badges, is_final=True)
            tracker.end(success=self._phase_goal_met(phase, start_badges, end_badges))

            bag_usage = self._compute_items_used(start_inventory, self._read_bag_inventory(core_env))
            tracker.items_used.extend(
                {"item": str(item_id), "used": amount} for item_id, amount in bag_usage.items()
            )

            if metrics_dir:
                tracker.save_metrics(output_dir=metrics_dir)

            summary = tracker.get_summary_stats()
            summary["badge_delta"] = end_badges - start_badges
            summary["items_used"] = sum(bag_usage.values())
            summary["duration"] = duration
            summary["unique_tiles"] = summary.get("unique_tiles_explored", 0)

            episode_summaries.append(
                {
                    "reward": summary["total_reward"],
                    "length": summary["total_steps"],
                    "duration": summary["total_duration_seconds"],
                    "pokemon_lost": summary["pokemon_fainted_player"],
                    "badge_delta": summary["badge_delta"],
                    "avg_hp": summary["final_avg_hp"],
                    "items_used": summary["items_used"],
                    "unique_tiles": summary["unique_tiles"],
                }
            )

        return {
            "phase_key": phase_key,
            "episodes": episode_summaries,
            "action_counts": action_counts,
        }

    def _extract_core_env(self, env):
        if hasattr(env, "envs"):
            candidate = env.envs[0]
        else:
            candidate = env
        visited = set()
        while hasattr(candidate, "env") and id(candidate) not in visited:
            visited.add(id(candidate))
            candidate = candidate.env
        return candidate

    def _capture_game_state(self, env) -> Dict[str, Any]:
        return {
            "x": int(env.read_m(0xD362)),
            "y": int(env.read_m(0xD361)),
            "map": int(env.read_m(0xD35E)),
            "hp": self._read_team_hp(env),
            "in_battle": bool(env.read_m(0xD057)),
            "badges": self._read_badges(env),
        }

    def _read_team_hp(self, env) -> List[int]:
        values: List[int] = []
        for addr in HP_ADDRESSES:
            hp_value = env.read_hp(addr)
            if hp_value > 0:
                values.append(hp_value)
        if not values:
            # fallback to aggregate fraction
            values.append(int(env.read_hp_fraction() * 100))
        return values

    def _read_badges(self, env) -> int:
        return int(env.read_m(BADGE_COUNT_ADDRESS))

    def _read_bag_inventory(self, env) -> Dict[int, int]:
        count = env.read_m(BAG_ITEM_COUNT)
        inventory: Dict[int, int] = {}
        cursor = BAG_ITEMS_START
        for _ in range(count):
            item_id = env.read_m(cursor)
            quantity = env.read_m(cursor + 1)
            if item_id == 0:
                break
            inventory[item_id] = quantity
            cursor += 2
        return inventory

    def _compute_items_used(self, start_inv: Dict[int, int], end_inv: Dict[int, int]) -> Dict[int, int]:
        usage: Dict[int, int] = {}
        for item_id, start_qty in start_inv.items():
            end_qty = end_inv.get(item_id, 0)
            diff = start_qty - end_qty
            if diff > 0:
                usage[item_id] = diff
        return usage

    def _phase_goal_met(self, phase: Dict[str, Any], start_badges: int, end_badges: int) -> bool:
        goal = phase.get("goal", {})
        if goal.get("type") == "badge":
            badge_bit = goal.get("badge_bit")
            if badge_bit is None:
                return False
            mask = 1 << badge_bit
            return bool(end_badges & mask) or (end_badges != start_badges)
        # Manual phases cannot be auto-evaluated; assume completion if we survived full episode
        return True

    def _scalar_action(self, action) -> int:
        if isinstance(action, np.ndarray):
            if action.ndim == 0:
                return int(action)
            if action.size == 1:
                return int(action.flatten()[0])
        return int(action)

    def _scalar_value(self, value) -> float:
        if isinstance(value, np.ndarray):
            if value.ndim == 0:
                return float(value)
            if value.size == 1:
                return float(value.flatten()[0])
        return float(value)
    
    def _calculate_convergence(self, rewards: List[float], window_size: int = 5) -> int:
        """
        Calculate number of episodes needed for convergence
        """
        if len(rewards) < window_size * 2:
            return len(rewards)
        
        # Calculate moving average
        moving_avg = []
        for i in range(window_size, len(rewards)):
            avg = np.mean(rewards[i-window_size:i])
            moving_avg.append(avg)
        
        # Find where moving average stabilizes (low variance)
        for i in range(window_size, len(moving_avg)):
            window_variance = np.var(moving_avg[i-window_size:i])
            if window_variance < np.var(rewards) * 0.1:  # 10% of total variance
                return i + window_size
        
        return len(rewards)
    
    def _estimate_exploration_efficiency(self, rewards: List[float], lengths: List[int]) -> float:
        """
        Estimate exploration efficiency based on reward progression
        """
        if not rewards:
            return 0.0
        
        # Simple metric: how quickly rewards improve
        max_reward = max(rewards)
        first_quarter_max = max(rewards[:len(rewards)//4]) if len(rewards) >= 4 else max(rewards)
        
        if max_reward == 0:
            return 0.0
        
        return first_quarter_max / max_reward
    
    def _calculate_learning_rate(self, rewards: List[float]) -> float:
        """
        Calculate learning rate as slope of reward progression
        """
        if len(rewards) < 2:
            return 0.0
        
        x = np.arange(len(rewards))
        slope, _ = np.polyfit(x, rewards, 1)
        return slope
    
    def _normalize_dict(self, d: Dict, total: int) -> Dict[str, float]:
        """
        Normalize dictionary values to proportions
        """
        if total == 0:
            return {k: 0.0 for k in d.keys()}
        return {k: v/total for k, v in d.items()}
    
    def generate_comparison_report(self):
        """
        Generate comprehensive comparison report
        """
        report = {
            'timestamp': datetime.now().isoformat(),
            'configuration': self.config,
            'environment_config': self.env_config,
            'results_summary': {},
            'detailed_comparison': {}
        }
        
        # Summary statistics
        for agent_name, metrics in self.results.items():
            report['results_summary'][agent_name] = {
                'mean_reward': np.mean(metrics.episode_rewards),
                'std_reward': np.std(metrics.episode_rewards),
                'max_reward': np.max(metrics.episode_rewards),
                'mean_length': np.mean(metrics.episode_lengths),
                'total_steps': metrics.total_steps,
                'total_time': metrics.total_time,
                'steps_per_second': metrics.total_steps / metrics.total_time,
                'convergence_episodes': metrics.convergence_episodes,
                'exploration_efficiency': metrics.exploration_efficiency,
                'performance_stability': metrics.performance_stability,
                'learning_rate': metrics.learning_rate,
                'pokemon_lost': metrics.pokemon_lost,
                'badge_delta': metrics.badge_delta,
                'avg_remaining_hp': metrics.avg_remaining_hp,
                'avg_items_used': metrics.avg_items_used,
                'avg_episode_time': metrics.avg_episode_time,
            }
        
        # Detailed comparison
        if 'PPO' in self.results and 'Epsilon_Greedy' in self.results:
            ppo = self.results['PPO']
            eg = self.results['Epsilon_Greedy']
            
            report['detailed_comparison'] = {
                'reward_advantage': {
                    'ppo_vs_epsilon_greedy': np.mean(ppo.episode_rewards) - np.mean(eg.episode_rewards),
                    'winner': 'PPO' if np.mean(ppo.episode_rewards) > np.mean(eg.episode_rewards) else 'Epsilon_Greedy'
                },
                'efficiency_advantage': {
                    'ppo_steps_per_second': ppo.total_steps / ppo.total_time,
                    'epsilon_greedy_steps_per_second': eg.total_steps / eg.total_time,
                    'winner': 'PPO' if (ppo.total_steps / ppo.total_time) > (eg.total_steps / eg.total_time) else 'Epsilon_Greedy'
                },
                'stability_comparison': {
                    'ppo_stability': ppo.performance_stability,
                    'epsilon_greedy_stability': eg.performance_stability,
                    'winner': 'PPO' if ppo.performance_stability < eg.performance_stability else 'Epsilon_Greedy'
                },
                'convergence_comparison': {
                    'ppo_convergence': ppo.convergence_episodes,
                    'epsilon_greedy_convergence': eg.convergence_episodes,
                    'winner': 'PPO' if ppo.convergence_episodes < eg.convergence_episodes else 'Epsilon_Greedy'
                },
                'exploration_comparison': {
                    'ppo_exploration': ppo.exploration_efficiency,
                    'epsilon_greedy_exploration': eg.exploration_efficiency,
                    'winner': 'PPO' if ppo.exploration_efficiency > eg.exploration_efficiency else 'Epsilon_Greedy'
                }
            }
        
        # Save report
        report_path = self.save_dir / f"comparison_report_{int(time.time())}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"Comparison report saved to {report_path}")
        
        # Print summary to console
        self.print_summary_report(report)
    
    def print_summary_report(self, report: Dict):
        """
        Print summary report to console
        """
        print("\n" + "="*80)
        print("AGENT COMPARISON SUMMARY")
        print("="*80)
        
        for agent_name, stats in report['results_summary'].items():
            print(f"\n{agent_name} Agent:")
            print(f"  Mean Reward: {stats['mean_reward']:.2f} ± {stats['std_reward']:.2f}")
            print(f"  Max Reward: {stats['max_reward']:.2f}")
            print(f"  Mean Episode Length: {stats['mean_length']:.1f}")
            print(f"  Steps per Second: {stats['steps_per_second']:.1f}")
            print(f"  Convergence Episodes: {stats['convergence_episodes']}")
            print(f"  Exploration Efficiency: {stats['exploration_efficiency']:.3f}")
            print(f"  Performance Stability: {stats['performance_stability']:.3f}")
            print(f"  Learning Rate: {stats['learning_rate']:.4f}")
            if 'pokemon_lost' in stats:
                print(f"  Avg Pokémon Lost: {stats['pokemon_lost']:.2f}")
                print(f"  Avg Badge Delta: {stats['badge_delta']:.2f}")
                print(f"  Avg Remaining HP: {stats['avg_remaining_hp']:.2f}")
                print(f"  Avg Items Used: {stats['avg_items_used']:.2f}")
                print(f"  Avg Episode Time (s): {stats['avg_episode_time']:.2f}")
        
        if 'detailed_comparison' in report:
            print(f"\nCOMPARISON WINNERS:")
            comp = report['detailed_comparison']
            print(f"  Reward: {comp['reward_advantage']['winner']}")
            print(f"  Efficiency: {comp['efficiency_advantage']['winner']}")
            print(f"  Stability: {comp['stability_comparison']['winner']}")
            print(f"  Convergence: {comp['convergence_comparison']['winner']}")
            print(f"  Exploration: {comp['exploration_comparison']['winner']}")
    
    def create_visualizations(self):
        """
        Create comparison visualizations
        """
        if not self.results:
            print("No results to visualize")
            return
        
        # Set up the plotting
        plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
        
        # 1. Episode Rewards Comparison
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Agent Comparison: PPO vs Epsilon Greedy', fontsize=16)
        
        # Plot 1: Episode rewards over time
        ax1 = axes[0, 0]
        for agent_name, metrics in self.results.items():
            ax1.plot(metrics.episode_rewards, label=agent_name, marker='o', alpha=0.7)
        ax1.set_title('Episode Rewards Over Time')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Reward')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Episode lengths
        ax2 = axes[0, 1]
        for agent_name, metrics in self.results.items():
            ax2.plot(metrics.episode_lengths, label=agent_name, marker='s', alpha=0.7)
        ax2.set_title('Episode Lengths Over Time')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Steps')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Reward distribution
        ax3 = axes[1, 0]
        reward_data = [metrics.episode_rewards for metrics in self.results.values()]
        agent_names = list(self.results.keys())
        ax3.boxplot(reward_data, labels=agent_names)
        ax3.set_title('Reward Distribution')
        ax3.set_ylabel('Reward')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Performance metrics comparison
        ax4 = axes[1, 1]
        metrics_names = ['Exploration Efficiency', 'Performance Stability', 'Learning Rate']
        x_pos = np.arange(len(metrics_names))
        
        for i, (agent_name, metrics) in enumerate(self.results.items()):
            values = [
                metrics.exploration_efficiency,
                1.0 / (1.0 + metrics.performance_stability),  # Inverse for better visualization
                max(0, metrics.learning_rate * 1000)  # Scale for visualization
            ]
            ax4.bar(x_pos + i*0.35, values, 0.35, label=agent_name, alpha=0.7)
        
        ax4.set_title('Performance Metrics Comparison')
        ax4.set_xlabel('Metrics')
        ax4.set_ylabel('Score')
        ax4.set_xticks(x_pos + 0.175)
        ax4.set_xticklabels(metrics_names, rotation=45)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save visualization
        viz_path = self.save_dir / f"comparison_visualization_{int(time.time())}.png"
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {viz_path}")
        
        plt.show()


def main():
    """
    Main function to run the comparison
    """
    # Environment configuration
    env_config = {
        'headless': True,
        'save_final_state': False,
        'early_stop': False,
        'action_freq': 24,
        'init_state': '../init.state',
        'max_steps': 40960,  # Reasonable episode length
        'print_rewards': False,
        'save_video': False,
        'fast_video': True,
        'session_path': Path('comparison_session'),
        'gb_path': '../PokemonRed.gb',
        'debug': False,
        'reward_scale': 0.5,
        'explore_weight': 0.25
    }
    
    # Comparison configuration
    comparison_config = {
        'num_episodes': 5,  # Start with fewer episodes for testing
        'max_steps_per_episode': 40960,
        'parallel_execution': False,
        'save_detailed_logs': True,
        'create_visualizations': True
    }
    
    # Epsilon Greedy agent configuration
    epsilon_config = {
        'epsilon_start': 0.6,
        'epsilon_min': 0.05,
        'epsilon_decay': 0.995,
        'scenario_detection_enabled': True
    }
    
    # Create comparator
    comparator = AgentComparator(
        env_config=env_config,
        comparison_config=comparison_config,
        save_dir="comparison_results"
    )
    
    # Run comparison
    print("Starting comprehensive agent comparison...")
    
    try:
        results = comparator.run_comparison(
            ppo_model_path=None,  # Will use random PPO if no model provided
            epsilon_config=epsilon_config
        )
        
        print("\nComparison completed successfully!")
        
    except Exception as e:
        print(f"Error during comparison: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()