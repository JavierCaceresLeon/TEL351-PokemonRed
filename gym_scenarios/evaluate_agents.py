#!/usr/bin/env python3
"""Run PPO agents across the eight gym scenarios and log performance metrics."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from stable_baselines3 import PPO

from baselines.memory_addresses import (
    BADGE_COUNT_ADDRESS,
    MAP_N_ADDRESS,
    X_POS_ADDRESS,
    Y_POS_ADDRESS,
)
from v2.red_gym_env_v2 import RedGymEnv


@dataclass
class PhaseGoal:
    goal_type: str
    badge_bit: Optional[int] = None
    map_id: Optional[int] = None
    x_min: Optional[int] = None
    x_max: Optional[int] = None
    y_min: Optional[int] = None
    y_max: Optional[int] = None
    note: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PhaseGoal":
        if data is None:
            return cls(goal_type="none")
        goal_type = data.get("type", "none")
        return cls(
            goal_type=goal_type,
            badge_bit=data.get("badge_bit"),
            map_id=data.get("map_id"),
            x_min=data.get("x", {}).get("min"),
            x_max=data.get("x", {}).get("max"),
            y_min=data.get("y", {}).get("min"),
            y_max=data.get("y", {}).get("max"),
            note=data.get("note"),
        )

    def evaluate(self, env: RedGymEnv, start_badge_byte: int) -> Optional[bool]:
        if self.goal_type in {"none", "manual"}:
            return None
        if self.goal_type == "badge":
            if self.badge_bit is None:
                return None
            mask = 1 << self.badge_bit
            start_flag = bool(start_badge_byte & mask)
            end_flag = bool(env.read_m(BADGE_COUNT_ADDRESS) & mask)
            if start_flag and not end_flag:
                return False
            return end_flag
        if self.goal_type == "coordinate":
            if self.map_id is None:
                return None
            cur_map = env.read_m(MAP_N_ADDRESS)
            if cur_map != self.map_id:
                return False
            x_pos = env.read_m(X_POS_ADDRESS)
            y_pos = env.read_m(Y_POS_ADDRESS)
            in_x = True
            in_y = True
            if self.x_min is not None and x_pos < self.x_min:
                in_x = False
            if self.x_max is not None and x_pos > self.x_max:
                in_x = False
            if self.y_min is not None and y_pos < self.y_min:
                in_y = False
            if self.y_max is not None and y_pos > self.y_max:
                in_y = False
            return in_x and in_y
        return None


@dataclass
class PhaseSpec:
    name: str
    description: str
    state_file: Path
    goal: PhaseGoal
    max_steps: int

    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Any],
        state_dir: Path,
        default_steps: int,
    ) -> "PhaseSpec":
        goal = PhaseGoal.from_dict(data.get("goal"))
        state_file = state_dir / Path(data["state_file"]).name
        return cls(
            name=data["name"],
            description=data.get("description", ""),
            state_file=state_file,
            goal=goal,
            max_steps=data.get("max_steps", default_steps),
        )


@dataclass
class ScenarioSpec:
    scenario_id: str
    leader: str
    badge_bit: int
    map_id: int
    phases: List[PhaseSpec]

    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Any],
        state_dir: Path,
        default_steps: int,
    ) -> "ScenarioSpec":
        phases = [
            PhaseSpec.from_dict(phase, state_dir, default_steps)
            for phase in data.get("phases", [])
        ]
        return cls(
            scenario_id=data["id"],
            leader=data["leader"],
            badge_bit=data["badge_bit"],
            map_id=data["map_id"],
            phases=phases,
        )


def load_scenarios(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    state_dir = Path(payload.get("state_directory", "gym_scenarios/state_files"))
    default_steps = int(payload.get("default_max_steps", 1800))
    scenarios = [
        ScenarioSpec.from_dict(entry, state_dir, default_steps)
        for entry in payload["scenarios"]
    ]
    return {
        "state_dir": state_dir,
        "default_steps": default_steps,
        "scenarios": scenarios,
    }


def build_env_config(
    base_config: Dict[str, Any],
    init_state: Path,
    session_dir: Path,
    max_steps: int,
) -> Dict[str, Any]:
    cfg = base_config.copy()
    cfg["init_state"] = str(init_state)
    cfg["session_path"] = session_dir
    cfg["max_steps"] = max_steps
    return cfg


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def summarize_metrics(metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not metrics:
        return {}
    avg_reward = float(np.mean([m["reward"] for m in metrics]))
    avg_steps = float(np.mean([m["steps"] for m in metrics]))
    duration = float(np.mean([m["duration_sec"] for m in metrics]))
    success_flags = [m["success"] for m in metrics if m["success"] is not None]
    if success_flags:
        success_rate = float(np.mean([1.0 if flag else 0.0 for flag in success_flags]))
    else:
        success_rate = None
    return {
        "episodes": len(metrics),
        "avg_reward": avg_reward,
        "avg_steps": avg_steps,
        "avg_duration_sec": duration,
        "success_rate": success_rate,
    }


def evaluate_phase(
    scenario: ScenarioSpec,
    phase: PhaseSpec,
    model: PPO,
    agent_name: str,
    base_config: Dict[str, Any],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    results: Dict[str, Any] = {
        "scenario": scenario.scenario_id,
        "phase": phase.name,
        "agent": agent_name,
        "state_file": str(phase.state_file),
        "goal_type": phase.goal.goal_type,
        "episodes": [],
        "warnings": [],
    }

    if not phase.state_file.exists():
        results["warnings"].append(
            f"Missing state file: {phase.state_file}. Skipping phase."
        )
        return results

    session_dir = ensure_dir(
        Path(args.session_root)
        / scenario.scenario_id
        / phase.name
        / agent_name
    )

    env_config = build_env_config(
        base_config,
        phase.state_file,
        session_dir,
        phase.max_steps,
    )

    env = RedGymEnv(env_config)
    try:
        for episode in range(args.episodes):
            obs, _ = env.reset(seed=args.seed + episode)
            start_badges = env.read_m(BADGE_COUNT_ADDRESS)
            metrics = {
                "episode": episode,
                "steps": 0,
                "reward": 0.0,
                "success": None,
                "duration_sec": 0.0,
                "final_badges": None,
                "final_coords": None,
            }
            start_time = time.time()
            for step in range(phase.max_steps):
                action, _ = model.predict(obs, deterministic=args.deterministic)
                obs, reward, terminated, truncated, _ = env.step(action)
                metrics["reward"] += float(reward)
                metrics["steps"] = step + 1
                if terminated or truncated:
                    break
            metrics["duration_sec"] = time.time() - start_time
            metrics["final_badges"] = int(env.read_m(BADGE_COUNT_ADDRESS))
            metrics["final_coords"] = {
                "x": int(env.read_m(X_POS_ADDRESS)),
                "y": int(env.read_m(Y_POS_ADDRESS)),
                "map": int(env.read_m(MAP_N_ADDRESS)),
            }
            metrics["success"] = phase.goal.evaluate(env, start_badges)
            results["episodes"].append(metrics)
    finally:
        env.pyboy.stop(save=False)

    results["summary"] = summarize_metrics(results["episodes"])
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare PPO agents over defined gym scenarios.",
    )
    parser.add_argument("--scenarios", default="gym_scenarios/scenarios.json")
    parser.add_argument("--baseline", required=True, help="Path to baseline PPO .zip")
    parser.add_argument("--improved", required=True, help="Path to improved PPO .zip")
    parser.add_argument("--gb-path", default="PokemonRed.gb")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--session-root", default="gym_scenarios/results")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--windowed",
        action="store_true",
        help="Render the PyBoy window (default runs headless).",
    )
    args = parser.parse_args()

    scenario_payload = load_scenarios(Path(args.scenarios))
    scenarios: List[ScenarioSpec] = scenario_payload["scenarios"]

    base_env_config = {
        "headless": not args.windowed,
        "save_final_state": False,
        "print_rewards": False,
        "action_freq": 24,
        "max_steps": scenario_payload["default_steps"],
        "save_video": False,
        "fast_video": False,
        "session_path": ensure_dir(Path(args.session_root) / "_scratch"),
        "gb_path": args.gb_path,
        "reward_scale": 0.5,
        "explore_weight": 0.25,
    }

    print("Loading baseline model...")
    baseline_model = PPO.load(args.baseline)
    print("Loading improved model...")
    improved_model = PPO.load(args.improved)

    timestamp_dir = ensure_dir(
        Path(args.session_root) / time.strftime("%Y%m%d_%H%M%S")
    )

    all_results: List[Dict[str, Any]] = []

    for scenario in scenarios:
        print(f"Evaluating scenario {scenario.scenario_id} ({scenario.leader})")
        for phase in scenario.phases:
            for agent_name, model in (
                ("baseline", baseline_model),
                ("improved", improved_model),
            ):
                result = evaluate_phase(
                    scenario,
                    phase,
                    model,
                    agent_name,
                    base_env_config,
                    args,
                )
                all_results.append(result)
                out_dir = ensure_dir(
                    timestamp_dir / scenario.scenario_id / phase.name
                )
                out_file = out_dir / f"{agent_name}.json"
                with out_file.open("w", encoding="utf-8") as handle:
                    json.dump(result, handle, indent=2)
                summary = result.get("summary") or {}
                avg_reward = summary.get("avg_reward")
                avg_steps = summary.get("avg_steps")
                success_rate = summary.get("success_rate")
                print(
                    f"  {phase.name} | {agent_name}: steps={avg_steps} "
                    f"reward={avg_reward} success={success_rate}"
                )

    aggregated = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "results": all_results,
    }
    with (timestamp_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(aggregated, handle, indent=2)


if __name__ == "__main__":
    main()
