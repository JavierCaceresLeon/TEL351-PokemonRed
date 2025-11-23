"""Observation and reward wrappers for the new agents."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
from gymnasium import ObservationWrapper, RewardWrapper, spaces


def _normalize(val: float, denom: float = 1.0) -> float:
    return float(val) / max(denom, 1e-6)


def _read_hp(env, start: int) -> int:
    return 256 * env.read_m(start) + env.read_m(start + 1)


class CombatObservationWrapper(ObservationWrapper):
    """Extends the observation with battle-centric statistics."""

    def __init__(self, env, history_len: int = 4):
        super().__init__(env)
        self.history_len = history_len
        self._battle_dim = 64
        base_spaces = dict(self.env.observation_space.spaces)
        base_spaces["battle_features"] = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self._battle_dim,),
            dtype=np.float32,
        )
        self.observation_space = spaces.Dict(base_spaces)
        self._damage_history = np.zeros((history_len,), dtype=np.float32)

    def observation(self, observation):
        observation["battle_features"] = self._compile_features()
        return observation

    def _compile_features(self) -> np.ndarray:
        party = []
        # party slots hp + status
        status_addrs = [0xD190, 0xD1BC, 0xD1E8, 0xD214, 0xD240, 0xD26C]
        for idx, addr in enumerate([0xD18C, 0xD1B8, 0xD1E4, 0xD210, 0xD23C, 0xD268]):
            cur_hp = _read_hp(self.env, addr)
            max_hp = _read_hp(self.env, addr + 2)
            status = self.env.read_m(status_addrs[idx])
            party.append(_normalize(cur_hp, max_hp))
            party.append(status / 255.0)
        # opponent
        opp_hp = _read_hp(self.env, 0xCFF3)
        opp_max = _read_hp(self.env, 0xCFF5)
        opp_level = self.env.read_m(0xCFE6) / 100.0
        opp_type = self.env.read_m(0xCFE7) / 15.0
        features = np.zeros((self._battle_dim,), dtype=np.float32)
        cursor = 0
        for value in party:
            features[cursor] = value
            cursor += 1
        features[cursor] = _normalize(opp_hp, opp_max)
        cursor += 1
        features[cursor] = opp_level
        cursor += 1
        features[cursor] = opp_type
        cursor += 1
        # damage velocity features
        self._damage_history = np.roll(self._damage_history, 1)
        self._damage_history[0] = party[0]  # proxy: leading slot hp ratio
        features[cursor:cursor + self.history_len] = self._damage_history
        return features


class PuzzleObservationWrapper(ObservationWrapper):
    """Adds puzzle/route planning signals to the observation dict."""

    def __init__(self, env, patch_radius: int = 16):
        super().__init__(env)
        self.patch_radius = patch_radius
        base_spaces = dict(self.env.observation_space.spaces)
        base_spaces["puzzle_features"] = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(patch_radius * patch_radius + 16,),
            dtype=np.float32,
        )
        self.observation_space = spaces.Dict(base_spaces)

    def observation(self, observation):
        observation["puzzle_features"] = self._compile_features()
        return observation

    def _compile_features(self) -> np.ndarray:
        coords = self.env.get_game_coords()
        grid = self.env.get_explore_map()
        center_x = min(max(coords[0], self.patch_radius), grid.shape[0] - self.patch_radius)
        center_y = min(max(coords[1], self.patch_radius), grid.shape[1] - self.patch_radius)
        patch = grid[
            center_x - self.patch_radius : center_x,
            center_y - self.patch_radius : center_y,
        ]
        patch = patch / 255.0
        feature_vec = np.zeros((self.patch_radius * self.patch_radius + 16,), dtype=np.float32)
        flat = patch.flatten()
        feature_vec[: flat.shape[0]] = flat
        # append location cues
        feature_vec[-4] = coords[0] / 255.0
        feature_vec[-3] = coords[1] / 255.0
        feature_vec[-2] = coords[2] / 255.0
        feature_vec[-1] = len(self.env.seen_coords) / 10_000.0
        return feature_vec


@dataclass
class _RewardState:
    last_player_hp: float = 1.0
    last_opp_hp: float = 1.0
    battle_wins: int = 0
    total_items_used: int = 0
    step_count: int = 0


class _CombatShaper:
    def __init__(self, risk_penalty: float = 0.15):
        self.risk_penalty = risk_penalty
        self.state = _RewardState()
        self._loss_history: list[float] = []

    def reset(self):
        self.state = _RewardState()
        self._loss_history.clear()

    def shape(self, env) -> float:
        # Use unwrapped to ensure access to base env methods like read_hp_fraction
        base_env = env.unwrapped
        player_hp = base_env.read_hp_fraction()
        opp_hp = _normalize(_read_hp(base_env, 0xCFF3), _read_hp(base_env, 0xCFF5))
        shaped = 4.0 * (self.state.last_opp_hp - opp_hp)
        shaped += 6.0 * (player_hp - self.state.last_player_hp)
        shaped -= 3.0 * max(0.0, self.state.last_player_hp - player_hp)
        if opp_hp <= 0.0:
            shaped += 25.0
        if player_hp <= 0.0:
            shaped -= 100.0
            self._loss_history.append(self.state.step_count)
        if self._loss_history:
            tail = np.percentile(np.array(self._loss_history, dtype=np.float32), 10)
            shaped -= self.risk_penalty * tail
        self.state.last_player_hp = player_hp
        self.state.last_opp_hp = opp_hp
        self.state.step_count += 1
        return shaped


class _PuzzleShaper:
    def __init__(self, step_penalty: float = -0.05):
        self.step_penalty = step_penalty
        self.prev_badges = 0
        self.prev_coord_count = 0

    def reset(self, env):
        base_env = env.unwrapped
        self.prev_badges = base_env.get_badges()
        self.prev_coord_count = len(base_env.seen_coords)

    def shape(self, env) -> float:
        base_env = env.unwrapped
        steps_bonus = self.step_penalty
        coord_bonus = 0.1 * max(0, len(base_env.seen_coords) - self.prev_coord_count)
        badge_bonus = 40.0 if base_env.get_badges() > self.prev_badges else 0.0
        non_leader_penalty = -25.0 if base_env.read_m(0xD057) != 0 else 0.0
        self.prev_badges = base_env.get_badges()
        self.prev_coord_count = len(base_env.seen_coords)
        return steps_bonus + coord_bonus + badge_bonus + non_leader_penalty


class CombatRewardWrapper(RewardWrapper):
    """Reward shaping specialized for combat excellence."""

    def __init__(self, env, risk_penalty: float = 0.15):
        super().__init__(env)
        self._logic = _CombatShaper(risk_penalty)

    def reset(self, **kwargs):
        self._logic.reset()
        return super().reset(**kwargs)

    def reward(self, reward):
        # Pass the unwrapped environment to the shaper logic
        return reward + self._logic.shape(self.env.unwrapped)


class PuzzleRewardWrapper(RewardWrapper):
    """Reward shaping for puzzle/route solving."""

    def __init__(self, env, step_penalty: float = -0.05):
        super().__init__(env)
        self._logic = _PuzzleShaper(step_penalty)

    def reset(self, **kwargs):
        self._logic.reset(self.env)
        return super().reset(**kwargs)

    def reward(self, reward):
        return reward + self._logic.shape(self.env)


class HybridRewardWrapper(RewardWrapper):
    """Combines combat and puzzle advantages dynamically."""

    def __init__(self, env, beta: Tuple[float, float, float] = (0.5, 0.1, 0.3)):
        super().__init__(env)
        self.beta = beta
        self._combat_logic = _CombatShaper()
        self._puzzle_logic = _PuzzleShaper()

    def reset(self, **kwargs):
        self._combat_logic.reset()
        self._puzzle_logic.reset(self.env)
        return super().reset(**kwargs)

    def reward(self, reward):
        combat_rew = self._combat_logic.shape(self.env)
        puzzle_rew = self._puzzle_logic.shape(self.env)
        badge_count = self.env.get_badges()
        items = self.env.read_m(0xD31E)
        usage_rate = items / 255.0
        bias = self.beta[0] + self.beta[1] * badge_count - self.beta[2] * usage_rate
        w = 1.0 / (1.0 + np.exp(-bias))
        shaped = w * combat_rew + (1 - w) * puzzle_rew - 0.1 * usage_rate
        return reward + shaped
