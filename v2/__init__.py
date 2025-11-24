"""
V2 Environment Package for Pokemon Red RL
=========================================

This package contains the improved version 2 environment for training
reinforcement learning agents in Pokemon Red.
"""

from .red_gym_env_v2 import RedGymEnv
from .global_map import local_to_global, GLOBAL_MAP_SHAPE

__all__ = ['RedGymEnv', 'local_to_global', 'GLOBAL_MAP_SHAPE']
