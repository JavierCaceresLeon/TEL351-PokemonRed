"""
Combat-Focused Red Gym Environment
Based on TEL351 red_gym_env_v2.py (PyBoy 2.x compatible) with combat-focused rewards
"""

import uuid
from pathlib import Path
from red_gym_env_v2 import RedGymEnv
import numpy as np

class CombatFocusedEnv(RedGymEnv):
    """
    Modified RedGymEnv that heavily rewards combat performance over exploration.
    """
    
    def __init__(self, config):
        super().__init__(config)
        
        # Combat-specific tracking
        self.battles_won = 0
        self.damage_dealt_total = 0
        self.damage_taken_total = 0
        self.prev_enemy_hp = 0
        self.prev_player_hp = 0
        self.in_battle_steps = 0
        
    def step(self, action):
        obs, base_reward, terminated, truncated, info = super().step(action)
        
        # Override reward with combat-focused calculation
        combat_reward = self._calculate_combat_reward()
        
        # Combine with small portion of base reward (keep some exploration incentive)
        final_reward = combat_reward + (base_reward * 0.1)
        
        return obs, final_reward, terminated, truncated, info
    
    def _calculate_combat_reward(self):
        """
        Calculate reward based on combat performance.
        """
        reward = 0.0
        
        # Check if in battle
        battle_type = self.read_m(0xD057)
        in_battle = battle_type > 0
        
        if in_battle:
            self.in_battle_steps += 1
            
            # Read HP values
            player_hp = self.read_hp(0xD16C)
            enemy_hp = self.read_hp(0xCFE6)
            
            # Damage dealt (positive reward)
            if self.prev_enemy_hp > 0:
                damage_dealt = self.prev_enemy_hp - enemy_hp
                if damage_dealt > 0:
                    self.damage_dealt_total += damage_dealt
                    reward += damage_dealt * 0.5  # Reward per HP of damage
            
            # Damage taken (penalty)
            if self.prev_player_hp > 0:
                damage_taken = self.prev_player_hp - player_hp
                if damage_taken > 0:
                    self.damage_taken_total += damage_taken
                    reward -= damage_taken * 0.3  # Smaller penalty
            
            # Victory bonus
            if enemy_hp == 0 and self.prev_enemy_hp > 0:
                reward += 100.0
                self.battles_won += 1
            
            # Defeat penalty
            if player_hp == 0:
                reward -= 50.0
            
            # Small bonus for being in battle
            reward += 0.05
            
            # Update tracking
            self.prev_enemy_hp = enemy_hp
            self.prev_player_hp = player_hp
            
        else:
            # Small penalty for not being in battle
            reward -= 0.02
            self.prev_enemy_hp = 0
            self.prev_player_hp = 0
        
        return reward
    
    def read_hp(self, addr):
        """Read 2-byte HP value (big-endian)."""
        high = self.read_m(addr)
        low = self.read_m(addr + 1)
        return (high << 8) | low

# Config helper
def make_combat_env_config(state_file, session_path, headless=True):
    """Create config for combat-focused environment."""
    return {
        "headless": headless,
        "save_final_state": True,
        "early_stop": False,
        "action_freq": 24,
        "init_state": state_file,
        "max_steps": 2048,
        "print_rewards": False,  # Too verbose for training
        "save_video": not headless,
        "fast_video": headless,
        "session_path": session_path,
        "gb_path": "PokemonRed.gb",
        "debug": False,
        "sim_frame_dist": 2_000_000.0,
        "use_screen_explore": True,
        "extra_buttons": False,
        "vec_dir": f"vec_bots/{str(uuid.uuid4())[:8]}",
        "frame_stacks": 3,
        "save_state": True,
        "render_mode": "rgb_array" if headless else "human"
    }
