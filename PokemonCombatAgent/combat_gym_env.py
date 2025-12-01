"""
Combat-Specialized Gym Environment for Pokemon Red

Based on PokemonRedExperiments/baselines/red_gym_env.py
but with reward function focused on battle performance.

Key Differences from Original:
- Rewards emphasize battle victories and HP conservation
- Tracks combat-specific metrics (damage dealt, type advantages)
- Penalizes wasteful resource usage (potions when high HP)
- Bonus for efficient battles (quick victories, low damage taken)
"""

import sys
import uuid 
import os
from math import floor, sqrt
import json
from pathlib import Path

import numpy as np
from einops import rearrange
import matplotlib.pyplot as plt
from skimage.transform import resize
from pyboy import PyBoy
import mediapy as media
import pandas as pd

from gymnasium import Env, spaces
from pyboy.utils import WindowEvent
from memory_addresses import *


class CombatGymEnv(Env):
    """
    Gymnasium environment specialized for Pokemon Red combat scenarios.
    
    Inherits the proven architecture from PokemonRedExperiments but modifies
    the reward function to prioritize battle performance.
    """

    def __init__(self, config=None):
        # ===== Base Configuration (from original) =====
        self.debug = config.get('debug', False)
        self.s_path = Path(config['session_path'])
        self.save_final_state = config.get('save_final_state', True)
        self.print_rewards = config.get('print_rewards', True)
        self.headless = config.get('headless', True)
        self.init_state = config['init_state']
        self.act_freq = config.get('action_freq', 24)
        self.max_steps = config.get('max_steps', 2048 * 8)
        self.early_stopping = config.get('early_stop', False)
        self.save_video = config.get('save_video', False)
        self.fast_video = config.get('fast_video', True)
        self.reward_scale = config.get('reward_scale', 1.0)
        self.instance_id = str(uuid.uuid4())[:8] if 'instance_id' not in config else config['instance_id']
        
        self.s_path.mkdir(exist_ok=True)
        self.reset_count = 0
        self.all_runs = []
        
        # ===== Combat-Specific Configuration =====
        self.combat_focus = config.get('combat_focus', True)  # Enable combat rewards
        self.type_bonus_scale = config.get('type_bonus_scale', 20.0)  # Reward for type advantages
        self.hp_efficiency_scale = config.get('hp_efficiency_scale', 50.0)  # Reward for HP conservation
        
        # ===== Observation/Action Spaces (same as original) =====
        self.valid_actions = [
            WindowEvent.PRESS_ARROW_DOWN,
            WindowEvent.PRESS_ARROW_LEFT,
            WindowEvent.PRESS_ARROW_RIGHT,
            WindowEvent.PRESS_ARROW_UP,
            WindowEvent.PRESS_BUTTON_A,
            WindowEvent.PRESS_BUTTON_B,
            WindowEvent.PRESS_BUTTON_START,  # Para menú
        ]
        
        self.release_arrow = [
            WindowEvent.RELEASE_ARROW_DOWN,
            WindowEvent.RELEASE_ARROW_LEFT,
            WindowEvent.RELEASE_ARROW_RIGHT,
            WindowEvent.RELEASE_ARROW_UP
        ]
        
        self.release_button = [
            WindowEvent.RELEASE_BUTTON_A,
            WindowEvent.RELEASE_BUTTON_B,
            WindowEvent.RELEASE_BUTTON_START,
        ]
        
        self.frame_stacks = 3
        self.output_shape = (36, 40, 3)
        self.mem_padding = 2
        self.memory_height = 8
        self.col_steps = 16
        self.output_full = (
            self.output_shape[0] * self.frame_stacks + 2 * (self.mem_padding + self.memory_height),
            self.output_shape[1],
            self.output_shape[2]
        )
        
        self.action_space = spaces.Discrete(len(self.valid_actions))
        self.observation_space = spaces.Box(low=0, high=255, shape=self.output_full, dtype=np.uint8)
        
        # ===== PyBoy Initialization =====
        # Use 'null' instead of 'headless' for PyBoy 2.6+
        window_type = 'null' if self.headless else 'SDL2'
        self.pyboy = PyBoy(
            config['gb_path'],
            window=window_type,
            sound=False,  # Disable sound to prevent buffer overrun errors
            sound_emulated=False,  # Completely disable sound emulation
        )
        
        # PyBoy 2.6+: screen property gives us access to screen buffer
        self.screen = self.pyboy.screen
        
        # Force initial screen update to initialize buffer properly
        self.pyboy.tick()
        
        if not self.headless:
            self.pyboy.set_emulation_speed(6)
        
        # ===== Combat Tracking Variables =====
        self.battle_count = 0
        self.battles_won = 0
        self.battles_lost = 0
        self.total_damage_dealt = 0
        self.total_damage_received = 0
        self.efficient_kills = 0  # Kills without losing Pokemon
        self.wasteful_heals = 0    # Heals when HP > 80%
        
        # Battle state tracking
        self.in_battle = False
        self.battle_start_hp = 0
        self.battle_start_enemy_hp = 0
        self.prev_enemy_hp = 0
        self.prev_party_hp = 0
        
        self.reset()

    def reset(self, seed=None, options=None):
        """Reset environment to initial state"""
        self.seed = seed
        
        # Load initial game state
        with open(self.init_state, "rb") as f:
            self.pyboy.load_state(f)
        
        # Initialize frame stacking
        self.recent_frames = np.zeros(
            (self.frame_stacks, self.output_shape[0], 
             self.output_shape[1], self.output_shape[2]),
            dtype=np.uint8)
        
        self.recent_memory = np.zeros((self.output_shape[1]*self.memory_height, 3), dtype=np.uint8)
        
        # Initialize stats tracking
        self.agent_stats = []
        
        # Combat metrics reset
        self.last_health = 1.0
        self.total_healing_rew = 0
        self.died_count = 0
        self.party_size = 0
        self.step_count = 0
        self.max_opponent_level = 0
        self.max_event_rew = 0
        self.max_level_rew = 0
        
        # Combat-specific resets
        self.in_battle = False
        self.battle_start_hp = 0
        self.prev_enemy_hp = 0
        self.prev_party_hp = 0
        
        # Initialize rewards
        self.progress_reward = self.get_game_state_reward()
        self.total_reward = sum([val for _, val in self.progress_reward.items()])
        
        self.reset_count += 1
        return self.render(), {}

    def step(self, action):
        """Execute one environment step"""
        
        # Execute action on emulator
        self.run_action_on_emulator(action)
        
        # Update observation
        self.recent_frames = np.roll(self.recent_frames, 1, axis=0)
        obs_memory = self.render()
        
        # Update combat state
        self.update_combat_state()
        
        # Track stats
        self.append_agent_stats(action)
        
        # Update healing rewards
        self.update_heal_reward()
        self.party_size = self.read_m(PARTY_SIZE_ADDRESS)
        
        # Calculate reward
        new_reward, new_prog = self.update_reward()
        
        self.last_health = self.read_hp_fraction()
        
        # Update memory visualization
        self.recent_memory = np.roll(self.recent_memory, 3)
        self.recent_memory[0, 0] = min(new_prog[0] * 64, 255)
        self.recent_memory[0, 1] = min(new_prog[1] * 64, 255)
        self.recent_memory[0, 2] = min(new_prog[2] * 128, 255)
        
        # Check if episode done
        step_limit_reached = self.check_if_done()
        
        # Save info if needed
        self.save_and_print_info(step_limit_reached, obs_memory)
        
        self.step_count += 1
        
        return obs_memory, new_reward * 0.1, False, step_limit_reached, {}

    def run_action_on_emulator(self, action):
        """Execute action on PyBoy emulator"""
        self.pyboy.send_input(self.valid_actions[action])
        
        for i in range(self.act_freq):
            # Release action after 8 frames (prevents stuck buttons)
            if i == 8:
                if action < 4:  # Arrow keys
                    self.pyboy.send_input(self.release_arrow[action])
                elif action < 7:  # A, B, Start buttons
                    self.pyboy.send_input(self.release_button[action - 4])
            
            self.pyboy.tick()

    def render(self, reduce_res=True, add_memory=True, update_mem=True):
        """Render current game screen with memory visualization"""
        # PyBoy 2.6+: use screen.ndarray to get (144, 160, 3) RGB array
        game_pixels_render = np.asarray(self.screen.ndarray, dtype=np.uint8)
        
        if reduce_res:
            game_pixels_render = (255 * resize(game_pixels_render, self.output_shape)).astype(np.uint8)
            if update_mem:
                self.recent_frames[0] = game_pixels_render
            if add_memory:
                pad = np.zeros(shape=(self.mem_padding, self.output_shape[1], 3), dtype=np.uint8)
                game_pixels_render = np.concatenate(
                    (
                        self.create_exploration_memory(),
                        pad,
                        self.create_recent_memory(),
                        pad,
                        rearrange(self.recent_frames, 'f h w c -> (f h) w c')
                    ),
                    axis=0)
        return game_pixels_render

    def create_exploration_memory(self):
        """Create visual memory of progress (level, HP, exploration)"""
        w = self.output_shape[1]
        h = self.memory_height
        
        def make_reward_channel(r_val):
            col_steps = self.col_steps
            max_r_val = (w-1) * h * col_steps
            r_val = min(r_val, max_r_val)
            row = floor(r_val / (h * col_steps))
            memory = np.zeros(shape=(h, w), dtype=np.uint8)
            memory[:, :row] = 255
            row_covered = row * h * col_steps
            col = floor((r_val - row_covered) / col_steps)
            memory[:col, row] = 255
            col_covered = col * col_steps
            last_pixel = floor(r_val - row_covered - col_covered) 
            memory[col, row] = last_pixel * (255 // col_steps)
            return memory
        
        level, hp, battles = self.group_rewards()
        full_memory = np.stack((
            make_reward_channel(level),
            make_reward_channel(hp),
            make_reward_channel(battles)
        ), axis=-1)
        
        # Highlight if badges obtained
        if self.get_badges() > 0:
            full_memory[:, -1, :] = 255
        
        return full_memory

    def create_recent_memory(self):
        """Create visualization of recent rewards"""
        return rearrange(
            self.recent_memory, 
            '(w h) c -> h w c', 
            h=self.memory_height)

    def update_combat_state(self):
        """
        Track battle state and compute combat-specific rewards.
        
        This is KEY for combat specialization.
        """
        # Check if we're in battle
        current_in_battle = self.read_m(IN_BATTLE_ADDRESS) > 0
        
        # Battle start detection
        if current_in_battle and not self.in_battle:
            self.in_battle = True
            self.battle_count += 1
            self.battle_start_hp = self.read_hp_fraction()
            self.prev_party_hp = self.battle_start_hp
            self.prev_enemy_hp = 1.0  # Enemy always starts at full HP
            self.battle_start_enemy_hp = 1.0
        
        # Battle end detection
        elif not current_in_battle and self.in_battle:
            self.in_battle = False
            
            # Determine battle outcome
            current_hp = self.read_hp_fraction()
            
            if current_hp > 0:
                # Victory!
                self.battles_won += 1
                
                # Check if it was an efficient kill (no Pokemon fainted)
                if self.died_count == 0 or current_hp > 0.8 * self.battle_start_hp:
                    self.efficient_kills += 1
            else:
                # Defeat
                self.battles_lost += 1
        
        # Track damage during battle
        if self.in_battle:
            current_party_hp = self.read_hp_fraction()
            
            # Damage received
            if current_party_hp < self.prev_party_hp:
                damage_received = self.prev_party_hp - current_party_hp
                self.total_damage_received += damage_received
            
            self.prev_party_hp = current_party_hp

    def get_game_state_reward(self, print_stats=False):
        """
        Compute reward based on game state.
        
        MODIFIED for combat focus:
        - Higher weights on battle victories
        - Bonus for HP efficiency
        - Penalty for wasteful healing
        - Reward for damage dealt
        """
        
        # Base rewards (from original)
        base_rewards = {
            'event': self.reward_scale * self.update_max_event_rew(),
            'level': self.reward_scale * self.get_levels_reward(),
            'heal': self.reward_scale * self.total_healing_rew * 10,
            'badge': self.reward_scale * self.get_badges() * 10,
            'dead': self.reward_scale * -5.0 * self.died_count,  # Increased penalty
        }
        
        # Combat-specific rewards
        combat_rewards = {}
        
        if self.combat_focus:
            # Victory bonus
            combat_rewards['victories'] = self.reward_scale * self.battles_won * 100.0
            
            # Efficient kill bonus
            combat_rewards['efficient'] = self.reward_scale * self.efficient_kills * 50.0
            
            # HP conservation bonus (only count after battles)
            if self.battle_count > 0:
                hp_efficiency = self.read_hp_fraction()
                combat_rewards['hp_conserve'] = self.reward_scale * self.hp_efficiency_scale * hp_efficiency
            
            # Penalty for wasteful heals
            combat_rewards['waste_heal'] = self.reward_scale * -10.0 * self.wasteful_heals
            
            # Opponent level bonus (fighting stronger opponents)
            combat_rewards['op_lvl'] = self.reward_scale * self.update_max_op_level() * 2.0
        
        # Merge rewards
        state_scores = {**base_rewards, **combat_rewards}
        
        return state_scores

    def group_rewards(self):
        """Group rewards for memory visualization"""
        prog = self.progress_reward
        
        # Modified to show: level, HP, battles won
        return (
            prog.get('level', 0) * 100 / self.reward_scale, 
            self.read_hp_fraction() * 2000,
            prog.get('victories', 0) * 10 / self.reward_scale
        )

    def update_reward(self):
        """Calculate reward delta"""
        old_prog = self.group_rewards()
        self.progress_reward = self.get_game_state_reward()
        new_prog = self.group_rewards()
        
        new_total = sum([val for _, val in self.progress_reward.items()])
        new_step = new_total - self.total_reward
        
        self.total_reward = new_total
        
        return (new_step, 
                (new_prog[0] - old_prog[0], 
                 new_prog[1] - old_prog[1], 
                 new_prog[2] - old_prog[2]))

    def check_if_done(self):
        """Check if episode should terminate"""
        if self.early_stopping:
            done = False
            if self.step_count > 128 and self.recent_memory.sum() < (255 * 1):
                done = True
        else:
            done = self.step_count >= self.max_steps
        
        return done

    def append_agent_stats(self, action):
        """Track agent statistics for analysis"""
        x_pos = self.read_m(X_POS_ADDRESS)
        y_pos = self.read_m(Y_POS_ADDRESS)
        map_n = self.read_m(MAP_N_ADDRESS)
        levels = [self.read_m(a) for a in LEVELS_ADDRESSES]
        
        self.agent_stats.append({
            'step': self.step_count,
            'x': x_pos,
            'y': y_pos,
            'map': map_n,
            'last_action': action,
            'pcount': self.read_m(PARTY_SIZE_ADDRESS),
            'levels': levels,
            'levels_sum': sum(levels),
            'hp': self.read_hp_fraction(),
            'deaths': self.died_count,
            'badge': self.get_badges(),
            'battles_won': self.battles_won,
            'battles_lost': self.battles_lost,
            'efficient_kills': self.efficient_kills,
            'in_battle': self.in_battle,
        })

    def save_and_print_info(self, done, obs_memory):
        """Print progress and save info"""
        if self.print_rewards:
            prog_string = f'step: {self.step_count:6d}'
            for key, val in self.progress_reward.items():
                prog_string += f' {key}: {val:5.2f}'
            prog_string += f' sum: {self.total_reward:5.2f}'
            prog_string += f' W/L: {self.battles_won}/{self.battles_lost}'
            print(f'\r{prog_string}', end='', flush=True)
        
        if self.step_count % 50 == 0:
            plt.imsave(
                self.s_path / Path(f'curframe_{self.instance_id}.jpeg'), 
                self.render(reduce_res=False))
        
        if self.print_rewards and done:
            print('', flush=True)
            if self.save_final_state:
                fs_path = self.s_path / Path('final_states')
                fs_path.mkdir(exist_ok=True)
                plt.imsave(
                    fs_path / Path(f'frame_r{self.total_reward:.4f}_{self.reset_count}.jpeg'), 
                    obs_memory)
        
        if done:
            self.all_runs.append(self.progress_reward)
            with open(self.s_path / Path(f'all_runs_{self.instance_id}.json'), 'w') as f:
                json.dump(self.all_runs, f)
            pd.DataFrame(self.agent_stats).to_csv(
                self.s_path / Path(f'agent_stats_{self.instance_id}.csv.gz'), 
                compression='gzip', mode='a')

    # ===== Helper Methods (from original) =====
    
    def read_m(self, addr):
        """Read memory at address"""
        return self.pyboy.memory[addr]
    
    def read_bit(self, addr, bit: int) -> bool:
        """Read specific bit from memory address"""
        return bin(256 + self.read_m(addr))[-bit-1] == '1'
    
    def get_levels_sum(self):
        """Sum of all Pokemon levels in party"""
        poke_levels = [max(self.read_m(a) - 2, 0) for a in LEVELS_ADDRESSES]
        return max(sum(poke_levels) - 4, 0)
    
    def get_levels_reward(self):
        """Reward for leveling up"""
        level_sum = self.get_levels_sum()
        self.max_level_rew = max(self.max_level_rew, level_sum * 0.5)
        return self.max_level_rew
    
    def get_badges(self):
        """Count number of gym badges"""
        return self.bit_count(self.read_m(BADGE_COUNT_ADDRESS))
    
    def update_heal_reward(self):
        """Track healing rewards"""
        cur_health = self.read_hp_fraction()
        
        if (cur_health > self.last_health and 
                self.read_m(PARTY_SIZE_ADDRESS) == self.party_size):
            if self.last_health > 0:
                heal_amount = cur_health - self.last_health
                
                # Check if heal was wasteful (HP > 80%)
                if self.last_health > 0.8:
                    self.wasteful_heals += 1
                else:
                    # Reward efficient healing
                    self.total_healing_rew += heal_amount * heal_amount
            else:
                self.died_count += 1
    
    def update_max_op_level(self):
        """Track max opponent level faced"""
        opponent_level = max([self.read_m(a) for a in OPPONENT_LEVELS_ADDRESSES]) - 5
        self.max_opponent_level = max(self.max_opponent_level, opponent_level)
        return self.max_opponent_level * 0.2
    
    def update_max_event_rew(self):
        """Track event flags progression"""
        cur_rew = self.get_all_events_reward()
        self.max_event_rew = max(cur_rew, self.max_event_rew)
        return self.max_event_rew
    
    def get_all_events_reward(self):
        """Count all event flags set"""
        event_flags_start = EVENT_FLAGS_START_ADDRESS
        event_flags_end = EVENT_FLAGS_END_ADDRESS
        base_event_flags = 13
        
        return max(
            sum([self.bit_count(self.read_m(i)) 
                 for i in range(event_flags_start, event_flags_end)]) - base_event_flags,
            0
        )
    
    def read_hp_fraction(self):
        """Read current HP as fraction of max HP"""
        hp_sum = sum([self.read_hp(add) for add in HP_ADDRESSES])
        max_hp_sum = sum([self.read_hp(add) for add in MAX_HP_ADDRESSES])
        max_hp_sum = max(max_hp_sum, 1)
        return hp_sum / max_hp_sum
    
    def read_hp(self, start):
        """Read 2-byte HP value"""
        return 256 * self.read_m(start) + self.read_m(start + 1)
    
    def bit_count(self, bits):
        """Count number of 1 bits"""
        return bin(bits).count('1')
