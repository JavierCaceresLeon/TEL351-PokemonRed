"""
Battle-Only Pokemon Red Environment
====================================

Specialized Gymnasium environment that ONLY trains on combat scenarios.
- Loads random gym leader battle states
- Heavily penalizes non-combat actions
- Rewards combat victories and efficient strategies
- Resets to new battle after win/loss
"""

import numpy as np
from pathlib import Path
import random
from pyboy import PyBoy
import gymnasium as gym
from skimage.transform import resize
from gymnasium import spaces


# Combat detection memory addresses (Game Boy memory map)
BATTLE_TYPE_ADDR = 0xD057  # 0=no battle, 1=wild, 2=trainer
IN_BATTLE_ADDR = 0xD05A    # 1=in battle menu
PLAYER_HP_ADDR = 0xD16C     # Current HP (2 bytes)
PLAYER_MAX_HP_ADDR = 0xD18D # Max HP (2 bytes)
ENEMY_HP_ADDR = 0xCFE6      # Enemy HP (2 bytes)
ENEMY_MAX_HP_ADDR = 0xCFF4  # Enemy max HP (2 bytes)
BADGES_ADDR = 0xD356        # Badge bits
PLAYER_LEVEL_ADDR = 0xD18C  # Active Pokemon level


class BattleOnlyEnv(gym.Env):
    """
    Environment that exclusively trains on gym leader battles.
    """
    
    def __init__(self, 
                 rom_path='PokemonRed.gb',
                 battle_states_dir='battle_states',
                 output_shape=(128, 40),
                 max_steps=2000,
                 headless=True):
        """
        Args:
            rom_path: Path to Pokemon Red ROM
            battle_states_dir: Directory containing *_battle.state files
            output_shape: Observation image dimensions
            max_steps: Max steps per battle
            headless: Run without window
        """
        super().__init__()
        
        self.rom_path = rom_path
        self.output_shape = output_shape
        self.max_steps = max_steps
        self.headless = headless
        
        # Load all available battle states
        states_path = Path(battle_states_dir)
        self.battle_states = sorted(states_path.glob('*_battle.state'))
        
        if not self.battle_states:
            raise FileNotFoundError(
                f"No battle states found in {battle_states_dir}. "
                "Expected files like pewter_battle.state, cerulean_battle.state, etc."
            )
        
        print(f"Loaded {len(self.battle_states)} battle scenarios:")
        for state in self.battle_states:
            print(f"  - {state.stem}")
        
        # Gymnasium spaces
        self.action_space = spaces.Discrete(9)  # 8 buttons + no-op
        self.observation_space = spaces.Box(
            low=0, high=255,
            shape=(output_shape[0], output_shape[1], 3),
            dtype=np.uint8
        )
        
        # Initialize PyBoy
        window = 'null' if headless else 'SDL2'
        self.pyboy = PyBoy(
            rom_path,
            window=window,
            sound=False,
            sound_emulated=False,
            cgb=False
        )
        self.screen = self.pyboy.screen
        
        # Action mapping
        self.actions = [
            [''],          # 0: No-op
            ['a'],         # 1: A (select/confirm)
            ['b'],         # 2: B (cancel/run)
            ['start'],     # 3: Start (pause)
            ['select'],    # 4: Select (switch pokemon)
            ['up'],        # 5: Up
            ['down'],      # 6: Down
            ['left'],      # 7: Left
            ['right']      # 8: Right
        ]
        
        # Episode tracking
        self.current_state_file = None
        self.step_count = 0
        self.initial_enemy_hp = 0
        self.initial_player_hp = 0
        self.damage_dealt = 0
        self.damage_taken = 0
        self.in_battle_steps = 0
        self.out_battle_steps = 0
        
    def reset(self, seed=None, options=None):
        """Reset to a random battle scenario."""
        super().reset(seed=seed)
        
        # Select random battle state
        self.current_state_file = random.choice(self.battle_states)
        
        # Load state
        try:
            with open(self.current_state_file, 'rb') as f:
                self.pyboy.load_state(f)
        except Exception as e:
            print(f"Warning: Failed to load {self.current_state_file.name}: {e}")
            # Fallback to first state
            with open(self.battle_states[0], 'rb') as f:
                self.pyboy.load_state(f)
        
        # Stabilize after load
        for _ in range(20):
            self.pyboy.tick()
        
        # Auto-initiate battle: press A slowly until battle starts
        # Battle states are saved just before trainer dialogue
        max_attempts = 100  # ~10 seconds worth
        for attempt in range(max_attempts):
            # Check if already in battle
            if self.is_in_battle():
                print(f"Battle started after {attempt} A presses ({self.current_state_file.stem})")
                break
            
            # Press A briefly
            self.pyboy.button_press('a')
            for _ in range(3):
                self.pyboy.tick()
            self.pyboy.button_release('a')
            
            # Wait before next press (dialogue needs time)
            for _ in range(12):
                self.pyboy.tick()
                # Check during wait too
                if self.is_in_battle():
                    print(f"Battle started during wait ({self.current_state_file.stem})")
                    break
        
        # Final stabilization in battle
        for _ in range(30):
            self.pyboy.tick()
        
        # Initialize tracking
        self.step_count = 0
        self.damage_dealt = 0
        self.damage_taken = 0
        self.in_battle_steps = 0
        self.out_battle_steps = 0
        
        # Record initial HPs
        self.initial_player_hp = self.read_hp(PLAYER_HP_ADDR)
        self.initial_enemy_hp = self.read_hp(ENEMY_HP_ADDR)
        
        obs = self.render()
        info = {
            'scenario': self.current_state_file.stem,
            'initial_player_hp': self.initial_player_hp,
            'initial_enemy_hp': self.initial_enemy_hp
        }
        
        return obs, info
    
    def step(self, action):
        """Execute action and return observation, reward, done, info."""
        self.step_count += 1
        
        # Record state before action
        prev_player_hp = self.read_hp(PLAYER_HP_ADDR)
        prev_enemy_hp = self.read_hp(ENEMY_HP_ADDR)
        was_in_battle = self.is_in_battle()
        
        # Execute action
        for button in self.actions[action]:
            if button:
                self.pyboy.button_press(button)
        
        # Advance game
        for _ in range(8):  # 8 frames per step (~60fps -> 7.5 actions/sec)
            self.pyboy.tick()
        
        # Release buttons
        for button in self.actions[action]:
            if button:
                self.pyboy.button_release(button)
        
        # Read new state
        curr_player_hp = self.read_hp(PLAYER_HP_ADDR)
        curr_enemy_hp = self.read_hp(ENEMY_HP_ADDR)
        is_in_battle = self.is_in_battle()
        
        # Track battle presence
        if is_in_battle:
            self.in_battle_steps += 1
        else:
            self.out_battle_steps += 1
        
        # Calculate reward
        reward = self.calculate_reward(
            prev_player_hp, curr_player_hp,
            prev_enemy_hp, curr_enemy_hp,
            was_in_battle, is_in_battle,
            action
        )
        
        # Check termination
        terminated = False
        truncated = False
        
        # Victory: enemy defeated
        if curr_enemy_hp == 0 and prev_enemy_hp > 0:
            reward += 100.0
            terminated = True
            
        # Defeat: player defeated
        if curr_player_hp == 0:
            reward -= 50.0
            terminated = True
        
        # Timeout
        if self.step_count >= self.max_steps:
            reward -= 10.0  # Penalty for not finishing
            truncated = True
        
        # Observation and info
        obs = self.render()
        info = {
            'scenario': self.current_state_file.stem,
            'player_hp': curr_player_hp,
            'enemy_hp': curr_enemy_hp,
            'in_battle': is_in_battle,
            'damage_dealt': self.damage_dealt,
            'damage_taken': self.damage_taken,
            'battle_ratio': self.in_battle_steps / max(1, self.step_count)
        }
        
        return obs, reward, terminated, truncated, info
    
    def calculate_reward(self, prev_player_hp, curr_player_hp,
                        prev_enemy_hp, curr_enemy_hp,
                        was_in_battle, is_in_battle, action):
        """
        Reward function optimized for combat efficiency.
        """
        reward = 0.0
        
        # Penalize being out of battle (reduced to avoid training collapse)
        if not is_in_battle:
            reward -= 0.1  # Moderate penalty
            return reward
        
        # Small reward for staying in battle + time pressure
        reward += 0.02  # Positive for being in battle
        reward -= 0.01  # Time penalty
        
        # Damage dealt to enemy (primary reward)
        damage_dealt = prev_enemy_hp - curr_enemy_hp
        if damage_dealt > 0:
            self.damage_dealt += damage_dealt
            # Scale reward by damage percentage
            damage_pct = damage_dealt / max(1, self.initial_enemy_hp)
            reward += 5.0 * damage_pct  # Up to 5.0 for 100% HP damage
        
        # Damage taken (penalty)
        damage_taken = prev_player_hp - curr_player_hp
        if damage_taken > 0:
            self.damage_taken += damage_taken
            # Larger penalty for taking damage
            damage_pct = damage_taken / max(1, self.initial_player_hp)
            reward -= 3.0 * damage_pct
        
        # Reward efficient HP ratio
        if curr_enemy_hp > 0 and curr_player_hp > 0:
            hp_ratio = curr_player_hp / max(1, curr_enemy_hp)
            reward += 0.01 * hp_ratio  # Bonus for HP advantage
        
        # Penalize running away or canceling
        if action == 2:  # B button (escape)
            if is_in_battle:
                reward -= 0.5
        
        return reward
    
    def render(self):
        """Get current screen observation."""
        # Get raw pixels (144x160x3 or 144x160x4 with alpha)
        game_pixels = np.asarray(self.screen.ndarray, dtype=np.uint8)
        
        # Remove alpha channel if present
        if game_pixels.shape[2] == 4:
            game_pixels = game_pixels[:, :, :3]
        
        # Resize to target shape (128, 40, 3)
        resized = resize(game_pixels, self.output_shape + (3,), anti_aliasing=False)
        obs = (255 * resized).astype(np.uint8)
        
        return obs
    
    def is_in_battle(self):
        """Check if currently in a battle."""
        try:
            battle_type = self.pyboy.memory[BATTLE_TYPE_ADDR]
            in_battle_flag = self.pyboy.memory[IN_BATTLE_ADDR]
            enemy_hp = self.read_hp(ENEMY_HP_ADDR)
            
            # Multiple detection methods for reliability
            is_battle = (battle_type > 0) or (in_battle_flag == 1) or (enemy_hp > 0)
            return is_battle
        except:
            return False
    
    def read_hp(self, addr):
        """Read 2-byte HP value from memory (big-endian)."""
        high = self.pyboy.memory[addr]
        low = self.pyboy.memory[addr + 1]
        return (high << 8) | low
    
    def close(self):
        """Cleanup resources."""
        if hasattr(self, 'pyboy'):
            self.pyboy.stop(save=False)
