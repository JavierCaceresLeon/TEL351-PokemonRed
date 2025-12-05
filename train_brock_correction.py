import copy
import sys
import uuid
import os
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.buffers import DictRolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback

# Paths básicos
BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR / "v2"))
os.chdir(BASE_DIR / "v2")

from red_gym_env_v2 import RedGymEnv  # type: ignore

# -----------------------------------------------------------------------------------
# CONFIGURACION "CORRECTION" - DISCIPLINA MILITAR
# -----------------------------------------------------------------------------------

TOTAL_TIMESTEPS = 1_000_000  # Sesion corta de correccion
NUM_CPU = 8
LEARNING_RATE = 5e-5   # Muy bajo para refinar sin romper
CLIP_RANGE = 0.1
GAMMA = 0.99
GAE_LAMBDA = 0.95
ENT_COEF = 0.0001      # CERO creatividad. Solo obediencia.
VF_COEF = 0.5
N_STEPS = 2048
BATCH_SIZE = 256
MAX_GRAD_NORM = 0.5
N_EPOCHS = 10

INIT_STATE = str(BASE_DIR / "pewter_gym_configured.state")
# CARGAMOS EL MODELO ULTIMATE
PRETRAINED_MODEL = str(BASE_DIR / "session_ultimate_38883696" / "ultimate_combat_final.zip")

SESSION_ID = f"correction_{str(uuid.uuid4())[:8]}"
SESSION_PATH = BASE_DIR / f"session_{SESSION_ID}"
CHECKPOINT_DIR = SESSION_PATH / "checkpoints"

# Direcciones de memoria
ADDR_IN_BATTLE = 0xD057
ADDR_BATTLE_MENU_ITEM = 0xCC2B
ADDR_PLAYER_HP_CURRENT = 0xD015
ADDR_PLAYER_HP_MAX = 0xD023
ADDR_ENEMY_HP_CURRENT = 0xCFE6
ADDR_ENEMY_HP_MAX = 0xCFF4
ADDR_MAP_ID = 0xD35E

class CombatCorrectionEnv(RedGymEnv):
    """
    Entorno de CORRECCION.
    Las recompensas por ganar se mantienen, pero los castigos por
    acciones estupidas (RUN, ITEM, PKMN) son SEVEROS.
    """
    def __init__(self, config=None):
        super().__init__(config)
        self.gym_map_id = 54
        self.debug = bool(config.get("combat_debug", False))

        self.prev_in_battle = False
        self.prev_enemy_hp = 0
        self.prev_player_hp = 0
        self.battle_steps = 0
        self.has_dealt_damage = False
        self.total_damage_dealt = 0
        self.first_damage_rewarded = False
        self.prev_menu_item = 0
        self.non_attack_menu_steps = 0

    def reset(self, seed=None, options=None):
        if options is None: options = {}
        obs, info = super().reset(seed=seed, options=options)
        self.prev_in_battle = False
        self.prev_enemy_hp = 0
        self.prev_player_hp = 0
        self.battle_steps = 0
        self.has_dealt_damage = False
        self.total_damage_dealt = 0
        self.first_damage_rewarded = False
        self.prev_menu_item = 0
        self.non_attack_menu_steps = 0
        return obs, info

    def _read_hp(self, addr_current, addr_max):
        try:
            current = self.read_m(addr_current) * 256 + self.read_m(addr_current + 1)
            max_hp = self.read_m(addr_max) * 256 + self.read_m(addr_max + 1)
            # Sanitizacion basica
            if max_hp > 1000: max_hp = 1000
            if current > 1000: current = 1000
            return current, max(max_hp, 1)
        except Exception:
            return 0, 1

    def _is_in_battle(self):
        return self.read_m(ADDR_IN_BATTLE) != 0

    def _get_menu_selection(self):
        try:
            return self.read_m(ADDR_BATTLE_MENU_ITEM)
        except Exception:
            return 0

    def update_reward(self):
        # self.update_initial_keys() # Removed as it does not exist
        step_reward = 0.0

        current_map = self.read_m(ADDR_MAP_ID)
        in_battle = self._is_in_battle()

        if current_map == 2:
            step_reward = -5.0
            self.step_count = self.max_steps
            return step_reward

        if in_battle:
            enemy_hp, enemy_max = self._read_hp(ADDR_ENEMY_HP_CURRENT, ADDR_ENEMY_HP_MAX)
            player_hp, player_max = self._read_hp(ADDR_PLAYER_HP_CURRENT, ADDR_PLAYER_HP_MAX)
            menu_item = self._get_menu_selection()

            if not self.prev_in_battle:
                self.prev_enemy_hp = enemy_hp
                self.prev_player_hp = player_hp
                self.battle_steps = 0
                self.has_dealt_damage = False
                self.first_damage_rewarded = False
                if self.debug: print(f"[CORRECTION] Batalla iniciada.")

            self.battle_steps += 1

            # --- CASTIGOS SEVEROS POR MENUS INCORRECTOS ---
            # 0=FIGHT, 1=PKMN, 2=ITEM, 3=RUN
            if menu_item == 0: # FIGHT
                if self.prev_menu_item != 0:
                    step_reward += 0.2 # Incentivo por volver al camino del bien
                self.non_attack_menu_steps = 0
            
            elif menu_item == 1 or menu_item == 2: # PKMN / ITEM
                step_reward -= 5.0 # CASTIGO FUERTE
                self.non_attack_menu_steps += 1
                if self.debug and self.battle_steps % 10 == 0:
                    print(f"[CORRECTION] Castigo por menu inutil (-5.0)")
            
            elif menu_item == 3: # RUN
                step_reward -= 10.0 # CASTIGO MAXIMO (LAVA)
                self.non_attack_menu_steps += 1
                if self.debug and self.battle_steps % 10 == 0:
                    print(f"[CORRECTION] Castigo por intentar huir (-10.0)")
            
            # Si insiste en quedarse en menus malos, el castigo escala
            if self.non_attack_menu_steps > 5:
                step_reward -= 2.0

            # --- DAÑO (Mantenemos la recompensa positiva) ---
            damage_dealt = self.prev_enemy_hp - enemy_hp
            
            # Sanitizacion de daño
            if damage_dealt > 100: damage_dealt = 0
            
            if damage_dealt > 0:
                self.has_dealt_damage = True
                self.total_damage_dealt += damage_dealt
                
                step_reward += damage_dealt * 0.5
                
                if not self.first_damage_rewarded:
                    step_reward += 5.0
                    self.first_damage_rewarded = True
                
                enemy_ratio = enemy_hp / max(enemy_max, 1)
                if enemy_ratio < 0.5 and (self.prev_enemy_hp / max(enemy_max, 1)) >= 0.5:
                    step_reward += 2.0

            # --- VICTORIA ---
            if enemy_hp == 0 and self.prev_enemy_hp > 0:
                step_reward += 20.0

            # --- DERROTA ---
            if player_hp == 0 and self.prev_player_hp > 0:
                step_reward -= 5.0

            self.prev_enemy_hp = enemy_hp
            self.prev_player_hp = player_hp
            self.prev_menu_item = menu_item

        else:
            if self.prev_in_battle:
                if not self.has_dealt_damage:
                    step_reward -= 10.0 # Huyo cobardemente
            
            self.battle_steps = 0
            self.has_dealt_damage = False
            self.non_attack_menu_steps = 0

        self.prev_in_battle = in_battle
        return step_reward

class CorrectionCallback(BaseCallback):
    def __init__(self, verbose=1):
        super().__init__(verbose)
        self.total_steps = 0
    
    def _on_step(self) -> bool:
        self.total_steps += 1
        if self.total_steps % 10000 == 0:
            print(f" > Correction Steps: {self.total_steps} / {TOTAL_TIMESTEPS}")
        return True

def constant_schedule(value: float):
    return lambda _: value

def make_env(rank: int, seed: int = 0):
    def _init():
        conf = {
            "headless": True,
            "save_final_state": True,
            "early_stop": False,
            "action_freq": 24,
            "init_state": INIT_STATE,
            "max_steps": 2048 * 8,
            "print_rewards": False,
            "save_video": False,
            "fast_video": True,
            "session_path": SESSION_PATH,
            "gb_path": str(BASE_DIR / "PokemonRed.gb"),
            "debug": False,
            "sim_frame_dist": 2_000_000.0,
            "extra_buttons": False,
            "combat_debug": False,
        }
        conf["instance_id"] = f"corr_{rank}_{str(uuid.uuid4())[:4]}"
        env = CombatCorrectionEnv(conf)
        env.reset(seed=seed + rank)
        return env
    set_random_seed(seed)
    return _init

if __name__ == "__main__":
    print("=" * 70)
    print("🔧 ENTRENAMIENTO DE CORRECCION - DISCIPLINA")
    print(f"   Session: {SESSION_ID}")
    print("=" * 70)

    SESSION_PATH.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    env_fns = [make_env(i) for i in range(NUM_CPU)]
    if NUM_CPU > 1:
        env = SubprocVecEnv(env_fns, start_method="spawn")
    else:
        env = DummyVecEnv(env_fns)

    if os.path.exists(PRETRAINED_MODEL):
        print(f"Cargando modelo ULTIMATE: {PRETRAINED_MODEL}")
        model = PPO.load(PRETRAINED_MODEL, env=env, device="auto")
    else:
        print(f"❌ ERROR: No se encuentra el modelo {PRETRAINED_MODEL}")
        sys.exit(1)

    # APLICAR CONFIGURACION DE CORRECCION
    model.learning_rate = constant_schedule(LEARNING_RATE)
    model.clip_range = constant_schedule(CLIP_RANGE)
    model.ent_coef = ENT_COEF # BAJAR ENTROPIA
    
    # Recrear buffer (mismo n_steps, asi que es seguro)
    model.rollout_buffer = DictRolloutBuffer(
        N_STEPS,
        model.observation_space,
        model.action_space,
        device=model.device,
        gamma=GAMMA,
        gae_lambda=GAE_LAMBDA,
        n_envs=env.num_envs,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=100_000 // NUM_CPU,
        save_path=str(CHECKPOINT_DIR),
        name_prefix="correction_combat"
    )
    monitor_callback = CorrectionCallback()

    print("\nIniciando correccion de comportamiento...")
    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=[checkpoint_callback, monitor_callback],
            progress_bar=True,
            reset_num_timesteps=True,
            tb_log_name="ppo_correction"
        )
    except KeyboardInterrupt:
        print("Interrumpido por usuario.")
    
    final_path = SESSION_PATH / "correction_combat_final.zip"
    model.save(str(final_path))
    print(f"Modelo corregido guardado en: {final_path}")
    env.close()
