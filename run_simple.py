"""
Ejecución y métricas del modelo PPO Ultimate en el mismo estado (Brock Gym).
"""
import os
import sys
import time
import uuid
import json
import numpy as np
from pathlib import Path
from datetime import datetime

# Intentar importar librerías de visualización
try:
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns
    HAS_VIZ = True
except ImportError:
    HAS_VIZ = False
    print("⚠️ Advertencia: matplotlib/pandas/seaborn no encontrados. Se generarán solo reportes de texto.")

# Configurar paths
BASE_DIR = Path(__file__).parent
V2_DIR = BASE_DIR / "v2"
sys.path.insert(0, str(V2_DIR))
os.chdir(V2_DIR)

from red_gym_env_v2 import RedGymEnv
from stable_baselines3 import PPO

# ============================================================================
# CONFIGURACION
# ============================================================================

# Modelo entrenado recientemente (Ultimate)
MODEL_NEW = BASE_DIR / "session_ultimate_38883696" / "ultimate_combat_final.zip"

# Estado inicial común (pewter_battle.state usado por run_pretrained_interactive.py)
INIT_STATE = str(BASE_DIR / "pewter_gym_configured.state")

SESSION_UUID = str(uuid.uuid4())[:8]
OUTPUT_DIR = BASE_DIR / "RESULTADOS" / f"eval_compare_{SESSION_UUID}"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# CLASE DE ENTORNO (CombatEvalEnv)
# ============================================================================

# Direcciones de memoria
ADDR_IN_BATTLE = 0xD057
ADDR_BATTLE_MENU_ITEM = 0xCC2B
ADDR_PLAYER_HP_CURRENT = 0xD015
ADDR_PLAYER_HP_MAX = 0xD023
ADDR_ENEMY_HP_CURRENT = 0xCFE6
ADDR_ENEMY_HP_MAX = 0xCFF4
ADDR_MAP_ID = 0xD35E

ACTION_LABELS = {
    0: "Down", 1: "Left", 2: "Right", 3: "Up",
    4: "A", 5: "B", 6: "Start", 7: "Select"
}

class CombatEvalEnv(RedGymEnv):
    """Entorno de evaluación con las mismas recompensas del entorno original."""
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
        # Simplificado: se mantiene el esquema base (igual que Ultimate)
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
                if self.debug: print(f"[EVAL] Batalla iniciada. HP: {enemy_hp}")

            self.battle_steps += 1

            if menu_item == 0:
                if self.prev_menu_item != 0:
                    step_reward += 0.1
                self.non_attack_menu_steps = 0
            elif menu_item in (1, 2):
                step_reward -= 0.2
                self.non_attack_menu_steps += 1
            elif menu_item == 3:
                step_reward -= 0.5
                self.non_attack_menu_steps += 1

            if self.non_attack_menu_steps > 5:
                step_reward -= 0.5

            damage_dealt = self.prev_enemy_hp - enemy_hp
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
                if enemy_ratio < 0.2 and (self.prev_enemy_hp / max(enemy_max, 1)) >= 0.2:
                    step_reward += 3.0

            damage_received = self.prev_player_hp - player_hp
            if damage_received > 0:
                step_reward -= damage_received * 0.1

            if player_hp == 0 and self.prev_player_hp > 0:
                step_reward -= 5.0

            if enemy_hp == 0 and self.prev_enemy_hp > 0:
                step_reward += 20.0

            self.prev_enemy_hp = enemy_hp
            self.prev_player_hp = player_hp
            self.prev_menu_item = menu_item

        else:
            if self.prev_in_battle and not self.has_dealt_damage:
                step_reward -= 5.0
            self.battle_steps = 0
            self.has_dealt_damage = False
            self.non_attack_menu_steps = 0

        self.prev_in_battle = in_battle
        return step_reward

# ============================================================================
# SISTEMA DE ESTADISTICAS
# ============================================================================

class BattleStats:
    def __init__(self, label):
        self.label = label
        self.episodes = []
        self.current_episode = self._new_episode()
        self.start_time = time.time()

    def _new_episode(self):
        return {
            "steps": [],
            "player_hp": [],
            "enemy_hp": [],
            "actions": [],
            "rewards": [],
            "damage_dealt": 0,
            "damage_received": 0,
            "enemy_faints": 0,
            "player_faints": 0,
            "win": False,
            "duration_steps": 0
        }

    def log_step(self, step, player_hp, enemy_hp, action, reward):
        self.current_episode["steps"].append(step)
        self.current_episode["player_hp"].append(player_hp)
        self.current_episode["enemy_hp"].append(enemy_hp)
        self.current_episode["actions"].append(int(action))
        self.current_episode["rewards"].append(float(reward))
        self.current_episode["duration_steps"] += 1

        # Detectar faint (transición >0 a 0)
        if len(self.current_episode["enemy_hp"]) > 1:
            prev_e = self.current_episode["enemy_hp"][-2]
            if prev_e > 0 and enemy_hp == 0:
                self.current_episode["enemy_faints"] += 1

        if len(self.current_episode["player_hp"]) > 1:
            prev_p = self.current_episode["player_hp"][-2]
            if prev_p > 0 and player_hp == 0:
                self.current_episode["player_faints"] += 1

    def end_episode(self, win=False):
        self.current_episode["win"] = win

        if len(self.current_episode["player_hp"]) > 1:
            p_hp = np.array(self.current_episode["player_hp"])
            e_hp = np.array(self.current_episode["enemy_hp"])

            diff_e = e_hp[:-1] - e_hp[1:]
            self.current_episode["damage_dealt"] = float(np.sum(diff_e[(diff_e > 0) & (diff_e < 100)]))

            diff_p = p_hp[:-1] - p_hp[1:]
            self.current_episode["damage_received"] = float(np.sum(diff_p[(diff_p > 0) & (diff_p < 100)]))

        self.episodes.append(self.current_episode)
        self.current_episode = self._new_episode()

    def summary(self):
        if not self.episodes:
            return None
        ep = self.episodes[-1]
        return {
            "label": self.label,
            "win": ep["win"],
            "damage_dealt": ep["damage_dealt"],
            "damage_received": ep["damage_received"],
            "enemy_faints": ep["enemy_faints"],
            "player_faints": ep["player_faints"],
            "steps": ep["duration_steps"],
            "actions": ep["actions"],
        }


def create_env(session_name: str, init_state: str = INIT_STATE, extra_buttons: bool = False):
    sess_path = Path(f"session_{session_name}_{str(uuid.uuid4())[:4]}")
    env_config = {
        "headless": False,
        "save_final_state": True,
        "early_stop": False,
        "action_freq": 24,
        "init_state": init_state,
        "max_steps": 2**20,
        "print_rewards": False,
        "save_video": False,
        "fast_video": True,
        "session_path": sess_path,
        "gb_path": str(BASE_DIR / "PokemonRed.gb"),
        "debug": False,
        "sim_frame_dist": 2_000_000.0,
        "extra_buttons": extra_buttons,
        "combat_debug": True,
    }
    return CombatEvalEnv(env_config)


def force_combat_action(action: int) -> int:
    # Bloquear Start/B y forzar A
    if action == 6 or action == 5:
        return 4
    return action


def step_agent(env, model, stats: BattleStats, step_limit=5000):
    obs, info = env.reset()
    was_in_battle = False
    step = 0
    while step < step_limit:
        step += 1
        action, _states = model.predict(obs, deterministic=True)
        action = force_combat_action(int(action))

        # Anti-stuck: si 20 acciones idénticas seguidas
        if step > 20:
            last_actions = stats.current_episode["actions"][-20:]
            if len(last_actions) == 20 and len(set(last_actions)) == 1 and last_actions[0] == action:
                action = int(np.random.choice([0, 1, 2, 3, 4]))

        obs, rewards, terminated, truncated, info = env.step(action)

        p_hp, _ = env._read_hp(ADDR_PLAYER_HP_CURRENT, ADDR_PLAYER_HP_MAX)
        e_hp, _ = env._read_hp(ADDR_ENEMY_HP_CURRENT, ADDR_ENEMY_HP_MAX)

        stats.log_step(step, p_hp, e_hp, action, rewards)

        in_battle = env.read_m(ADDR_IN_BATTLE)
        current_map = env.read_m(ADDR_MAP_ID)
        if in_battle:
            was_in_battle = True

        if was_in_battle and in_battle == 0:
            # Fuera de batalla tras haber estado peleando
            if current_map == 54:
                stats.end_episode(win=True)
            else:
                stats.end_episode(win=False)
            break

        if terminated or truncated:
            stats.end_episode(win=(e_hp == 0))
            break

    return stats


def compare_and_report(stats_new: BattleStats):
    s_new = stats_new.summary()

    def fmt_bool(v):
        return "✔" if v else "✖"

    print("\n" + "="*70)
    print("EVALUACION DEL MODELO")
    print("="*70)
    header = f"{'Modelo':25} | {'Win':3} | {'Daño hecho':10} | {'Daño recibido':14} | {'Faints rival':12} | {'Faints propios':14} | {'Pasos':6}"
    print(header)
    print("-"*len(header))
    if s_new:
        print(f"{s_new['label']:25} | {fmt_bool(s_new['win'])}  | {s_new['damage_dealt']:10.1f} | {s_new['damage_received']:14.1f} | {s_new['enemy_faints']:12d} | {s_new['player_faints']:14d} | {s_new['steps']:6d}")

    # Acciones principales (top 5)
    if HAS_VIZ:
        sns.set_theme(style="darkgrid")
    if s_new:
        actions = s_new["actions"]
        if actions:
            counts = {}
            for a in actions:
                counts[a] = counts.get(a, 0) + 1
            top = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:5]
            top_str = ", ".join([f"{ACTION_LABELS.get(k, k)}: {v}" for k, v in top])
            print(f"Acciones más usadas ({s_new['label']}): {top_str}")

    # Guardar JSON comparativo
    data = {"new": s_new, "timestamp": datetime.now().isoformat()}
    with open(OUTPUT_DIR / "comparison.json", "w") as f:
        json.dump(data, f, indent=4)
    print(f"\n📄 Comparación guardada en: {OUTPUT_DIR / 'comparison.json'}")


if __name__ == '__main__':
    print("="*70)
    print("🤖 EVALUACION MODELO - POKEMON RED (Brock Gym)")
    print("="*70)
    print(f"Estado inicial: {INIT_STATE}")
    print(f"Modelo nuevo    : {MODEL_NEW}")
    env_new = create_env("new", init_state=INIT_STATE, extra_buttons=False)

    # Cargar modelos
    if not MODEL_NEW.exists():
        print(f"❌ No se encontró el modelo nuevo en {MODEL_NEW}")
        sys.exit(1)

    model_new = PPO.load(MODEL_NEW, env=env_new, custom_objects={'lr_schedule': 0, 'clip_range': 0})
    stats_new = BattleStats("Ultimate (nuevo)")

    print("\n🎮 Ejecutando el modelo Ultimate...")

    try:
        step_limit = 5000
        obs_new, _ = env_new.reset()
        was_in_battle_new = False
        step = 0

        while step < step_limit:
            step += 1

            # --- Modelo nuevo ---
            action_new, _ = model_new.predict(obs_new, deterministic=True)
            action_new = force_combat_action(int(action_new))
            if step > 20:
                last_actions = stats_new.current_episode["actions"][-20:]
                if len(last_actions) == 20 and len(set(last_actions)) == 1 and last_actions[0] == action_new:
                    action_new = int(np.random.choice([0,1,2,3,4]))
            obs_new, rew_new, term_new, trunc_new, _ = env_new.step(action_new)
            p_hp_new, _ = env_new._read_hp(ADDR_PLAYER_HP_CURRENT, ADDR_PLAYER_HP_MAX)
            e_hp_new, _ = env_new._read_hp(ADDR_ENEMY_HP_CURRENT, ADDR_ENEMY_HP_MAX)
            stats_new.log_step(step, p_hp_new, e_hp_new, action_new, rew_new)
            in_battle_new = env_new.read_m(ADDR_IN_BATTLE)
            map_new = env_new.read_m(ADDR_MAP_ID)
            if in_battle_new:
                was_in_battle_new = True

            # Fin de batalla nuevo
            if was_in_battle_new and in_battle_new == 0:
                stats_new.end_episode(win=(map_new == 54))
                was_in_battle_new = False
            if stats_new.episodes:
                break

            if term_new or trunc_new:
                stats_new.end_episode(win=(e_hp_new == 0))
            if stats_new.episodes:
                break

        # Si alguno no cerró episodio, cerrarlo como derrota
        if not stats_new.episodes:
            stats_new.end_episode(win=False)
    except KeyboardInterrupt:
        print("\n🛑 Interrumpido por usuario.")
    finally:
        env_new.close()
        compare_and_report(stats_new)
        print("\nEvaluación finalizada.")
