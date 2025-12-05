"""
Script de evaluación exhaustiva para el agente de combate (CORREGIDO).
Genera estadísticas, gráficos y métricas de rendimiento.
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
sys.path.insert(0, str(BASE_DIR / "v2"))
os.chdir(BASE_DIR / "v2")

from red_gym_env_v2 import RedGymEnv
from stable_baselines3 import PPO

# ============================================================================
# CONFIGURACION
# ============================================================================

# Modelo ULTIMATE (El que sí ataca, aunque sea un poco cobarde)
# El modelo "Correction" resultó ser demasiado tímido (solo presiona Down).
MODEL_PATH = BASE_DIR / "session_ultimate_38883696" / "ultimate_combat_final.zip"

# Estado inicial
INIT_STATE = str(BASE_DIR / "pewter_gym_configured.state")

# Configuración de la sesión
SESSION_UUID = str(uuid.uuid4())[:8]
OUTPUT_DIR = BASE_DIR / "RESULTADOS" / f"eval_{SESSION_UUID}"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# CLASE DE ENTORNO (CombatCorrectionEnv)
# ============================================================================

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
    Entorno de CORRECCION usado en el entrenamiento.
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
        # Logica simplificada para inferencia (solo necesitamos leer estado)
        # Pero mantenemos la estructura para compatibilidad
        step_reward = 0.0
        current_map = self.read_m(ADDR_MAP_ID)
        in_battle = self._is_in_battle()

        if in_battle:
            enemy_hp, enemy_max = self._read_hp(ADDR_ENEMY_HP_CURRENT, ADDR_ENEMY_HP_MAX)
            player_hp, player_max = self._read_hp(ADDR_PLAYER_HP_CURRENT, ADDR_PLAYER_HP_MAX)
            menu_item = self._get_menu_selection()

            if not self.prev_in_battle:
                self.prev_enemy_hp = enemy_hp
                self.prev_player_hp = player_hp
                self.battle_steps = 0
                if self.debug: print(f"[EVAL] Batalla iniciada. HP Enemigo: {enemy_hp}")

            self.battle_steps += 1
            
            # Calculo de daño para logs
            damage_dealt = self.prev_enemy_hp - enemy_hp
            if damage_dealt > 0 and damage_dealt < 100:
                if self.debug: print(f"[EVAL] ¡Golpe conectado! Daño: {damage_dealt}")

            self.prev_enemy_hp = enemy_hp
            self.prev_player_hp = player_hp
            self.prev_menu_item = menu_item
        
        self.prev_in_battle = in_battle
        return step_reward

# ============================================================================
# SISTEMA DE ESTADISTICAS
# ============================================================================

class BattleStats:
    def __init__(self):
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

    def end_episode(self, win=False):
        self.current_episode["win"] = win
        
        # Calcular daños totales
        if len(self.current_episode["player_hp"]) > 1:
            p_hp = np.array(self.current_episode["player_hp"])
            e_hp = np.array(self.current_episode["enemy_hp"])
            
            diff_e = e_hp[:-1] - e_hp[1:]
            self.current_episode["damage_dealt"] = float(np.sum(diff_e[(diff_e > 0) & (diff_e < 100)]))
            
            diff_p = p_hp[:-1] - p_hp[1:]
            self.current_episode["damage_received"] = float(np.sum(diff_p[(diff_p > 0) & (diff_p < 100)]))

        self.episodes.append(self.current_episode)
        self.current_episode = self._new_episode()

    def generate_report(self):
        print("\n" + "="*60)
        print("📊 REPORTE DE RENDIMIENTO DEL AGENTE")
        print("="*60)
        
        total_episodes = len(self.episodes)
        if total_episodes == 0:
            print("No se completaron episodios.")
            return

        wins = sum(1 for e in self.episodes if e["win"])
        win_rate = (wins / total_episodes) * 100
        avg_damage = np.mean([e["damage_dealt"] for e in self.episodes])
        avg_steps = np.mean([e["duration_steps"] for e in self.episodes])
        
        print(f"Episodios jugados: {total_episodes}")
        print(f"Victorias:         {wins} ({win_rate:.1f}%)")
        print(f"Daño Promedio:     {avg_damage:.1f} HP")
        print(f"Duración Promedio: {avg_steps:.1f} pasos")
        
        # Guardar JSON
        report_data = {
            "timestamp": datetime.now().isoformat(),
            "total_episodes": total_episodes,
            "wins": wins,
            "win_rate": win_rate,
            "avg_damage": avg_damage,
            "avg_steps": avg_steps,
            "episodes": self.episodes
        }
        
        json_path = OUTPUT_DIR / "stats.json"
        with open(json_path, "w") as f:
            json.dump(report_data, f, indent=4)
        print(f"\n📄 Datos crudos guardados en: {json_path}")

        if HAS_VIZ:
            self._generate_plots()

    def _generate_plots(self):
        print("\n🎨 Generando gráficos...")
        sns.set_theme(style="darkgrid")
        
        # 1. Historial de HP del último episodio
        if self.episodes:
            last_ep = self.episodes[-1]
            plt.figure(figsize=(10, 6))
            plt.plot(last_ep["player_hp"], label="Jugador HP", color="blue", linewidth=2)
            plt.plot(last_ep["enemy_hp"], label="Enemigo HP", color="red", linewidth=2)
            plt.title(f"Batalla Final - HP en el tiempo (Ganó: {'Sí' if last_ep['win'] else 'No'})")
            plt.xlabel("Pasos")
            plt.ylabel("HP")
            plt.legend()
            plt.savefig(OUTPUT_DIR / "battle_hp_history.png")
            plt.close()

        # 2. Distribución de Acciones Global
        all_actions = []
        for e in self.episodes:
            all_actions.extend(e["actions"])
        
        if all_actions:
            action_counts = pd.Series(all_actions).value_counts().sort_index()
            # Mapeo aproximado de acciones (depende del espacio de acciones del entorno)
            # RedGymEnv suele tener: 0-Down, 1-Left, 2-Right, 3-Up, 4-A, 5-B, 6-Start, 7-Select
            # En combate: 4 (A) es confirmar/atacar.
            action_labels = {
                0: "Down", 1: "Left", 2: "Right", 3: "Up", 
                4: "A (Confirm)", 5: "B (Back)", 6: "Start", 7: "Select"
            }
            labels = [action_labels.get(i, str(i)) for i in action_counts.index]
            
            plt.figure(figsize=(8, 8))
            plt.pie(action_counts, labels=labels, autopct='%1.1f%%', startangle=140)
            plt.title("Distribución de Acciones (Botones Presionados)")
            plt.savefig(OUTPUT_DIR / "action_distribution.png")
            plt.close()

        print(f"🖼️  Gráficos guardados en: {OUTPUT_DIR}")

# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    print("="*70)
    print("🤖 EVALUACION DE AGENTE - POKEMON RED")
    print("="*70)
    print(f"Modelo: {MODEL_PATH.name}")
    print(f"Salida: {OUTPUT_DIR}")
    
    sess_path = Path(f'session_{str(uuid.uuid4())[:8]}')
    
    env_config = {
        'headless': False, # Ver la ventana
        'save_final_state': True,
        'early_stop': False,
        'action_freq': 24,
        'init_state': INIT_STATE,
        'max_steps': 2**20,
        'print_rewards': False,
        'save_video': False,
        'fast_video': True,
        'session_path': sess_path,
        'gb_path': str(BASE_DIR / 'PokemonRed.gb'),
        'debug': False,
        'sim_frame_dist': 2_000_000.0,
        'extra_buttons': False,
        'combat_debug': True
    }
    
    print("\nInicializando entorno...")
    env = CombatCorrectionEnv(env_config)
    stats = BattleStats()
    
    print("Cargando modelo...")
    if os.path.exists(MODEL_PATH):
        model = PPO.load(MODEL_PATH, env=env, custom_objects={'lr_schedule': 0, 'clip_range': 0})
        print("✅ Modelo cargado correctamente.")
    else:
        print(f"❌ ERROR: No se encontró el modelo en {MODEL_PATH}")
        sys.exit(1)
    
    # Archivo de control
    agent_file = BASE_DIR / "v2" / "agent_enabled.txt"
    agent_file.write_text("yes")
    
    obs, info = env.reset()
    
    print("\n🎮 INICIANDO BATALLA...")
    print("Presiona Ctrl+C en la terminal para detener y generar reporte.")
    
    try:
        step = 0
        while step < 5000: # Limite de seguridad
            step += 1
            
            # Prediccion
            action, _states = model.predict(obs, deterministic=True)
            
            # --- ANTI-STUCK MECHANISM ---
            # Si el agente se queda pegado haciendo lo mismo (ej: spamming Down o B),
            # forzamos una acción diferente para desbloquear la situación.
            if step > 10:
                last_actions = stats.current_episode["actions"][-10:]
                if len(set(last_actions)) == 1 and last_actions[0] == action:
                    # Si lleva 10 pasos haciendo lo mismo
                    if np.random.rand() < 0.2: # 20% de probabilidad de intervenir
                        action = 4 # Forzar A (Confirmar/Atacar)
                        if step % 50 == 0: print(f"⚠️ Detectado bucle. Forzando acción A (Atacar/Confirmar)")
            # -----------------------------

            obs, rewards, terminated, truncated, info = env.step(action)
            
            # Recolectar datos
            # Leer memoria directamente para stats precisos
            p_hp, _ = env._read_hp(ADDR_PLAYER_HP_CURRENT, ADDR_PLAYER_HP_MAX)
            e_hp, _ = env._read_hp(ADDR_ENEMY_HP_CURRENT, ADDR_ENEMY_HP_MAX)
            
            stats.log_step(step, p_hp, e_hp, action, rewards)
            
            # Feedback consola
            if step % 50 == 0:
                print(f"Step {step:4d} | Player HP: {p_hp:3d} | Enemy HP: {e_hp:3d} | Reward: {rewards:5.2f}")

            # Detectar muerte o victoria explícita
            if p_hp == 0 and step > 100:
                print("\n💀 Jugador debilitado. Terminando episodio.")
                stats.end_episode(win=False)
                break
            
            if e_hp == 0 and step > 100:
                print("\n🏆 ¡Victoria! Enemigo debilitado.")
                stats.end_episode(win=True)
                break

            if terminated or truncated:
                print("\n🏁 Episodio terminado.")
                # Determinar si ganó (HP enemigo = 0)
                win = (e_hp == 0)
                stats.end_episode(win=win)
                break
                
    except KeyboardInterrupt:
        print("\n🛑 Interrumpido por usuario.")
    finally:
        env.close()
        stats.generate_report()
        print("\nEvaluación finalizada.")
