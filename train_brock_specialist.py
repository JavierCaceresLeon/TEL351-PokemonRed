"""
=============================================================================
ENTRENAMIENTO ESPECIALISTA EN COMBATE - BROCK GYM
=============================================================================
Script optimizado para entrenar un agente que ATAQUE en combates.

PROBLEMA ANTERIOR:
- clip_range: 0 -> El modelo no podia actualizar su politica
- clip_fraction: 0.997 -> 99.7% de updates rechazados
- approx_kl: 5e-05 -> Modelo completamente estancado

SOLUCION:
- clip_range CONSTANTE en 0.2 (nunca decae a 0)
- Recompensas explicitas por ATACAR (boton A en batalla)
- Penalizacion por solo moverse sin atacar
- Mayor entropia para forzar exploracion de ataques
=============================================================================
"""

import copy
import sys
import uuid
import os
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.utils import set_random_seed

# Configurar paths
BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR / "v2"))
os.chdir(BASE_DIR / "v2")

from red_gym_env_v2 import RedGymEnv  # type: ignore

# ============================================================================
# CONFIGURACION OPTIMIZADA PARA COMBATE
# ============================================================================

# Parametros de entrenamiento - OPTIMIZADOS
TOTAL_TIMESTEPS = 2_000_000  # Mas tiempo para aprender a atacar
CHECKPOINT_FREQ = 500_000
NUM_CPU = 8  # Numero de entornos paralelos
DEBUG_COMBAT_LOGS = False  # Cambia a True para ver prints detallados

# PPO Hiperparametros - CRITICOS PARA QUE APRENDA
LEARNING_RATE = 3e-4        # Mas alto para aprender rapido
CLIP_RANGE = 0.2            # CONSTANTE - nunca 0!
GAMMA = 0.98                # Menos descuento = mas enfocado en recompensa inmediata
GAE_LAMBDA = 0.95           # Estimacion de ventaja
ENT_COEF = 0.08             # MAS ENTROPIA = mas exploracion
VF_COEF = 0.5               # Coeficiente del critico
N_STEPS = 1024              # Menos pasos = updates mas frecuentes
BATCH_SIZE = 128            # Batches mas pequenos = mas updates  
MAX_GRAD_NORM = 0.5         # Clipping de gradientes
N_EPOCHS = 6                # Mas epocas por update

# Archivos
INIT_STATE = str(BASE_DIR / "pewter_gym_configured.state")
PRETRAINED_MODEL = str(BASE_DIR / "session_combat_6ad3ab4f" / "combat_specialist_final.zip")
SESSION_ID = str(uuid.uuid4())[:8]
SESSION_PATH = BASE_DIR / f"session_combat_{SESSION_ID}"
CHECKPOINT_DIR = SESSION_PATH / "checkpoints"

# ============================================================================
# DIRECCIONES DE MEMORIA POKEMON RED - COMBATE
# ============================================================================

ADDR_IN_BATTLE = 0xD057      # 0 = no batalla, 1+ = en batalla
ADDR_BATTLE_TYPE = 0xD05A    # Tipo de batalla
ADDR_CURRENT_MENU = 0xCC26   # Menu actual en batalla
ADDR_CURSOR_POS = 0xCC2A     # Posicion del cursor
ADDR_BATTLE_MENU_ITEM = 0xCC2B  # Item seleccionado en menu batalla (0=FIGHT, 1=PKMN, 2=ITEM, 3=RUN)

# HP del jugador (Pokemon activo)
ADDR_PLAYER_HP_CURRENT = 0xD015
ADDR_PLAYER_HP_MAX = 0xD023

# HP del oponente
ADDR_ENEMY_HP_CURRENT = 0xCFE6
ADDR_ENEMY_HP_MAX = 0xCFF4

# Mapa actual
ADDR_MAP_ID = 0xD35E

# ============================================================================
# ENTORNO ULTRA-COMBATE v2 - MAXIMO CASTIGO POR NO ATACAR
# ============================================================================

class CombatFocusedEnv(RedGymEnv):
    """
    Entorno EXTREMO que OBLIGA al agente a atacar.
    
    FILOSOFIA: Cada paso en batalla sin hacer daño = CASTIGO
    El unico camino hacia recompensa positiva es ATACAR.
    
    Sistema de recompensas v2:
    - CADA STEP en batalla sin daño: -2 (acumulativo, duele!)
    - Primer ataque exitoso: +500 (JACKPOT para que entienda)
    - Daño subsecuente: +5 por HP de daño
    - Victoria sobre Pokemon: +1000
    - Intentar huir: -200 (detectado por menu)
    - Seleccionar Pokemon/Item en batalla: -50
    - Perder batalla: -500
    """
    
    def __init__(self, config=None):
        super().__init__(config)
        self.gym_map_id = 54  # Pewter Gym
        self.debug = bool(config.get('combat_debug', False))
        
        # Estado de combate
        self.prev_in_battle = False
        self.prev_enemy_hp = 0
        self.prev_player_hp = 0
        self.initial_enemy_hp = 0  # HP al inicio de batalla
        self.battle_steps = 0
        self.has_dealt_damage = False  # Flag: ya hizo daño alguna vez?
        self.total_damage_dealt = 0
        self.first_damage_rewarded = False  # Solo dar bonus una vez
        
        # Tracking de menu
        self.prev_menu_item = 0
        self.flee_attempts = 0
        
    def reset(self, seed=None, options={}):
        obs, info = super().reset(seed=seed, options=options)
        self.prev_in_battle = False
        self.prev_enemy_hp = 0
        self.prev_player_hp = 0
        self.initial_enemy_hp = 0
        self.battle_steps = 0
        self.has_dealt_damage = False
        self.total_damage_dealt = 0
        self.first_damage_rewarded = False
        self.prev_menu_item = 0
        self.flee_attempts = 0
        return obs, info
    
    def _read_hp(self, addr_current, addr_max):
        """Lee HP actual y maximo."""
        try:
            current = self.read_m(addr_current) * 256 + self.read_m(addr_current + 1)
            max_hp = self.read_m(addr_max) * 256 + self.read_m(addr_max + 1)
            return current, max(max_hp, 1)
        except:
            return 0, 1
    
    def _is_in_battle(self):
        return self.read_m(ADDR_IN_BATTLE) != 0
    
    def _get_menu_selection(self):
        """0=FIGHT, 1=PKMN, 2=ITEM, 3=RUN"""
        try:
            return self.read_m(ADDR_BATTLE_MENU_ITEM)
        except:
            return 0
        
    def update_reward(self):
        """
        Sistema de recompensas BRUTAL - solo atacar da recompensa.
        """
        step_reward = super().update_reward()
        
        current_map = self.read_m(ADDR_MAP_ID)
        in_battle = self._is_in_battle()
        
        # === SALIR DEL GIMNASIO = MUERTE ===
        if current_map == 2:
            step_reward -= 2000
            self.step_count = self.max_steps
            return step_reward
        
        # === LOGICA DE COMBATE ===
        if in_battle:
            enemy_hp, enemy_max = self._read_hp(ADDR_ENEMY_HP_CURRENT, ADDR_ENEMY_HP_MAX)
            player_hp, player_max = self._read_hp(ADDR_PLAYER_HP_CURRENT, ADDR_PLAYER_HP_MAX)
            menu_item = self._get_menu_selection()
            
            # --- INICIO DE BATALLA ---
            if not self.prev_in_battle:
                self.initial_enemy_hp = enemy_hp
                self.prev_enemy_hp = enemy_hp
                self.prev_player_hp = player_hp
                self.battle_steps = 0
                self.has_dealt_damage = False
                self.first_damage_rewarded = False
                if self.debug:
                    print(f"[BATALLA] Iniciada! Enemy HP: {enemy_hp}")
            
            self.battle_steps += 1
            
            # --- DETECTAR INTENTO DE HUIR (menu item 3) ---
            if menu_item == 3:  # RUN seleccionado
                step_reward -= 200
                self.flee_attempts += 1
                if self.debug:
                    print(f"[BATALLA] INTENTO DE HUIR DETECTADO! Penalizacion -200")
            
            # --- DETECTAR SELECCION DE POKEMON/ITEM (no pelear) ---
            if menu_item == 1:  # PKMN
                step_reward -= 30
            if menu_item == 2:  # ITEM
                step_reward -= 30
            
            # --- DAÑO INFLIGIDO ---
            damage_dealt = self.prev_enemy_hp - enemy_hp
            
            if damage_dealt > 0:
                self.has_dealt_damage = True
                self.total_damage_dealt += damage_dealt
                
                # PRIMER ATAQUE = JACKPOT MASIVO
                if not self.first_damage_rewarded:
                    step_reward += 500  # BOOM! Aprende que atacar es bueno
                    self.first_damage_rewarded = True
                    if self.debug:
                        print(f"[BATALLA] PRIMER DAÑO! +500 bonus. Daño: {damage_dealt}")
                else:
                    # Daño subsecuente
                    step_reward += damage_dealt * 5
                    step_reward += 50  # Bonus por atacar
                
                # Bonus si enemigo bajo de vida
                enemy_ratio = enemy_hp / max(enemy_max, 1)
                if enemy_ratio < 0.3:
                    step_reward += 100
                    
            # --- ENEMIGO DERROTADO ---
            if enemy_hp == 0 and self.prev_enemy_hp > 0:
                step_reward += 1000  # VICTORIA!
                if self.debug:
                    print(f"[BATALLA] VICTORIA! Pokemon enemigo derrotado!")
            
            # --- DAÑO RECIBIDO (penalizacion leve) ---
            damage_received = self.prev_player_hp - player_hp
            if damage_received > 0:
                step_reward -= damage_received * 0.3
            
            # --- POKEMON PROPIO DERROTADO ---
            if player_hp == 0 and self.prev_player_hp > 0:
                step_reward -= 300
                if self.debug:
                    print(f"[BATALLA] Pokemon propio derrotado!")
            
            # === CASTIGO POR CADA PASO SIN HACER DAÑO ===
            # Esto es lo MAS IMPORTANTE - forzar a atacar
            if damage_dealt <= 0:
                # Penalizacion que CRECE con el tiempo
                idle_penalty = -2 - (self.battle_steps // 20)  # -2, -3, -4, etc.
                step_reward += idle_penalty
                
                # Cada 30 pasos sin daño, penalizacion extra
                if self.battle_steps % 30 == 0 and not self.has_dealt_damage:
                    step_reward -= 50
                    if self.debug:
                        print(f"[BATALLA] {self.battle_steps} pasos sin atacar! -50")
            
            # Actualizar estado
            self.prev_enemy_hp = enemy_hp
            self.prev_player_hp = player_hp
            self.prev_menu_item = menu_item
            
        else:
            # --- FUERA DE BATALLA ---
            if self.prev_in_battle:
                # Batalla terminó
                if self.has_dealt_damage:
                    step_reward += 100  # Bonus por haber peleado
                    if self.debug:
                        print(f"[BATALLA] Terminada. Daño total: {self.total_damage_dealt}")
                else:
                    # Termino sin atacar (huyo?)
                    step_reward -= 300
                    if self.debug:
                        print(f"[BATALLA] Terminada SIN ATACAR! Penalizacion -300")
                    
                self.total_damage_dealt = 0
            
            self.battle_steps = 0
            self.has_dealt_damage = False
        
        self.prev_in_battle = in_battle
        return step_reward


# ============================================================================
# CALLBACK PARA MONITOREO
# ============================================================================

class CombatMonitorCallback(BaseCallback):
    """Callback para monitorear progreso de combate."""
    
    def __init__(self, verbose=1):
        super().__init__(verbose)
        self.episode_rewards = []
        self.battles_won = 0
        self.total_damage = 0
        
    def _on_step(self) -> bool:
        # Imprimir estadisticas cada N pasos
        if self.n_calls % 10000 == 0:
            if hasattr(self.model, 'logger'):
                # Verificar que clip_range NO sea 0
                clip_range = self.model.clip_range(1.0)  # Evaluar schedule
                lr = self.model.learning_rate
                if callable(lr):
                    lr = lr(1.0)
                    
                print(f"\n[Step {self.n_calls}] clip_range={clip_range:.3f}, lr={lr:.6f}")
                
                if clip_range < 0.05:
                    print("⚠️  WARNING: clip_range muy bajo! El modelo no puede aprender.")
        
        return True


def constant_schedule(value: float):
    """Schedule constante que NUNCA decae a 0."""
    return lambda _: value


def make_env(rank, env_conf, seed=0):
    """
    Crea entorno de combate para entrenamiento paralelo.
    """
    def _init():
        local_conf = copy.deepcopy(env_conf)
        local_conf['instance_id'] = f'{rank}_{str(uuid.uuid4())[:4]}'
        # Solo el primer entorno imprime logs detallados si se activa DEBUG_COMBAT_LOGS
        local_conf['combat_debug'] = bool(local_conf.get('combat_debug', False) or (DEBUG_COMBAT_LOGS and rank == 0))
        env = CombatFocusedEnv(local_conf)
        env.reset(seed=(seed + rank))
        return env
    set_random_seed(seed)
    return _init


if __name__ == '__main__':
    print("="*70)
    print("🥊 ENTRENAMIENTO ULTRA-COMBATE - BROCK SPECIALIST 🥊")
    print(f"   Session: {SESSION_ID}")
    print("="*70)
    print(f"""
CONFIGURACION:
  - Total Steps: {TOTAL_TIMESTEPS:,}
  - Clip Range: {CLIP_RANGE} (CONSTANTE - nunca 0!)
  - Learning Rate: {LEARNING_RATE}
  - Entropia: {ENT_COEF} (alta para explorar ataques)
  - CPUs: {NUM_CPU}
""")
    
    # Configuración del entorno
    env_config = {
        'headless': True,
        'save_final_state': True,
        'early_stop': False,
        'action_freq': 24,
        'init_state': INIT_STATE,
        'max_steps': 2048 * 8,  # Mas pasos para tener tiempo de pelear
        'print_rewards': False,
        'save_video': False,
        'fast_video': True,
        'session_path': SESSION_PATH,
        'gb_path': str(BASE_DIR / 'PokemonRed.gb'),
        'debug': False,
        'sim_frame_dist': 2_000_000.0,
        'extra_buttons': False,
        'combat_debug': False
    }
    
    # Crear entornos paralelos
    print(f"Creando {NUM_CPU} entornos paralelos...")
    env_fns = [make_env(i, env_config) for i in range(NUM_CPU)]
    if NUM_CPU == 1:
        env = DummyVecEnv(env_fns)
    else:
        try:
            env = SubprocVecEnv(env_fns, start_method='spawn')
        except Exception as exc:
            print("⚠️  SubprocVecEnv fallo o se detuvo. Reintentando con DummyVecEnv (sin multiproceso).", exc)
            env = DummyVecEnv(env_fns)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Cargar modelo pre-entrenado
    print(f"Cargando modelo base: {PRETRAINED_MODEL}")
    
    # Verificar si hay checkpoints previos para reanudar
    checkpoints = sorted(list(CHECKPOINT_DIR.glob("*.zip")))
    if checkpoints:
        latest_checkpoint = checkpoints[-1]
        print(f"✅ Reanudando desde checkpoint: {latest_checkpoint}")
        model = PPO.load(
            latest_checkpoint, 
            env=env, 
            device='auto'
        )
    elif os.path.exists(PRETRAINED_MODEL):
        print(f"📦 Cargando modelo pre-entrenado: {PRETRAINED_MODEL}")
        model = PPO.load(
            PRETRAINED_MODEL, 
            env=env, 
            device='auto'
        )
        print("✅ Modelo cargado. Aplicando configuracion de combate...")
    else:
        print("⚠️  No hay modelo previo. Creando modelo NUEVO desde cero...")
        model = PPO(
            "MultiInputPolicy",
            env,
            learning_rate=LEARNING_RATE,
            n_steps=N_STEPS,
            batch_size=BATCH_SIZE,
            n_epochs=N_EPOCHS,
            gamma=GAMMA,
            gae_lambda=GAE_LAMBDA,
            clip_range=CLIP_RANGE,
            ent_coef=ENT_COEF,
            vf_coef=VF_COEF,
            max_grad_norm=MAX_GRAD_NORM,
            verbose=1,
            device='auto'
        )

    # === APLICAR CONFIGURACION CORRECTA ===
    # CRITICO: Usar schedules CONSTANTES para que nunca decaigan a 0
    model.learning_rate = LEARNING_RATE
    model.lr_schedule = constant_schedule(LEARNING_RATE)
    model.clip_range = constant_schedule(CLIP_RANGE)  # NUNCA 0!
    model.gamma = GAMMA
    model.gae_lambda = GAE_LAMBDA
    model.ent_coef = ENT_COEF
    model.vf_coef = VF_COEF
    model.n_steps = N_STEPS
    model.batch_size = BATCH_SIZE
    model.max_grad_norm = MAX_GRAD_NORM
    
    # === RECREAR ROLLOUT BUFFER CON NUEVA CONFIGURACION ===
    # Esto es CRITICO cuando se carga un modelo con n_steps diferente
    from stable_baselines3.common.buffers import DictRolloutBuffer
    model.rollout_buffer = DictRolloutBuffer(
        N_STEPS,
        model.observation_space,
        model.action_space,
        device=model.device,
        gamma=GAMMA,
        gae_lambda=GAE_LAMBDA,
        n_envs=NUM_CPU,
    )
    print("✅ Rollout buffer recreado con n_steps correcto")
    
    print(f"""
✅ CONFIGURACION PPO APLICADA:
   - learning_rate: {LEARNING_RATE}
   - clip_range: {CLIP_RANGE} (CONSTANTE)
   - ent_coef: {ENT_COEF}
   - n_steps: {N_STEPS}
   - batch_size: {BATCH_SIZE}
""")
        
    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=CHECKPOINT_FREQ // NUM_CPU, 
        save_path=str(CHECKPOINT_DIR),
        name_prefix='combat_specialist'
    )
    monitor_callback = CombatMonitorCallback(verbose=1)
    callbacks = [checkpoint_callback, monitor_callback]
    
    # Entrenar
    print(f"\n🚀 Iniciando entrenamiento por {TOTAL_TIMESTEPS:,} pasos...")
    print("="*70)
    
    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS, 
            callback=callbacks,
            progress_bar=True,
            reset_num_timesteps=True  # REINICIAR conteo para evitar error de buffer
        )
        
        # Guardar modelo final
        final_path = SESSION_PATH / "combat_specialist_final.zip"
        model.save(str(final_path))
        print(f"\n✅ Entrenamiento finalizado!")
        print(f"   Modelo guardado en: {final_path}")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Entrenamiento interrumpido por usuario.")
        interrupt_path = SESSION_PATH / "combat_specialist_interrupted.zip"
        model.save(str(interrupt_path))
        print(f"   Modelo guardado en: {interrupt_path}")
        
    finally:
        env.close()
        print("\n🏁 Sesion terminada.")
