import os
from os.path import exists
from pathlib import Path
import uuid
import time
import glob
import sys

# Add v2 directory to path to enable local imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from red_gym_env_v2 import RedGymEnv
from stable_baselines3 import A2C, PPO
from stable_baselines3.common import env_checker
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed

import json

# Add v2 directory to path to enable local imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))
sys.path.append(str(current_dir.parent / "gym_scenarios"))

from red_gym_env_v2 import RedGymEnv
from stable_baselines3 import A2C, PPO
from stable_baselines3.common import env_checker
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from gym_memory_addresses import *

def inject_gym_config(env, scenario_path):
    """Inyecta la configuración del gimnasio en el emulador en ejecución"""
    config_path = Path(scenario_path).parent / "team_config.json"
    if not config_path.exists():
        print(f"Advertencia: No se encontró config en {config_path}")
        return

    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    print(f"Inyectando configuración: {config.get('gym_name', 'Unknown')}")
    
    pyboy = env.pyboy
    
    def write_mem(addr, val):
        if hasattr(pyboy, "set_memory_value"):
            pyboy.set_memory_value(addr, val)
        else:
            pyboy.memory[addr] = val & 0xFF

    def write_word(addr, val):
        write_mem(addr, (val >> 8) & 0xFF)
        write_mem(addr + 1, val & 0xFF)

    def write_bcd(val):
        return ((val // 10) << 4) | (val % 10)

    # 1. Equipo
    team = config.get('player_team', [])
    write_mem(PARTY_SIZE_ADDRESS, len(team))
    for i, poke in enumerate(team):
        slot = poke.get('slot', 1) - 1
        if 0 <= slot < 6:
            write_mem(PARTY_ADDRESSES[slot], poke.get('species_id', 0))
            write_mem(LEVELS_ADDRESSES[slot], poke.get('level', 5))
            write_word(HP_ADDRESSES[slot], poke.get('current_hp', 20))
            write_word(MAX_HP_ADDRESSES[slot], poke.get('max_hp', 20))
            # Moves could be added here if needed

    # 2. Items
    items = config.get('bag_items', [])
    item_count = min(len(items), 20)
    # BAG_ITEM_COUNT = 0xD31D (Need to verify this address or import it)
    # Assuming it's imported from gym_memory_addresses
    if 'BAG_ITEM_COUNT' in globals():
        write_mem(BAG_ITEM_COUNT, item_count)
        for i, item in enumerate(items[:20]):
            base = BAG_ITEMS_START + (i * 2)
            write_mem(base, item.get('item_id', 0))
            write_mem(base + 1, item.get('quantity', 1))
        write_mem(BAG_ITEMS_START + (item_count * 2), 0xFF)

    # 3. Dinero y Medallas
    money = config.get('money', 0)
    write_mem(MONEY_ADDRESS_1, write_bcd(money // 10000))
    write_mem(MONEY_ADDRESS_2, write_bcd((money // 100) % 100))
    write_mem(MONEY_ADDRESS_3, write_bcd(money % 100))
    
    write_mem(BADGE_COUNT_ADDRESS, config.get('badge_bits', 0))

    # 4. Posición (Lo más crítico)
    start_pos = config.get('start_position', {'x': 4, 'y': 13})
    map_id = config.get('map_id', 0)
    
    print(f"Iniciando secuencia de Warp a Mapa {map_id} ({start_pos['x']}, {start_pos['y']})...")
    
    # --- MÉTODO DE WARP SEGURO ---
    # En lugar de sobrescribir la memoria del mapa actual (lo que causa glitches gráficos),
    # le decimos al juego que ejecute una transición de warp normal.
    
    # 1. Configurar destino del warp
    write_mem(0xD365, map_id)          # wWarpDestMap
    write_mem(0xD366, start_pos['x'])  # wWarpDestX
    write_mem(0xD367, start_pos['y'])  # wWarpDestY
    
    # 2. Configurar flag de warp pendiente (wd72d)
    # Bit 3 (0x08) indica "Warp transition"
    # Leemos el valor actual para no romper otros flags
    if hasattr(pyboy, "get_memory_value"):
        current_wd72d = pyboy.get_memory_value(0xD12B)
    else:
        current_wd72d = pyboy.memory[0xD12B]
    
    write_mem(0xD12B, current_wd72d | 0x08)
    
    # 3. Resetear estado de script del mapa para evitar conflictos
    write_mem(0xD35D, 0x00) # wMapPalOffset (Script index?) - A veces ayuda limpiar esto
    
    print("Warp programado. El juego debería realizar la transición en breve.")
    return map_id, start_pos

def make_env(rank, env_conf, seed=0):
    def _init():
        env = RedGymEnv(env_conf)
        return env
    set_random_seed(seed)
    return _init

def get_most_recent_zip_with_age(folder_path):
    # Ensure folder exists
    if not os.path.exists(folder_path):
        return None, None
        
    zip_files = glob.glob(os.path.join(folder_path, "*.zip"))
    
    if not zip_files:
        return None, None
    
    most_recent_zip = max(zip_files, key=os.path.getmtime)
    current_time = time.time()
    modification_time = os.path.getmtime(most_recent_zip)
    age_in_hours = (current_time - modification_time) / 3600
    return most_recent_zip, age_in_hours

def list_scenarios():
    # Buscamos en la carpeta gym_scenarios que está un nivel arriba de v2
    scenarios_dir = Path('../gym_scenarios')
    if not scenarios_dir.exists():
        print(f"No se encontró el directorio: {scenarios_dir.resolve()}")
        return []
    
    scenarios = []
    # Iterar sobre las carpetas que empiezan con 'gym'
    for item in scenarios_dir.iterdir():
        if item.is_dir() and item.name.startswith('gym'):
            state_file = item / 'gym_scenario.state'
            if state_file.exists():
                scenarios.append((item.name, str(state_file)))
    
    # Ordenar por nombre
    return sorted(scenarios, key=lambda x: x[0])

if __name__ == '__main__':
    print("="*60)
    print(" CARGADOR DE ESCENARIOS DE GIMNASIO (MODO DIRECTO)")
    print("="*60)

    # Configuración directa del escenario (Hardcoded)
    # Puedes cambiar esto manualmente para probar otros gimnasios
    SCENARIO_ID = "gym1_pewter_brock" 
    
    scenario_path = Path(f'../gym_scenarios/{SCENARIO_ID}/gym_scenario.state')
    
    if not scenario_path.exists():
        print(f"Error: No se encuentra el archivo de estado en: {scenario_path}")
        print("Verifica que la carpeta 'gym_scenarios' exista y tenga el nombre correcto.")
        sys.exit(1)

    print(f"\n✓ Cargando escenario: {SCENARIO_ID}")
    print(f"  Archivo de estado: {scenario_path}")

    # 2. Configuración del entorno
    sess_path = Path(f'session_{str(uuid.uuid4())[:8]}')
    ep_length = 2**23  # Episodio muy largo para permitir exploración

    # MODIFICACIÓN: Usar has_pokedex.state como base para evitar bug de pantalla de título
    base_state_path = '../has_pokedex.state'
    if not os.path.exists(base_state_path):
        print(f"Advertencia: No se encontró {base_state_path}, usando init.state")
        base_state_path = '../init.state'

    env_config = {
        'headless': False, 
        'save_final_state': True, 
        'early_stop': False,
        'action_freq': 24, 
        'init_state': base_state_path,  # <--- USAR ESTADO BASE LIMPIO
        'max_steps': ep_length, 
        'print_rewards': True, 
        'save_video': False, 
        'fast_video': True, 
        'session_path': sess_path,
        'gb_path': '../PokemonRed.gb', 
        'debug': False, 
        'sim_frame_dist': 2_000_000.0, 
        'extra_buttons': False
    }
    
    print("\nInicializando entorno...")
    env = make_env(0, env_config)()
    
    # 3. Cargar modelo
    # Intentamos buscar en 'runs' (local) o '../models' (global)
    checkpoint_dirs = ["runs", "../models", "../models_local"]
    most_recent_checkpoint = None
    time_since = 0
    
    for d in checkpoint_dirs:
        cp, ts = get_most_recent_zip_with_age(d)
        if cp:
            most_recent_checkpoint = cp
            time_since = ts
            break
            
    if most_recent_checkpoint is not None:
        file_name = most_recent_checkpoint
        print(f"\n✓ Usando checkpoint: {file_name}")
        print(f"  Antigüedad: {time_since:.1f} horas")
        
        print('\nCargando modelo PPO...')
        try:
            model = PPO.load(file_name, env=env, custom_objects={'lr_schedule': 0, 'clip_range': 0})
        except Exception as e:
            print(f"Error cargando el modelo: {e}")
            print("Intentando cargar sin custom_objects...")
            model = PPO.load(file_name, env=env)
        
        print("\n¡Listo! La simulación comenzará en breve.")
        print("Presiona 'M' en la ventana de PyBoy (si está configurado) o edita 'agent_enabled.txt' para pausar el agente.")
        
        obs, info = env.reset()

        # --- INYECCIÓN DE ESCENARIO ---
        print("\n[!] Inyectando configuración del escenario...")
        target_map_id, target_pos = inject_gym_config(env, scenario_path)
        # ------------------------------
        
        # --- FASE DE CALENTAMIENTO ---
        print("\n[!] Calentando emulador (3 segundos)...")
        print("    Esto permite que el warp se ejecute y el mapa cargue.")
        
        # Reactivamos renderizado para ver si el warp ocurre
        for i in range(180): 
            env.pyboy.tick(1, True)
            if i % 60 == 0:
                env.render()
                
        # Actualizar observación tras el calentamiento e inyección
        try:
            obs = env._get_obs()
        except:
            print("Advertencia: No se pudo obtener observación inicial tras inyección.")
            obs = env.reset() # Fallback
        
        print("✓ Calentamiento finalizado. Iniciando agente.\n")
        # -----------------------------

        # Variables para feedback
        step = 0
        total_reward = 0
        start_time = time.time()
        
        while True:
            try:
                # Control simple mediante archivo de texto
                if os.path.exists("agent_enabled.txt"):
                    with open("agent_enabled.txt", "r") as f:
                        content = f.read().strip()
                        agent_enabled = content.startswith("yes")
                else:
                    # Crear el archivo si no existe
                    with open("agent_enabled.txt", "w") as f:
                        f.write("yes")
                    agent_enabled = True
            except:
                agent_enabled = True

            # --- DELIMITACIÓN DEL MAPA ---
            # Si el agente sale del mapa objetivo, usamos el Warp Seguro para traerlo de vuelta
            current_map = env.read_m(0xD35E) # MAP_N_ADDRESS
            if current_map != target_map_id:
                # print(f"\n[!] Agente salió del mapa. Usando Warp para volver...")
                # 1. Configurar destino
                env.pyboy.set_memory_value(0xD365, target_map_id)
                env.pyboy.set_memory_value(0xD366, target_pos['x'])
                env.pyboy.set_memory_value(0xD367, target_pos['y'])
                # 2. Activar Warp
                current_wd72d = env.pyboy.get_memory_value(0xD12B)
                env.pyboy.set_memory_value(0xD12B, current_wd72d | 0x08)
            # -----------------------------

            if agent_enabled:
                action, _states = model.predict(obs, deterministic=False)
                obs, rewards, terminated, truncated, info = env.step(action)
                total_reward += rewards
            else:
                # Modo manual: el agente no actúa, solo avanza el emulador
                env.pyboy.tick(1, True)
                obs = env._get_obs()
                truncated = env.step_count >= env.max_steps - 1
                rewards = 0
            
            env.render()
            
            # Feedback en consola cada 60 pasos (aprox 1-2 segundos)
            step += 1
            if step % 60 == 0:
                elapsed = time.time() - start_time
                fps = step / (elapsed + 1e-9)
                # Usamos \r para sobrescribir la línea y no llenar la consola
                print(f"\rPasos: {step} | Recompensa Total: {total_reward:.2f} | FPS: {fps:.1f} | Acción: {action if agent_enabled else 'Manual'}   ", end="", flush=True)

            if truncated:
                break
        env.close()
    else:
        print("\n[!] ERROR: No se encontraron checkpoints (.zip) en 'runs', '../models' o '../models_local'.")
        print("Por favor asegúrate de tener un modelo entrenado para ejecutar.")
