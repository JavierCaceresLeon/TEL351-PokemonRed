import os
from os.path import exists
from pathlib import Path
import uuid
import time
import glob
from red_gym_env_v2 import RedGymEnv
from stable_baselines3 import A2C, PPO
from stable_baselines3.common import env_checker
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import CheckpointCallback

def make_env(rank, env_conf, seed=0):
    """
    Utility function for multiprocessed env.
    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the initial seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        env = RedGymEnv(env_conf)
        #env.seed(seed + rank)
        return env
    set_random_seed(seed)
    return _init

def get_most_recent_zip_with_age(folder_path):
    # Get all zip files in the folder
    zip_files = glob.glob(os.path.join(folder_path, "*.zip"))
    
    if not zip_files:
        return None, None  # Return None if no zip files are found
    
    # Find the most recently modified zip file
    most_recent_zip = max(zip_files, key=os.path.getmtime)
    
    # Calculate how old the file is in hours
    current_time = time.time()
    modification_time = os.path.getmtime(most_recent_zip)
    age_in_hours = (current_time - modification_time) / 3600  # Convert seconds to hours
    
    return most_recent_zip, age_in_hours

if __name__ == '__main__':

    sess_path = Path(f'session_{str(uuid.uuid4())[:8]}')
    ep_length = 2**23
    
    # ========================================================================
    # CONFIGURACIÓN: Usar el mismo escenario que el agente especializado
    # ========================================================================
    SCENARIO_ID = 'pewter_brock'  # Cambia a 'cerulean_misty', etc. según necesites
    PHASE_NAME = 'battle'         # 'battle' o 'puzzle'
    
    # Cargar configuración del escenario desde scenarios.json
    import json
    scenarios_path = 'gym_scenarios/scenarios.json'
    if os.path.exists(scenarios_path):
        with open(scenarios_path, 'r') as f:
            scenarios_data = json.load(f)
        
        # Buscar el escenario
        scenario = next((s for s in scenarios_data['scenarios'] if s['id'] == SCENARIO_ID), None)
        if scenario:
            # Buscar la fase específica
            phase = next((p for p in scenario['phases'] if p['name'] == PHASE_NAME), None)
            if phase:
                state_file = phase['state_file']
                print(f"✅ Usando escenario: {SCENARIO_ID} ({PHASE_NAME})")
                print(f"   State file: {state_file}")
            else:
                print(f"⚠️  Fase '{PHASE_NAME}' no encontrada en {SCENARIO_ID}, usando default")
                state_file = 'init.state'
        else:
            print(f"⚠️  Escenario '{SCENARIO_ID}' no encontrado, usando default")
            state_file = 'init.state'
    else:
        print(f"⚠️  {scenarios_path} no encontrado, usando default")
        state_file = 'init.state'
    
    # Verificar que el archivo .state existe
    if not os.path.exists(state_file):
        print(f"❌ ERROR: State file no encontrado: {state_file}")
        print("Genera los archivos .state con: python generate_gym_states.py")
        exit(1)

    env_config = {
                'headless': False, 'save_final_state': True, 'early_stop': False,
                'action_freq': 24, 'init_state': state_file, 'max_steps': ep_length, 
                'print_rewards': True, 'save_video': False, 'fast_video': True, 'session_path': sess_path,
                'gb_path': 'PokemonRed.gb', 'debug': False, 'sim_frame_dist': 2_000_000.0, 'extra_buttons': False
            }
    
    num_cpu = 1 #64 #46  # Also sets the number of episodes per training iteration
    env = make_env(0, env_config)() #SubprocVecEnv([make_env(i, env_config) for i in range(num_cpu)])
    
    #env_checker.check_env(env)
    most_recent_checkpoint, time_since = get_most_recent_zip_with_age("runs")
    file_name = None
    if most_recent_checkpoint is not None:
        file_name = most_recent_checkpoint
        print(f"using checkpoint: {file_name}, which is {time_since} hours old")
    else:
        # Intentar con el modelo conocido
        file_name = "v2/runs/poke_26214400.zip"
        if not os.path.exists(file_name):
            print(f"ERROR: No se encontró checkpoint en 'runs/' ni en {file_name}")
            print("Descarga el modelo baseline o especifica la ruta correcta.")
            exit(1)
        print(f"Usando modelo por defecto: {file_name}")
    
    # could optionally manually specify a checkpoint here
    #file_name = "runs/poke_41943040_steps.zip"
    print('\nloading checkpoint')
    model = PPO.load(file_name, env=env, custom_objects={'lr_schedule': 0, 'clip_range': 0})
        
    #keyboard.on_press_key("M", toggle_agent)
    obs, info = env.reset()
    while True:
        try:
            with open("agent_enabled.txt", "r") as f:
                agent_enabled = f.readlines()[0].startswith("yes")
        except:
            agent_enabled = False
        if agent_enabled:
            action, _states = model.predict(obs, deterministic=False)
            obs, rewards, terminated, truncated, info = env.step(action)
        else:
            env.pyboy.tick(1, True)
            obs = env._get_obs()
            truncated = env.step_count >= env.max_steps - 1
        env.render()
        if truncated:
            break
    env.close()


