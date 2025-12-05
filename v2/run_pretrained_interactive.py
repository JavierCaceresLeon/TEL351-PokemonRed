import os
from os.path import exists
from pathlib import Path
import uuid
import time
import glob
import sys
import numpy as np
from pathlib import Path

# Add v2 directory to path to enable local imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))
# Add parent directory to enable importing advanced_agents and other modules
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))

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

    # Usar estado del gimnasio de Brock desde gym_scenarios
    gym_state = str(parent_dir / "gym_scenarios" / "state_files" / "pewter_battle.state")
    
    env_config = {
                'headless': False, 'save_final_state': True, 'early_stop': False,
                'action_freq': 24, 'init_state': gym_state, 'max_steps': ep_length, 
                'print_rewards': True, 'save_video': False, 'fast_video': True, 'session_path': sess_path,
                'gb_path': str(parent_dir / 'PokemonRed.gb'), 'debug': False, 'sim_frame_dist': 2_000_000.0, 'extra_buttons': False
            }
    
    print(f"Cargando estado: {gym_state}")
    
    num_cpu = 1 #64 #46  # Also sets the number of episodes per training iteration
    env = make_env(0, env_config)() #SubprocVecEnv([make_env(i, env_config) for i in range(num_cpu)])
    
    # Cargar el estado ANTES del env_checker para evitar corrupcion
    with open(gym_state, "rb") as f:
        env.pyboy.load_state(f)
    print("Estado del gimnasio cargado correctamente")
    
    # Saltar env_checker - puede causar problemas con estados personalizados
    # env_checker.check_env(env)
    
    # Directly specify the checkpoint to use
    file_name = "C:\\Users\\Cris\\Documents\\GitHub\\TEL351-PokemonRed\\PokemonCombatAgent\\combat_agent_final_continued.zip"
    
    # Verify the file exists
    if not os.path.exists(file_name):
        print(f"ERROR: Checkpoint not found: {file_name}")
        sys.exit(1)
    
    print(f"Using checkpoint: {file_name}")
    model = PPO.load(file_name, env=env, custom_objects={'lr_schedule': 0, 'clip_range': 0})
        
    # Primero hacer reset normal para inicializar todas las variables
    obs, info = env.reset()
    
    # Luego recargar el estado del gimnasio
    with open(gym_state, "rb") as f:
        env.pyboy.load_state(f)
    print("Estado del gimnasio recargado - Ahora en el gimnasio de Brock")
    
    # Actualizar la observacion con el nuevo estado
    obs = env._get_obs()
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


