from os.path import exists
import argparse
import glob
import os
from pathlib import Path
import uuid
from red_gym_env import RedGymEnv
from stable_baselines3 import A2C, PPO
from stable_baselines3.common import env_checker
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import CheckpointCallback
import cv2
import numpy as np
from global_map import local_to_global

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

if __name__ == '__main__':

    sess_path = Path(f'session_{str(uuid.uuid4())[:8]}')
    ep_length = 2**23

    env_config = {
                'headless': False, 'save_final_state': True, 'early_stop': False,
                'action_freq': 24, 'init_state': '../has_pokedex_nballs.state', 'max_steps': ep_length, 
                'print_rewards': True, 'save_video': False, 'fast_video': True, 'session_path': sess_path,
                'gb_path': '../PokemonRed.gb', 'debug': False, 'sim_frame_dist': 2_000_000.0, 'extra_buttons': True
            }
    
    num_cpu = 1 #64 #46  # Also sets the number of episodes per training iteration
    env = make_env(0, env_config)() #SubprocVecEnv([make_env(i, env_config) for i in range(num_cpu)])
    
    #env_checker.check_env(env)
    file_name = 'session_4da05e87_main_good/poke_439746560_steps'

    parser = argparse.ArgumentParser(description="Run pretrained model interactively")
    parser.add_argument('--checkpoint', '-c', default=file_name, help='Path to a checkpoint file or folder (will search for poke_*_steps.zip)')
    parser.add_argument('--no-model', action='store_true', help='Run without loading a model')
    parser.add_argument('--show-map', action='store_true', help='Show the agent position on the full map')
    args = parser.parse_args()

    def find_latest_checkpoint(path_str):
        if not path_str:
            return None
        p = Path(path_str)
        # If path given points directly to a file, return it
        if p.exists() and p.is_file():
            return p
        # Sometimes the file is provided without a .zip suffix
        if p.with_suffix('.zip').exists():
            return p.with_suffix('.zip')
        # If path is a directory, search for poke_*_steps.zip
        if p.exists() and p.is_dir():
            zips = sorted(p.glob('poke_*_steps.zip'), key=os.path.getmtime, reverse=True)
            if zips:
                return zips[0]
        # Search parent directory for poke_*_steps.zip
        parent = p.parent if p.parent.exists() else Path('.')
        zips = sorted(parent.glob('poke_*_steps.zip'), key=os.path.getmtime, reverse=True)
        if zips:
            return zips[0]
        # Search across session_* folders as a last resort
        all_zips = sorted(Path('.').glob('session_*/poke_*_steps.zip'), key=os.path.getmtime, reverse=True)
        if all_zips:
            return all_zips[0]
        return None

    print('\nloading checkpoint')
    model = None
    if not args.no_model:
        candidate = find_latest_checkpoint(args.checkpoint)
        if candidate is None:
            print(f'Checkpoint not found: {args.checkpoint} — running without model.\nYou can pass --checkpoint PATH or place a poke_*_steps.zip under a session_* folder.')
        else:
            # stable-baselines3 can accept the path without the .zip suffix — normalize accordingly
            try:
                load_path = str(candidate)
                if load_path.endswith('.zip'):
                    load_path = str(Path(load_path).with_suffix(''))
                print(f'Loading checkpoint: {candidate}')
                model = PPO.load(load_path, env=env, custom_objects={'lr_schedule': 0, 'clip_range': 0})
            except Exception as e:
                print('Error loading model; proceeding without model. Exception:', e)
                model = None
        
    #keyboard.on_press_key("M", toggle_agent)
    obs, info = env.reset()
    
    # Map visualization setup
    if args.show_map:
        script_dir = Path(__file__).parent
        map_img_path = script_dir.parent / 'visualization' / 'poke_map' / 'pokemap_full_calibrated_CROPPED_1.png'
        
        if map_img_path.exists():
            full_map_bg = cv2.imread(str(map_img_path))
            # Resize for better visibility if needed, but let's keep original first
            print(f"Map loaded from {map_img_path}")
        else:
            print(f"Map image not found at {map_img_path}. Map visualization disabled.")
            args.show_map = False

    while True:
        action = 7 # pass action
        try:
            with open("agent_enabled.txt", "r") as f:
                agent_enabled = f.readlines()[0].startswith("yes")
        except:
            agent_enabled = False
        if agent_enabled:
            if model is None:
                # No model available — use a random action if agent requested
                action = env.action_space.sample()
            else:
                action, _states = model.predict(obs, deterministic=False)
        obs, rewards, terminated, truncated, info = env.step(action)
        env.render()
        
        if args.show_map:
            try:
                # Get coordinates
                x_pos = env.read_m(0xD362)
                y_pos = env.read_m(0xD361)
                map_n = env.read_m(0xD35E)
                
                # Convert to global
                gy, gx = local_to_global(y_pos, x_pos, map_n)
                
                # Update the background with the path (small blue dot for history)
                cv2.circle(full_map_bg, (gx, gy), 2, (255, 0, 0), -1) 
                
                # Create a view for the current frame
                map_view = full_map_bg.copy()
                
                # Draw current position (larger red dot)
                cv2.circle(map_view, (gx, gy), 4, (0, 0, 255), -1) 
                
                # Show
                cv2.imshow('Global Map', map_view)
                cv2.waitKey(1)
            except Exception as e:
                print(f"Error updating map: {e}")

        if args.no_model:
            # No model — random action
            action = env.action_space.sample()
            obs, rewards, terminated, truncated, info = env.step(action)
            env.render()
            continue

        if truncated:
            break
    env.close()
