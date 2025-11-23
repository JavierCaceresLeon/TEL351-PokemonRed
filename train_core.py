import json
import os
import sys

# FIX: Allow multiple OpenMP runtimes (common issue with PyTorch on Windows)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from pathlib import Path
import types

# Ensure project and baselines modules are importable before local imports
project_path = os.getcwd()
if project_path not in sys.path:
    sys.path.append(project_path)

baselines_path = os.path.join(project_path, 'baselines')
if baselines_path not in sys.path:
    sys.path.append(baselines_path)

from gymnasium import spaces
from stable_baselines3 import PPO
from v2.red_gym_env_v2 import RedGymEnv

from advanced_agents.train_agents import _base_env_config
from advanced_agents.combat_apex_agent import CombatApexAgent, CombatAgentConfig
from advanced_agents.puzzle_speed_agent import PuzzleSpeedAgent, PuzzleAgentConfig
from advanced_agents.hybrid_sage_agent import HybridSageAgent, HybridAgentConfig

def run_training(agent_type, scenario_id, phase_name, timesteps, headless, run_name_prefix=""):
    # Load scenario
    scenarios_path = Path('gym_scenarios/scenarios.json')
    if not scenarios_path.exists():
        raise FileNotFoundError("gym_scenarios/scenarios.json not found")
        
    with open(scenarios_path, 'r') as f:
        scenarios_data = json.load(f)
    
    scenario = next((s for s in scenarios_data['scenarios'] if s['id'] == scenario_id), None)
    if not scenario:
        raise ValueError(f"Scenario {scenario_id} not found")
        
    phase = next((p for p in scenario['phases'] if p['name'] == phase_name), None)
    if not phase:
        raise ValueError(f"Phase {phase_name} not found in {scenario_id}")
        
    state_file_path = phase['state_file']
    
    # Fallback logic for state file
    if not os.path.exists(state_file_path):
        print(f"Warning: State file {state_file_path} not found.")
        if os.path.exists('init.state'):
             print("Using init.state as fallback.")
             state_file_path = 'init.state'
        else:
             # Try to find any state file in the directory
             state_dir = Path('gym_scenarios/state_files')
             if state_dir.exists():
                 states = list(state_dir.glob('*.state'))
                 if states:
                     state_file_path = str(states[0])
                     print(f"Using fallback state from directory: {state_file_path}")
                 else:
                     raise FileNotFoundError(f"State file {state_file_path} and no fallback states found.")
             else:
                 raise FileNotFoundError(f"State file {state_file_path} and init.state not found.")

    print(f"Using state file: {state_file_path}")

    # Env config
    env_overrides = {
        "init_state": state_file_path,
        "headless": headless,
        "save_video": False,
        "gb_path": "PokemonRed.gb",
        "session_path": Path(f"sessions/{agent_type}_{scenario_id}_{phase_name}"),
        # Ensure we use the correct render mode for local visualization if not headless
        "render_mode": "rgb_array" if headless else "human",
        # Avoid ultra-fast emulation when showing the SDL2 window
        "fast_video": headless
    }
    
    # Agent Config
    if agent_type == "baseline":
        # Baseline uses standard PPO with RedGymEnv
        env_config = _base_env_config(env_overrides)
        env = RedGymEnv(env_config)
        
        # Handle Dict observation space for baseline too
        if isinstance(env.observation_space, spaces.Dict):
            policy = "MultiInputPolicy"
        else:
            policy = "CnnPolicy"
            
        print(f"Starting training: {agent_type} on {scenario_id} ({phase_name})")
        print(f"Headless: {headless}")
        
        model = PPO(policy, env, verbose=1, tensorboard_log="advanced_agents/runs")
        try:
            model.learn(total_timesteps=timesteps)
        except KeyboardInterrupt:
            print("Training interrupted by user. Saving model...")
            
        save_dir = Path("models")
        save_dir.mkdir(exist_ok=True)
        save_path = save_dir / f"{agent_type}_{scenario_id}_{phase_name}"
        model.save(save_path)
        print(f"Model saved to {save_path}")
        return

    elif agent_type == "combat":
        config_cls = CombatAgentConfig
        agent_cls = CombatApexAgent
    elif agent_type == "puzzle":
        config_cls = PuzzleAgentConfig
        agent_cls = PuzzleSpeedAgent
    elif agent_type == "hybrid":
        config_cls = HybridAgentConfig
        agent_cls = HybridSageAgent
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")
        
    config = config_cls(
        env_config=_base_env_config(env_overrides),
        total_timesteps=timesteps,
        tensorboard_log="advanced_agents/runs"
    )
    
    agent = agent_cls(config)
    
    # Dict observation space fix
    env_for_check = agent.make_env()
    obs_space = getattr(env_for_check, 'observation_space', None)
    if isinstance(obs_space, spaces.Dict):
        print("Dict observation space detected -> switching policy to MultiInputPolicy")
        agent.policy_name = types.MethodType(lambda self: "MultiInputPolicy", agent)
    env_for_check.close()
        
    print(f"Starting training: {agent_type} on {scenario_id} ({phase_name})")
    print(f"Headless: {headless}")
    
    runtime = None
    try:
        runtime = agent.train()
    except KeyboardInterrupt:
        print("Training interrupted by user. Attempting to save partial model (if available)...")
    
    model = getattr(runtime, "model", None)
    if model is None:
        model = getattr(agent, "_model", None)
    
    if model is None:
        print("No trained model available to save.")
        return

    save_dir = Path("models")
    save_dir.mkdir(exist_ok=True)
    save_path = save_dir / f"{agent_type}_{scenario_id}_{phase_name}"
    model.save(save_path)
    print(f"Model saved to {save_path}")
