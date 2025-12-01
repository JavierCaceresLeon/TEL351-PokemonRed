"""
Quick diagnostic to see what actions the model is taking
"""
from pathlib import Path
from stable_baselines3 import PPO
from red_gym_env_v2 import RedGymEnv
from battle_only_actions import BattleOnlyActions

# Create env
config = {
    'headless': False,
    'save_final_state': False,
    'early_stop': False,
    'action_freq': 24,
    'init_state': 'generated_battle_states/clean_pewter_gym.state',
    'max_steps': 500,
    'print_rewards': False,
    'save_video': False,
    'fast_video': False,
    'session_path': Path('debug_session'),
    'gb_path': 'PokemonRed.gb',
    'debug': False,
    'sim_frame_dist': 2_000_000.0,
    'extra_buttons': False
}

env = RedGymEnv(config)
env = BattleOnlyActions(env)

# Load model
model = PPO.load('sessions/combat_agent_final_battle_loop/combat_agent_final_battle_loop.zip', env=env)

obs, info = env.reset()

print("\n🔍 Diagnostic - First 50 actions:")
print("Action mapping: 0=A, 1=UP, 2=DOWN")
print("-" * 40)

action_counts = {0: 0, 1: 0, 2: 0}

for i in range(50):
    action, _states = model.predict(obs, deterministic=False)
    action_int = int(action.item() if hasattr(action, 'item') else action)
    action_counts[action_int] += 1
    
    action_name = {0: 'A', 1: 'UP', 2: 'DOWN'}[action_int]
    print(f"Step {i+1:3d}: {action_name} (raw={action_int})")
    
    obs, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        print("\nEpisode ended")
        break

print("\n📊 Action distribution:")
print(f"  A (0):    {action_counts[0]} ({action_counts[0]/50*100:.1f}%)")
print(f"  UP (1):   {action_counts[1]} ({action_counts[1]/50*100:.1f}%)")
print(f"  DOWN (2): {action_counts[2]} ({action_counts[2]/50*100:.1f}%)")

env.close()
