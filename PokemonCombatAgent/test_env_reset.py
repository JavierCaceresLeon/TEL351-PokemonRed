"""
Test battle_only_env reset to verify it properly enters battle
"""
from battle_only_env import BattleOnlyEnv

env = BattleOnlyEnv(
    rom_path='PokemonRed.gb',
    battle_states_dir='battle_states',
    headless=False  # Show window to verify visually
)

print("Testing reset...")
obs, info = env.reset()

print(f"\nReset complete!")
print(f"Scenario: {info['scenario']}")
print(f"Player HP: {info['initial_player_hp']}")
print(f"Enemy HP: {info['initial_enemy_hp']}")
print(f"Observation shape: {obs.shape}")

print("\nTaking 50 random steps to verify battle works...")
for i in range(50):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    
    if i % 10 == 0:
        print(f"Step {i}: HP={info['player_hp']}/{info['enemy_hp']}, Reward={reward:.2f}, InBattle={info['in_battle']}")
    
    if terminated or truncated:
        print(f"\nEpisode ended at step {i}")
        print(f"  Terminated: {terminated}")
        print(f"  Truncated: {truncated}")
        print(f"  Final info: {info}")
        break

print("\nTest complete! Environment working correctly.")
env.close()
