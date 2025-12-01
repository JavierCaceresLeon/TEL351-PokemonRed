"""
Interactive Demo: Watch trained combat agent play Pokemon Red

This script loads a trained model and lets you watch it play,
or take control manually.
"""

import argparse
import time
from pathlib import Path
from combat_gym_env import CombatGymEnv
from stable_baselines3 import PPO


def run_interactive(model_path, num_episodes=5, enable_agent=True):
    """
    Run trained agent interactively.
    
    Args:
        model_path: Path to trained model checkpoint
        num_episodes: Number of episodes to run
        enable_agent: If True, agent plays. If False, human control.
    """
    
    print(f"=" * 60)
    print(f"Interactive Combat Agent Demo")
    print(f"=" * 60)
    print(f"Model: {model_path}")
    print(f"Episodes: {num_episodes}")
    print(f"Mode: {'Agent' if enable_agent else 'Human'}")
    print(f"=" * 60)
    
    # Load model
    if enable_agent:
        print("\nLoading trained model...")
        model = PPO.load(model_path)
        print("Model loaded successfully!")
    
    # Environment config (NOT headless, we want to see it)
    env_config = {
        'headless': False,  # Show GUI
        'save_final_state': True,
        'early_stop': False,
        'action_freq': 24,
        'init_state': '../has_pokedex_nballs.state',
        'max_steps': 5000,
        'print_rewards': True,
        'save_video': False,
        'session_path': Path('./demo_session'),
        'gb_path': '../PokemonRed.gb',
        'debug': False,
        'combat_focus': True,
        'reward_scale': 1.0,
    }
    
    # Run episodes
    for episode in range(num_episodes):
        print(f"\n{'=' * 60}")
        print(f"Episode {episode + 1}/{num_episodes}")
        print(f"{'=' * 60}\n")
        
        env = CombatGymEnv(env_config)
        obs, _ = env.reset()
        
        episode_reward = 0
        step = 0
        done = False
        truncated = False
        
        while not (done or truncated):
            # Agent or human control
            if enable_agent:
                action, _ = model.predict(obs, deterministic=True)
            else:
                # For human control, could implement keyboard input
                # For now, just use random actions
                action = env.action_space.sample()
            
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            step += 1
            
            # Slow down for human viewing
            time.sleep(0.01)  # 100 FPS (adjust as needed)
            
            # Print progress every 100 steps
            if step % 100 == 0:
                print(f"  Step {step}: Reward={episode_reward:.2f}, "
                      f"HP={env.read_hp_fraction():.2%}, "
                      f"Battles={env.battles_won}/{env.battles_lost}")
            
            # Safety limit
            if step > 10000:
                print("  (Reached safety limit)")
                break
        
        # Episode summary
        print(f"\n{'-' * 60}")
        print(f"Episode {episode + 1} Summary:")
        print(f"  Total Reward: {episode_reward:.2f}")
        print(f"  Steps: {step}")
        print(f"  Battles Won: {env.battles_won}")
        print(f"  Battles Lost: {env.battles_lost}")
        print(f"  Win Rate: {env.battles_won / max(env.battles_won + env.battles_lost, 1):.1%}")
        print(f"  Final HP: {env.read_hp_fraction():.2%}")
        print(f"  Deaths: {env.died_count}")
        print(f"  Badges: {env.get_badges()}")
        print(f"{'-' * 60}")
        
        # Wait between episodes
        if episode < num_episodes - 1:
            print("\nPress Enter for next episode...")
            input()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Watch trained combat agent play interactively',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained model (without .zip extension)')
    parser.add_argument('--episodes', type=int, default=5,
                        help='Number of episodes to run')
    parser.add_argument('--agent', action='store_true',
                        help='Let agent play (default)')
    parser.add_argument('--human', dest='agent', action='store_false',
                        help='Human control (random for now)')
    parser.set_defaults(agent=True)
    
    args = parser.parse_args()
    
    run_interactive(args.model, args.episodes, args.agent)
