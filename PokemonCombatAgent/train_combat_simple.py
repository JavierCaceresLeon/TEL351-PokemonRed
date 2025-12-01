"""
Train Combat Agent using proven RedGymEnv with combat-focused rewards
"""

import argparse
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
import torch

from combat_focused_env import CombatFocusedEnv, make_combat_env_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--state', type=str, default='has_pokedex_nballs.state',
                        help='Initial state file')
    parser.add_argument('--timesteps', type=int, default=500_000)
    parser.add_argument('--session-name', type=str, default='combat_agent')
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--checkpoint-freq', type=int, default=50_000)
    
    args = parser.parse_args()
    
    # Setup paths
    session_path = Path('sessions') / args.session_name
    session_path.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("Training Combat-Focused Pokemon Red Agent")
    print("="*60)
    print(f"Initial state: {args.state}")
    print(f"Timesteps: {args.timesteps:,}")
    print(f"Session: {session_path}")
    print(f"Headless: {args.headless}")
    
    # GPU detection
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU")
    print("="*60)
    print()
    
    # Create environment
    config = make_combat_env_config(
        state_file=args.state,
        session_path=session_path,
        headless=args.headless
    )
    
    env = CombatFocusedEnv(config)
    
    # Detect policy type based on observation space
    from gymnasium import spaces
    if isinstance(env.observation_space, spaces.Dict):
        policy = 'MultiInputPolicy'
        print("Using MultiInputPolicy (Dict observation space)")
    else:
        policy = 'CnnPolicy'
        print("Using CnnPolicy (array observation space)")
    
    # Create model
    print("Creating PPO model...")
    model = PPO(
        policy,
        env,
        verbose=1,
        n_steps=2048,
        batch_size=128,
        n_epochs=4,
        gamma=0.99,
        learning_rate=0.0003,
        ent_coef=0.01,
        device=device,
        tensorboard_log=str(session_path)
    )
    
    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=args.checkpoint_freq,
        save_path=str(session_path / 'checkpoints'),
        name_prefix='combat_agent',
        save_replay_buffer=False
    )
    
    print(f"\nStarting training for {args.timesteps:,} timesteps...")
    print()
    
    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=checkpoint_callback,
            progress_bar=True
        )
        
        # Save final model
        final_path = session_path / 'combat_agent_final.zip'
        model.save(str(final_path))
        print(f"\n✅ Training complete! Model saved to: {final_path}")
        
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted")
        interrupt_path = session_path / 'combat_agent_interrupted.zip'
        model.save(str(interrupt_path))
        print(f"Saved to: {interrupt_path}")
    
    finally:
        env.close()


if __name__ == '__main__':
    main()
