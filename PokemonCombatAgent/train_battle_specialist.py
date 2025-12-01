"""
Train Battle Specialist Agent
==============================

Trains a PPO agent EXCLUSIVELY on gym leader battle scenarios.
Uses battle_only_env.py which rotates through saved battle states.
"""

import argparse
import os
import platform
from datetime import datetime
import uuid
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.callbacks import CheckpointCallback

from battle_only_env import BattleOnlyEnv


def make_env(rom_path, battle_states_dir, headless=True):
    """Create a battle-only environment."""
    def _init():
        return BattleOnlyEnv(
            rom_path=rom_path,
            battle_states_dir=battle_states_dir,
            output_shape=(128, 40),
            max_steps=2000,  # Enough time for complete battle
            headless=headless
        )
    return _init


def main(args):
    """Main training loop."""
    
    # Generate session ID
    session_id = uuid.uuid4().hex[:8]
    session_name = f"battle_specialist_{session_id}"
    session_dir = os.path.join('sessions', session_name)
    os.makedirs(session_dir, exist_ok=True)
    
    print("=" * 60)
    print("Training Battle Specialist Agent")
    print("=" * 60)
    print(f"Session: {session_name}")
    print(f"Output: {session_dir}")
    print(f"Timesteps: {args.timesteps:,}")
    print(f"Parallel envs: {args.num_envs}")
    print(f"Battle states: {args.battle_states_dir}")
    print("=" * 60)
    print()
    
    # Create vectorized environments
    print(f"Initializing {args.num_envs} parallel battle environments...")
    
    # Use DummyVecEnv (thread-based) for Windows compatibility
    if platform.system() == 'Windows':
        print("Using DummyVecEnv (thread-based) for Windows compatibility")
        env = DummyVecEnv([
            make_env(args.rom, args.battle_states_dir, args.headless)
            for _ in range(args.num_envs)
        ])
    else:
        # Could use SubprocVecEnv on Linux/Mac for better parallelism
        env = DummyVecEnv([
            make_env(args.rom, args.battle_states_dir, args.headless)
            for _ in range(args.num_envs)
        ])
    
    # Wrap for channel-first images (PyTorch expects [C, H, W])
    env = VecTransposeImage(env)
    
    print()
    
    # GPU detection
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        print(f"GPU detected: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
    else:
        print("No GPU detected, using CPU")
    
    # Create or load model
    model_path = os.path.join(session_dir, 'battle_specialist.zip')
    
    if args.resume and os.path.exists(model_path):
        print(f"\nResuming from checkpoint: {model_path}")
        model = PPO.load(model_path, env=env, device=device)
    else:
        print("\nCreating new PPO model with battle-optimized configuration...")
        print(f"  Device: {device.upper()} ({'GPU acceleration enabled!' if device=='cuda' else 'CPU only'})")
        print(f"  Policy: CnnPolicy")
        print(f"  n_steps: {args.max_steps}")
        print(f"  batch_size: {args.batch_size}")
        print(f"  n_epochs: {args.n_epochs}")
        print(f"  gamma: 0.99")
        print(f"  learning_rate: {args.learning_rate}")
        print(f"  ent_coef: {args.ent_coef}")
        
        model = PPO(
            'CnnPolicy',
            env,
            verbose=1,
            n_steps=args.max_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=0.99,
            learning_rate=args.learning_rate,
            ent_coef=args.ent_coef,
            device=device,
            tensorboard_log=session_dir
        )
    
    # Setup checkpointing
    checkpoint_callback = CheckpointCallback(
        save_freq=args.checkpoint_freq // args.num_envs,
        save_path=session_dir,
        name_prefix='battle_specialist',
        save_replay_buffer=False,
        save_vecnormalize=False
    )
    
    # Training info
    steps_per_env = args.max_steps
    total_steps_per_update = steps_per_env * args.num_envs
    num_updates = args.timesteps // total_steps_per_update
    
    print()
    print("=" * 60)
    print("Starting training...")
    print(f"Total timesteps: {args.timesteps:,}")
    print(f"Steps per environment: {steps_per_env:,}")
    print(f"Timesteps per iteration: {total_steps_per_update:,}")
    print(f"Total iterations: {num_updates}")
    print(f"Checkpoint frequency: {args.checkpoint_freq:,} timesteps")
    print("=" * 60)
    print()
    
    # Train
    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=checkpoint_callback,
            progress_bar=True
        )
        
        # Save final model
        final_path = os.path.join(session_dir, 'battle_specialist_final.zip')
        model.save(final_path)
        print(f"\n✅ Training complete! Final model saved to: {final_path}")
        
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
        interrupt_path = os.path.join(session_dir, 'battle_specialist_interrupted.zip')
        model.save(interrupt_path)
        print(f"Saved interrupted model to: {interrupt_path}")
    
    finally:
        env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train battle specialist PPO agent')
    
    # Environment
    parser.add_argument('--rom', type=str, default='PokemonRed.gb',
                        help='Path to Pokemon Red ROM')
    parser.add_argument('--battle-states-dir', type=str, default='battle_states',
                        help='Directory containing *_battle.state files')
    parser.add_argument('--headless', action='store_true',
                        help='Run without display window')
    
    # Training
    parser.add_argument('--timesteps', type=int, default=500_000,
                        help='Total timesteps to train (default: 500K for battle specialist)')
    parser.add_argument('--num-envs', type=int, default=2,
                        help='Number of parallel environments (2-4 recommended)')
    parser.add_argument('--checkpoint-freq', type=int, default=25_000,
                        help='Save checkpoint every N timesteps')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from latest checkpoint')
    
    # PPO hyperparameters (optimized for battles)
    parser.add_argument('--max-steps', type=int, default=2048,
                        help='Steps per environment per update (balanced for battle completion)')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='PPO batch size')
    parser.add_argument('--n-epochs', type=int, default=4,
                        help='PPO epochs per update')
    parser.add_argument('--learning-rate', type=float, default=0.0003,
                        help='Learning rate')
    parser.add_argument('--ent-coef', type=float, default=0.01,
                        help='Entropy coefficient for exploration')
    
    args = parser.parse_args()
    main(args)
