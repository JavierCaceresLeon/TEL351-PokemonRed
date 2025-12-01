"""
Training script for Combat-Specialized Pokemon Red Agent

Based on PokemonRedExperiments/baselines/run_baseline_parallel.py
but configured for combat scenarios.
"""

import argparse
from pathlib import Path
import uuid
import platform
from combat_gym_env import CombatGymEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import CheckpointCallback


def make_env(rank, env_conf, seed=0):
    """
    Create environment instance for parallel training.
    
    Args:
        rank: Process rank (for seeding)
        env_conf: Environment configuration dict
        seed: Base random seed
    """
    def _init():
        env = CombatGymEnv(env_conf)
        env.reset(seed=(seed + rank))
        return env
    
    set_random_seed(seed)
    return _init


def main(args):
    """Main training loop"""
    
    # Create session directory
    session_name = args.session_name if args.session_name else f'combat_session_{str(uuid.uuid4())[:8]}'
    sess_path = Path(args.output_dir) / session_name
    sess_path.mkdir(parents=True, exist_ok=True)
    
    print(f"=" * 60)
    print(f"Training Combat-Specialized Pokemon Red Agent")
    print(f"=" * 60)
    print(f"Session: {session_name}")
    print(f"Output: {sess_path}")
    print(f"Timesteps: {args.timesteps:,}")
    print(f"Parallel envs: {args.num_envs}")
    print(f"=" * 60)
    
    # Environment configuration
    env_config = {
        'headless': args.headless,
        'save_final_state': True,
        'early_stop': False,
        'action_freq': args.action_freq,
        'init_state': args.init_state,
        'max_steps': args.max_steps,
        'print_rewards': False,  # Disabled for cleaner training output
        'save_video': False,
        'fast_video': True,
        'session_path': sess_path,
        'gb_path': args.rom_path,
        'debug': False,
        'reward_scale': args.reward_scale,
        
        # Combat-specific config
        'combat_focus': True,
        'type_bonus_scale': 20.0,
        'hp_efficiency_scale': 50.0,
    }
    
    # Create parallel environments
    print(f"\nInitializing {args.num_envs} parallel environments...")
    
    # Use DummyVecEnv on Windows (thread-based) to avoid multiprocessing issues with PyBoy
    # Use SubprocVecEnv on Linux/Mac (process-based) for better performance
    if platform.system() == 'Windows':
        print("Using DummyVecEnv (thread-based) for Windows compatibility")
        env = DummyVecEnv([make_env(i, env_config, seed=args.seed) for i in range(args.num_envs)])
    else:
        print("Using SubprocVecEnv (process-based) for optimal performance")
        env = SubprocVecEnv([make_env(i, env_config, seed=args.seed) for i in range(args.num_envs)])
    
    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=args.checkpoint_freq,
        save_path=sess_path,
        name_prefix='combat_agent'
    )
    
    # Detect device (CUDA GPU if available, otherwise CPU)
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load or create model
    if args.load_checkpoint and Path(args.load_checkpoint + '.zip').exists():
        print(f"\nLoading checkpoint: {args.load_checkpoint}")
        model = PPO.load(args.load_checkpoint, env=env, device=device)
        
        # Update configuration (in case we changed number of envs)
        model.n_steps = args.max_steps
        model.n_envs = args.num_envs
        model.rollout_buffer.buffer_size = args.max_steps
        model.rollout_buffer.n_envs = args.num_envs
        model.rollout_buffer.reset()
        
    else:
        print("\nCreating new PPO model with combat-optimized configuration...")
        print(f"  Device: {device.upper()} {'(GPU acceleration enabled!)' if device == 'cuda' else '(CPU mode)'}")
        print(f"  Policy: CnnPolicy")
        print(f"  n_steps: {args.max_steps}")
        print(f"  batch_size: {args.batch_size}")
        print(f"  n_epochs: {args.n_epochs}")
        print(f"  gamma: {args.gamma}")
        print(f"  learning_rate: {args.learning_rate}")
        print(f"  ent_coef: {args.ent_coef}")
        
        model = PPO(
            'CnnPolicy',
            env,
            verbose=1,
            n_steps=args.max_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=args.gamma,
            learning_rate=args.learning_rate,
            ent_coef=args.ent_coef,
            clip_range=args.clip_range,
            device=device,  # Use GPU if available
        )
    
    # Training loop
    print(f"\n{'=' * 60}")
    print(f"Starting training...")
    print(f"Total timesteps: {args.timesteps:,}")
    print(f"Timesteps per iteration: {args.max_steps * args.num_envs:,}")
    print(f"Total iterations: {args.timesteps // (args.max_steps * args.num_envs)}")
    print(f"{'=' * 60}\n")
    
    total_timesteps_per_iteration = args.max_steps * args.num_envs
    
    model.learn(
        total_timesteps=args.timesteps,
        callback=checkpoint_callback,
        progress_bar=True
    )
    
    # Save final model
    final_model_path = sess_path / 'combat_agent_final'
    model.save(final_model_path)
    print(f"\n{'=' * 60}")
    print(f"Training completed!")
    print(f"Final model saved to: {final_model_path}.zip")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train Combat-Specialized Pokemon Red Agent',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Paths
    parser.add_argument('--rom-path', type=str, default='./PokemonRed.gb',
                        help='Path to Pokemon Red ROM')
    parser.add_argument('--init-state', type=str, default='./has_pokedex_nballs.state',
                        help='Initial game state file')
    parser.add_argument('--output-dir', type=str, default='./sessions',
                        help='Output directory for sessions')
    parser.add_argument('--session-name', type=str, default='',
                        help='Session name (auto-generated if not provided)')
    
    # Training parameters
    parser.add_argument('--timesteps', type=int, default=1_000_000,
                        help='Total training timesteps')
    parser.add_argument('--num-envs', type=int, default=4,
                        help='Number of parallel environments (4 recommended for Windows DummyVecEnv)')
    parser.add_argument('--max-steps', type=int, default=2048,
                        help='Steps per environment per update (reduced for memory efficiency)')
    parser.add_argument('--checkpoint-freq', type=int, default=50_000,
                        help='Save checkpoint every N steps')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed')
    
    # PPO hyperparameters
    parser.add_argument('--batch-size', type=int, default=128,
                        help='PPO batch size (reduced for memory efficiency)')
    parser.add_argument('--n-epochs', type=int, default=4,
                        help='PPO number of epochs per update (increased to compensate for smaller batch)')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='Discount factor')
    parser.add_argument('--learning-rate', type=float, default=3e-4,
                        help='Learning rate')
    parser.add_argument('--ent-coef', type=float, default=0.01,
                        help='Entropy coefficient')
    parser.add_argument('--clip-range', type=float, default=0.2,
                        help='PPO clip range')
    
    # Environment parameters
    parser.add_argument('--action-freq', type=int, default=24,
                        help='Frames per action')
    parser.add_argument('--reward-scale', type=float, default=1.0,
                        help='Global reward scaling factor')
    parser.add_argument('--headless', action='store_true',
                        help='Run without GUI')
    parser.add_argument('--no-headless', dest='headless', action='store_false',
                        help='Show GUI (for debugging)')
    parser.set_defaults(headless=True)
    
    # Checkpointing
    parser.add_argument('--load-checkpoint', type=str, default='',
                        help='Path to checkpoint to resume from (without .zip)')
    
    args = parser.parse_args()
    
    main(args)
