#!/usr/bin/env python3
"""
Continue training from the last checkpoint
"""
import argparse
from pathlib import Path
from stable_baselines3 import PPO
from combat_focused_env import CombatFocusedEnv, make_combat_env_config
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.callbacks import CheckpointCallback
import torch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint .zip file")
    parser.add_argument("--timesteps", type=int, default=100000, help="Additional timesteps to train")
    parser.add_argument("--session-name", type=str, default="combat_agent_final", help="Session name for logs")
    args = parser.parse_args()
    
    # Check GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    print("=" * 60)
    print("Continuing Combat Agent Training")
    print("=" * 60)
    print(f"Loading from: {args.checkpoint}")
    print(f"Additional timesteps: {args.timesteps:,}")
    print(f"Session: sessions\\{args.session_name}")
    print(f"Device: {device}")
    print("=" * 60)
    print()
    
    # Create environment
    session_dir = Path(f"sessions/{args.session_name}")
    config = make_combat_env_config(
        state_file="has_pokedex_nballs.state",
        session_path=session_dir,
        headless=True
    )
    env = CombatFocusedEnv(config)
    env = DummyVecEnv([lambda: env])
    env = VecTransposeImage(env)
    
    # Load model from checkpoint
    print(f"Loading model from {args.checkpoint}...")
    model = PPO.load(args.checkpoint, env=env, device=device)
    print("Model loaded successfully!")
    
    # Setup callbacks
    session_dir = Path(f"sessions/{args.session_name}")
    checkpoint_dir = session_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path=str(checkpoint_dir),
        name_prefix="combat_agent",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )
    
    # Continue training
    print(f"\nContinuing training for {args.timesteps:,} additional timesteps...")
    print()
    
    model.learn(
        total_timesteps=args.timesteps,
        callback=checkpoint_callback,
        progress_bar=True
    )
    
    # Save final model
    final_path = f"{args.session_name}_continued.zip"
    model.save(final_path)
    print(f"\n{'=' * 60}")
    print(f"Training complete! Model saved to: {final_path}")
    print(f"Original checkpoint preserved at: {args.checkpoint}")
    print(f"{'=' * 60}")

if __name__ == "__main__":
    main()
