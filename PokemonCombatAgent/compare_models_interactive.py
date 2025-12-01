"""
Compare Combat-Focused Agent vs Baseline Agent interactively
Shows models sequentially (one at a time) to avoid SDL2 conflicts
Saves results to JSON file for analysis
"""
import argparse
from pathlib import Path
import time
import sys
import json
import numpy as np
from datetime import datetime
from stable_baselines3 import PPO

# Import environment (works with both repos)
try:
    from red_gym_env_v2 import RedGymEnv
    from battle_only_actions import BattleOnlyActions
except ImportError:
    sys.path.append('../PokemonRedExperiments/v2')
    from red_gym_env_v2 import RedGymEnv
    from battle_only_actions import BattleOnlyActions

class BattleComparison:
    def __init__(self, combat_model_path, baseline_model_path, battle_state):
        self.combat_model_path = combat_model_path
        self.baseline_model_path = baseline_model_path
        self.battle_state = battle_state
        
        # Don't create environments yet - will create them one at a time
        print("Models ready to load sequentially...")
    
    def _create_env(self, window_title="Pokemon Red", use_reduced_actions=False):
        """Create a RedGymEnv instance
        
        Args:
            window_title: Title for the PyBoy window
            use_reduced_actions: If True, wrap with BattleOnlyActions (3 actions instead of 7)
        """
        config = {
            'headless': False,
            'save_final_state': False,
            'early_stop': False,
            'action_freq': 24,
            'init_state': self.battle_state,
            'max_steps': 10000,  # Increased for gym battles
            'print_rewards': True,
            'save_video': False,
            'fast_video': False,
            'session_path': Path('comparison_session'),
            'gb_path': 'PokemonRed.gb',
            'debug': False,
            'sim_frame_dist': 2_000_000.0,
            'extra_buttons': False
        }
        
        env = RedGymEnv(config)
        
        # Apply BattleOnlyActions wrapper if requested
        if use_reduced_actions:
            env = BattleOnlyActions(env)
        
        # Set window title if possible
        try:
            if hasattr(env.pyboy, 'set_title'):
                env.pyboy.set_title(window_title)
        except:
            pass
        
        return env
    
    def run_single_battle(self, model_path, agent_name, max_steps=5000):
        """Run a single battle episode and collect metrics"""
        print(f"\n{'='*60}")
        print(f"Running: {agent_name}")
        print(f"{'='*60}\n")
        
        # Detect if model needs reduced actions by trying to load it first
        use_reduced_actions = False
        try:
            # Try loading with a temporary env to check action space
            temp_env = self._create_env(window_title=agent_name, use_reduced_actions=False)
            test_model = PPO.load(model_path, env=temp_env, custom_objects={'lr_schedule': 0, 'clip_range': 0})
            temp_env.close()
            del test_model
            print("✓ Model compatible with standard actions (7)")
        except ValueError as e:
            if "Action spaces do not match" in str(e):
                print("✓ Model requires reduced actions (3) - using BattleOnlyActions wrapper")
                use_reduced_actions = True
            else:
                raise
        
        # Create environment with correct action space
        env = self._create_env(window_title=agent_name, use_reduced_actions=use_reduced_actions)
        
        # Load model
        print(f"Loading model from: {model_path}")
        model = PPO.load(model_path, env=env, custom_objects={'lr_schedule': 0, 'clip_range': 0})
        print("Model loaded!\n")
        
        obs, info = env.reset()
        
        # WARM-UP: Press A until we reach the actual battle menu
        # (skip intro messages like "Wild PIDGEY appeared!")
        print("🎮 Warm-up: Pressing A to skip intro messages...")
        battle_type = env.read_m(0xD057)
        warmup_steps = 0
        max_warmup = 50
        
        while warmup_steps < max_warmup and battle_type > 0:
            # Press A (action 0 in reduced space, or just use raw env)
            if hasattr(env, 'action_space') and env.action_space.n == 3:
                # Reduced action space: 0=A
                obs, _, _, _, _ = env.step(0)
            else:
                # Full action space: 0=A
                obs, _, _, _, _ = env.step(0)
            
            warmup_steps += 1
            battle_type = env.read_m(0xD057)
            env.render()
            
            # Check if we're in the battle menu (look for specific memory state)
            # In Pokemon Red, when in battle menu, certain memory addresses change
            # For simplicity, we'll just do a few A presses
            if warmup_steps >= 5:
                break
        
        print(f"✓ Warm-up complete ({warmup_steps} steps)\n")
        
        metrics = {
            'total_reward': 0,
            'steps': 0,
            'hp_dealt': 0,
            'hp_taken': 0,
            'battles_won': 0,
            'final_hp': 0,
            'initial_enemy_hp': 0,
            'time_in_battle': 0,
            'actions': []
        }
        
        in_battle = False
        initial_player_hp = None
        initial_enemy_hp = None
        
        for step in range(max_steps):
            # Get action from model
            action, _states = model.predict(obs, deterministic=False)
            metrics['actions'].append(int(action))
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            metrics['total_reward'] += reward
            metrics['steps'] += 1
            
            # Track battle metrics
            battle_type = env.read_m(0xD057)
            
            if battle_type > 0 and not in_battle:
                # Battle started
                in_battle = True
                initial_player_hp = env.read_hp(0xD16C)
                initial_enemy_hp = env.read_hp(0xCFE6)
                metrics['initial_enemy_hp'] = initial_enemy_hp
                print(f"⚔️  Battle started! Player HP: {initial_player_hp}, Enemy HP: {initial_enemy_hp}")
            
            if in_battle:
                metrics['time_in_battle'] += 1
                current_enemy_hp = env.read_hp(0xCFE6)
                
                # Check if battle ended
                if battle_type == 0:
                    current_player_hp = env.read_hp(0xD16C)
                    metrics['final_hp'] = current_player_hp
                    
                    if current_enemy_hp == 0:
                        metrics['battles_won'] += 1
                        print(f"✅ Battle won! Player HP remaining: {current_player_hp}")
                    else:
                        print(f"❌ Battle lost or fled. Player HP: {current_player_hp}")
                    
                    if initial_player_hp and initial_enemy_hp:
                        metrics['hp_dealt'] = initial_enemy_hp - current_enemy_hp
                        metrics['hp_taken'] = initial_player_hp - current_player_hp
                    
                    break
            
            # Render
            env.render()
            
            if terminated or truncated:
                print(f"Episode ended: terminated={terminated}, truncated={truncated}")
                break
        
        # Clean up
        env.close()
        del model
        del env
        
        print(f"\n✅ {agent_name} episode complete!")
        
        return metrics
    
    def compare(self, num_episodes=1, max_steps=5000):
        """Run comparison across multiple episodes"""
        print("\n" + "="*60)
        print("BATTLE COMPARISON: Combat Agent vs Baseline Agent")
        print("="*60)
        print(f"Battle State: {self.battle_state}")
        print(f"Episodes: {num_episodes}")
        print(f"Max steps per episode: {max_steps}")
        print("="*60 + "\n")
        
        combat_results = []
        baseline_results = []
        
        for episode in range(num_episodes):
            print(f"\n{'#'*60}")
            print(f"EPISODE {episode + 1}/{num_episodes}")
            print(f"{'#'*60}")
            
            # Run combat agent
            combat_metrics = self.run_single_battle(
                self.combat_model_path, 
                f"Combat-Focused Agent (Episode {episode+1})",
                max_steps=max_steps
            )
            combat_results.append(combat_metrics)
            
            time.sleep(1)  # Brief pause between runs
            
            # Run baseline agent
            baseline_metrics = self.run_single_battle(
                self.baseline_model_path,
                f"Baseline Agent (Episode {episode+1})",
                max_steps=max_steps
            )
            baseline_results.append(baseline_metrics)
        
        # Print comparison summary
        self._print_summary(combat_results, baseline_results)
        
        # Save results to JSON
        results_file = self._save_results(combat_results, baseline_results)
        
        return combat_results, baseline_results, results_file
    
    def _print_summary(self, combat_results, baseline_results):
        """Print detailed comparison summary"""
        print("\n" + "="*60)
        print("COMPARISON SUMMARY")
        print("="*60)
        
        def avg(results, key):
            values = [r[key] for r in results]
            return np.mean(values) if values else 0
        
        metrics = [
            ('Total Reward', 'total_reward'),
            ('Steps Taken', 'steps'),
            ('HP Dealt', 'hp_dealt'),
            ('HP Taken', 'hp_taken'),
            ('Battles Won', 'battles_won'),
            ('Final HP', 'final_hp'),
            ('Time in Battle', 'time_in_battle')
        ]
        
        print(f"\n{'Metric':<20} {'Combat Agent':>15} {'Baseline':>15} {'Difference':>15}")
        print("-" * 70)
        
        for name, key in metrics:
            combat_avg = avg(combat_results, key)
            baseline_avg = avg(baseline_results, key)
            diff = combat_avg - baseline_avg
            
            print(f"{name:<20} {combat_avg:>15.2f} {baseline_avg:>15.2f} {diff:>15.2f}")
        
        # Win rate
        combat_wins = sum(r['battles_won'] for r in combat_results)
        baseline_wins = sum(r['battles_won'] for r in baseline_results)
        total = len(combat_results)
        
        print("\n" + "="*60)
        print(f"Combat Agent Win Rate: {combat_wins}/{total} ({100*combat_wins/total:.1f}%)")
        print(f"Baseline Win Rate: {baseline_wins}/{total} ({100*baseline_wins/total:.1f}%)")
        print("="*60 + "\n")
    
    def _save_results(self, combat_results, baseline_results):
        """Save results to JSON file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = Path("comparison_results")
        results_dir.mkdir(exist_ok=True)
        
        results_file = results_dir / f"comparison_{timestamp}.json"
        
        # Convert numpy types to native Python types for JSON serialization
        def convert_to_native(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, list):
                return [convert_to_native(item) for item in obj]
            elif isinstance(obj, dict):
                return {key: convert_to_native(value) for key, value in obj.items()}
            return obj
        
        results = {
            "timestamp": timestamp,
            "battle_state": str(self.battle_state),
            "combat_model": str(self.combat_model_path),
            "baseline_model": str(self.baseline_model_path),
            "num_episodes": len(combat_results),
            "combat_results": convert_to_native(combat_results),
            "baseline_results": convert_to_native(baseline_results),
            "summary": {
                "combat": {
                    "avg_reward": float(np.mean([r['total_reward'] for r in combat_results])),
                    "avg_steps": float(np.mean([r['steps'] for r in combat_results])),
                    "avg_hp_dealt": float(np.mean([r['hp_dealt'] for r in combat_results])),
                    "avg_hp_taken": float(np.mean([r['hp_taken'] for r in combat_results])),
                    "win_rate": sum(r['battles_won'] for r in combat_results) / len(combat_results)
                },
                "baseline": {
                    "avg_reward": float(np.mean([r['total_reward'] for r in baseline_results])),
                    "avg_steps": float(np.mean([r['steps'] for r in baseline_results])),
                    "avg_hp_dealt": float(np.mean([r['hp_dealt'] for r in baseline_results])),
                    "avg_hp_taken": float(np.mean([r['hp_taken'] for r in baseline_results])),
                    "win_rate": sum(r['battles_won'] for r in baseline_results) / len(baseline_results)
                }
            }
        }
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"📊 Results saved to: {results_file}\n")
        return results_file
    
    def cleanup(self):
        """Cleanup method (environments are now created/destroyed per episode)"""
        pass


def main():
    parser = argparse.ArgumentParser(description="Compare Combat Agent vs Baseline")
    parser.add_argument(
        '--combat-model',
        type=str,
        default='sessions/combat_agent_final/combat_agent_final.zip',
        help='Path to combat-focused model'
    )
    parser.add_argument(
        '--baseline-model',
        type=str,
        required=True,
        help='Path to baseline model from PokemonRedExperiments'
    )
    parser.add_argument(
        '--battle-state',
        type=str,
        default='battle_states/pewter_battle.state',
        help='Battle state file to use'
    )
    parser.add_argument(
        '--episodes',
        type=int,
        default=1,
        help='Number of episodes to run for each agent'
    )
    parser.add_argument(
        '--max-steps',
        type=int,
        default=5000,
        help='Maximum steps per episode before stopping'
    )
    
    args = parser.parse_args()
    
    # Verify files exist
    if not Path(args.combat_model).exists():
        print(f"❌ Combat model not found: {args.combat_model}")
        return
    
    if not Path(args.baseline_model).exists():
        print(f"❌ Baseline model not found: {args.baseline_model}")
        return
    
    if not Path(args.battle_state).exists():
        print(f"❌ Battle state not found: {args.battle_state}")
        return
    
    # Run comparison
    comparison = BattleComparison(
        args.combat_model,
        args.baseline_model,
        args.battle_state
    )
    
    try:
        combat_results, baseline_results, results_file = comparison.compare(
            args.episodes,
            max_steps=args.max_steps
        )
        print(f"\n{'='*60}")
        print(f"✅ Comparison complete!")
        print(f"📊 Results saved to: {results_file}")
        print(f"{'='*60}\n")
    finally:
        comparison.cleanup()


if __name__ == '__main__':
    main()
