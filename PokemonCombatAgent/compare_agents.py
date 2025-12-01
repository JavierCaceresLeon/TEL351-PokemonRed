"""
Compare Combat Agent vs Baseline PPO Agent

Evaluates both agents on the same scenarios and generates comparison metrics.
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from combat_gym_env import CombatGymEnv
from stable_baselines3 import PPO
from scipy import stats
import json


class AgentEvaluator:
    """Evaluate agent performance in combat scenarios"""
    
    def __init__(self, env_config):
        self.env_config = env_config
        self.metrics = []
    
    def evaluate_agent(self, model_path, num_episodes=100, agent_name="Agent"):
        """
        Evaluate agent over multiple episodes.
        
        Returns:
            dict: Aggregated metrics
        """
        print(f"\n{'=' * 60}")
        print(f"Evaluating {agent_name}")
        print(f"Model: {model_path}")
        print(f"Episodes: {num_episodes}")
        print(f"{'=' * 60}\n")
        
        # Load model
        model = PPO.load(model_path)
        
        # Run episodes
        episode_metrics = []
        
        for episode in range(num_episodes):
            # Create fresh environment
            env = CombatGymEnv(self.env_config)
            obs, _ = env.reset()
            
            episode_reward = 0
            done = False
            truncated = False
            steps = 0
            
            # Track episode-specific metrics
            start_hp = env.read_hp_fraction()
            battles_won_start = env.battles_won
            battles_lost_start = env.battles_lost
            efficient_kills_start = env.efficient_kills
            
            while not (done or truncated):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = env.step(action)
                episode_reward += reward
                steps += 1
                
                # Safety limit
                if steps > 10000:
                    break
            
            # Collect metrics
            end_hp = env.read_hp_fraction()
            battles_won = env.battles_won - battles_won_start
            battles_lost = env.battles_lost - battles_lost_start
            efficient_kills = env.efficient_kills - efficient_kills_start
            deaths = env.died_count
            badges = env.get_badges()
            
            metrics = {
                'episode': episode + 1,
                'agent': agent_name,
                'total_reward': episode_reward,
                'steps': steps,
                'battles_won': battles_won,
                'battles_lost': battles_lost,
                'win_rate': battles_won / max(battles_won + battles_lost, 1),
                'efficient_kills': efficient_kills,
                'hp_start': start_hp,
                'hp_end': end_hp,
                'hp_conserved': end_hp / max(start_hp, 0.01),
                'deaths': deaths,
                'badges': badges,
            }
            
            episode_metrics.append(metrics)
            
            # Progress
            if (episode + 1) % 10 == 0:
                print(f"  Episode {episode + 1}/{num_episodes} - "
                      f"Reward: {episode_reward:.2f}, "
                      f"Wins: {battles_won}, "
                      f"HP: {end_hp:.2%}")
        
        return episode_metrics
    
    def compare_agents(self, combat_metrics, baseline_metrics):
        """
        Statistical comparison between two agents.
        
        Returns:
            dict: Comparison results with p-values
        """
        print(f"\n{'=' * 60}")
        print(f"Statistical Comparison")
        print(f"{'=' * 60}\n")
        
        combat_df = pd.DataFrame(combat_metrics)
        baseline_df = pd.DataFrame(baseline_metrics)
        
        comparison = {}
        
        # Compare key metrics
        metrics_to_compare = [
            ('total_reward', 'Total Reward'),
            ('win_rate', 'Win Rate'),
            ('hp_conserved', 'HP Conservation'),
            ('deaths', 'Deaths per Episode'),
            ('efficient_kills', 'Efficient Kills'),
        ]
        
        for metric_key, metric_name in metrics_to_compare:
            combat_values = combat_df[metric_key].values
            baseline_values = baseline_df[metric_key].values
            
            # T-test
            t_stat, p_value = stats.ttest_ind(combat_values, baseline_values)
            
            combat_mean = np.mean(combat_values)
            baseline_mean = np.mean(baseline_values)
            combat_std = np.std(combat_values)
            baseline_std = np.std(baseline_values)
            
            # Effect size (Cohen's d)
            pooled_std = np.sqrt((combat_std**2 + baseline_std**2) / 2)
            cohens_d = (combat_mean - baseline_mean) / max(pooled_std, 0.001)
            
            comparison[metric_key] = {
                'metric_name': metric_name,
                'combat_mean': combat_mean,
                'combat_std': combat_std,
                'baseline_mean': baseline_mean,
                'baseline_std': baseline_std,
                'difference': combat_mean - baseline_mean,
                'percent_change': ((combat_mean - baseline_mean) / max(abs(baseline_mean), 0.001)) * 100,
                't_statistic': t_stat,
                'p_value': p_value,
                'cohens_d': cohens_d,
                'significant': p_value < 0.05,
            }
            
            # Print results
            print(f"{metric_name}:")
            print(f"  Combat Agent:   {combat_mean:.4f} ± {combat_std:.4f}")
            print(f"  Baseline Agent: {baseline_mean:.4f} ± {baseline_std:.4f}")
            print(f"  Difference:     {combat_mean - baseline_mean:+.4f} "
                  f"({comparison[metric_key]['percent_change']:+.1f}%)")
            print(f"  p-value:        {p_value:.4f} {'✓ SIGNIFICANT' if p_value < 0.05 else '✗ not significant'}")
            print(f"  Cohen's d:      {cohens_d:.3f}\n")
        
        return comparison
    
    def save_results(self, combat_metrics, baseline_metrics, comparison, output_path):
        """Save all results to files"""
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save raw metrics
        pd.DataFrame(combat_metrics).to_csv(output_path / 'combat_agent_metrics.csv', index=False)
        pd.DataFrame(baseline_metrics).to_csv(output_path / 'baseline_agent_metrics.csv', index=False)
        
        # Save comparison
        comparison_df = pd.DataFrame.from_dict(comparison, orient='index')
        comparison_df.to_csv(output_path / 'comparison_results.csv')
        
        # Save summary JSON
        summary = {
            'combat_agent': {
                'total_episodes': len(combat_metrics),
                'mean_reward': float(np.mean([m['total_reward'] for m in combat_metrics])),
                'mean_win_rate': float(np.mean([m['win_rate'] for m in combat_metrics])),
                'mean_hp_conserved': float(np.mean([m['hp_conserved'] for m in combat_metrics])),
            },
            'baseline_agent': {
                'total_episodes': len(baseline_metrics),
                'mean_reward': float(np.mean([m['total_reward'] for m in baseline_metrics])),
                'mean_win_rate': float(np.mean([m['win_rate'] for m in baseline_metrics])),
                'mean_hp_conserved': float(np.mean([m['hp_conserved'] for m in baseline_metrics])),
            },
            'significant_improvements': [
                key for key, val in comparison.items() 
                if val['significant'] and val['difference'] > 0
            ]
        }
        
        with open(output_path / 'summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n{'=' * 60}")
        print(f"Results saved to {output_path}/")
        print(f"  - combat_agent_metrics.csv")
        print(f"  - baseline_agent_metrics.csv")
        print(f"  - comparison_results.csv")
        print(f"  - summary.json")
        print(f"{'=' * 60}")


def main(args):
    """Main comparison pipeline"""
    
    # Environment configuration (same for both agents)
    env_config = {
        'headless': True,
        'save_final_state': False,
        'early_stop': False,
        'action_freq': args.action_freq,
        'init_state': args.init_state,
        'max_steps': args.max_steps,
        'print_rewards': False,
        'save_video': False,
        'session_path': Path('./temp_eval'),
        'gb_path': args.rom_path,
        'debug': False,
        'combat_focus': True,
        'reward_scale': 1.0,
    }
    
    evaluator = AgentEvaluator(env_config)
    
    # Evaluate combat agent
    if args.combat_agent:
        combat_metrics = evaluator.evaluate_agent(
            args.combat_agent,
            num_episodes=args.episodes,
            agent_name="Combat Agent"
        )
    else:
        print("No combat agent specified, skipping...")
        combat_metrics = []
    
    # Evaluate baseline agent
    if args.baseline_agent:
        baseline_metrics = evaluator.evaluate_agent(
            args.baseline_agent,
            num_episodes=args.episodes,
            agent_name="Baseline PPO"
        )
    else:
        print("No baseline agent specified, skipping...")
        baseline_metrics = []
    
    # Compare if both agents evaluated
    if combat_metrics and baseline_metrics:
        comparison = evaluator.compare_agents(combat_metrics, baseline_metrics)
        evaluator.save_results(combat_metrics, baseline_metrics, comparison, args.output_dir)
    elif combat_metrics:
        # Just save combat agent metrics
        output_path = Path(args.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(combat_metrics).to_csv(output_path / 'combat_agent_metrics.csv', index=False)
        print(f"\nResults saved to {output_path}/combat_agent_metrics.csv")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Compare Combat Agent vs Baseline PPO',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Agent paths
    parser.add_argument('--combat-agent', type=str, default='',
                        help='Path to combat agent checkpoint (without .zip)')
    parser.add_argument('--baseline-agent', type=str, default='',
                        help='Path to baseline PPO checkpoint (without .zip)')
    
    # Evaluation parameters
    parser.add_argument('--episodes', type=int, default=100,
                        help='Number of episodes to evaluate each agent')
    parser.add_argument('--output-dir', type=str, default='./comparison_results',
                        help='Output directory for results')
    
    # Environment parameters
    parser.add_argument('--rom-path', type=str, default='../PokemonRed.gb',
                        help='Path to Pokemon Red ROM')
    parser.add_argument('--init-state', type=str, default='../has_pokedex_nballs.state',
                        help='Initial game state file')
    parser.add_argument('--max-steps', type=int, default=5000,
                        help='Max steps per episode')
    parser.add_argument('--action-freq', type=int, default=24,
                        help='Frames per action')
    
    args = parser.parse_args()
    
    if not args.combat_agent and not args.baseline_agent:
        parser.error("At least one of --combat-agent or --baseline-agent must be specified")
    
    main(args)
