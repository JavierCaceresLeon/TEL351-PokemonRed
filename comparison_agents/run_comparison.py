"""
Main Execution Script for Pokemon Red Agent Comparison
======================================================

This script runs the comprehensive comparison between PPO and Epsilon Greedy agents
in the Pokemon Red v2 environment.
"""

import sys
import argparse
from pathlib import Path
import json
import numpy as np

# Add paths
sys.path.append('../v2')
sys.path.append('.')

from agent_comparison import AgentComparator
from metrics_analyzer import MetricsAnalyzer
from epsilon_greedy_agent import EpsilonGreedyAgent
from v2_agent import V2EpsilonGreedyAgent


def setup_environment_config(headless: bool = True, 
                           max_steps: int = 40960,
                           session_name: str = "comparison_session") -> dict:
    """
    Setup environment configuration for the comparison
    """
    return {
        'headless': headless,
        'save_final_state': False,
        'early_stop': False,
        'action_freq': 24,
        'init_state': '../init.state',
        'max_steps': max_steps,
        'print_rewards': False,
        'save_video': False,
        'fast_video': True,
        'session_path': Path(session_name),
        'gb_path': '../PokemonRed.gb',
        'debug': False,
        'reward_scale': 0.5,
        'explore_weight': 0.25
    }


def setup_comparison_config(num_episodes: int = 5,
                          parallel: bool = False,
                          create_viz: bool = True) -> dict:
    """
    Setup comparison configuration
    """
    return {
        'num_episodes': num_episodes,
        'max_steps_per_episode': 40960,
        'parallel_execution': parallel,
        'save_detailed_logs': True,
        'create_visualizations': create_viz,
        'metrics_to_compare': [
            'episode_rewards', 'episode_lengths', 'exploration_efficiency',
            'convergence_rate', 'stability', 'scenario_adaptation'
        ]
    }


def setup_epsilon_config(epsilon_start: float = 0.5,
                        epsilon_min: float = 0.05,
                        epsilon_decay: float = 0.995,
                        scenario_detection: bool = True) -> dict:
    """
    Setup Epsilon Greedy agent configuration
    """
    return {
        'epsilon_start': epsilon_start,
        'epsilon_min': epsilon_min,
        'epsilon_decay': epsilon_decay,
        'scenario_detection_enabled': scenario_detection
    }


def run_standalone_epsilon_greedy(env_config: dict, 
                                 agent_config: dict,
                                 num_episodes: int = 3) -> dict:
    """
    Run standalone Epsilon Greedy agent for testing
    """
    print("Running standalone Epsilon Greedy agent...")
    
    try:
        # Create agent wrapper
        agent_wrapper = V2EpsilonGreedyAgent(
            env_config=env_config,
            agent_config=agent_config,
            enable_logging=True
        )
        
        # Run episodes
        results = agent_wrapper.run_multiple_episodes(
            num_episodes=num_episodes,
            max_steps_per_episode=env_config['max_steps'],
            save_results=True
        )
        
        # Get summary
        summary = agent_wrapper.get_summary_statistics()
        
        agent_wrapper.close()
        
        return {
            'success': True,
            'results': results,
            'summary': summary
        }
        
    except Exception as e:
        print(f"Error running standalone Epsilon Greedy: {e}")
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': str(e)}


def run_comprehensive_comparison(env_config: dict,
                               comparison_config: dict,
                               epsilon_config: dict,
                               ppo_model_path: str = None) -> dict:
    """
    Run comprehensive comparison between agents
    """
    print("Running comprehensive agent comparison...")
    
    try:
        # Create comparator
        comparator = AgentComparator(
            env_config=env_config,
            comparison_config=comparison_config,
            save_dir="comparison_results"
        )
        
        # Run comparison
        results = comparator.run_comparison(
            ppo_model_path=ppo_model_path,
            epsilon_config=epsilon_config
        )
        
        return {
            'success': True,
            'results': results,
            'save_dir': str(comparator.save_dir)
        }
        
    except Exception as e:
        print(f"Error during comprehensive comparison: {e}")
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': str(e)}


def run_detailed_metrics_analysis(comparison_results: dict) -> dict:
    """
    Run detailed metrics analysis on comparison results
    """
    print("Running detailed metrics analysis...")
    
    try:
        # Create metrics analyzer
        analyzer = MetricsAnalyzer("detailed_metrics_analysis")
        
        # Extract metrics from comparison results
        agent_metrics = {}
        
        for agent_name, metrics in comparison_results.items():
            if hasattr(metrics, 'episode_rewards'):
                # Calculate comprehensive metrics
                detailed_metrics = analyzer.calculate_comprehensive_metrics(
                    agent_name=agent_name,
                    episode_rewards=metrics.episode_rewards,
                    episode_lengths=metrics.episode_lengths,
                    episode_times=[1.0] * len(metrics.episode_rewards),  # Placeholder
                    game_states=None
                )
                agent_metrics[agent_name] = detailed_metrics
        
        # Perform comparison if we have metrics
        if agent_metrics:
            comparison_analysis = analyzer.compare_agents(
                agent_metrics=agent_metrics,
                create_plots=True
            )
            
            return {
                'success': True,
                'analysis': comparison_analysis,
                'save_dir': str(analyzer.save_dir)
            }
        else:
            return {'success': False, 'error': 'No valid metrics found'}
        
    except Exception as e:
        print(f"Error during metrics analysis: {e}")
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': str(e)}


def main():
    """
    Main execution function with command line interface
    """
    parser = argparse.ArgumentParser(description='Pokemon Red Agent Comparison')
    
    parser.add_argument('--mode', choices=['standalone', 'comparison', 'analysis', 'full'],
                       default='full', help='Execution mode')
    
    parser.add_argument('--episodes', type=int, default=5,
                       help='Number of episodes to run')
    
    parser.add_argument('--headless', action='store_true', default=True,
                       help='Run in headless mode')
    
    parser.add_argument('--max-steps', type=int, default=40960,
                       help='Maximum steps per episode')
    
    parser.add_argument('--epsilon-start', type=float, default=0.5,
                       help='Starting epsilon value')
    
    parser.add_argument('--epsilon-decay', type=float, default=0.995,
                       help='Epsilon decay rate')
    
    parser.add_argument('--ppo-model', type=str, default=None,
                       help='Path to pre-trained PPO model')
    
    parser.add_argument('--parallel', action='store_true', default=False,
                       help='Run agents in parallel')
    
    parser.add_argument('--no-viz', action='store_true', default=False,
                       help='Skip visualizations')
    
    args = parser.parse_args()
    
    # Setup configurations
    env_config = setup_environment_config(
        headless=args.headless,
        max_steps=args.max_steps
    )
    
    comparison_config = setup_comparison_config(
        num_episodes=args.episodes,
        parallel=args.parallel,
        create_viz=not args.no_viz
    )
    
    epsilon_config = setup_epsilon_config(
        epsilon_start=args.epsilon_start,
        epsilon_decay=args.epsilon_decay
    )
    
    print("Pokemon Red Agent Comparison")
    print("=" * 50)
    print(f"Mode: {args.mode}")
    print(f"Episodes: {args.episodes}")
    print(f"Max Steps: {args.max_steps}")
    print(f"Epsilon Start: {args.epsilon_start}")
    print(f"Epsilon Decay: {args.epsilon_decay}")
    print("")
    
    # Execute based on mode
    if args.mode == 'standalone':
        # Run only Epsilon Greedy agent
        result = run_standalone_epsilon_greedy(
            env_config=env_config,
            agent_config=epsilon_config,
            num_episodes=args.episodes
        )
        
        if result['success']:
            print("Standalone Epsilon Greedy execution completed successfully!")
            print(f"Summary: {result['summary']}")
        else:
            print(f"Standalone execution failed: {result['error']}")
    
    elif args.mode == 'comparison':
        # Run agent comparison
        result = run_comprehensive_comparison(
            env_config=env_config,
            comparison_config=comparison_config,
            epsilon_config=epsilon_config,
            ppo_model_path=args.ppo_model
        )
        
        if result['success']:
            print("Agent comparison completed successfully!")
            print(f"Results saved to: {result['save_dir']}")
        else:
            print(f"Comparison failed: {result['error']}")
    
    elif args.mode == 'analysis':
        # Run detailed analysis (requires existing results)
        print("Analysis mode requires existing comparison results")
        print("Please run comparison mode first")
    
    elif args.mode == 'full':
        # Run complete pipeline
        print("Running full comparison pipeline...")
        
        # Step 1: Comprehensive comparison
        comparison_result = run_comprehensive_comparison(
            env_config=env_config,
            comparison_config=comparison_config,
            epsilon_config=epsilon_config,
            ppo_model_path=args.ppo_model
        )
        
        if comparison_result['success']:
            print("✓ Comprehensive comparison completed")
            
            # Step 2: Detailed metrics analysis
            metrics_result = run_detailed_metrics_analysis(
                comparison_result['results']
            )
            
            if metrics_result['success']:
                print("✓ Detailed metrics analysis completed")
                print(f"Analysis saved to: {metrics_result['save_dir']}")
            else:
                print(f"✗ Metrics analysis failed: {metrics_result['error']}")
        else:
            print(f"✗ Comprehensive comparison failed: {comparison_result['error']}")
    
    print("\nExecution completed!")


if __name__ == "__main__":
    main()
