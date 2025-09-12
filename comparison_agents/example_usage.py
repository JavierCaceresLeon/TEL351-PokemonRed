"""
Example Usage Script for Pokemon Red Agent Comparison
====================================================

This script demonstrates how to use the Pokemon Red agent comparison system
with different configurations and scenarios.
"""

import sys
import time
from pathlib import Path

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from config import Config, get_experiment_config, setup_experiment_environment
from run_comparison import (
    run_standalone_epsilon_greedy,
    run_comprehensive_comparison,
    run_detailed_metrics_analysis
)


def example_1_basic_comparison():
    """
    Example 1: Basic comparison between PPO and Epsilon Greedy
    """
    print("="*60)
    print("EXAMPLE 1: Basic Agent Comparison")
    print("="*60)
    
    # Setup experiment
    experiment_name = f"basic_comparison_{int(time.time())}"
    experiment_config = get_experiment_config(
        experiment_name=experiment_name,
        num_episodes=3,  # Small number for quick demo
        max_steps=20000  # Shorter episodes
    )
    
    print(f"Experiment: {experiment_name}")
    print(f"Episodes: {experiment_config['comparison']['num_episodes']}")
    print(f"Max Steps: {experiment_config['env']['max_steps']}")
    
    try:
        # Run comparison
        result = run_comprehensive_comparison(
            env_config=experiment_config['env'],
            comparison_config=experiment_config['comparison'],
            epsilon_config=experiment_config['epsilon'],
            ppo_model_path=None  # Use random PPO for demo
        )
        
        if result['success']:
            print("✓ Basic comparison completed successfully!")
            print(f"Results saved to: {result['save_dir']}")
            
            # Print summary
            for agent_name, metrics in result['results'].items():
                print(f"\n{agent_name} Summary:")
                print(f"  Episodes: {len(metrics.episode_rewards)}")
                print(f"  Mean Reward: {sum(metrics.episode_rewards)/len(metrics.episode_rewards):.2f}")
                print(f"  Total Steps: {metrics.total_steps}")
        else:
            print(f"✗ Comparison failed: {result['error']}")
            
    except Exception as e:
        print(f"Error in basic comparison: {e}")


def example_2_epsilon_only():
    """
    Example 2: Run only Epsilon Greedy agent with different configurations
    """
    print("\n" + "="*60)
    print("EXAMPLE 2: Epsilon Greedy Agent Only")
    print("="*60)
    
    # Test different epsilon configurations
    epsilon_configs = [
        {
            'name': 'aggressive_exploration',
            'config': {'epsilon_start': 0.8, 'epsilon_decay': 0.99, 'epsilon_min': 0.1}
        },
        {
            'name': 'conservative_exploration', 
            'config': {'epsilon_start': 0.3, 'epsilon_decay': 0.995, 'epsilon_min': 0.01}
        },
        {
            'name': 'balanced_exploration',
            'config': {'epsilon_start': 0.5, 'epsilon_decay': 0.995, 'epsilon_min': 0.05}
        }
    ]
    
    results = {}
    
    for test_config in epsilon_configs:
        print(f"\nTesting: {test_config['name']}")
        print(f"Config: {test_config['config']}")
        
        # Setup environment
        env_config = Config.get_env_config(
            max_steps=15000,
            headless=True
        )
        
        try:
            result = run_standalone_epsilon_greedy(
                env_config=env_config,
                agent_config=test_config['config'],
                num_episodes=2  # Quick test
            )
            
            if result['success']:
                print(f"✓ {test_config['name']} completed")
                print(f"  Mean Reward: {result['summary']['mean_episode_reward']:.2f}")
                print(f"  Episodes: {result['summary']['total_episodes']}")
                results[test_config['name']] = result
            else:
                print(f"✗ {test_config['name']} failed: {result['error']}")
                
        except Exception as e:
            print(f"Error in {test_config['name']}: {e}")
    
    # Compare results
    if results:
        print(f"\nComparison of Epsilon Configurations:")
        print("-" * 50)
        for name, result in results.items():
            summary = result['summary']
            print(f"{name:25s}: {summary['mean_episode_reward']:8.2f} reward")


def example_3_custom_scenarios():
    """
    Example 3: Test agent behavior in different game scenarios
    """
    print("\n" + "="*60)
    print("EXAMPLE 3: Custom Scenario Testing")
    print("="*60)
    
    # Different scenario configurations
    scenarios = [
        {
            'name': 'exploration_focused',
            'description': 'Optimized for map exploration',
            'config': {
                'epsilon_start': 0.7,
                'epsilon_decay': 0.992,
                'scenario_detection_enabled': True
            }
        },
        {
            'name': 'progression_focused', 
            'description': 'Optimized for game progression',
            'config': {
                'epsilon_start': 0.4,
                'epsilon_decay': 0.998,
                'scenario_detection_enabled': True
            }
        }
    ]
    
    print("Testing different scenario configurations...")
    
    for scenario in scenarios:
        print(f"\nScenario: {scenario['name']}")
        print(f"Description: {scenario['description']}")
        
        # Setup environment with scenario-specific settings
        env_config = Config.get_env_config(
            max_steps=25000,
            explore_weight=0.5 if 'exploration' in scenario['name'] else 0.2
        )
        
        try:
            result = run_standalone_epsilon_greedy(
                env_config=env_config,
                agent_config=scenario['config'],
                num_episodes=2
            )
            
            if result['success']:
                print(f"✓ Scenario test completed")
                summary = result['summary']
                print(f"  Average Reward: {summary['mean_episode_reward']:.2f}")
                print(f"  Average Length: {summary['mean_episode_length']:.1f}")
                
                # Extract some scenario-specific metrics from the last episode
                if result['results']:
                    last_episode = result['results'][-1]
                    print(f"  Final Epsilon: {last_episode['final_epsilon']:.3f}")
                    print(f"  Exploration Efficiency: {last_episode['exploration_efficiency']:.3f}")
            else:
                print(f"✗ Scenario test failed: {result['error']}")
                
        except Exception as e:
            print(f"Error in scenario {scenario['name']}: {e}")


def example_4_performance_analysis():
    """
    Example 4: Detailed performance analysis
    """
    print("\n" + "="*60)
    print("EXAMPLE 4: Performance Analysis Demo")
    print("="*60)
    
    # This example shows how to use the metrics analyzer independently
    try:
        from metrics_analyzer import MetricsAnalyzer
        
        # Create sample data for demonstration
        sample_data = {
            'Agent_A': {
                'episode_rewards': [10, 15, 20, 22, 25, 27, 30, 32, 35, 38],
                'episode_lengths': [1000, 950, 900, 880, 850, 820, 800, 780, 760, 740],
                'episode_times': [50, 48, 45, 44, 42, 41, 40, 39, 38, 37]
            },
            'Agent_B': {
                'episode_rewards': [8, 12, 18, 24, 28, 30, 31, 34, 36, 39],
                'episode_lengths': [1100, 1000, 950, 900, 850, 820, 810, 790, 770, 750],
                'episode_times': [55, 50, 48, 45, 43, 41, 40, 39, 38, 37]
            }
        }
        
        # Create analyzer
        analyzer = MetricsAnalyzer("example_analysis")
        
        # Calculate metrics for each agent
        agent_metrics = {}
        for agent_name, data in sample_data.items():
            metrics = analyzer.calculate_comprehensive_metrics(
                agent_name=agent_name,
                episode_rewards=data['episode_rewards'],
                episode_lengths=data['episode_lengths'],
                episode_times=data['episode_times'],
                game_states=None
            )
            agent_metrics[agent_name] = metrics
            
            print(f"\n{agent_name} Metrics:")
            print(f"  Mean Reward: {metrics.mean_reward:.2f}")
            print(f"  Stability: {metrics.performance_stability:.3f}")
            print(f"  Learning Rate: {metrics.learning_rate:.4f}")
            print(f"  Exploration Efficiency: {metrics.exploration_efficiency:.3f}")
        
        # Compare agents
        print(f"\nRunning comparative analysis...")
        comparison_results = analyzer.compare_agents(
            agent_metrics=agent_metrics,
            create_plots=False  # Skip plots for this demo
        )
        
        print(f"✓ Performance analysis completed")
        print(f"Recommendations:")
        for rec in comparison_results['recommendations']:
            print(f"  - {rec}")
            
    except Exception as e:
        print(f"Error in performance analysis: {e}")


def example_5_configuration_testing():
    """
    Example 5: Test different configuration setups
    """
    print("\n" + "="*60)
    print("EXAMPLE 5: Configuration Testing")
    print("="*60)
    
    # Test configuration validation and setup
    try:
        from config import Config
        
        # Test basic configuration
        env_config = Config.get_env_config(max_steps=30000)
        print(f"✓ Environment config validation: {Config.validate_config(env_config, 'env')}")
        
        epsilon_config = Config.get_epsilon_config(epsilon_start=0.6)
        print(f"✓ Epsilon config validation: {Config.validate_config(epsilon_config, 'epsilon')}")
        
        comparison_config = Config.get_comparison_config(num_episodes=3)
        print(f"✓ Comparison config validation: {Config.validate_config(comparison_config, 'comparison')}")
        
        # Test output directory setup
        output_dirs = Config.setup_output_directories()
        print(f"✓ Output directories created: {len(output_dirs)} directories")
        
        # Test scenario weights
        for scenario in ['exploration', 'battle', 'navigation']:
            weights = Config.get_scenario_weights(scenario)
            print(f"✓ {scenario} weights: {len(weights)} parameters")
        
        # Test experiment setup
        experiment_dirs = setup_experiment_environment("test_experiment")
        print(f"✓ Experiment environment: {len(experiment_dirs)} subdirectories")
        
        print(f"Configuration testing completed successfully!")
        
    except Exception as e:
        print(f"Error in configuration testing: {e}")


def main():
    """
    Main function to run all examples
    """
    print("Pokemon Red Agent Comparison - Example Usage")
    print("=" * 80)
    print("This script demonstrates various features of the comparison system.")
    print("Note: Examples use reduced episodes/steps for demonstration purposes.")
    print()
    
    # Ask user which examples to run
    examples = [
        ("Basic Comparison", example_1_basic_comparison),
        ("Epsilon Greedy Only", example_2_epsilon_only),
        ("Custom Scenarios", example_3_custom_scenarios),
        ("Performance Analysis", example_4_performance_analysis),
        ("Configuration Testing", example_5_configuration_testing)
    ]
    
    print("Available examples:")
    for i, (name, _) in enumerate(examples, 1):
        print(f"  {i}. {name}")
    print("  0. Run all examples")
    print()
    
    try:
        choice = input("Enter your choice (0-5): ").strip()
        
        if choice == "0":
            # Run all examples
            for name, func in examples:
                print(f"\n{'='*20} Running: {name} {'='*20}")
                func()
        elif choice in [str(i) for i in range(1, len(examples) + 1)]:
            # Run specific example
            idx = int(choice) - 1
            name, func = examples[idx]
            print(f"\n{'='*20} Running: {name} {'='*20}")
            func()
        else:
            print("Invalid choice. Running configuration testing example...")
            example_5_configuration_testing()
            
    except KeyboardInterrupt:
        print("\nExecution interrupted by user.")
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*80)
    print("Example execution completed!")
    print("Check the output directories for generated files and results.")


if __name__ == "__main__":
    main()