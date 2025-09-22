"""
Interactive A* Search Agent for Pokemon Red
==========================================

Run the A* search agent interactively with real-time visualization and control.
This script provides intelligent pathfinding and goal-directed exploration.
"""

import sys
import os
import numpy as np
import time
from pathlib import Path

# Add the comparison_agents directory to the path
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

# Import necessary modules
from v2_astar_agent import V2AStarAgent


def main():
    """Main function to run the A* agent interactively"""
    
    print("Interactive A* Search Agent for Pokemon Red")
    print("Intelligent pathfinding with goal-directed exploration")
    print("Press Ctrl+C to stop at any time")
    print()
    
    # Session and environment configuration
    sess_path = Path(f'astar_session_{str(time.time_ns())[:8]}')
    ep_length = 2**23
    
    # Environment configuration for interactive v2 mode
    env_config = {
        'headless': False,
        'save_final_state': True,
        'early_stop': False,
        'action_freq': 24,
        'init_state': '../has_pokedex.state',
        'max_steps': ep_length,
        'print_rewards': True,
        'save_video': False,
        'fast_video': True,
        'session_path': sess_path,
        'gb_path': '../PokemonRed.gb',
        'debug': False,
        'sim_frame_dist': 2_000_000.0
    }
    
    # A* specific configuration
    agent_config = {
        'exploration_bonus': 0.2,
        'goal_reward_bonus': 2.0,
        'stuck_threshold': 50,
        'path_planning_interval': 10,
        'max_path_length': 100,
        'heuristic_weight': 1.5
    }
    
    print("A* Agent Configuration:")
    print(f"   Exploration Bonus: {agent_config['exploration_bonus']}")
    print(f"   Goal Reward Bonus: {agent_config['goal_reward_bonus']}")
    print(f"   Stuck Threshold: {agent_config['stuck_threshold']}")
    print(f"   Heuristic Weight: {agent_config['heuristic_weight']}")
    print(f"   Max Path Length: {agent_config['max_path_length']}")
    print()
    
    try:
        # Initialize A* agent
        print("Initializing A* Search Agent...")
        agent = V2AStarAgent(env_config, agent_config, enable_logging=True)
        print("A* Agent ready!")
        print()
        
        # Configuration options
        print("Interactive Mode Configuration:")
        print("   Game Boy window will open")
        print("   Real-time metrics displayed")
        print("   Intelligent pathfinding active")
        print("   Goal-directed exploration enabled")
        print()
        
        # Get user preferences
        max_episodes = None
        max_steps_per_episode = None
        
        try:
            episodes_input = input("Max episodes (press Enter for unlimited): ").strip()
            if episodes_input:
                max_episodes = int(episodes_input)
        except ValueError:
            print("Invalid input, using unlimited episodes")
        
        try:
            steps_input = input("🚶 Max steps per episode (press Enter for default 50000): ").strip()
            if steps_input:
                max_steps_per_episode = int(steps_input)
        except ValueError:
            print("Invalid input, using default 50000 steps")
        
        print()
        print("Starting A* Interactive Session...")
        print("Watch the agent use intelligent pathfinding!")
        print("Press Ctrl+C in terminal to stop")
        print("=" * 50)
        
        # Run interactive session
        agent.run_interactive(max_episodes, max_steps_per_episode)
        
    except KeyboardInterrupt:
        print("\nInteractive session stopped by user")
    except Exception as e:
        print(f"\nError: {e}")
        print("Check your environment setup and dependencies")
    
    print("\nA* Interactive session completed!")


if __name__ == "__main__":
    main()