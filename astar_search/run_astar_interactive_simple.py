"""
Simple Interactive A* Agent for Pokemon Red
==========================================

Run the A* agent continuously like epsilon greedy - no episodes, just one session.
"""

import time
import numpy as np
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from astar_search.v2_astar_agent import V2AStarAgent

if __name__ == "__main__":
    print(" Interactive A* Search Agent for Pokemon Red")
    print(" Intelligent pathfinding with goal-directed exploration")
    print("  Press Ctrl+C to stop at any time")
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
        'stuck_threshold': 50,
    }
    
    print("  A* Agent Configuration:")
    print(f"   Exploration Bonus: {agent_config['exploration_bonus']}")
    print(f"   Stuck Threshold: {agent_config['stuck_threshold']}")
    print()
    
    try:
        # Initialize A* agent
        print("Initializing A* Search Agent...")
        agent = V2AStarAgent(env_config, agent_config, enable_logging=True)
        print("A* Agent ready!")
        print()
        
        print("Resetting environment...")
        observation, info = agent.env.reset()
        print("Environment reset complete")
        print()
        
        print("Interactive Mode Active:")
        print("   Game Boy window is open")
        print("   A* pathfinding in progress")
        print("   Goal-directed exploration enabled")
        print()
        
        # Continuous execution like epsilon greedy
        step = 0
        total_reward = 0
        start_time = time.time()
        
        print("Starting A* continuous session...")
        print("Press Ctrl+C in terminal to stop")
        print("=" * 50)
        
        while True:
            try:
                # Extract game state and enhanced observation like the wrapper does
                game_state = agent.extract_game_state(observation)
                enhanced_obs = agent.enhance_observation_with_heuristics(observation)
                
                # Get action from A* agent directly
                action, decision_info = agent.agent.select_action(enhanced_obs, game_state)
                
                # Execute action
                observation, reward, terminated, truncated, info = agent.env.step(action)
                agent.env.render()
                
                step += 1
                total_reward += reward
                
                # Print progress every 1000 steps
                if step % 1000 == 0:
                    elapsed = time.time() - start_time
                    steps_per_sec = step / elapsed if elapsed > 0 else 0
                    unique_positions = len(getattr(agent.agent, 'visited_positions', set()))
                    current_objective = getattr(agent.agent, 'current_objective', 'exploring')
                    print(f"Step {step:,} | Reward: {total_reward:.2f} | "
                          f"Positions: {unique_positions} | Objective: {current_objective} | "
                          f"Speed: {steps_per_sec:.1f} steps/sec")
                
                if terminated or truncated:
                    print("Episode terminated, resetting...")
                    observation, info = agent.env.reset()
                    total_reward = 0
                
            except KeyboardInterrupt:
                print("\nStopping A* session...")
                break
            except Exception as e:
                print(f"\nError during execution: {e}")
                break
        
        # Final statistics
        elapsed = time.time() - start_time
        print("\n" + "=" * 50)
        print(" A* Session Results:")
        print(f" Total Steps: {step:,}")
        print(f" Total Reward: {total_reward:.2f}")
        print(f"  Session Duration: {elapsed:.1f} seconds")
        if elapsed > 0:
            print(f" Average Speed: {step/elapsed:.1f} steps/second")
        print(" A* session completed!")
        
    except Exception as e:
        print(f" Error initializing A* agent: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        try:
            if 'agent' in locals():
                agent.env.close()
                print(" Environment closed")
        except:
            pass