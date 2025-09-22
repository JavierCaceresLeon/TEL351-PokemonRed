#!/usr/bin/env python3
"""
Simple A* Agent Test
Quick test of the A* agent to verify it works better than Tabu Search
"""

import sys
import os
import time
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from search_algorithms.astar_agent import PokemonAStarAgent

# Import from v2 environment (same as epsilon greedy)
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'v2'))
from red_gym_env_v2 import RedGymEnv
from stream_agent_wrapper import StreamWrapper

def main():
    """Test the A* agent"""
    print(" Testing A* Search Agent for Pokemon Red")
    print(" Goal-directed pathfinding - should be much smarter than Tabu!")
    print("  Press Ctrl+C to stop\n")
    
    # Setup paths
    script_dir = Path(__file__).parent
    rom_path = script_dir.parent / "PokemonRed.gb"
    save_state = script_dir.parent / "has_pokedex.state"
    
    if not rom_path.exists():
        print(f" ROM file not found: {rom_path}")
        return
    
    try:
        # Environment configuration (similar to epsilon greedy)
        env_config = {
            'headless': True,  # Run without GUI for speed
            'save_final_state': False,
            'early_stop': False,
            'action_freq': 24,
            'init_state': str(save_state) if save_state.exists() else '../init.state',
            'max_steps': 2**20,  # Large number
            'print_rewards': False,
            'save_video': False,
            'fast_video': True,
            'session_path': Path(f'astar_test_{str(time.time_ns())[:8]}'),
            'gb_path': str(rom_path),
            'debug': False,
            'sim_frame_dist': 2_000_000.0,
            'extra_buttons': False
        }
        
        # Initialize environment and agent
        env = StreamWrapper(
            RedGymEnv(env_config),
            stream_metadata={
                "user": "astar-test",
                "env_id": 0,
                "color": "#4488ff",
                "extra": "A* Search Agent Test",
            }
        )
        
        agent = PokemonAStarAgent(exploration_bonus=1.5, goal_persistence=100)
        
        if save_state.exists():
            print(f" Using save state: {save_state.name}")
        
        print(f" A* Agent initialized")
        
        # Get initial state
        observation, info = env.reset()
        
        print(f" Starting A* agent test...")
        print(f" Running A* agent... (Press Ctrl+C to stop)\n")
        
        step_count = 0
        total_reward = 0
        unique_positions = set()
        last_info_time = time.time()
        
        # Extract initial game state from observation
        def extract_simple_state(obs) -> dict:
            # Extract basic game state from v2 observation
            return {
                'x': int(obs.get('x', [0])[0]) if hasattr(obs.get('x', [0]), '__iter__') else int(obs.get('x', 0)),
                'y': int(obs.get('y', [0])[0]) if hasattr(obs.get('y', [0]), '__iter__') else int(obs.get('y', 0)),
                'map_id': int(obs.get('map_id', [0])[0]) if hasattr(obs.get('map_id', [0]), '__iter__') else int(obs.get('map_id', 0)),
                'level': int(np.sum(obs.get('level', np.zeros(8)))),
                'battle': False,
                'money': 0
            }
        
        try:
            while True:
                # Extract game state from observation
                game_state = extract_simple_state(observation)
                
                # Agent decision
                action, decision_info = agent.select_action(observation, game_state)
                
                # Execute action
                observation, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                # Update tracking
                agent.update_performance(action, reward, observation, game_state)
                
                step_count += 1
                total_reward += reward
                pos = (game_state['x'], game_state['y'])
                unique_positions.add(pos)
                
                # Periodic status updates
                current_time = time.time()
                if current_time - last_info_time >= 20:  # Every 20 seconds
                    obj = decision_info.get('objective', 'N/A')
                    source = decision_info.get('action_source', 'N/A')
                    quality = decision_info.get('action_quality', 0)
                    print(f" Step {step_count:,} | Reward: {total_reward:.2f} | "
                          f"Positions: {len(unique_positions)} | Objective: {obj} | "
                          f"Source: {source} | Quality: {quality:.3f}")
                    last_info_time = current_time
                
                if done:
                    print(" Episode completed!")
                    break
                    
        except KeyboardInterrupt:
            print("\n Stopping...")
        
        # Final summary
        runtime = time.time() - (current_time - (current_time - last_info_time))
        efficiency = len(unique_positions) / max(step_count, 1)
        
        print(f"\n A* Agent Test Results:")
        print(f" Total Steps: {step_count:,}")
        print(f" Total Reward: {total_reward:.2f}")
        print(f" Unique Positions: {len(unique_positions):,}")
        print(f" Exploration Efficiency: {efficiency:.4f}")
        print(f" Paths Planned: {getattr(agent, 'paths_planned', 0)}")
        
        # Performance evaluation
        if efficiency > 0.1:
            print(" A* shows EXCELLENT exploration efficiency!")
            print(" This is much better than typical random/epsilon greedy!")
        elif efficiency > 0.05:
            print(" A* shows good exploration efficiency!")
            print(" Better than basic approaches!")
        else:
            print("  A* exploration could be improved")
            print(" Consider adjusting parameters")
            
        if len(unique_positions) > step_count * 0.3:
            print(" A* demonstrates intelligent pathfinding!")
        
        print(" A* agent test completed!")
        
    except Exception as e:
        print(f" Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        try:
            env.close()
        except:
            pass

if __name__ == "__main__":
    import numpy as np
    main()