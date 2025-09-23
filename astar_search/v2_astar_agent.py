"""
V2 Agent Wrapper for A* Search Algorithm
========================================

This module integrates the A* Search agent with the Pokemon Red v2 environment,
providing a direct comparison alternative to other agents.
"""

import sys
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Import v2 environment
from v2.red_gym_env import RedGymEnv
from v2.stream_wrapper import StreamWrapper

# Import our A* agent
from astar_search.astar_agent import PokemonAStarAgent


class V2AStarAgent:
    """
    Wrapper class that adapts the A* Search agent to work with the v2 environment
    """
    
    def __init__(self, 
                 env_config: Dict,
                 agent_config: Dict = None,
                 enable_logging: bool = True):
        
        self.env_config = env_config
        self.enable_logging = enable_logging
        
        # Default agent configuration for A*
        default_agent_config = {
            'exploration_bonus': 0.2,
            'goal_reward_bonus': 2.0,
            'stuck_threshold': 50,
            'path_planning_interval': 10,
            'max_path_length': 100,
            'heuristic_weight': 1.5
        }
        
        if agent_config:
            default_agent_config.update(agent_config)
        
        self.agent_config = default_agent_config
        
        # Initialize environment
        self.env = StreamWrapper(
            RedGymEnv(env_config),
            stream_metadata={
                "user": "astar-v2",
                "env_id": 0,
                "color": "#2299ff",
                "extra": "A* Search Pathfinding Agent",
            }
        )
        
        # Initialize agent with only valid parameters
        valid_agent_params = {
            'exploration_bonus': self.agent_config.get('exploration_bonus', 0.2),
            'goal_persistence': self.agent_config.get('stuck_threshold', 50)
        }
        self.agent = PokemonAStarAgent(**valid_agent_params)
        
        # Performance tracking
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_metrics = []
        self.total_steps = 0
        
        # State tracking for enhanced heuristics
        self.previous_game_state = None
        self.stagnation_counter = 0
        self.paths_planned = 0
        
    def extract_game_state(self, observation: Dict) -> Dict:
        """
        Extract relevant game state information from observation
        """
        game_state = {
            'health': observation.get('health', np.array([1.0]))[0],
            'level_sum': np.sum(observation.get('level', np.zeros(8))),
            'badges': np.sum(observation.get('badges', np.zeros(8))),
            'events': np.sum(observation.get('events', np.zeros(100))),
            'map_data': observation.get('map', np.zeros((48, 48, 1))),
            'recent_actions': observation.get('recent_actions', np.zeros(3)),
            'screen': observation.get('screens', np.zeros((72, 80, 3))),
            'position': observation.get('position', np.array([0, 0]))
        }
        
        return game_state
    
    def detect_progress(self, current_state: Dict, previous_state: Optional[Dict]) -> bool:
        """
        Detect if the agent is making meaningful progress
        """
        if previous_state is None:
            return True
            
        # Check for progress indicators
        progress_indicators = [
            current_state['badges'] > previous_state['badges'],
            current_state['events'] > previous_state['events'],
            current_state['level_sum'] > previous_state['level_sum'],
            not np.array_equal(current_state['map_data'], previous_state['map_data']),
            not np.array_equal(current_state['position'], previous_state['position'])
        ]
        
        return any(progress_indicators)
    
    def enhance_observation_with_heuristics(self, observation: Dict) -> Dict:
        """
        Enhance observation with additional heuristic information for A*
        """
        enhanced_obs = observation.copy()
        current_game_state = self.extract_game_state(observation)
        
        # Detect stagnation
        if not self.detect_progress(current_game_state, self.previous_game_state):
            self.stagnation_counter += 1
        else:
            self.stagnation_counter = 0
        
        # Add A* specific information
        enhanced_obs['stagnation_level'] = min(self.stagnation_counter / 100.0, 1.0)
        enhanced_obs['paths_planned'] = self.paths_planned
        
        # Add exploration density for A* goal selection
        map_data = current_game_state['map_data']
        if map_data.size > 0:
            exploration_density = np.sum(map_data > 0) / map_data.size
            enhanced_obs['exploration_density'] = exploration_density
        else:
            enhanced_obs['exploration_density'] = 0.0
        
        # Store current state for next comparison
        self.previous_game_state = current_game_state
        
        return enhanced_obs
    
    def run_episode(self, max_steps: int = None) -> Dict:
        """
        Run a single episode with the A* agent
        """
        if max_steps is None:
            max_steps = self.env_config.get('max_steps', 50000)
        
        observation, info = self.env.reset()
        self.agent.reset()
        
        episode_reward = 0
        episode_length = 0
        episode_start_time = time.time()
        
        # A* specific tracking
        positions_visited = set()
        paths_planned_episode = 0
        
        while episode_length < max_steps:
            try:
                # Extract game state from observation
                game_state = self.extract_game_state(observation)
                
                # Enhance observation with heuristics
                enhanced_obs = self.enhance_observation_with_heuristics(observation)
                
                # Get action from A* agent
                action, decision_info = self.agent.select_action(enhanced_obs, game_state)
                
                # Track if A* planned a new path
                if hasattr(self.agent, 'last_planning_step') and self.agent.last_planning_step == episode_length:
                    paths_planned_episode += 1
                    self.paths_planned += 1
                
                # Execute action
                observation, reward, terminated, truncated, info = self.env.step(action)
                
                # Track position for A* metrics
                if 'position' in observation:
                    pos = tuple(observation['position'])
                    positions_visited.add(pos)
                
                episode_reward += reward
                episode_length += 1
                self.total_steps += 1
                
                # Print progress for A*
                if self.enable_logging and episode_length % 500 == 0:
                    unique_positions = len(positions_visited)
                    efficiency = unique_positions / episode_length if episode_length > 0 else 0
                    current_objective = getattr(self.agent, 'current_objective', 'exploring')
                    action_source = getattr(self.agent, 'last_action_source', 'unknown')
                    
                    print(f"📊 Step {episode_length:,} | Reward: {episode_reward:.2f} | "
                          f"Positions: {unique_positions} | Objective: {current_objective} | "
                          f"Source: {action_source} | Quality: {efficiency:.3f}")
                
                if terminated or truncated:
                    break
                    
            except KeyboardInterrupt:
                print("\n🛑 Stopping...")
                break
            except Exception as e:
                print(f"❌ Error during episode: {e}")
                break
        
        episode_time = time.time() - episode_start_time
        
        # Calculate A* specific metrics
        unique_positions = len(positions_visited)
        exploration_efficiency = unique_positions / episode_length if episode_length > 0 else 0
        
        episode_metrics = {
            'reward': episode_reward,
            'length': episode_length,
            'time': episode_time,
            'unique_positions': unique_positions,
            'exploration_efficiency': exploration_efficiency,
            'paths_planned': paths_planned_episode,
            'steps_per_second': episode_length / episode_time if episode_time > 0 else 0
        }
        
        # Store metrics
        self.episode_rewards.append(episode_reward)
        self.episode_lengths.append(episode_length)
        self.episode_metrics.append(episode_metrics)
        
        return episode_metrics
    
    def run_interactive(self, max_episodes: int = None, max_steps_per_episode: int = None):
        """
        Run the A* agent interactively with real-time feedback
        """
        episode_count = 0
        
        try:
            while max_episodes is None or episode_count < max_episodes:
                episode_count += 1
                
                print(f"\n🌟 A* Episode {episode_count}")
                print("🎯 Intelligent pathfinding with goal-directed exploration")
                print("⏹️  Press Ctrl+C to stop")
                
                episode_metrics = self.run_episode(max_steps_per_episode)
                
                # Display episode results
                print(f"\n🎯 A* Episode {episode_count} Results:")
                print(f"🚶 Steps: {episode_metrics['length']:,}")
                print(f"🏆 Reward: {episode_metrics['reward']:.2f}")
                print(f"📍 Unique Positions: {episode_metrics['unique_positions']}")
                print(f"🔄 Exploration Efficiency: {episode_metrics['exploration_efficiency']:.4f}")
                print(f"🧭 Paths Planned: {episode_metrics['paths_planned']}")
                print(f"⚡ Steps/Second: {episode_metrics['steps_per_second']:.1f}")
                
                if episode_metrics['exploration_efficiency'] < 0.001:
                    print("⚠️  Low exploration efficiency - A* may need tuning")
                
                if episode_metrics['paths_planned'] == 0:
                    print("⚠️  No paths planned - Check A* goal selection")
                else:
                    print(f"✅ A* planning active - {episode_metrics['paths_planned']} paths computed")
                
        except KeyboardInterrupt:
            print(f"\n🛑 Interactive session stopped after {episode_count} episodes")
        
        # Final summary
        if self.episode_metrics:
            avg_reward = np.mean(self.episode_rewards)
            avg_length = np.mean(self.episode_lengths)
            total_unique_positions = sum(m['unique_positions'] for m in self.episode_metrics)
            total_paths_planned = sum(m['paths_planned'] for m in self.episode_metrics)
            
            print(f"\n📊 A* Agent Summary ({episode_count} episodes):")
            print(f"🏆 Average Reward: {avg_reward:.2f}")
            print(f"🚶 Average Steps: {avg_length:.0f}")
            print(f"🌍 Total Positions Explored: {total_unique_positions}")
            print(f"🧭 Total Paths Planned: {total_paths_planned}")
            print(f"📈 Overall Steps: {self.total_steps:,}")
    
    def get_performance_summary(self) -> Dict:
        """
        Get comprehensive performance summary for A*
        """
        if not self.episode_metrics:
            return {}
        
        return {
            'total_episodes': len(self.episode_metrics),
            'total_steps': self.total_steps,
            'average_reward': np.mean(self.episode_rewards),
            'average_length': np.mean(self.episode_lengths),
            'total_paths_planned': sum(m['paths_planned'] for m in self.episode_metrics),
            'average_exploration_efficiency': np.mean([m['exploration_efficiency'] for m in self.episode_metrics]),
            'total_unique_positions': sum(m['unique_positions'] for m in self.episode_metrics)
        }