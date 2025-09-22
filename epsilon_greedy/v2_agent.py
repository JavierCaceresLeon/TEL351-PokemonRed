"""
V2 Agent Wrapper for Epsilon Greedy Algorithm
==============================================

This module integrates the Epsilon Greedy agent with the Pokemon Red v2 environment,
providing a direct comparison alternative to the PPO-based agent.
"""

import sys
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Import v2 environment
from v2.red_gym_env_v2 import RedGymEnv
from v2.stream_agent_wrapper import StreamWrapper

# Import our epsilon greedy agent
from epsilon_greedy.epsilon_greedy_agent import EpsilonGreedyAgent, GameScenario


class V2EpsilonGreedyAgent:
    """
    Wrapper class that adapts the Epsilon Greedy agent to work with the v2 environment
    """
    
    def __init__(self, 
                 env_config: Dict,
                 agent_config: Dict = None,
                 enable_logging: bool = True):
        
        self.env_config = env_config
        self.enable_logging = enable_logging
        
        # Default agent configuration
        default_agent_config = {
            'epsilon_start': 0.4,
            'epsilon_min': 0.05,
            'epsilon_decay': 0.9995,
            'scenario_detection_enabled': True
        }
        
        if agent_config:
            default_agent_config.update(agent_config)
        
        self.agent_config = default_agent_config
        
        # Initialize environment
        self.env = StreamWrapper(
            RedGymEnv(env_config),
            stream_metadata={
                "user": "epsilon-greedy-v2",
                "env_id": 0,
                "color": "#44aa77",
                "extra": "Epsilon Greedy Heuristic Agent",
            }
        )
        
        # Initialize agent
        self.agent = EpsilonGreedyAgent(**self.agent_config)
        
        # Performance tracking
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_metrics = []
        self.total_steps = 0
        
        # State tracking for enhanced heuristics
        self.previous_game_state = None
        self.stagnation_counter = 0
        self.exploration_map = None
        
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
            'screen': observation.get('screens', np.zeros((72, 80, 3)))
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
            not np.array_equal(current_state['map_data'], previous_state['map_data'])
        ]
        
        return any(progress_indicators)
    
    def enhance_observation_with_heuristics(self, observation: Dict) -> Dict:
        """
        Enhance observation with additional heuristic information
        """
        enhanced_obs = observation.copy()
        current_game_state = self.extract_game_state(observation)
        
        # Detect stagnation
        if not self.detect_progress(current_game_state, self.previous_game_state):
            self.stagnation_counter += 1
        else:
            self.stagnation_counter = 0
        
        # Add stagnation information
        enhanced_obs['stagnation_level'] = min(self.stagnation_counter / 100.0, 1.0)
        
        # Add exploration density
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
        Run a single episode with the epsilon greedy agent
        """
        if max_steps is None:
            max_steps = self.env_config.get('max_steps', 50000)
        
        observation, info = self.env.reset()
        self.agent.reset()
        
        episode_reward = 0
        episode_length = 0
        episode_start_time = time.time()
        
        # Episode-specific metrics
        scenario_counts = {scenario: 0 for scenario in GameScenario}
        action_counts = {action: 0 for action in range(7)}
        rewards_over_time = []
        
        done = False
        step = 0
        
        while not done and step < max_steps:
            # Enhance observation with heuristic information
            enhanced_obs = self.enhance_observation_with_heuristics(observation)
            
            # Update agent's position tracking
            self.agent.update_position(enhanced_obs)
            
            # Select action using epsilon greedy strategy
            action_start_time = time.time()
            action = self.agent.select_action(enhanced_obs)
            action_time = time.time() - action_start_time
            
            # Execute action
            observation, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            
            # Update metrics
            episode_reward += reward
            episode_length += 1
            self.total_steps += 1
            
            # Track scenario and action usage
            current_scenario = self.agent.detect_scenario(enhanced_obs)
            scenario_counts[current_scenario] += 1
            action_counts[action] += 1
            
            rewards_over_time.append(reward)
            
            # Log progress periodically
            if self.enable_logging and step % 1000 == 0:
                agent_metrics = self.agent.get_performance_metrics()
                print(f"Step {step}: Reward={reward:.3f}, Scenario={current_scenario.value}, "
                      f"Epsilon={agent_metrics['current_epsilon']:.3f}")
            
            step += 1
        
        episode_time = time.time() - episode_start_time
        
        # Compile episode metrics
        agent_metrics = self.agent.get_performance_metrics()
        
        episode_metrics = {
            'episode_reward': episode_reward,
            'episode_length': episode_length,
            'episode_time': episode_time,
            'steps_per_second': episode_length / episode_time if episode_time > 0 else 0,
            'average_reward': episode_reward / episode_length if episode_length > 0 else 0,
            'final_epsilon': agent_metrics['current_epsilon'],
            'scenario_distribution': {s.value: count/episode_length for s, count in scenario_counts.items()},
            'action_distribution': {f'action_{a}': count/episode_length for a, count in action_counts.items()},
            'exploration_efficiency': agent_metrics['exploration_efficiency'],
            'total_steps': self.total_steps,
            'stagnation_final': self.stagnation_counter,
            'rewards_std': np.std(rewards_over_time) if rewards_over_time else 0,
            'rewards_trend': np.polyfit(range(len(rewards_over_time)), rewards_over_time, 1)[0] if len(rewards_over_time) > 1 else 0
        }
        
        # Store episode data
        self.episode_rewards.append(episode_reward)
        self.episode_lengths.append(episode_length)
        self.episode_metrics.append(episode_metrics)
        
        return episode_metrics
    
    def run_multiple_episodes(self, 
                            num_episodes: int, 
                            max_steps_per_episode: int = None,
                            save_results: bool = True) -> List[Dict]:
        """
        Run multiple episodes and collect comprehensive metrics
        """
        print(f"Running {num_episodes} episodes with Epsilon Greedy Agent...")
        
        all_metrics = []
        
        for episode in range(num_episodes):
            print(f"\n--- Episode {episode + 1}/{num_episodes} ---")
            
            episode_metrics = self.run_episode(max_steps_per_episode)
            all_metrics.append(episode_metrics)
            
            print(f"Episode {episode + 1} Results:")
            print(f"  Reward: {episode_metrics['episode_reward']:.2f}")
            print(f"  Length: {episode_metrics['episode_length']}")
            print(f"  Avg Reward: {episode_metrics['average_reward']:.4f}")
            print(f"  Exploration Efficiency: {episode_metrics['exploration_efficiency']:.3f}")
            print(f"  Final Epsilon: {episode_metrics['final_epsilon']:.3f}")
        
        # Save results if requested
        if save_results:
            self.save_results(all_metrics)
        
        return all_metrics
    
    def save_results(self, metrics: List[Dict], filename: Optional[str] = None):
        """
        Save episode results to file
        """
        if filename is None:
            timestamp = int(time.time())
            filename = f"epsilon_greedy_results_{timestamp}.json"
        
        # Prepare data for saving
        save_data = {
            'agent_config': self.agent_config,
            'env_config': self.env_config,
            'episode_metrics': metrics,
            'summary_stats': self.get_summary_statistics()
        }
        
        save_path = Path(filename)
        
        import json
        with open(save_path, 'w') as f:
            json.dump(save_data, f, indent=2, default=str)
        
        print(f"Results saved to {save_path}")
    
    def get_summary_statistics(self) -> Dict:
        """
        Get summary statistics across all episodes
        """
        if not self.episode_rewards:
            return {}
        
        return {
            'total_episodes': len(self.episode_rewards),
            'total_steps': self.total_steps,
            'mean_episode_reward': np.mean(self.episode_rewards),
            'std_episode_reward': np.std(self.episode_rewards),
            'max_episode_reward': np.max(self.episode_rewards),
            'min_episode_reward': np.min(self.episode_rewards),
            'mean_episode_length': np.mean(self.episode_lengths),
            'std_episode_length': np.std(self.episode_lengths),
            'reward_improvement_trend': np.polyfit(range(len(self.episode_rewards)), self.episode_rewards, 1)[0] if len(self.episode_rewards) > 1 else 0
        }
    
    def close(self):
        """
        Clean up environment
        """
        self.env.close()


def main():
    """
    Main function to run the epsilon greedy agent
    """
    # Environment configuration (matching v2 settings)
    env_config = {
        'headless': True,
        'save_final_state': False,
        'early_stop': False,
        'action_freq': 24,
        'init_state': '../init.state',
        'max_steps': 2048 * 20,  # Shorter episodes for testing
        'print_rewards': True,
        'save_video': False,
        'fast_video': True,
        'session_path': Path('epsilon_greedy_session'),
        'gb_path': '../PokemonRed.gb',
        'debug': False,
        'reward_scale': 0.5,
        'explore_weight': 0.25
    }
    
    # Agent configuration
    agent_config = {
        'epsilon_start': 0.5,
        'epsilon_min': 0.05,
        'epsilon_decay': 0.9995,
        'scenario_detection_enabled': True
    }
    
    # Create and run agent
    v2_agent = V2EpsilonGreedyAgent(
        env_config=env_config,
        agent_config=agent_config,
        enable_logging=True
    )
    
    try:
        # Run episodes
        results = v2_agent.run_multiple_episodes(
            num_episodes=5,
            max_steps_per_episode=env_config['max_steps'],
            save_results=True
        )
        
        # Print summary
        summary = v2_agent.get_summary_statistics()
        print("\n" + "="*50)
        print("EPSILON GREEDY AGENT SUMMARY")
        print("="*50)
        for key, value in summary.items():
            print(f"{key}: {value}")
        
    finally:
        v2_agent.close()


if __name__ == "__main__":
    main()
