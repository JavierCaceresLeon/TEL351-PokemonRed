"""
Epsilon Greedy Agent with Variable Epsilon
Allows testing different epsilon values to study exploration vs exploitation trade-offs
"""

import numpy as np
import random
from collections import defaultdict, deque
import time
import pickle
import os

class VariableEpsilonGreedyAgent:
    def __init__(self, env, epsilon=0.1, alpha=0.1, gamma=0.95, max_memory=10000):
        """
        Initialize Variable Epsilon Greedy Agent
        
        Args:
            env: Pokemon environment
            epsilon: Exploration rate (0.0 = pure exploitation, 1.0 = pure exploration)
            alpha: Learning rate for Q-values
            gamma: Discount factor for future rewards
            max_memory: Maximum number of states to remember
        """
        self.env = env
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.max_memory = max_memory
        
        # Q-table: state -> action -> value
        self.q_table = defaultdict(lambda: defaultdict(float))
        
        # Memory for recent states (to avoid loops)
        self.recent_states = deque(maxlen=50)
        
        # Action space
        self.actions = [0, 1, 2, 3, 4, 5, 6, 7]  # 8 possible actions
        
        # Statistics
        self.total_actions = 0
        self.exploration_actions = 0
        self.exploitation_actions = 0
        self.rewards_history = []
        
        # Performance tracking
        self.episode_rewards = []
        self.epsilon_history = []
        
        print(f"Initialized Variable Epsilon Greedy Agent with epsilon={epsilon}")
        print(f"  - High epsilon ({epsilon:.1f}) = More exploration")
        print(f"  - Low epsilon ({epsilon:.1f}) = More exploitation")
    
    def set_epsilon(self, new_epsilon):
        """Change epsilon value during runtime"""
        old_epsilon = self.epsilon
        self.epsilon = new_epsilon
        print(f"Epsilon changed from {old_epsilon:.3f} to {new_epsilon:.3f}")
    
    def get_state_key(self, obs):
        """Convert observation to hashable state key"""
        try:
            if hasattr(obs, 'flatten'):
                # If it's a numpy array, flatten and take a hash
                flattened = obs.flatten()
                # Take every 10th element to reduce dimensionality
                sample = flattened[::10]
                return tuple(sample.astype(int)[:50])  # Limit to 50 elements
            else:
                # If it's already a simple format
                return str(obs)[:100]  # Limit string length
        except:
            return str(obs)[:100]
    
    def select_action(self, obs):
        """Select action using epsilon-greedy strategy"""
        state_key = self.get_state_key(obs)
        self.total_actions += 1
        
        # Epsilon-greedy decision
        if random.random() < self.epsilon:
            # Exploration: random action
            action = random.choice(self.actions)
            self.exploration_actions += 1
            action_type = "EXPLORE"
        else:
            # Exploitation: best known action
            q_values = self.q_table[state_key]
            if q_values:
                action = max(q_values.keys(), key=lambda a: q_values[a])
            else:
                action = random.choice(self.actions)
            self.exploitation_actions += 1
            action_type = "EXPLOIT"
        
        # Avoid repeating recent states
        if state_key in self.recent_states:
            action = random.choice(self.actions)
            action_type = "ANTI_LOOP"
        
        self.recent_states.append(state_key)
        
        # Log action details every 100 actions
        if self.total_actions % 100 == 0:
            exp_rate = self.exploration_actions / self.total_actions * 100
            exp_rate = self.exploitation_actions / self.total_actions * 100
            print(f"Action {self.total_actions}: {action_type} (epsilon={self.epsilon:.3f}, Explore: {exp_rate:.1f}%)")
        
        return action
    
    def update_q_value(self, state, action, reward, next_state):
        """Update Q-value using Q-learning formula"""
        state_key = self.get_state_key(state)
        next_state_key = self.get_state_key(next_state)
        
        # Current Q-value
        current_q = self.q_table[state_key][action]
        
        # Best next Q-value
        next_q_values = self.q_table[next_state_key]
        if next_q_values:
            max_next_q = max(next_q_values.values())
        else:
            max_next_q = 0
        
        # Q-learning update
        new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
        self.q_table[state_key][action] = new_q
        
        # Track rewards
        self.rewards_history.append(reward)
    
    def get_statistics(self):
        """Get current agent statistics"""
        if self.total_actions == 0:
            return {
                'epsilon': self.epsilon,
                'total_actions': 0,
                'exploration_rate': 0,
                'exploitation_rate': 0,
                'avg_reward': 0,
                'q_table_size': 0
            }
        
        exploration_rate = self.exploration_actions / self.total_actions
        exploitation_rate = self.exploitation_actions / self.total_actions
        avg_reward = np.mean(self.rewards_history[-1000:]) if self.rewards_history else 0
        
        return {
            'epsilon': self.epsilon,
            'total_actions': self.total_actions,
            'exploration_rate': exploration_rate,
            'exploitation_rate': exploitation_rate,
            'avg_reward': avg_reward,
            'q_table_size': len(self.q_table),
            'recent_rewards': self.rewards_history[-10:] if self.rewards_history else []
        }
    
    def print_detailed_stats(self):
        """Print comprehensive statistics"""
        stats = self.get_statistics()
        print(f"\n=== Variable Epsilon Greedy Agent Stats ===")
        print(f"Epsilon (exploration rate): {stats['epsilon']:.3f}")
        print(f"Total Actions: {stats['total_actions']}")
        print(f"Exploration Actions: {stats['exploration_rate']:.1%}")
        print(f"Exploitation Actions: {stats['exploitation_rate']:.1%}")
        print(f"Average Reward (last 1000): {stats['avg_reward']:.3f}")
        print(f"Q-Table Size: {stats['q_table_size']} states")
        print(f"Recent Rewards: {stats['recent_rewards']}")
        print("=" * 45)
    
    def save_agent(self, filename):
        """Save agent state to file"""
        agent_data = {
            'q_table': dict(self.q_table),
            'epsilon': self.epsilon,
            'alpha': self.alpha,
            'gamma': self.gamma,
            'total_actions': self.total_actions,
            'exploration_actions': self.exploration_actions,
            'exploitation_actions': self.exploitation_actions,
            'rewards_history': self.rewards_history,
            'episode_rewards': self.episode_rewards
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(agent_data, f)
        print(f"Agent saved to {filename}")
    
    def load_agent(self, filename):
        """Load agent state from file"""
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                agent_data = pickle.load(f)
            
            self.q_table = defaultdict(lambda: defaultdict(float), agent_data['q_table'])
            self.epsilon = agent_data['epsilon']
            self.alpha = agent_data['alpha']
            self.gamma = agent_data['gamma']
            self.total_actions = agent_data['total_actions']
            self.exploration_actions = agent_data['exploration_actions']
            self.exploitation_actions = agent_data['exploitation_actions']
            self.rewards_history = agent_data['rewards_history']
            self.episode_rewards = agent_data.get('episode_rewards', [])
            
            print(f"Agent loaded from {filename}")
            print(f"Loaded Q-table with {len(self.q_table)} states")
        else:
            print(f"File {filename} not found, starting fresh")


# Epsilon preset configurations for easy testing
EPSILON_CONFIGS = {
    'very_high_exploration': 0.9,    # 90% exploration - almost random
    'high_exploration': 0.7,         # 70% exploration - lots of exploration
    'balanced': 0.5,                 # 50% exploration - balanced approach
    'moderate_exploitation': 0.3,    # 30% exploration - more exploitation
    'low_exploration': 0.1,          # 10% exploration - mostly exploitation
    'very_low_exploration': 0.05,    # 5% exploration - almost pure exploitation
    'pure_exploitation': 0.01        # 1% exploration - nearly greedy
}

def create_agent_with_preset(env, preset_name, **kwargs):
    """Create agent with preset epsilon configuration"""
    if preset_name not in EPSILON_CONFIGS:
        print(f"Unknown preset: {preset_name}")
        print(f"Available presets: {list(EPSILON_CONFIGS.keys())}")
        return None
    
    epsilon = EPSILON_CONFIGS[preset_name]
    agent = VariableEpsilonGreedyAgent(env, epsilon=epsilon, **kwargs)
    print(f"Created agent with '{preset_name}' preset (epsilon={epsilon})")
    return agent

if __name__ == "__main__":
    print("Variable Epsilon Greedy Agent")
    print("Available epsilon presets:")
    for name, epsilon in EPSILON_CONFIGS.items():
        print(f"  {name}: epsilon={epsilon}")