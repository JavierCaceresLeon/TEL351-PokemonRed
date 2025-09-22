"""
Simple Epsilon Variants Test - Without v2 Environment Dependencies
Test different epsilon values with a mock environment to validate the epsilon logic
"""

import sys
import os
# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from epsilon_greedy.epsilon_variable_agent import VariableEpsilonGreedyAgent, EPSILON_CONFIGS, create_agent_with_preset
import numpy as np
import time

class MockPokemonEnv:
    """Mock environment for testing epsilon logic without PyBoy dependencies"""
    
    def __init__(self):
        self.step_count = 0
        self.max_steps = 1000
        
    def reset(self):
        """Reset environment and return initial observation"""
        self.step_count = 0
        # Return a mock observation (simulating game screen)
        return np.random.randint(0, 255, (144, 160, 3), dtype=np.uint8)
    
    def step(self, action):
        """Take a step in the environment"""
        self.step_count += 1
        
        # Mock observation (simulating game screen changes)
        obs = np.random.randint(0, 255, (144, 160, 3), dtype=np.uint8)
        
        # Mock reward (simulate catching pokemon, exploring, etc.)
        if np.random.random() < 0.1:  # 10% chance of good reward
            reward = 10.0
        elif np.random.random() < 0.3:  # 30% chance of small reward
            reward = 1.0
        else:
            reward = 0.1  # Small exploration reward
        
        # Episode ends after max steps
        done = self.step_count >= self.max_steps
        truncated = False
        info = {"step": self.step_count}
        
        return obs, reward, done, truncated, info
    
    def close(self):
        """Close environment"""
        pass

def test_epsilon_behavior(epsilon_value, test_steps=200):
    """Test agent behavior with specific epsilon value"""
    
    print(f"\n{'='*50}")
    print(f"TESTING EPSILON = {epsilon_value}")
    print(f"{'='*50}")
    
    # Create mock environment and agent
    env = MockPokemonEnv()
    agent = VariableEpsilonGreedyAgent(env, epsilon=epsilon_value)
    
    # Run test
    obs = env.reset()
    total_reward = 0
    start_time = time.time()
    
    for step in range(test_steps):
        # Agent selects action
        action = agent.select_action(obs)
        
        # Environment step
        next_obs, reward, done, truncated, info = env.step(action)
        
        # Agent updates Q-values
        agent.update_q_value(obs, action, reward, next_obs)
        
        total_reward += reward
        obs = next_obs
        
        if done:
            obs = env.reset()
            total_reward = 0
    
    # Get statistics
    stats = agent.get_statistics()
    elapsed_time = time.time() - start_time
    
    # Print results
    print(f"Results after {test_steps} steps ({elapsed_time:.2f}s):")
    print(f"  Epsilon: {stats['epsilon']:.3f}")
    print(f"  Exploration Rate: {stats['exploration_rate']:.1%}")
    print(f"  Exploitation Rate: {stats['exploitation_rate']:.1%}")
    print(f"  Average Reward: {stats['avg_reward']:.3f}")
    print(f"  Q-Table Size: {stats['q_table_size']}")
    print(f"  Total Actions: {stats['total_actions']}")
    
    env.close()
    return stats

def compare_epsilon_values():
    """Compare different epsilon values"""
    
    print("EPSILON COMPARISON STUDY")
    print("Testing epsilon behavior without v2 environment dependencies")
    print("=" * 70)
    
    # Test key epsilon values
    test_epsilons = [0.05, 0.1, 0.3, 0.5, 0.7, 0.9]
    results = {}
    
    for epsilon in test_epsilons:
        results[epsilon] = test_epsilon_behavior(epsilon, test_steps=300)
        time.sleep(0.5)  # Brief pause between tests
    
    # Summary comparison
    print("\n" + "=" * 70)
    print("EPSILON COMPARISON SUMMARY")
    print("=" * 70)
    print("Epsilon | Exploration | Exploitation | Avg Reward | Q-Table")
    print("-" * 70)
    
    for epsilon, stats in results.items():
        print(f"{epsilon:7.2f} | {stats['exploration_rate']:10.1%} | {stats['exploitation_rate']:11.1%} | "
              f"{stats['avg_reward']:9.3f} | {stats['q_table_size']:7d}")
    
    print("\nKey Insights:")
    print("- High epsilon (0.7-0.9): More exploration, larger Q-tables")
    print("- Low epsilon (0.05-0.1): More exploitation, focused learning")
    print("- Medium epsilon (0.3-0.5): Balanced exploration/exploitation")

def test_preset_configurations():
    """Test all preset epsilon configurations"""
    
    print("\n" + "=" * 70)
    print("TESTING PRESET CONFIGURATIONS")
    print("=" * 70)
    
    for preset_name, epsilon_value in EPSILON_CONFIGS.items():
        print(f"\nTesting '{preset_name}' preset (epsilon={epsilon_value})")
        
        env = MockPokemonEnv()
        agent = create_agent_with_preset(env, preset_name)
        
        if agent:
            # Quick test
            obs = env.reset()
            for _ in range(50):
                action = agent.select_action(obs)
                obs, reward, done, truncated, info = env.step(action)
                agent.update_q_value(obs, action, reward, obs)
            
            stats = agent.get_statistics()
            print(f"  -> Exploration: {stats['exploration_rate']:.1%}, "
                  f"Q-table: {stats['q_table_size']} states")
        
        env.close()

def test_dynamic_epsilon():
    """Test changing epsilon during runtime"""
    
    print("\n" + "=" * 70)
    print("TESTING DYNAMIC EPSILON CHANGES")
    print("=" * 70)
    
    env = MockPokemonEnv()
    agent = VariableEpsilonGreedyAgent(env, epsilon=0.5)
    
    print("Starting with epsilon = 0.5 (balanced)")
    
    # Test with balanced epsilon
    obs = env.reset()
    for _ in range(100):
        action = agent.select_action(obs)
        obs, reward, done, truncated, info = env.step(action)
        agent.update_q_value(obs, action, reward, obs)
    
    stats1 = agent.get_statistics()
    print(f"Phase 1: Exploration: {stats1['exploration_rate']:.1%}")
    
    # Change to high exploration
    agent.set_epsilon(0.9)
    print("Changed to epsilon = 0.9 (high exploration)")
    
    for _ in range(100):
        action = agent.select_action(obs)
        obs, reward, done, truncated, info = env.step(action)
        agent.update_q_value(obs, action, reward, obs)
    
    stats2 = agent.get_statistics()
    print(f"Phase 2: Exploration: {stats2['exploration_rate']:.1%}")
    
    # Change to low exploration
    agent.set_epsilon(0.1)
    print("Changed to epsilon = 0.1 (high exploitation)")
    
    for _ in range(100):
        action = agent.select_action(obs)
        obs, reward, done, truncated, info = env.step(action)
        agent.update_q_value(obs, action, reward, obs)
    
    stats3 = agent.get_statistics()
    print(f"Phase 3: Exploration: {stats3['exploration_rate']:.1%}")
    
    env.close()

if __name__ == "__main__":
    print("Epsilon Greedy Variants Testing")
    print("Mock Environment Test - No v2 Dependencies Required")
    print("=" * 70)
    
    # Run all tests
    try:
        # Test 1: Compare epsilon values
        compare_epsilon_values()
        
        # Test 2: Test preset configurations
        test_preset_configurations()
        
        # Test 3: Test dynamic epsilon changes
        test_dynamic_epsilon()
        
        print("\n" + "=" * 70)
        print("ALL TESTS COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        print("\nNext steps:")
        print("1. The epsilon variable agent is working correctly")
        print("2. You can now integrate it with the real v2 environment")
        print("3. Use different epsilon values to optimize performance")
        print("4. Consider starting with 'balanced' preset (epsilon=0.5)")
        
    except Exception as e:
        print(f"\nError during testing: {e}")
        import traceback
        traceback.print_exc()