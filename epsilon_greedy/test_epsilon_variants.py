"""
Interactive Epsilon Greedy Testing Script
Test different epsilon values to study exploration vs exploitation performance
"""

import sys
import os
# Add parent directory to path to access v2 modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from v2.red_gym_env_v2 import RedGymEnv
from v2.stream_agent_wrapper import StreamWrapper
from epsilon_greedy.epsilon_variable_agent import VariableEpsilonGreedyAgent, EPSILON_CONFIGS
import time

def run_epsilon_experiment(epsilon_value, steps=1000, save_name=None):
    """Run experiment with specific epsilon value"""
    
    print(f"\n{'='*60}")
    print(f"STARTING EPSILON EXPERIMENT: ε = {epsilon_value}")
    print(f"{'='*60}")
    
    # Initialize environment
    print("Initializing Pokemon Red environment...")
    env_config = {
        'headless': False,
        'save_final_state': True,
        'early_stop': False,
        'action_freq': 24,
        'init_state': './init.state',
        'max_steps': 2**20,
        'print_rewards': True,
        'save_video': False,
        'fast_video': True,
        'session_path': './session_epsilon_test',
        'gb_path': './PokemonRed.gb',
        'debug': False,
        'sim_frame_dist': 2_000_000.0
    }
    
    env = RedGymEnv(env_config)
    env = StreamWrapper(
        env, 
        session_path="./session_epsilon_test",
        save_video=False
    )
    
    # Initialize agent
    agent = VariableEpsilonGreedyAgent(env, epsilon=epsilon_value)
    
    # Load previous training if available
    if save_name:
        agent.load_agent(f"./epsilon_greedy/agent_{save_name}.pkl")
    
    print(f"Starting experiment with {steps} steps...")
    print("Press Ctrl+C to stop early and see results")
    
    try:
        obs = env.reset()
        total_reward = 0
        step_count = 0
        
        start_time = time.time()
        
        for step in range(steps):
            # Select action
            action = agent.select_action(obs)
            
            # Take step in environment
            next_obs, reward, done, truncated, info = env.step(action)
            
            # Update agent
            agent.update_q_value(obs, action, reward, next_obs)
            
            # Track progress
            total_reward += reward
            step_count += 1
            obs = next_obs
            
            # Print progress every 100 steps
            if step % 100 == 0:
                elapsed = time.time() - start_time
                print(f"Step {step}/{steps} | Reward: {total_reward:.2f} | Time: {elapsed:.1f}s")
                agent.print_detailed_stats()
            
            if done or truncated:
                print(f"Episode ended at step {step}")
                obs = env.reset()
                total_reward = 0
        
        # Final statistics
        print(f"\n{'='*60}")
        print(f"EXPERIMENT COMPLETED: ε = {epsilon_value}")
        print(f"{'='*60}")
        agent.print_detailed_stats()
        
        # Save agent if name provided
        if save_name:
            agent.save_agent(f"./epsilon_greedy/agent_{save_name}.pkl")
            print(f"Agent saved as: agent_{save_name}.pkl")
        
        return agent.get_statistics()
        
    except KeyboardInterrupt:
        print(f"\n\nExperiment stopped by user at step {step_count}")
        agent.print_detailed_stats()
        
        if save_name:
            agent.save_agent(f"./epsilon_greedy/agent_{save_name}.pkl")
            print(f"Agent saved as: agent_{save_name}.pkl")
        
        return agent.get_statistics()
    
    finally:
        env.close()

def run_preset_experiment(preset_name, steps=1000):
    """Run experiment with preset epsilon configuration"""
    if preset_name not in EPSILON_CONFIGS:
        print(f"Unknown preset: {preset_name}")
        print(f"Available presets: {list(EPSILON_CONFIGS.keys())}")
        return None
    
    epsilon = EPSILON_CONFIGS[preset_name]
    return run_epsilon_experiment(epsilon, steps, save_name=preset_name)

def compare_epsilon_values():
    """Compare multiple epsilon values"""
    print("EPSILON COMPARISON STUDY")
    print("=" * 50)
    
    # Test different epsilon values
    test_epsilons = [0.1, 0.3, 0.5, 0.7, 0.9]
    results = {}
    
    for epsilon in test_epsilons:
        print(f"\nTesting epsilon = {epsilon}")
        input("Press Enter to continue...")
        results[epsilon] = run_epsilon_experiment(epsilon, steps=500, save_name=f"eps_{epsilon}")
    
    # Summary
    print("\n" + "=" * 60)
    print("EPSILON COMPARISON RESULTS")
    print("=" * 60)
    
    for epsilon, stats in results.items():
        print(f"ε = {epsilon}:")
        print(f"  Exploration Rate: {stats['exploration_rate']:.1%}")
        print(f"  Average Reward: {stats['avg_reward']:.3f}")
        print(f"  Q-Table Size: {stats['q_table_size']}")
        print()

def interactive_epsilon_testing():
    """Interactive epsilon testing with user input"""
    print("INTERACTIVE EPSILON TESTING")
    print("=" * 40)
    print("Available presets:")
    for name, epsilon in EPSILON_CONFIGS.items():
        print(f"  {name}: ε={epsilon}")
    print()
    
    while True:
        print("\nChoose testing mode:")
        print("1. Test specific epsilon value")
        print("2. Test preset configuration")
        print("3. Compare multiple epsilons")
        print("4. Exit")
        
        choice = input("Enter choice (1-4): ").strip()
        
        if choice == "1":
            try:
                epsilon = float(input("Enter epsilon value (0.0-1.0): "))
                if 0 <= epsilon <= 1:
                    steps = int(input("Enter number of steps (default 1000): ") or "1000")
                    run_epsilon_experiment(epsilon, steps, save_name=f"custom_{epsilon}")
                else:
                    print("Epsilon must be between 0.0 and 1.0")
            except ValueError:
                print("Invalid input")
        
        elif choice == "2":
            preset = input("Enter preset name: ").strip()
            steps = int(input("Enter number of steps (default 1000): ") or "1000")
            run_preset_experiment(preset, steps)
        
        elif choice == "3":
            compare_epsilon_values()
        
        elif choice == "4":
            break
        
        else:
            print("Invalid choice")

if __name__ == "__main__":
    print("Epsilon Greedy Testing Script")
    print("Study the impact of different epsilon values on agent performance")
    
    if len(sys.argv) > 1:
        # Command line usage
        if sys.argv[1] in EPSILON_CONFIGS:
            run_preset_experiment(sys.argv[1])
        else:
            try:
                epsilon = float(sys.argv[1])
                run_epsilon_experiment(epsilon)
            except ValueError:
                print(f"Invalid epsilon value: {sys.argv[1]}")
    else:
        # Interactive mode
        interactive_epsilon_testing()