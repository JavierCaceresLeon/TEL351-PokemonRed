#!/usr/bin/env python3
"""
Quick Professional Algorithm Test Suite
======================================

Quick test suite to verify algorithm implementations work correctly
with PyBoy and generate some sample data.
"""

import os
import sys
import time
import traceback
from pathlib import Path

# Add necessary paths
sys.path.append(str(Path(__file__).parent / 'comparison_agents'))
sys.path.append(str(Path(__file__).parent / 'comparison_agents' / 'search_algorithms'))
sys.path.append(str(Path(__file__).parent / 'v2'))

def test_epsilon_greedy_agent():
    """Test enhanced Epsilon-Greedy agent"""
    print("Testing Enhanced Epsilon-Greedy Agent...")
    
    try:
        from epsilon_greedy_agent import EpsilonGreedyAgent
        
        agent = EpsilonGreedyAgent(
            epsilon_start=0.5,
            epsilon_min=0.05,
            epsilon_decay=0.995
        )
        
        # Test with mock observation
        mock_obs = {
            'screen': [[0] * 80] * 72,
            'x': 5, 'y': 3,
            'health': [1.0],
            'badges': [0] * 8,
            'events': [0] * 100
        }
        
        # Test action selection
        for i in range(10):
            action = agent.get_action(mock_obs)
            print(f"  Step {i+1}: Action {action}, Epsilon: {agent.epsilon:.3f}")
        
        print("✅ Epsilon-Greedy Agent: PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Epsilon-Greedy Agent: FAILED - {e}")
        traceback.print_exc()
        return False

def test_astar_agent():
    """Test enhanced A* agent"""
    print("\\nTesting Enhanced A* Agent...")
    
    try:
        from astar_agent import AStarAgent
        
        agent = AStarAgent()
        
        # Test with mock observation
        mock_obs = {
            'screen': [[0] * 80] * 72,
            'x': 2, 'y': 2,
            'reward': 0.1
        }
        
        # Test action selection
        for i in range(10):
            action = agent.get_action(mock_obs)
            print(f"  Step {i+1}: Action {action}, Position: {agent.current_position}")
        
        print("✅ A* Agent: PASSED")
        return True
        
    except Exception as e:
        print(f"❌ A* Agent: FAILED - {e}")
        traceback.print_exc()
        return False

def test_algorithm_imports():
    """Test that all algorithm modules can be imported"""
    print("\\nTesting Algorithm Imports...")
    
    test_results = {}
    
    # Test algorithms
    algorithms = [
        ('epsilon_greedy_agent', 'EpsilonGreedyAgent'),
        ('astar_agent', 'AStarAgent'),
        ('bfs_agent', 'BFSAgent'),
        ('tabu_agent', 'TabuSearchAgent'),
        ('hill_climbing_agent', 'HillClimbingAgent'),
        ('simulated_annealing_agent', 'SimulatedAnnealingAgent')
    ]
    
    for module_name, class_name in algorithms:
        try:
            module = __import__(module_name)
            agent_class = getattr(module, class_name)
            print(f"  ✅ {class_name}: Import successful")
            test_results[module_name] = True
        except Exception as e:
            print(f"  ❌ {class_name}: Import failed - {e}")
            test_results[module_name] = False
    
    success_rate = sum(test_results.values()) / len(test_results) * 100
    print(f"\\nImport Success Rate: {success_rate:.1f}% ({sum(test_results.values())}/{len(test_results)})")
    
    return success_rate > 80

def check_pyboy_compatibility():
    """Check PyBoy and environment compatibility"""
    print("\\nChecking PyBoy Compatibility...")
    
    try:
        # Check if PyBoy ROM exists
        rom_path = Path('PokemonRed.gb')
        if rom_path.exists():
            print("  ✅ Pokemon Red ROM: Found")
        else:
            print("  ⚠️  Pokemon Red ROM: Not found (required for full execution)")
        
        # Check if v2 environment can be imported
        try:
            from red_gym_env_v2 import RedGymEnv
            print("  ✅ PyBoy Environment: Import successful")
        except Exception as e:
            print(f"  ❌ PyBoy Environment: Import failed - {e}")
            return False
        
        # Check if states exist
        state_files = ['has_pokedex.state', 'init.state']
        for state_file in state_files:
            if Path(state_file).exists():
                print(f"  ✅ {state_file}: Found")
            else:
                print(f"  ⚠️  {state_file}: Not found")
        
        return True
        
    except Exception as e:
        print(f"  ❌ PyBoy Compatibility Check: Failed - {e}")
        return False

def test_professional_executor():
    """Test the professional algorithm executor"""
    print("\\nTesting Professional Algorithm Executor...")
    
    try:
        from professional_algorithm_executor import ProfessionalAlgorithmExecutor
        
        executor = ProfessionalAlgorithmExecutor()
        print("  ✅ Professional Executor: Initialized successfully")
        
        # Check algorithm configurations
        configs = executor.algorithm_configs
        print(f"  ✅ Algorithm Configurations: {len(configs)} algorithms configured")
        
        for algo_name in list(configs.keys())[:3]:  # Test first 3
            config = configs[algo_name]
            print(f"    - {algo_name}: Max steps {config.max_steps}, Time limit {config.time_limit}s")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Professional Executor: Failed - {e}")
        traceback.print_exc()
        return False

def check_results_structure():
    """Check existing results structure"""
    print("\\nChecking Results Structure...")
    
    results_dir = Path('RESULTADOS')
    if not results_dir.exists():
        print("  ⚠️  RESULTADOS directory not found")
        return False
    
    # Count files by algorithm type
    algorithm_dirs = {
        'ppo': 'PPO',
        'epsilon_greedy': 'Epsilon-Greedy',
        'search_algorithms': 'Search Algorithms'
    }
    
    total_files = 0
    for algo_dir, algo_name in algorithm_dirs.items():
        algo_path = results_dir / algo_dir
        if algo_path.exists():
            csv_files = list(algo_path.rglob('*.csv'))
            total_files += len(csv_files)
            print(f"  ✅ {algo_name}: {len(csv_files)} result files")
        else:
            print(f"  ⚠️  {algo_name}: Directory not found")
    
    print(f"  📊 Total result files: {total_files}")
    return total_files > 50

def main():
    """Main test function"""
    print("🔧 PROFESSIONAL ALGORITHM TEST SUITE")
    print("=" * 50)
    
    test_results = []
    
    # Run tests
    test_results.append(("Algorithm Imports", test_algorithm_imports()))
    test_results.append(("PyBoy Compatibility", check_pyboy_compatibility()))
    test_results.append(("Epsilon-Greedy Agent", test_epsilon_greedy_agent()))
    test_results.append(("A* Agent", test_astar_agent()))
    test_results.append(("Professional Executor", test_professional_executor()))
    test_results.append(("Results Structure", check_results_structure()))
    
    # Summary
    print("\\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    
    passed_tests = 0
    for test_name, result in test_results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:<25} {status}")
        if result:
            passed_tests += 1
    
    success_rate = passed_tests / len(test_results) * 100
    print(f"\\nOverall Success Rate: {success_rate:.1f}% ({passed_tests}/{len(test_results)})")
    
    if success_rate >= 80:
        print("🎉 SYSTEM READY FOR PROFESSIONAL ALGORITHM EXECUTION")
    else:
        print("⚠️  SYSTEM NEEDS ATTENTION BEFORE FULL EXECUTION")
    
    print("=" * 50)

if __name__ == "__main__":
    main()