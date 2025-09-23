#!/usr/bin/env python3
"""
Test a single algorithm with the professional executor
"""

import sys
import time
from pathlib import Path

# Add necessary paths
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent / 'comparison_agents'))
sys.path.append(str(Path(__file__).parent / 'comparison_agents' / 'search_algorithms'))
sys.path.append(str(Path(__file__).parent / 'v2'))

from professional_algorithm_executor import ProfessionalAlgorithmExecutor

def test_single_algorithm():
    """Test a single algorithm quickly"""
    print("🧪 TESTING SINGLE ALGORITHM")
    print("=" * 50)
    
    # Create executor
    executor = ProfessionalAlgorithmExecutor()
    
    # Test epsilon-greedy with short execution
    config = executor.algorithm_configs['epsilon_greedy_balanceada']
    config.max_steps = 50  # Very short test
    config.time_limit = 30.0  # 30 seconds
    config.visualization_enabled = True
    config.real_time_factor = 5.0  # Faster execution
    
    print(f"Testing: {config.algorithm} ({config.variant})")
    print(f"Max steps: {config.max_steps}, Time limit: {config.time_limit}s")
    
    try:
        result = executor.execute_algorithm(config)
        
        if result and result.success:
            print("✅ Algorithm test SUCCESSFUL!")
            print(f"📊 Steps completed: {result.total_steps}")
            print(f"🏆 Total reward: {result.total_reward}")
            print(f"⏱️ Execution time: {result.execution_time:.2f}s")
        else:
            print("❌ Algorithm test FAILED!")
            print(f"Error: {result.error_message if result else 'No result returned'}")
    
    except Exception as e:
        print(f"❌ Exception during test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_single_algorithm()