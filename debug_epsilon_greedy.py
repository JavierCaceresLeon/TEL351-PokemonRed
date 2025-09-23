#!/usr/bin/env python3
"""
Debug test for epsilon greedy agent
"""

import sys
import traceback
from pathlib import Path

# Add necessary paths
sys.path.append(str(Path(__file__).parent / 'comparison_agents'))

def test_epsilon_greedy_basic():
    """Test epsilon greedy agent with basic mock observation"""
    print("🔍 DEBUGGING EPSILON GREEDY AGENT")
    print("=" * 50)
    
    try:
        from epsilon_greedy_agent import EpsilonGreedyAgent
        print("✅ Import successful")
        
        # Create agent with correct parameters
        agent = EpsilonGreedyAgent(epsilon_start=0.3, epsilon_min=0.05, epsilon_decay=0.995)
        print("✅ Agent created")
        
        # Create basic mock observation
        mock_obs = {
            'location': {'x': 0, 'y': 0, 'map_id': 1},
            'player': {'health': 100, 'level': 5},
            'badges': 0,
            'events': 5,  # Should be a number, not dict
            'seen_coords': set(),
            'action_history': []
        }
        print("✅ Mock observation created")
        
        # Test get_action
        action = agent.get_action(mock_obs)
        print(f"✅ Action selected: {action}")
        
        # Test multiple steps
        for i in range(5):
            action = agent.get_action(mock_obs)
            print(f"Step {i+1}: Action {action}, Epsilon {agent.epsilon:.3f}")
        
        print("🎉 All tests passed!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    test_epsilon_greedy_basic()