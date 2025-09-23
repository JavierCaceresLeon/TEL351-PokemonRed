"""
A* Search Agent for Pokemon Red Environment
==========================================

This agent implements the A* search algorithm with sophisticated heuristics
for navigating the Pokemon Red environment efficiently.

Key Features:
- A* pathfinding with Manhattan distance heuristic
- Priority queue for optimal node exploration
- Goal-directed navigation
- Dynamic heuristic adaptation
- Memory-efficient state representation
"""

import numpy as np
import heapq
import json
from typing import Dict, Tuple, List, Optional, Set
from dataclasses import dataclass
from enum import Enum
import random
from collections import deque
import time


class GameScenario(Enum):
    """Different scenarios the agent can encounter"""
    EXPLORATION = "exploration"
    NAVIGATION = "navigation"
    PROGRESSION = "progression"
    STUCK = "stuck"


@dataclass
class AStarNode:
    """Node for A* search"""
    position: Tuple[int, int]
    g_cost: float  # Cost from start
    h_cost: float  # Heuristic cost to goal
    f_cost: float  # Total cost (g + h)
    parent: Optional['AStarNode'] = None
    action: Optional[int] = None
    
    def __lt__(self, other):
        return self.f_cost < other.f_cost


@dataclass
class HeuristicWeights:
    """Weights for different heuristic components"""
    distance_weight: float = 1.0
    exploration_bonus: float = 0.3
    obstacle_penalty: float = 2.0
    goal_attraction: float = 1.5


class AStarAgent:
    """
    A* Search Agent for Pokemon Red Environment
    
    This agent uses A* search algorithm to find optimal paths to objectives
    in the Pokemon Red game environment.
    """
    
    def __init__(self):
        """Initialize A* agent with default configuration"""
        self.action_space = 7  # UP, DOWN, LEFT, RIGHT, A, B, START
        self.actions = {
            0: "UP",
            1: "DOWN", 
            2: "LEFT",
            3: "RIGHT",
            4: "A",
            5: "B",
            6: "START"
        }
        
        # A* specific parameters
        self.open_set = []  # Priority queue for nodes to explore
        self.closed_set = set()  # Explored nodes
        self.came_from = {}  # Parent tracking
        self.g_score = {}  # Cost from start
        self.f_score = {}  # Total estimated cost
        
        # Game state tracking
        self.current_position = (0, 0)
        self.goal_position = (5, 3)  # Approximate location of Pokemon selection
        self.visited_positions = set()
        self.position_counts = {}
        self.last_positions = deque(maxlen=10)
        
        # Heuristics and weights
        self.weights = HeuristicWeights()
        self.scenario = GameScenario.EXPLORATION
        
        # Action tracking
        self.action_history = deque(maxlen=20)
        self.last_rewards = deque(maxlen=5)
        self.step_count = 0
        self.stuck_counter = 0
        
        # Game knowledge
        self.known_objectives = [
            (5, 3),   # Pokemon selection area
            (3, 2),   # Professor Oak's position
            (4, 6),   # Exit from player's house
            (5, 8),   # Route to lab
        ]
        
        print("A* Agent initialized with pathfinding capabilities")
    
    def manhattan_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """Calculate Manhattan distance between two positions"""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    def euclidean_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """Calculate Euclidean distance between two positions"""
        return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
    
    def calculate_heuristic(self, position: Tuple[int, int], goal: Tuple[int, int]) -> float:
        """
        Calculate A* heuristic cost from position to goal
        Uses Manhattan distance as base with additional factors
        """
        base_distance = self.manhattan_distance(position, goal)
        
        # Add exploration bonus for unvisited areas
        exploration_bonus = 0
        if position not in self.visited_positions:
            exploration_bonus = -self.weights.exploration_bonus
        
        # Penalty for frequently visited positions (avoid loops)
        visit_penalty = 0
        if position in self.position_counts:
            visit_penalty = self.position_counts[position] * 0.1
        
        # Goal attraction increases as we get closer
        goal_distance = self.manhattan_distance(position, goal)
        if goal_distance < 3:
            goal_attraction = -self.weights.goal_attraction * (3 - goal_distance)
        else:
            goal_attraction = 0
        
        total_heuristic = (base_distance * self.weights.distance_weight + 
                          visit_penalty + exploration_bonus + goal_attraction)
        
        return max(0, total_heuristic)  # Ensure non-negative
    
    def get_neighbors(self, position: Tuple[int, int]) -> List[Tuple[Tuple[int, int], int]]:
        """Get valid neighboring positions and corresponding actions"""
        neighbors = []
        
        # Movement actions (UP, DOWN, LEFT, RIGHT)
        moves = [
            ((-1, 0), 0),  # UP
            ((1, 0), 1),   # DOWN
            ((0, -1), 2),  # LEFT
            ((0, 1), 3),   # RIGHT
        ]
        
        for (dx, dy), action in moves:
            new_pos = (position[0] + dx, position[1] + dy)
            # Basic bounds checking (simplified for game environment)
            if (0 <= new_pos[0] <= 20 and 0 <= new_pos[1] <= 20):
                neighbors.append((new_pos, action))
        
        # Add interaction actions (A, B, START) at current position
        for action in [4, 5, 6]:  # A, B, START
            neighbors.append((position, action))
        
        return neighbors
    
    def reconstruct_path(self, came_from: Dict, current: Tuple[int, int]) -> List[int]:
        """Reconstruct path from start to goal"""
        path = []
        while current in came_from:
            path.append(came_from[current][1])  # Action taken to reach current
            current = came_from[current][0]    # Parent position
        
        path.reverse()
        return path
    
    def astar_search(self, start: Tuple[int, int], goal: Tuple[int, int], max_iterations: int = 100) -> List[int]:
        """
        Perform A* search from start to goal
        Returns list of actions to reach goal
        """
        open_set = []
        closed_set = set()
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self.calculate_heuristic(start, goal)}
        
        heapq.heappush(open_set, (f_score[start], start))
        
        iterations = 0
        while open_set and iterations < max_iterations:
            iterations += 1
            
            current_f, current = heapq.heappop(open_set)
            
            if current == goal or self.manhattan_distance(current, goal) < 1:
                # Found goal or close enough
                return self.reconstruct_path(came_from, current)
            
            closed_set.add(current)
            
            for neighbor, action in self.get_neighbors(current):
                if neighbor in closed_set:
                    continue
                
                tentative_g = g_score[current] + 1  # Each step costs 1
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = (current, action)
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self.calculate_heuristic(neighbor, goal)
                    
                    # Add to open set if not already there
                    if neighbor not in [item[1] for item in open_set]:
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))
        
        # No path found, return random action
        return [random.randint(0, 6)]
    
    def extract_position_from_observation(self, observation) -> Tuple[int, int]:
        """
        Extract position from game observation
        Simplified position extraction for demonstration
        """
        # In a real implementation, this would analyze the screen pixels
        # For now, we'll use a simplified approach with some randomness
        # to simulate position changes
        
        if hasattr(observation, 'shape') and len(observation.shape) == 3:
            # Use screen data to estimate position
            screen_hash = hash(observation.tobytes()) % 100
            x = (screen_hash % 10) + (self.step_count % 10)
            y = ((screen_hash // 10) % 10) + ((self.step_count // 10) % 10)
            return (x % 20, y % 20)
        else:
            # Fallback position estimation
            return (self.step_count % 20, (self.step_count // 20) % 20)
    
    def detect_scenario(self, observation, reward: float) -> GameScenario:
        """Detect current game scenario"""
        # Check if stuck (repeated positions)
        if len(self.last_positions) >= 5:
            if len(set(self.last_positions)) <= 2:
                return GameScenario.STUCK
        
        # Check for progression (positive reward)
        if reward > 0.1:
            return GameScenario.PROGRESSION
        
        # Default to exploration or navigation
        if self.manhattan_distance(self.current_position, self.goal_position) > 5:
            return GameScenario.EXPLORATION
        else:
            return GameScenario.NAVIGATION
    
    def get_action(self, observation: Dict) -> int:
        """
        PyBoy-compatible action selection method
        """
        reward = observation.get('reward', 0.0)
        return self.select_action(observation, reward)
    
    def select_action(self, observation, reward: float = 0.0) -> int:
        """
        Enhanced A* action selection with professional improvements
        """
        import time
        start_time = time.time()
        
        self.step_count += 1
        
        # Enhanced position extraction from observation
        self.current_position = self.extract_position_from_observation(observation)
        
        # Update tracking with better memory management
        self._update_position_tracking(reward)
        
        # Enhanced scenario detection
        self.scenario = self.detect_scenario_enhanced(observation, reward)
        
        # Smart goal selection based on scenario and game state
        current_goal = self._select_optimal_goal(observation)
        
        # Enhanced A* search with adaptive parameters
        path = self.astar_search_enhanced(self.current_position, current_goal)
        
        # Action selection with fallback strategies
        selected_action = self._select_action_from_path(path, observation)
        
        # Update history with size management
        self.action_history.append(selected_action)
        if len(self.action_history) > 100:
            self.action_history.pop(0)
        
        return selected_action
    
    def _update_position_tracking(self, reward: float):
        """Enhanced position tracking with memory optimization"""
        self.last_positions.append(self.current_position)
        if len(self.last_positions) > 15:
            self.last_positions.popleft()
        
        self.visited_positions.add(self.current_position)
        self.position_counts[self.current_position] = self.position_counts.get(self.current_position, 0) + 1
        
        self.last_rewards.append(reward)
        if len(self.last_rewards) > 10:
            self.last_rewards.popleft()
    
    def detect_scenario_enhanced(self, observation: Dict, reward: float) -> GameScenario:
        """Enhanced scenario detection with multiple criteria"""
        # Check for stuck behavior with multiple indicators
        if self._is_stuck_comprehensive():
            return GameScenario.STUCK
        
        # Check for progression with reward and game state analysis
        if self._is_making_progress(observation, reward):
            return GameScenario.PROGRESSION
        
        # Check if close to known objectives
        if self._is_near_objective():
            return GameScenario.NAVIGATION
        
        # Default to exploration
        return GameScenario.EXPLORATION
    
    def _is_stuck_comprehensive(self) -> bool:
        """Comprehensive stuck detection with multiple criteria"""
        if len(self.last_positions) < 8:
            return False
        
        # Position diversity check
        unique_positions = len(set(self.last_positions))
        position_diversity = unique_positions / len(self.last_positions)
        
        # Action pattern check
        if len(self.action_history) >= 6:
            recent_actions = list(self.action_history)[-6:]
            action_diversity = len(set(recent_actions)) / len(recent_actions)
        else:
            action_diversity = 1.0
        
        # Check for oscillation (back-and-forth movement)
        oscillation_detected = self._detect_oscillation()
        
        return (position_diversity < 0.4 or 
                action_diversity < 0.4 or 
                oscillation_detected)
    
    def _detect_oscillation(self) -> bool:
        """Detect oscillatory movement patterns"""
        if len(self.last_positions) < 6:
            return False
        
        recent_pos = list(self.last_positions)[-6:]
        # Check for A-B-A-B pattern
        for i in range(0, len(recent_pos) - 3, 2):
            if (recent_pos[i] == recent_pos[i+2] and 
                recent_pos[i+1] == recent_pos[i+3] and
                recent_pos[i] != recent_pos[i+1]):
                return True
        
        return False
    
    def _is_making_progress(self, observation: Dict, reward: float) -> bool:
        """Enhanced progress detection"""
        # Reward-based progress
        if reward > 0.1:
            return True
        
        # Game state progress (if available)
        if 'badges' in observation or 'events' in observation:
            badges = np.sum(observation.get('badges', np.zeros(8)))
            events = np.sum(observation.get('events', np.zeros(100)))
            return badges > 0 or events > 5
        
        # Position-based progress (moving towards unexplored areas)
        if len(self.last_positions) >= 5:
            recent_positions = list(self.last_positions)[-5:]
            if len(set(recent_positions)) >= 4:  # Good position diversity
                return True
        
        return False
    
    def _is_near_objective(self) -> bool:
        """Check if near any known objective"""
        for objective in self.known_objectives:
            if self.manhattan_distance(self.current_position, objective) <= 3:
                return True
        return False
    
    def _select_optimal_goal(self, observation: Dict) -> Tuple[int, int]:
        """Smart goal selection based on current state and scenario"""
        if self.scenario == GameScenario.STUCK:
            # Find furthest unvisited position
            return self._find_exploration_goal()
        
        elif self.scenario == GameScenario.PROGRESSION:
            # Continue towards main progression goal
            return self.goal_position
        
        elif self.scenario == GameScenario.NAVIGATION:
            # Find nearest unvisited objective
            unvisited_objectives = [obj for obj in self.known_objectives 
                                  if self.position_counts.get(obj, 0) < 3]
            if unvisited_objectives:
                return min(unvisited_objectives,
                          key=lambda pos: self.manhattan_distance(self.current_position, pos))
        
        # Default goal selection
        return self._adaptive_goal_selection()
    
    def _find_exploration_goal(self) -> Tuple[int, int]:
        """Find optimal exploration target"""
        # Create exploration candidates
        candidates = []
        for x in range(max(0, self.current_position[0] - 8), 
                      min(21, self.current_position[0] + 9)):
            for y in range(max(0, self.current_position[1] - 8),
                          min(21, self.current_position[1] + 9)):
                if (x, y) not in self.visited_positions:
                    candidates.append((x, y))
        
        if candidates:
            # Choose candidate that balances distance and exploration value
            def exploration_score(pos):
                distance = self.manhattan_distance(self.current_position, pos)
                visit_count = self.position_counts.get(pos, 0)
                return -distance - visit_count * 2  # Prefer closer, less visited
            
            return max(candidates, key=exploration_score)
        
        return self.goal_position
    
    def _adaptive_goal_selection(self) -> Tuple[int, int]:
        """Adaptive goal selection based on exploration progress"""
        exploration_ratio = len(self.visited_positions) / max(self.step_count / 10, 1)
        
        if exploration_ratio < 0.3:  # Need more exploration
            return self._find_exploration_goal()
        else:  # Focus on main objective
            return self.goal_position
    
    def astar_search_enhanced(self, start: Tuple[int, int], goal: Tuple[int, int]) -> List[int]:
        """Enhanced A* search with adaptive parameters"""
        # Adjust max iterations based on distance and scenario
        base_iterations = 50
        distance_factor = min(3, self.manhattan_distance(start, goal) / 5)
        scenario_factor = 2 if self.scenario == GameScenario.STUCK else 1
        max_iterations = int(base_iterations * distance_factor * scenario_factor)
        
        return self.astar_search(start, goal, max_iterations)
    
    def _select_action_from_path(self, path: List[int], observation: Dict) -> int:
        """Select action from path with intelligent fallbacks"""
        if path and len(path) > 0:
            planned_action = path[0]
            
            # Validate action against recent history to avoid loops
            if self._is_action_safe(planned_action):
                return planned_action
        
        # Fallback strategies
        return self._select_fallback_action(observation)
    
    def _is_action_safe(self, action: int) -> bool:
        """Check if action is safe to avoid immediate cycles"""
        if len(self.action_history) < 4:
            return True
        
        # Check for immediate action repetition
        recent_actions = list(self.action_history)[-4:]
        if recent_actions.count(action) >= 3:
            return False
        
        # Check for movement-based oscillation
        if action in [0, 1, 2, 3]:  # Movement actions
            opposite_actions = {0: 1, 1: 0, 2: 3, 3: 2}
            opposite = opposite_actions.get(action)
            if (len(self.action_history) >= 2 and 
                list(self.action_history)[-1] == opposite and 
                list(self.action_history)[-2] == action):
                return False
        
        return True
    
    def _select_fallback_action(self, observation: Dict) -> int:
        """Intelligent fallback action selection"""
        # Strategy 1: Smart random movement avoiding recent actions
        recent_actions = list(self.action_history)[-5:] if len(self.action_history) >= 5 else []
        movement_actions = [0, 1, 2, 3]  # UP, DOWN, LEFT, RIGHT
        
        # Prefer movement actions not recently used
        available_movements = [a for a in movement_actions if recent_actions.count(a) < 2]
        
        if available_movements:
            return random.choice(available_movements)
        
        # Strategy 2: Interaction action if stuck
        if self.scenario == GameScenario.STUCK:
            return 4  # A button for interaction
        
        # Strategy 3: Completely random
        return random.randint(0, 6)
    
    def get_agent_info(self) -> Dict:
        """Return information about the agent's current state"""
        return {
            "algorithm": "A*",
            "current_position": self.current_position,
            "goal_position": self.goal_position,
            "scenario": self.scenario.value,
            "visited_positions": len(self.visited_positions),
            "step_count": self.step_count,
            "recent_actions": list(self.action_history)[-5:],
            "position_repeats": self.position_counts.get(self.current_position, 0)
        }
    
    def reset(self):
        """Reset agent state for new episode"""
        self.current_position = (0, 0)
        self.visited_positions.clear()
        self.position_counts.clear()
        self.last_positions.clear()
        self.action_history.clear()
        self.last_rewards.clear()
        self.step_count = 0
        self.stuck_counter = 0
        
        # Clear A* specific data structures
        self.open_set.clear()
        self.closed_set.clear()
        self.came_from.clear()
        self.g_score.clear()
        self.f_score.clear()
        
        print("A* Agent reset for new episode")
