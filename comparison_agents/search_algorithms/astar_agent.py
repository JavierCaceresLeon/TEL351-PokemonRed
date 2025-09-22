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
    
    def select_action(self, observation, reward: float = 0.0) -> int:
        """
        Select next action using A* search
        """
        self.step_count += 1
        
        # Extract current position from observation
        self.current_position = self.extract_position_from_observation(observation)
        
        # Update tracking
        self.last_positions.append(self.current_position)
        self.visited_positions.add(self.current_position)
        self.position_counts[self.current_position] = self.position_counts.get(self.current_position, 0) + 1
        self.last_rewards.append(reward)
        
        # Detect current scenario
        self.scenario = self.detect_scenario(observation, reward)
        
        # Choose goal based on scenario
        if self.scenario == GameScenario.STUCK:
            # If stuck, try to move to an unvisited area
            unvisited_goals = [(x, y) for x in range(20) for y in range(20) 
                             if (x, y) not in self.visited_positions]
            if unvisited_goals:
                current_goal = min(unvisited_goals, 
                                 key=lambda pos: self.manhattan_distance(self.current_position, pos))
            else:
                current_goal = self.goal_position
        elif self.scenario == GameScenario.PROGRESSION:
            # If making progress, continue toward main goal
            current_goal = self.goal_position
        else:
            # Default goal selection
            if self.step_count < 100:
                # Early game: explore known objectives
                current_goal = min(self.known_objectives,
                                 key=lambda pos: self.manhattan_distance(self.current_position, pos))
            else:
                # Late game: focus on main goal
                current_goal = self.goal_position
        
        # Perform A* search to find path
        path = self.astar_search(self.current_position, current_goal, max_iterations=50)
        
        if path:
            selected_action = path[0]
        else:
            # Fallback: random action if no path found
            selected_action = random.randint(0, 6)
        
        # Update action history
        self.action_history.append(selected_action)
        
        return selected_action
    
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
