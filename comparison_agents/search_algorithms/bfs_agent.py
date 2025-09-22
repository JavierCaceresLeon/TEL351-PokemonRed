"""
Breadth-First Search Agent for Pokemon Red Environment
=====================================================

This agent implements the BFS search algorithm for systematic exploration
of the Pokemon Red environment.

Key Features:
- BFS for complete state space exploration
- Level-by-level exploration
- Optimal path finding for unweighted graphs
- Memory of explored states
- Goal-directed behavior with systematic backup
"""

import numpy as np
from collections import deque
import json
from typing import Dict, Tuple, List, Optional, Set
from dataclasses import dataclass
from enum import Enum
import random
import time


class GameScenario(Enum):
    """Different scenarios the agent can encounter"""
    EXPLORATION = "exploration"
    NAVIGATION = "navigation" 
    PROGRESSION = "progression"
    STUCK = "stuck"


@dataclass
class BFSNode:
    """Node for BFS search"""
    position: Tuple[int, int]
    depth: int
    path: List[int]  # Actions taken to reach this node
    parent: Optional['BFSNode'] = None


class BFSAgent:
    """
    Breadth-First Search Agent for Pokemon Red Environment
    
    This agent uses BFS to systematically explore the game state space
    and find optimal paths to objectives.
    """
    
    def __init__(self):
        """Initialize BFS agent with default configuration"""
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
        
        # BFS specific data structures
        self.queue = deque()  # BFS queue
        self.visited = set()  # Visited states
        self.level_count = {}  # States at each level
        self.current_path = []  # Current path being executed
        self.path_index = 0   # Current position in path
        
        # Game state tracking
        self.current_position = (0, 0)
        self.goal_position = (5, 3)  # Pokemon selection area
        self.visited_positions = set()
        self.position_counts = {}
        self.last_positions = deque(maxlen=15)
        
        # Search parameters
        self.max_depth = 20
        self.exploration_radius = 3
        self.search_complete = False
        
        # Action tracking
        self.action_history = deque(maxlen=25)
        self.last_rewards = deque(maxlen=5)
        self.step_count = 0
        self.stuck_counter = 0
        
        # Performance metrics
        self.nodes_explored = 0
        self.search_time = 0
        self.paths_found = 0
        
        # Game objectives
        self.objectives = [
            (5, 3),   # Pokemon selection
            (3, 2),   # Professor Oak
            (4, 6),   # House exit
            (5, 8),   # Lab entrance
            (6, 4),   # Alternative position
        ]
        
        print("BFS Agent initialized with systematic exploration")
    
    def get_neighbors(self, position: Tuple[int, int]) -> List[Tuple[Tuple[int, int], int]]:
        """Get valid neighboring positions and corresponding actions"""
        neighbors = []
        
        # Movement actions
        moves = [
            ((-1, 0), 0),  # UP
            ((1, 0), 1),   # DOWN
            ((0, -1), 2),  # LEFT
            ((0, 1), 3),   # RIGHT
        ]
        
        for (dx, dy), action in moves:
            new_pos = (position[0] + dx, position[1] + dy)
            # Basic bounds checking
            if (0 <= new_pos[0] <= 25 and 0 <= new_pos[1] <= 25):
                neighbors.append((new_pos, action))
        
        # Interaction actions at current position
        for action in [4, 5, 6]:  # A, B, START
            neighbors.append((position, action))
        
        return neighbors
    
    def bfs_search(self, start: Tuple[int, int], goal: Tuple[int, int], max_nodes: int = 200) -> List[int]:
        """
        Perform BFS search from start to goal
        Returns list of actions to reach goal
        """
        start_time = time.time()
        
        if start == goal:
            return []
        
        queue = deque([BFSNode(start, 0, [])])
        visited = {start}
        nodes_explored = 0
        
        while queue and nodes_explored < max_nodes:
            current_node = queue.popleft()
            nodes_explored += 1
            
            # Check if we've reached the goal or are very close
            if (current_node.position == goal or 
                abs(current_node.position[0] - goal[0]) + abs(current_node.position[1] - goal[1]) <= 1):
                
                self.nodes_explored += nodes_explored
                self.search_time += time.time() - start_time
                self.paths_found += 1
                return current_node.path
            
            # Explore neighbors
            for neighbor_pos, action in self.get_neighbors(current_node.position):
                if neighbor_pos not in visited and current_node.depth < self.max_depth:
                    visited.add(neighbor_pos)
                    new_path = current_node.path + [action]
                    new_node = BFSNode(neighbor_pos, current_node.depth + 1, new_path, current_node)
                    queue.append(new_node)
        
        self.nodes_explored += nodes_explored
        self.search_time += time.time() - start_time
        
        # No path found, return empty list
        return []
    
    def extract_position_from_observation(self, observation) -> Tuple[int, int]:
        """
        Extract position from game observation
        Enhanced position extraction with BFS-specific logic
        """
        if hasattr(observation, 'shape') and len(observation.shape) == 3:
            # Use screen content for position estimation
            screen_sum = np.sum(observation)
            screen_hash = hash(observation.tobytes()) % 1000
            
            # More stable position calculation
            x = (screen_hash % 25) + (self.step_count % 3)
            y = ((screen_hash // 25) % 25) + ((self.step_count // 3) % 3)
            
            # Add some deterministic variation based on action history
            if len(self.action_history) > 0:
                last_action = self.action_history[-1]
                if last_action == 0:  # UP
                    y = max(0, y - 1)
                elif last_action == 1:  # DOWN
                    y = min(24, y + 1)
                elif last_action == 2:  # LEFT
                    x = max(0, x - 1)
                elif last_action == 3:  # RIGHT
                    x = min(24, x + 1)
            
            return (x % 25, y % 25)
        else:
            # Fallback with BFS-specific movement pattern
            return ((self.step_count % 25), ((self.step_count // 25) % 25))
    
    def detect_scenario(self, observation, reward: float) -> GameScenario:
        """Detect current game scenario for BFS strategy"""
        # Check if stuck (repeated positions)
        if len(self.last_positions) >= 8:
            unique_positions = len(set(self.last_positions))
            if unique_positions <= 2:
                return GameScenario.STUCK
        
        # Check for progression
        if reward > 0.05:
            return GameScenario.PROGRESSION
        
        # Determine if we should explore or navigate
        distance_to_goal = abs(self.current_position[0] - self.goal_position[0]) + abs(self.current_position[1] - self.goal_position[1])
        
        if distance_to_goal > 8:
            return GameScenario.EXPLORATION
        else:
            return GameScenario.NAVIGATION
    
    def select_best_objective(self) -> Tuple[int, int]:
        """Select best objective based on BFS analysis"""
        # Calculate distances to all objectives
        objective_scores = []
        
        for obj in self.objectives:
            distance = abs(self.current_position[0] - obj[0]) + abs(self.current_position[1] - obj[1])
            visit_count = self.position_counts.get(obj, 0)
            
            # Score: lower distance is better, higher visit count is worse
            score = distance + (visit_count * 2)
            objective_scores.append((score, obj))
        
        # Return objective with lowest score
        objective_scores.sort()
        return objective_scores[0][1]
    
    def execute_systematic_exploration(self) -> int:
        """Execute systematic BFS-based exploration"""
        # Define exploration pattern based on current position
        x, y = self.current_position
        
        # BFS-style systematic movement
        if self.step_count % 4 == 0:
            return 3  # RIGHT
        elif self.step_count % 4 == 1:
            return 1  # DOWN
        elif self.step_count % 4 == 2:
            return 2  # LEFT
        else:
            return 0  # UP
    
    def select_action(self, observation, reward: float = 0.0) -> int:
        """
        Select next action using BFS strategy
        """
        self.step_count += 1
        
        # Extract current position
        self.current_position = self.extract_position_from_observation(observation)
        
        # Update tracking
        self.last_positions.append(self.current_position)
        self.visited_positions.add(self.current_position)
        self.position_counts[self.current_position] = self.position_counts.get(self.current_position, 0) + 1
        self.last_rewards.append(reward)
        
        # Detect scenario
        scenario = self.detect_scenario(observation, reward)
        
        # Select action based on scenario
        if scenario == GameScenario.STUCK:
            # If stuck, use systematic exploration
            selected_action = self.execute_systematic_exploration()
            self.stuck_counter += 1
            
        elif scenario == GameScenario.PROGRESSION:
            # If making progress, continue with current strategy
            if len(self.current_path) > self.path_index:
                selected_action = self.current_path[self.path_index]
                self.path_index += 1
            else:
                # Generate new path to goal
                goal = self.select_best_objective()
                self.current_path = self.bfs_search(self.current_position, goal, max_nodes=150)
                self.path_index = 0
                
                if self.current_path and len(self.current_path) > 0:
                    selected_action = self.current_path[0]
                    self.path_index = 1
                else:
                    selected_action = random.randint(0, 6)
        
        else:
            # Default BFS behavior
            if len(self.current_path) > self.path_index:
                # Continue executing current path
                selected_action = self.current_path[self.path_index]
                self.path_index += 1
            else:
                # Generate new path
                if scenario == GameScenario.EXPLORATION:
                    # Explore systematically
                    unvisited_positions = [(x, y) for x in range(25) for y in range(25) 
                                         if (x, y) not in self.visited_positions]
                    
                    if unvisited_positions and len(unvisited_positions) > 5:
                        # Choose closest unvisited position
                        goal = min(unvisited_positions, 
                                 key=lambda pos: abs(pos[0] - self.current_position[0]) + 
                                                 abs(pos[1] - self.current_position[1]))
                    else:
                        goal = self.select_best_objective()
                else:
                    # Navigate to best objective
                    goal = self.select_best_objective()
                
                # Perform BFS search
                self.current_path = self.bfs_search(self.current_position, goal, max_nodes=120)
                self.path_index = 0
                
                if self.current_path and len(self.current_path) > 0:
                    selected_action = self.current_path[0]
                    self.path_index = 1
                else:
                    # Fallback to systematic exploration
                    selected_action = self.execute_systematic_exploration()
        
        # Update action history
        self.action_history.append(selected_action)
        
        return selected_action
    
    def get_agent_info(self) -> Dict:
        """Return information about the agent's current state"""
        return {
            "algorithm": "BFS",
            "current_position": self.current_position,
            "goal_position": self.goal_position,
            "visited_positions": len(self.visited_positions),
            "step_count": self.step_count,
            "nodes_explored": self.nodes_explored,
            "paths_found": self.paths_found,
            "search_time": round(self.search_time, 3),
            "current_path_length": len(self.current_path),
            "path_progress": f"{self.path_index}/{len(self.current_path)}",
            "stuck_counter": self.stuck_counter,
            "recent_actions": list(self.action_history)[-5:]
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
        
        # Reset BFS specific structures
        self.queue.clear()
        self.visited.clear()
        self.level_count.clear()
        self.current_path.clear()
        self.path_index = 0
        self.search_complete = False
        
        # Reset performance metrics
        self.nodes_explored = 0
        self.search_time = 0
        self.paths_found = 0
        
        print("BFS Agent reset for new episode")