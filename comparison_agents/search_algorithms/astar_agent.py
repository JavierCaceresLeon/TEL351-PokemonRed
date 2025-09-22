"""
A* Search Agent for Pokemon Red
Enhanced A* implementation with intelligent heuristics and goal-directed behavior.
Much more effective than epsilon greedy with proper pathfinding and memory.
"""

import heapq
import random
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from collections import deque, defaultdict
from enum import Enum
import math

class GameObjective(Enum):
    """Different objectives the agent can pursue"""
    EXPLORE_MAP = "explore_map"
    FIND_POKEMON = "find_pokemon"
    LEVEL_UP = "level_up"
    FIND_ITEMS = "find_items"
    PROGRESS_STORY = "progress_story"
    ESCAPE_STUCK = "escape_stuck"

class AStarNode:
    """Node for A* pathfinding"""
    def __init__(self, position: Tuple[int, int], g_cost: float = 0, h_cost: float = 0, parent=None, action: int = None):
        self.position = position
        self.g_cost = g_cost  # Cost from start
        self.h_cost = h_cost  # Heuristic cost to goal
        self.f_cost = g_cost + h_cost  # Total cost
        self.parent = parent
        self.action = action  # Action that led to this node
    
    def __lt__(self, other):
        return self.f_cost < other.f_cost
    
    def __eq__(self, other):
        return self.position == other.position
    
    def __hash__(self):
        return hash(self.position)

class PokemonAStarAgent:
    """
    Advanced A* agent for Pokemon Red with intelligent goal selection and pathfinding.
    Significantly more effective than epsilon greedy with proper planning.
    """
    
    def __init__(self, exploration_bonus: float = 1.5, goal_persistence: int = 100):
        # Core A* parameters
        self.exploration_bonus = exploration_bonus
        self.goal_persistence = goal_persistence
        
        # Game state tracking
        self.visited_positions = set()
        self.position_rewards = defaultdict(float)
        self.position_visit_count = defaultdict(int)
        self.last_positions = deque(maxlen=20)
        
        # Current objective and planning
        self.current_objective = GameObjective.EXPLORE_MAP
        self.current_goal_position = None
        self.steps_on_current_goal = 0
        self.planned_path = deque()
        
        # Memory and learning
        self.successful_paths = {}  # position -> (action, reward)
        self.failed_positions = set()
        self.high_reward_areas = set()
        self.pokemon_encounters = set()
        
        # Action control
        self.valid_actions = [0, 1, 2, 3, 4, 5, 6]  # UP, DOWN, LEFT, RIGHT, A, B, START
        self.movement_actions = [0, 1, 2, 3]  # Movement only
        self.interaction_actions = [4, 5]  # A, B
        
        # Anti-stuck mechanisms
        self.stuck_counter = 0
        self.last_reward = 0
        self.steps_without_progress = 0
        self.emergency_exploration_mode = False
        
        # Performance tracking
        self.step_count = 0
        self.total_reward = 0
        self.decisions_made = 0
        self.successful_explorations = 0
        
    def select_action(self, observation, game_state: Dict) -> Tuple[int, Dict]:
        """
        Select action using A* pathfinding with intelligent goal selection
        """
        self.step_count += 1
        self.decisions_made += 1
        
        # Get current position
        current_pos = (game_state.get('x', 0), game_state.get('y', 0))
        self.last_positions.append(current_pos)
        self.visited_positions.add(current_pos)
        self.position_visit_count[current_pos] += 1
        
        # Detect if stuck
        stuck = self.detect_stuck_situation(current_pos, game_state)
        
        # Update objective based on game state
        self.update_objective(game_state, stuck)
        
        # If we have a planned path and not stuck, follow it
        if self.planned_path and not stuck and not self.emergency_exploration_mode:
            planned_action = self.planned_path.popleft()
            action_quality = self.evaluate_action_quality(planned_action, current_pos, game_state)
            
            decision_info = {
                'agent_type': 'astar',
                'objective': self.current_objective.value,
                'action_source': 'planned_path',
                'planned_path_length': len(self.planned_path),
                'action_quality': action_quality,
                'stuck_counter': self.stuck_counter,
                'visited_positions': len(self.visited_positions),
                'steps_without_progress': self.steps_without_progress
            }
            
            return planned_action, decision_info
        
        # Clear planned path if stuck or in emergency mode
        self.planned_path.clear()
        
        # Select goal and plan path
        goal_position = self.select_goal(current_pos, game_state)
        
        if goal_position:
            # Plan path to goal using A*
            path = self.plan_path_astar(current_pos, goal_position, game_state)
            if path and len(path) > 1:
                self.planned_path = deque(path[1:])  # Exclude current position
                next_action = self.planned_path.popleft() if self.planned_path else self.get_best_local_action(current_pos, game_state)
            else:
                next_action = self.get_best_local_action(current_pos, game_state)
        else:
            # No clear goal, use local optimization
            next_action = self.get_best_local_action(current_pos, game_state)
        
        # Evaluate action quality
        action_quality = self.evaluate_action_quality(next_action, current_pos, game_state)
        
        # Prepare detailed decision info
        decision_info = {
            'agent_type': 'astar',
            'objective': self.current_objective.value,
            'action_source': 'astar_planning' if goal_position else 'local_optimization',
            'goal_position': goal_position,
            'planned_path_length': len(self.planned_path),
            'action_quality': action_quality,
            'stuck_counter': self.stuck_counter,
            'visited_positions': len(self.visited_positions),
            'steps_without_progress': self.steps_without_progress,
            'emergency_mode': self.emergency_exploration_mode,
            'high_reward_areas': len(self.high_reward_areas)
        }
        
        return next_action, decision_info
    
    def detect_stuck_situation(self, current_pos: Tuple[int, int], game_state: Dict) -> bool:
        """Detect if the agent is stuck in a loop or unproductive area"""
        # Check for position loops
        if len(self.last_positions) >= 10:
            recent_positions = list(self.last_positions)[-10:]
            unique_positions = set(recent_positions)
            if len(unique_positions) <= 3:  # Only 3 or fewer unique positions in last 10 steps
                self.stuck_counter += 1
                return True
        
        # Check for repeated visits to same position
        if self.position_visit_count[current_pos] > 15:
            self.stuck_counter += 1
            return True
        
        # Check for lack of progress
        if self.steps_without_progress > 50:
            self.stuck_counter += 1
            return True
        
        # Reset stuck counter if not stuck
        if self.stuck_counter > 0:
            self.stuck_counter = max(0, self.stuck_counter - 1)
        
        return self.stuck_counter > 5
    
    def update_objective(self, game_state: Dict, stuck: bool):
        """Update current objective based on game state and progress"""
        self.steps_on_current_goal += 1
        
        # Emergency exploration if stuck
        if stuck or self.stuck_counter > 10:
            self.current_objective = GameObjective.ESCAPE_STUCK
            self.emergency_exploration_mode = True
            self.steps_on_current_goal = 0
            return
        
        # Reset emergency mode if not stuck
        if not stuck and self.emergency_exploration_mode:
            self.emergency_exploration_mode = False
        
        # Switch objectives based on game state and time spent
        if game_state.get('battle', False):
            self.current_objective = GameObjective.FIND_POKEMON
        elif self.steps_on_current_goal > self.goal_persistence:
            # Rotate objectives
            objectives = [GameObjective.EXPLORE_MAP, GameObjective.FIND_POKEMON, GameObjective.PROGRESS_STORY]
            current_idx = objectives.index(self.current_objective) if self.current_objective in objectives else 0
            self.current_objective = objectives[(current_idx + 1) % len(objectives)]
            self.steps_on_current_goal = 0
    
    def select_goal(self, current_pos: Tuple[int, int], game_state: Dict) -> Optional[Tuple[int, int]]:
        """Select goal position based on current objective"""
        if self.current_objective == GameObjective.ESCAPE_STUCK or self.emergency_exploration_mode:
            # Find farthest unvisited position
            return self.find_escape_position(current_pos)
        
        elif self.current_objective == GameObjective.EXPLORE_MAP:
            # Find interesting unexplored area
            return self.find_exploration_target(current_pos)
        
        elif self.current_objective == GameObjective.FIND_POKEMON:
            # Go to areas where we've had encounters
            return self.find_pokemon_area(current_pos)
        
        elif self.current_objective == GameObjective.PROGRESS_STORY:
            # Find areas with potential progression
            return self.find_progression_target(current_pos)
        
        return None
    
    def find_escape_position(self, current_pos: Tuple[int, int]) -> Tuple[int, int]:
        """Find position to escape stuck situation"""
        # Find the position we've visited least recently and is far away
        min_visits = float('inf')
        best_position = None
        
        for pos in self.visited_positions:
            visits = self.position_visit_count[pos]
            distance = self.manhattan_distance(current_pos, pos)
            
            # Prefer far positions with few visits
            score = visits - distance * 0.1
            
            if score < min_visits:
                min_visits = score
                best_position = pos
        
        # If no good visited position, generate a random far position
        if not best_position:
            best_position = (
                current_pos[0] + random.randint(-10, 10),
                current_pos[1] + random.randint(-10, 10)
            )
        
        return best_position
    
    def find_exploration_target(self, current_pos: Tuple[int, int]) -> Tuple[int, int]:
        """Find good position for exploration"""
        # Generate potential exploration targets around current position
        candidates = []
        
        for dx in range(-8, 9, 2):
            for dy in range(-8, 9, 2):
                candidate_pos = (current_pos[0] + dx, current_pos[1] + dy)
                
                if candidate_pos not in self.visited_positions:
                    distance = abs(dx) + abs(dy)
                    # Prefer positions at medium distance
                    if 3 <= distance <= 8:
                        candidates.append(candidate_pos)
        
        if candidates:
            return random.choice(candidates)
        
        # Fallback: find least visited area
        min_visits = float('inf')
        best_pos = None
        
        for pos in self.visited_positions:
            visits = self.position_visit_count[pos]
            if visits < min_visits:
                min_visits = visits
                best_pos = pos
        
        return best_pos
    
    def find_pokemon_area(self, current_pos: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        """Find area likely to have Pokemon encounters"""
        if self.pokemon_encounters:
            # Go to a previous encounter location
            return random.choice(list(self.pokemon_encounters))
        
        # Explore grass areas (heuristic: positions we haven't fully explored)
        return self.find_exploration_target(current_pos)
    
    def find_progression_target(self, current_pos: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        """Find area that might lead to story progression"""
        if self.high_reward_areas:
            # Go to areas where we got high rewards
            return random.choice(list(self.high_reward_areas))
        
        # Default to exploration
        return self.find_exploration_target(current_pos)
    
    def plan_path_astar(self, start: Tuple[int, int], goal: Tuple[int, int], game_state: Dict) -> List[int]:
        """Plan path from start to goal using A* algorithm"""
        if start == goal:
            return []
        
        open_set = []
        closed_set = set()
        
        start_node = AStarNode(start, 0, self.heuristic(start, goal))
        heapq.heappush(open_set, start_node)
        
        nodes = {start: start_node}
        
        max_iterations = 100  # Prevent infinite loops
        iterations = 0
        
        while open_set and iterations < max_iterations:
            iterations += 1
            current = heapq.heappop(open_set)
            
            if current.position == goal:
                # Reconstruct path
                path = []
                while current.parent:
                    path.append(current.action)
                    current = current.parent
                path.reverse()
                return path
            
            closed_set.add(current.position)
            
            # Explore neighbors
            for action in self.movement_actions:
                neighbor_pos = self.get_next_position(current.position, action)
                
                if neighbor_pos in closed_set:
                    continue
                
                # Calculate costs
                movement_cost = 1.0
                
                # Add penalties for visited positions
                if neighbor_pos in self.position_visit_count:
                    movement_cost += self.position_visit_count[neighbor_pos] * 0.1
                
                # Add penalties for failed positions
                if neighbor_pos in self.failed_positions:
                    movement_cost += 2.0
                
                g_cost = current.g_cost + movement_cost
                h_cost = self.heuristic(neighbor_pos, goal)
                
                if neighbor_pos in nodes:
                    existing_node = nodes[neighbor_pos]
                    if g_cost < existing_node.g_cost:
                        existing_node.g_cost = g_cost
                        existing_node.f_cost = g_cost + h_cost
                        existing_node.parent = current
                        existing_node.action = action
                else:
                    new_node = AStarNode(neighbor_pos, g_cost, h_cost, current, action)
                    nodes[neighbor_pos] = new_node
                    heapq.heappush(open_set, new_node)
        
        # No path found, return empty
        return []
    
    def get_next_position(self, pos: Tuple[int, int], action: int) -> Tuple[int, int]:
        """Get next position after taking action"""
        x, y = pos
        if action == 0:  # UP
            return (x, y - 1)
        elif action == 1:  # DOWN
            return (x, y + 1)
        elif action == 2:  # LEFT
            return (x - 1, y)
        elif action == 3:  # RIGHT
            return (x + 1, y)
        return pos  # Non-movement action
    
    def heuristic(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """Calculate heuristic distance between positions"""
        base_distance = self.manhattan_distance(pos1, pos2)
        
        # Add bonus for unexplored areas
        if pos1 not in self.visited_positions:
            base_distance *= 0.8  # Make unexplored areas more attractive
        
        # Add penalty for highly visited areas
        visit_penalty = self.position_visit_count.get(pos1, 0) * 0.1
        
        return base_distance + visit_penalty
    
    def manhattan_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """Calculate Manhattan distance between two positions"""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    def get_best_local_action(self, current_pos: Tuple[int, int], game_state: Dict) -> int:
        """Get best action using local optimization when no clear path"""
        action_scores = {}
        
        for action in self.valid_actions:
            score = self.evaluate_action_quality(action, current_pos, game_state)
            action_scores[action] = score
        
        # Choose best action
        best_action = max(action_scores, key=action_scores.get)
        
        # Add some randomness to prevent getting stuck in local optima
        if random.random() < 0.1:  # 10% chance
            return random.choice(self.valid_actions)
        
        return best_action
    
    def evaluate_action_quality(self, action: int, current_pos: Tuple[int, int], game_state: Dict) -> float:
        """Evaluate quality of an action"""
        base_quality = 0.5
        
        # Movement actions
        if action in self.movement_actions:
            next_pos = self.get_next_position(current_pos, action)
            
            # Bonus for unexplored positions
            if next_pos not in self.visited_positions:
                base_quality += 1.0
            
            # Penalty for highly visited positions
            visit_count = self.position_visit_count.get(next_pos, 0)
            base_quality -= visit_count * 0.1
            
            # Bonus for distance from current area
            avg_x = sum(pos[0] for pos in self.last_positions) / len(self.last_positions) if self.last_positions else current_pos[0]
            avg_y = sum(pos[1] for pos in self.last_positions) / len(self.last_positions) if self.last_positions else current_pos[1]
            distance_from_center = abs(next_pos[0] - avg_x) + abs(next_pos[1] - avg_y)
            base_quality += min(distance_from_center * 0.1, 0.5)
            
            # Penalty for failed positions
            if next_pos in self.failed_positions:
                base_quality -= 0.5
        
        # Interaction actions
        elif action in self.interaction_actions:
            # Bonus for interaction in unexplored areas
            if current_pos not in self.visited_positions or self.position_visit_count[current_pos] <= 2:
                base_quality += 0.3
            
            # Bonus if we're in a potential high-reward area
            if current_pos in self.high_reward_areas:
                base_quality += 0.4
        
        # Menu action (START) - be very conservative
        elif action == 6:
            # Only use START sparingly and when not stuck
            if self.stuck_counter == 0 and random.random() < 0.05:  # 5% chance
                base_quality += 0.1
            else:
                base_quality -= 0.8  # Strong penalty
        
        return base_quality
    
    def update_performance(self, action: int, reward: float, observation, game_state: Dict):
        """Update agent's performance tracking and learning"""
        self.total_reward += reward
        current_pos = (game_state.get('x', 0), game_state.get('y', 0))
        
        # Update position rewards
        self.position_rewards[current_pos] += reward
        
        # Track high reward areas
        if reward > 1.0:
            self.high_reward_areas.add(current_pos)
            self.steps_without_progress = 0
        else:
            self.steps_without_progress += 1
        
        # Track Pokemon encounters
        if game_state.get('battle', False):
            self.pokemon_encounters.add(current_pos)
        
        # Track failed positions (areas with negative rewards)
        if reward < -0.1:
            self.failed_positions.add(current_pos)
        
        # Update successful paths
        if reward > 0.5:
            self.successful_paths[current_pos] = (action, reward)
        
        self.last_reward = reward
    
    def reset(self):
        """Reset agent for new episode"""
        self.planned_path.clear()
        self.last_positions.clear()
        self.stuck_counter = 0
        self.steps_without_progress = 0
        self.emergency_exploration_mode = False
        self.current_objective = GameObjective.EXPLORE_MAP
        self.steps_on_current_goal = 0
        self.step_count = 0
        
        # Keep learned knowledge between episodes
        # self.visited_positions.clear()  # Keep this for learning
        # self.successful_paths.clear()   # Keep this for learning
