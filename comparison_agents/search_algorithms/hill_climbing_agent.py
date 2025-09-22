"""
Hill Climbing Agent for Pokemon Red Environment
==============================================

This agent implements the Hill Climbing algorithm for local optimization
in the Pokemon Red environment.

Key Features:
- Greedy local search
- Multiple restart capability
- Adaptive step size
- Plateau escape mechanisms
- Best-improvement and first-improvement variants
"""

import numpy as np
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
    PLATEAU = "plateau"


class HillClimbingVariant(Enum):
    """Different Hill Climbing variants"""
    STEEPEST_ASCENT = "steepest_ascent"
    FIRST_IMPROVEMENT = "first_improvement"
    RANDOM_RESTART = "random_restart"
    STOCHASTIC = "stochastic"


@dataclass
class HCState:
    """State representation for Hill Climbing"""
    position: Tuple[int, int]
    fitness: float
    actions_taken: List[int]
    evaluations: int = 0


class HillClimbingAgent:
    """
    Hill Climbing Agent for Pokemon Red Environment
    
    This agent uses hill climbing search to find local optima in the
    action space, with mechanisms to escape local maxima.
    """
    
    def __init__(self, variant: HillClimbingVariant = HillClimbingVariant.STEEPEST_ASCENT):
        """Initialize Hill Climbing agent"""
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
        
        # Hill Climbing configuration
        self.variant = variant
        self.current_state = HCState((0, 0), -float('inf'), [])
        self.best_state = HCState((0, 0), -float('inf'), [])
        self.plateau_threshold = 15  # Steps without improvement
        self.restart_threshold = 50  # Steps before random restart
        
        # Game state tracking
        self.current_position = (0, 0)
        self.goal_position = (5, 3)  # Pokemon selection area
        self.visited_positions = set()
        self.position_counts = {}
        self.last_positions = deque(maxlen=10)
        
        # Fitness calculation components
        self.distance_weight = -1.0  # Negative because we want to minimize distance
        self.exploration_bonus = 2.0
        self.progress_multiplier = 5.0
        self.repetition_penalty = -3.0
        
        # Algorithm state
        self.steps_without_improvement = 0
        self.plateau_escapes = 0
        self.random_restarts = 0
        self.evaluations_count = 0
        
        # Action tracking
        self.action_history = deque(maxlen=25)
        self.last_rewards = deque(maxlen=6)
        self.step_count = 0
        self.local_maxima_detected = False
        
        # Performance tracking
        self.fitness_history = deque(maxlen=50)
        self.best_fitness_found = -float('inf')
        self.improvement_steps = []
        
        # Objectives
        self.objectives = [
            (5, 3),   # Pokemon selection
            (3, 2),   # Professor Oak
            (4, 6),   # House exit
            (5, 8),   # Lab entrance
            (7, 4),   # Alternative position
            (2, 7),   # Exploration target
        ]
        
        # Adaptive parameters
        self.exploration_probability = 0.1
        self.step_size = 1
        self.momentum = 0.0
        self.last_improvement_direction = None
        
        print(f"Hill Climbing Agent initialized with {variant.value} strategy")
    
    def calculate_fitness(self, position: Tuple[int, int], action_sequence: List[int] = None) -> float:
        """
        Calculate fitness of a state
        Higher fitness is better (maximization problem)
        """
        fitness = 0.0
        
        # Distance to closest objective (negative distance for maximization)
        min_distance = min(abs(position[0] - obj[0]) + abs(position[1] - obj[1]) 
                          for obj in self.objectives)
        fitness += min_distance * self.distance_weight
        
        # Exploration bonus for visiting new positions
        if position not in self.visited_positions:
            fitness += self.exploration_bonus
        else:
            # Penalty for revisiting positions too frequently
            visit_count = self.position_counts.get(position, 0)
            if visit_count > 3:
                fitness += self.repetition_penalty * (visit_count - 3)
        
        # Progress bonus based on recent rewards
        if len(self.last_rewards) > 0:
            recent_reward_sum = sum(self.last_rewards)
            if recent_reward_sum > 0:
                fitness += recent_reward_sum * self.progress_multiplier
        
        # Bonus for being close to any objective
        if min_distance <= 2:
            fitness += 10.0  # High bonus for reaching objectives
        elif min_distance <= 5:
            fitness += 5.0   # Medium bonus for being close
        
        # Path efficiency bonus
        if action_sequence and len(action_sequence) > 0:
            # Prefer shorter, more direct paths
            path_efficiency = max(0, 20 - len(action_sequence))
            fitness += path_efficiency * 0.1
            
            # Bonus for diverse action sequences (avoid repetition)
            if len(action_sequence) >= 3:
                unique_actions = len(set(action_sequence[-3:]))
                fitness += unique_actions * 0.5
        
        # Penalize being stuck in small areas
        if len(self.last_positions) >= 6:
            unique_recent = len(set(list(self.last_positions)[-6:]))
            if unique_recent <= 2:
                fitness -= 8.0  # Penalty for being stuck
        
        return fitness
    
    def get_neighbor_actions(self) -> List[int]:
        """Get all possible neighboring actions"""
        return list(range(self.action_space))
    
    def evaluate_action(self, action: int) -> Tuple[Tuple[int, int], float]:
        """
        Evaluate the fitness of taking a specific action
        Returns new position and fitness value
        """
        # Simulate position change based on action
        new_position = self.current_position
        
        if action == 0:  # UP
            new_position = (max(0, self.current_position[0] - 1), self.current_position[1])
        elif action == 1:  # DOWN
            new_position = (min(24, self.current_position[0] + 1), self.current_position[1])
        elif action == 2:  # LEFT
            new_position = (self.current_position[0], max(0, self.current_position[1] - 1))
        elif action == 3:  # RIGHT
            new_position = (self.current_position[0], min(24, self.current_position[1] + 1))
        # For A, B, START actions, position typically stays the same
        
        # Calculate fitness of the new position
        new_action_sequence = list(self.action_history) + [action]
        fitness = self.calculate_fitness(new_position, new_action_sequence)
        
        self.evaluations_count += 1
        return new_position, fitness
    
    def steepest_ascent_selection(self) -> int:
        """Select action using steepest ascent hill climbing"""
        best_action = 0
        best_fitness = -float('inf')
        best_position = self.current_position
        
        # Evaluate all possible actions
        for action in self.get_neighbor_actions():
            position, fitness = self.evaluate_action(action)
            
            if fitness > best_fitness:
                best_fitness = fitness
                best_action = action
                best_position = position
        
        # Check if this is an improvement
        if best_fitness > self.current_state.fitness:
            self.steps_without_improvement = 0
            self.last_improvement_direction = best_action
            self.improvement_steps.append(self.step_count)
        else:
            self.steps_without_improvement += 1
        
        return best_action
    
    def first_improvement_selection(self) -> int:
        """Select action using first improvement hill climbing"""
        # Randomize the order of action evaluation
        actions = self.get_neighbor_actions()
        random.shuffle(actions)
        
        for action in actions:
            position, fitness = self.evaluate_action(action)
            
            # Return first action that improves fitness
            if fitness > self.current_state.fitness:
                self.steps_without_improvement = 0
                self.last_improvement_direction = action
                self.improvement_steps.append(self.step_count)
                return action
        
        # No improvement found
        self.steps_without_improvement += 1
        
        # Return best action among evaluated
        return self.steepest_ascent_selection()
    
    def stochastic_selection(self) -> int:
        """Select action using stochastic hill climbing"""
        # Evaluate all actions
        actions_fitness = []
        for action in self.get_neighbor_actions():
            position, fitness = self.evaluate_action(action)
            actions_fitness.append((action, fitness))
        
        # Sort by fitness (descending)
        actions_fitness.sort(key=lambda x: x[1], reverse=True)
        
        # Use softmax-like selection biased toward better actions
        fitnesses = [f for _, f in actions_fitness]
        
        # Convert to probabilities (with temperature)
        temperature = max(0.1, 1.0 - (self.step_count / 1000))  # Decrease temperature over time
        if max(fitnesses) - min(fitnesses) > 0:
            exp_values = np.exp(np.array(fitnesses) / temperature)
            probabilities = exp_values / np.sum(exp_values)
        else:
            # If all fitnesses are equal, use uniform distribution
            probabilities = np.ones(len(fitnesses)) / len(fitnesses)
        
        # Select action based on probabilities
        selected_idx = np.random.choice(len(actions_fitness), p=probabilities)
        selected_action = actions_fitness[selected_idx][0]
        selected_fitness = actions_fitness[selected_idx][1]
        
        # Track improvement
        if selected_fitness > self.current_state.fitness:
            self.steps_without_improvement = 0
            self.last_improvement_direction = selected_action
            self.improvement_steps.append(self.step_count)
        else:
            self.steps_without_improvement += 1
        
        return selected_action
    
    def random_restart(self) -> int:
        """Perform random restart"""
        self.random_restarts += 1
        self.steps_without_improvement = 0
        self.local_maxima_detected = False
        
        # Reset some tracking to encourage exploration
        if len(self.visited_positions) > 20:
            # Keep some visited positions but reduce their penalty
            positions_to_keep = list(self.visited_positions)[-10:]
            self.visited_positions = set(positions_to_keep)
            self.position_counts = {pos: max(1, count // 2) 
                                   for pos, count in self.position_counts.items() 
                                   if pos in positions_to_keep}
        
        print(f"Random restart #{self.random_restarts} at step {self.step_count}")
        return random.randint(0, 6)
    
    def escape_plateau(self) -> int:
        """Attempt to escape plateau/local maximum"""
        self.plateau_escapes += 1
        
        # Try actions that haven't been used recently
        recent_actions = set(list(self.action_history)[-5:])
        available_actions = [a for a in range(self.action_space) if a not in recent_actions]
        
        if available_actions:
            selected = random.choice(available_actions)
        else:
            # If all actions were used recently, try the opposite of the last action
            if len(self.action_history) > 0:
                last_action = self.action_history[-1]
                if last_action == 0:    # UP -> DOWN
                    selected = 1
                elif last_action == 1:  # DOWN -> UP
                    selected = 0
                elif last_action == 2:  # LEFT -> RIGHT
                    selected = 3
                elif last_action == 3:  # RIGHT -> LEFT
                    selected = 2
                else:
                    selected = random.randint(0, 3)  # Random movement
            else:
                selected = random.randint(0, 6)
        
        print(f"Plateau escape #{self.plateau_escapes} at step {self.step_count}")
        return selected
    
    def extract_position_from_observation(self, observation) -> Tuple[int, int]:
        """Extract position from game observation"""
        if hasattr(observation, 'shape') and len(observation.shape) == 3:
            # Use observation data for position estimation
            obs_hash = hash(observation.tobytes()) % 1500
            
            # Base position calculation
            base_x = obs_hash % 25
            base_y = (obs_hash // 25) % 25
            
            # Apply movement based on action history for consistency
            movement_x = movement_y = 0
            if len(self.action_history) >= 1:
                last_action = self.action_history[-1]
                if last_action == 0:    # UP
                    movement_y -= 1
                elif last_action == 1:  # DOWN
                    movement_y += 1
                elif last_action == 2:  # LEFT
                    movement_x -= 1
                elif last_action == 3:  # RIGHT
                    movement_x += 1
            
            final_x = (base_x + movement_x) % 25
            final_y = (base_y + movement_y) % 25
            
            return (final_x, final_y)
        else:
            # Fallback
            return (self.step_count % 25, (self.step_count // 25) % 25)
    
    def detect_scenario(self, observation, reward: float) -> GameScenario:
        """Detect current game scenario"""
        # Check if we're on a plateau
        if self.steps_without_improvement >= self.plateau_threshold:
            return GameScenario.PLATEAU
        
        # Check if stuck
        if len(self.last_positions) >= 8:
            unique_positions = len(set(self.last_positions))
            if unique_positions <= 2:
                return GameScenario.STUCK
        
        # Check for progression
        if reward > 0.1 or (len(self.fitness_history) > 0 and 
                           self.current_state.fitness > max(self.fitness_history[-5:] if len(self.fitness_history) >= 5 else [0])):
            return GameScenario.PROGRESSION
        
        # Distance-based scenario detection
        min_distance = min(abs(self.current_position[0] - obj[0]) + abs(self.current_position[1] - obj[1]) 
                          for obj in self.objectives)
        
        if min_distance > 8:
            return GameScenario.EXPLORATION
        else:
            return GameScenario.NAVIGATION
    
    def select_action(self, observation, reward: float = 0.0) -> int:
        """
        Select next action using Hill Climbing
        """
        self.step_count += 1
        
        # Extract current position
        self.current_position = self.extract_position_from_observation(observation)
        
        # Update tracking
        self.last_positions.append(self.current_position)
        self.visited_positions.add(self.current_position)
        self.position_counts[self.current_position] = self.position_counts.get(self.current_position, 0) + 1
        self.last_rewards.append(reward)
        
        # Calculate current fitness
        current_fitness = self.calculate_fitness(self.current_position, list(self.action_history))
        self.current_state = HCState(self.current_position, current_fitness, list(self.action_history))
        
        # Update best state
        if current_fitness > self.best_state.fitness:
            self.best_state = HCState(self.current_position, current_fitness, list(self.action_history))
            self.best_fitness_found = current_fitness
        
        # Update fitness history
        self.fitness_history.append(current_fitness)
        
        # Detect scenario
        scenario = self.detect_scenario(observation, reward)
        
        # Select action based on scenario and variant
        if scenario == GameScenario.PLATEAU and self.steps_without_improvement >= self.plateau_threshold:
            if self.variant == HillClimbingVariant.RANDOM_RESTART and self.steps_without_improvement >= self.restart_threshold:
                selected_action = self.random_restart()
            else:
                selected_action = self.escape_plateau()
        
        elif scenario == GameScenario.STUCK:
            # Force exploration when stuck
            selected_action = self.escape_plateau()
        
        else:
            # Normal hill climbing based on variant
            if self.variant == HillClimbingVariant.STEEPEST_ASCENT:
                selected_action = self.steepest_ascent_selection()
            elif self.variant == HillClimbingVariant.FIRST_IMPROVEMENT:
                selected_action = self.first_improvement_selection()
            elif self.variant == HillClimbingVariant.STOCHASTIC:
                selected_action = self.stochastic_selection()
            else:  # Default to steepest ascent
                selected_action = self.steepest_ascent_selection()
        
        # Update action history
        self.action_history.append(selected_action)
        
        return selected_action
    
    def get_agent_info(self) -> Dict:
        """Return information about the agent's current state"""
        return {
            "algorithm": f"Hill Climbing ({self.variant.value})",
            "current_position": self.current_position,
            "current_fitness": round(self.current_state.fitness, 3),
            "best_fitness": round(self.best_state.fitness, 3),
            "steps_without_improvement": self.steps_without_improvement,
            "plateau_escapes": self.plateau_escapes,
            "random_restarts": self.random_restarts,
            "evaluations": self.evaluations_count,
            "visited_positions": len(self.visited_positions),
            "step_count": self.step_count,
            "recent_actions": list(self.action_history)[-5:],
            "improvement_frequency": len(self.improvement_steps) / max(1, self.step_count)
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
        self.steps_without_improvement = 0
        self.plateau_escapes = 0
        self.random_restarts = 0
        self.evaluations_count = 0
        self.local_maxima_detected = False
        
        # Reset states
        self.current_state = HCState((0, 0), -float('inf'), [])
        self.best_state = HCState((0, 0), -float('inf'), [])
        
        # Reset performance tracking
        self.fitness_history.clear()
        self.best_fitness_found = -float('inf')
        self.improvement_steps.clear()
        self.last_improvement_direction = None
        
        print(f"Hill Climbing Agent ({self.variant.value}) reset for new episode")