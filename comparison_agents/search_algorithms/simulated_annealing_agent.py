"""
Simulated Annealing Agent for Pokemon Red Environment
===================================================

This agent implements the Simulated Annealing algorithm for stochastic optimization
in the Pokemon Red environment.

Key Features:
- Temperature-based acceptance of moves
- Cooling schedule for exploration-exploitation balance
- Energy function based on game objectives
- Adaptive parameter adjustment
- Memory of best solutions found
"""

import numpy as np
import math
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
class SAState:
    """State representation for Simulated Annealing"""
    position: Tuple[int, int]
    energy: float
    actions_taken: List[int]
    visit_count: int = 0


@dataclass
class CoolingSchedule:
    """Parameters for temperature cooling"""
    initial_temp: float = 100.0
    final_temp: float = 0.1
    cooling_rate: float = 0.95
    reheat_threshold: int = 50  # Steps without improvement before reheating


class SimulatedAnnealingAgent:
    """
    Simulated Annealing Agent for Pokemon Red Environment
    
    This agent uses simulated annealing to escape local optima and find
    globally optimal solutions for navigation and objective completion.
    """
    
    def __init__(self):
        """Initialize Simulated Annealing agent"""
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
        
        # Simulated Annealing parameters
        self.cooling_schedule = CoolingSchedule()
        self.current_temp = self.cooling_schedule.initial_temp
        self.initial_temp = self.cooling_schedule.initial_temp
        
        # State tracking
        self.current_state = SAState((0, 0), float('inf'), [])
        self.best_state = SAState((0, 0), float('inf'), [])
        self.previous_state = SAState((0, 0), float('inf'), [])
        
        # Game state tracking
        self.current_position = (0, 0)
        self.goal_position = (5, 3)  # Pokemon selection area
        self.visited_positions = set()
        self.position_counts = {}
        self.last_positions = deque(maxlen=12)
        
        # Energy calculation components
        self.distance_weight = 1.0
        self.exploration_weight = 0.5
        self.repetition_penalty = 2.0
        self.progress_bonus = -3.0
        
        # Action tracking
        self.action_history = deque(maxlen=30)
        self.last_rewards = deque(maxlen=8)
        self.step_count = 0
        self.steps_without_improvement = 0
        self.accepted_moves = 0
        self.rejected_moves = 0
        
        # Algorithm state
        self.reheating_enabled = True
        self.adaptive_cooling = True
        self.last_improvement_step = 0
        
        # Objectives and targets
        self.objectives = [
            (5, 3),   # Pokemon selection
            (3, 2),   # Professor Oak
            (4, 6),   # House exit
            (5, 8),   # Lab entrance
            (6, 4),   # Alternative target
            (2, 5),   # Exploration point
        ]
        
        # Performance tracking
        self.energy_history = deque(maxlen=100)
        self.temperature_history = deque(maxlen=100)
        self.acceptance_rate = 0.0
        
        print("Simulated Annealing Agent initialized with adaptive cooling")
    
    def calculate_energy(self, position: Tuple[int, int], action_sequence: List[int] = None) -> float:
        """
        Calculate energy (cost) of a state
        Lower energy is better (minimization problem)
        """
        energy = 0.0
        
        # Distance to closest objective (primary component)
        min_distance = min(abs(position[0] - obj[0]) + abs(position[1] - obj[1]) 
                          for obj in self.objectives)
        energy += min_distance * self.distance_weight
        
        # Exploration component (encourage visiting new areas)
        if position not in self.visited_positions:
            energy += self.exploration_weight * (-2)  # Bonus for new positions
        else:
            # Penalty for revisiting positions
            visit_count = self.position_counts.get(position, 0)
            energy += visit_count * self.repetition_penalty
        
        # Progress tracking (based on rewards)
        if len(self.last_rewards) > 0:
            recent_reward = sum(self.last_rewards) / len(self.last_rewards)
            if recent_reward > 0:
                energy += self.progress_bonus  # Bonus for positive rewards
        
        # Path efficiency (if action sequence provided)
        if action_sequence:
            # Penalize very long sequences
            if len(action_sequence) > 10:
                energy += (len(action_sequence) - 10) * 0.1
            
            # Penalize repetitive action patterns
            if len(action_sequence) >= 3:
                if action_sequence[-1] == action_sequence[-2] == action_sequence[-3]:
                    energy += 1.0  # Penalty for repeating same action
        
        # Stuck detection penalty
        if len(self.last_positions) >= 6:
            unique_recent = len(set(list(self.last_positions)[-6:]))
            if unique_recent <= 2:
                energy += 5.0  # High penalty for being stuck
        
        return energy
    
    def acceptance_probability(self, current_energy: float, new_energy: float, temperature: float) -> float:
        """
        Calculate probability of accepting a move
        Uses classic Boltzmann distribution
        """
        if new_energy < current_energy:
            return 1.0  # Always accept better solutions
        
        if temperature <= 0:
            return 0.0  # No random moves at zero temperature
        
        energy_diff = new_energy - current_energy
        try:
            probability = math.exp(-energy_diff / temperature)
            return min(1.0, probability)
        except OverflowError:
            return 0.0
    
    def update_temperature(self):
        """Update temperature according to cooling schedule"""
        if self.adaptive_cooling:
            # Adaptive cooling based on acceptance rate
            if self.step_count > 10:
                recent_accepted = self.accepted_moves
                recent_total = self.accepted_moves + self.rejected_moves
                if recent_total > 0:
                    self.acceptance_rate = recent_accepted / recent_total
                
                # Adjust cooling rate based on acceptance rate
                if self.acceptance_rate > 0.7:
                    # Cool faster if accepting too many moves
                    cooling_factor = self.cooling_schedule.cooling_rate * 0.95
                elif self.acceptance_rate < 0.3:
                    # Cool slower if rejecting too many moves
                    cooling_factor = self.cooling_schedule.cooling_rate * 1.05
                else:
                    cooling_factor = self.cooling_schedule.cooling_rate
                
                self.current_temp *= cooling_factor
        else:
            # Standard geometric cooling
            self.current_temp *= self.cooling_schedule.cooling_rate
        
        # Ensure temperature doesn't go below minimum
        self.current_temp = max(self.current_temp, self.cooling_schedule.final_temp)
        
        # Reheating mechanism
        if (self.reheating_enabled and 
            self.steps_without_improvement > self.cooling_schedule.reheat_threshold):
            self.current_temp = self.initial_temp * 0.5  # Partial reheat
            self.steps_without_improvement = 0
            print(f"Reheating at step {self.step_count}")
    
    def generate_neighbor_action(self, current_action: int = None) -> int:
        """
        Generate a neighboring action for the current state
        """
        # Probability distribution over actions based on current state
        action_probs = np.ones(self.action_space)
        
        # Bias towards movement actions early in the game
        if self.step_count < 200:
            action_probs[:4] *= 2.0  # Favor UP, DOWN, LEFT, RIGHT
        
        # Avoid recently used actions to encourage exploration
        if len(self.action_history) > 0:
            recent_actions = list(self.action_history)[-3:]
            for action in recent_actions:
                action_probs[action] *= 0.7
        
        # If we're close to objectives, favor interaction actions
        min_distance = min(abs(self.current_position[0] - obj[0]) + abs(self.current_position[1] - obj[1]) 
                          for obj in self.objectives)
        if min_distance <= 2:
            action_probs[4:] *= 3.0  # Favor A, B, START
        
        # Normalize probabilities
        action_probs /= np.sum(action_probs)
        
        # Sample action
        return np.random.choice(self.action_space, p=action_probs)
    
    def extract_position_from_observation(self, observation) -> Tuple[int, int]:
        """
        Extract position from game observation
        Enhanced for Simulated Annealing with more stable tracking
        """
        if hasattr(observation, 'shape') and len(observation.shape) == 3:
            # Use observation content for position estimation
            obs_mean = np.mean(observation)
            obs_hash = hash(observation.tobytes()) % 2000
            
            # More sophisticated position calculation
            base_x = (obs_hash % 20)
            base_y = ((obs_hash // 20) % 20)
            
            # Apply movement based on recent actions
            movement_x = movement_y = 0
            if len(self.action_history) >= 2:
                for action in list(self.action_history)[-2:]:
                    if action == 0:    # UP
                        movement_y -= 1
                    elif action == 1:  # DOWN
                        movement_y += 1
                    elif action == 2:  # LEFT
                        movement_x -= 1
                    elif action == 3:  # RIGHT
                        movement_x += 1
            
            final_x = (base_x + movement_x) % 20
            final_y = (base_y + movement_y) % 20
            
            return (final_x, final_y)
        else:
            # Fallback position calculation
            return (self.step_count % 20, (self.step_count // 20) % 20)
    
    def detect_scenario(self, observation, reward: float) -> GameScenario:
        """Detect current game scenario"""
        # Check if stuck
        if len(self.last_positions) >= 8:
            unique_positions = len(set(self.last_positions))
            if unique_positions <= 2:
                return GameScenario.STUCK
        
        # Check for progression
        if reward > 0.1 or (len(self.last_rewards) > 0 and sum(self.last_rewards) > 0.2):
            return GameScenario.PROGRESSION
        
        # Distance-based scenario detection
        min_distance = min(abs(self.current_position[0] - obj[0]) + abs(self.current_position[1] - obj[1]) 
                          for obj in self.objectives)
        
        if min_distance > 7:
            return GameScenario.EXPLORATION
        else:
            return GameScenario.NAVIGATION
    
    def select_action(self, observation, reward: float = 0.0) -> int:
        """
        Select next action using Simulated Annealing
        """
        self.step_count += 1
        
        # Extract current position
        self.current_position = self.extract_position_from_observation(observation)
        
        # Update tracking
        self.last_positions.append(self.current_position)
        self.visited_positions.add(self.current_position)
        self.position_counts[self.current_position] = self.position_counts.get(self.current_position, 0) + 1
        self.last_rewards.append(reward)
        
        # Calculate current state energy
        current_energy = self.calculate_energy(self.current_position, list(self.action_history))
        self.current_state = SAState(self.current_position, current_energy, list(self.action_history))
        
        # Update best state if current is better
        if current_energy < self.best_state.energy:
            self.best_state = SAState(self.current_position, current_energy, list(self.action_history))
            self.last_improvement_step = self.step_count
            self.steps_without_improvement = 0
        else:
            self.steps_without_improvement += 1
        
        # Generate candidate action
        candidate_action = self.generate_neighbor_action()
        
        # Simulate the effect of the candidate action on position
        # (This is a simplified simulation)
        candidate_position = self.current_position
        if candidate_action == 0:  # UP
            candidate_position = (max(0, self.current_position[0] - 1), self.current_position[1])
        elif candidate_action == 1:  # DOWN
            candidate_position = (min(19, self.current_position[0] + 1), self.current_position[1])
        elif candidate_action == 2:  # LEFT
            candidate_position = (self.current_position[0], max(0, self.current_position[1] - 1))
        elif candidate_action == 3:  # RIGHT
            candidate_position = (self.current_position[0], min(19, self.current_position[1] + 1))
        # For A, B, START actions, position stays the same
        
        # Calculate energy of candidate state
        candidate_energy = self.calculate_energy(candidate_position, list(self.action_history) + [candidate_action])
        
        # Decide whether to accept the candidate move
        accept_probability = self.acceptance_probability(current_energy, candidate_energy, self.current_temp)
        
        if random.random() < accept_probability:
            # Accept the move
            selected_action = candidate_action
            self.accepted_moves += 1
        else:
            # Reject the move - choose based on current best strategy
            self.rejected_moves += 1
            scenario = self.detect_scenario(observation, reward)
            
            if scenario == GameScenario.STUCK:
                # If stuck, try a random action to escape
                selected_action = random.randint(0, 6)
            elif scenario == GameScenario.PROGRESSION:
                # If making progress, continue with actions that led to progress
                if len(self.best_state.actions_taken) > 0:
                    selected_action = self.best_state.actions_taken[-1]
                else:
                    selected_action = candidate_action
            else:
                # Default: use the candidate action anyway with some probability
                if random.random() < 0.3:
                    selected_action = candidate_action
                else:
                    # Choose action that moves towards closest objective
                    min_dist = float('inf')
                    best_action = 0
                    for action in range(4):  # Only movement actions
                        test_pos = self.current_position
                        if action == 0:    # UP
                            test_pos = (max(0, self.current_position[0] - 1), self.current_position[1])
                        elif action == 1:  # DOWN
                            test_pos = (min(19, self.current_position[0] + 1), self.current_position[1])
                        elif action == 2:  # LEFT
                            test_pos = (self.current_position[0], max(0, self.current_position[1] - 1))
                        elif action == 3:  # RIGHT
                            test_pos = (self.current_position[0], min(19, self.current_position[1] + 1))
                        
                        dist = min(abs(test_pos[0] - obj[0]) + abs(test_pos[1] - obj[1]) 
                                  for obj in self.objectives)
                        if dist < min_dist:
                            min_dist = dist
                            best_action = action
                    
                    selected_action = best_action
        
        # Update temperature
        self.update_temperature()
        
        # Update action history
        self.action_history.append(selected_action)
        
        # Update performance tracking
        self.energy_history.append(current_energy)
        self.temperature_history.append(self.current_temp)
        
        return selected_action
    
    def get_agent_info(self) -> Dict:
        """Return information about the agent's current state"""
        return {
            "algorithm": "Simulated Annealing",
            "current_position": self.current_position,
            "current_energy": round(self.current_state.energy, 3),
            "best_energy": round(self.best_state.energy, 3),
            "temperature": round(self.current_temp, 3),
            "acceptance_rate": round(self.acceptance_rate, 3),
            "accepted_moves": self.accepted_moves,
            "rejected_moves": self.rejected_moves,
            "steps_without_improvement": self.steps_without_improvement,
            "visited_positions": len(self.visited_positions),
            "step_count": self.step_count,
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
        self.steps_without_improvement = 0
        self.accepted_moves = 0
        self.rejected_moves = 0
        self.last_improvement_step = 0
        
        # Reset SA specific parameters
        self.current_temp = self.initial_temp
        self.current_state = SAState((0, 0), float('inf'), [])
        self.best_state = SAState((0, 0), float('inf'), [])
        self.previous_state = SAState((0, 0), float('inf'), [])
        
        # Reset performance tracking
        self.energy_history.clear()
        self.temperature_history.clear()
        self.acceptance_rate = 0.0
        
        print("Simulated Annealing Agent reset for new episode")