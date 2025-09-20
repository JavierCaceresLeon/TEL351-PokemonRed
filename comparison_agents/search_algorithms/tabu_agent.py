"""
Tabu Search Agent with Advanced Heuristics for Pokemon Red Environment
======================================================================

This agent implements a Tabu Search algorithm with sophisticated heuristics
that adapt to different game scenarios. It maintains a tabu list to avoid
cycling through recently visited states/actions.

Key Features:
- Tabu list to prevent cycling
- Multi-scenario heuristics (exploration, battle, navigation, progression)
- Memory-based decision making
- Aspiration criteria for exceptional moves
- Real-time adaptation without training episodes
"""

import numpy as np
import json
from typing import Dict, Tuple, List, Optional, Set
from dataclasses import dataclass
from enum import Enum
import random
from collections import deque


class GameScenario(Enum):
    """Different scenarios the agent can encounter"""
    EXPLORATION = "exploration"  # General map exploration
    BATTLE = "battle"           # In combat
    NAVIGATION = "navigation"   # Moving to specific objectives
    PROGRESSION = "progression" # Key game events (gym battles, catching pokemon)
    STUCK = "stuck"            # Repetitive behavior detection


@dataclass
class HeuristicWeights:
    """Weights for different heuristic components"""
    exploration: float = 1.0
    objective_distance: float = 1.5
    health_consideration: float = 0.8
    level_progression: float = 1.2
    map_familiarity: float = 0.6
    event_completion: float = 2.0


@dataclass
class TabuMove:
    """Represents a tabu move with its attributes"""
    action: int
    state_hash: str
    iteration: int
    scenario: GameScenario
    
    def __eq__(self, other):
        if not isinstance(other, TabuMove):
            return False
        return self.action == other.action and self.state_hash == other.state_hash
    
    def __hash__(self):
        return hash((self.action, self.state_hash))


class TabuSearchAgent:
    """
    Advanced Tabu Search Agent for Pokemon Red
    
    This agent uses tabu search to avoid cycling through states while
    using heuristic functions to evaluate the quality of different actions.
    """
    
    def __init__(self, 
                 tabu_tenure: int = 7,
                 max_tabu_size: int = 50,
                 aspiration_threshold: float = 1.5,
                 scenario_detection_enabled: bool = True):
        
        self.tabu_tenure = tabu_tenure  # How long moves stay in tabu list
        self.max_tabu_size = max_tabu_size
        self.aspiration_threshold = aspiration_threshold  # Factor for aspiration criteria
        self.scenario_detection_enabled = scenario_detection_enabled
        
        # Tabu search components
        self.tabu_list: Set[TabuMove] = set()
        self.iteration_count = 0
        self.best_solution_quality = float('-inf')
        self.current_solution_quality = 0.0
        
        # Recent states tracking for cycle detection
        self.recent_states = deque(maxlen=20)
        self.recent_actions = deque(maxlen=10)
        self.position_sequence = deque(maxlen=15)  # Track recent positions
        
        # Anti-cycling mechanisms
        self.cycle_detection_window = 8
        self.forced_exploration_counter = 0
        self.last_significant_reward = 0
        self.steps_without_progress = 0
        self.direction_bias = None  # Encourage continuing in one direction
        self.bias_steps_remaining = 0
        
        # Enhanced anti-cycling tracking
        self.position_visit_count = {}  # Count visits to each position
        self.direction_preference = {0: 0.1, 1: 0.1, 2: 0.1, 3: 0.1}  # Direction preference
        self.step_count = 0  # For tracking overall progress
        
        # Game state tracking for better decision making
        self.stuck_counter = 0
        self.exploration_bonus = 1.0
        self.last_positions = deque(maxlen=10)
        
        # Menu and action control
        self.recent_menu_actions = deque(maxlen=15)  # Track recent START/SELECT actions
        self.menu_cooldown = 0  # Frames to wait before allowing START again
        self.consecutive_menu_presses = 0  # Count consecutive menu button presses
        self.last_menu_press = -10  # Frame when last menu button was pressed
        self.menu_spam_threshold = 3  # Max consecutive menu presses before penalty
        
        # Action mappings (from red_gym_env_v2.py)
        self.valid_actions = [
            0,  # WindowEvent.PRESS_ARROW_DOWN
            1,  # WindowEvent.PRESS_ARROW_LEFT
            2,  # WindowEvent.PRESS_ARROW_RIGHT
            3,  # WindowEvent.PRESS_ARROW_UP
            4,  # WindowEvent.PRESS_BUTTON_A
            5,  # WindowEvent.PRESS_BUTTON_B
            6,  # WindowEvent.PRESS_BUTTON_START
        ]
        
        # Movement actions only
        self.movement_actions = [0, 1, 2, 3]  # UP, DOWN, LEFT, RIGHT
        
        # Heuristic weights for different scenarios
        self.scenario_weights = {
            GameScenario.EXPLORATION: HeuristicWeights(
                exploration=1.5, objective_distance=0.8, health_consideration=0.6,
                level_progression=0.4, map_familiarity=1.0, event_completion=0.8
            ),
            GameScenario.BATTLE: HeuristicWeights(
                exploration=0.2, objective_distance=0.3, health_consideration=2.0,
                level_progression=1.5, map_familiarity=0.1, event_completion=0.5
            ),
            GameScenario.NAVIGATION: HeuristicWeights(
                exploration=0.6, objective_distance=2.0, health_consideration=0.8,
                level_progression=0.6, map_familiarity=1.2, event_completion=1.0
            ),
            GameScenario.PROGRESSION: HeuristicWeights(
                exploration=0.8, objective_distance=1.0, health_consideration=1.0,
                level_progression=2.0, map_familiarity=0.8, event_completion=2.5
            ),
            GameScenario.STUCK: HeuristicWeights(
                exploration=2.0, objective_distance=1.5, health_consideration=0.5,
                level_progression=0.8, map_familiarity=0.3, event_completion=1.0
            )
        }
        
        # State tracking for heuristics
        self.previous_positions = []
        self.previous_rewards = []
        self.step_count = 0
        self.last_reward_gain = 0
        self.stuck_counter = 0
        
        # Game state memory
        self.position_visits = {}
        self.action_rewards = {action: [] for action in self.valid_actions}
    
    def is_menu_spam(self) -> bool:
        """Detect if the agent is spamming menu buttons"""
        if len(self.recent_menu_actions) < 3:
            return False
        
        # Count recent START/SELECT actions
        recent_menu_count = sum(1 for action in self.recent_menu_actions if action in [6])  # START only
        
        # Consider it spam if more than 2 out of last 5 actions were menu actions
        if len(self.recent_menu_actions) >= 5:
            recent_5 = list(self.recent_menu_actions)[-5:]
            menu_ratio = sum(1 for action in recent_5 if action in [6]) / 5
            return menu_ratio > 0.4  # More than 40% menu actions = spam
        
        return recent_menu_count >= 3  # 3 or more menu actions in recent history
    
    def update_menu_tracking(self, action: int):
        """Update menu action tracking"""
        self.recent_menu_actions.append(action)
        
        # Update cooldown
        if self.menu_cooldown > 0:
            self.menu_cooldown -= 1
        
        # Track consecutive menu presses
        if action == 6:  # START button
            if self.step_count - self.last_menu_press <= 2:  # Within 2 steps
                self.consecutive_menu_presses += 1
            else:
                self.consecutive_menu_presses = 1
            self.last_menu_press = self.step_count
            
            # Set cooldown if spamming
            if self.consecutive_menu_presses >= self.menu_spam_threshold:
                self.menu_cooldown = 10  # 10 frame cooldown
        else:
            # Reset consecutive count if not menu action
            if action in self.movement_actions or action in [4, 5]:  # Movement or A/B
                self.consecutive_menu_presses = 0
    
    def get_menu_penalty(self, action: int) -> float:
        """Calculate penalty for menu actions to prevent spamming"""
        if action != 6:  # Not START button
            return 0.0
        
        penalty = 0.0
        
        # Strong penalty if in cooldown
        if self.menu_cooldown > 0:
            penalty += 0.8  # Very high penalty
        
        # Penalty based on recent usage
        if self.is_menu_spam():
            penalty += 0.6  # High penalty for spam
        
        # Penalty for consecutive presses
        if self.consecutive_menu_presses >= 2:
            penalty += 0.3 * self.consecutive_menu_presses
        
        # Penalty if used recently
        recent_starts = sum(1 for action in list(self.recent_menu_actions)[-5:] if action == 6)
        if recent_starts >= 2:
            penalty += 0.4  # Moderate penalty for recent use
        
        return min(penalty, 0.95)  # Cap penalty at 95%

    def get_state_hash(self, observation) -> str:
        """Create a hash representation of the current state"""
        try:
            # Handle different observation types
            if isinstance(observation, dict):
                # If observation is a dictionary (v2 environment format)
                # Use screen data if available
                if 'screens' in observation:
                    screen_data = observation['screens']
                    if isinstance(screen_data, np.ndarray) and len(screen_data.shape) > 1:
                        sample_points = screen_data[::8, ::8].flatten()[:50]
                        return str(hash(tuple(sample_points.astype(int))))
                
                # Fall back to using key game state values
                hash_values = []
                for key in ['health', 'level', 'badges', 'events']:
                    if key in observation:
                        val = observation[key]
                        if isinstance(val, np.ndarray):
                            hash_values.extend(val.flatten()[:5].astype(int))
                        else:
                            hash_values.append(int(val) if val is not None else 0)
                
                return str(hash(tuple(hash_values[:20])))  # Limit to 20 values
                
            elif isinstance(observation, np.ndarray):
                # If observation is a numpy array
                if len(observation.shape) > 1:
                    # If observation is an image, sample key pixels
                    sample_points = observation[::8, ::8].flatten()[:50]
                    return str(hash(tuple(sample_points.astype(int))))
                else:
                    # If observation is a vector, use it directly
                    return str(hash(tuple(observation.astype(int))))
            
            else:
                # Fallback for other types
                return str(hash(str(observation)[:100]))  # Use string representation
                
        except Exception as e:
            # If all else fails, use iteration count as hash
            return str(self.iteration_count)
    
    def update_tabu_list(self, action: int, state_hash: str, scenario: GameScenario):
        """Update the tabu list with the current move"""
        # Remove old tabu moves based on tenure
        current_iteration = self.iteration_count
        self.tabu_list = {
            move for move in self.tabu_list 
            if current_iteration - move.iteration < self.tabu_tenure
        }
        
        # Add current move to tabu list
        tabu_move = TabuMove(action, state_hash, current_iteration, scenario)
        self.tabu_list.add(tabu_move)
        
        # Limit tabu list size
        if len(self.tabu_list) > self.max_tabu_size:
            # Remove oldest move
            oldest_move = min(self.tabu_list, key=lambda x: x.iteration)
            self.tabu_list.remove(oldest_move)
    
    def is_tabu(self, action: int, state_hash: str) -> bool:
        """Check if a move is in the tabu list"""
        test_move = TabuMove(action, state_hash, 0, GameScenario.EXPLORATION)
        return any(move.action == action and move.state_hash == state_hash 
                  for move in self.tabu_list)
    
    def detect_cycles(self) -> bool:
        """Detect if the agent is stuck in a cycle"""
        if len(self.position_sequence) < self.cycle_detection_window:
            return False
        
        # Check for position cycles
        recent_positions = list(self.position_sequence)[-self.cycle_detection_window:]
        unique_positions = len(set(recent_positions))
        
        # If we're visiting very few unique positions, we might be cycling
        if unique_positions <= 3:
            return True
        
        # Check for action cycles
        if len(self.recent_actions) >= 6:
            recent_actions = list(self.recent_actions)[-6:]
            # Check if we're repeating the same 2-3 action sequence
            if len(set(recent_actions)) <= 2:
                return True
        
        return False
    
    def get_anti_cycle_action(self) -> int:
        """Get an action specifically designed to break cycles"""
        # Forced exploration: choose a random action we haven't used much recently
        action_counts = {}
        recent_actions = list(self.recent_actions)[-10:] if len(self.recent_actions) >= 10 else list(self.recent_actions)
        
        # Create candidate actions (exclude problematic menu actions if spamming)
        candidate_actions = self.valid_actions.copy()
        
        # If we're spamming menu, remove START from candidates
        if self.is_menu_spam() or self.menu_cooldown > 0:
            candidate_actions = [a for a in candidate_actions if a != 6]  # Remove START
        
        for action in candidate_actions:
            action_counts[action] = recent_actions.count(action)
        
        # Choose the least used action from candidates
        if candidate_actions:
            least_used_action = min(action_counts, key=action_counts.get)
        else:
            # Fallback to movement actions only
            movement_counts = {a: recent_actions.count(a) for a in self.movement_actions}
            least_used_action = min(movement_counts, key=movement_counts.get)
        
        # If it's a movement action, set a direction bias
        if least_used_action in self.movement_actions:
            self.direction_bias = least_used_action
            self.bias_steps_remaining = 5  # Continue in this direction for 5 steps
        
        return least_used_action
    
    def aspiration_criteria(self, move_quality: float) -> bool:
        """Check if a move satisfies aspiration criteria (overrides tabu)"""
        return move_quality > self.best_solution_quality * self.aspiration_threshold
    
    def detect_scenario(self, observation, game_state: Dict) -> GameScenario:
        """Detect current game scenario based on observation and game state"""
        if not self.scenario_detection_enabled:
            return GameScenario.EXPLORATION
        
        try:
            # Priority 1: Check if we're stuck in cycles
            if self.detect_cycles() or self.forced_exploration_counter > 0:
                if self.forced_exploration_counter > 0:
                    self.forced_exploration_counter -= 1
                return GameScenario.STUCK
            
            # Priority 2: Check for battle scenario
            if game_state.get('battle', False) or game_state.get('battle_type', 0) > 0:
                return GameScenario.BATTLE
            
            # Priority 3: Check for progression scenario (high rewards, new areas)
            if len(self.previous_rewards) >= 3:
                recent_rewards = self.previous_rewards[-3:]
                avg_recent = sum(recent_rewards) / len(recent_rewards)
                if avg_recent > self.last_significant_reward + 0.1:
                    self.last_significant_reward = avg_recent
                    self.steps_without_progress = 0
                    return GameScenario.PROGRESSION
            
            # Track steps without progress
            self.steps_without_progress += 1
            
            # Priority 4: Force exploration if no progress for too long
            if self.steps_without_progress > 50:
                self.forced_exploration_counter = 10
                self.steps_without_progress = 0
                return GameScenario.STUCK
            
            # Priority 5: Check for navigation scenario (consistent movement direction)
            if len(self.recent_actions) >= 4:
                recent_movement = [a for a in self.recent_actions if a in self.movement_actions]
                if len(recent_movement) >= 3:
                    direction_consistency = len(set(recent_movement[-3:])) <= 2
                    if direction_consistency and not self.detect_cycles():
                        return GameScenario.NAVIGATION
            
            # Default to exploration
            return GameScenario.EXPLORATION
            
        except Exception as e:
            print(f"Warning: Scenario detection failed: {e}")
            return GameScenario.EXPLORATION
    
    def calculate_move_quality(self, action: int, observation, 
                              game_state: Dict, scenario: GameScenario) -> float:
        """Calculate the quality/utility of a potential move"""
        weights = self.scenario_weights[scenario]
        quality = 0.0
        
        try:
            # Base exploration bonus (encourage trying different actions)
            action_rewards = self.action_rewards.get(action, [])
            action_frequency = len(action_rewards)
            if action_frequency == 0:
                quality += weights.exploration * 0.5  # Bonus for unexplored actions
            else:
                # Safe access to recent rewards
                recent_rewards = action_rewards[-5:] if len(action_rewards) >= 5 else action_rewards
                if recent_rewards:
                    avg_reward = np.mean(recent_rewards)
                    quality += weights.exploration * avg_reward * 0.1
            
            # Health consideration
            player_hp = game_state.get('hp', 100)
            max_hp = game_state.get('max_hp', 100)
            health_ratio = player_hp / max_hp if max_hp > 0 else 1.0
            
            if scenario == GameScenario.BATTLE:
                # In battle, prioritize survival actions
                if action == 4:  # A button (attack/select)
                    quality += weights.health_consideration * 0.3
                elif action == 5 and health_ratio < 0.3:  # B button (run/cancel) when low health
                    quality += weights.health_consideration * 0.4
            else:
                # Outside battle, health is less critical but still considered
                quality += weights.health_consideration * health_ratio * 0.1
            
            # Level progression consideration
            level = game_state.get('level', 1)
            quality += weights.level_progression * (level / 100.0) * 0.2
            
            # Event completion bonus
            badges = game_state.get('badges', 0)
            pcount = game_state.get('pcount', 0)
            
            event_score = (badges * 0.3 + pcount * 0.1)
            quality += weights.event_completion * event_score * 0.1
            
            # ENHANCED ANTI-CYCLING: Strong penalties for recently used actions
            if len(self.recent_actions) >= 3:
                recent_actions_list = list(self.recent_actions)
                recent_count = recent_actions_list[-5:].count(action)  # Check last 5 actions
                quality -= recent_count * 0.3  # Increased penalty
            
            # DIRECTION BIAS: If we have a direction bias, strongly favor it
            if self.bias_steps_remaining > 0 and action == self.direction_bias:
                quality += 1.0  # Strong bias towards continuing direction
            
            # CYCLE BREAKING: Enhanced when stuck
            if scenario == GameScenario.STUCK:
                recent_actions_list = list(self.recent_actions)
                recent_frequency = recent_actions_list[-8:].count(action) if len(recent_actions_list) >= 8 else recent_actions_list.count(action)
                
                if recent_frequency == 0:
                    quality += 0.8  # Big bonus for unused actions
                elif recent_frequency <= 1:
                    quality += 0.4  # Medium bonus for rarely used
                else:
                    quality -= 0.6  # Strong penalty for frequently used
                
                # Extra bonus for movement actions to encourage exploration
                if action in self.movement_actions:
                    quality += 0.3
                
                # Add more randomness to break deterministic cycles
                quality += random.uniform(-0.2, 0.3)
            
            # EXPLORATION BONUS: Penalize staying in same areas
            current_pos = (game_state.get('x', 0), game_state.get('y', 0))
            if current_pos in self.position_visits:
                visit_count = self.position_visits[current_pos]
                # Penalty for staying in heavily visited areas
                if visit_count > 5:
                    quality -= 0.2
                elif visit_count > 10:
                    quality -= 0.4
            
            # MENU SPAM PREVENTION: Apply heavy penalties for menu abuse
            menu_penalty = self.get_menu_penalty(action)
            quality -= menu_penalty
            
            # Scenario-specific bonuses
            if scenario == GameScenario.STUCK:
                # When stuck, prefer movement actions and random exploration
                if action in self.movement_actions:
                    quality += 0.3
                quality += random.uniform(-0.1, 0.2)  # Add randomness to break cycles
            
            elif scenario == GameScenario.PROGRESSION:
                # During progression, prefer interaction actions BUT with menu control
                if action == 4:  # A button - safe to encourage
                    quality += 0.2
                elif action == 6:  # START button - be very careful
                    # Only allow START if not spamming and not in cooldown
                    if not self.is_menu_spam() and self.menu_cooldown == 0:
                        quality += 0.1  # Reduced bonus
                    else:
                        quality -= 0.5  # Strong penalty if spamming
            
            return quality
            
        except Exception as e:
            print(f"Warning: Quality calculation failed: {e}")
            return random.uniform(0, 0.1)  # Return small random value as fallback
    
    def select_action(self, observation, game_state: Dict) -> Tuple[int, Dict]:
        """
        Select the best non-tabu action using tabu search principles with enhanced anti-cycling
        """
        self.iteration_count += 1
        self.step_count += 1
        
        # Track position for cycle detection
        current_pos = (game_state.get('x', 0), game_state.get('y', 0))
        self.position_sequence.append(current_pos)
        
        # Detect current scenario
        scenario = self.detect_scenario(observation, game_state)
        
        # FORCED CYCLE BREAKING: If we detect cycles, force a different action
        if scenario == GameScenario.STUCK or self.detect_cycles():
            if random.random() < 0.3:  # 30% chance to force anti-cycle action
                forced_action = self.get_anti_cycle_action()
                decision_info = {
                    'agent_type': 'tabu_search',
                    'scenario': scenario.value,
                    'action_qualities': {forced_action: 1.0},
                    'selected_quality': 1.0,
                    'best_quality': self.best_solution_quality,
                    'tabu_list_size': len(self.tabu_list),
                    'iteration': self.iteration_count,
                    'non_tabu_count': 1,
                    'aspiration_used': False,
                    'stuck_counter': self.stuck_counter,
                    'forced_anti_cycle': True
                }
                
                # Track the action
                self.recent_actions.append(forced_action)
                self.recent_states.append(str(self.iteration_count))
                
                # Update menu tracking for forced actions too
                self.update_menu_tracking(forced_action)
                
                return forced_action, decision_info
        
        # Normal tabu search logic
        state_hash = self.get_state_hash(observation)
        
        # Evaluate all possible actions
        action_qualities = {}
        non_tabu_actions = []
        
        for action in self.valid_actions:
            quality = self.calculate_move_quality(action, observation, game_state, scenario)
            action_qualities[action] = quality
            
            # Check if action is tabu
            if not self.is_tabu(action, state_hash):
                non_tabu_actions.append((action, quality))
            elif self.aspiration_criteria(quality):
                # Override tabu if aspiration criteria is met
                non_tabu_actions.append((action, quality))
        
        # Select best action from non-tabu actions
        if non_tabu_actions:
            # Sort by quality and select best
            non_tabu_actions.sort(key=lambda x: x[1], reverse=True)
            selected_action = non_tabu_actions[0][0]
            self.current_solution_quality = non_tabu_actions[0][1]
        else:
            # If all actions are tabu (shouldn't happen with aspiration), 
            # select the least recently added to tabu list
            if self.tabu_list:
                oldest_tabu = min(self.tabu_list, key=lambda x: x.iteration)
                selected_action = oldest_tabu.action
                self.current_solution_quality = action_qualities[selected_action]
            else:
                # Fallback to random action
                selected_action = random.choice(self.valid_actions)
                self.current_solution_quality = action_qualities[selected_action]
        
        # Update tabu list with selected action
        self.update_tabu_list(selected_action, state_hash, scenario)
        
        # Update best solution if current is better
        if self.current_solution_quality > self.best_solution_quality:
            self.best_solution_quality = self.current_solution_quality
        
        # Track recent actions and states
        self.recent_actions.append(selected_action)
        self.recent_states.append(state_hash)
        
        # Update menu tracking to prevent spam
        self.update_menu_tracking(selected_action)
        
        # Prepare decision info for metrics
        decision_info = {
            'agent_type': 'tabu_search',
            'scenario': scenario.value,
            'action_qualities': action_qualities,
            'selected_quality': self.current_solution_quality,
            'best_quality': self.best_solution_quality,
            'tabu_list_size': len(self.tabu_list),
            'iteration': self.iteration_count,
            'non_tabu_count': len(non_tabu_actions),
            'aspiration_used': selected_action in [a for a, q in action_qualities.items() 
                                                 if self.is_tabu(a, state_hash) and 
                                                 self.aspiration_criteria(q)],
            'stuck_counter': self.stuck_counter,
            'forced_anti_cycle': False,
            'cycles_detected': self.detect_cycles(),
            'menu_cooldown': self.menu_cooldown,
            'consecutive_menu_presses': self.consecutive_menu_presses,
            'is_menu_spam': self.is_menu_spam(),
            'menu_penalty_applied': self.get_menu_penalty(selected_action)
        }
        
        return selected_action, decision_info
    
    def update_performance(self, action: int, reward: float, observation, 
                          game_state: Dict):
        """Update agent's performance tracking"""
        # Track rewards for this action
        if action not in self.action_rewards:
            self.action_rewards[action] = []
        
        self.action_rewards[action].append(reward)
        if len(self.action_rewards[action]) > 20:  # Keep only recent rewards
            self.action_rewards[action] = self.action_rewards[action][-20:]
        
        # Track general performance
        self.previous_rewards.append(reward)
        if len(self.previous_rewards) > 50:  # Keep only recent rewards
            self.previous_rewards = self.previous_rewards[-50:]
        
        # Track position for exploration metrics
        current_pos = (game_state.get('x', 0), game_state.get('y', 0))
        self.previous_positions.append(current_pos)
        if len(self.previous_positions) > 30:
            self.previous_positions = self.previous_positions[-30:]
        
        # Update position visit counts
        if current_pos not in self.position_visits:
            self.position_visits[current_pos] = 0
        self.position_visits[current_pos] += 1
    
    def get_exploration_metrics(self) -> Dict:
        """Get metrics about exploration and tabu search performance"""
        unique_positions = len(self.position_visits)
        total_visits = sum(self.position_visits.values())
        
        return {
            'unique_positions_visited': unique_positions,
            'total_position_visits': total_visits,
            'exploration_efficiency': unique_positions / max(total_visits, 1),
            'average_action_reward': {
                action: np.mean(rewards[-10:]) if rewards else 0.0 
                for action, rewards in self.action_rewards.items()
            },
            'tabu_statistics': {
                'current_tabu_size': len(self.tabu_list),
                'max_tabu_size': self.max_tabu_size,
                'tabu_tenure': self.tabu_tenure,
                'current_iteration': self.iteration_count
            },
            'performance_metrics': {
                'best_solution_quality': self.best_solution_quality,
                'current_solution_quality': self.current_solution_quality,
                'stuck_episodes': self.stuck_counter
            }
        }
    
    def reset_episode(self):
        """Reset episode-specific tracking"""
        self.recent_actions.clear()
        self.recent_states.clear()
        self.stuck_counter = 0
        self.iteration_count = 0
        # Keep tabu list and learned patterns between episodes
    
    def save_agent_state(self, filepath: str):
        """Save agent state to file"""
        state = {
            'tabu_tenure': self.tabu_tenure,
            'max_tabu_size': self.max_tabu_size,
            'aspiration_threshold': self.aspiration_threshold,
            'iteration_count': self.iteration_count,
            'best_solution_quality': self.best_solution_quality,
            'position_visits': self.position_visits,
            'action_rewards': {k: v[-10:] for k, v in self.action_rewards.items()},  # Save only recent
            'exploration_metrics': self.get_exploration_metrics()
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2, default=str)
    
    def load_agent_state(self, filepath: str):
        """Load agent state from file"""
        try:
            with open(filepath, 'r') as f:
                state = json.load(f)
            
            self.tabu_tenure = state.get('tabu_tenure', self.tabu_tenure)
            self.max_tabu_size = state.get('max_tabu_size', self.max_tabu_size)
            self.aspiration_threshold = state.get('aspiration_threshold', self.aspiration_threshold)
            self.iteration_count = state.get('iteration_count', 0)
            self.best_solution_quality = state.get('best_solution_quality', float('-inf'))
            self.position_visits = state.get('position_visits', {})
            
            action_rewards = state.get('action_rewards', {})
            for action_str, rewards in action_rewards.items():
                action = int(action_str)
                if action in self.action_rewards:
                    self.action_rewards[action] = rewards
                    
        except Exception as e:
            print(f"Warning: Could not load agent state: {e}")
