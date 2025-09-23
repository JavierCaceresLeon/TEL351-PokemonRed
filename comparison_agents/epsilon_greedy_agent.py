"""
Epsilon Greedy Agent with Advanced Heuristics for Pokemon Red Environment
=========================================================================

This agent implements an epsilon-greedy search algorithm with sophisticated heuristics
that adapt to different game scenarios. Unlike PPO which relies on episodic learning,
this agent makes decisions based on the current state using heuristic functions.

Key Features:
- Multi-scenario heuristics (exploration, battle, navigation, progression)
- Dynamic epsilon decay based on progress
- State-aware decision making
- Real-time adaptation without training episodes
"""

import numpy as np
import json
from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass
from enum import Enum
import random


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


class EpsilonGreedyAgent:
    """
    Advanced Epsilon Greedy Agent for Pokemon Red
    
    This agent uses heuristic functions to evaluate the quality of different actions
    and selects the best action with probability (1-epsilon) or a random action
    with probability epsilon.
    """
    
    def __init__(self, 
                 epsilon_start: float = 0.3,
                 epsilon_min: float = 0.05,
                 epsilon_decay: float = 0.995,
                 scenario_detection_enabled: bool = True):
        
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.scenario_detection_enabled = scenario_detection_enabled
        
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
        
        # State tracking
        self.previous_positions = []
        self.action_history = []
        self.scenario_history = []
        self.step_count = 0
        
        # Objectives and targets
        self.current_objectives = self._initialize_objectives()
        
        # Performance tracking
        self.decision_times = []
        self.heuristic_scores = []
        
    def _initialize_objectives(self) -> Dict:
        """Initialize game objectives and their priorities"""
        return {
            'gym_badges': {'priority': 10, 'completed': 0, 'total': 8},
            'pokemon_caught': {'priority': 8, 'completed': 0, 'target': 10},
            'areas_explored': {'priority': 6, 'completed': 0, 'target': 50},
            'levels_gained': {'priority': 7, 'completed': 0, 'target': 100},
            'key_events': {'priority': 9, 'completed': 0, 'target': 20}
        }
    
    def detect_scenario(self, observation: Dict) -> GameScenario:
        """
        Detect the current game scenario based on observation
        """
        if not self.scenario_detection_enabled:
            return GameScenario.EXPLORATION
            
        # Check if in battle (simplified detection)
        # In actual implementation, would check specific memory addresses
        if self._is_in_battle(observation):
            return GameScenario.BATTLE
            
        # Check if stuck (repetitive movement)
        if self._is_stuck():
            return GameScenario.STUCK
            
        # Check if progressing towards key objectives
        if self._is_progressing(observation):
            return GameScenario.PROGRESSION
            
        # Check if navigating to specific target
        if self._has_clear_objective(observation):
            return GameScenario.NAVIGATION
            
        # Default to exploration
        return GameScenario.EXPLORATION
    
    def _is_in_battle(self, observation: Dict) -> bool:
        """Detect if currently in a battle"""
        # This would check specific memory addresses or screen patterns
        # For now, simplified implementation
        return False
    
    def _is_stuck(self) -> bool:
        """Detect if agent is stuck in repetitive behavior"""
        if len(self.previous_positions) < 10:
            return False
            
        # Check if last 10 positions are very similar
        recent_positions = self.previous_positions[-10:]
        unique_positions = len(set(map(tuple, recent_positions)))
        
        return unique_positions < 3
    
    def _is_progressing(self, observation: Dict) -> bool:
        """Check if making progress towards key objectives"""
        # Check badges, events, levels
        badges = np.sum(observation.get('badges', np.zeros(8)))
        events = np.sum(observation.get('events', np.zeros(100)))
        
        return badges > 0 or events > 10
    
    def _has_clear_objective(self, observation: Dict) -> bool:
        """Check if there's a clear navigation objective"""
        # Simplified: assume we have clear objective if we have some progress
        return True
    
    def calculate_heuristic_score(self, action: int, observation: Dict, scenario: GameScenario) -> float:
        """
        Calculate heuristic score for a given action in current scenario
        """
        weights = self.scenario_weights[scenario]
        score = 0.0
        
        # 1. Exploration heuristic
        exploration_score = self._calculate_exploration_score(action, observation)
        score += weights.exploration * exploration_score
        
        # 2. Objective distance heuristic
        objective_score = self._calculate_objective_score(action, observation)
        score += weights.objective_distance * objective_score
        
        # 3. Health consideration heuristic
        health_score = self._calculate_health_score(action, observation)
        score += weights.health_consideration * health_score
        
        # 4. Level progression heuristic
        level_score = self._calculate_level_score(action, observation)
        score += weights.level_progression * level_score
        
        # 5. Map familiarity heuristic
        familiarity_score = self._calculate_familiarity_score(action, observation)
        score += weights.map_familiarity * familiarity_score
        
        # 6. Event completion heuristic
        event_score = self._calculate_event_score(action, observation)
        score += weights.event_completion * event_score
        
        # 7. Scenario-specific bonuses
        scenario_bonus = self._calculate_scenario_bonus(action, observation, scenario)
        score += scenario_bonus
        
        return score
    
    def _calculate_exploration_score(self, action: int, observation: Dict) -> float:
        """Score based on exploration potential"""
        if action not in self.movement_actions:
            return 0.0
            
        # Favor actions that lead to unexplored areas
        # This would use the map information and current position
        map_data = observation.get('map', np.zeros((48, 48, 1)))
        
        # Simple heuristic: favor directions with lower values (unexplored)
        # In practice, would be more sophisticated
        return np.random.random() * 0.5  # Placeholder
    
    def _calculate_objective_score(self, action: int, observation: Dict) -> float:
        """Score based on progress towards objectives"""
        if action not in self.movement_actions:
            # Non-movement actions
            if action == 4:  # A button - interaction
                return 0.8
            elif action == 5:  # B button - cancel/back
                return 0.2
            else:  # Start button
                return 0.1
                
        # Movement actions - favor movement towards objectives
        return np.random.random() * 0.6  # Placeholder
    
    def _calculate_health_score(self, action: int, observation: Dict) -> float:
        """Score based on health considerations"""
        health = observation.get('health', np.array([1.0]))[0]
        
        if health < 0.3:
            # Low health - favor defensive actions or healing
            if action == 6:  # Start button (menu)
                return 1.0
            elif action in self.movement_actions:
                return 0.3  # Cautious movement
            else:
                return 0.5
        else:
            # Good health - normal scoring
            return 0.5
    
    def _calculate_level_score(self, action: int, observation: Dict) -> float:
        """Score based on level progression opportunities"""
        # Favor actions that might lead to battles or training
        if action == 4:  # A button - might trigger battles
            return 0.7
        elif action in self.movement_actions:
            return 0.4  # Movement can lead to encounters
        else:
            return 0.2
    
    def _calculate_familiarity_score(self, action: int, observation: Dict) -> float:
        """Score based on area familiarity"""
        # Penalty for staying in very familiar areas too long
        if len(self.previous_positions) > 5:
            # Check if current area is overly familiar
            return np.random.random() * 0.3
        return 0.5
    
    def _calculate_event_score(self, action: int, observation: Dict) -> float:
        """Score based on event progression potential"""
        events = observation.get('events', np.zeros(100))
        recent_events = np.sum(events)
        
        if action == 4:  # A button - interaction might trigger events
            return 0.9
        elif action in self.movement_actions:
            return 0.4
        else:
            return 0.2
    
    def _calculate_scenario_bonus(self, action: int, observation: Dict, scenario: GameScenario) -> float:
        """Additional scenario-specific scoring"""
        if scenario == GameScenario.BATTLE:
            # In battle, favor A button (attack) and strategic moves
            if action == 4:
                return 1.0
            return 0.1
            
        elif scenario == GameScenario.STUCK:
            # When stuck, favor random exploration
            if action in self.movement_actions:
                # Favor less recently used directions
                if len(self.action_history) > 0:
                    if action != self.action_history[-1]:
                        return 0.8
                return 0.3
            return 0.2
            
        elif scenario == GameScenario.PROGRESSION:
            # Favor interaction and forward movement
            if action == 4:
                return 0.9
            elif action == 3:  # UP - often progression direction
                return 0.6
            return 0.3
            
        return 0.0
    
    def get_action(self, observation: Dict) -> int:
        """
        Enhanced PyBoy-compatible action selection method
        """
        return self.select_action(observation)
    
    def select_action(self, observation: Dict) -> int:
        """
        Enhanced epsilon-greedy action selection with professional improvements
        """
        import time
        start_time = time.time()
        
        self.step_count += 1
        
        # Update state tracking with enhanced observation processing
        self._update_state_tracking(observation)
        
        # Detect current scenario with improved logic
        current_scenario = self.detect_scenario(observation)
        self.scenario_history.append(current_scenario)
        
        # Enhanced epsilon-greedy decision with anti-cycling
        if np.random.random() < self.epsilon:
            # Smart exploration - avoid recent actions and favor diverse exploration
            action = self._select_smart_exploration_action()
        else:
            # Enhanced greedy action based on comprehensive heuristics
            action = self._select_enhanced_greedy_action(observation, current_scenario)
        
        # Apply action filtering to prevent problematic sequences
        action = self._apply_action_filtering(action)
        
        # Update epsilon with adaptive strategy
        self._update_epsilon_adaptively(current_scenario)
        
        # Update history with size management
        self.action_history.append(action)
        if len(self.action_history) > 100:
            self.action_history.pop(0)
        
        # Track performance metrics
        decision_time = time.time() - start_time
        self.decision_times.append(decision_time)
        if len(self.decision_times) > 1000:
            self.decision_times.pop(0)
            
        return action
    
    def _update_state_tracking(self, observation: Dict):
        """Enhanced state tracking with better observation processing"""
        # Extract position information
        position = self._extract_position_from_observation(observation)
        self.previous_positions.append(position)
        if len(self.previous_positions) > 50:
            self.previous_positions.pop(0)
    
    def _extract_position_from_observation(self, observation: Dict) -> Tuple[int, int, int]:
        """Extract position from observation with multiple fallback methods"""
        try:
            # Method 1: Direct position extraction
            if 'x' in observation and 'y' in observation:
                x = int(observation['x'])
                y = int(observation['y'])
                map_id = int(observation.get('map', 0))
                return (x, y, map_id)
            
            # Method 2: Extract from screen if available
            if 'screen' in observation:
                screen = observation['screen']
                # Use screen hash as position approximation
                screen_hash = hash(screen.tobytes()) % 10000
                return (screen_hash % 100, screen_hash // 100, 0)
            
            # Method 3: Use step count as fallback
            return (self.step_count % 100, (self.step_count // 100) % 100, 0)
            
        except Exception:
            return (0, 0, 0)
    
    def _select_smart_exploration_action(self) -> int:
        """Enhanced exploration that considers action history and scenarios"""
        # Avoid actions used frequently in recent history
        recent_actions = self.action_history[-10:] if len(self.action_history) >= 10 else self.action_history
        action_frequency = {action: recent_actions.count(action) for action in self.valid_actions}
        
        # Find least used actions
        min_frequency = min(action_frequency.values()) if action_frequency else 0
        least_used_actions = [action for action, freq in action_frequency.items() if freq == min_frequency]
        
        # If stuck, prioritize movement actions
        if self._is_stuck_enhanced():
            movement_options = [action for action in least_used_actions if action in self.movement_actions]
            if movement_options:
                return np.random.choice(movement_options)
        
        # General exploration
        return np.random.choice(least_used_actions if least_used_actions else self.valid_actions)
    
    def _select_enhanced_greedy_action(self, observation: Dict, scenario: GameScenario) -> int:
        """Enhanced greedy selection with improved heuristics"""
        action_scores = {}
        
        for action in self.valid_actions:
            # Calculate comprehensive score
            base_score = self.calculate_heuristic_score(action, observation, scenario)
            
            # Add anti-cycling bonus
            cycling_penalty = self._calculate_cycling_penalty(action)
            
            # Add scenario-specific bonus
            scenario_bonus = self._calculate_enhanced_scenario_bonus(action, scenario)
            
            # Combine scores
            total_score = base_score - cycling_penalty + scenario_bonus
            action_scores[action] = total_score
        
        # Select best action
        best_action = max(action_scores.items(), key=lambda x: x[1])[0]
        
        # Store for analysis
        self.heuristic_scores.append(action_scores)
        if len(self.heuristic_scores) > 100:
            self.heuristic_scores.pop(0)
        
        return best_action
    
    def _calculate_cycling_penalty(self, action: int) -> float:
        """Calculate penalty for repetitive actions"""
        if len(self.action_history) < 5:
            return 0.0
        
        recent_actions = self.action_history[-5:]
        action_count = recent_actions.count(action)
        
        # Exponential penalty for repetition
        return action_count * 0.3
    
    def _calculate_enhanced_scenario_bonus(self, action: int, scenario: GameScenario) -> float:
        """Enhanced scenario-specific action bonuses"""
        bonuses = {
            GameScenario.EXPLORATION: {
                **{move_action: 0.5 for move_action in self.movement_actions},
                4: 0.3  # A button for interaction
            },
            GameScenario.BATTLE: {
                4: 1.0,  # A button priority in battle
                5: 0.3   # B button for escape
            },
            GameScenario.NAVIGATION: {
                **{move_action: 0.7 for move_action in self.movement_actions},
                4: 0.4
            },
            GameScenario.PROGRESSION: {
                4: 0.9,  # A button for progression
                3: 0.6,  # UP often leads to progression
                0: 0.4   # DOWN as secondary
            },
            GameScenario.STUCK: {
                **{move_action: 0.8 for move_action in self.movement_actions}
            }
        }
        
        return bonuses.get(scenario, {}).get(action, 0.0)
    
    def _apply_action_filtering(self, action: int) -> int:
        """Apply filters to prevent problematic action sequences"""
        # Prevent excessive menu actions
        if action == 6:  # START button
            recent_menu_count = sum(1 for a in self.action_history[-3:] if a == 6)
            if recent_menu_count >= 2:
                return np.random.choice(self.movement_actions)
        
        # Prevent action loops (same action repeated too many times)
        if len(self.action_history) >= 4:
            if all(a == action for a in self.action_history[-4:]):
                different_actions = [a for a in self.valid_actions if a != action]
                return np.random.choice(different_actions) if different_actions else action
        
        return action
    
    def _update_epsilon_adaptively(self, scenario: GameScenario):
        """Update epsilon with scenario-aware adaptive strategy"""
        # Base decay
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        # Scenario-specific adjustments
        if scenario == GameScenario.STUCK:
            # Increase exploration when stuck
            self.epsilon = min(0.5, self.epsilon * 1.3)
        elif scenario == GameScenario.PROGRESSION:
            # Reduce exploration when progressing well
            self.epsilon = max(self.epsilon_min, self.epsilon * 0.95)
        
        # Performance-based adjustments
        if len(self.scenario_history) >= 20:
            recent_stuck_ratio = sum(1 for s in self.scenario_history[-20:] if s == GameScenario.STUCK) / 20
            if recent_stuck_ratio > 0.3:  # Too much stuck time
                self.epsilon = min(0.6, self.epsilon * 1.2)
    
    def _is_stuck_enhanced(self) -> bool:
        """Enhanced stuck detection with multiple criteria"""
        if len(self.previous_positions) < 10:
            return False
        
        # Check position diversity
        recent_positions = self.previous_positions[-10:]
        unique_positions = len(set(recent_positions))
        position_diversity = unique_positions / len(recent_positions)
        
        # Check action diversity
        if len(self.action_history) >= 10:
            recent_actions = self.action_history[-10:]
            unique_actions = len(set(recent_actions))
            action_diversity = unique_actions / len(recent_actions)
        else:
            action_diversity = 1.0
        
        # Consider stuck if low diversity in both position and actions
        return position_diversity < 0.4 and action_diversity < 0.5
    
    def update_position(self, observation: Dict):
        """Update position tracking"""
        # Extract position from observation (simplified)
        # In real implementation, would extract x, y, map from memory
        position = [0, 0, 0]  # Placeholder
        
        self.previous_positions.append(position)
        if len(self.previous_positions) > 50:
            self.previous_positions.pop(0)
    
    def get_performance_metrics(self) -> Dict:
        """Get performance metrics for analysis"""
        return {
            'current_epsilon': self.epsilon,
            'step_count': self.step_count,
            'scenario_distribution': self._get_scenario_distribution(),
            'average_heuristic_scores': self._get_average_heuristic_scores(),
            'exploration_efficiency': self._calculate_exploration_efficiency()
        }
    
    def _get_scenario_distribution(self) -> Dict:
        """Get distribution of scenarios encountered"""
        if not self.scenario_history:
            return {}
            
        from collections import Counter
        scenario_counts = Counter(self.scenario_history)
        total = len(self.scenario_history)
        
        return {scenario.value: count/total for scenario, count in scenario_counts.items()}
    
    def _get_average_heuristic_scores(self) -> Dict:
        """Get average heuristic scores by action"""
        if not self.heuristic_scores:
            return {}
            
        action_scores = {action: [] for action in self.valid_actions}
        
        for score_dict in self.heuristic_scores:
            for action, score in score_dict.items():
                action_scores[action].append(score)
        
        return {action: np.mean(scores) if scores else 0.0 
                for action, scores in action_scores.items()}
    
    def _calculate_exploration_efficiency(self) -> float:
        """Calculate exploration efficiency metric"""
        if len(self.previous_positions) < 2:
            return 0.0
            
        unique_positions = len(set(map(tuple, self.previous_positions)))
        total_positions = len(self.previous_positions)
        
        return unique_positions / total_positions
    
    def reset(self):
        """Reset agent state for new episode"""
        self.previous_positions = []
        self.action_history = []
        self.scenario_history = []
        self.step_count = 0
        self.heuristic_scores = []
        
        # Keep epsilon for continued learning across episodes
        # self.epsilon = reset to initial value if desired