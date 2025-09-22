"""
Search Algorithms Package for Pokemon Red Environment
====================================================

This package contains various search algorithms implemented for the Pokemon Red environment:
- A* Search: Optimal pathfinding with heuristics
- Breadth-First Search: Systematic level-by-level exploration
- Simulated Annealing: Stochastic optimization with temperature cooling
- Hill Climbing: Local optimization with plateau escape mechanisms
- Tabu Search: Memory-based search with tabu list
"""

from .astar_agent import AStarAgent
from .bfs_agent import BFSAgent
from .simulated_annealing_agent import SimulatedAnnealingAgent
from .hill_climbing_agent import HillClimbingAgent, HillClimbingVariant
from .tabu_agent import TabuSearchAgent

__all__ = [
    'AStarAgent',
    'BFSAgent', 
    'SimulatedAnnealingAgent',
    'HillClimbingAgent',
    'HillClimbingVariant',
    'TabuSearchAgent'
]