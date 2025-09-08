"""
Paquete de comparación de agentes para Pokémon Red
Compara diferentes enfoques para salir de la habitación inicial
"""

__version__ = "1.0.0"
__author__ = "TEL351 Project"
__description__ = "Comparación entre agente entrenado (v2) y algoritmos de búsqueda clásicos"

# Importaciones principales
try:
    from .config import ComparisonConfig, QuickTestConfig, IntensiveTestConfig, DebugConfig
    from .search_env import SearchEnvironment
    
    # Algoritmos de búsqueda
    from .search_algorithms.astar_agent import AStarAgent
    from .search_algorithms.tabu_agent import TabuSearchAgent
    
    # Agente entrenado (opcional si no hay stable_baselines3)
    try:
        from .v2_agent import V2TrainedAgent
    except ImportError:
        V2TrainedAgent = None
    
    __all__ = [
        'ComparisonConfig',
        'QuickTestConfig', 
        'IntensiveTestConfig',
        'DebugConfig',
        'SearchEnvironment',
        'AStarAgent',
        'TabuSearchAgent',
        'V2TrainedAgent'
    ]
    
except ImportError as e:
    # Si hay errores de importación, definir una lista mínima
    __all__ = []
    print(f"Warning: Some modules could not be imported: {e}")
