"""
Wrapper para reducir el espacio de acciones solo a acciones válidas en batalla
"""
import gymnasium as gym
import numpy as np

class BattleOnlyActions(gym.ActionWrapper):
    """
    Reduce el espacio de acciones a solo las relevantes en batalla.
    
    Acciones originales (0-8):
    0: A (confirmar/atacar)
    1: B (cancelar/retroceder)
    2: UP
    3: DOWN
    4: LEFT
    5: RIGHT
    6: START
    7: SELECT
    8: NO_OP
    
    Acciones en batalla (reducidas 0-2):
    0: A (atacar/confirmar)
    1: UP (navegar menú hacia arriba)
    2: DOWN (navegar menú hacia abajo)
    """
    
    def __init__(self, env):
        super().__init__(env)
        
        # Mapeo: acción reducida -> acción original
        self.action_map = {
            0: 0,  # A
            1: 2,  # UP
            2: 3,  # DOWN
        }
        
        # Nuevo espacio de acciones reducido
        self.action_space = gym.spaces.Discrete(3)

    def __getattr__(self, name):
        """Forward attribute access to the wrapped env so wrappers can access
        attributes like `pyboy` or `memory`.
        """
        return getattr(self.env, name)
    
    def action(self, action):
        """Convierte acción reducida a acción original"""
        # Convert numpy array to int if needed
        if hasattr(action, 'item'):
            action = action.item()
        return self.action_map[int(action)]
