"""
Entorno simplificado para algoritmos de búsqueda clásicos
Basado en red_gym_env_v2.py pero adaptado para búsqueda
"""

import uuid
import json
import time
from pathlib import Path
import numpy as np
from pyboy import PyBoy
from pyboy.utils import WindowEvent
import sys
import os

# Agregar el directorio v2 al path para importar dependencias
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'v2'))
from global_map import local_to_global, GLOBAL_MAP_SHAPE

class SearchEnvironment:
    """
    Entorno simplificado para algoritmos de búsqueda
    Objetivo: Salir de la habitación inicial de Pokémon Red
    """
    
    def __init__(self, config=None):
        self.init_state = config.get("init_state", "../init.state")
        self.gb_path = config.get("gb_path", "../PokemonRed.gb")
        self.headless = config.get("headless", True)
        self.max_steps = config.get("max_steps", 1000)
        self.session_path = config.get("session_path", Path("search_session"))
        
        # Inicializar PyBoy
        self.pyboy = PyBoy(
            self.gb_path,
            window="null" if self.headless else "SDL2"
        )
        
        # Cargar estado inicial
        if self.init_state:
            with open(self.init_state, 'rb') as f:
                self.pyboy.load_state(f)
        
        self.screen = self.pyboy.screen
        
        # Definir acciones posibles
        self.actions = [
            WindowEvent.PRESS_BUTTON_A,
            WindowEvent.PRESS_BUTTON_B, 
            WindowEvent.PRESS_ARROW_DOWN,
            WindowEvent.PRESS_ARROW_LEFT,
            WindowEvent.PRESS_ARROW_RIGHT,
            WindowEvent.PRESS_ARROW_UP,
            WindowEvent.RELEASE_BUTTON_A,
            WindowEvent.RELEASE_BUTTON_B,
            WindowEvent.RELEASE_ARROW_DOWN,
            WindowEvent.RELEASE_ARROW_LEFT,
            WindowEvent.RELEASE_ARROW_RIGHT,
            WindowEvent.RELEASE_ARROW_UP
        ]
        
        # Variables de estado
        self.step_count = 0
        self.initial_position = None
        self.session_path.mkdir(exist_ok=True)
        
    def reset(self):
        """Reset del entorno"""
        self.step_count = 0
        if self.init_state:
            with open(self.init_state, 'rb') as f:
                self.pyboy.load_state(f)
        
        # Obtener posición inicial
        self.initial_position = self.get_player_position()
        return self.get_state()
    
    def get_player_position(self):
        """Obtener posición del jugador"""
        player_x = self.pyboy.memory[0xD362]
        player_y = self.pyboy.memory[0xD361]
        map_n = self.pyboy.memory[0xD35E]
        return (player_x, player_y, map_n)
    
    def get_state(self):
        """Obtener estado actual del juego"""
        position = self.get_player_position()
        
        # Obtener screen como estado
        screen_obs = self.screen.ndarray
        
        return {
            'position': position,
            'screen': screen_obs,
            'step_count': self.step_count,
            'map_id': position[2]
        }
    
    def step(self, action_idx):
        """Ejecutar una acción"""
        if action_idx < 0 or action_idx >= len(self.actions):
            raise ValueError(f"Acción inválida: {action_idx}")
        
        action = self.actions[action_idx]
        
        # Ejecutar acción
        self.pyboy.send_input(action)
        
        # Avanzar frames
        for _ in range(24):  # action_freq similar al v2
            self.pyboy.tick()
        
        self.step_count += 1
        
        # Obtener nuevo estado
        new_state = self.get_state()
        
        # Calcular recompensa
        reward = self.calculate_reward(new_state)
        
        # Verificar si terminó
        done = self.is_done(new_state)
        
        return new_state, reward, done
    
    def calculate_reward(self, state):
        """Calcular recompensa basada en el progreso"""
        current_pos = state['position']
        
        # Recompensa por salir de la habitación inicial (map_id diferente)
        if self.initial_position and current_pos[2] != self.initial_position[2]:
            return 100.0  # Gran recompensa por salir de la habitación
        
        # Recompensa por moverse (exploración)
        if self.initial_position:
            distance = abs(current_pos[0] - self.initial_position[0]) + abs(current_pos[1] - self.initial_position[1])
            return distance * 0.1
        
        return -0.01  # Penalización pequeña por paso
    
    def is_done(self, state):
        """Verificar si el episodio terminó"""
        current_pos = state['position']
        
        # Terminó si salió de la habitación inicial
        if self.initial_position and current_pos[2] != self.initial_position[2]:
            return True
        
        # Terminó si alcanzó max_steps
        if self.step_count >= self.max_steps:
            return True
        
        return False
    
    def get_valid_actions(self, state):
        """Obtener acciones válidas desde el estado actual"""
        # Por simplicidad, todas las acciones son válidas
        # En una implementación más sofisticada, se podrían filtrar
        return list(range(len(self.actions)))
    
    def close(self):
        """Cerrar el entorno"""
        if hasattr(self, 'pyboy'):
            self.pyboy.stop()
