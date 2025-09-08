"""
Implementación del algoritmo A* para Pokémon Red
Objetivo: Encontrar el camino óptimo para salir de la habitación inicial
"""

import heapq
import time
from typing import List, Tuple, Dict, Set
import numpy as np
from collections import defaultdict

class Node:
    """Nodo para el algoritmo A*"""
    
    def __init__(self, state, parent=None, action=None, g_cost=0, h_cost=0):
        self.state = state
        self.parent = parent
        self.action = action
        self.g_cost = g_cost  # Costo desde el inicio
        self.h_cost = h_cost  # Heurística
        self.f_cost = g_cost + h_cost  # Costo total
        
    def __lt__(self, other):
        return self.f_cost < other.f_cost
    
    def __eq__(self, other):
        if not isinstance(other, Node):
            return False
        return self.get_state_key() == other.get_state_key()
    
    def __hash__(self):
        return hash(self.get_state_key())
    
    def get_state_key(self):
        """Generar clave única para el estado"""
        pos = self.state['position']
        return (pos[0], pos[1], pos[2])

class AStarAgent:
    """Agente que usa A* para navegar"""
    
    def __init__(self, environment, max_search_depth=1000):
        self.env = environment
        self.max_search_depth = max_search_depth
        self.search_stats = {
            'nodes_explored': 0,
            'search_time': 0,
            'path_length': 0,
            'success': False
        }
        
    def heuristic(self, state):
        """
        Función heurística para A*
        Estima la distancia al objetivo (salir de la habitación)
        """
        current_pos = state['position']
        
        # Si ya salió de la habitación inicial, heurística = 0
        if self.env.initial_position and current_pos[2] != self.env.initial_position[2]:
            return 0
        
        # Heurística basada en distancia a la puerta (esquina de la habitación)
        # Asumiendo que las puertas están en los bordes
        room_center_x, room_center_y = 5, 5  # Centro aproximado de la habitación inicial
        
        # Distancia Manhattan a los bordes de la habitación
        distance_to_edge = min(
            abs(current_pos[0] - 0),  # Borde izquierdo
            abs(current_pos[0] - 10), # Borde derecho
            abs(current_pos[1] - 0),  # Borde superior
            abs(current_pos[1] - 10)  # Borde inferior
        )
        
        return distance_to_edge
    
    def get_neighbors(self, node):
        """Obtener estados vecinos válidos"""
        neighbors = []
        valid_actions = self.env.get_valid_actions(node.state)
        
        # Guardar estado actual
        current_state = self.env.get_state()
        
        for action in valid_actions:
            try:
                # Probar la acción
                new_state, reward, done = self.env.step(action)
                
                # Crear nuevo nodo
                g_cost = node.g_cost + 1  # Cada paso cuesta 1
                h_cost = self.heuristic(new_state)
                
                neighbor = Node(
                    state=new_state,
                    parent=node,
                    action=action,
                    g_cost=g_cost,
                    h_cost=h_cost
                )
                
                neighbors.append(neighbor)
                
            except Exception as e:
                print(f"Error al probar acción {action}: {e}")
                continue
        
        return neighbors
    
    def search(self):
        """Ejecutar búsqueda A*"""
        start_time = time.time()
        
        # Reset del entorno
        initial_state = self.env.reset()
        
        # Nodo inicial
        start_node = Node(
            state=initial_state,
            g_cost=0,
            h_cost=self.heuristic(initial_state)
        )
        
        # Estructuras de datos para A*
        open_set = [start_node]
        closed_set: Set[Node] = set()
        g_scores = defaultdict(lambda: float('inf'))
        g_scores[start_node.get_state_key()] = 0
        
        self.search_stats['nodes_explored'] = 0
        
        while open_set and self.search_stats['nodes_explored'] < self.max_search_depth:
            # Obtener nodo con menor f_cost
            current_node = heapq.heappop(open_set)
            
            # Verificar si llegamos al objetivo
            if self.is_goal(current_node.state):
                self.search_stats['search_time'] = time.time() - start_time
                self.search_stats['success'] = True
                self.search_stats['path_length'] = current_node.g_cost
                return self.reconstruct_path(current_node)
            
            # Añadir a closed_set
            closed_set.add(current_node)
            self.search_stats['nodes_explored'] += 1
            
            # Explorar vecinos
            neighbors = self.get_neighbors(current_node)
            
            for neighbor in neighbors:
                if neighbor in closed_set:
                    continue
                
                state_key = neighbor.get_state_key()
                tentative_g = current_node.g_cost + 1
                
                # Si encontramos un camino mejor
                if tentative_g < g_scores[state_key]:
                    g_scores[state_key] = tentative_g
                    neighbor.g_cost = tentative_g
                    neighbor.f_cost = tentative_g + neighbor.h_cost
                    neighbor.parent = current_node
                    
                    # Añadir a open_set si no está
                    if neighbor not in open_set:
                        heapq.heappush(open_set, neighbor)
        
        # No se encontró solución
        self.search_stats['search_time'] = time.time() - start_time
        self.search_stats['success'] = False
        return []
    
    def is_goal(self, state):
        """Verificar si el estado es el objetivo"""
        return self.env.is_done(state) and self.env.calculate_reward(state) > 50
    
    def reconstruct_path(self, node):
        """Reconstruir el camino desde el nodo objetivo"""
        path = []
        current = node
        
        while current.parent is not None:
            path.append(current.action)
            current = current.parent
        
        path.reverse()
        return path
    
    def execute_plan(self, plan):
        """Ejecutar un plan de acciones"""
        self.env.reset()
        
        results = {
            'steps': len(plan),
            'success': False,
            'final_state': None,
            'execution_time': 0
        }
        
        start_time = time.time()
        
        for i, action in enumerate(plan):
            state, reward, done = self.env.step(action)
            
            if done:
                results['success'] = reward > 50  # Salió de la habitación
                results['final_state'] = state
                break
        
        results['execution_time'] = time.time() - start_time
        return results
    
    def get_stats(self):
        """Obtener estadísticas de la búsqueda"""
        return self.search_stats.copy()
