"""
Implementación del algoritmo Tabú Search para Pokémon Red
Objetivo: Encontrar una solución para salir de la habitación inicial usando búsqueda local
"""

import time
import random
from typing import List, Tuple, Dict, Set
import numpy as np
from collections import deque

class TabuSearchAgent:
    """Agente que usa Tabú Search para navegar"""
    
    def __init__(self, environment, max_iterations=1000, tabu_size=50):
        self.env = environment
        self.max_iterations = max_iterations
        self.tabu_size = tabu_size
        self.tabu_list = deque(maxlen=tabu_size)
        self.search_stats = {
            'iterations': 0,
            'search_time': 0,
            'best_solution_length': 0,
            'success': False,
            'best_fitness': float('-inf')
        }
        
    def get_state_key(self, state):
        """Generar clave única para el estado"""
        pos = state['position']
        return (pos[0], pos[1], pos[2])
    
    def evaluate_fitness(self, state):
        """
        Función de evaluación para Tabú Search
        Mayor fitness = mejor estado
        """
        current_pos = state['position']
        
        # Si salió de la habitación inicial, fitness muy alto
        if self.env.initial_position and current_pos[2] != self.env.initial_position[2]:
            return 1000
        
        # Fitness basado en distancia a los bordes (posibles salidas)
        # Mientras más cerca del borde, mejor fitness
        distance_to_edges = [
            current_pos[0],      # Distancia al borde izquierdo
            10 - current_pos[0], # Distancia al borde derecho  
            current_pos[1],      # Distancia al borde superior
            10 - current_pos[1]  # Distancia al borde inferior
        ]
        
        min_distance_to_edge = min(distance_to_edges)
        
        # Fitness inverso a la distancia mínima al borde
        fitness = 100 - min_distance_to_edge
        
        # Bonificación por exploración (alejarse del centro)
        center_x, center_y = 5, 5
        distance_from_center = abs(current_pos[0] - center_x) + abs(current_pos[1] - center_y)
        fitness += distance_from_center * 5
        
        return fitness
    
    def get_neighbors(self, current_solution):
        """
        Generar soluciones vecinas modificando la solución actual
        """
        neighbors = []
        
        # Tipo 1: Añadir una acción aleatoria al final
        for _ in range(5):  # Generar 5 vecinos de este tipo
            new_solution = current_solution.copy()
            random_action = random.randint(0, len(self.env.actions) - 1)
            new_solution.append(random_action)
            neighbors.append(new_solution)
        
        # Tipo 2: Modificar una acción existente (si la solución no está vacía)
        if current_solution:
            for i in range(min(len(current_solution), 3)):  # Modificar hasta 3 posiciones
                new_solution = current_solution.copy()
                new_solution[i] = random.randint(0, len(self.env.actions) - 1)
                neighbors.append(new_solution)
        
        # Tipo 3: Eliminar la última acción (si hay más de 1)
        if len(current_solution) > 1:
            new_solution = current_solution[:-1]
            neighbors.append(new_solution)
        
        return neighbors
    
    def evaluate_solution(self, solution):
        """Evaluar una solución ejecutándola en el entorno"""
        self.env.reset()
        
        max_fitness = float('-inf')
        final_state = None
        success = False
        
        try:
            for action in solution:
                state, reward, done = self.env.step(action)
                fitness = self.evaluate_fitness(state)
                
                if fitness > max_fitness:
                    max_fitness = fitness
                    final_state = state
                
                if done:
                    if reward > 50:  # Salió de la habitación
                        success = True
                    break
                    
        except Exception as e:
            print(f"Error evaluando solución: {e}")
            return float('-inf'), None, False
        
        return max_fitness, final_state, success
    
    def is_tabu(self, solution):
        """Verificar si una solución está en la lista tabú"""
        # Para simplificar, consideramos tabú las últimas N acciones
        if len(solution) > 0:
            last_actions = tuple(solution[-min(3, len(solution)):])
            return last_actions in self.tabu_list
        return False
    
    def add_to_tabu(self, solution):
        """Añadir solución a la lista tabú"""
        if len(solution) > 0:
            last_actions = tuple(solution[-min(3, len(solution)):])
            self.tabu_list.append(last_actions)
    
    def search(self):
        """Ejecutar Tabú Search"""
        start_time = time.time()
        
        # Solución inicial: secuencia vacía
        current_solution = []
        best_solution = []
        
        current_fitness, _, _ = self.evaluate_solution(current_solution)
        best_fitness = current_fitness
        
        self.search_stats['iterations'] = 0
        
        for iteration in range(self.max_iterations):
            self.search_stats['iterations'] = iteration + 1
            
            # Generar vecinos
            neighbors = self.get_neighbors(current_solution)
            
            # Evaluar vecinos y encontrar el mejor no-tabú
            best_neighbor = None
            best_neighbor_fitness = float('-inf')
            best_neighbor_success = False
            
            for neighbor in neighbors:
                # Limitar longitud de solución para evitar explosión
                if len(neighbor) > 50:
                    continue
                
                if not self.is_tabu(neighbor):
                    fitness, state, success = self.evaluate_solution(neighbor)
                    
                    if success:
                        # Si encontramos una solución exitosa, terminar
                        self.search_stats['search_time'] = time.time() - start_time
                        self.search_stats['success'] = True
                        self.search_stats['best_solution_length'] = len(neighbor)
                        self.search_stats['best_fitness'] = fitness
                        return neighbor
                    
                    if fitness > best_neighbor_fitness:
                        best_neighbor = neighbor
                        best_neighbor_fitness = fitness
                        best_neighbor_success = success
            
            # Si no hay vecinos válidos, generar solución aleatoria
            if best_neighbor is None:
                best_neighbor = [random.randint(0, len(self.env.actions) - 1) 
                               for _ in range(random.randint(1, 10))]
                best_neighbor_fitness, _, best_neighbor_success = self.evaluate_solution(best_neighbor)
            
            # Actualizar solución actual
            self.add_to_tabu(current_solution)
            current_solution = best_neighbor
            current_fitness = best_neighbor_fitness
            
            # Actualizar mejor solución
            if current_fitness > best_fitness:
                best_solution = current_solution.copy()
                best_fitness = current_fitness
            
            # Criterio de parada temprana si la fitness es muy alta
            if current_fitness > 500:
                break
        
        self.search_stats['search_time'] = time.time() - start_time
        self.search_stats['best_solution_length'] = len(best_solution)
        self.search_stats['best_fitness'] = best_fitness
        
        return best_solution
    
    def execute_plan(self, plan):
        """Ejecutar un plan de acciones"""
        self.env.reset()
        
        results = {
            'steps': len(plan),
            'success': False,
            'final_state': None,
            'execution_time': 0,
            'max_fitness': float('-inf')
        }
        
        start_time = time.time()
        
        for i, action in enumerate(plan):
            try:
                state, reward, done = self.env.step(action)
                fitness = self.evaluate_fitness(state)
                
                if fitness > results['max_fitness']:
                    results['max_fitness'] = fitness
                
                if done:
                    results['success'] = reward > 50  # Salió de la habitación
                    results['final_state'] = state
                    break
                    
            except Exception as e:
                print(f"Error ejecutando acción {action}: {e}")
                break
        
        results['execution_time'] = time.time() - start_time
        return results
    
    def get_stats(self):
        """Obtener estadísticas de la búsqueda"""
        return self.search_stats.copy()
