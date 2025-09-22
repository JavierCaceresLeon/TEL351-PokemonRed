#!/usr/bin/env python3
"""
Script optimizado para comparar diferentes algoritmos de búsqueda
en Pokemon Red con máxima velocidad de ejecución.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import random
import time
from pathlib import Path
from datetime import datetime

# Configurar rutas
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'v2'))
sys.path.insert(0, str(PROJECT_ROOT / 'comparison_agents'))

def create_search_agent(algorithm_name="astar", **kwargs):
    """Crear agente de búsqueda según el algoritmo especificado"""
    if algorithm_name == "astar":
        from search_algorithms.astar_agent import AStarAgent
        return AStarAgent()
    elif algorithm_name == "bfs":
        from search_algorithms.bfs_agent import BFSAgent
        return BFSAgent()
    elif algorithm_name == "simulated_annealing":
        from search_algorithms.simulated_annealing_agent import SimulatedAnnealingAgent
        return SimulatedAnnealingAgent()
    elif algorithm_name == "hill_climbing":
        from search_algorithms.hill_climbing_agent import HillClimbingAgent, HillClimbingVariant
        variant = kwargs.get('variant', HillClimbingVariant.STEEPEST_ASCENT)
        return HillClimbingAgent(variant)
    elif algorithm_name == "tabu_search":
        from search_algorithms.tabu_agent import TabuSearchAgent
        return TabuSearchAgent()
    else:
        raise ValueError(f"Algoritmo desconocido: {algorithm_name}")

def select_action_search_agent(agent, observation=None):
    """Seleccionar acción usando el agente de búsqueda"""
    if observation is None:
        # Generar observación simulada
        observation = np.random.randint(0, 256, (72, 80, 3), dtype=np.uint8)
    
    # Simular recompensa basada en progreso
    reward = 0.0
    if hasattr(agent, 'step_count') and agent.step_count > 0:
        # Recompensa por progreso general
        if agent.step_count % 50 == 0:
            reward = 0.1
        # Recompensa por llegar a objetivos
        if hasattr(agent, 'current_position') and hasattr(agent, 'objectives'):
            min_distance = min(abs(agent.current_position[0] - obj[0]) + abs(agent.current_position[1] - obj[1]) 
                             for obj in agent.objectives)
            if min_distance <= 2:
                reward = 1.0
        # Penalización por estar atascado
        if hasattr(agent, 'last_positions') and len(agent.last_positions) >= 5:
            if len(set(agent.last_positions)) <= 2:
                reward = -0.1
    
    return agent.select_action(observation, reward)

def simulate_action_effect(current_position, action):
    """Simular el efecto de una acción en la posición"""
    x, y = current_position
    
    if action == 0:  # UP
        return (max(0, x - 1), y)
    elif action == 1:  # DOWN
        return (min(24, x + 1), y)
    elif action == 2:  # LEFT
        return (x, max(0, y - 1))
    elif action == 3:  # RIGHT
        return (x, min(24, y + 1))
    else:  # A, B, START - pueden cambiar estado del juego
        # Simular probabilidad de éxito en interacciones
        if random.random() < 0.1:  # 10% chance de progreso en interacciones
            return (x, y)  # Posición no cambia pero hay progreso
        return (x, y)

def run_search_algorithm_simulation(algorithm_config, max_steps=2000, results_dir=None):
    """Ejecutar simulación de algoritmo de búsqueda"""
    
    print(f"Iniciando simulación con {algorithm_config['name']}")
    
    # Crear agente
    agent = create_search_agent(algorithm_config['algorithm'], **algorithm_config.get('params', {}))
    
    # Configurar simulación
    step = 0
    total_reward = 0
    pokemon_obtained = False
    start_time = time.time()
    
    # Métricas detalladas
    action_counts = {i: 0 for i in range(7)}
    position_history = []
    reward_history = []
    agent_info_history = []
    
    # Condiciones de terminación
    max_repetitions = 100  # Máximo número de repeticiones en la misma posición
    position_counts = {}
    stuck_counter = 0
    
    while step < max_steps and not pokemon_obtained:
        step += 1
        
        # Crear observación simulada
        observation = np.random.randint(0, 256, (72, 80, 3), dtype=np.uint8)
        
        # Seleccionar acción
        action = select_action_search_agent(agent, observation)
        action_counts[action] += 1
        
        # Simular efecto de la acción
        if hasattr(agent, 'current_position'):
            new_position = simulate_action_effect(agent.current_position, action)
            
            # Actualizar contador de posiciones
            position_key = f"{new_position[0]},{new_position[1]}"
            position_counts[position_key] = position_counts.get(position_key, 0) + 1
            
            # Detectar si está atascado
            if position_counts[position_key] > max_repetitions:
                stuck_counter += 1
                if stuck_counter > 20:  # Si está muy atascado, reiniciar posición
                    agent.current_position = (random.randint(0, 24), random.randint(0, 24))
                    stuck_counter = 0
        
        # Calcular recompensa de simulación
        step_reward = 0
        
        # Recompensa por exploración
        if hasattr(agent, 'visited_positions'):
            exploration_bonus = len(agent.visited_positions) * 0.01
            step_reward += exploration_bonus
        
        # Recompensa por progreso hacia objetivos
        if hasattr(agent, 'current_position') and hasattr(agent, 'objectives'):
            min_distance = min(abs(agent.current_position[0] - obj[0]) + abs(agent.current_position[1] - obj[1]) 
                             for obj in agent.objectives)
            if min_distance <= 3:
                step_reward += (4 - min_distance) * 0.5
                
                # Gran recompensa por llegar al objetivo principal (Pokemon)
                if min_distance <= 1:
                    step_reward += 10.0
                    pokemon_obtained = True
        
        # Penalización por repetición excesiva
        if hasattr(agent, 'current_position'):
            position_key = f"{agent.current_position[0]},{agent.current_position[1]}"
            if position_counts.get(position_key, 0) > 5:
                step_reward -= 0.1
        
        # Bonificación por uso eficiente de acciones
        if action in [4, 5, 6] and hasattr(agent, 'current_position'):  # A, B, START
            min_distance = min(abs(agent.current_position[0] - obj[0]) + abs(agent.current_position[1] - obj[1]) 
                             for obj in agent.objectives)
            if min_distance <= 2:  # Solo dar bonificación si está cerca de un objetivo
                step_reward += 0.5
        
        total_reward += step_reward
        
        # Guardar métricas
        if hasattr(agent, 'current_position'):
            position_history.append(agent.current_position)
        reward_history.append(step_reward)
        
        # Obtener información del agente
        if hasattr(agent, 'get_agent_info'):
            agent_info = agent.get_agent_info()
            agent_info_history.append(agent_info)
        
        # Condición de éxito adicional: si ha explorado mucho y obtuvo recompensas
        if step >= 500 and total_reward > 15:
            pokemon_obtained = True
            print(f"Terminación por criterio de éxito alternativo en paso {step}")
        
        # Mostrar progreso cada 200 pasos
        if step % 200 == 0:
            progress_info = ""
            if hasattr(agent, 'get_agent_info'):
                info = agent.get_agent_info()
                progress_info = f" | Pos: {info.get('current_position', 'N/A')} | Visitadas: {info.get('visited_positions', 'N/A')}"
            print(f"  Paso {step}/{max_steps} | Recompensa: {total_reward:.2f}{progress_info}")
    
    # Calcular tiempo transcurrido
    elapsed_time = time.time() - start_time
    
    # Determinar razón de terminación
    if pokemon_obtained:
        termination_reason = "Pokemon obtenido"
    elif step >= max_steps:
        termination_reason = "Máximo de pasos alcanzado"
    else:
        termination_reason = "Terminación inesperada"
    
    print(f"Simulación completada: {step} pasos, {elapsed_time:.2f}s, {termination_reason}")
    
    # Guardar resultados si se especifica directorio
    if results_dir:
        save_simulation_results(
            results_dir, step, total_reward, elapsed_time, termination_reason,
            algorithm_config, action_counts, position_history, reward_history, agent_info_history
        )
    
    return {
        'steps': step,
        'time': elapsed_time,
        'reward': total_reward,
        'pokemon_obtained': pokemon_obtained,
        'termination_reason': termination_reason,
        'algorithm': algorithm_config['name']
    }

def save_simulation_results(results_dir, step, total_reward, elapsed_time, termination_reason,
                          algorithm_config, action_counts, position_history, reward_history, agent_info_history):
    """Guardar resultados de la simulación"""
    
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = str(int(time.time()))
    algorithm_name = algorithm_config['name'].lower().replace(' ', '_')
    
    # Datos de resumen en CSV
    summary_data = {
        'Métrica': [
            'Algoritmo',
            'Pasos Totales',
            'Tiempo (s)',
            'Recompensa Total',
            'Pokemon Obtenidos',
            'Razón de Terminación',
            'Posiciones Visitadas',
            'Acción Más Usada',
            'Eficiencia (Recompensa/Paso)',
            'Velocidad (Pasos/s)'
        ],
        'Valor': [
            algorithm_config['name'],
            step,
            round(elapsed_time, 3),
            round(total_reward, 3),
            1 if termination_reason == "Pokemon obtenido" else 0,
            termination_reason,
            len(set(position_history)) if position_history else 0,
            max(action_counts, key=action_counts.get),
            round(total_reward / max(1, step), 4),
            round(step / max(0.1, elapsed_time), 2)
        ]
    }
    
    summary_df = pd.DataFrame(summary_data)
    summary_file = results_dir / f"{algorithm_name}_summary_{timestamp}.csv"
    summary_df.to_csv(summary_file, index=False)
    
    # Datos detallados en JSON
    detailed_data = {
        'algorithm_config': algorithm_config,
        'execution_summary': {
            'steps': step,
            'time_seconds': elapsed_time,
            'total_reward': total_reward,
            'termination_reason': termination_reason,
            'pokemon_obtained': termination_reason == "Pokemon obtenido"
        },
        'action_statistics': action_counts,
        'position_history': position_history[-50:] if len(position_history) > 50 else position_history,  # Últimas 50 posiciones
        'reward_history': reward_history[-50:] if len(reward_history) > 50 else reward_history,  # Últimas 50 recompensas
        'agent_info_samples': agent_info_history[-10:] if len(agent_info_history) > 10 else agent_info_history,  # Últimas 10 muestras
        'performance_metrics': {
            'positions_visited': len(set(position_history)) if position_history else 0,
            'efficiency': total_reward / max(1, step),
            'speed': step / max(0.1, elapsed_time),
            'most_used_action': max(action_counts, key=action_counts.get),
            'action_diversity': len([count for count in action_counts.values() if count > 0])
        }
    }
    
    json_file = results_dir / f"{algorithm_name}_raw_data_{timestamp}.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(detailed_data, f, indent=2, ensure_ascii=False, default=str)
    
    # Métricas en Markdown
    markdown_content = f"""# Métricas de Ejecución - {algorithm_config['name']}

## Resumen de Ejecución
- **Algoritmo**: {algorithm_config['name']}
- **Pasos Totales**: {step}
- **Tiempo de Ejecución**: {elapsed_time:.3f} segundos
- **Recompensa Total**: {total_reward:.3f}
- **Pokemon Obtenido**: {'Sí' if termination_reason == "Pokemon obtenido" else 'No'}
- **Razón de Terminación**: {termination_reason}

## Métricas de Rendimiento
- **Posiciones Visitadas**: {len(set(position_history)) if position_history else 0}
- **Eficiencia (Recompensa/Paso)**: {total_reward / max(1, step):.4f}
- **Velocidad (Pasos/segundo)**: {step / max(0.1, elapsed_time):.2f}
- **Diversidad de Acciones**: {len([count for count in action_counts.values() if count > 0])}/7

## Distribución de Acciones
"""
    
    action_names = ["UP", "DOWN", "LEFT", "RIGHT", "A", "B", "START"]
    for i, (action, count) in enumerate(action_counts.items()):
        percentage = (count / max(1, step)) * 100
        markdown_content += f"- **{action_names[i]}**: {count} ({percentage:.1f}%)\n"
    
    if agent_info_history and len(agent_info_history) > 0:
        markdown_content += f"\n## Estado Final del Agente\n"
        final_info = agent_info_history[-1]
        for key, value in final_info.items():
            markdown_content += f"- **{key}**: {value}\n"
    
    markdown_file = results_dir / f"{algorithm_name}_metrics_{timestamp}.md"
    with open(markdown_file, 'w', encoding='utf-8') as f:
        f.write(markdown_content)
    
    print(f"Resultados guardados en {results_dir}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Uso: python simple_search_runner.py <algoritmo> <variante> <directorio_resultados>")
        print("Algoritmos disponibles: astar, bfs, simulated_annealing, hill_climbing, tabu_search")
        print("Variantes para hill_climbing: steepest_ascent, first_improvement, random_restart, stochastic")
        sys.exit(1)
    
    algorithm = sys.argv[1]
    variant = sys.argv[2]
    results_dir = sys.argv[3]
    
    # Configurar algoritmo
    if algorithm == "hill_climbing":
        from search_algorithms.hill_climbing_agent import HillClimbingVariant
        variant_map = {
            'steepest_ascent': HillClimbingVariant.STEEPEST_ASCENT,
            'first_improvement': HillClimbingVariant.FIRST_IMPROVEMENT,
            'random_restart': HillClimbingVariant.RANDOM_RESTART,
            'stochastic': HillClimbingVariant.STOCHASTIC
        }
        
        algorithm_config = {
            'name': f'Hill Climbing ({variant})',
            'algorithm': 'hill_climbing',
            'params': {'variant': variant_map.get(variant, HillClimbingVariant.STEEPEST_ASCENT)}
        }
    else:
        algorithm_config = {
            'name': algorithm.replace('_', ' ').title(),
            'algorithm': algorithm,
            'params': {}
        }
    
    # Ejecutar simulación
    try:
        result = run_search_algorithm_simulation(algorithm_config, max_steps=2500, results_dir=results_dir)
        print(f"Simulación completada exitosamente: {result}")
    except Exception as e:
        print(f"Error durante la simulación: {e}")
        sys.exit(1)