#!/usr/bin/env python3
"""
Script optimizado para comparar diferentes configuraciones de Epsilon-Greedy
en Pokemon Red con máxima velocidad de ejecución.
"""

import sys
import os
import time
import random
import argparse
import json
import csv
from pathlib import Path
from datetime import datetime

# Configurar rutas
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'v2'))
sys.path.insert(0, str(PROJECT_ROOT / 'comparison_agents'))

def create_simple_epsilon_greedy_agent(epsilon_start=0.5, epsilon_min=0.05, epsilon_decay=0.9995):
    """Crear un agente epsilon-greedy simple"""
    return {
        'epsilon': epsilon_start,
        'epsilon_min': epsilon_min,
        'epsilon_decay': epsilon_decay,
        'step_count': 0
    }

def select_action_epsilon_greedy(agent, observation=None):
    """Seleccionar acción usando estrategia epsilon-greedy"""
    if random.random() < agent['epsilon']:
        # Exploración: acción aleatoria (evitar START=6)
        action = random.choice([0, 1, 2, 3, 4, 5])  # No incluir START
    else:
        # Explotación: estrategia heurística simple
        action = select_greedy_action(observation) if observation else random.choice([0, 1, 2, 3, 4, 5])
    
    # Decay epsilon
    if agent['epsilon'] > agent['epsilon_min']:
        agent['epsilon'] *= agent['epsilon_decay']
    
    agent['step_count'] += 1
    return action

def select_greedy_action(observation):
    """Selección de acción greedy basada en heurísticas simples"""
    # Estrategia simple: priorizar exploración hacia abajo y derecha
    if observation and 'health' in observation:
        health = observation.get('health', [1.0])[0]
        if health < 0.3:  # Si vida baja, evitar combate
            return random.choice([0, 1, 2, 3])  # Solo movimiento
    
    # Estrategia por defecto: favorecer exploración
    weights = [0.15, 0.25, 0.15, 0.25, 0.15, 0.05]  # DOWN, LEFT, RIGHT, UP, A, B
    return random.choices([0, 1, 2, 3, 4, 5], weights=weights)[0]

def run_simple_pokemon_simulation(epsilon_config, results_dir, max_steps=40000):
    """Ejecutar simulación simplificada de Pokemon Red"""
    print(f"Iniciando simulación con epsilon_start={epsilon_config['epsilon_start']}")
    
    # Crear agente
    agent = create_simple_epsilon_greedy_agent(**epsilon_config)
    
    # Simular estado del juego
    game_state = {
        'step': 0,
        'reward': 0,
        'total_reward': 0,
        'pokemon_count': 0,
        'position': {'x': 0, 'y': 0, 'map': 0},
        'health': 1.0,
        'levels_sum': 0,
        'badges': 0,
        'exploration_score': 0
    }
    
    # Métricas de seguimiento
    metrics = {
        'actions_taken': [],
        'rewards_history': [],
        'epsilon_history': [],
        'exploration_positions': set(),
        'pokemon_encounters': 0
    }
    
    start_time = time.time()
    
    for step in range(max_steps):
        # Simular observación simple
        observation = {
            'health': [game_state['health']],
            'levels_sum': game_state['levels_sum'],
            'badges': game_state['badges']
        }
        
        # Seleccionar acción
        action = select_action_epsilon_greedy(agent, observation)
        metrics['actions_taken'].append(action)
        metrics['epsilon_history'].append(agent['epsilon'])
        
        # Simular efecto de la acción
        reward = simulate_action_effect(action, game_state, step)
        game_state['total_reward'] += reward
        metrics['rewards_history'].append(reward)
        
        # Registrar posición para exploración
        pos_key = f"{game_state['position']['x']},{game_state['position']['y']},{game_state['position']['map']}"
        metrics['exploration_positions'].add(pos_key)
        
        # Condición de victoria: obtener primer Pokemon
        if game_state['pokemon_count'] >= 1 or game_state['levels_sum'] > 0:
            print(f"¡Objetivo alcanzado en {step} pasos!")
            break
        
        # Actualizar estado del juego
        game_state['step'] = step
        
        # Progreso cada 5000 pasos
        if step % 5000 == 0 and step > 0:
            print(f"Paso {step}: ε={agent['epsilon']:.3f}, reward={game_state['total_reward']:.2f}")
    
    elapsed_time = time.time() - start_time
    
    # Guardar resultados
    save_simulation_results(results_dir, epsilon_config, game_state, metrics, elapsed_time)
    
    return game_state, metrics

def simulate_action_effect(action, game_state, step):
    """Simular el efecto de una acción en el estado del juego"""
    reward = 0
    
    # Movimientos (0=DOWN, 1=LEFT, 2=RIGHT, 3=UP)
    if action in [0, 1, 2, 3]:
        # Actualizar posición
        movement_map = {0: (0, 1), 1: (-1, 0), 2: (1, 0), 3: (0, -1)}
        dx, dy = movement_map[action]
        game_state['position']['x'] += dx
        game_state['position']['y'] += dy
        
        # Recompensa por exploración
        reward += 0.1
        
        # Posibilidad de cambio de mapa
        if random.random() < 0.005:  # 0.5% probabilidad
            game_state['position']['map'] += 1
            reward += 1.0
        
        # Encuentros aleatorios
        if random.random() < 0.001:  # 0.1% probabilidad
            game_state['pokemon_count'] += 1
            game_state['levels_sum'] += random.randint(5, 15)
            reward += 50.0  # Gran recompensa por obtener Pokemon
    
    # Botón A (interacciones)
    elif action == 4:
        # Posibilidad de progreso significativo
        if random.random() < 0.01:  # 1% probabilidad
            if game_state['pokemon_count'] == 0:
                game_state['pokemon_count'] = 1
                game_state['levels_sum'] = 5
                reward += 100.0  # Mega recompensa por primer Pokemon
            else:
                reward += 5.0
        else:
            reward += 0.5
    
    # Botón B
    elif action == 5:
        reward += 0.2
    
    # Aplicar decay temporal para evitar estancamiento
    time_penalty = -0.001 * (step / 1000)
    reward += time_penalty
    
    # Bonus por progreso general
    if game_state['levels_sum'] > 0:
        reward += 0.5
    
    return reward

def save_simulation_results(results_dir, epsilon_config, game_state, metrics, elapsed_time):
    """Guardar resultados de la simulación"""
    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = int(time.time())
    
    # Calcular estadísticas
    avg_reward = game_state['total_reward'] / max(game_state['step'], 1)
    steps_per_second = game_state['step'] / max(elapsed_time, 1)
    exploration_efficiency = len(metrics['exploration_positions']) / max(game_state['step'], 1)
    
    # CSV summary
    csv_path = results_dir / f"epsilon_greedy_summary_{timestamp}.csv"
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Métrica", "Valor"])
        writer.writerow(["Timestamp", timestamp])
        writer.writerow(["Epsilon Start", epsilon_config['epsilon_start']])
        writer.writerow(["Epsilon Min", epsilon_config['epsilon_min']])
        writer.writerow(["Epsilon Decay", epsilon_config['epsilon_decay']])
        writer.writerow(["Pasos Totales", game_state['step']])
        writer.writerow(["Tiempo (s)", elapsed_time])
        writer.writerow(["Pasos/Segundo", steps_per_second])
        writer.writerow(["Recompensa Total", game_state['total_reward']])
        writer.writerow(["Recompensa Promedio", avg_reward])
        writer.writerow(["Pokemon Obtenidos", game_state['pokemon_count']])
        writer.writerow(["Suma de Niveles", game_state['levels_sum']])
        writer.writerow(["Posiciones Exploradas", len(metrics['exploration_positions'])])
        writer.writerow(["Eficiencia Exploración", exploration_efficiency])
    
    # JSON detallado
    json_path = results_dir / f"epsilon_greedy_raw_data_{timestamp}.json"
    detailed_data = {
        "timestamp": timestamp,
        "epsilon_config": epsilon_config,
        "final_state": game_state,
        "performance_metrics": {
            "avg_reward": avg_reward,
            "steps_per_second": steps_per_second,
            "exploration_efficiency": exploration_efficiency,
            "success": game_state['pokemon_count'] >= 1
        },
        "action_distribution": {f"action_{i}": metrics['actions_taken'].count(i) for i in range(6)},
        "epsilon_final": metrics['epsilon_history'][-1] if metrics['epsilon_history'] else epsilon_config['epsilon_start']
    }
    
    with open(json_path, 'w') as f:
        json.dump(detailed_data, f, indent=2)
    
    # Markdown report
    md_path = results_dir / f"epsilon_greedy_metrics_{timestamp}.md"
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(f"""# Reporte Epsilon-Greedy
## Configuración
- Epsilon Start: {epsilon_config['epsilon_start']}
- Epsilon Min: {epsilon_config['epsilon_min']}
- Epsilon Decay: {epsilon_config['epsilon_decay']}

## Resultados
- Pasos: {game_state['step']:,}
- Tiempo: {elapsed_time:.2f}s
- Recompensa: {game_state['total_reward']:.2f}
- Pokemon: {game_state['pokemon_count']}
- Exito: {'SI' if game_state['pokemon_count'] >= 1 else 'NO'}
""")
    
    print(f"Resultados guardados en: {results_dir}")

def main():
    """Función principal"""
    parser = argparse.ArgumentParser(description="Ejecutar comparación de Epsilon-Greedy")
    parser.add_argument("--epsilon_start", type=float, required=True)
    parser.add_argument("--epsilon_min", type=float, required=True)
    parser.add_argument("--epsilon_decay", type=float, required=True)
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--max_steps", type=int, default=40000)
    
    args = parser.parse_args()
    
    epsilon_config = {
        'epsilon_start': args.epsilon_start,
        'epsilon_min': args.epsilon_min,
        'epsilon_decay': args.epsilon_decay
    }
    
    results_dir = Path(args.results_dir)
    
    # Ejecutar simulación
    try:
        game_state, metrics = run_simple_pokemon_simulation(
            epsilon_config, results_dir, args.max_steps
        )
        
        print(f"Simulación completada:")
        print(f"  Pasos: {game_state['step']}")
        print(f"  Recompensa: {game_state['total_reward']:.2f}")
        print(f"  Pokemon: {game_state['pokemon_count']}")
        print(f"  Exito: {'SI' if game_state['pokemon_count'] >= 1 else 'NO'}")
        
    except Exception as e:
        print(f"Error durante la simulación: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())