"""
Interactive A* Search Agent for Pokemon Red
==========================================

Run the A* search agent interactively with real-time visualization and control.
This script provides intelligent pathfinding and goal-directed exploration.
"""

import sys
import os
import numpy as np
import time
from pathlib import Path

# Add the comparison_agents directory to the path
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

# Import necessary modules
from config import get_v2_config
from v2_astar_agent import V2AStarAgent


def main():
    """Main function to run the A* agent interactively"""
    
    print("Interactive A* Search Agent for Pokemon Red")
    print("Intelligent pathfinding with goal-directed exploration")
    print("Press Ctrl+C to stop at any time")
    print()
    
    # Configuration for A* agent
    env_config = get_v2_config()
    
    # A* specific configuration
    agent_config = {
        'exploration_bonus': 0.2,
        'goal_reward_bonus': 2.0,
        'stuck_threshold': 50,
        'path_planning_interval': 10,
        'max_path_length': 100,
        'heuristic_weight': 1.5
    }
    
    print("A* Agent Configuration:")
    print(f"   Exploration Bonus: {agent_config['exploration_bonus']}")
    print(f"   Goal Reward Bonus: {agent_config['goal_reward_bonus']}")
    print(f"   Stuck Threshold: {agent_config['stuck_threshold']}")
    print(f"   Heuristic Weight: {agent_config['heuristic_weight']}")
    print(f"   Max Path Length: {agent_config['max_path_length']}")
    print()
    
    try:
        # Initialize A* agent
        print("Initializing A* Search Agent...")
        agent = V2AStarAgent(env_config, agent_config, enable_logging=True)
        print("A* Agent ready!")
        print()
        
        # Configuration options
        print("Interactive Mode Configuration:")
        print("   Game Boy window will open")
        print("   Real-time metrics displayed")
        print("   Intelligent pathfinding active")
        print("   Goal-directed exploration enabled")
        print()
        
        # Get user preferences
        max_episodes = None
        max_steps_per_episode = None
        
        try:
            episodes_input = input(" Max episodes (press Enter for unlimited): ").strip()
            if episodes_input:
                max_episodes = int(episodes_input)
        except ValueError:
            print("Invalid input, using unlimited episodes")
        
        try:
            steps_input = input("🚶 Max steps per episode (press Enter for default 50000): ").strip()
            if steps_input:
                max_steps_per_episode = int(steps_input)
        except ValueError:
            print("Invalid input, using default 50000 steps")
        
        print()
        print("Starting A* Interactive Session...")
        print("Watch the agent use intelligent pathfinding!")
        print("Press Ctrl+C in terminal to stop")
        print("=" * 50)
        
        # Run interactive session
        agent.run_interactive(max_episodes, max_steps_per_episode)
        
    except KeyboardInterrupt:
        print("\nInteractive session stopped by user")
    except Exception as e:
        print(f"\nError: {e}")
        print("Check your environment setup and dependencies")
    
    print("\nA* Interactive session completed!")


if __name__ == "__main__":
    main()

if __name__ == "__main__":
    print("Inicializando Interactive A* Search Agent...")
    print("Pathfinding inteligente con objetivos dirigidos")
    
    # Session and environment configuration
    sess_path = Path(f'astar_session_{str(time.time_ns())[:8]}')
    ep_length = 2**23
    env_config = {
        'headless': False,
        'save_final_state': True,
        'early_stop': False,
        'action_freq': 24,
        'init_state': '../has_pokedex.state',  # Use pokedex state for A* testing
        'max_steps': ep_length,
        'print_rewards': True,
        'save_video': False,
        'fast_video': True,
        'session_path': sess_path,
        'gb_path': '../PokemonRed.gb',
        'debug': False,
        'sim_frame_dist': 2_000_000.0,
        'extra_buttons': False
    }

    print("Configuración del entorno lista")
    
    # Initialize agent wrapper
    try:
        print("Inicializando agente V2AStar...")
        agent = V2AStarAgent(env_config, enable_logging=True)
        print("Agente A* creado correctamente")
        
        print("Reseteando entorno...")
        observation, info = agent.env.reset()
        agent.agent.reset()
        print("Entorno reseteado correctamente")
        
    except Exception as e:
        print(f"Error inicializando agente: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

    import psutil
    import json
    import csv
    from datetime import datetime
    process = psutil.Process()
    start_time = time.time()
    step = 0
    episode_reward = 0

    got_starter = False
    got_pokemon = False
    import random
    import os
    
    # VARIABLES PARA MÉTRICAS AVANZADAS A*
    action_history = []
    reward_history = []
    pathfinding_stats = {"paths_planned": 0, "successful_paths": 0, "stuck_escapes": 0}
    objective_usage = {"explore": 0, "find_pokemon": 0, "progress_story": 0, "escape_stuck": 0}
    position_history = []
    detailed_stats = {
        "max_reward": 0,
        "min_reward": float('inf'),
        "total_actions": 0,
        "unique_positions": set(),
        "time_per_100_steps": [],
        "memory_usage_history": [],
        "cpu_usage_history": [],
        "planned_moves": 0,
        "intelligent_decisions": 0
    }
    
    def save_metrics(reason=""):
        """Función MEJORADA para guardar métricas completas del A*"""
        elapsed = time.time() - start_time
        mem_info = process.memory_info()
        results_dir = Path(__file__).parent / "results"
        results_dir.mkdir(exist_ok=True)
        timestamp = int(time.time())
        
        # Calcular estadísticas avanzadas del A*
        avg_reward_per_step = episode_reward / max(step, 1)
        steps_per_second = step / max(elapsed, 1)
        avg_memory = sum(detailed_stats["memory_usage_history"]) / max(len(detailed_stats["memory_usage_history"]), 1)
        pathfinding_success_rate = pathfinding_stats["successful_paths"] / max(pathfinding_stats["paths_planned"], 1) * 100
        
        scenario_text = "A* Search - Exploración Inteligente"
        if reason:
            scenario_text += f" ({reason})"
        
        # MARKDOWN DETALLADO PARA A*
        metrics_path = results_dir / f"astar_agent_metrics_{timestamp}.md"
        markdown_report = f"""
---
# 🧭 Informe Completo: A* Search Agent
## {scenario_text}

### **Rendimiento Principal**
- **Recompensa Total:** `{episode_reward:.2f}`
- **Recompensa Máxima:** `{detailed_stats['max_reward']:.2f}`
- **Recompensa Mínima:** `{detailed_stats['min_reward']:.2f}`
- **Recompensa Promedio/Paso:** `{avg_reward_per_step:.4f}`
- **Pasos Totales:** `{step:,}`
- **Escenario:** {scenario_text}

### **Análisis Temporal**
- **Tiempo Total:** `{elapsed:.2f}` segundos ({elapsed/60:.2f} minutos)
- **Pasos por Segundo:** `{steps_per_second:.2f}`
- **Tiempo Promedio/Paso:** `{elapsed/max(step,1)*1000:.2f}` ms

### **Estadísticas de Pathfinding A***
- **Rutas Planificadas:** {pathfinding_stats['paths_planned']:,}
- **Rutas Exitosas:** {pathfinding_stats['successful_paths']:,}
- **Tasa de Éxito:** `{pathfinding_success_rate:.1f}%`
- **Escapes de Bucles:** {pathfinding_stats['stuck_escapes']:,}
- **Movimientos Planificados:** {detailed_stats['planned_moves']:,}
- **Decisiones Inteligentes:** {detailed_stats['intelligent_decisions']:,}

### **Uso de Objetivos A***
- **Exploración:** {objective_usage['explore']:,} veces ({objective_usage['explore']/max(step,1)*100:.1f}%)
- **Buscar Pokémon:** {objective_usage['find_pokemon']:,} veces ({objective_usage['find_pokemon']/max(step,1)*100:.1f}%)
- **Progreso Historia:** {objective_usage['progress_story']:,} veces ({objective_usage['progress_story']/max(step,1)*100:.1f}%)
- **Escapar Bucles:** {objective_usage['escape_stuck']:,} veces ({objective_usage['escape_stuck']/max(step,1)*100:.1f}%)

### **Uso de Recursos del Sistema**
- **Memoria Actual:** `{mem_info.rss / (1024*1024):.2f}` MB
- **Memoria Promedio:** `{avg_memory:.2f}` MB
- **CPU Actual:** `{process.cpu_percent(interval=0.1):.1f}%`
- **Posiciones Únicas Visitadas:** {len(detailed_stats['unique_positions']):,}

### **Estadísticas de Acciones**
- **Total de Acciones:** {detailed_stats['total_actions']:,}
- **Distribución de Acciones:** {dict(sorted([(k,v) for k,v in zip(['↑','↓','←','→','A','B','START'], [action_history.count(i) for i in range(7)])], key=lambda x: x[1], reverse=True))}

### **Configuración del Agente A***
```yaml
Algoritmo: A* Search con Pathfinding Inteligente
Heurística: Distancia Manhattan + Objetivo Específico
Anti-Stuck: Detección automática de bucles
Exploración: Dirigida por objetivos adaptativos
```

### **Análisis de Rendimiento**
```
Eficiencia de Exploración: {len(detailed_stats['unique_positions'])/max(step,1)*100:.3f}%
Inteligencia de Movimiento: {detailed_stats['intelligent_decisions']/max(step,1)*100:.2f}%
Calidad de Pathfinding: {pathfinding_success_rate:.1f}%
```

---
*Generado automáticamente por A* Interactive Agent el {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
        
        # Escribir el reporte markdown
        with open(metrics_path, 'w', encoding='utf-8') as f:
            f.write(markdown_report)
        
        # JSON COMPACTO PARA ANÁLISIS
        json_path = results_dir / f"astar_agent_results_{timestamp}.json"
        json_data = {
            "algorithm": "astar_search",
            "timestamp": timestamp,
            "execution_time": elapsed,
            "total_steps": step,
            "total_reward": episode_reward,
            "avg_reward_per_step": avg_reward_per_step,
            "steps_per_second": steps_per_second,
            "unique_positions": len(detailed_stats['unique_positions']),
            "pathfinding_stats": pathfinding_stats,
            "objective_usage": objective_usage,
            "system_metrics": {
                "memory_mb": mem_info.rss / (1024*1024),
                "avg_memory_mb": avg_memory,
                "cpu_percent": process.cpu_percent(interval=0.1)
            },
            "efficiency_metrics": {
                "exploration_efficiency": len(detailed_stats['unique_positions'])/max(step,1),
                "pathfinding_success_rate": pathfinding_success_rate,
                "intelligence_ratio": detailed_stats['intelligent_decisions']/max(step,1)
            }
        }
        
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=2)
        
        print(f"Métricas A* guardadas: {metrics_path.name} y {json_path.name}")
        return metrics_path, json_path

    print("\nINSTRUCCIONES:")
    print("Presiona Ctrl+C para detener")
    print("El agente A* usará pathfinding inteligente")
    print("Objetivos adaptativos: exploración → búsqueda → progreso")
    print("Anti-stuck automático y detección de bucles")
    print("\nIniciando sesión interactiva A*...")
    
    try:
        observation, info = agent.env.reset()
        confirmation_steps = 0
        max_confirmation_steps = 5
        last_positions = []  # Para tracking de posiciones A*
        
        print(f"Objetivo: Explorar inteligentemente con A* search")
        print(f"Paso inicial: recompensa={episode_reward:.2f}")
        
        while True:
            # Usar el agente A* para elegir acción
            action = agent.agent.act(observation)
            
            # Tracking de acciones y métricas A*
            action_history.append(action)
            detailed_stats['total_actions'] += 1
            
            # Verificar si el A* está planificando rutas
            if hasattr(agent.agent, 'current_path') and agent.agent.current_path:
                detailed_stats['planned_moves'] += 1
                pathfinding_stats['paths_planned'] += 1
                if len(agent.agent.current_path) > 1:
                    pathfinding_stats['successful_paths'] += 1
            
            # Verificar uso de objetivos A*
            if hasattr(agent.agent, 'current_objective'):
                obj_name = agent.agent.current_objective.name.lower() if agent.agent.current_objective else 'explore'
                if obj_name in objective_usage:
                    objective_usage[obj_name] += 1
                    detailed_stats['intelligent_decisions'] += 1
            
            # Ejecutar paso
            observation, reward, terminated, truncated, info = agent.env.step(action)
            episode_reward += reward
            step += 1
            
            # Tracking de posiciones únicas
            if hasattr(observation, 'get'):
                x, y = observation.get('x', 0), observation.get('y', 0)
                detailed_stats['unique_positions'].add((x, y))
                last_positions.append((x, y))
                if len(last_positions) > 100:  # Mantener últimas 100 posiciones
                    last_positions.pop(0)
            
            # Actualizar estadísticas
            detailed_stats['max_reward'] = max(detailed_stats['max_reward'], reward)
            detailed_stats['min_reward'] = min(detailed_stats['min_reward'], reward)
            reward_history.append(reward)
            
            # Logging cada 1000 pasos
            if step % 1000 == 0:
                mem_usage = process.memory_info().rss / (1024*1024)
                detailed_stats['memory_usage_history'].append(mem_usage)
                detailed_stats['cpu_usage_history'].append(process.cpu_percent(interval=0.1))
                
                print(f"Paso {step:,} | Recompensa: {episode_reward:.2f} | "
                      f"Posiciones únicas: {len(detailed_stats['unique_positions'])} | "
                      f"Rutas: {pathfinding_stats['paths_planned']} | "
                      f"Memoria: {mem_usage:.1f}MB")
            
            # DETECCIÓN DE POKÉMON/OBJETIVOS (similar al epsilon greedy)
            pcount = observation.get('pokemon_count', 0) if hasattr(observation, 'get') else 0
            party_size = observation.get('party_size', 0) if hasattr(observation, 'get') else 0
            levels_sum = observation.get('levels_sum', 0) if hasattr(observation, 'get') else 0
            badges = observation.get('badges', None) if hasattr(observation, 'get') else None
            
            # Lógica de detección similar al epsilon greedy pero adaptada para A*
            if pcount > 0 or party_size > 0 or levels_sum > 0:
                got_starter = True
                print(f"\nA* DETECCIÓN! (Paso {step}) - pcount: {pcount}, party: {party_size}, levels: {levels_sum}")
            
            if got_starter:
                try:
                    badges_val = 0
                    if badges is not None:
                        if hasattr(badges, '__len__'):
                            badges_val = sum(badges) if hasattr(badges, 'sum') else sum(list(badges))
                        else:
                            badges_val = badges
                    
                    if badges_val > 0 or any(observation.get(key, 0) > 0 for key in ['starter_id', 'pokemon_seen', 'pokemon_caught']):
                        got_pokemon = True
                        print(f"\nA* ÉXITO! (Paso {step}) - badges: {badges_val}, objetivos alcanzados")
                except Exception as e:
                    pass
            
            # SISTEMA DE CONFIRMACIÓN A*
            if got_starter or got_pokemon:
                confirmation_steps += 1
                print(f"Confirmando objetivo A*... {confirmation_steps}/{max_confirmation_steps}")
                
                if confirmation_steps >= 1:
                    print("\n" + "="*60)
                    print("DETENCIÓN AUTOMÁTICA A* EJECUTADA")
                    print(f"OBJETIVO A* CONFIRMADO en paso {step}")
                    print(f"Estado: pcount={pcount}, levels_sum={levels_sum}")
                    print(f"Rutas planificadas: {pathfinding_stats['paths_planned']}")
                    print(f"Posiciones únicas: {len(detailed_stats['unique_positions'])}")
                    
                    try:
                        save_metrics(f"A* Pokémon obtenido - paso {step}")
                    except Exception as e:
                        print(f" Error guardando métricas A*: {e}")
                    
                    try:
                        print(" Cerrando entorno A*...")
                        agent.env.close()
                        print(" Entorno A* cerrado correctamente")
                    except Exception as e:
                        print(f" Error cerrando entorno A*: {e}")
                    
                    print(" PROGRAMA A* TERMINADO AUTOMÁTICAMENTE")
                    print("="*60)
                    
                    import sys
                    sys.exit(0)
                        
            else:
                confirmation_steps = 0
                
            # Límite de seguridad A*
            if step > 100000:
                print(f"\n Límite de pasos A* alcanzado ({step}). Deteniendo...")
                save_metrics("Límite de pasos A* alcanzado")
                agent.env.close()
                os._exit(0)
                
            if terminated or truncated or step >= ep_length:
                print("Episode finished. Resetting...")
                observation, info = agent.env.reset()
                agent.agent.reset()
                step = 0
                episode_reward = 0
                confirmation_steps = 0
                
    except KeyboardInterrupt:
        print("\n\n Interrumpido por el usuario. Guardando métricas A*...")
        save_metrics("Interrumpido por usuario")
        agent.env.close()
        print(" Exiting A* interactive session.")