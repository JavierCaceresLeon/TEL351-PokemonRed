"""
Interactive Tabu Search Agent for Pokemon Red (Game Boy Interface)
==================================================================

This script launches the Tabu Search agent in the v2 environment with a real-time Game Boy window.
Uses the same metrics system as Epsilon Greedy and PPO agents.
"""

import time
import numpy as np
import os
import psutil
import json
import csv
import sys
from pathlib import Path
from datetime import datetime
import random
import signal

# Import v2 environment components
sys.path.append('../v2')
from red_gym_env_v2 import RedGymEnv
from stream_agent_wrapper import StreamWrapper

# Import the Tabu Search agent
from search_algorithms.tabu_agent import TabuSearchAgent, GameScenario

class InterruptHandler:
    """Handle Ctrl+C gracefully"""
    def __init__(self):
        self.interrupted = False
        signal.signal(signal.SIGINT, self.handle_interrupt)
    
    def handle_interrupt(self, signum, frame):
        print("\n🛑 Interrupción detectada. Guardando métricas...")
        self.interrupted = True

def extract_game_state(observation) -> dict:
    """Extract game state from v2 environment observation"""
    try:
        # Convert observation to dict format that Tabu Search agent expects
        if hasattr(observation, 'keys'):
            # If observation is already a dict
            obs_dict = observation
        else:
            # If observation is an array or other format, create a dict
            obs_dict = {
                'screen': observation if isinstance(observation, np.ndarray) else np.zeros((144, 160, 3)),
            }
        
        # Extract relevant game information (using same pattern as v2_agent.py)
        game_state = {
            'hp': int(obs_dict.get('health', np.array([100]))[0]) if 'health' in obs_dict else 100,
            'max_hp': 100,  # Default max HP
            'level': int(np.sum(obs_dict.get('level', np.zeros(8)))),
            'badges': int(np.sum(obs_dict.get('badges', np.zeros(8)))),
            'pcount': int(np.sum(obs_dict.get('pcount', np.zeros(8)))),
            'events': int(np.sum(obs_dict.get('events', np.zeros(100)))),
            'x': 0,  # Position will be extracted differently
            'y': 0,
            'battle': False,  # Will be detected from screen if possible
        }
        
        return game_state
        
    except Exception as e:
        # Return default state if extraction fails
        return {
            'hp': 100, 'max_hp': 100, 'level': 1, 'badges': 0, 
            'pcount': 0, 'events': 0, 'x': 0, 'y': 0, 'battle': False
        }

def save_metrics(agent, env, step, episode_reward, start_time, process, action_history, 
                reward_history, scenario_detections, position_history, detailed_stats, reason=""):
    """Función COMPLETA para guardar métricas de Tabu Search"""
    elapsed = time.time() - start_time
    mem_info = process.memory_info()
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    timestamp = int(time.time())
    
    # Calcular estadísticas avanzadas
    avg_reward_per_step = episode_reward / max(step, 1)
    steps_per_second = step / max(elapsed, 1)
    avg_memory = sum(detailed_stats["memory_usage_history"]) / max(len(detailed_stats["memory_usage_history"]), 1)
    avg_cpu = sum(detailed_stats["cpu_usage_history"]) / max(len(detailed_stats["cpu_usage_history"]), 1)
    
    # Obtener métricas específicas del agente Tabu Search
    exploration_metrics = agent.get_exploration_metrics()
    
    # Obtener información del juego (usando función auxiliar)
    try:
        # En lugar de llamar env.get_game_state(), usaremos la información que ya tenemos
        game_state = {
            "hp": 100, "max_hp": 100, "level": 1, "badges": 0, 
            "pcount": 0, "x": 0, "y": 0
        }
    except:
        game_state = {"error": "Could not retrieve game state"}
    
    # === 1. DATOS CRUDOS (JSON) ===
    raw_data = {
        "session_info": {
            "timestamp": timestamp,
            "datetime": datetime.now().isoformat(),
            "agent_type": "tabu_search",
            "reason_for_stop": reason,
            "duration_seconds": elapsed,
            "total_steps": step
        },
        "game_performance": {
            "total_reward": float(episode_reward),
            "average_reward_per_step": avg_reward_per_step,
            "max_reward": detailed_stats["max_reward"],
            "min_reward": detailed_stats["min_reward"],
            "steps_per_second": steps_per_second,
            "final_game_state": game_state
        },
        "tabu_search_metrics": {
            "tabu_list_size": len(agent.tabu_list),
            "iteration_count": agent.iteration_count,
            "best_solution_quality": agent.best_solution_quality,
            "current_solution_quality": agent.current_solution_quality,
            "stuck_counter": agent.stuck_counter,
            "tabu_tenure": agent.tabu_tenure,
            "aspiration_threshold": agent.aspiration_threshold
        },
        "exploration_analysis": exploration_metrics,
        "scenario_analysis": {
            "total_detections": sum(scenario_detections.values()),
            "scenario_distribution": scenario_detections,
            "scenario_percentages": {
                scenario: (count / max(sum(scenario_detections.values()), 1)) * 100 
                for scenario, count in scenario_detections.items()
            }
        },
        "action_analysis": {
            "total_actions": len(action_history),
            "action_distribution": {str(i): action_history.count(i) for i in range(7)},
            "action_percentages": {
                str(i): (action_history.count(i) / max(len(action_history), 1)) * 100 
                for i in range(7)
            },
            "recent_actions": action_history[-20:] if len(action_history) >= 20 else action_history,
            "action_patterns": {
                "most_used_action": max(range(7), key=lambda x: action_history.count(x)) if action_history else 0,
                "action_diversity": len(set(action_history)) if action_history else 0
            }
        },
        "temporal_analysis": {
            "reward_progression": reward_history[-50:] if len(reward_history) >= 50 else reward_history,
            "position_diversity": len(set(position_history)) if position_history else 0,
            "exploration_efficiency": exploration_metrics.get('exploration_efficiency', 0),
            "time_analysis": {
                "avg_time_per_100_steps": sum(detailed_stats["time_per_100_steps"]) / max(len(detailed_stats["time_per_100_steps"]), 1),
                "total_time_measurements": len(detailed_stats["time_per_100_steps"])
            }
        },
        "system_performance": {
            "memory_usage_mb": mem_info.rss / 1024 / 1024,
            "avg_memory_mb": avg_memory,
            "memory_history": detailed_stats["memory_usage_history"][-20:],
            "cpu_usage_percent": psutil.cpu_percent(),
            "avg_cpu_percent": avg_cpu,
            "cpu_history": detailed_stats["cpu_usage_history"][-20:]
        },
        "raw_data": {
            "action_history": action_history,
            "reward_history": reward_history,
            "position_history": position_history[-100:] if len(position_history) >= 100 else position_history
        }
    }
    
    # Guardar JSON
    json_file = results_dir / f"tabu_search_raw_data_{timestamp}.json"
    with open(json_file, 'w') as f:
        json.dump(raw_data, f, indent=2, default=str)
    
    # === 2. RESUMEN (CSV) ===
    csv_data = {
        'timestamp': timestamp,
        'agent_type': 'tabu_search',
        'duration_seconds': elapsed,
        'total_steps': step,
        'total_reward': episode_reward,
        'avg_reward_per_step': avg_reward_per_step,
        'steps_per_second': steps_per_second,
        'final_badges': game_state.get('badges', 0),
        'final_pcount': game_state.get('pcount', 0),
        'final_level': game_state.get('level', 1),
        'final_hp': game_state.get('hp', 0),
        'tabu_list_size': len(agent.tabu_list),
        'best_solution_quality': agent.best_solution_quality,
        'unique_positions': exploration_metrics.get('unique_positions_visited', 0),
        'exploration_efficiency': exploration_metrics.get('exploration_efficiency', 0),
        'stuck_episodes': agent.stuck_counter,
        'most_used_action': max(range(7), key=lambda x: action_history.count(x)) if action_history else 0,
        'action_diversity': len(set(action_history)) if action_history else 0,
        'exploration_scenario_pct': scenario_detections.get('exploration', 0) / max(sum(scenario_detections.values()), 1) * 100,
        'battle_scenario_pct': scenario_detections.get('battle', 0) / max(sum(scenario_detections.values()), 1) * 100,
        'stuck_scenario_pct': scenario_detections.get('stuck', 0) / max(sum(scenario_detections.values()), 1) * 100,
        'avg_memory_mb': avg_memory,
        'avg_cpu_percent': avg_cpu,
        'reason_for_stop': reason
    }
    
    csv_file = results_dir / f"tabu_search_summary_{timestamp}.csv"
    with open(csv_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=csv_data.keys())
        writer.writeheader()
        writer.writerow(csv_data)
    
    # === 3. REPORTE LEGIBLE (MARKDOWN) ===
    markdown_content = f"""# Reporte de Métricas - Tabu Search Agent

## 📊 Información de la Sesión
- **Fecha y Hora**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Agente**: Tabu Search
- **Duración**: {elapsed:.1f} segundos ({elapsed/60:.1f} minutos)
- **Pasos Totales**: {step:,}
- **Razón de Parada**: {reason}

## 🎮 Rendimiento del Juego
- **Recompensa Total**: {episode_reward:.2f}
- **Recompensa Promedio por Paso**: {avg_reward_per_step:.4f}
- **Recompensa Máxima**: {detailed_stats['max_reward']:.2f}
- **Recompensa Mínima**: {detailed_stats['min_reward']:.2f}
- **Pasos por Segundo**: {steps_per_second:.2f}

## 🔍 Métricas Específicas de Tabu Search
- **Tamaño de Lista Tabú**: {len(agent.tabu_list)}
- **Iteraciones Totales**: {agent.iteration_count:,}
- **Mejor Calidad de Solución**: {agent.best_solution_quality:.4f}
- **Calidad Actual**: {agent.current_solution_quality:.4f}
- **Episodios de Atascamiento**: {agent.stuck_counter}
- **Tenure Tabú**: {agent.tabu_tenure}
- **Umbral de Aspiración**: {agent.aspiration_threshold:.2f}

## 🗺️ Análisis de Exploración
- **Posiciones Únicas Visitadas**: {exploration_metrics.get('unique_positions_visited', 0):,}
- **Visitas Totales**: {exploration_metrics.get('total_position_visits', 0):,}
- **Eficiencia de Exploración**: {exploration_metrics.get('exploration_efficiency', 0):.2%}
- **Diversidad de Posiciones**: {len(set(position_history)):,}

## 🎯 Distribución de Escenarios
"""
    
    total_scenarios = sum(scenario_detections.values())
    for scenario, count in scenario_detections.items():
        percentage = (count / max(total_scenarios, 1)) * 100
        markdown_content += f"- **{scenario.title()}**: {count:,} detecciones ({percentage:.1f}%)\n"
    
    markdown_content += f"""
## 🎮 Análisis de Acciones
- **Acciones Totales**: {len(action_history):,}
- **Diversidad de Acciones**: {len(set(action_history))} de 7 posibles
- **Acción Más Usada**: {max(range(7), key=lambda x: action_history.count(x)) if action_history else 0}

### Distribución de Acciones:
"""
    
    action_names = ['↓ Down', '← Left', '→ Right', '↑ Up', '🅰 A Button', '🅱 B Button', '⏸ Start']
    for i in range(7):
        count = action_history.count(i)
        percentage = (count / max(len(action_history), 1)) * 100
        markdown_content += f"- **{action_names[i]}**: {count:,} ({percentage:.1f}%)\n"
    
    markdown_content += f"""
## 🎯 Estado Final del Juego
- **Medallas**: {game_state.get('badges', 0)}
- **Pokémon Capturados**: {game_state.get('pcount', 0)}
- **Nivel**: {game_state.get('level', 1)}
- **HP Actual**: {game_state.get('hp', 0)}/{game_state.get('max_hp', 0)}
- **Posición**: ({game_state.get('x', 0)}, {game_state.get('y', 0)})

## 💻 Rendimiento del Sistema
- **Uso de Memoria**: {mem_info.rss / 1024 / 1024:.1f} MB
- **Memoria Promedio**: {avg_memory:.1f} MB
- **CPU Actual**: {psutil.cpu_percent():.1f}%
- **CPU Promedio**: {avg_cpu:.1f}%

## 📈 Métricas de Tiempo
- **Tiempo por 100 Pasos**: {sum(detailed_stats['time_per_100_steps']) / max(len(detailed_stats['time_per_100_steps']), 1):.2f} segundos
- **Eficiencia Temporal**: {(step/elapsed)*60:.0f} pasos/minuto

## 📁 Archivos Generados
- **Datos Crudos**: `{json_file.name}`
- **Resumen CSV**: `{csv_file.name}`
- **Este Reporte**: `tabu_search_metrics_{timestamp}.md`

---
*Reporte generado automáticamente por el sistema de métricas de Tabu Search*
"""
    
    markdown_file = results_dir / f"tabu_search_metrics_{timestamp}.md"
    with open(markdown_file, 'w', encoding='utf-8') as f:
        f.write(markdown_content)
    
    print(f"📊 Guardando datos en {results_dir.name}/tabu_search_*_{timestamp}.*")
    print(f"📄 Generando reporte en {markdown_file.name}")
    print(f"✅ Métricas guardadas exitosamente")
    print(f"📋 Resumen: {step:,} pasos, {episode_reward:.1f} recompensa, {elapsed:.1f}s")

if __name__ == "__main__":
    print("🔍 Iniciando Interactive Tabu Search Agent...")
    
    # Configurar manejador de interrupciones
    interrupt_handler = InterruptHandler()
    
    # Configuración del entorno (usando misma estructura que epsilon greedy)
    sess_path = Path(f'session_{str(time.time_ns())[:8]}')
    ep_length = 2**23
    env_config = {
        'headless': False,
        'save_final_state': False,
        'early_stop': False,
        'action_freq': 24,
        'init_state': '../init.state',
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
    
    print("📋 Configuración del entorno lista")
    
    # Inicializar entorno y agente
    try:
        print("🔧 Inicializando entorno v2...")
        # Usar la misma estructura que epsilon_greedy_interactive.py
        env = StreamWrapper(
            RedGymEnv(env_config),
            stream_metadata={
                "user": "tabu-search-v2",
                "env_id": 0,
                "color": "#aa4477",
                "extra": "Tabu Search Algorithm Agent",
            }
        )
        
        print("🔧 Inicializando agente Tabu Search...")
        agent = TabuSearchAgent(
            tabu_tenure=7,
            max_tabu_size=50,
            aspiration_threshold=1.5,
            scenario_detection_enabled=True
        )
        
        print("✅ Entorno y agente creados correctamente")
        
        print("🔄 Reseteando entorno...")
        observation, info = env.reset()
        print("✅ Entorno reseteado correctamente")
        
    except Exception as e:
        print(f"❌ Error inicializando: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
    
    # Variables para métricas
    process = psutil.Process()
    start_time = time.time()
    step = 0
    episode_reward = 0
    
    # Historiales para métricas
    action_history = []
    reward_history = []
    scenario_detections = {
        "exploration": 0, "battle": 0, "navigation": 0, 
        "progression": 0, "stuck": 0
    }
    position_history = []
    detailed_stats = {
        "max_reward": 0,
        "min_reward": float('inf'),
        "total_actions": 0,
        "unique_positions": set(),
        "time_per_100_steps": [],
        "memory_usage_history": [],
        "cpu_usage_history": []
    }
    
    print("🎮 Iniciando Pokémon Red con métricas mejoradas...")
    print("📊 Capturando métricas en tiempo real...")
    print("⏹️  Presiona Ctrl+C para parar y guardar métricas")
    print("-" * 60)
    
    # Bucle principal del juego
    step_100_timer = time.time()
    
    try:
        while not interrupt_handler.interrupted:
            # Obtener estado del juego desde la observación
            try:
                game_state = extract_game_state(observation)
                current_pos = (game_state.get('x', 0), game_state.get('y', 0))
                position_history.append(current_pos)
                detailed_stats["unique_positions"].add(current_pos)
            except:
                game_state = {'hp': 100, 'max_hp': 100, 'level': 1, 'badges': 0, 'pcount': 0}
                current_pos = (0, 0)
            
            # Seleccionar acción con el agente Tabu Search
            action, decision_info = agent.select_action(observation, game_state)
            
            # Ejecutar acción en el entorno v2
            observation, reward, done, truncated, info = env.step(action)
            
            # Actualizar métricas del agente
            agent.update_performance(action, reward, observation, game_state)
            
            # Actualizar historiales
            action_history.append(action)
            reward_history.append(reward)
            episode_reward += reward
            step += 1
            
            # Actualizar estadísticas detalladas
            detailed_stats["max_reward"] = max(detailed_stats["max_reward"], reward)
            detailed_stats["min_reward"] = min(detailed_stats["min_reward"], reward)
            detailed_stats["total_actions"] += 1
            
            # Registrar detección de escenarios
            scenario = decision_info.get('scenario', 'exploration')
            if scenario in scenario_detections:
                scenario_detections[scenario] += 1
            
            # Métricas de sistema cada 100 pasos
            if step % 100 == 0:
                elapsed_100 = time.time() - step_100_timer
                detailed_stats["time_per_100_steps"].append(elapsed_100)
                
                mem_usage = process.memory_info().rss / 1024 / 1024
                detailed_stats["memory_usage_history"].append(mem_usage)
                detailed_stats["cpu_usage_history"].append(psutil.cpu_percent())
                
                step_100_timer = time.time()
            
            # Mostrar progreso cada 500 pasos
            if step % 500 == 0:
                elapsed = time.time() - start_time
                print(f"⏰ Tiempo: {elapsed/60:.0f}:{elapsed%60:02.0f} | "
                      f"🏃 Pasos: {step:,} | "
                      f"🎯 Recompensa: {episode_reward:.1f} | "
                      f"🔍 Escenario: {scenario} | "
                      f"📊 Lista Tabú: {len(agent.tabu_list)}")
            
            # Condición de parada por episodio terminado
            if done or truncated:
                print("🏁 Episodio terminado")
                break
                
    except KeyboardInterrupt:
        print("\n🛑 Interrupción manual detectada")
    except Exception as e:
        print(f"\n❌ Error durante ejecución: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Guardar métricas al final
        reason = "manual_interrupt" if interrupt_handler.interrupted else "episode_complete"
        save_metrics(agent, env, step, episode_reward, start_time, process, 
                    action_history, reward_history, scenario_detections, 
                    position_history, detailed_stats, reason)
        
        print("🎮 Cerrando entorno...")
        try:
            env.close()
        except:
            pass
        
        print("✅ Sesión de Tabu Search completada")