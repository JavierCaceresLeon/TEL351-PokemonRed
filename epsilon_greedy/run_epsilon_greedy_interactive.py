"""
Interactive Epsilon Greedy Agent for Pokemon Red (Game Boy Interface)
=====================================================================

This script launches the Epsilon Greedy agent in the v2 environment with a real-time Game Boy window.
Inspired by v2/run_pretrained_interactive.py, it allows interactive control and visualization.
"""

import time
import numpy as np
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from epsilon_greedy.v2_agent import V2EpsilonGreedyAgent

if __name__ == "__main__":
    print(" Inicializando Interactive Epsilon Greedy Agent...")
    
    # Session and environment configuration
    sess_path = Path(f'session_{str(time.time_ns())[:8]}')
    ep_length = 2**23
    env_config = {
        'headless': False,
        'save_final_state': True,
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

    print("Configuración del entorno lista")
    
    # Initialize agent wrapper
    try:
        print("Inicializando agente V2EpsilonGreedy...")
        agent = V2EpsilonGreedyAgent(env_config, enable_logging=True)
        print("Agente creado correctamente")
        
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
    
    # VARIABLES PARA MÉTRICAS AVANZADAS
    action_history = []
    reward_history = []
    heuristic_usage = {"explore": 0, "battle": 0, "menu": 0, "overworld": 0, "start": 0}
    scenario_detections = {"explore": 0, "battle": 0, "menu": 0, "overworld": 0, "start": 0}
    epsilon_history = []
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
    
    def save_metrics(reason=""):
        """Función MEJORADA para guardar métricas completas"""
        elapsed = time.time() - start_time
        mem_info = process.memory_info()
        results_dir = Path(__file__).parent / "results"
        results_dir.mkdir(exist_ok=True)
        timestamp = int(time.time())
        
        # Calcular estadísticas avanzadas
        avg_reward_per_step = episode_reward / max(step, 1)
        steps_per_second = step / max(elapsed, 1)
        avg_memory = sum(detailed_stats["memory_usage_history"]) / max(len(detailed_stats["memory_usage_history"]), 1)
        
        scenario_text = "Elección de Pokémon inicial" if got_starter else "Primer Pokémon obtenido"
        if reason:
            scenario_text += f" ({reason})"
        
        # MARKDOWN DETALLADO
        metrics_path = results_dir / f"epsilon_greedy_metrics_{timestamp}.md"
        markdown_report = f"""
---
# Informe Completo: Epsilon Greedy Agent
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

### **Uso de Heurísticas**
- **Exploración:** {heuristic_usage['explore']:,} veces ({heuristic_usage['explore']/max(step,1)*100:.1f}%)
- **Combate:** {heuristic_usage['battle']:,} veces ({heuristic_usage['battle']/max(step,1)*100:.1f}%)
- **Menús:** {heuristic_usage['menu']:,} veces ({heuristic_usage['menu']/max(step,1)*100:.1f}%)
- **Mundo Abierto:** {heuristic_usage['overworld']:,} veces ({heuristic_usage['overworld']/max(step,1)*100:.1f}%)
- **Inicio:** {heuristic_usage['start']:,} veces ({heuristic_usage['start']/max(step,1)*100:.1f}%)

### **Detección de Escenarios**
- **Exploración:** {scenario_detections['explore']:,} detecciones
- **Combate:** {scenario_detections['battle']:,} detecciones
- **Menús:** {scenario_detections['menu']:,} detecciones
- **Mundo Abierto:** {scenario_detections['overworld']:,} detecciones
- **Inicio:** {scenario_detections['start']:,} detecciones

### **Uso de Recursos del Sistema**
- **Memoria Actual:** `{mem_info.rss / (1024*1024):.2f}` MB
- **Memoria Promedio:** `{avg_memory:.2f}` MB
- **CPU Actual:** `{process.cpu_percent(interval=0.1):.1f}%`
- **Posiciones Únicas Visitadas:** {len(detailed_stats['unique_positions']):,}

### **Estadísticas de Acciones**
- **Total de Acciones:** {detailed_stats['total_actions']:,}
- **Distribución de Acciones:** {dict(sorted([(k,v) for k,v in zip(['↑','↓','←','→','A','B','START'], [action_history.count(i) for i in range(7)])], key=lambda x: x[1], reverse=True))}

### **Configuración del Agente**
- **Algoritmo:** Epsilon Greedy con Heurísticas
- **Epsilon Inicial:** Variable según escenario
- **Tiempo de Entrenamiento:** 0s (sin entrenamiento previo)
- **Versión del Entorno:** Pokemon Red v2

### **Notas Adicionales**
- Generado automáticamente el {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
- Sesión ID: {timestamp}
- Razón de finalización: {reason if reason else "Detección automática"}

---
"""
        
        with open(metrics_path, "w", encoding="utf-8") as f:
            f.write(markdown_report)
        
        # GUARDAR DATOS CRUDOS EN JSON
        json_path = results_dir / f"epsilon_greedy_raw_data_{timestamp}.json"
        raw_data = {
            "timestamp": timestamp,
            "session_info": {
                "total_steps": step,
                "total_reward": episode_reward,
                "elapsed_time": elapsed,
                "reason": reason,
                "scenario": scenario_text
            },
            "performance": {
                "avg_reward_per_step": avg_reward_per_step,
                "steps_per_second": steps_per_second,
                "max_reward": detailed_stats['max_reward'],
                "min_reward": detailed_stats['min_reward']
            },
            "heuristics": heuristic_usage,
            "scenarios": scenario_detections,
            "system_resources": {
                "memory_mb": mem_info.rss / (1024*1024),
                "avg_memory_mb": avg_memory,
                "cpu_percent": process.cpu_percent(interval=0.1),
                "unique_positions": len(detailed_stats['unique_positions'])
            },
            "action_history": action_history[-1000:],  # Últimas 1000 acciones
            "reward_history": reward_history[-1000:],  # Últimas 1000 recompensas
            "epsilon_history": epsilon_history[-1000:]  # Últimos 1000 epsilons
        }
        
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(raw_data, f, indent=2)
        
        # GUARDAR CSV PARA ANÁLISIS
        csv_path = results_dir / f"epsilon_greedy_summary_{timestamp}.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Métrica", "Valor"])
            writer.writerow(["Timestamp", timestamp])
            writer.writerow(["Pasos Totales", step])
            writer.writerow(["Recompensa Total", episode_reward])
            writer.writerow(["Tiempo (s)", elapsed])
            writer.writerow(["Pasos/Segundo", steps_per_second])
            writer.writerow(["Memoria (MB)", mem_info.rss / (1024*1024)])
            writer.writerow(["Razón", reason])
            writer.writerow(["Escenario", scenario_text])
        
        print(f"\n MÉTRICAS GUARDADAS:")
        print(f" Markdown: {metrics_path.name}")
        print(f" JSON: {json_path.name}")
        print(f" CSV: {csv_path.name}")
        print(f" Directorio: {results_dir}")
        
        return metrics_path
    
    print("\n[Interactive Epsilon Greedy Agent] Ejecutando hasta elegir el Pokémon inicial o tener al menos 1 Pokémon en el equipo...")
    
    # Variables para detección múltiple y robusta
    confirmation_steps = 0
    max_confirmation_steps = 2  # Reducido para detección más rápida
    
    try:
        while True:
            enhanced_obs = agent.enhance_observation_with_heuristics(observation)
            agent.agent.update_position(enhanced_obs)
            
            # Selección de acción evitando START (índice 6) - CON MANEJO DE ERRORES
            try:
                action = agent.agent.select_action(enhanced_obs)
                # Si action es un array de numpy, tomar el primer elemento
                if hasattr(action, 'shape') and action.shape:
                    action = int(action.item() if hasattr(action, 'item') else action[0])
                elif isinstance(action, (list, tuple)):
                    action = int(action[0])
                else:
                    action = int(action)
                    
                # Evitar botón START (índice 6)
                if action == 6:
                    action = random.choice([i for i in range(7) if i != 6])
                    
            except Exception as e:
                print(f" Error en selección de acción: {e}")
                # Acción por defecto: moverse hacia abajo
                action = 1
            
            observation, reward, terminated, truncated, info = agent.env.step(action)
            agent.env.render()
            step += 1
            episode_reward += reward
            
            # ========== CAPTURA DE MÉTRICAS EN TIEMPO REAL ==========
            # Registrar acción y recompensa
            action_history.append(action)
            reward_history.append(reward)
            
            # Actualizar estadísticas detalladas
            detailed_stats['max_reward'] = max(detailed_stats['max_reward'], episode_reward)
            detailed_stats['min_reward'] = min(detailed_stats['min_reward'], reward)
            detailed_stats['total_actions'] += 1
            
            # Registrar posición actual si está disponible
            try:
                current_pos = (enhanced_obs.get('x', 0), enhanced_obs.get('y', 0))
                detailed_stats['unique_positions'].add(current_pos)
            except:
                pass
            
            # Obtener información del agente sobre escenario y heurística usada
            try:
                current_scenario = agent.agent.current_scenario if hasattr(agent.agent, 'current_scenario') else 'unknown'
                current_heuristic = agent.agent.current_heuristic if hasattr(agent.agent, 'current_heuristic') else 'unknown'
                current_epsilon = agent.agent.epsilon if hasattr(agent.agent, 'epsilon') else 0.1
                
                # Registrar uso de heurísticas y escenarios
                if current_scenario in scenario_detections:
                    scenario_detections[current_scenario] += 1
                if current_heuristic in heuristic_usage:
                    heuristic_usage[current_heuristic] += 1
                    
                epsilon_history.append(current_epsilon)
            except:
                epsilon_history.append(0.1)  # Valor por defecto
            
            # Capturar uso de recursos cada 100 pasos
            if step % 100 == 0:
                try:
                    current_memory = process.memory_info().rss / (1024*1024)
                    current_cpu = process.cpu_percent()
                    detailed_stats['memory_usage_history'].append(current_memory)
                    detailed_stats['cpu_usage_history'].append(current_cpu)
                    
                    elapsed_100 = time.time() - start_time
                    detailed_stats['time_per_100_steps'].append(elapsed_100)
                except:
                    pass
            
            # SISTEMA DE DETECCIÓN MÚLTIPLE Y ROBUSTO
            # Método 1: Verificar 'pcount' (cantidad de Pokémon)
            pcount = observation.get('pcount', 0)
            
            # Método 2: Verificar 'levels_sum' (suma de niveles)
            levels_sum = observation.get('levels_sum', 0)
            
            # Método 3: Verificar eventos - CON MANEJO SEGURO
            events = observation.get('events', None)
            got_starter_event = False
            try:
                if events is not None and hasattr(events, '__len__') and len(events) > 2:
                    if hasattr(events, 'item'):  # Array de numpy
                        event_val = events[2].item() if hasattr(events[2], 'item') else events[2]
                    else:  # Lista normal
                        event_val = events[2]
                    
                    if event_val > 0:
                        got_starter_event = True
                        got_starter = True
                        print(f"\n Evento 'Got Starter' detectado! (Paso {step})")
            except Exception as e:
                # Si hay error con eventos, continuar
                pass
            
            # Método 4: Verificar array de niveles directamente - CON MANEJO SEGURO
            levels = observation.get('levels', [])
            has_pokemon_by_levels = False
            try:
                if levels is not None and hasattr(levels, '__len__') and len(levels) > 0:
                    # Convertir a lista si es array de numpy
                    if hasattr(levels, 'tolist'):
                        levels_list = levels.tolist()
                    else:
                        levels_list = list(levels)
                    has_pokemon_by_levels = any(level > 0 for level in levels_list)
            except Exception as e:
                # Si hay error con levels, continuar
                pass
            
            # Método 5: Verificar party_size si existe
            party_size = observation.get('party_size', 0)
            
            # Método 6: Verificar badges si existe (Pokédex obtenido)
            badges = observation.get('badges', 0)
            
            # DETECCIÓN PRINCIPAL: CUALQUIER MÉTODO POSITIVO
            if pcount >= 1:
                got_pokemon = True
                print(f"\n DETECCIÓN por pcount! (Paso {step}, Cantidad: {pcount})")
            elif levels_sum > 0:
                got_pokemon = True
                print(f"\n DETECCIÓN por levels_sum! (Paso {step}, Nivel total: {levels_sum})")
            elif has_pokemon_by_levels:
                got_pokemon = True
                print(f"\n DETECCIÓN por levels array! (Paso {step}, Levels: {levels})")
            elif party_size > 0:
                got_pokemon = True
                print(f"\n DETECCIÓN por party_size! (Paso {step}, Tamaño: {party_size})")
            elif got_starter_event:
                got_pokemon = True
                print(f"\n DETECCIÓN por evento Got Starter! (Paso {step})")
            
            # DEBUG EXTENDIDO: Imprimir todos los campos relevantes cada 50 pasos
            if step % 50 == 0:
                print(f"[Debug {step}] pcount: {pcount}, levels_sum: {levels_sum}, party_size: {party_size}, badges: {badges}")
                
                # Manejo seguro de arrays para debug
                levels_str = "None"
                if levels is not None:
                    try:
                        if hasattr(levels, '__len__') and len(levels) > 0:
                            levels_str = str(levels[:6])
                        else:
                            levels_str = str(levels)
                    except:
                        levels_str = "Error"
                
                events_str = "None"
                if events is not None:
                    try:
                        if hasattr(events, '__len__') and len(events) > 0:
                            events_str = str(events[:5])
                        else:
                            events_str = str(events)
                    except:
                        events_str = "Error"
                
                print(f"            levels: {levels_str}, events: {events_str}")
                
                # Imprimir TODAS las claves de observation para debug
                if step % 200 == 0:  # Cada 200 pasos mostrar todas las claves
                    print(f"[Debug {step}] Todas las claves de observation: {list(observation.keys())}")
            
            # DETECCIÓN ADICIONAL: Verificar cambios en la observación
            if step > 1000:  # Después de 1000 pasos, ser más agresivo
                # Si hay cualquier cambio en badges, levels, o eventos
                try:
                    badges_val = 0
                    if badges is not None:
                        if hasattr(badges, '__len__'):
                            badges_val = sum(badges) if hasattr(badges, 'sum') else sum(list(badges))
                        else:
                            badges_val = badges
                    
                    if badges_val > 0 or any(observation.get(key, 0) > 0 for key in ['starter_id', 'pokemon_seen', 'pokemon_caught']):
                        got_pokemon = True
                        print(f"\n DETECCIÓN AGRESIVA! (Paso {step}) - badges: {badges_val}, campos especiales detectados")
                except Exception as e:
                    # Si hay error en detección agresiva, continuar
                    pass
            
            # SISTEMA DE CONFIRMACIÓN INMEDIATO Y ROBUSTO
            if got_starter or got_pokemon:
                confirmation_steps += 1
                print(f" Confirmando objetivo... {confirmation_steps}/{max_confirmation_steps}")
                
                # DETENCIÓN INMEDIATA - SIN DEMORA
                if confirmation_steps >= 1:  # Solo 1 confirmación para detección instantánea
                    print("\n" + "="*60)
                    print(" DETENCIÓN AUTOMÁTICA EJECUTADA 🚨")
                    print(f" OBJETIVO CONFIRMADO en paso {step}")
                    print(f" Estado: pcount={pcount}, levels_sum={levels_sum}")
                    try:
                        badges_str = str(badges) if badges is not None else "None"
                        print(f" Party size: {party_size}, badges: {badges_str}")
                    except:
                        print(f" Party size: {party_size}, badges: Error")
                    
                    # Guardar métricas INMEDIATAMENTE
                    try:
                        save_metrics(f"Pokémon obtenido - paso {step}")
                    except Exception as e:
                        print(f" Error guardando métricas: {e}")
                    
                    # Cerrar entorno INMEDIATAMENTE
                    try:
                        print(" Cerrando entorno...")
                        agent.env.close()
                        print(" Entorno cerrado correctamente")
                    except Exception as e:
                        print(f" Error cerrando entorno: {e}")
                    
                    print(" PROGRAMA TERMINADO AUTOMÁTICAMENTE")
                    print("="*60)
                    
                    # MÚLTIPLES MÉTODOS DE SALIDA FORZADA
                    try:
                        import sys
                        sys.exit(0)
                    except:
                        pass
                    
                    try:
                        os._exit(0)
                    except:
                        pass
                    
                    # Si todo falla, terminar de forma abrupta
                    exit(0)
                        
            else:
                confirmation_steps = 0  # Reset si no se detecta objetivo
                
            # Límite de seguridad: detener después de muchos pasos sin objetivo
            if step > 100000:  # 100k pasos máximo
                print(f"\n Límite de pasos alcanzado ({step}). Deteniendo...")
                save_metrics("Límite de pasos alcanzado")
                agent.env.close()
                os._exit(0)
                
            if terminated or truncated or step >= ep_length:
                print("Episode finished. Resetting...")
                observation, info = agent.env.reset()
                agent.agent.reset()
                step = 0
                episode_reward = 0
                confirmation_steps = 0  # Reset confirmación en reset
                
    except KeyboardInterrupt:
        print("\n\nInterrumpido por el usuario. Guardando métricas...")
        save_metrics("Interrumpido por usuario")
        agent.env.close()
        print("Exiting interactive session.")
