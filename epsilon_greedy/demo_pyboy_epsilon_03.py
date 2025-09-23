"""
Demo Automático: Epsilon 0.3 con PyBoy Emulador
===============================================

Demo automático basado en run_epsilon_greedy_interactive.py que muestra el 
emulador PyBoy ejecutándose con epsilon fijo de 0.3 (exploración moderada).
No requiere input del usuario y se ejecuta automáticamente.
"""

import time
import numpy as np
import os
import sys
import psutil
import json
import csv
from datetime import datetime
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from epsilon_greedy.v2_agent import V2EpsilonGreedyAgent

class AutoEpsilonDemo03:
    """Demo automático con epsilon 0.3"""
    
    def __init__(self):
        self.target_epsilon = 0.3
        self.max_steps = 50000  # Límite automático para demo
        self.stats_interval = 100  # Mostrar stats cada 100 pasos
        
        # Variables para métricas como run_epsilon_greedy_interactive.py
        self.process = psutil.Process()
        self.start_time = None
        self.action_history = []
        self.reward_history = []
        self.epsilon_history = []
        self.position_history = []
        self.heuristic_usage = {"explore": 0, "battle": 0, "menu": 0, "overworld": 0, "start": 0}
        self.scenario_detections = {"explore": 0, "battle": 0, "menu": 0, "overworld": 0, "start": 0}
        self.detailed_stats = {
            "max_reward": 0,
            "min_reward": float('inf'),
            "total_actions": 0,
            "unique_positions": set(),
            "time_per_100_steps": [],
            "memory_usage_history": [],
            "cpu_usage_history": []
        }
        
    def save_metrics(self, reason="", step=0, episode_reward=0):
        """Función para guardar métricas completas como run_epsilon_greedy_interactive.py"""
        if self.start_time is None:
            return
            
        elapsed = time.time() - self.start_time
        mem_info = self.process.memory_info()
        results_dir = Path(__file__).parent / "results"
        results_dir.mkdir(exist_ok=True)
        timestamp = int(time.time())
        
        # Calcular estadísticas avanzadas
        avg_reward_per_step = episode_reward / max(step, 1)
        steps_per_second = step / max(elapsed, 1)
        avg_memory = sum(self.detailed_stats["memory_usage_history"]) / max(len(self.detailed_stats["memory_usage_history"]), 1) if self.detailed_stats["memory_usage_history"] else 0
        
        scenario_text = "Demo Epsilon 0.3 - Exploración Moderada"
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
- **Recompensa Máxima:** `{self.detailed_stats['max_reward']:.2f}`
- **Recompensa Mínima:** `{self.detailed_stats['min_reward']:.2f}`
- **Recompensa Promedio/Paso:** `{avg_reward_per_step:.4f}`
- **Pasos Totales:** `{step:,}`
- **Escenario:** {scenario_text}

### **Análisis Temporal**
- **Tiempo Total:** `{elapsed:.2f}` segundos ({elapsed/60:.2f} minutos)
- **Pasos por Segundo:** `{steps_per_second:.2f}`
- **Tiempo Promedio/Paso:** `{elapsed/max(step,1)*1000:.2f}` ms

### **Uso de Heurísticas**
- **Exploración:** {self.heuristic_usage['explore']:,} veces ({self.heuristic_usage['explore']/max(step,1)*100:.1f}%)
- **Combate:** {self.heuristic_usage['battle']:,} veces ({self.heuristic_usage['battle']/max(step,1)*100:.1f}%)
- **Menús:** {self.heuristic_usage['menu']:,} veces ({self.heuristic_usage['menu']/max(step,1)*100:.1f}%)
- **Mundo Abierto:** {self.heuristic_usage['overworld']:,} veces ({self.heuristic_usage['overworld']/max(step,1)*100:.1f}%)
- **Inicio:** {self.heuristic_usage['start']:,} veces ({self.heuristic_usage['start']/max(step,1)*100:.1f}%)

### **Detección de Escenarios**
- **Exploración:** {self.scenario_detections['explore']:,} detecciones
- **Combate:** {self.scenario_detections['battle']:,} detecciones
- **Menús:** {self.scenario_detections['menu']:,} detecciones
- **Mundo Abierto:** {self.scenario_detections['overworld']:,} detecciones
- **Inicio:** {self.scenario_detections['start']:,} detecciones

### **Uso de Recursos del Sistema**
- **Memoria Actual:** `{mem_info.rss / (1024*1024):.2f}` MB
- **Memoria Promedio:** `{avg_memory:.2f}` MB
- **CPU Actual:** `{self.process.cpu_percent(interval=0.1):.1f}%`
- **Posiciones Únicas Visitadas:** {len(self.detailed_stats['unique_positions']):,}

### **Estadísticas de Acciones**
- **Total de Acciones:** {self.detailed_stats['total_actions']:,}
- **Distribución de Acciones:** {dict(sorted([(k,v) for k,v in zip(['↑','↓','←','→','A','B','START'], [self.action_history.count(i) for i in range(7)])], key=lambda x: x[1], reverse=True))}

### **Configuración del Agente**
- **Algoritmo:** Epsilon Greedy con Heurísticas
- **Epsilon Inicial:** {self.target_epsilon} (fijo)
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
                "max_reward": self.detailed_stats['max_reward'],
                "min_reward": self.detailed_stats['min_reward']
            },
            "heuristics": self.heuristic_usage,
            "scenarios": self.scenario_detections,
            "system_resources": {
                "memory_mb": mem_info.rss / (1024*1024),
                "avg_memory_mb": avg_memory,
                "cpu_percent": self.process.cpu_percent(interval=0.1),
                "unique_positions": len(self.detailed_stats['unique_positions'])
            },
            "action_history": self.action_history[-1000:],  # Últimas 1000 acciones
            "reward_history": self.reward_history[-1000:],  # Últimas 1000 recompensas
            "epsilon_history": self.epsilon_history[-1000:]  # Últimos 1000 epsilons
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
        
    def run_demo(self):
        print("="*60)
        print(" [EPSILON-0.3-MODERADO] DEMO AUTOMÁTICO: EPSILON 0.3 (EXPLORACIÓN MODERADA)")
        print("="*60)
        print(f" [EPSILON-0.3-MODERADO] Epsilon fijo: {self.target_epsilon}")
        print(f" [EPSILON-0.3-MODERADO] Máximo pasos: {self.max_steps:,}")
        print(f" [EPSILON-0.3-MODERADO] PyBoy emulador se mostrará automáticamente")
        print(f" [EPSILON-0.3-MODERADO] La demo se ejecuta sola, sin input requerido")
        print("="*60)
        
        # Session and environment configuration
        sess_path = Path(f'demo_epsilon_03_session_{str(time.time_ns())[:8]}')
        env_config = {
            'headless': False,  # IMPORTANTE: mostrar emulador
            'save_final_state': True,
            'early_stop': False,
            'action_freq': 24,
            'init_state': '../init.state',
            'max_steps': self.max_steps,
            'print_rewards': True,
            'save_video': False,
            'fast_video': True,
            'session_path': sess_path,
            'gb_path': '../PokemonRed.gb',
            'debug': False,
            'sim_frame_dist': 2_000_000.0,
            'extra_buttons': False
        }

        print(" [EPSILON-0.3-MODERADO] Configurando entorno PyBoy...")
        
        # Configuración del agente con epsilon fijo
        agent_config = {
            'epsilon_start': self.target_epsilon,
            'epsilon_min': self.target_epsilon,    # Mismo valor = no decay
            'epsilon_decay': 1.0,                  # Sin decay
            'scenario_detection_enabled': True
        }
        
        # Initialize agent wrapper
        try:
            print(" [EPSILON-0.3-MODERADO] Inicializando agente con epsilon fijo 0.3...")
            agent = V2EpsilonGreedyAgent(env_config, agent_config, enable_logging=True)
            
            # PERSONALIZAR VENTANA PYBOY PARA EPSILON 0.3 - IDENTIFICACIÓN MEJORADA
            window_title = "POKEMON RED ===>>> EPSILON 0.3 MODERADO <<<==== 30-EXPLORA 70-EXPLOTA"
            if hasattr(agent.env, 'env') and hasattr(agent.env.env, 'pyboy'):
                pyboy = agent.env.env.pyboy
                
                # Múltiples intentos para establecer título de ventana
                try:
                    if hasattr(pyboy, 'set_window_title'):
                        pyboy.set_window_title(window_title)
                    elif hasattr(pyboy, '_window_title'):
                        pyboy._window_title = window_title
                    elif hasattr(pyboy, 'window'):
                        if hasattr(pyboy.window, 'set_title'):
                            pyboy.window.set_title(window_title)
                except Exception as e:
                    print(f" [EPSILON-0.3-MODERADO] Advertencia: No se pudo establecer título de ventana: {e}")
                
                # Intentar posicionar ventana en esquina superior izquierda
                try:
                    if hasattr(pyboy, 'window'):
                        if hasattr(pyboy.window, 'set_position'):
                            pyboy.window.set_position(100, 100)  # Esquina superior izquierda
                except Exception as e:
                    print(f" [EPSILON-0.3-MODERADO] Advertencia: No se pudo posicionar ventana: {e}")
            
            # También personalizar los metadatos del stream
            if hasattr(agent.env, 'stream_metadata'):
                agent.env.stream_metadata.update({
                    "user": "EPSILON-0.3-MODERADO-DEMO",
                    "env_id": "E03",
                    "identifier": "MODERADO",
                    "epsilon_value": "0.3",
                    "behavior": "30% Exploración - 70% Explotación",
                    "extra": "Demo Épsilon 0.3 - Comportamiento Balanceado",
                })
            
            print(" [EPSILON-0.3-MODERADO] Agente creado correctamente")
            
            print(" [EPSILON-0.3-MODERADO] Reseteando entorno...")
            observation, info = agent.env.reset()
            agent.agent.reset()
            
            # FORZAR epsilon a 0.3 para asegurar que sea fijo
            agent.agent.epsilon = self.target_epsilon
            print(f" [EPSILON-0.3-MODERADO] Epsilon configurado: {agent.agent.epsilon}")
            print(f" [EPSILON-0.3-MODERADO] Ventana identificada como: EPSILON 0.3 (MODERADO)")
            
        except Exception as e:
            print(f" [EPSILON-0.3-MODERADO] Error inicializando agente: {e}")
            import traceback
            traceback.print_exc()
            return

        # Variables de tracking
        step = 0
        episode_reward = 0
        self.start_time = time.time()  # Inicializar start_time
        last_stats_time = self.start_time
        
        # Tracking para estadísticas
        action_counts = [0] * 8  # 8 acciones posibles
        total_exploration = 0
        total_exploitation = 0
        
        print(" [EPSILON-0.3-MODERADO] DEMO INICIADA! El emulador PyBoy debería aparecer ahora...")
        print(" [EPSILON-0.3-MODERADO] El agente explorará con epsilon=0.3 (30% exploración, 70% explotación)")
        print(" [EPSILON-0.3-MODERADO] Estadísticas se mostrarán automáticamente...")
        print("-" * 60)
        
        try:
            while step < self.max_steps:
                # Obtener observación mejorada
                enhanced_obs = agent.enhance_observation_with_heuristics(observation)
                agent.agent.update_position(enhanced_obs)
                
                # FORZAR epsilon a 0.3 en cada paso (por seguridad)
                agent.agent.epsilon = self.target_epsilon
                
                # Selección de acción
                try:
                    action = agent.agent.select_action(enhanced_obs)
                    
                    # Normalizar acción
                    if hasattr(action, 'shape') and action.shape:
                        action = int(action.item() if hasattr(action, 'item') else action[0])
                    elif isinstance(action, (list, tuple)):
                        action = int(action[0])
                    else:
                        action = int(action)
                    
                    # BLOQUEAR BOTÓN START (índice 6) para evitar menús problemáticos
                    if action == 6:
                        # Reemplazar con acción de movimiento aleatoria
                        action = np.random.choice([0, 1, 2, 3])  # Solo movimientos: ↑, ↓, ←, →
                        print(" START bloqueado - usando movimiento aleatorio")
                    
                    # Asegurar acción válida (0-7, pero START ya está bloqueado)
                    action = max(0, min(7, action))
                    
                    # Tracking de exploración vs explotación
                    if hasattr(agent.agent, 'last_action_was_exploration'):
                        if agent.agent.last_action_was_exploration:
                            total_exploration += 1
                        else:
                            total_exploitation += 1
                    
                    action_counts[action] += 1
                    
                except Exception as e:
                    print(f" Error en selección de acción: {e}")
                    action = 1  # Acción por defecto: abajo
                
                # Ejecutar paso en el entorno
                observation, reward, terminated, truncated, info = agent.env.step(action)
                agent.env.render()  # IMPORTANTE: renderizar para mostrar emulador
                
                step += 1
                episode_reward += reward
                
                # ========== CAPTURA DE MÉTRICAS EN TIEMPO REAL ==========
                # Registrar acción y recompensa
                self.action_history.append(action)
                self.reward_history.append(reward)
                
                # Actualizar estadísticas detalladas
                self.detailed_stats['max_reward'] = max(self.detailed_stats['max_reward'], episode_reward)
                self.detailed_stats['min_reward'] = min(self.detailed_stats['min_reward'], reward)
                self.detailed_stats['total_actions'] += 1
                
                # Registrar posición actual si está disponible
                try:
                    current_pos = (enhanced_obs.get('x', 0), enhanced_obs.get('y', 0))
                    self.detailed_stats['unique_positions'].add(current_pos)
                except:
                    pass
                
                # Obtener información del agente sobre escenario y heurística usada
                try:
                    current_scenario = agent.agent.current_scenario if hasattr(agent.agent, 'current_scenario') else 'unknown'
                    current_heuristic = agent.agent.current_heuristic if hasattr(agent.agent, 'current_heuristic') else 'unknown'
                    current_epsilon = agent.agent.epsilon if hasattr(agent.agent, 'epsilon') else self.target_epsilon
                    
                    # Registrar uso de heurísticas y escenarios
                    if current_scenario in self.scenario_detections:
                        self.scenario_detections[current_scenario] += 1
                    if current_heuristic in self.heuristic_usage:
                        self.heuristic_usage[current_heuristic] += 1
                        
                    self.epsilon_history.append(current_epsilon)
                except:
                    self.epsilon_history.append(self.target_epsilon)  # Valor por defecto
                
                # Capturar uso de recursos cada 100 pasos
                if step % 100 == 0:
                    try:
                        current_memory = self.process.memory_info().rss / (1024*1024)
                        current_cpu = self.process.cpu_percent()
                        self.detailed_stats['memory_usage_history'].append(current_memory)
                        self.detailed_stats['cpu_usage_history'].append(current_cpu)
                        
                        elapsed_100 = time.time() - self.start_time
                        self.detailed_stats['time_per_100_steps'].append(elapsed_100)
                    except:
                        pass
                
                # Mostrar estadísticas periódicamente
                current_time = time.time()
                if current_time - last_stats_time >= 10:  # Cada 10 segundos
                    self._print_demo_stats(step, episode_reward, self.start_time, action_counts, 
                                         total_exploration, total_exploitation)
                    last_stats_time = current_time
                
                # Condiciones de parada automática
                if terminated or truncated:
                    print(f" Episodio terminado en paso {step}")
                    break
                
                # Detección simple de objetivo (Pokemon obtenido)
                pcount = observation.get('pcount', 0)
                if pcount >= 1:
                    print(f" ¡OBJETIVO ALCANZADO! Pokemon obtenido en paso {step}")
                    print(f" Demo completada exitosamente")
                    try:
                        self.save_metrics(f"Objetivo alcanzado - Pokemon obtenido", step, episode_reward)
                        self._metrics_saved = True
                    except Exception as e:
                        print(f" Error guardando métricas: {e}")
                    break
                
        except KeyboardInterrupt:
            print("\n Demo interrumpida por el usuario (Ctrl+C)")
            print(" Guardando métricas antes de salir...")
            try:
                self.save_metrics("Interrumpido por usuario (Ctrl+C)", step, episode_reward)
                self._metrics_saved = True
            except Exception as e:
                print(f" Error guardando métricas: {e}")
            
        except Exception as e:
            print(f"\n Error durante la demo: {e}")
            print(" Guardando métricas antes de salir...")
            try:
                self.save_metrics(f"Error durante ejecución: {str(e)}", step, episode_reward)
                self._metrics_saved = True
            except Exception as save_error:
                print(f" Error guardando métricas: {save_error}")
            import traceback
            traceback.print_exc()
            
        finally:
            # Estadísticas finales
            elapsed = time.time() - self.start_time
            print("\n" + "="*60)
            print(" ESTADÍSTICAS FINALES DE LA DEMO")
            print("="*60)
            print(f" Tiempo total: {elapsed:.1f} segundos ({elapsed/60:.1f} minutos)")
            print(f" Pasos ejecutados: {step:,}")
            print(f" Epsilon usado: {self.target_epsilon} (fijo)")
            print(f" Recompensa total: {episode_reward:.2f}")
            
            if step > 0:
                print(f" Pasos por segundo: {step/elapsed:.1f}")
                print(f" Recompensa promedio: {episode_reward/step:.4f}")
                
                # Estadísticas de exploración
                total_actions = total_exploration + total_exploitation
                if total_actions > 0:
                    print(f" Exploración: {total_exploration:,} ({total_exploration/total_actions:.1%})")
                    print(f" Explotación: {total_exploitation:,} ({total_exploitation/total_actions:.1%})")
                
                # Top acciones
                action_names = ['↑', '↓', '←', '→', 'A', 'B', 'START', 'SELECT']
                print(f" Acciones más usadas:")
                for i, count in enumerate(action_counts):
                    if count > 0:
                        print(f"   {action_names[i]}: {count:,} veces ({count/step:.1%})")
            
            print("="*60)
            
            # Guardar métricas finales si no se guardaron antes
            try:
                if not hasattr(self, '_metrics_saved'):
                    self.save_metrics("Demo finalizada normalmente", step, episode_reward)
                    self._metrics_saved = True
            except Exception as e:
                print(f" Error guardando métricas finales: {e}")
            
            # Cerrar entorno
            try:
                agent.env.close()
                print(" Entorno cerrado correctamente")
            except Exception as e:
                print(f" Error cerrando entorno: {e}")
            
            print(" Demo finalizada")
    
    def _print_demo_stats(self, step, episode_reward, start_time, action_counts, 
                         total_exploration, total_exploitation):
        """Imprime estadísticas periódicas de la demo"""
        elapsed = time.time() - start_time
        
        print(f" [Paso {step:,}] Tiempo: {elapsed:.1f}s | "
              f"Recompensa: {episode_reward:.2f} | "
              f"Epsilon: {self.target_epsilon}")
        
        if step > 0:
            total_actions = total_exploration + total_exploitation
            if total_actions > 0:
                print(f"     Exploración: {total_exploration/total_actions:.1%} | "
                      f" Explotación: {total_exploitation/total_actions:.1%} | "
                      f" {step/elapsed:.1f} pasos/s")


if __name__ == "__main__":
    print("Iniciando Demo Automático: Epsilon 0.3 con PyBoy Emulador...")
    
    demo = AutoEpsilonDemo03()
    demo.run_demo()