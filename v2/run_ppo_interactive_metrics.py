"""
PPO Agent Interactivo con Sistema de Métricas Completas
======================================================

Script mejorado para ejecutar el agente PPO preentrenado con captura detallada de métricas
y análisis completo de rendimiento, compatible con el sistema de métricas de Epsilon Greedy.
"""

import os
import time
import json
import csv
import psutil
import numpy as np
from os.path import exists
from pathlib import Path
import uuid
import glob
from datetime import datetime
from red_gym_env_v2 import RedGymEnv
from stable_baselines3 import A2C, PPO
from stable_baselines3.common import env_checker
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import CheckpointCallback

def make_env(rank, env_conf, seed=0):
    """
    Utility function for multiprocessed env.
    """
    def _init():
        env = RedGymEnv(env_conf)
        return env
    set_random_seed(seed)
    return _init

def get_most_recent_zip_with_age(folder_path):
    # Get all zip files in the folder
    zip_files = glob.glob(os.path.join(folder_path, "*.zip"))
    
    if not zip_files:
        return None, None
    
    # Find the most recently modified zip file
    most_recent_zip = max(zip_files, key=os.path.getmtime)
    
    # Calculate how old the file is in hours
    current_time = time.time()
    modification_time = os.path.getmtime(most_recent_zip)
    age_in_hours = (current_time - modification_time) / 3600
    
    return most_recent_zip, age_in_hours

def save_ppo_metrics(step, episode_reward, start_time, process, action_history, reward_history, 
                     detailed_stats, model_path="", reason=""):
    """Función para guardar métricas completas del agente PPO"""
    elapsed = time.time() - start_time
    mem_info = process.memory_info()
    results_dir = Path(__file__).parent / "ppo_results"
    results_dir.mkdir(exist_ok=True)
    timestamp = int(time.time())
    
    # Calcular estadísticas avanzadas
    avg_reward_per_step = episode_reward / max(step, 1)
    steps_per_second = step / max(elapsed, 1)
    avg_memory = sum(detailed_stats["memory_usage_history"]) / max(len(detailed_stats["memory_usage_history"]), 1)
    
    scenario_text = "Pokémon Red - Agente PPO Preentrenado"
    if reason:
        scenario_text += f" ({reason})"
    
    # MARKDOWN DETALLADO
    metrics_path = results_dir / f"ppo_metrics_{timestamp}.md"
    markdown_report = f"""
---
# 🤖 Informe Completo: PPO Agent (Deep Learning)
## {scenario_text}

### 🎯 **Rendimiento Principal**
- **Recompensa Total:** `{episode_reward:.2f}`
- **Recompensa Máxima:** `{detailed_stats['max_reward']:.2f}`
- **Recompensa Mínima:** `{detailed_stats['min_reward']:.2f}`
- **Recompensa Promedio/Paso:** `{avg_reward_per_step:.4f}`
- **Pasos Totales:** `{step:,}`
- **Tipo de Agente:** PPO (Proximal Policy Optimization)

### ⏱️ **Análisis Temporal**
- **Tiempo Total:** `{elapsed:.2f}` segundos ({elapsed/60:.2f} minutos)
- **Pasos por Segundo:** `{steps_per_second:.2f}`
- **Tiempo Promedio/Paso:** `{elapsed/max(step,1)*1000:.2f}` ms

### 🧠 **Información del Modelo**
- **Algoritmo:** PPO (Proximal Policy Optimization)
- **Modelo Cargado:** `{model_path}`
- **Modo:** Determinístico = False
- **Estado:** Modelo preentrenado cargado desde checkpoint

### 💻 **Uso de Recursos del Sistema**
- **Memoria Actual:** `{mem_info.rss / (1024*1024):.2f}` MB
- **Memoria Promedio:** `{avg_memory:.2f}` MB
- **CPU Actual:** `{process.cpu_percent(interval=0.1):.1f}%`
- **Posiciones Únicas Visitadas:** {len(detailed_stats['unique_positions']):,}

### 📈 **Estadísticas de Acciones**
- **Total de Acciones:** {detailed_stats['total_actions']:,}
- **Distribución de Acciones:** {dict(sorted([(k,v) for k,v in zip(['↑','↓','←','→','A','B','START'], [action_history.count(i) for i in range(7)])], key=lambda x: x[1], reverse=True))}

### 📊 **Análisis de Recompensas**
- **Recompensa Media por Acción:** {np.mean(reward_history[-1000:]):.4f} (últimas 1000 acciones)
- **Desviación Estándar:** {np.std(reward_history[-1000:]):.4f}
- **Recompensas Positivas:** {sum(1 for r in reward_history if r > 0):,} ({sum(1 for r in reward_history if r > 0)/len(reward_history)*100:.1f}%)
- **Recompensas Negativas:** {sum(1 for r in reward_history if r < 0):,} ({sum(1 for r in reward_history if r < 0)/len(reward_history)*100:.1f}%)

### 🎮 **Comportamiento del Agente**
- **Modo de Control:** Automático (IA controlando completamente)
- **Predicciones Realizadas:** {step:,}
- **Exploración:** Controlada por política entrenada
- **Episodios Completados:** Variable (depende de duración)

### 🔧 **Configuración del Entorno**
- **Juego:** Pokemon Red (Game Boy)
- **Frecuencia de Acción:** 24 frames por acción
- **Estado Inicial:** init.state
- **Máximos Pasos:** {2**23:,}
- **Visualización:** Activada (ventana Game Boy)

### 📝 **Notas Adicionales**
- Generado automáticamente el {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
- Sesión ID: {timestamp}
- Razón de finalización: {reason if reason else "Manual por usuario"}
- Agente preentrenado usando reinforcement learning

### 🆚 **Comparación con Epsilon Greedy**
- **Ventaja PPO:** Aprendizaje por experiencia, comportamiento más sofisticado
- **Entrenamiento:** Miles de horas de experiencia vs. heurísticas manuales
- **Consistencia:** Más predecible en situaciones conocidas
- **Adaptabilidad:** Mejor manejo de situaciones complejas

---
"""
    
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write(markdown_report)
    
    # GUARDAR DATOS CRUDOS EN JSON
    json_path = results_dir / f"ppo_raw_data_{timestamp}.json"
    raw_data = {
        "timestamp": timestamp,
        "session_info": {
            "total_steps": step,
            "total_reward": episode_reward,
            "elapsed_time": elapsed,
            "reason": reason,
            "scenario": scenario_text,
            "model_path": model_path
        },
        "performance": {
            "avg_reward_per_step": avg_reward_per_step,
            "steps_per_second": steps_per_second,
            "max_reward": detailed_stats['max_reward'],
            "min_reward": detailed_stats['min_reward']
        },
        "system_resources": {
            "memory_mb": mem_info.rss / (1024*1024),
            "avg_memory_mb": avg_memory,
            "cpu_percent": process.cpu_percent(interval=0.1),
            "unique_positions": len(detailed_stats['unique_positions'])
        },
        "action_history": action_history[-1000:],  # Últimas 1000 acciones
        "reward_history": reward_history[-1000:],  # Últimas 1000 recompensas
        "agent_type": "PPO",
        "model_info": {
            "algorithm": "Proximal Policy Optimization",
            "deterministic": False,
            "pretrained": True
        }
    }
    
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(raw_data, f, indent=2)
    
    # GUARDAR CSV PARA ANÁLISIS
    csv_path = results_dir / f"ppo_summary_{timestamp}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Métrica", "Valor"])
        writer.writerow(["Timestamp", timestamp])
        writer.writerow(["Agente", "PPO"])
        writer.writerow(["Pasos Totales", step])
        writer.writerow(["Recompensa Total", episode_reward])
        writer.writerow(["Tiempo (s)", elapsed])
        writer.writerow(["Pasos/Segundo", steps_per_second])
        writer.writerow(["Memoria (MB)", mem_info.rss / (1024*1024)])
        writer.writerow(["Modelo", model_path])
        writer.writerow(["Razón", reason])
    
    print(f"\n🤖 MÉTRICAS PPO GUARDADAS:")
    print(f"📄 Markdown: {metrics_path.name}")
    print(f"🔢 JSON: {json_path.name}")
    print(f"📈 CSV: {csv_path.name}")
    print(f"📁 Directorio: {results_dir}")
    
    return metrics_path

if __name__ == '__main__':
    print("🤖 PPO Agent con Sistema de Métricas Avanzadas")
    print("=" * 60)
    
    sess_path = Path(f'ppo_session_{str(uuid.uuid4())[:8]}')
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
    
    num_cpu = 1
    env = make_env(0, env_config)()
    
    # Buscar el checkpoint más reciente
    most_recent_checkpoint, time_since = get_most_recent_zip_with_age("runs")
    if most_recent_checkpoint is not None:
        file_name = most_recent_checkpoint
        print(f"📁 Usando checkpoint: {file_name}")
        print(f"⏰ Edad del archivo: {time_since:.2f} horas")
    else:
        print("❌ No se encontraron checkpoints en la carpeta 'runs'")
        exit(1)
    
    print('🔄 Cargando modelo PPO...')
    model = PPO.load(file_name, env=env, custom_objects={'lr_schedule': 0, 'clip_range': 0})
    print('✅ Modelo cargado correctamente')
    
    # VARIABLES PARA MÉTRICAS AVANZADAS
    process = psutil.Process()
    start_time = time.time()
    step = 0
    episode_reward = 0
    action_history = []
    reward_history = []
    detailed_stats = {
        "max_reward": 0,
        "min_reward": float('inf'),
        "total_actions": 0,
        "unique_positions": set(),
        "memory_usage_history": [],
        "cpu_usage_history": []
    }
    
    print("\n🎮 Iniciando ejecución del agente PPO...")
    print("🛑 Presiona Ctrl+C para parar y generar métricas completas")
    
    obs, info = env.reset()
    
    try:
        while True:
            # Verificar si el agente está habilitado
            try:
                with open("agent_enabled.txt", "r") as f:
                    agent_enabled = f.readlines()[0].startswith("yes")
            except:
                agent_enabled = True  # Por defecto habilitado
                
            if agent_enabled:
                # PPO predice la acción
                action, _states = model.predict(obs, deterministic=False)
                obs, rewards, terminated, truncated, info = env.step(action)
                
                # ========== CAPTURA DE MÉTRICAS EN TIEMPO REAL ==========
                step += 1
                reward = rewards if isinstance(rewards, (int, float)) else rewards[0]
                episode_reward += reward
                
                # Registrar acción y recompensa
                action_history.append(int(action) if hasattr(action, 'item') else int(action[0]) if hasattr(action, '__len__') else action)
                reward_history.append(reward)
                
                # Actualizar estadísticas detalladas
                detailed_stats['max_reward'] = max(detailed_stats['max_reward'], episode_reward)
                detailed_stats['min_reward'] = min(detailed_stats['min_reward'], reward)
                detailed_stats['total_actions'] += 1
                
                # Registrar posición si está disponible
                try:
                    current_pos = (obs.get('x', 0), obs.get('y', 0)) if isinstance(obs, dict) else (0, 0)
                    detailed_stats['unique_positions'].add(current_pos)
                except:
                    pass
                
                # Capturar uso de recursos cada 100 pasos
                if step % 100 == 0:
                    try:
                        current_memory = process.memory_info().rss / (1024*1024)
                        current_cpu = process.cpu_percent()
                        detailed_stats['memory_usage_history'].append(current_memory)
                        detailed_stats['cpu_usage_history'].append(current_cpu)
                    except:
                        pass
                
                # Debug cada 200 pasos
                if step % 200 == 0:
                    elapsed = time.time() - start_time
                    print(f"[PPO {step:4d}] Recompensa: {episode_reward:.2f} (t={elapsed:.1f}s)")
                
            else:
                # Modo manual/pausa
                env.pyboy.tick(1, True)
                obs = env._get_obs()
                truncated = env.step_count >= env.max_steps - 1
                
            env.render()
            
            if terminated or truncated:
                break
                
    except KeyboardInterrupt:
        print("\n\n🛑 Interrumpido por el usuario. Generando métricas completas...")
        save_ppo_metrics(step, episode_reward, start_time, process, action_history, 
                         reward_history, detailed_stats, file_name, "Interrumpido por usuario")
        
    except Exception as e:
        print(f"\n❌ Error durante ejecución: {e}")
        save_ppo_metrics(step, episode_reward, start_time, process, action_history,
                         reward_history, detailed_stats, file_name, f"Error: {str(e)}")
    
    finally:
        print("🔒 Cerrando entorno...")
        env.close()
        print("👋 Sesión finalizada")