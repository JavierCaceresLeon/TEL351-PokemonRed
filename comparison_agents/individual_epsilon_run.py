
import sys
import time
import numpy as np
import os
from pathlib import Path
import argparse
import psutil
import json
import csv
from datetime import datetime

# Añadir la ruta de v2 para importar RedGymEnv y del directorio comparison_agents
sys.path.append(str(Path(__file__).resolve().parent.parent / 'v2'))
sys.path.append(str(Path(__file__).resolve().parent))

try:
    from red_gym_env_v2 import RedGymEnv
    from v2_agent import V2EpsilonGreedyAgent
except ImportError as e:
    print(f"Error importando módulos: {e}")
    sys.exit(1)

def save_metrics(results_dir, step, episode_reward, start_time, reason="", agent=None, detailed_stats=None, action_history=None, reward_history=None, epsilon_history=None):
    """Función MEJORADA para guardar métricas completas"""
    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = int(time.time())
    process = psutil.Process()
    elapsed = time.time() - start_time
    mem_info = process.memory_info()

    # Extraer datos del agente si está disponible
    heuristic_usage = {"explore": 0, "battle": 0, "menu": 0, "overworld": 0, "start": 0}
    scenario_detections = {"explore": 0, "battle": 0, "menu": 0, "overworld": 0, "start": 0}
    if agent and hasattr(agent, 'agent'):
        if hasattr(agent.agent, 'heuristic_usage'):
            heuristic_usage = agent.agent.heuristic_usage
        if hasattr(agent.agent, 'scenario_detections'):
            scenario_detections = agent.agent.scenario_detections

    # Calcular estadísticas avanzadas
    avg_reward_per_step = episode_reward / max(step, 1)
    steps_per_second = step / max(elapsed, 1)
    avg_memory = sum(detailed_stats["memory_usage_history"]) / max(len(detailed_stats["memory_usage_history"]), 1) if detailed_stats and detailed_stats["memory_usage_history"] else 0

    scenario_text = "Elección de Pokémon inicial"
    if reason:
        scenario_text += f" ({reason})"

    # MARKDOWN DETALLADO
    metrics_path = results_dir / f"epsilon_greedy_metrics_{timestamp}.md"
    markdown_report = f"""
# 📊 Informe: Epsilon Greedy Agent
- **Pasos Totales:** {step:,}
- **Tiempo Total:** {elapsed:.2f} s
- **Pasos/Segundo:** {steps_per_second:.2f}
- **Recompensa Total:** {episode_reward:.2f}
- **Razón de finalización:** {reason}
"""
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write(markdown_report)

    # DATOS CRUDOS EN JSON
    json_path = results_dir / f"epsilon_greedy_raw_data_{timestamp}.json"
    raw_data = {
        "timestamp": timestamp,
        "session_info": {"total_steps": step, "total_reward": episode_reward, "elapsed_time": elapsed, "reason": reason},
        "performance": {"avg_reward_per_step": avg_reward_per_step, "steps_per_second": steps_per_second},
        "heuristics": heuristic_usage, "scenarios": scenario_detections,
        "system_resources": {"memory_mb": mem_info.rss / (1024*1024), "avg_memory_mb": avg_memory},
        "action_history": action_history[-1000:] if action_history else [],
        "reward_history": reward_history[-1000:] if reward_history else [],
        "epsilon_history": epsilon_history[-1000:] if epsilon_history else []
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(raw_data, f, indent=2)

    # RESUMEN EN CSV
    csv_path = results_dir / f"epsilon_greedy_summary_{timestamp}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Métrica", "Valor"])
        writer.writerow(["Timestamp", timestamp])
        writer.writerow(["Pasos Totales", step])
        writer.writerow(["Recompensa Total", episode_reward])
        writer.writerow(["Tiempo (s)", elapsed])
        writer.writerow(["Pasos/Segundo", steps_per_second])
        writer.writerow(["Razón", reason])

    print(f"📊 Métricas guardadas en: {results_dir.name}")
    return metrics_path

def run_single_epsilon_greedy_episode(epsilon_start, epsilon_min, epsilon_decay, results_dir):
    """
    Ejecuta un único episodio del agente Epsilon Greedy con la configuración dada.
    """
    print(f"🧬 Configuración: ε_start={epsilon_start}, ε_min={epsilon_min}, ε_decay={epsilon_decay}")
    
    sess_path = Path(f'session_{str(time.time_ns())[:8]}')
    ep_length = 2**23  # Un número muy grande, la detención es por evento
    
    env_config = {
        'headless': True,  # MÁXIMA VELOCIDAD
        'save_final_state': False,
        'early_stop': False,
        'action_freq': 24,
        'init_state': str(Path(__file__).parent.parent / 'init.state'),
        'max_steps': ep_length,
        'print_rewards': False, # Desactivado para no saturar logs
        'save_video': False,
        'fast_video': False,
        'session_path': sess_path,
        'gb_path': str(Path(__file__).parent.parent / 'PokemonRed.gb'),
        'debug': False,
        'sim_frame_dist': 2_000_000.0,
        'extra_buttons': False,
        'emulation_speed': 0 # 0 para velocidad ilimitada en headless
    }

    agent_config = {
        'epsilon_start': epsilon_start,
        'epsilon_min': epsilon_min,
        'epsilon_decay': epsilon_decay,
        'scenario_detection_enabled': True
    }

    try:
        agent = V2EpsilonGreedyAgent(env_config, agent_config=agent_config, enable_logging=False)
        # Forzar velocidad de emulación
        agent.env.pyboy.set_emulation_speed(0)
        
        observation, info = agent.env.reset()
        agent.agent.reset()
    except Exception as e:
        print(f"❌ Error inicializando agente: {e}")
        import traceback
        traceback.print_exc()
        return

    start_time = time.time()
    step = 0
    episode_reward = 0
    
    action_history, reward_history, epsilon_history = [], [], []
    detailed_stats = {"memory_usage_history": [], "cpu_usage_history": []}
    process = psutil.Process()

    confirmation_steps = 0
    max_confirmation_steps = 2

    try:
        while True:
            enhanced_obs = agent.enhance_observation_with_heuristics(observation)
            agent.agent.update_position(enhanced_obs)
            
            action = agent.agent.select_action(enhanced_obs)
            if hasattr(action, 'item'): action = action.item()

            observation, reward, terminated, truncated, info = agent.env.step(action)
            
            step += 1
            episode_reward += reward
            
            action_history.append(action)
            reward_history.append(reward)
            if hasattr(agent.agent, 'epsilon'):
                epsilon_history.append(agent.agent.epsilon)

            if step % 100 == 0:
                try:
                    detailed_stats['memory_usage_history'].append(process.memory_info().rss / (1024*1024))
                    detailed_stats['cpu_usage_history'].append(process.cpu_percent())
                except: pass

            # Sistema de detección robusto
            pcount = observation.get('pcount', 0)
            levels_sum = observation.get('levels_sum', 0)
            party_size = observation.get('party_size', 0)
            
            got_pokemon = pcount >= 1 or levels_sum > 0 or party_size > 0

            if got_pokemon:
                confirmation_steps += 1
                if confirmation_steps >= max_confirmation_steps:
                    print(f"✅ Objetivo alcanzado en {step} pasos.")
                    save_metrics(results_dir, step, episode_reward, start_time, "Pokémon obtenido", agent, detailed_stats, action_history, reward_history, epsilon_history)
                    agent.env.close()
                    return
            else:
                confirmation_steps = 0
            
            if step > 40000: # Límite de seguridad
                print("⚠️ Límite de pasos alcanzado.")
                save_metrics(results_dir, step, episode_reward, start_time, "Límite de pasos", agent, detailed_stats, action_history, reward_history, epsilon_history)
                agent.env.close()
                return
                
    except Exception as e:
        print(f"❌ Error durante la ejecución: {e}")
        import traceback
        traceback.print_exc()
        save_metrics(results_dir, step, episode_reward, start_time, f"Error: {e}", agent, detailed_stats, action_history, reward_history, epsilon_history)
        agent.env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ejecutar una única instancia de Epsilon Greedy.")
    parser.add_argument("--epsilon_start", type=float, required=True)
    parser.add_argument("--epsilon_min", type=float, required=True)
    parser.add_argument("--epsilon_decay", type=float, required=True)
    parser.add_argument("--results_dir", type=str, required=True)
    args = parser.parse_args()

    run_single_epsilon_greedy_episode(
        epsilon_start=args.epsilon_start,
        epsilon_min=args.epsilon_min,
        epsilon_decay=args.epsilon_decay,
        results_dir=Path(args.results_dir)
    )
