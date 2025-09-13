"""
Interactive Epsilon Greedy Agent for Pokemon Red (Game Boy Interface)
=====================================================================

This script launches the Epsilon Greedy agent in the v2 environment with a real-time Game Boy window.
Inspired by v2/run_pretrained_interactive.py, it allows interactive control and visualization.
"""

import time
from pathlib import Path
from v2_agent import V2EpsilonGreedyAgent

if __name__ == "__main__":
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

    # Initialize agent wrapper
    agent = V2EpsilonGreedyAgent(env_config, enable_logging=True)
    observation, info = agent.env.reset()
    agent.agent.reset()


    import psutil
    process = psutil.Process()
    start_time = time.time()
    step = 0
    episode_reward = 0

    got_starter = False
    got_pokemon = False
    print("\n[Interactive Epsilon Greedy Agent] Ejecutando hasta elegir el Pokémon inicial o tener al menos 1 Pokémon en el equipo...")
    try:
        while not (got_starter or got_pokemon):
            enhanced_obs = agent.enhance_observation_with_heuristics(observation)
            agent.agent.update_position(enhanced_obs)
            # Selección de acción evitando START (índice 6)
            action = agent.agent.select_action(enhanced_obs)
            if action == 6:
                # Si la acción es START, elige aleatoriamente otra acción válida (0-5)
                import random
                action = random.choice([i for i in range(7) if i != 6])
            observation, reward, terminated, truncated, info = agent.env.step(action)
            agent.env.render()
            step += 1
            episode_reward += reward
            # Detectar evento 'Got Starter' en observation['events']
            events = observation.get('events', None)
            if events is not None and len(events) > 2 and events[2] > 0:
                got_starter = True
            # Detectar si el agente tiene al menos 1 Pokémon en el equipo
            pcount = observation.get('pcount', 0)
            if pcount >= 1:
                got_pokemon = True
            if got_starter or got_pokemon:
                elapsed = time.time() - start_time
                mem_info = process.memory_info()
                import os
                results_dir = Path(__file__).parent / "results"
                results_dir.mkdir(exist_ok=True)
                timestamp = int(time.time())
                metrics_path = results_dir / f"epsilon_greedy_metrics_{timestamp}.md"
                markdown_report = f"\n---\n# Informe de Métricas: Epsilon Greedy - Elección de Pokémon Inicial\n\n"
                markdown_report += f"**Rendimiento:**\n- Recompensa total: `{episode_reward:.2f}`\n- Pasos realizados: `{step}`\n- Escenario: {'Elección de Pokémon inicial' if got_starter else 'Primer Pokémon obtenido'}\n\n"
                markdown_report += f"**Tiempo en encontrar la solución:**\n- Tiempo total: `{elapsed:.2f}` segundos\n\n"
                markdown_report += f"**Uso de recursos:**\n- Memoria usada: `{mem_info.rss / (1024*1024):.2f}` MB\n- CPU (%): `{process.cpu_percent(interval=0.1)}`\n\n"
                markdown_report += f"**Cantidad de pasos:**\n- Total de pasos: `{step}`\n\n"
                markdown_report += f"**Tiempo de entrenamiento:**\n- Epsilon Greedy no requiere entrenamiento previo (tiempo = `0s`)\n\n"
                markdown_report += "---\n"
                with open(metrics_path, "w", encoding="utf-8") as f:
                    f.write(markdown_report)
                print(markdown_report)
                print(f"[Métricas guardadas en: {metrics_path.resolve()}]")
                agent.env.close()
                break
            if terminated or truncated or step >= ep_length:
                print("Episode finished. Resetting...")
                observation, info = agent.env.reset()
                agent.agent.reset()
                step = 0
                episode_reward = 0
    except KeyboardInterrupt:
        print("Exiting interactive session.")
        agent.env.close()
