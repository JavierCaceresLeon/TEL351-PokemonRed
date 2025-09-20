"""
Script de Prueba Simple - Detección de Pokémon Garantizada
==========================================================

Este es un script simplificado que garantiza la detención automática
cuando el agente obtiene su primer Pokémon, sin depender de múltiples métodos.
"""

import time
import numpy as np
import os
from pathlib import Path
from v2_agent import V2EpsilonGreedyAgent

if __name__ == "__main__":
    # Session y configuración mínima
    sess_path = Path(f'session_test_{str(time.time_ns())[:8]}')
    env_config = {
        'headless': False,
        'save_final_state': True,
        'early_stop': False,
        'action_freq': 24,
        'init_state': '../init.state',
        'max_steps': 50000,  # Límite más bajo para pruebas
        'print_rewards': False,  # Menos ruido en consola
        'save_video': False,
        'fast_video': True,
        'session_path': sess_path,
        'gb_path': '../PokemonRed.gb',
        'debug': False,
        'sim_frame_dist': 2_000_000.0,
        'extra_buttons': False
    }

    # Inicializar agente
    agent = V2EpsilonGreedyAgent(env_config, enable_logging=False)
    observation, info = agent.env.reset()
    agent.agent.reset()

    import psutil
    import random
    process = psutil.Process()
    start_time = time.time()
    step = 0
    episode_reward = 0
    
    print("🎮 SCRIPT DE PRUEBA - Detección Simple de Pokémon")
    print("=" * 50)
    print("Objetivo: Obtener el primer Pokémon y cerrar automáticamente")
    print("Método: Detección simple por pcount >= 1")
    print("=" * 50)
    
    try:
        while True:
            # Ejecutar paso del agente
            enhanced_obs = agent.enhance_observation_with_heuristics(observation)
            agent.agent.update_position(enhanced_obs)
            action = agent.agent.select_action(enhanced_obs)
            
            # Evitar START
            if action == 6:
                action = random.choice([i for i in range(7) if i != 6])
            
            observation, reward, terminated, truncated, info = agent.env.step(action)
            agent.env.render()
            step += 1
            episode_reward += reward
            
            # DETECCIÓN SIMPLE Y DIRECTA
            pcount = observation.get('pcount', 0)
            
            # Mostrar estado cada 100 pasos
            if step % 100 == 0:
                print(f"Paso {step}: pcount={pcount}, reward={reward:.2f}")
            
            # CONDICIÓN DE PARADA SIMPLE
            if pcount >= 1:
                elapsed = time.time() - start_time
                mem_info = process.memory_info()
                
                print("\n" + "🎉" * 20)
                print("✅ ¡POKÉMON DETECTADO!")
                print(f"📊 Paso: {step}")
                print(f"📊 Pokémon en equipo: {pcount}")
                print(f"⏱️ Tiempo: {elapsed:.2f} segundos")
                print(f"🏆 Recompensa total: {episode_reward:.2f}")
                print(f"💾 Memoria: {mem_info.rss / (1024*1024):.2f} MB")
                
                # Guardar métricas simples
                results_dir = Path(__file__).parent / "results"
                results_dir.mkdir(exist_ok=True)
                timestamp = int(time.time())
                metrics_path = results_dir / f"simple_test_metrics_{timestamp}.md"
                
                with open(metrics_path, "w") as f:
                    f.write(f"# Test Simple - Pokémon Obtenido\n\n")
                    f.write(f"- **Paso:** {step}\n")
                    f.write(f"- **Pokémon:** {pcount}\n")
                    f.write(f"- **Tiempo:** {elapsed:.2f}s\n")
                    f.write(f"- **Recompensa:** {episode_reward:.2f}\n")
                    f.write(f"- **Memoria:** {mem_info.rss / (1024*1024):.2f} MB\n")
                
                print(f"💾 Métricas guardadas: {metrics_path}")
                
                # CERRAR INMEDIATAMENTE
                print("🔒 Cerrando Game Boy...")
                agent.env.close()
                print("🏁 ¡PRUEBA COMPLETADA EXITOSAMENTE!")
                print("🎉" * 20)
                
                # SALIDA FORZADA
                os._exit(0)
            
            # Reset si termina episodio
            if terminated or truncated:
                print(f"Episode terminado en paso {step}. Reiniciando...")
                observation, info = agent.env.reset()
                agent.agent.reset()
                step = 0
                episode_reward = 0
                
    except KeyboardInterrupt:
        print("\n🛑 Interrumpido por usuario")
        agent.env.close()
        print("Test terminado manualmente")