"""
Detector Ultra Simple - Solo por Recompensa
==========================================

Este script detecta el primer Pokémon basándose únicamente en el cambio de recompensa.
Cuando la recompensa total supera un umbral específico, asume que se obtuvo el Pokémon.
"""

import time
import numpy as np
import os
import sys
from pathlib import Path
from v2_agent import V2EpsilonGreedyAgent
import signal

def ultra_force_exit():
    """Salida ultra-forzada sin demoras"""
    print("💥 SALIDA ULTRA-FORZADA")
    try:
        os.kill(os.getpid(), signal.SIGTERM)
    except:
        pass
    try:
        os._exit(0)
    except:
        pass

def save_ultra_metrics(step, reward, elapsed):
    """Guardar métricas ultra-rápido"""
    try:
        results_dir = Path(__file__).parent / "results"
        results_dir.mkdir(exist_ok=True)
        timestamp = int(time.time())
        metrics_path = results_dir / f"ultra_detector_metrics_{timestamp}.md"
        
        report = f"""
---
# Detector Ultra Simple - Pokémon por Recompensa ✅

**Método:** Detección por umbral de recompensa
**Resultado:** ÉXITO
**Pasos:** {step}
**Recompensa:** {reward:.2f}
**Tiempo:** {elapsed:.2f} segundos

---
"""
        
        with open(metrics_path, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"💾 Métricas: {metrics_path}")
        return True
    except:
        return False

if __name__ == "__main__":
    print("⚡ DETECTOR ULTRA SIMPLE - SOLO RECOMPENSA")
    print("=" * 50)
    
    # Configuración ultra-básica
    sess_path = Path(f'ultra_session_{str(time.time_ns())[:8]}')
    env_config = {
        'headless': False,
        'save_final_state': False,  # Más rápido
        'early_stop': False,
        'action_freq': 24,
        'init_state': '../init.state',
        'max_steps': 30000,
        'print_rewards': False,
        'save_video': False,
        'fast_video': True,
        'session_path': sess_path,
        'gb_path': '../PokemonRed.gb',
        'debug': False,
        'sim_frame_dist': 2_000_000.0,
        'extra_buttons': False
    }

    # Inicializar
    try:
        print("🚀 Inicializando agente ultra-simple...")
        agent = V2EpsilonGreedyAgent(env_config, enable_logging=False)
        observation, info = agent.env.reset()
        agent.agent.reset()
        print("✅ Listo")
    except Exception as e:
        print(f"❌ Error: {e}")
        ultra_force_exit()

    # Variables
    start_time = time.time()
    step = 0
    episode_reward = 0
    pokemon_detected = False
    
    # UMBRALES DE DETECCIÓN
    REWARD_THRESHOLD = 40.0  # Umbral de recompensa que indica Pokémon obtenido
    MAX_STEPS = 30000
    
    print(f"🎯 Umbral de recompensa: {REWARD_THRESHOLD}")
    print("📊 Iniciando detección...")
    
    try:
        while not pokemon_detected and step < MAX_STEPS:
            # Acción simple
            try:
                enhanced_obs = agent.enhance_observation_with_heuristics(observation)
                agent.agent.update_position(enhanced_obs)
                action = agent.agent.select_action(enhanced_obs)
                
                # Simplificar action si es array
                if hasattr(action, 'item'):
                    action = action.item()
                elif hasattr(action, '__len__'):
                    action = action[0]
                action = int(action)
                
                # Evitar START
                if action == 6:
                    action = 1  # Simplemente mover hacia abajo
                    
            except:
                action = 1  # Acción por defecto
            
            # Ejecutar
            observation, reward, terminated, truncated, info = agent.env.step(action)
            agent.env.render()
            step += 1
            episode_reward += reward
            
            # DETECCIÓN POR RECOMPENSA SIMPLE
            if episode_reward >= REWARD_THRESHOLD:
                pokemon_detected = True
                print(f"\n🎯 ¡POKÉMON DETECTADO POR RECOMPENSA!")
                print(f"   Paso: {step}, Recompensa: {episode_reward:.2f}")
            
            # Debug cada 200 pasos
            if step % 200 == 0:
                elapsed = time.time() - start_time
                print(f"[{step:4d}] Recompensa: {episode_reward:.1f} (t={elapsed:.1f}s)")
            
            # TERMINACIÓN INMEDIATA
            if pokemon_detected:
                elapsed = time.time() - start_time
                print("\n" + "🎊" * 20)
                print("💥 POKÉMON DETECTADO - TERMINACIÓN INMEDIATA")
                print(f"📊 Paso: {step}, Recompensa: {episode_reward:.2f}")
                print(f"⏱️ Tiempo: {elapsed:.2f}s")
                print("🎊" * 20)
                
                # Guardar métricas ULTRA-RÁPIDO
                save_ultra_metrics(step, episode_reward, elapsed)
                
                # CERRAR TODO INMEDIATAMENTE
                try:
                    agent.env.close()
                except:
                    pass
                
                print("💥 SALIENDO AHORA...")
                ultra_force_exit()
            
            # Reset si termina episodio
            if terminated or truncated:
                observation, info = agent.env.reset()
                agent.agent.reset()
                
    except KeyboardInterrupt:
        print("\n⚠️ Interrumpido")
        elapsed = time.time() - start_time
        save_ultra_metrics(step, episode_reward, elapsed)
        ultra_force_exit()
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        ultra_force_exit()

    # Si llega aquí, no encontró Pokémon
    print("⚠️ No se detectó Pokémon por recompensa")
    ultra_force_exit()