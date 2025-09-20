"""
Detector Simple de Pokémon - Script de Respaldo Robusto
======================================================

Este script está diseñado específicamente para detectar de forma INMEDIATA
cuando se obtiene el primer Pokémon y TERMINAR AUTOMÁTICAMENTE sin intervención manual.

Es un script de respaldo simplificado que garantiza el cierre automático.
"""

import time
import numpy as np
import os
import sys
from pathlib import Path
from v2_agent import V2EpsilonGreedyAgent
import psutil
import signal

def force_exit():
    """Función de salida forzada múltiple"""
    print("🚨 EJECUTANDO SALIDA FORZADA...")
    try:
        # Método 1: sys.exit
        sys.exit(0)
    except:
        pass
    
    try:
        # Método 2: os._exit
        os._exit(0)
    except:
        pass
    
    try:
        # Método 3: signal
        os.kill(os.getpid(), signal.SIGTERM)
    except:
        pass

def save_simple_metrics(step, reward, elapsed):
    """Función simplificada para guardar métricas"""
    try:
        results_dir = Path(__file__).parent / "results"
        results_dir.mkdir(exist_ok=True)
        timestamp = int(time.time())
        metrics_path = results_dir / f"simple_detector_metrics_{timestamp}.md"
        
        report = f"""
---
# Detector Simple - Pokémon Obtenido ✅

**Resultado:** ÉXITO - Pokémon detectado automáticamente
**Pasos:** {step}
**Recompensa:** {reward:.2f}
**Tiempo:** {elapsed:.2f} segundos
**Timestamp:** {timestamp}

---
"""
        
        with open(metrics_path, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"✅ Métricas guardadas: {metrics_path}")
        return True
    except Exception as e:
        print(f"⚠️ Error guardando métricas: {e}")
        return False

if __name__ == "__main__":
    print("🔥 DETECTOR SIMPLE DE POKÉMON - INICIO")
    print("=" * 50)
    
    # Configuración mínima del entorno
    sess_path = Path(f'simple_session_{str(time.time_ns())[:8]}')
    env_config = {
        'headless': False,
        'save_final_state': True,
        'early_stop': False,
        'action_freq': 24,
        'init_state': '../init.state',
        'max_steps': 50000,  # Límite más bajo
        'print_rewards': False,  # Reducir ruido
        'save_video': False,
        'fast_video': True,
        'session_path': sess_path,
        'gb_path': '../PokemonRed.gb',
        'debug': False,
        'sim_frame_dist': 2_000_000.0,
        'extra_buttons': False
    }

    # Inicializar agente
    try:
        print("🔧 Creando agente V2EpsilonGreedy...")
        agent = V2EpsilonGreedyAgent(env_config, enable_logging=False)  # Sin logging para simplicidad
        print("🔄 Reseteando entorno...")
        observation, info = agent.env.reset()
        agent.agent.reset()
        print("✅ Agente inicializado correctamente")
    except Exception as e:
        print(f"❌ Error inicializando agente: {e}")
        import traceback
        traceback.print_exc()
        force_exit()

    # Variables de control
    start_time = time.time()
    step = 0
    episode_reward = 0
    
    # Variables de detección simplificada
    pokemon_detected = False
    
    print("🎯 Iniciando búsqueda del primer Pokémon...")
    print("📊 Detección activa - el programa se cerrará AUTOMÁTICAMENTE al obtener Pokémon\n")
    
    try:
        while not pokemon_detected and step < 50000:
            # Obtener observación mejorada
            enhanced_obs = agent.enhance_observation_with_heuristics(observation)
            agent.agent.update_position(enhanced_obs)
            
            # Seleccionar acción (evitar START)
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
                    action = np.random.choice([0, 1, 2, 3, 4, 5])
                    
            except Exception as e:
                print(f"⚠️ Error en selección de acción: {e}")
                # Acción por defecto: moverse hacia abajo
                action = 1
            
            # Ejecutar acción
            observation, reward, terminated, truncated, info = agent.env.step(action)
            agent.env.render()
            step += 1
            episode_reward += reward
            
            # ========== DETECCIÓN PRINCIPAL ==========
            # Método 1: pcount (cantidad de Pokémon en equipo)
            pcount = observation.get('pcount', 0)
            if pcount >= 1:
                pokemon_detected = True
                print(f"\n🎯 ¡POKÉMON DETECTADO! (Método: pcount)")
                print(f"   Paso: {step}, Cantidad: {pcount}")
            
            # Método 2: levels_sum (suma de niveles)
            if not pokemon_detected:
                levels_sum = observation.get('levels_sum', 0)
                if levels_sum > 0:
                    pokemon_detected = True
                    print(f"\n🎯 ¡POKÉMON DETECTADO! (Método: levels_sum)")
                    print(f"   Paso: {step}, Niveles totales: {levels_sum}")
            
            # Método 3: Verificar niveles individuales
            if not pokemon_detected:
                levels = observation.get('levels', [])
                try:
                    if levels is not None and hasattr(levels, '__len__') and len(levels) > 0:
                        # Convertir a lista si es array de numpy
                        if hasattr(levels, 'tolist'):
                            levels_list = levels.tolist()
                        else:
                            levels_list = list(levels)
                        
                        if any(level > 0 for level in levels_list[:6]):  # Solo los primeros 6 slots del equipo
                            pokemon_detected = True
                            print(f"\n🎯 ¡POKÉMON DETECTADO! (Método: levels individuales)")
                            print(f"   Paso: {step}, Levels: {levels_list[:6]}")
                except Exception as e:
                    pass
            
            # Método 4: Verificar eventos - CON MANEJO SEGURO
            if not pokemon_detected:
                events = observation.get('events', [])
                try:
                    # Verificar si events es un array de numpy o lista
                    if events is not None and hasattr(events, '__len__') and len(events) > 2:
                        if hasattr(events, 'item'):  # Array de numpy
                            event_val = events[2].item() if hasattr(events[2], 'item') else events[2]
                        else:  # Lista normal
                            event_val = events[2]
                        
                        if event_val > 0:
                            pokemon_detected = True
                            print(f"\n🎯 ¡POKÉMON DETECTADO! (Método: eventos)")
                            print(f"   Paso: {step}, Evento: {event_val}")
                except Exception as e:
                    # Si hay error con eventos, continuar sin problema
                    pass
            
            # Método 5: Verificar cambios en badges o party_size
            if not pokemon_detected:
                party_size = observation.get('party_size', 0)
                badges = observation.get('badges', 0)
                
                try:
                    badges_val = 0
                    if badges is not None:
                        if hasattr(badges, '__len__'):
                            badges_val = sum(badges) if hasattr(badges, 'sum') else sum(list(badges))
                        else:
                            badges_val = badges
                    
                    if party_size > 0 or badges_val > 0:
                        pokemon_detected = True
                        print(f"\n🎯 ¡POKÉMON DETECTADO! (Método: party/badges)")
                        print(f"   Paso: {step}, Party size: {party_size}, Badges: {badges_val}")
                except Exception as e:
                    pass
            
            # Debug cada 100 pasos con más información
            if step % 100 == 0:
                elapsed = time.time() - start_time
                # Obtener valores para debug
                levels_sum = observation.get('levels_sum', 0)
                party_size = observation.get('party_size', 0)
                
                print(f"[{step:4d}] Buscando... (t={elapsed:.1f}s, pcount={pcount}, levels_sum={levels_sum}, party={party_size})")
                
                # Debug más detallado cada 500 pasos
                if step % 500 == 0:
                    levels = observation.get('levels', [])
                    events = observation.get('events', [])
                    try:
                        levels_str = str(levels[:6]) if levels is not None and hasattr(levels, '__len__') else str(levels)
                        events_str = str(events[:5]) if events is not None and hasattr(events, '__len__') else str(events)
                        print(f"    Detalles: levels={levels_str}, events={events_str}")
                        print(f"    Claves disponibles: {list(observation.keys())[:10]}...")  # Primeras 10 claves
                    except:
                        pass
            
            # DETECCIÓN EXITOSA - TERMINAR INMEDIATAMENTE
            if pokemon_detected:
                elapsed = time.time() - start_time
                print("\n" + "="*60)
                print("🏆 ¡ÉXITO! PRIMER POKÉMON OBTENIDO")
                print(f"📊 Paso: {step}")
                print(f"⏱️  Tiempo: {elapsed:.2f} segundos")
                print(f"🎯 Recompensa total: {episode_reward:.2f}")
                print("="*60)
                
                # Guardar métricas
                save_simple_metrics(step, episode_reward, elapsed)
                
                # TERMINACIÓN INMEDIATA MÚLTIPLE
                print("🚀 INICIANDO TERMINACIÓN INMEDIATA...")
                
                # Cerrar entorno ANTES de salir
                try:
                    print("🔒 Cerrando entorno...")
                    agent.env.close()
                    print("🔒 Entorno cerrado")
                except Exception as e:
                    print(f"⚠️ Error cerrando entorno: {e}")
                
                # SALIDA FORZADA INMEDIATA - SIN DEMORA
                print("� EJECUTANDO SALIDA FORZADA INMEDIATA...")
                
                # Usar múltiples métodos de salida en paralelo
                import threading
                import time as time_mod
                
                def force_exit_delayed():
                    time_mod.sleep(0.1)  # Pequeña demora para permitir print
                    force_exit()
                
                # Crear thread para salida forzada
                exit_thread = threading.Thread(target=force_exit_delayed)
                exit_thread.daemon = True
                exit_thread.start()
                
                # También intentar salida inmediata
                force_exit()
            
            # Reset si el episodio termina
            if terminated or truncated:
                print(f"[{step}] Episode terminado, reiniciando...")
                observation, info = agent.env.reset()
                agent.agent.reset()
                
    except KeyboardInterrupt:
        print("\n⚠️ Interrumpido por usuario")
        elapsed = time.time() - start_time
        save_simple_metrics(step, episode_reward, elapsed)
        try:
            agent.env.close()
        except:
            pass
        print("👋 Cerrando...")
        force_exit()
    
    except Exception as e:
        print(f"\n❌ Error inesperado: {e}")
        try:
            agent.env.close()
        except:
            pass
        force_exit()

    # Si llegamos aquí sin detectar Pokémon (no debería pasar)
    print("⚠️ Límite de pasos alcanzado sin detectar Pokémon")
    try:
        agent.env.close()
    except:
        pass
    force_exit()