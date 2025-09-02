#!/usr/bin/env python3
"""
Script para ejecutar sesiones controladas de Pokemon Red
que guarden datos completos sin interrupción manual
"""

import sys
import time
import signal
from pathlib import Path
from red_gym_env_v2 import RedGymEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import uuid

class ControlledSession:
    def __init__(self, max_steps=10000, save_frequency=500):
        self.max_steps = max_steps
        self.save_frequency = save_frequency
        self.running = True
        
        # Configurar handler para Ctrl+C
        signal.signal(signal.SIGINT, self.signal_handler)
        
    def signal_handler(self, signum, frame):
        print(f"\n\nSeñal de interrupción recibida. Guardando sesión...")
        self.running = False
        
    def run_session(self, checkpoint_path=None):
        """Ejecuta una sesión controlada con guardado automático"""
        
        session_id = str(uuid.uuid4())[:8]
        sess_path = Path(f'v2/session_{session_id}')
        sess_path.mkdir(exist_ok=True)
        
        print(f"Iniciando sesión controlada: {session_id}")
        print(f"Directorio de sesión: {sess_path}")
        print(f"Pasos máximos: {self.max_steps}")
        print(f"Guardado automático cada: {self.save_frequency} pasos")
        print("Presiona Ctrl+C para terminar graciosamente\n")
        
        # Configuración del entorno
        env_config = {
            'headless': False, 
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
            'reward_scale': 0.5, 
            'explore_weight': 0.25
        }
        
        # Crear entorno
        env = RedGymEnv(env_config)
        env = DummyVecEnv([lambda: env])
        
        # Cargar modelo
        if checkpoint_path is None:
            checkpoint_path = 'v2/runs/poke_26214400'
            
        print(f"Cargando modelo: {checkpoint_path}")
        try:
            model = PPO.load(checkpoint_path, env=env)
            print("Modelo cargado exitosamente")
        except Exception as e:
            print(f"Error cargando modelo: {e}")
            return
            
        # Ejecutar sesión
        obs = env.reset()
        step_count = 0
        last_save = 0
        
        start_time = time.time()
        
        try:
            while self.running and step_count < self.max_steps:
                # Predicción del modelo
                action, _ = model.predict(obs, deterministic=False)
                
                # Ejecutar acción
                obs, rewards, dones, info = env.step(action)
                step_count += 1
                
                # Guardado automático
                if step_count - last_save >= self.save_frequency:
                    self.save_progress(sess_path, step_count, env, start_time)
                    last_save = step_count
                    
                # Reiniciar si termina el episodio
                if dones[0]:
                    print(f"\nEpisodio completado en paso {step_count}")
                    obs = env.reset()
                    
        except KeyboardInterrupt:
            print("\nInterrupción detectada por KeyboardInterrupt")
            
        # Guardado final
        self.save_progress(sess_path, step_count, env, start_time, final=True)
        
        elapsed_time = time.time() - start_time
        print(f"\nSesión completada:")
        print(f"  Pasos ejecutados: {step_count}")
        print(f"  Tiempo transcurrido: {elapsed_time:.1f} segundos")
        print(f"  Pasos por segundo: {step_count/elapsed_time:.1f}")
        print(f"  Datos guardados en: {sess_path}")
        
        env.close()
        
    def save_progress(self, sess_path, step_count, env, start_time, final=False):
        """Guarda el progreso actual"""
        elapsed = time.time() - start_time
        
        if final:
            print(f"\nGuardado final - Paso {step_count} ({elapsed:.1f}s)")
        else:
            print(f"\nGuardado automático - Paso {step_count} ({elapsed:.1f}s)")
            
        # Aquí se pueden agregar más operaciones de guardado
        # como screenshots, estadísticas adicionales, etc.

def main():
    if len(sys.argv) > 1:
        try:
            max_steps = int(sys.argv[1])
        except ValueError:
            print("Error: El primer argumento debe ser un número (max_steps)")
            return
    else:
        max_steps = 10000
        
    save_freq = 500
    if len(sys.argv) > 2:
        try:
            save_freq = int(sys.argv[2])
        except ValueError:
            print("Error: El segundo argumento debe ser un número (save_frequency)")
            return
            
    checkpoint = None
    if len(sys.argv) > 3:
        checkpoint = sys.argv[3]
        
    print("=== Sesión Controlada de Pokemon Red ===")
    print(f"Uso: python {sys.argv[0]} [max_steps] [save_frequency] [checkpoint_path]")
    print(f"Parámetros actuales:")
    print(f"  max_steps: {max_steps}")
    print(f"  save_frequency: {save_freq}")
    print(f"  checkpoint: {checkpoint or 'v2/runs/poke_26214400'}")
    print()
    
    session = ControlledSession(max_steps=max_steps, save_frequency=save_freq)
    session.run_session(checkpoint_path=checkpoint)

if __name__ == "__main__":
    main()
