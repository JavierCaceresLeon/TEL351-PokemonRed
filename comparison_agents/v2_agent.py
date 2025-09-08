"""
Wrapper para el agente entrenado de v2
Permite usar el modelo PPO entrenado con la misma interfaz que los algoritmos de búsqueda
"""

import sys
import os
import time
import glob
from pathlib import Path

# Agregar el directorio v2 al path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'v2'))

try:
    from stable_baselines3 import PPO
    from red_gym_env_v2 import RedGymEnv
except ImportError as e:
    print(f"Error importing v2 dependencies: {e}")
    print("Make sure stable_baselines3 and other dependencies are installed")

class V2TrainedAgent:
    """Wrapper para el agente entrenado con PPO en v2"""
    
    def __init__(self, environment_config=None, model_path=None):
        self.env_config = environment_config or {
            'headless': True,
            'save_final_state': False,
            'early_stop': False,
            'action_freq': 24,
            'init_state': '../init.state',
            'max_steps': 1000,
            'print_rewards': False,
            'save_video': False,
            'fast_video': True,
            'session_path': Path("v2_agent_session"),
            'gb_path': '../PokemonRed.gb',
            'debug': False,
            'reward_scale': 0.5,
            'explore_weight': 0.25
        }
        
        # Inicializar entorno
        self.env = None
        self.model = None
        self.model_path = model_path
        
        self.search_stats = {
            'steps_taken': 0,
            'execution_time': 0,
            'success': False,
            'model_loaded': False
        }
        
        self._load_model()
        self._setup_environment()
    
    def _find_latest_model(self):
        """Buscar el modelo más reciente en v2/runs"""
        runs_dir = Path(__file__).parent.parent.parent / "v2" / "runs"
        
        if not runs_dir.exists():
            print(f"Directorio runs no encontrado: {runs_dir}")
            return None
        
        # Buscar archivos .zip en runs
        zip_files = list(runs_dir.glob("*.zip"))
        
        if not zip_files:
            print("No se encontraron modelos .zip en v2/runs")
            return None
        
        # Obtener el más reciente
        latest_model = max(zip_files, key=os.path.getmtime)
        print(f"Modelo más reciente encontrado: {latest_model}")
        return str(latest_model)
    
    def _load_model(self):
        """Cargar el modelo PPO entrenado"""
        try:
            if self.model_path is None:
                self.model_path = self._find_latest_model()
            
            if self.model_path is None:
                raise FileNotFoundError("No se encontró ningún modelo entrenado")
            
            print(f"Cargando modelo desde: {self.model_path}")
            self.model = PPO.load(self.model_path)
            self.search_stats['model_loaded'] = True
            print("Modelo cargado exitosamente")
            
        except Exception as e:
            print(f"Error cargando modelo: {e}")
            self.search_stats['model_loaded'] = False
    
    def _setup_environment(self):
        """Configurar el entorno"""
        try:
            self.env = RedGymEnv(self.env_config)
            print("Entorno v2 configurado exitosamente")
        except Exception as e:
            print(f"Error configurando entorno: {e}")
            self.env = None
    
    def get_state_key(self, obs):
        """Generar clave del estado para comparación"""
        # Simplificado - en una implementación real podríamos extraer más información
        return str(hash(obs.tobytes() if isinstance(obs, type(None)) else str(obs)))
    
    def search(self):
        """
        'Búsqueda' usando el modelo entrenado
        Retorna la secuencia de acciones hasta salir de la habitación
        """
        if not self.model or not self.env:
            print("Modelo o entorno no disponible")
            return []
        
        start_time = time.time()
        actions_taken = []
        
        try:
            # Reset del entorno
            obs, info = self.env.reset()
            done = False
            step_count = 0
            max_steps = self.env_config['max_steps']
            
            # Obtener posición inicial
            initial_map_id = self.env.read_m(0xD35E)
            
            while not done and step_count < max_steps:
                # Predecir acción usando el modelo
                action, _states = self.model.predict(obs, deterministic=True)
                actions_taken.append(int(action))
                
                # Ejecutar acción
                obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                step_count += 1
                
                # Verificar si salió de la habitación inicial
                current_map_id = self.env.read_m(0xD35E)
                if current_map_id != initial_map_id:
                    print(f"¡Salió de la habitación inicial en {step_count} pasos!")
                    self.search_stats['success'] = True
                    break
                
                # Verificar recompensa alta (posible éxito)
                if reward > 50:
                    self.search_stats['success'] = True
                    break
            
            self.search_stats['steps_taken'] = step_count
            self.search_stats['execution_time'] = time.time() - start_time
            
            print(f"Agente v2 completó en {step_count} pasos")
            return actions_taken
            
        except Exception as e:
            print(f"Error durante la búsqueda con modelo v2: {e}")
            self.search_stats['execution_time'] = time.time() - start_time
            return actions_taken
    
    def execute_plan(self, plan=None):
        """
        Ejecutar un plan específico o usar el modelo para generar uno
        """
        if plan is None:
            plan = self.search()
        
        start_time = time.time()
        
        results = {
            'steps': len(plan),
            'success': self.search_stats['success'],
            'execution_time': time.time() - start_time,
            'final_state': None
        }
        
        return results
    
    def get_stats(self):
        """Obtener estadísticas del agente"""
        return self.search_stats.copy()
    
    def close(self):
        """Cerrar recursos"""
        if self.env:
            try:
                self.env.close()
            except:
                pass
