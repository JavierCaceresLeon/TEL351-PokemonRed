"""
Archivo de configuración para la comparación de agentes
Centraliza todos los parámetros configurables
"""

from pathlib import Path

class ComparisonConfig:
    """Configuración centralizada para la comparación de agentes"""
    
    # Configuración de archivos
    INIT_STATE = '../init.state'
    GB_PATH = '../PokemonRed.gb'
    V2_RUNS_DIR = '../v2/runs'
    
    # Configuración general del experimento
    NUM_RUNS = 3  # Número de ejecuciones por agente
    MAX_STEPS = 500  # Máximo de pasos por intento
    HEADLESS = True  # Ejecutar sin interfaz gráfica
    
    # Configuración específica de A*
    ASTAR_CONFIG = {
        'max_search_depth': 500,  # Máximo de nodos a explorar
        'heuristic_weight': 1.0,  # Peso de la heurística (1.0 = A* estándar)
    }
    
    # Configuración específica de Tabú Search
    TABU_CONFIG = {
        'max_iterations': 200,  # Máximo de iteraciones
        'tabu_size': 30,        # Tamaño de la lista tabú
        'neighbor_count': 8,    # Número de vecinos a generar
        'max_solution_length': 50,  # Máximo de acciones en una solución
    }
    
    # Configuración específica del agente V2
    V2_CONFIG = {
        'action_freq': 24,      # Frecuencia de acción (frames por acción)
        'reward_scale': 0.5,    # Escala de recompensas
        'explore_weight': 0.25, # Peso de exploración
        'deterministic': True,  # Predicciones deterministas
    }
    
    # Configuración de resultados
    RESULTS_CONFIG = {
        'save_detailed_logs': True,     # Guardar logs detallados
        'save_action_sequences': True,  # Guardar secuencias de acciones
        'create_visualizations': False, # Crear gráficos (requiere matplotlib)
    }
    
    # Configuración de debugging
    DEBUG_CONFIG = {
        'verbose': True,           # Mostrar información detallada
        'save_intermediate_states': False,  # Guardar estados intermedios
        'log_actions': False,      # Log de cada acción
    }
    
    @classmethod
    def get_search_env_config(cls):
        """Configuración para el entorno de búsqueda"""
        return {
            'init_state': cls.INIT_STATE,
            'gb_path': cls.GB_PATH,
            'headless': cls.HEADLESS,
            'max_steps': cls.MAX_STEPS,
            'session_path': Path('comparison_session'),
        }
    
    @classmethod
    def get_v2_env_config(cls, session_suffix=""):
        """Configuración para el entorno V2"""
        return {
            'headless': cls.HEADLESS,
            'save_final_state': False,
            'early_stop': False,
            'action_freq': cls.V2_CONFIG['action_freq'],
            'init_state': cls.INIT_STATE,
            'max_steps': cls.MAX_STEPS,
            'print_rewards': cls.DEBUG_CONFIG['verbose'],
            'save_video': False,
            'fast_video': True,
            'session_path': Path(f'comparison_session_v2{session_suffix}'),
            'gb_path': cls.GB_PATH,
            'debug': cls.DEBUG_CONFIG['verbose'],
            'reward_scale': cls.V2_CONFIG['reward_scale'],
            'explore_weight': cls.V2_CONFIG['explore_weight'],
        }
    
    @classmethod
    def validate_config(cls):
        """Validar que la configuración es válida"""
        errors = []
        
        # Verificar archivos requeridos
        if not Path(cls.INIT_STATE).exists():
            errors.append(f"Estado inicial no encontrado: {cls.INIT_STATE}")
        
        if not Path(cls.GB_PATH).exists():
            errors.append(f"ROM no encontrada: {cls.GB_PATH}")
        
        # Verificar parámetros numéricos
        if cls.NUM_RUNS <= 0:
            errors.append("NUM_RUNS debe ser mayor que 0")
        
        if cls.MAX_STEPS <= 0:
            errors.append("MAX_STEPS debe ser mayor que 0")
        
        # Verificar configuración de A*
        if cls.ASTAR_CONFIG['max_search_depth'] <= 0:
            errors.append("max_search_depth de A* debe ser mayor que 0")
        
        # Verificar configuración de Tabú
        if cls.TABU_CONFIG['max_iterations'] <= 0:
            errors.append("max_iterations de Tabú debe ser mayor que 0")
        
        if cls.TABU_CONFIG['tabu_size'] <= 0:
            errors.append("tabu_size debe ser mayor que 0")
        
        return errors

# Configuraciones predefinidas para diferentes escenarios

class QuickTestConfig(ComparisonConfig):
    """Configuración para pruebas rápidas"""
    NUM_RUNS = 1
    MAX_STEPS = 100
    
    ASTAR_CONFIG = {
        'max_search_depth': 100,
        'heuristic_weight': 1.0,
    }
    
    TABU_CONFIG = {
        'max_iterations': 50,
        'tabu_size': 15,
        'neighbor_count': 5,
        'max_solution_length': 25,
    }

class IntensiveTestConfig(ComparisonConfig):
    """Configuración para pruebas intensivas"""
    NUM_RUNS = 10
    MAX_STEPS = 2000
    
    ASTAR_CONFIG = {
        'max_search_depth': 2000,
        'heuristic_weight': 1.0,
    }
    
    TABU_CONFIG = {
        'max_iterations': 1000,
        'tabu_size': 100,
        'neighbor_count': 15,
        'max_solution_length': 200,
    }

class DebugConfig(ComparisonConfig):
    """Configuración para debugging"""
    NUM_RUNS = 1
    MAX_STEPS = 50
    HEADLESS = False  # Mostrar ventana del juego
    
    DEBUG_CONFIG = {
        'verbose': True,
        'save_intermediate_states': True,
        'log_actions': True,
    }
    
    RESULTS_CONFIG = {
        'save_detailed_logs': True,
        'save_action_sequences': True,
        'create_visualizations': True,
    }
