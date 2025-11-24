"""
Script interactivo para ejecutar el CombatApexAgent entrenado.
Equivalente a run_pretrained_interactive.py pero para tu modelo especializado.

Uso:
    python run_combat_agent_interactive.py --scenario pewter_brock --phase battle
"""

import argparse
import os
import sys
import json
from pathlib import Path

# Configuración de rutas
project_path = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_path))
sys.path.insert(0, str(project_path / 'baselines'))

# Fix OpenMP conflict
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from stable_baselines3 import PPO
from advanced_agents.combat_apex_agent import CombatApexAgent, CombatAgentConfig
from advanced_agents.train_agents import _base_env_config


def load_scenarios():
    """Carga los escenarios disponibles desde scenarios.json"""
    scenario_path = project_path / 'gym_scenarios' / 'scenarios.json'
    with open(scenario_path, 'r') as f:
        data = json.load(f)
    return {s['id']: s for s in data['scenarios']}


def resolve_phase(scenarios, scenario_id, phase_name):
    """Encuentra la fase específica en un escenario"""
    scenario = scenarios.get(scenario_id)
    if not scenario:
        raise ValueError(f"Escenario '{scenario_id}' no encontrado")
    
    phase = next((p for p in scenario['phases'] if p['name'] == phase_name), None)
    if not phase:
        raise ValueError(f"Fase '{phase_name}' no encontrada en {scenario_id}")
    
    return phase


def ensure_state_file(state_file_path):
    """Verifica que el archivo .state exista"""
    abs_path = project_path / state_file_path
    if not abs_path.exists():
        raise FileNotFoundError(
            f"Archivo de estado no encontrado: {abs_path}\n"
            "Genera los archivos .state con generate_gym_states.py"
        )
    return str(abs_path)


def build_env_config(state_file_path, headless=False):
    """Construye la configuración del entorno"""
    env_overrides = {
        'init_state': state_file_path,
        'headless': headless,
        'save_video': False,
        'gb_path': str(project_path / 'PokemonRed.gb'),
        'session_path': str(project_path / 'sessions' / f"interactive_{Path(state_file_path).name}"),
        'render_mode': 'human' if not headless else 'rgb_array',
        'fast_video': False  # Siempre False en modo interactivo para mejor visualización
    }
    return _base_env_config(env_overrides)


def run_interactive(scenario_id, phase_name, headless=False, max_steps=10000):
    """
    Ejecuta el agente de combate entrenado en modo interactivo.
    
    Args:
        scenario_id: ID del escenario (ej: 'pewter_brock')
        phase_name: Nombre de la fase (ej: 'battle')
        headless: Si True, no muestra ventana del emulador
        max_steps: Máximo de pasos por episodio
    """
    print(f"\n{'='*60}")
    print(f"  EJECUTANDO COMBAT APEX AGENT - MODO INTERACTIVO")
    print(f"  Escenario: {scenario_id} | Fase: {phase_name}")
    print(f"{'='*60}\n")
    
    # 1. Cargar escenarios y resolver fase
    scenarios = load_scenarios()
    phase = resolve_phase(scenarios, scenario_id, phase_name)
    state_file_path = ensure_state_file(phase['state_file'])
    
    # 2. Configurar entorno
    env_config = build_env_config(state_file_path, headless=headless)
    agent_config = CombatAgentConfig(
        env_config=env_config,
        total_timesteps=1000  # No importa para inferencia
    )
    
    # 3. Cargar modelo entrenado
    model_path = project_path / 'models_local' / 'combat' / f"{scenario_id}_{phase_name}.zip"
    
    if not model_path.exists():
        raise FileNotFoundError(
            f"Modelo no encontrado: {model_path}\n"
            f"Entrena primero el modelo usando Local_Train.ipynb"
        )
    
    print(f"Cargando modelo desde: {model_path}")
    agent_wrapper = CombatApexAgent(agent_config)
    env = agent_wrapper.make_env()
    model = PPO.load(str(model_path))
    
    # Mostrar información del modelo
    try:
        num_timesteps = model.num_timesteps
        print(f"   Pasos de entrenamiento del modelo: {num_timesteps:,}")
        if num_timesteps < 10000:
            print(f"   ADVERTENCIA: Modelo sub-entrenado (<10k pasos)")
            print(f"   Recomendación: Entrena con 40,000+ pasos en Local_Train.ipynb")
    except:
        print(f"No se pudo determinar los pasos de entrenamiento")
    
    # Acceder al entorno base para obtener PyBoy
    base_env = env
    while hasattr(base_env, 'env'):
        base_env = base_env.env
    
    # Verificar si tenemos acceso a PyBoy
    if hasattr(base_env, 'pyboy') and not headless:
        print(f"Ventana de PyBoy inicializada (SDL2)")
    elif headless:
        print(f"Modo headless activado (sin ventana)")
    
    # 4. Ejecutar episodio
    if not headless:
        print(f"\nIniciando episodio (máx {max_steps} pasos)...")
        print(f"BUSCA la ventana de PyBoy (puede estar minimizada o detrás de otras ventanas)")
        print(f"   Título: 'PyBoy' o 'Pok\u00e9mon Red'\n")
    else:
        print(f"\nIniciando episodio en modo headless (máx {max_steps} pasos)...\n")
    
    # Reset puede retornar (obs,) o (obs, info) dependiendo del wrapper
    reset_result = env.reset()
    obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result
    done = False
    truncated = False
    step_count = 0
    total_reward = 0
    
    try:
        while not done and not truncated and step_count < max_steps:
            # Predicción determinista (sin exploración)
            action, _states = model.predict(obs, deterministic=True)
            step_result = env.step(action)
            
            # Renderizar frame si no es headless
            if not headless and hasattr(env, 'render'):
                env.render()
            
            # Manejar diferentes formatos de retorno
            if len(step_result) == 5:
                obs, reward, done, truncated, info = step_result
            elif len(step_result) == 4:
                obs, reward, done, info = step_result
                truncated = False
            else:
                raise ValueError(f"Formato inesperado de step(): {len(step_result)} valores")
            
            # Convertir reward a escalar (puede ser numpy array)
            reward_scalar = float(reward.item() if hasattr(reward, 'item') else reward)
            total_reward += reward_scalar
            step_count += 1
            
            # Mostrar progreso cada 100 pasos con más detalles
            if step_count % 100 == 0:
                # Convertir action a entero si es array
                action_int = int(action.item() if hasattr(action, 'item') else action)
                action_names = ['DOWN', 'LEFT', 'RIGHT', 'UP', 'A', 'B', 'START']
                action_str = action_names[action_int] if action_int < len(action_names) else str(action_int)
                print(f"Paso {step_count}/{max_steps} | Reward: {total_reward:.2f} (+{reward_scalar:.2f}) | Acción: {action_str}")
            
            # Detectar eventos importantes del info dict
            if 'event' in info:
                print(f"Evento: {info['event']}")
    
    except KeyboardInterrupt:
        print("\n\nEjecución interrumpida por el usuario (Ctrl+C)")
    
    finally:
        env.close()
        
        print(f"\n{'='*60}")
        print(f"  RESUMEN DEL EPISODIO")
        print(f"{'='*60}")
        print(f"  Pasos ejecutados: {step_count}")
        print(f"  Reward total: {total_reward:.2f}")
        print(f"  Reward promedio/paso: {total_reward/step_count:.4f}" if step_count > 0 else "  N/A")
        print(f"  Estado final: {'Completado' if done else 'Truncado' if truncated else 'Máx pasos alcanzado'}")
        print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Ejecuta un agente Combat Apex entrenado en modo interactivo'
    )
    parser.add_argument(
        '--scenario',
        type=str,
        default='pewter_brock',
        help='ID del escenario (default: pewter_brock)'
    )
    parser.add_argument(
        '--phase',
        type=str,
        default='battle',
        help='Nombre de la fase (default: battle)'
    )
    parser.add_argument(
        '--headless',
        action='store_true',
        help='Ejecutar sin ventana del emulador'
    )
    parser.add_argument(
        '--max-steps',
        type=int,
        default=10000,
        help='Máximo de pasos por episodio (default: 10000)'
    )
    
    args = parser.parse_args()
    
    run_interactive(
        scenario_id=args.scenario,
        phase_name=args.phase,
        headless=args.headless,
        max_steps=args.max_steps
    )


if __name__ == '__main__':
    main()
