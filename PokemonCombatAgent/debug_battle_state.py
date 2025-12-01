"""
Script de diagnóstico profundo para entender por qué el agente no pelea.
Ejecuta el modelo paso a paso mostrando:
- Frame actual del juego
- Acción predicha con probabilidades
- Estado de memoria del juego
- Distribución de acciones
"""

import numpy as np
from pathlib import Path
from red_gym_env_v2 import RedGymEnv
from battle_only_actions import BattleOnlyActions
from stable_baselines3 import PPO
import matplotlib.pyplot as plt
from collections import Counter
import json

def analyze_game_state(pyboy):
    """Analiza el estado actual del juego en detalle."""
    memory = pyboy.memory
    
    # Direcciones de memoria críticas
    party_count = memory[0xD163]
    in_battle = memory[0xD057]
    battle_type = memory[0xD05A]
    
    # HP del jugador
    player_hp = memory[0xD016] << 8 | memory[0xD017]
    player_max_hp = memory[0xD018] << 8 | memory[0xD019]
    
    # HP del enemigo
    enemy_hp = memory[0xCFE7] << 8 | memory[0xCFE8]
    enemy_max_hp = memory[0xCFE9] << 8 | memory[0xCFEA]
    
    # Estado del diálogo/texto
    text_box_active = memory[0xCC57]
    menu_selection = memory[0xCC24]
    
    state = {
        'in_battle': in_battle,
        'battle_type': battle_type,
        'player_hp': player_hp,
        'player_max_hp': player_max_hp,
        'enemy_hp': enemy_hp,
        'enemy_max_hp': enemy_max_hp,
        'text_box_active': text_box_active,
        'menu_selection': menu_selection,
        'party_count': party_count
    }
    
    return state

def print_state_info(state, step):
    """Imprime información del estado de forma legible."""
    print(f"\n{'='*60}")
    print(f"STEP {step}")
    print(f"{'='*60}")
    print(f"En batalla: {state['in_battle'] != 0}")
    print(f"Tipo de batalla: {state['battle_type']}")
    print(f"HP Jugador: {state['player_hp']}/{state['player_max_hp']}")
    print(f"HP Enemigo: {state['enemy_hp']}/{state['enemy_max_hp']}")
    print(f"Texto activo: {state['text_box_active']}")
    print(f"Selección menú: {state['menu_selection']}")
    print(f"Pokemon en party: {state['party_count']}")

def run_diagnostic(model_path, state_file, max_steps=100, render=False):
    """Ejecuta diagnóstico detallado del modelo."""
    
    print(f"\n{'='*70}")
    print(f"DIAGNÓSTICO DE BATALLA - Combat Agent")
    print(f"{'='*70}")
    print(f"Modelo: {model_path}")
    print(f"State file: {state_file}")
    print(f"Max steps: {max_steps}")
    print(f"{'='*70}\n")
    
    # Crear ambiente
    config = {
        'headless': not render,
        'init_state': state_file,
        'action_freq': 24,
        'max_steps': max_steps,
        'save_final_state': False,
        'print_rewards': False,
        'gb_path': 'PokemonRed.gb',
        'session_path': Path('diagnostic_session'),
        'debug': False,
        'save_video': False,
        'fast_video': False,
        'video_interval': 256 * 60 * 1,
        'frame_stacks': 3,
        'explore_weight': 1,
        'use_screen_explore': True,
        'similar_frame_dist': 500
    }
    
    env = RedGymEnv(config)
    env = BattleOnlyActions(env)
    
    # Cargar modelo
    print("Cargando modelo...")
    model = PPO.load(model_path)
    print("✓ Modelo cargado\n")
    
    # Reset
    obs, _ = env.reset()
    
    # Análisis inicial
    print("\n" + "="*60)
    print("ESTADO INICIAL DEL JUEGO")
    print("="*60)
    initial_state = analyze_game_state(env.pyboy)
    print_state_info(initial_state, 0)
    
    # Guardar acciones
    actions_taken = []
    states_log = []
    action_probs_log = []
    
    for step in range(max_steps):
        # Predecir acción
        action, _states = model.predict(obs, deterministic=False)
        
        # Obtener probabilidades de todas las acciones
        obs_tensor = model.policy.obs_to_tensor(obs)[0]
        with np.errstate(all='ignore'):
            distribution = model.policy.get_distribution(obs_tensor)
            probs = distribution.distribution.probs.detach().cpu().numpy()[0]
        
        # Ejecutar acción
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Analizar estado
        current_state = analyze_game_state(env.pyboy)
        
        # Guardar datos
        actions_taken.append(int(action))
        states_log.append(current_state)
        action_probs_log.append(probs.tolist())
        
        # Mostrar cada 10 pasos o si hay cambio significativo
        if step % 10 == 0 or step < 10:
            print_state_info(current_state, step)
            print(f"\nAcción tomada: {action} ({'A' if action == 0 else 'UP' if action == 1 else 'DOWN'})")
            print(f"Probabilidades: A={probs[0]:.3f}, UP={probs[1]:.3f}, DOWN={probs[2]:.3f}")
            print(f"Reward: {reward:.2f}")
        
        if terminated or truncated:
            print(f"\n{'='*60}")
            print(f"EPISODIO TERMINADO EN STEP {step}")
            print(f"Terminated: {terminated}, Truncated: {truncated}")
            print(f"{'='*60}")
            break
    
    env.close()
    
    # Análisis de acciones
    print(f"\n{'='*70}")
    print("ANÁLISIS DE DISTRIBUCIÓN DE ACCIONES")
    print(f"{'='*70}")
    
    action_counts = Counter(actions_taken)
    total = len(actions_taken)
    
    action_names = {0: 'A (Confirmar)', 1: 'UP (Subir)', 2: 'DOWN (Bajar)'}
    
    for action_id in sorted(action_counts.keys()):
        count = action_counts[action_id]
        pct = (count / total) * 100
        print(f"{action_names[action_id]}: {count:4d} veces ({pct:5.1f}%)")
        print(f"  {'█' * int(pct)}")
    
    # Análisis de probabilidades promedio
    print(f"\n{'='*70}")
    print("PROBABILIDADES PROMEDIO DE CADA ACCIÓN")
    print(f"{'='*70}")
    
    avg_probs = np.mean(action_probs_log, axis=0)
    for i, prob in enumerate(avg_probs):
        print(f"{action_names[i]}: {prob:.3f} ({prob*100:.1f}%)")
    
    # Verificar si el estado cambió
    print(f"\n{'='*70}")
    print("CAMBIOS EN ESTADO DEL JUEGO")
    print(f"{'='*70}")
    
    initial = states_log[0]
    final = states_log[-1]
    
    print(f"HP Jugador:  {initial['player_hp']} → {final['player_hp']} (Δ: {final['player_hp'] - initial['player_hp']})")
    print(f"HP Enemigo:  {initial['enemy_hp']} → {final['enemy_hp']} (Δ: {final['enemy_hp'] - initial['enemy_hp']})")
    print(f"En batalla:  {initial['in_battle']} → {final['in_battle']}")
    print(f"Texto activo: {initial['text_box_active']} → {final['text_box_active']}")
    
    # Guardar diagnóstico completo
    diagnostic_data = {
        'model_path': str(model_path),
        'state_file': str(state_file),
        'total_steps': len(actions_taken),
        'actions': actions_taken,
        'action_distribution': dict(action_counts),
        'avg_probabilities': avg_probs.tolist(),
        'initial_state': {k: int(v) if isinstance(v, (np.integer, np.int64)) else v 
                         for k, v in initial.items()},
        'final_state': {k: int(v) if isinstance(v, (np.integer, np.int64)) else v 
                       for k, v in final.items()},
        'states_timeline': [{k: int(v) if isinstance(v, (np.integer, np.int64)) else v 
                            for k, v in s.items()} for s in states_log]
    }
    
    output_file = Path('diagnostic_results') / 'battle_diagnostic.json'
    output_file.parent.mkdir(exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(diagnostic_data, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"✓ Diagnóstico guardado en: {output_file}")
    print(f"{'='*70}\n")
    
    return diagnostic_data

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Diagnóstico detallado de batalla')
    parser.add_argument('--model', type=str, 
                       default='sessions/combat_agent_final_battle_loop/combat_agent_final_battle_loop.zip',
                       help='Ruta al modelo')
    parser.add_argument('--state', type=str,
                       default='generated_battle_states/clean_pewter_gym.state',
                       help='Archivo de estado')
    parser.add_argument('--steps', type=int, default=100,
                       help='Máximo de pasos')
    parser.add_argument('--render', action='store_true',
                       help='Mostrar ventana del juego')
    
    args = parser.parse_args()
    
    run_diagnostic(
        model_path=args.model,
        state_file=args.state,
        max_steps=args.steps,
        render=args.render
    )
