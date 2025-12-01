"""
Usa el modelo baseline para llegar a un gimnasio y guardar el estado JUSTO antes de batalla
"""

import argparse
from pathlib import Path
from red_gym_env_v2 import RedGymEnv
from stable_baselines3 import PPO
import time

def create_env(init_state, headless=False):
    """Crear ambiente para navegación"""
    config = {
        'headless': headless,
        'save_final_state': True,
        'early_stop': False,
        'action_freq': 24,
        'init_state': init_state,
        'max_steps': 50000,  # Suficiente para llegar a gimnasio
        'print_rewards': False,
        'save_video': False,
        'fast_video': False,
        'session_path': Path('temp_session'),
        'gb_path': 'PokemonRed.gb',
        'debug': False,
        'sim_frame_dist': 2_000_000.0,
        'extra_buttons': False
    }
    return RedGymEnv(config)

def save_state_at_position(pyboy, save_path, description):
    """Guardar estado del juego"""
    with open(save_path, "wb") as f:
        pyboy.save_state(f)
    print(f"✅ Estado guardado: {save_path}")
    print(f"   Descripción: {description}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, 
                        default='../PokemonRedExperiments/v2/runs/poke_26214400.zip',
                        help='Modelo baseline preentrenado')
    parser.add_argument('--init-state', type=str, 
                        default='has_pokedex_nballs.state',
                        help='Estado inicial')
    parser.add_argument('--target-gym', type=str, choices=['pewter', 'cerulean', 'vermilion'],
                        default='pewter',
                        help='Gimnasio objetivo')
    parser.add_argument('--headless', action='store_true',
                        help='Ejecutar sin ventana')
    
    args = parser.parse_args()
    
    print("="*60)
    print("🎮 Generador de Estados de Batalla Limpios")
    print("="*60)
    print(f"Modelo: {args.model}")
    print(f"Estado inicial: {args.init_state}")
    print(f"Gimnasio objetivo: {args.target_gym}")
    print("="*60)
    print()
    
    # Coordenadas de gimnasios (X, Y, Map)
    gym_locations = {
        'pewter': {
            'map_id': 52,  # Pewter Gym
            'name': 'Pewter City Gym (Brock)',
            'leader_map': 52,
            'save_name': 'clean_pewter_gym.state'
        },
        'cerulean': {
            'map_id': 65,  # Cerulean Gym
            'name': 'Cerulean City Gym (Misty)',
            'leader_map': 65,
            'save_name': 'clean_cerulean_gym.state'
        },
        'vermilion': {
            'map_id': 92,  # Vermilion Gym
            'name': 'Vermilion City Gym (Lt. Surge)',
            'leader_map': 92,
            'save_name': 'clean_vermilion_gym.state'
        }
    }
    
    target = gym_locations[args.target_gym]
    
    # Crear ambiente
    print("Cargando ambiente y modelo...")
    env = create_env(args.init_state, args.headless)
    model = PPO.load(args.model, env=env, custom_objects={'lr_schedule': 0, 'clip_range': 0})
    print("✅ Modelo cargado\n")
    
    obs, info = env.reset()
    
    # Directorio para guardar estados
    output_dir = Path('generated_battle_states')
    output_dir.mkdir(exist_ok=True)
    
    print(f"🎯 Objetivo: Detectar INICIO de batalla en {target['name']}")
    print("Ejecutando modelo baseline para navegar...")
    print("Presiona 'S' cuando veas que ESTÁ EN BATALLA\n")
    
    step = 0
    last_map = -1
    last_battle_type = 0
    states_saved = []
    in_gym = False
    battle_just_started = False
    frames_in_battle = 0
    
    try:
        while step < 50000:
            # Leer posición actual ANTES de step
            current_map = env.read_m(0xD35E)  # Map ID
            x_pos = env.read_m(0xD362)  # X position
            y_pos = env.read_m(0xD361)  # Y position
            battle_type = env.read_m(0xD057)  # Battle type (0=none, 1=wild, 2=trainer)
            
            # Detectar cambio de mapa
            if current_map != last_map:
                print(f"Step {step:6d} | Mapa: {current_map:3d} | Pos: ({x_pos:3d}, {y_pos:3d}) | Battle: {battle_type}")
                last_map = current_map
                
                # Marcar si estamos en el gimnasio
                if current_map == target['map_id']:
                    if not in_gym:
                        print(f"\n✅ Entramos a {target['name']}")
                        print(f"   Esperando que entre en batalla...\n")
                        in_gym = True
                else:
                    if in_gym:
                        print(f"⚠️  Salió del gimnasio\n")
                    in_gym = False
                    frames_in_battle = 0
            
            # DETECTAR INICIO DE BATALLA (transición de 0 a 1 o 2)
            if battle_type > 0 and last_battle_type == 0:
                battle_just_started = True
                frames_in_battle = 0
                print(f"\n⚔️  ¡BATALLA DETECTADA en step {step}!")
                print(f"   Battle type: {battle_type} (1=wild, 2=trainer)")
                print(f"   Mapa: {current_map}")
                print(f"   En gimnasio: {in_gym}")
            
            # Si estamos en batalla, contar frames
            if battle_type > 0:
                frames_in_battle += 1
                
                # Guardar después de 5 frames en batalla (más estable)
                if battle_just_started and frames_in_battle == 5:
                    print(f"   Estabilizando batalla... ({frames_in_battle} frames)")
                    
                    # Verificar que REALMENTE estamos en batalla
                    check_battle = env.read_m(0xD057)
                    if check_battle > 0:
                        save_path = output_dir / f"{target['save_name']}"
                        save_state_at_position(
                            env.pyboy,
                            save_path,
                            f"{target['name']} - BATALLA ACTIVA - Step {step}"
                        )
                        states_saved.append(str(save_path))
                        
                        print(f"\n🎉 Estado guardado CON BATALLA ACTIVA!")
                        print(f"   Battle type confirmado: {check_battle}")
                        print("Deteniendo ejecución...\n")
                        break
                    else:
                        print(f"   ⚠️  Batalla terminó antes de guardar")
                        battle_just_started = False
            else:
                # No hay batalla
                if battle_just_started:
                    print(f"   ⚠️  Batalla terminó rápidamente")
                battle_just_started = False
                frames_in_battle = 0
            
            last_battle_type = battle_type
            
            # Predecir acción con el modelo
            action, _states = model.predict(obs, deterministic=False)
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Renderizar si no es headless
            if not args.headless:
                env.render()
            
            step += 1
            
            if terminated or truncated:
                print(f"\nEpisodio terminado en step {step}")
                break
    
    except KeyboardInterrupt:
        print("\n\n⚠️ Ejecución detenida por el usuario")
    
    finally:
        # Guardar estado actual al detener
        final_save_path = output_dir / f"manual_save_{args.target_gym}.state"
        save_state_at_position(
            env.pyboy,
            final_save_path,
            f"Estado manual - Step {step}"
        )
        states_saved.append(str(final_save_path))
        
        env.close()
    
    print("\n" + "="*60)
    print("📁 Estados guardados:")
    print("="*60)
    for state_path in states_saved:
        print(f"  • {state_path}")
    print("="*60)
    print(f"\n💡 Usa estos estados para comparar:")
    print(f"python compare_models_interactive.py \\")
    print(f"  --combat-model sessions\\combat_agent_final\\combat_agent_final.zip \\")
    print(f"  --baseline-model {args.model} \\")
    print(f"  --battle-state {states_saved[0] if states_saved else 'generated_battle_states/...'} \\")
    print(f"  --episodes 10 --max-steps 2000")
    print()

if __name__ == '__main__':
    main()
