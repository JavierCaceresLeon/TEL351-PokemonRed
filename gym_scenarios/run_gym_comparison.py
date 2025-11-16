"""
Script de Comparación: PPO Base vs PPO Reentrenado en Gimnasios
================================================================

Este script ejecuta y compara dos agentes PPO (base y reentrenado) en cada uno
de los 8 escenarios de gimnasios, midiendo su desempeño en puzzles y combates.

Uso:
    python run_gym_comparison.py --gym 1
    python run_gym_comparison.py --all
    python run_gym_comparison.py --gym 3 --model-base path/to/base.zip --model-retrained path/to/retrained.zip
"""

import argparse
import sys
import time
import json
from pathlib import Path
import numpy as np

# Agregar paths necesarios
sys.path.append(str(Path(__file__).parent.parent / "v2"))
sys.path.append(str(Path(__file__).parent.parent / "baselines"))
sys.path.append(str(Path(__file__).parent))

from gym_metrics import GymMetricsTracker, compare_agents
from stable_baselines3 import PPO
from red_gym_env_v2 import RedGymEnv


class GymScenarioRunner:
    """Ejecuta agentes en escenarios de gimnasios con métricas"""
    
    def __init__(self, gym_number, state_file, config_file, max_steps=10000):
        """
        Inicializa el runner
        
        Args:
            gym_number: Número del gimnasio (1-8)
            state_file: Path al archivo .state del gimnasio
            config_file: Path al archivo team_config.json
            max_steps: Máximo de pasos permitidos
        """
        self.gym_number = gym_number
        self.state_file = Path(state_file)
        self.config_file = Path(config_file)
        self.max_steps = max_steps
        
        # Cargar configuración del gimnasio
        with open(config_file, 'r', encoding='utf-8') as f:
            self.gym_config = json.load(f)
        
        self.gym_name = self.gym_config.get('gym_name', f'Gym {gym_number}')
        
        print(f"\n{'='*70}")
        print(f"🏟️  ESCENARIO DE GIMNASIO {gym_number}: {self.gym_name}")
        print(f"{'='*70}")
    
    def create_env(self, headless=True):
        """Crea el ambiente de gimnasio"""
        rom_path = Path(__file__).parent.parent / "PokemonRed.gb"
        session_path = Path(__file__).parent / f"gym{self.gym_number}_session"
        session_path.mkdir(exist_ok=True)
        
        env_config = {
            'headless': headless,
            'save_final_state': True,
            'print_rewards': True,
            'gb_path': str(rom_path),
            'debug': False,
            'sim_frame_dist': 2_000_000.0,
            'use_screen_explore': True,
            'extra_buttons': False,
            'explore_weight': 1,
            'reward_scale': 1,
            'action_freq': 24,
            'init_state': str(self.state_file),
            'max_steps': self.max_steps,
            'early_stop': False,
            'save_video': False,
            'fast_video': True,
            'session_path': session_path,
            'instance_id': f'gym{self.gym_number}'
        }
        
        return RedGymEnv(env_config)
    
    def run_agent(self, model_path, agent_name, headless=True, deterministic=False):
        """
        Ejecuta un agente PPO en el gimnasio
        
        Args:
            model_path: Path al modelo PPO (.zip)
            agent_name: Nombre del agente ("PPO_Base", "PPO_Retrained")
            headless: Si ejecutar sin interfaz gráfica
            deterministic: Si usar política determinística
        
        Returns:
            Diccionario con métricas del agente
        """
        print(f"\n{'─'*70}")
        print(f"🤖 Ejecutando agente: {agent_name}")
        print(f"   Modelo: {model_path}")
        print(f"{'─'*70}\n")
        
        # Cargar modelo
        if not Path(model_path).exists():
            print(f"❌ Error: Modelo no encontrado: {model_path}")
            return None
        
        model = PPO.load(model_path)
        print(f"✓ Modelo cargado exitosamente")
        
        # Crear ambiente
        env = self.create_env(headless=headless)
        obs, info = env.reset()
        print(f"✓ Ambiente inicializado")
        
        # Iniciar tracker de métricas
        tracker = GymMetricsTracker(
            gym_number=self.gym_number,
            agent_name=agent_name,
            gym_name=self.gym_name
        )
        tracker.start()
        
        # Ejecutar episodio
        done = False
        truncated = False
        step = 0
        last_hp = [100, 100, 100, 100, 100, 100]  # Asumimos HP inicial completo
        
        print(f"\n▶️  Iniciando episodio (máx {self.max_steps} pasos)...\n")
        
        while not (done or truncated) and step < self.max_steps:
            # Predecir acción
            action, _states = model.predict(obs, deterministic=deterministic)
            
            # Ejecutar acción
            obs, reward, done, truncated, info = env.step(action)
            
            # Extraer estado del juego desde el ambiente
            game_state = {
                'x': env.read_m(0xD362),
                'y': env.read_m(0xD361),
                'map': env.read_m(0xD35E),
                'hp': [env.read_hp_fraction()],  # HP total del equipo
                'in_battle': False  # TODO: detectar batalla
            }
            
            # Registrar paso
            tracker.record_step(action, reward, game_state)
            
            # Detectar eventos especiales
            current_hp = env.read_hp_fraction()
            
            # TODO: Mejorar detección de eventos
            # - Batalla iniciada/terminada
            # - Puzzle resuelto
            # - Items usados
            
            step += 1
            
            # Mostrar progreso cada 500 pasos
            if step % 500 == 0:
                print(f"  📍 Paso {step}/{self.max_steps} - Recompensa acumulada: {tracker.total_reward:.2f}")
        
        # Determinar éxito
        # TODO: Implementar detección de victoria en gimnasio
        success = tracker.total_reward > 100  # Placeholder
        
        # Finalizar tracking
        tracker.record_team_state([env.read_hp_fraction()], is_final=True)
        tracker.end(success=success)
        
        # Guardar métricas
        output_dir = Path(__file__).parent / f"gym{self.gym_number}_{self.gym_name.lower().replace(' ', '_').replace('-', '')}" / "results"
        tracker.save_metrics(output_dir=output_dir)
        
        # Cerrar ambiente
        env.close()
        
        print(f"\n✓ Ejecución completada para {agent_name}")
        
        return {
            'metadata': tracker.metadata,
            'summary': tracker.get_summary_stats()
        }
    
    def compare_models(self, model_base_path, model_retrained_path, headless=True):
        """
        Compara dos modelos PPO en el gimnasio
        
        Args:
            model_base_path: Path al modelo base
            model_retrained_path: Path al modelo reentrenado
            headless: Si ejecutar sin interfaz gráfica
        
        Returns:
            Diccionario con comparación de métricas
        """
        print(f"\n{'*'*70}")
        print(f"🔬 COMPARACIÓN DE AGENTES - GIMNASIO {self.gym_number}")
        print(f"{'*'*70}")
        
        results = []
        
        # Ejecutar agente base
        print("\n[1/2] Ejecutando PPO Base...")
        result_base = self.run_agent(
            model_path=model_base_path,
            agent_name="PPO_Base",
            headless=headless
        )
        if result_base:
            results.append(result_base)
        
        # Pequeña pausa entre ejecuciones
        time.sleep(2)
        
        # Ejecutar agente reentrenado
        print("\n[2/2] Ejecutando PPO Reentrenado...")
        result_retrained = self.run_agent(
            model_path=model_retrained_path,
            agent_name="PPO_Retrained",
            headless=headless
        )
        if result_retrained:
            results.append(result_retrained)
        
        # Comparar resultados
        if len(results) == 2:
            comparison = self._compare_results(result_base, result_retrained)
            
            # Guardar comparación
            output_dir = Path(__file__).parent / f"gym{self.gym_number}_{self.gym_name.lower().replace(' ', '_').replace('-', '')}" / "results"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            comparison_file = output_dir / f"comparison_{int(time.time())}.json"
            with open(comparison_file, 'w', encoding='utf-8') as f:
                json.dump(comparison, f, indent=2, ensure_ascii=False)
            
            print(f"\n✓ Comparación guardada: {comparison_file}")
            
            # Mostrar resumen
            self._print_comparison_summary(comparison)
            
            return comparison
        else:
            print("\n❌ No se pudieron comparar ambos agentes")
            return None
    
    def _compare_results(self, result_base, result_retrained):
        """Compara dos resultados y calcula diferencias"""
        base_summary = result_base['summary']
        retrained_summary = result_retrained['summary']
        
        comparison = {
            'gym_number': self.gym_number,
            'gym_name': self.gym_name,
            'base': result_base,
            'retrained': result_retrained,
            'differences': {}
        }
        
        # Métricas a comparar
        metrics_to_compare = [
            'total_duration_seconds',
            'total_steps',
            'total_reward',
            'avg_reward_per_step',
            'success',
            'battle_won',
            'puzzle_solved',
            'unique_tiles_explored',
            'potions_used'
        ]
        
        for metric in metrics_to_compare:
            if metric in base_summary and metric in retrained_summary:
                base_val = base_summary[metric]
                retrained_val = retrained_summary[metric]
                
                diff = {
                    'base': base_val,
                    'retrained': retrained_val,
                }
                
                # Calcular diferencia/mejora
                if isinstance(base_val, (int, float)) and isinstance(retrained_val, (int, float)):
                    diff['absolute_diff'] = retrained_val - base_val
                    if base_val != 0:
                        diff['percent_change'] = ((retrained_val - base_val) / base_val) * 100
                    else:
                        diff['percent_change'] = 0
                    
                    # Determinar si es mejora
                    # Para tiempo y pasos, menos es mejor
                    if metric in ['total_duration_seconds', 'total_steps', 'potions_used']:
                        diff['improved'] = retrained_val < base_val
                    # Para rewards y exploración, más es mejor
                    else:
                        diff['improved'] = retrained_val > base_val
                
                comparison['differences'][metric] = diff
        
        return comparison
    
    def _print_comparison_summary(self, comparison):
        """Imprime resumen de comparación en consola"""
        print(f"\n{'='*70}")
        print(f"📊 RESUMEN DE COMPARACIÓN - GIMNASIO {self.gym_number}")
        print(f"{'='*70}\n")
        
        diffs = comparison['differences']
        
        for metric, data in diffs.items():
            base_val = data['base']
            retrained_val = data['retrained']
            
            # Formatear valores
            if isinstance(base_val, bool):
                base_str = "✓" if base_val else "✗"
                retrained_str = "✓" if retrained_val else "✗"
                diff_str = ""
            elif isinstance(base_val, float):
                base_str = f"{base_val:.2f}"
                retrained_str = f"{retrained_val:.2f}"
                if 'percent_change' in data:
                    pct = data['percent_change']
                    arrow = "📈" if data.get('improved', False) else "📉"
                    diff_str = f" {arrow} {pct:+.1f}%"
                else:
                    diff_str = ""
            else:
                base_str = str(base_val)
                retrained_str = str(retrained_val)
                if 'percent_change' in data:
                    pct = data['percent_change']
                    arrow = "📈" if data.get('improved', False) else "📉"
                    diff_str = f" {arrow} {pct:+.1f}%"
                else:
                    diff_str = ""
            
            print(f"  {metric:.<35} Base: {base_str:>12} | Retrained: {retrained_str:>12}{diff_str}")
        
        print(f"\n{'='*70}\n")


def main():
    """Función principal"""
    parser = argparse.ArgumentParser(
        description="Compara agentes PPO en escenarios de gimnasios Pokémon"
    )
    
    parser.add_argument(
        '--gym',
        type=int,
        choices=range(1, 9),
        help='Número del gimnasio a evaluar (1-8)'
    )
    
    parser.add_argument(
        '--all',
        action='store_true',
        help='Evaluar todos los 8 gimnasios'
    )
    
    parser.add_argument(
        '--model-base',
        type=str,
        default='../v2/ppo_session_bf67d815/model_99000000.zip',
        help='Path al modelo PPO base'
    )
    
    parser.add_argument(
        '--model-retrained',
        type=str,
        default='../v2/ppo_session_fb8123f4/model_retrained.zip',
        help='Path al modelo PPO reentrenado'
    )
    
    parser.add_argument(
        '--headless',
        action='store_true',
        default=True,
        help='Ejecutar sin interfaz gráfica'
    )
    
    parser.add_argument(
        '--max-steps',
        type=int,
        default=10000,
        help='Máximo de pasos por episodio'
    )
    
    args = parser.parse_args()
    
    # Determinar qué gimnasios evaluar
    if args.all:
        gyms_to_evaluate = range(1, 9)
    elif args.gym:
        gyms_to_evaluate = [args.gym]
    else:
        print("❌ Error: Especifica --gym N o --all")
        return
    
    # Información de gimnasios
    gym_folders = {
        1: "gym1_pewter_brock",
        2: "gym2_cerulean_misty",
        3: "gym3_vermilion_lt_surge",
        4: "gym4_celadon_erika",
        5: "gym5_fuchsia_koga",
        6: "gym6_saffron_sabrina",
        7: "gym7_cinnabar_blaine",
        8: "gym8_viridian_giovanni"
    }
    
    base_path = Path(__file__).parent
    
    # Evaluar cada gimnasio
    all_comparisons = []
    
    for gym_num in gyms_to_evaluate:
        gym_folder = gym_folders[gym_num]
        gym_path = base_path / gym_folder
        
        state_file = gym_path / "gym_scenario.state"
        config_file = gym_path / "team_config.json"
        
        # Verificar archivos
        if not state_file.exists():
            print(f"\n⚠️  Advertencia: Estado no encontrado para gimnasio {gym_num}")
            print(f"   Ejecuta primero: python generate_gym_states.py")
            continue
        
        if not config_file.exists():
            print(f"\n⚠️  Advertencia: Config no encontrado para gimnasio {gym_num}")
            continue
        
        # Crear runner y ejecutar comparación
        runner = GymScenarioRunner(
            gym_number=gym_num,
            state_file=state_file,
            config_file=config_file,
            max_steps=args.max_steps
        )
        
        comparison = runner.compare_models(
            model_base_path=args.model_base,
            model_retrained_path=args.model_retrained,
            headless=args.headless
        )
        
        if comparison:
            all_comparisons.append(comparison)
    
    # Resumen final
    if all_comparisons:
        print(f"\n{'*'*70}")
        print(f"✓ EVALUACIÓN COMPLETADA")
        print(f"  Gimnasios evaluados: {len(all_comparisons)}")
        print(f"{'*'*70}\n")
    else:
        print("\n❌ No se completaron evaluaciones")


if __name__ == "__main__":
    main()
