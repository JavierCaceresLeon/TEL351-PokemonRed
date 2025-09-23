#!/usr/bin/env python3
"""
Professional PyBoy-Compatible Algorithm Executor
==============================================

Enhanced execution system for all algorithms with PyBoy visualization support.
This system ensures all algorithms can be executed in a visualizable environment
with proper timing, metrics collection, and professional implementation standards.

Features:
- PyBoy compatible execution environment
- Real-time visualization support
- Professional timing control
- Comprehensive metrics collection
- Enhanced algorithm implementations
- Robust error handling and logging
"""

import os
import sys
import time
import json
import argparse
import numpy as np
import threading
import queue
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import psutil
import gc

# Add paths for different algorithm modules
sys.path.append(str(Path(__file__).parent / 'comparison_agents'))
sys.path.append(str(Path(__file__).parent / 'comparison_agents' / 'search_algorithms'))
sys.path.append(str(Path(__file__).parent / 'v2'))

@dataclass
class ExecutionConfig:
    """Configuration for algorithm execution"""
    algorithm: str
    variant: str = "default"
    max_steps: int = 3000
    time_limit: float = 300.0  # 5 minutes
    save_frequency: int = 100
    visualization_enabled: bool = True
    real_time_factor: float = 1.0
    metrics_collection: bool = True

@dataclass
class ExecutionResult:
    """Results from algorithm execution"""
    algorithm: str
    variant: str
    success: bool
    execution_time: float
    total_steps: int
    total_reward: float
    final_state: Dict
    metrics_history: List[Dict]
    error_message: Optional[str] = None

class ProfessionalAlgorithmExecutor:
    """
    Professional-grade algorithm executor with PyBoy integration
    """
    
    def __init__(self, results_dir: str = "RESULTADOS"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        # Algorithm configurations
        self.algorithm_configs = self._initialize_algorithm_configs()
        
        # Execution state
        self.current_execution = None
        self.execution_results = []
        
        # Performance monitoring
        self.process = psutil.Process()
        
    def _initialize_algorithm_configs(self) -> Dict[str, ExecutionConfig]:
        """Initialize configurations for all algorithms"""
        return {
            'ppo': ExecutionConfig(
                algorithm='ppo',
                max_steps=1000,  # Reduced for faster testing
                time_limit=120.0,  # 2 minutes
                real_time_factor=8.0
            ),
            'epsilon_greedy_alta_exploracion': ExecutionConfig(
                algorithm='epsilon_greedy',
                variant='alta_exploracion',
                max_steps=800,
                time_limit=100.0,
                real_time_factor=10.0
            ),
            'epsilon_greedy_balanceada': ExecutionConfig(
                algorithm='epsilon_greedy',
                variant='balanceada',
                max_steps=800,
                time_limit=100.0,
                real_time_factor=10.0
            ),
            'epsilon_greedy_conservadora': ExecutionConfig(
                algorithm='epsilon_greedy',
                variant='conservadora',
                max_steps=800,
                time_limit=100.0,
                real_time_factor=10.0
            ),
            'astar': ExecutionConfig(
                algorithm='astar',
                max_steps=600,
                time_limit=80.0,
                real_time_factor=12.0
            ),
            'bfs': ExecutionConfig(
                algorithm='bfs',
                max_steps=500,
                time_limit=90.0,
                real_time_factor=12.0
            ),
            'tabu_search': ExecutionConfig(
                algorithm='tabu_search',
                max_steps=700,
                time_limit=85.0,
                real_time_factor=10.0
            ),
            'hill_climbing_steepest': ExecutionConfig(
                algorithm='hill_climbing',
                variant='steepest_ascent',
                max_steps=600,
                time_limit=75.0,
                real_time_factor=12.0
            ),
            'hill_climbing_first': ExecutionConfig(
                algorithm='hill_climbing',
                variant='first_improvement',
                max_steps=600,
                time_limit=75.0,
                real_time_factor=12.0
            ),
            'hill_climbing_restart': ExecutionConfig(
                algorithm='hill_climbing',
                variant='random_restart',
                max_steps=600,
                time_limit=75.0,
                real_time_factor=12.0
            ),
            'simulated_annealing': ExecutionConfig(
                algorithm='simulated_annealing',
                max_steps=700,
                time_limit=90.0,
                real_time_factor=10.0
            )
        }
    
    def create_enhanced_environment(self, config: ExecutionConfig):
        """Create enhanced PyBoy environment for algorithm execution"""
        try:
            # Import PyBoy environment
            from red_gym_env_v2 import RedGymEnv
            from stream_agent_wrapper import StreamWrapper
            # Validar ROM disponible
            rom_path = Path('PokemonRed.gb')
            if not rom_path.exists():
                raise FileNotFoundError("No se encontró PokemonRed.gb en el directorio raíz. Copia la ROM legal y verifica el SHA-1 requerido.")
            
            # Enhanced environment configuration
            env_config = {
                'headless': not config.visualization_enabled,
                'save_final_state': True,
                'save_screenshot': config.visualization_enabled,
                'save_video': config.visualization_enabled,
                'print_rewards': True,
                'gb_path': 'PokemonRed.gb',
                'debug': False,
                'init_state': 'init.state',  # Start in protagonist's room
                'action_freq': max(1, int(config.real_time_factor)),
                'max_steps': config.max_steps,
                'save_freq': config.save_frequency,
                'fast_video': True,
                'session_path': Path(f'session_{config.algorithm}_{config.variant}_{int(time.time())}')
            }
            
            # Create base environment
            base_env = RedGymEnv(env_config)
            
            # Wrap with streaming if visualization enabled
            if config.visualization_enabled:
                env = StreamWrapper(
                    base_env,
                    stream_metadata={
                        "user": f"{config.algorithm}-professional",
                        "env_id": 0,
                        "color": self._get_algorithm_color(config.algorithm),
                        "extra": f"Professional {config.algorithm.upper()} Implementation"
                    }
                )
            else:
                env = base_env
            
            return env
            
        except Exception as e:
            print(f"Error creating environment: {e}")
            return None
    
    def _get_algorithm_color(self, algorithm: str) -> str:
        """Get distinctive color for algorithm visualization"""
        colors = {
            'ppo': '#8B5CF6',
            'epsilon_greedy': '#10B981', 
            'astar': '#3B82F6',
            'bfs': '#06B6D4',
            'tabu_search': '#8B5A2B',
            'hill_climbing': '#F59E0B',
            'simulated_annealing': '#EF4444'
        }
        return colors.get(algorithm, '#6B7280')
    
    def create_enhanced_agent(self, config: ExecutionConfig):
        """Create enhanced agent implementation"""
        try:
            if config.algorithm == 'ppo':
                return self._create_ppo_agent(config)
            elif config.algorithm == 'epsilon_greedy':
                return self._create_epsilon_greedy_agent(config)
            elif config.algorithm == 'astar':
                return self._create_astar_agent(config)
            elif config.algorithm == 'bfs':
                return self._create_bfs_agent(config)
            elif config.algorithm == 'tabu_search':
                return self._create_tabu_agent(config)
            elif config.algorithm == 'hill_climbing':
                return self._create_hill_climbing_agent(config)
            elif config.algorithm == 'simulated_annealing':
                return self._create_simulated_annealing_agent(config)
            else:
                raise ValueError(f"Unknown algorithm: {config.algorithm}")
                
        except Exception as e:
            print(f"Error creating agent for {config.algorithm}: {e}")
            return None
    
    def _create_ppo_agent(self, config: ExecutionConfig):
        """Create enhanced PPO agent"""
        try:
            from stable_baselines3 import PPO
        except Exception as e:
            raise ImportError(f"Stable-Baselines3/Torch no disponibles para PPO: {e}")
        
        # Load the best available PPO model
        model_paths = [
            'v2/runs/poke_26214400.zip',
            'v2/runs/poke_*.zip'
        ]
        
        for path_pattern in model_paths:
            matches = list(Path('.').glob(path_pattern))
            if matches:
                model_path = max(matches, key=lambda p: p.stat().st_mtime)
                print(f"Loading PPO model: {model_path}")
                return PPO.load(str(model_path))
        # Si no se encontró un modelo PPO, lanzar excepción al momento de crear el agente
        raise FileNotFoundError("No PPO model found")
    
    def _create_epsilon_greedy_agent(self, config: ExecutionConfig):
        """Create enhanced Epsilon-Greedy agent"""
        from epsilon_greedy_agent import EpsilonGreedyAgent
        
        # Enhanced configurations for different variants
        variant_configs = {
            'alta_exploracion': {
                'epsilon_start': 0.9,
                'epsilon_min': 0.1,
                'epsilon_decay': 0.999,
                'scenario_detection_enabled': True
            },
            'balanceada': {
                'epsilon_start': 0.5,
                'epsilon_min': 0.05,
                'epsilon_decay': 0.9995,
                'scenario_detection_enabled': True
            },
            'conservadora': {
                'epsilon_start': 0.3,
                'epsilon_min': 0.01,
                'epsilon_decay': 0.9998,
                'scenario_detection_enabled': True
            },
            'default': {
                'epsilon_start': 0.4,
                'epsilon_min': 0.05,
                'epsilon_decay': 0.995,
                'scenario_detection_enabled': True
            }
        }
        
        agent_config = variant_configs.get(config.variant, variant_configs['default'])
        return EpsilonGreedyAgent(**agent_config)
    
    def _create_astar_agent(self, config: ExecutionConfig):
        """Create enhanced A* agent"""
        from astar_agent import AStarAgent
        return AStarAgent()
    
    def _create_bfs_agent(self, config: ExecutionConfig):
        """Create enhanced BFS agent"""
        from bfs_agent import BFSAgent
        return BFSAgent()
    
    def _create_tabu_agent(self, config: ExecutionConfig):
        """Create enhanced Tabu Search agent"""
        from tabu_agent import TabuSearchAgent
        return TabuSearchAgent(
            tabu_tenure=7,
            max_tabu_size=50,
            aspiration_threshold=1.5,
            scenario_detection_enabled=True
        )
    
    def _create_hill_climbing_agent(self, config: ExecutionConfig):
        """Create enhanced Hill Climbing agent"""
        from hill_climbing_agent import HillClimbingAgent, HillClimbingVariant
        
        variant_map = {
            'steepest_ascent': HillClimbingVariant.STEEPEST_ASCENT,
            'first_improvement': HillClimbingVariant.FIRST_IMPROVEMENT,
            'random_restart': HillClimbingVariant.RANDOM_RESTART,
            'default': HillClimbingVariant.STEEPEST_ASCENT
        }
        
        variant = variant_map.get(config.variant, HillClimbingVariant.STEEPEST_ASCENT)
        return HillClimbingAgent(variant=variant)
    
    def _create_simulated_annealing_agent(self, config: ExecutionConfig):
        """Create enhanced Simulated Annealing agent"""
        from simulated_annealing_agent import SimulatedAnnealingAgent
        return SimulatedAnnealingAgent()
    
    def execute_algorithm(self, config: ExecutionConfig) -> ExecutionResult:
        """Execute single algorithm with comprehensive monitoring"""
        print(f"\\nExecuting {config.algorithm} ({config.variant})...")
        print(f"Max steps: {config.max_steps}, Time limit: {config.time_limit}s")
        
        start_time = time.time()
        self.current_execution = config
        
        try:
            # Create environment and agent
            env = self.create_enhanced_environment(config)
            if env is None:
                raise RuntimeError("Failed to create environment")
            
            agent = self.create_enhanced_agent(config)
            if agent is None:
                raise RuntimeError("Failed to create agent")
            
            # Initialize metrics collection
            metrics_history = []
            step_count = 0
            total_reward = 0.0
            last_reward = 0.0
            
            # Reset environment with proper handling
            reset_result = env.reset()
            if isinstance(reset_result, tuple):
                observation, info = reset_result
            else:
                observation = reset_result
                info = {}
            
            print(f"Starting execution with visualization: {config.visualization_enabled}")
            
            # Main execution loop
            while step_count < config.max_steps:
                step_start_time = time.time()
                
                # Get action from agent
                if config.algorithm == 'ppo':
                    # For PPO, handle observation properly
                    if isinstance(observation, tuple):
                        obs_for_agent = observation[0] if len(observation) > 0 else observation
                    else:
                        obs_for_agent = observation
                    action, _ = agent.predict(obs_for_agent, deterministic=False)
                else:
                    # For search algorithms, provide observation as dict
                    obs_dict = self._convert_observation_to_dict(observation)
                    if hasattr(agent, 'get_action'):
                        action = agent.get_action(obs_dict)
                    elif hasattr(agent, 'select_action'):
                        # Pass last_reward when supported
                        try:
                            action = agent.select_action(obs_dict, last_reward)
                        except TypeError:
                            action = agent.select_action(obs_dict)
                    else:
                        raise AttributeError(f"Agent {type(agent).__name__} has no get_action/select_action method")
                
                # Execute action in environment
                step_result = env.step(action)
                
                # Handle different return formats
                if len(step_result) == 4:
                    observation, reward, done, info = step_result
                elif len(step_result) == 5:
                    observation, reward, done, truncated, info = step_result
                else:
                    raise ValueError(f"Unexpected step result format: {len(step_result)} values")
                
                step_count += 1
                total_reward += reward
                last_reward = reward
                
                # Collect metrics
                if config.metrics_collection and step_count % 50 == 0:
                    metrics = {
                        'step': step_count,
                        'reward': reward,
                        'total_reward': total_reward,
                        'timestamp': time.time(),
                        'memory_usage': self.process.memory_info().rss / 1024 / 1024,
                        'cpu_percent': self.process.cpu_percent()
                    }
                    metrics_history.append(metrics)
                
                # Check time limit
                if time.time() - start_time > config.time_limit:
                    print(f"Time limit reached: {config.time_limit}s")
                    break
                
                # Check if done
                if done:
                    print(f"Episode completed at step {step_count}")
                    break
                
                # Apply real-time factor for visualization (reduced sleep time)
                if config.visualization_enabled and config.real_time_factor > 1:
                    sleep_time = min(0.016 * (config.real_time_factor - 1), 0.05)  # Cap at 50ms
                    time.sleep(sleep_time)
            
            execution_time = time.time() - start_time
            # Ajuste a tiempo real percibido si se acelera la emulación
            reported_time = execution_time * (config.real_time_factor if config.real_time_factor and config.real_time_factor > 1 else 1.0)
            
            # Get final state
            final_state = {
                'total_steps': step_count,
                'total_reward': total_reward,
                'execution_time_wall': execution_time,
                'execution_time_reported': reported_time,
                'success': True
            }
            
            # Clean up
            env.close()
            gc.collect()
            
            result = ExecutionResult(
                algorithm=config.algorithm,
                variant=config.variant,
                success=True,
                execution_time=reported_time,
                total_steps=step_count,
                total_reward=total_reward,
                final_state=final_state,
                metrics_history=metrics_history
            )
            
            print(f"Execution completed successfully:")
            print(f"  Time: {execution_time:.2f}s")
            print(f"  Steps: {step_count}")
            print(f"  Reward: {total_reward:.2f}")
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            reported_time = execution_time * (config.real_time_factor if config.real_time_factor and config.real_time_factor > 1 else 1.0)
            error_msg = str(e)
            
            print(f"Execution failed: {error_msg}")
            
            return ExecutionResult(
                algorithm=config.algorithm,
                variant=config.variant,
                success=False,
                execution_time=reported_time,
                total_steps=0,
                total_reward=0.0,
                final_state={},
                metrics_history=[],
                error_message=error_msg
            )
    
    def _convert_observation_to_dict(self, observation) -> Dict:
        """Convert environment observation to dictionary format for search algorithms"""
        if isinstance(observation, dict):
            return observation
        
        # Handle tuple observations (observation, info)
        if isinstance(observation, tuple):
            actual_obs = observation[0] if len(observation) > 0 else observation
            return self._convert_observation_to_dict(actual_obs)
        
        # For numerical observations (like info dict)
        if isinstance(observation, (int, float)):
            return {
                'screen': np.zeros((144, 160, 3)),
                'location': {'x': 0, 'y': 0, 'map_id': 1},
                'player': {'health': float(observation) if observation > 0 else 100, 'level': 5},
                'badges': 0,
                'events': 0,
                'seen_coords': set(),
                'action_history': []
            }
        
        # For array observations, create a compatible dictionary
        return {
            'screen': observation if hasattr(observation, 'shape') else np.zeros((144, 160, 3)),
            'location': {'x': 0, 'y': 0, 'map_id': 1},
            'player': {'health': 100, 'level': 5},
            'badges': 0,
            'events': 0,
            'seen_coords': set(),
            'action_history': []
        }
    
    def save_execution_results(self, result: ExecutionResult):
        """Save execution results to files"""
        timestamp = int(time.time())
        
        # Create algorithm-specific directory
        algo_dir = self.results_dir / "enhanced_execution" / f"{result.algorithm}_{result.variant}"
        algo_dir.mkdir(parents=True, exist_ok=True)
        
        # Save summary CSV
        # Usar tiempo reportado (equivalente en tiempo real) si está disponible
        time_seconds = result.execution_time if result.execution_time is not None else 0.0
        summary_data = {
            'Métrica': [
                'Pasos Totales',
                'Tiempo (s)',
                'Recompensa Total',
                'Eficiencia (Rew/Paso)',
                'Velocidad (Pasos/s)',
                'Estado de Ejecución'
            ],
            'Valor': [
                result.total_steps,
                time_seconds,
                result.total_reward,
                result.total_reward / max(result.total_steps, 1),
                result.total_steps / max(time_seconds, 0.1),
                'Exitoso' if result.success else 'Fallido'
            ]
        }
        
        import pandas as pd
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(algo_dir / f"enhanced_summary_{timestamp}.csv", index=False)
        
        # Save detailed results
        detailed_results = {
            'algorithm': result.algorithm,
            'variant': result.variant,
            'success': result.success,
            'execution_time': result.execution_time,
            'total_steps': result.total_steps,
            'total_reward': result.total_reward,
            'final_state': result.final_state,
            'metrics_history': result.metrics_history,
            'error_message': result.error_message,
            'timestamp': timestamp
        }
        
        with open(algo_dir / f"enhanced_detailed_{timestamp}.json", 'w') as f:
            json.dump(detailed_results, f, indent=2, default=str)
        
        # Save Markdown report
        self._save_markdown_report(result, algo_dir, timestamp)
    
    def _save_markdown_report(self, result: ExecutionResult, output_dir: Path, timestamp: int):
        """Save detailed markdown report"""
        report = f"""# Reporte de Ejecución Profesional: {result.algorithm.upper()}

## Configuración
- **Algoritmo:** {result.algorithm}
- **Variante:** {result.variant}
- **Timestamp:** {timestamp}
- **Fecha:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Resultados de Rendimiento

### Métricas Principales
- **Estado de Ejecución:** {'✅ Exitoso' if result.success else '❌ Fallido'}
- **Tiempo de Ejecución (equivalente tiempo real):** {result.execution_time:.2f} segundos
- **Pasos Totales:** {result.total_steps:,}
- **Recompensa Total:** {result.total_reward:.2f}
- **Eficiencia:** {result.total_reward / max(result.total_steps, 1):.4f} recompensa/paso
- **Velocidad:** {result.total_steps / max(result.execution_time, 0.1):.1f} pasos/segundo

### Análisis de Rendimiento
"""

        if result.success:
            report += (
                "\nLa ejecución se completó exitosamente con un rendimiento "
                f"{'excelente' if result.total_reward > 1000 else 'bueno' if result.total_reward > 500 else 'básico'}.\n\n"
                "**Indicadores de Calidad:**\n"
                f"- Eficiencia de recompensa: "
                f"{'Alta' if result.total_reward / max(result.total_steps, 1) > 0.1 else 'Media' if result.total_reward / max(result.total_steps, 1) > 0.05 else 'Baja'}\n"
                f"- Velocidad de ejecución: "
                f"{'Rápida' if result.total_steps / max(result.execution_time, 0.1) > 50 else 'Media' if result.total_steps / max(result.execution_time, 0.1) > 20 else 'Lenta'}\n"
                f"- Estabilidad: {'Estable' if len(result.metrics_history) > 10 else 'Limitada'}\n"
            )
        else:
            report += (
                f"\nLa ejecución falló con el siguiente error: {result.error_message}\n\n"
                "**Análisis del Fallo:**\n"
                f"- Tiempo transcurrido antes del fallo: {result.execution_time:.2f} segundos\n"
                "- Posibles causas: Error de implementación, falta de dependencias, o configuración incorrecta\n"
            )

        report += (
            "\n\n### Historial de Métricas\n"
            f"Se recolectaron {len(result.metrics_history)} puntos de datos durante la ejecución.\n\n"
            "### Estado Final\n"
            f"{json.dumps(result.final_state, indent=2)}\n\n"
            "---\n"
            "*Reporte generado automáticamente por el Sistema de Ejecución Profesional*\n"
        )

        with open(output_dir / f"enhanced_report_{timestamp}.md", 'w', encoding='utf-8') as f:
            f.write(report)
    
    def run_comprehensive_test_suite(self, algorithms: Optional[List[str]] = None):
        """Run comprehensive test suite for all algorithms"""
        if algorithms is None:
            algorithms = list(self.algorithm_configs.keys())
        
        print("Starting Comprehensive Professional Algorithm Test Suite")
        print("=" * 70)
        print(f"Algorithms to test: {len(algorithms)}")
        print(f"Estimated total time: {sum(self.algorithm_configs[alg].time_limit for alg in algorithms) / 60:.1f} minutes")
        print("=" * 70)
        
        successful_executions = 0
        failed_executions = 0
        
        for algo_name in algorithms:
            if algo_name not in self.algorithm_configs:
                print(f"Warning: Unknown algorithm {algo_name}, skipping...")
                continue
            
            config = self.algorithm_configs[algo_name]
            
            try:
                result = self.execute_algorithm(config)
                self.execution_results.append(result)
                self.save_execution_results(result)
                
                if result.success:
                    successful_executions += 1
                    print(f"✅ {algo_name}: Success")
                else:
                    failed_executions += 1
                    print(f"❌ {algo_name}: Failed - {result.error_message}")
                
            except Exception as e:
                failed_executions += 1
                print(f"💥 {algo_name}: Critical Error - {e}")
                continue
            
            # Brief pause between executions
            time.sleep(2)
        
        print("\\n" + "=" * 70)
        print("COMPREHENSIVE TEST SUITE COMPLETED")
        print(f"✅ Successful: {successful_executions}")
        print(f"❌ Failed: {failed_executions}")
        print(f"📊 Success Rate: {successful_executions / (successful_executions + failed_executions) * 100:.1f}%")
        print("=" * 70)
        
        return self.execution_results

def main():
    """CLI profesional para ejecutar algoritmos de forma flexible"""
    parser = argparse.ArgumentParser(description="Professional Algorithm Executor (PyBoy-compatible)")
    parser.add_argument("--algos", nargs="*", help="Claves de algoritmos a ejecutar (por defecto: todos)")
    parser.add_argument("--headless", action="store_true", help="Ejecutar sin visualización PyBoy")
    parser.add_argument("--max-steps", type=int, help="Sobrescribir max_steps para todas las ejecuciones")
    parser.add_argument("--time-limit", type=float, help="Sobrescribir time_limit (s) para todas las ejecuciones")
    parser.add_argument("--no-metrics", action="store_true", help="Desactivar colección de métricas")
    parser.add_argument("--real-time-factor", type=float, help="Sobrescribir real_time_factor para todas las ejecuciones")
    parser.add_argument("--skip-ppo", action="store_true", help="Omitir PPO si faltan dependencias/modelos")

    args = parser.parse_args()

    executor = ProfessionalAlgorithmExecutor()

    # Construir lista de algoritmos
    algorithms = args.algos if args.algos else list(executor.algorithm_configs.keys())
    if args.skip_ppo:
        algorithms = [a for a in algorithms if a != 'ppo']

    # Aplicar overrides globales
    for key in algorithms:
        if key not in executor.algorithm_configs:
            print(f"Aviso: algoritmo desconocido '{key}', se omite.")
            continue
        cfg = executor.algorithm_configs[key]
        if args.headless:
            cfg.visualization_enabled = False
        if args.max_steps is not None:
            cfg.max_steps = args.max_steps
        if args.time_limit is not None:
            cfg.time_limit = args.time_limit
        if args.no_metrics:
            cfg.metrics_collection = False
        if args.real_time_factor is not None:
            cfg.real_time_factor = args.real_time_factor

    # Ejecutar suite
    try:
        results = executor.run_comprehensive_test_suite(algorithms)
    except Exception as e:
        print(f"Error al ejecutar la suite: {e}")
        results = []

    print(f"\nExecution completed. {len(results)} algorithm runs processed.")
    print("Results saved to RESULTADOS/enhanced_execution/")

if __name__ == "__main__":
    main()