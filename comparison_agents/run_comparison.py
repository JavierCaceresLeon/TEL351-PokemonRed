"""
Script principal para comparar diferentes agentes en Pokémon Red
Compara: Agente entrenado (v2), A*, Tabú Search
Objetivo: Salir de la habitación inicial
"""

import sys
import os
import time
import json
from pathlib import Path
from datetime import datetime

# Agregar directorios al path
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))
sys.path.append(str(current_dir / "search_algorithms"))

# Importar agentes
try:
    from search_env import SearchEnvironment
    from search_algorithms.astar_agent import AStarAgent
    from search_algorithms.tabu_agent import TabuSearchAgent
    from v2_agent import V2TrainedAgent
except ImportError as e:
    print(f"Error importando módulos: {e}")
    print("Asegúrate de que todas las dependencias estén instaladas")

class AgentComparison:
    """Clase para comparar diferentes agentes"""
    
    def __init__(self, config=None):
        self.config = config or {
            'init_state': '../init.state',
            'gb_path': '../PokemonRed.gb',
            'headless': True,
            'max_steps': 1000,
            'session_path': Path('comparison_session'),
            'num_runs': 5  # Número de ejecuciones por agente
        }
        
        self.results = {
            'v2_agent': [],
            'astar_agent': [],
            'tabu_agent': []
        }
        
        self.results_dir = Path(__file__).parent / "results"
        self.results_dir.mkdir(exist_ok=True)
    
    def test_search_environment(self):
        """Probar que el entorno de búsqueda funciona"""
        print("Probando entorno de búsqueda...")
        try:
            env = SearchEnvironment(self.config)
            state = env.reset()
            print(f"Estado inicial: {state['position']}")
            
            # Probar algunas acciones
            for i in range(5):
                actions = env.get_valid_actions(state)
                action = actions[0] if actions else 0
                state, reward, done = env.step(action)
                print(f"Paso {i+1}: Posición {state['position']}, Recompensa: {reward:.2f}")
                
                if done:
                    print("Episodio terminado")
                    break
            
            env.close()
            print("✓ Entorno de búsqueda funciona correctamente")
            return True
            
        except Exception as e:
            print(f"✗ Error en entorno de búsqueda: {e}")
            return False
    
    def run_astar_agent(self, run_id):
        """Ejecutar agente A*"""
        print(f"Ejecutando A* - Run {run_id + 1}")
        
        try:
            env = SearchEnvironment(self.config)
            agent = AStarAgent(env, max_search_depth=500)
            
            start_time = time.time()
            
            # Búsqueda
            plan = agent.search()
            search_time = time.time() - start_time
            
            # Ejecutar plan
            execution_results = agent.execute_plan(plan)
            
            # Recopilar resultados
            stats = agent.get_stats()
            
            result = {
                'run_id': run_id,
                'agent_type': 'astar',
                'search_time': search_time,
                'execution_time': execution_results['execution_time'],
                'total_time': search_time + execution_results['execution_time'],
                'plan_length': len(plan),
                'success': execution_results['success'],
                'nodes_explored': stats['nodes_explored'],
                'timestamp': datetime.now().isoformat()
            }
            
            env.close()
            return result
            
        except Exception as e:
            print(f"Error en A*: {e}")
            return {
                'run_id': run_id,
                'agent_type': 'astar',
                'error': str(e),
                'success': False,
                'timestamp': datetime.now().isoformat()
            }
    
    def run_tabu_agent(self, run_id):
        """Ejecutar agente Tabú Search"""
        print(f"Ejecutando Tabú Search - Run {run_id + 1}")
        
        try:
            env = SearchEnvironment(self.config)
            agent = TabuSearchAgent(env, max_iterations=200, tabu_size=30)
            
            start_time = time.time()
            
            # Búsqueda
            plan = agent.search()
            search_time = time.time() - start_time
            
            # Ejecutar plan
            execution_results = agent.execute_plan(plan)
            
            # Recopilar resultados
            stats = agent.get_stats()
            
            result = {
                'run_id': run_id,
                'agent_type': 'tabu',
                'search_time': search_time,
                'execution_time': execution_results['execution_time'],
                'total_time': search_time + execution_results['execution_time'],
                'plan_length': len(plan),
                'success': execution_results['success'],
                'iterations': stats['iterations'],
                'best_fitness': stats['best_fitness'],
                'timestamp': datetime.now().isoformat()
            }
            
            env.close()
            return result
            
        except Exception as e:
            print(f"Error en Tabú Search: {e}")
            return {
                'run_id': run_id,
                'agent_type': 'tabu',
                'error': str(e),
                'success': False,
                'timestamp': datetime.now().isoformat()
            }
    
    def run_v2_agent(self, run_id):
        """Ejecutar agente entrenado v2"""
        print(f"Ejecutando Agente V2 - Run {run_id + 1}")
        
        try:
            # Configuración específica para v2
            v2_config = {
                'headless': True,
                'save_final_state': False,
                'early_stop': False,
                'action_freq': 24,
                'init_state': self.config['init_state'],
                'max_steps': self.config['max_steps'],
                'print_rewards': False,
                'save_video': False,
                'fast_video': True,
                'session_path': self.config['session_path'] / f"v2_run_{run_id}",
                'gb_path': self.config['gb_path'],
                'debug': False,
                'reward_scale': 0.5,
                'explore_weight': 0.25
            }
            
            agent = V2TrainedAgent(v2_config)
            
            if not agent.search_stats['model_loaded']:
                return {
                    'run_id': run_id,
                    'agent_type': 'v2',
                    'error': 'No se pudo cargar el modelo',
                    'success': False,
                    'timestamp': datetime.now().isoformat()
                }
            
            start_time = time.time()
            
            # Ejecutar agente
            plan = agent.search()
            total_time = time.time() - start_time
            
            # Recopilar resultados
            stats = agent.get_stats()
            
            result = {
                'run_id': run_id,
                'agent_type': 'v2',
                'search_time': 0,  # El modelo predice directamente
                'execution_time': stats['execution_time'],
                'total_time': total_time,
                'plan_length': len(plan),
                'steps_taken': stats['steps_taken'],
                'success': stats['success'],
                'timestamp': datetime.now().isoformat()
            }
            
            agent.close()
            return result
            
        except Exception as e:
            print(f"Error en agente V2: {e}")
            return {
                'run_id': run_id,
                'agent_type': 'v2',
                'error': str(e),
                'success': False,
                'timestamp': datetime.now().isoformat()
            }
    
    def run_comparison(self):
        """Ejecutar comparación completa"""
        print("=== Iniciando Comparación de Agentes ===")
        print(f"Configuración: {self.config}")
        
        # Probar entorno primero
        if not self.test_search_environment():
            print("Abortando comparación debido a problemas con el entorno")
            return
        
        num_runs = self.config['num_runs']
        
        # Ejecutar cada agente múltiples veces
        print(f"\nEjecutando {num_runs} pruebas por agente...")
        
        # A*
        print("\n--- Ejecutando A* ---")
        for i in range(num_runs):
            result = self.run_astar_agent(i)
            self.results['astar_agent'].append(result)
            print(f"A* Run {i+1}: {'✓' if result.get('success', False) else '✗'}")
        
        # Tabú Search
        print("\n--- Ejecutando Tabú Search ---")
        for i in range(num_runs):
            result = self.run_tabu_agent(i)
            self.results['tabu_agent'].append(result)
            print(f"Tabú Run {i+1}: {'✓' if result.get('success', False) else '✗'}")
        
        # V2 Agent
        print("\n--- Ejecutando Agente V2 ---")
        for i in range(num_runs):
            result = self.run_v2_agent(i)
            self.results['v2_agent'].append(result)
            print(f"V2 Run {i+1}: {'✓' if result.get('success', False) else '✗'}")
        
        # Guardar y analizar resultados
        self.save_results()
        self.analyze_results()
    
    def save_results(self):
        """Guardar resultados en archivo JSON"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.results_dir / f"comparison_results_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\nResultados guardados en: {filename}")
    
    def analyze_results(self):
        """Analizar y mostrar estadísticas de los resultados"""
        print("\n=== ANÁLISIS DE RESULTADOS ===")
        
        for agent_type, results in self.results.items():
            if not results:
                continue
                
            print(f"\n--- {agent_type.upper()} ---")
            
            # Filtrar resultados exitosos
            successful_runs = [r for r in results if r.get('success', False)]
            failed_runs = [r for r in results if not r.get('success', False)]
            
            success_rate = len(successful_runs) / len(results) * 100
            print(f"Tasa de éxito: {success_rate:.1f}% ({len(successful_runs)}/{len(results)})")
            
            if successful_runs:
                times = [r.get('total_time', 0) for r in successful_runs]
                steps = [r.get('plan_length', 0) for r in successful_runs]
                
                print(f"Tiempo promedio: {sum(times)/len(times):.2f}s")
                print(f"Tiempo mínimo: {min(times):.2f}s")
                print(f"Tiempo máximo: {max(times):.2f}s")
                print(f"Pasos promedio: {sum(steps)/len(steps):.1f}")
                print(f"Pasos mínimo: {min(steps)}")
                print(f"Pasos máximo: {max(steps)}")
            
            if failed_runs:
                print(f"Fallos: {len(failed_runs)}")
                errors = [r.get('error', 'Unknown') for r in failed_runs if 'error' in r]
                if errors:
                    print(f"Errores comunes: {set(errors)}")

def main():
    """Función principal"""
    print("Comparación de Agentes - Pokémon Red")
    print("Objetivo: Salir de la habitación inicial")
    
    # Configuración
    config = {
        'init_state': '../init.state',
        'gb_path': '../PokemonRed.gb',
        'headless': True,
        'max_steps': 500,  # Reducido para pruebas más rápidas
        'session_path': Path('comparison_session'),
        'num_runs': 3  # Reducido para pruebas iniciales
    }
    
    # Ejecutar comparación
    comparison = AgentComparison(config)
    comparison.run_comparison()

if __name__ == "__main__":
    main()
