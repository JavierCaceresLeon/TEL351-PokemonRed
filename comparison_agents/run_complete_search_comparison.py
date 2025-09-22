#!/usr/bin/env python3
"""
Script maestro para ejecutar comparaciones de diferentes algoritmos de búsqueda
en Pokemon Red a máxima velocidad.
"""

import os
import sys
import subprocess
import time
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

def run_search_comparison_suite():
    """Ejecutar suite completa de comparaciones de algoritmos de búsqueda"""
    
    # Algoritmos de búsqueda a probar
    search_algorithms = [
        {
            'name': 'A* Search',
            'algorithm': 'astar',
            'variant': 'default',
            'description': 'Búsqueda A* con heurística de distancia Manhattan'
        },
        {
            'name': 'Breadth-First Search',
            'algorithm': 'bfs',
            'variant': 'default',
            'description': 'Búsqueda por amplitud sistemática'
        },
        {
            'name': 'Simulated Annealing',
            'algorithm': 'simulated_annealing',
            'variant': 'default',
            'description': 'Recocido simulado con enfriamiento adaptativo'
        },
        {
            'name': 'Hill Climbing (Steepest Ascent)',
            'algorithm': 'hill_climbing',
            'variant': 'steepest_ascent',
            'description': 'Hill Climbing con selección de mejor mejora'
        },
        {
            'name': 'Hill Climbing (First Improvement)',
            'algorithm': 'hill_climbing',
            'variant': 'first_improvement',
            'description': 'Hill Climbing con primera mejora encontrada'
        },
        {
            'name': 'Hill Climbing (Random Restart)',
            'algorithm': 'hill_climbing',
            'variant': 'random_restart',
            'description': 'Hill Climbing con reinicio aleatorio'
        },
        {
            'name': 'Tabu Search',
            'algorithm': 'tabu_search',
            'variant': 'default',
            'description': 'Búsqueda Tabú con lista de movimientos prohibidos'
        }
    ]
    
    # Configuración de ejecución
    runs_per_algorithm = 11
    base_results_dir = Path("RESULTADOS/search_algorithms_comparison")
    
    print("=" * 80)
    print("🔍 COMPARACIÓN DE ALGORITMOS DE BÚSQUEDA - POKEMON RED")
    print("=" * 80)
    print(f"📊 Configuraciones a probar: {len(search_algorithms)}")
    print(f"🔄 Ejecuciones por configuración: {runs_per_algorithm}")
    print(f"📈 Total de ejecuciones: {len(search_algorithms) * runs_per_algorithm}")
    print(f"💾 Directorio de resultados: {base_results_dir}")
    print("=" * 80)
    
    # Verificar que el script runner existe
    runner_script = Path("comparison_agents/simple_search_runner.py")
    if not runner_script.exists():
        print(f"❌ Error: No se encuentra el script {runner_script}")
        return False
    
    # Ejecutar cada configuración
    total_executions = 0
    successful_executions = 0
    failed_executions = 0
    
    for config in search_algorithms:
        algorithm_name = config['algorithm']
        variant_name = config['variant']
        config_name = f"{algorithm_name}_{variant_name}".replace(' ', '_').lower()
        
        print(f"\n🚀 Ejecutando: {config['name']}")
        print(f"📝 Descripción: {config['description']}")
        print(f"⚙️  Configuración: {config_name}")
        
        # Crear directorio para esta configuración
        config_results_dir = base_results_dir / config_name
        config_results_dir.mkdir(parents=True, exist_ok=True)
        
        # Ejecutar múltiples runs de esta configuración
        config_successful = 0
        config_failed = 0
        config_start_time = time.time()
        
        for run_num in range(1, runs_per_algorithm + 1):
            total_executions += 1
            run_dir = config_results_dir / str(run_num)
            run_dir.mkdir(exist_ok=True)
            
            print(f"  🔄 Ejecución {run_num}/{runs_per_algorithm}... ", end="", flush=True)
            
            try:
                # Ejecutar el script de simulación
                cmd = [
                    sys.executable,
                    str(runner_script),
                    algorithm_name,
                    variant_name,
                    str(run_dir)
                ]
                
                # Ejecutar con timeout para evitar colgamientos
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=300,  # 5 minutos timeout máximo por ejecución
                    cwd=Path.cwd()
                )
                
                if result.returncode == 0:
                    print("✅ Éxito")
                    successful_executions += 1
                    config_successful += 1
                else:
                    print(f"❌ Error (código {result.returncode})")
                    if result.stderr:
                        print(f"    Error: {result.stderr[:100]}...")
                    failed_executions += 1
                    config_failed += 1
                
            except subprocess.TimeoutExpired:
                print("⏰ Timeout")
                failed_executions += 1
                config_failed += 1
            except Exception as e:
                print(f"❌ Excepción: {str(e)[:50]}...")
                failed_executions += 1
                config_failed += 1
        
        config_elapsed = time.time() - config_start_time
        print(f"  📊 Configuración completada: {config_successful}/{runs_per_algorithm} éxitos")
        print(f"  ⏱️  Tiempo de configuración: {config_elapsed:.1f}s")
        
        # Crear resumen de la configuración
        create_config_summary(config_results_dir, config, config_successful, config_failed, config_elapsed)
    
    print("\n" + "=" * 80)
    print("📈 RESUMEN FINAL DE EJECUCIONES")
    print("=" * 80)
    print(f"✅ Ejecuciones exitosas: {successful_executions}/{total_executions}")
    print(f"❌ Ejecuciones fallidas: {failed_executions}/{total_executions}")
    print(f"📊 Tasa de éxito: {(successful_executions/total_executions)*100:.1f}%")
    print(f"💾 Resultados guardados en: {base_results_dir}")
    
    if successful_executions > 0:
        print("\n🎯 ¡Comparación completada! Ejecutar análisis de resultados...")
        return True
    else:
        print("\n❌ No se completaron ejecuciones exitosas.")
        return False

def create_config_summary(config_dir, config_info, successful_runs, failed_runs, elapsed_time):
    """Crear resumen de una configuración específica"""
    
    summary = {
        'algorithm_info': config_info,
        'execution_summary': {
            'total_runs': successful_runs + failed_runs,
            'successful_runs': successful_runs,
            'failed_runs': failed_runs,
            'success_rate': successful_runs / (successful_runs + failed_runs) if (successful_runs + failed_runs) > 0 else 0,
            'total_time_seconds': elapsed_time,
            'average_time_per_run': elapsed_time / (successful_runs + failed_runs) if (successful_runs + failed_runs) > 0 else 0
        },
        'timestamp': time.time(),
        'status': 'completed'
    }
    
    summary_file = config_dir / 'config_summary.json'
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

def main():
    """Función principal"""
    
    print("🎮 Iniciando comparación de algoritmos de búsqueda para Pokemon Red")
    print("⚡ Configuración optimizada para máxima velocidad")
    
    # Verificar entorno
    if not Path("PokemonRed.gb").exists():
        print("⚠️  Advertencia: No se encuentra PokemonRed.gb (necesario para referencia)")
    
    if not Path("comparison_agents").exists():
        print("❌ Error: Directorio comparison_agents no encontrado")
        return False
    
    # Ejecutar suite de comparaciones
    success = run_search_comparison_suite()
    
    if success:
        print("\n🎉 ¡Suite de comparaciones completada exitosamente!")
        print("📊 Próximo paso: Ejecutar análisis y generar visualizaciones")
        print("💡 Comando sugerido: python analyze_search_comparison.py")
    else:
        print("\n❌ La suite de comparaciones falló. Revisar logs para más detalles.")
    
    return success

if __name__ == "__main__":
    try:
        success = main()
        exit_code = 0 if success else 1
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n⏹️  Ejecución interrumpida por el usuario")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Error inesperado: {e}")
        sys.exit(1)