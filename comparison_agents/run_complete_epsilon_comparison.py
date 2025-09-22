#!/usr/bin/env python3
"""
Script maestro para ejecutar comparaciones de diferentes configuraciones Epsilon-Greedy
en Pokemon Red a máxima velocidad.
"""

import sys
import os
import time
import subprocess
from pathlib import Path
import shutil

def run_epsilon_comparison_suite():
    """Ejecutar suite completa de comparaciones epsilon-greedy"""
    
    # Configuraciones de epsilon a probar
    epsilon_configs = [
        {'epsilon_start': 0.9, 'epsilon_min': 0.1, 'epsilon_decay': 0.999, 'name': 'alta_exploracion'},
        {'epsilon_start': 0.7, 'epsilon_min': 0.05, 'epsilon_decay': 0.9995, 'name': 'moderada_alta'},
        {'epsilon_start': 0.5, 'epsilon_min': 0.05, 'epsilon_decay': 0.9995, 'name': 'balanceada'},
        {'epsilon_start': 0.3, 'epsilon_min': 0.01, 'epsilon_decay': 0.9998, 'name': 'conservadora'},
        {'epsilon_start': 0.1, 'epsilon_min': 0.01, 'epsilon_decay': 0.9999, 'name': 'muy_greedy'}
    ]
    
    # Directorio base de resultados
    base_results_dir = Path(__file__).parent.parent / "RESULTADOS" / "epsilon_greedy_comparison"
    
    # Limpiar resultados anteriores
    if base_results_dir.exists():
        print(f"🧹 Limpiando resultados anteriores: {base_results_dir}")
        shutil.rmtree(base_results_dir)
    
    print("="*80)
    print("🔬 INICIANDO COMPARACIÓN COMPLETA DE EPSILON-GREEDY")
    print("="*80)
    print(f"Configuraciones a probar: {len(epsilon_configs)}")
    print(f"Ejecuciones por configuración: 11")
    print(f"Total de ejecuciones: {len(epsilon_configs) * 11}")
    print()
    
    total_start_time = time.time()
    successful_runs = 0
    failed_runs = 0
    
    for config_idx, config in enumerate(epsilon_configs, 1):
        print(f"\n{'='*60}")
        print(f"📊 CONFIGURACIÓN {config_idx}/{len(epsilon_configs)}: {config['name'].upper()}")
        print(f"   ε_start={config['epsilon_start']}, ε_min={config['epsilon_min']}, ε_decay={config['epsilon_decay']}")
        print(f"{'='*60}")
        
        config_start_time = time.time()
        config_successful = 0
        config_failed = 0
        
        for run_idx in range(1, 12):  # 11 ejecuciones
            print(f"\n🚀 Ejecutando {config['name']} - Run {run_idx}/11...")
            
            # Crear directorio de resultados para esta ejecución
            run_results_dir = base_results_dir / config['name'] / str(run_idx)
            run_results_dir.mkdir(parents=True, exist_ok=True)
            
            # Preparar comando
            script_path = Path(__file__).parent / "simple_epsilon_runner.py"
            command = [
                sys.executable,
                str(script_path),
                "--epsilon_start", str(config['epsilon_start']),
                "--epsilon_min", str(config['epsilon_min']),
                "--epsilon_decay", str(config['epsilon_decay']),
                "--results_dir", str(run_results_dir),
                "--max_steps", "15000"
            ]
            
            # Ejecutar con timeout
            run_start_time = time.time()
            try:
                result = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    timeout=300  # 5 minutos max por ejecución
                )
                
                run_elapsed = time.time() - run_start_time
                
                if result.returncode == 0:
                    print(f"   ✅ Completado en {run_elapsed:.1f}s")
                    successful_runs += 1
                    config_successful += 1
                else:
                    print(f"   ❌ Falló (código {result.returncode})")
                    print(f"   Error: {result.stderr[:200]}...")
                    failed_runs += 1
                    config_failed += 1
                    
            except subprocess.TimeoutExpired:
                print(f"   ⏰ Timeout después de 5 minutos")
                failed_runs += 1
                config_failed += 1
                
            except Exception as e:
                print(f"   ❌ Error inesperado: {e}")
                failed_runs += 1
                config_failed += 1
        
        config_elapsed = time.time() - config_start_time
        print(f"\n📈 Configuración {config['name']} completada:")
        print(f"   Exitosos: {config_successful}/11")
        print(f"   Fallidos: {config_failed}/11")
        print(f"   Tiempo total: {config_elapsed:.1f}s")
        print(f"   Tiempo promedio por run: {config_elapsed/11:.1f}s")
    
    total_elapsed = time.time() - total_start_time
    
    print("\n" + "="*80)
    print("🎉 COMPARACIÓN COMPLETADA")
    print("="*80)
    print(f"Total exitosos: {successful_runs}")
    print(f"Total fallidos: {failed_runs}")
    print(f"Tiempo total: {total_elapsed/60:.1f} minutos")
    print(f"Resultados guardados en: {base_results_dir}")
    
    return successful_runs, failed_runs

def main():
    """Función principal"""
    print("Iniciando comparación de configuraciones Epsilon-Greedy...")
    
    try:
        successful, failed = run_epsilon_comparison_suite()
        
        if successful > 0:
            print(f"\n🎯 Listo para generar gráficos y análisis con {successful} ejecuciones exitosas.")
            return 0
        else:
            print(f"\n❌ No se completaron ejecuciones exitosas.")
            return 1
            
    except KeyboardInterrupt:
        print("\n\n⚠️ Proceso interrumpido por el usuario.")
        return 1
    except Exception as e:
        print(f"\n❌ Error inesperado: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())