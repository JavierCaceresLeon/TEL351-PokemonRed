
import sys
import time
import numpy as np
import os
from pathlib import Path
import subprocess

# Añadir la ruta de v2 para importar RedGymEnv
sys.path.append(str(Path(__file__).parent.parent / 'v2'))
from v2_agent import V2EpsilonGreedyAgent

def run_comparison_for_epsilon(epsilon_config, num_runs=11):
    """
    Ejecuta una serie de comparaciones para una configuración de épsilon dada.
    """
    epsilon_label = f"eps_start_{epsilon_config['epsilon_start']}".replace('.', '_')
    print(f"🚀 Ejecutando {num_runs} veces para la configuración: {epsilon_label}")

    for i in range(num_runs):
        print(f"--- Inicio de la ejecución {i+1}/{num_runs} para {epsilon_label} ---")
        
        # Usamos subprocess para aislar cada ejecución y asegurar que los recursos se liberen
        # correctamente. Esto es más robusto que llamar a una función en un bucle.
        
        # Construir la ruta de resultados
        results_dir = Path(__file__).parent.parent / "RESULTADOS" / "epsilon_greedy_variations" / epsilon_label / str(i+1)
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Comando para ejecutar el script de agente individual
        command = [
            sys.executable,
            str(Path(__file__).parent / "individual_epsilon_run.py"),
            "--epsilon_start", str(epsilon_config['epsilon_start']),
            "--epsilon_min", str(epsilon_config['epsilon_min']),
            "--epsilon_decay", str(epsilon_config['epsilon_decay']),
            "--results_dir", str(results_dir)
        ]
        
        try:
            # Redirigir la salida para evitar que el log principal se sature
            with open(results_dir / "output.log", "w") as log_file:
                process = subprocess.Popen(command, stdout=log_file, stderr=subprocess.STDOUT)
                process.wait(timeout=1800) # Timeout de 30 minutos por si algo se cuelga

            if process.returncode == 0:
                print(f"✅ Ejecución {i+1} para {epsilon_label} completada. Resultados en: {results_dir}")
            else:
                print(f"⚠️ Ejecución {i+1} para {epsilon_label} falló con código {process.returncode}. Revisa output.log.")

        except subprocess.TimeoutExpired:
            print(f"❌ Ejecución {i+1} para {epsilon_label} excedió el tiempo límite. Proceso terminado.")
            process.kill()
        except Exception as e:
            print(f"❌ Error inesperado en la ejecución {i+1} para {epsilon_label}: {e}")

        time.sleep(2) # Pequeña pausa entre ejecuciones

if __name__ == "__main__":
    # Definir las 5 configuraciones de épsilon a comparar
    epsilon_configs = [
        {'epsilon_start': 0.9, 'epsilon_min': 0.1, 'epsilon_decay': 0.999},  # Alta exploración
        {'epsilon_start': 0.7, 'epsilon_min': 0.1, 'epsilon_decay': 0.9995}, # Exploración moderada-alta
        {'epsilon_start': 0.5, 'epsilon_min': 0.05, 'epsilon_decay': 0.9995},# Configuración base
        {'epsilon_start': 0.3, 'epsilon_min': 0.01, 'epsilon_decay': 0.9998},# Baja exploración
        {'epsilon_start': 0.1, 'epsilon_min': 0.01, 'epsilon_decay': 0.9999} # Muy baja exploración (casi greedy)
    ]

    # Limpiar directorio de resultados anterior si existe
    base_results_dir = Path(__file__).parent.parent / "RESULTADOS" / "epsilon_greedy_variations"
    if base_results_dir.exists():
        import shutil
        print(f"🧹 Limpiando directorio de resultados anterior: {base_results_dir}")
        shutil.rmtree(base_results_dir)
    
    print("="*50)
    print("🔬 INICIANDO COMPARACIÓN DE CONFIGURACIONES EPSILON-GREEDY 🔬")
    print("="*50)

    for config in epsilon_configs:
        run_comparison_for_epsilon(config, num_runs=11)

    print("\n🎉 Todas las comparaciones han finalizado.")
