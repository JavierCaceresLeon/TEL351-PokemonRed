#!/usr/bin/env python3
"""
Ejecución Paralela Visualizable en PyBoy de TODOS los algoritmos
================================================================

Lanza cada algoritmo en un proceso separado, con su propia ventana PyBoy,
partiendo desde el estado inicial del protagonista (init.state). Ajusta el
reporte de tiempo a tiempo real según real_time_factor.
"""
import os
import sys
import time
import multiprocessing as mp
from pathlib import Path

# Asegurar imports locales
sys.path.append(str(Path(__file__).parent))

from professional_algorithm_executor import ProfessionalAlgorithmExecutor


def _run_single(algo_key: str, cfg_overrides: dict | None = None) -> None:
    executor = ProfessionalAlgorithmExecutor()
    if algo_key not in executor.algorithm_configs:
        print(f"Algoritmo desconocido: {algo_key}")
        return

    cfg = executor.algorithm_configs[algo_key]
    # Forzar visualización y estado inicial
    cfg.visualization_enabled = True
    # Aceleración controlada: mantener < 12 para estabilidad
    cfg.real_time_factor = min(getattr(cfg, 'real_time_factor', 6.0) or 6.0, 12.0)
    # Aplicar overrides si vienen
    if cfg_overrides:
        for k, v in cfg_overrides.items():
            setattr(cfg, k, v)

    print(f"[Child:{os.getpid()}] Ejecutando {cfg.algorithm} ({cfg.variant}) con ventana PyBoy...")
    res = executor.execute_algorithm(cfg)
    executor.save_execution_results(res)
    status = 'OK' if res.success else f"FAIL ({res.error_message})"
    print(f"[Child:{os.getpid()}] {cfg.algorithm}/{cfg.variant} => {status}")


def run_all(parallelism: int | None = None) -> None:
    executor = ProfessionalAlgorithmExecutor()
    algo_keys = list(executor.algorithm_configs.keys())

    # Afinar parámetros para demostración visual estable
    overrides = {
        'ppo': { 'max_steps': 600, 'time_limit': 90.0 },
        'epsilon_greedy_alta_exploracion': { 'max_steps': 600, 'time_limit': 80.0 },
        'epsilon_greedy_balanceada': { 'max_steps': 600, 'time_limit': 80.0 },
        'epsilon_greedy_conservadora': { 'max_steps': 600, 'time_limit': 80.0 },
        'astar': { 'max_steps': 500, 'time_limit': 70.0 },
        'bfs': { 'max_steps': 500, 'time_limit': 70.0 },
        'tabu_search': { 'max_steps': 600, 'time_limit': 80.0 },
        'hill_climbing_steepest': { 'max_steps': 500, 'time_limit': 70.0 },
        'hill_climbing_first': { 'max_steps': 500, 'time_limit': 70.0 },
        'hill_climbing_restart': { 'max_steps': 500, 'time_limit': 70.0 },
        'simulated_annealing': { 'max_steps': 600, 'time_limit': 80.0 },
    }

    # Por defecto, ejecutar todos en paralelo (hasta número de CPUs-1)
    if parallelism is None:
        try:
            parallelism = max(1, mp.cpu_count() - 1)
        except Exception:
            parallelism = 2

    print("=== Lanzando ejecución paralela de algoritmos con PyBoy ===")
    print(f"Algoritmos: {len(algo_keys)} | Paralelismo: {parallelism}")

    with mp.Pool(processes=parallelism) as pool:
        tasks = []
        for key in algo_keys:
            ov = overrides.get(key, {})
            tasks.append(pool.apply_async(_run_single, (key, ov)))
        # Esperar a que terminen
        for t in tasks:
            try:
                t.get()
            except Exception as e:
                print(f"Error en subproceso: {e}")

    print("\n=== Ejecución paralela completada ===")
    print("Resultados guardados en RESULTADOS/enhanced_execution/")


if __name__ == "__main__":
    # Permitir seleccionar paralelismo por variable de entorno
    par = os.environ.get('ALGOS_PAR', '').strip()
    p = None
    if par.isdigit():
        p = int(par)
    run_all(p)
