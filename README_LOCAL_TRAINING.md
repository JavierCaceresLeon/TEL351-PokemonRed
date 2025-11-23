# Guía de Entrenamiento Local para Agentes Especializados

Este documento explica cómo entrenar los 3 tipos de agentes avanzados (Combat, Puzzle, Hybrid) y el agente Baseline (PPO v2) en los 8 escenarios definidos, utilizando scripts locales con visualización opcional.

## Requisitos Previos

Asegúrate de tener instaladas las dependencias (PyBoy, Stable-Baselines3, etc.) y de estar en el entorno virtual correcto.

## Estructura de Scripts

Se han creado 4 scripts principales, uno para cada tipo de agente:

1.  `train_combat_agent.py`: Entrena el agente especializado en combate (`CombatApexAgent`).
2.  `train_puzzle_agent.py`: Entrena el agente especializado en puzzles (`PuzzleSpeedAgent`).
3.  `train_hybrid_agent.py`: Entrena el agente híbrido (`HybridSageAgent`).
4.  `train_baseline_agent.py`: Entrena un agente PPO estándar (v2) para comparación.

Todos estos scripts utilizan `train_core.py` como núcleo común.

## Cómo Ejecutar

Cada script acepta los siguientes argumentos:

*   `--scenario`: El ID del escenario (ej. `pewter_brock`, `cerulean_misty`). Ver `gym_scenarios/scenarios.json` para la lista completa.
*   `--phase`: La fase del escenario (`battle` o `puzzle`). Por defecto depende del agente, pero se recomienda especificarlo.
*   `--timesteps`: Número de pasos de entrenamiento (por defecto 100000).
*   `--no-headless`: **Importante**. Usa esta bandera para ver la ventana del GameBoy mientras entrena. Si no se pone, corre en modo "headless" (sin ventana).
*   `--run-name`: Prefijo opcional para el nombre del experimento en los logs.

### Ejemplos

**1. Entrenar Agente de Combate en Pewter Gym (Batalla contra Brock) con visualización:**

```powershell
python train_combat_agent.py --scenario pewter_brock --phase battle --no-headless
```

**2. Entrenar Agente de Puzzle en Pewter Gym (Llegar a Brock) sin visualización (más rápido):**

```powershell
python train_puzzle_agent.py --scenario pewter_brock --phase puzzle
```

**3. Entrenar Agente Baseline (PPO v2) para comparar:**

```powershell
python train_baseline_agent.py --scenario pewter_brock --phase battle --no-headless
```

## Métricas y Comparación

Los entrenamientos guardan logs compatibles con Tensorboard en la carpeta `advanced_agents/runs`.

Para ver las gráficas y comparar el rendimiento de los agentes:

```powershell
tensorboard --logdir advanced_agents/runs
```

Abre el navegador en `http://localhost:6006`. Podrás ver curvas de recompensa, longitud de episodios, etc., y comparar "combat_apex" vs "baseline".

## Modelos Guardados

Los modelos entrenados se guardan automáticamente en la carpeta `models/` con el formato `{tipo_agente}_{escenario}_{fase}.zip`.
