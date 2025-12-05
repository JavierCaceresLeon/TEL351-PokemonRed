# Demo: Modelo PPO Ultimate

Este branch incluye solo lo necesario para ejecutar el modelo entrenado

## Contenido clave

- `session_ultimate_38883696/ultimate_combat_final.zip`: checkpoint del modelo Ultimate (16M steps, ~2 semanas de entrenamiento).
- `pewter_gym_configured.state`: estado inicial para replicar la evaluación.
- `run_simple.py`: script de evaluación (abre una ventana, ejecuta el modelo y guarda métricas).
- `requirements.txt`: dependencias para recrear el entorno.

## Cómo ejecutar

1. Instalar dependencias:

```bash
pip install -r requirements.txt
```

1. Ejecutar la demo desde la raíz del repo:

```bash
python run_simple.py
```

1. Se abrirá la ventana de emulación y, al terminar, se generará `RESULTADOS/eval_compare_<uuid>/comparison.json` con las métricas.

## Métricas obtenidas (última corrida)

```text
======================================================================
EVALUACION DEL MODELO
======================================================================
Modelo                    | Win | Daño hecho | Daño recibido  | Faints rival | Faints propios | Pasos
------------------------------------------------------------------------------------------------------
Ultimate (nuevo)          | ✖  |       27.0 |           87.0 |            1 |              3 |    386
```

## Función de recompensa usada

Resumen de `CombatEvalEnv.update_reward` (entorno v2):

- **Daño infligido**: recompensa proporcional; primer golpe +5.0.
- **Umbrales de HP rival**: +2.0 al bajar de 50%, +3.0 al bajar de 20%.
- **Daño recibido**: penalización proporcional.
- **Derrotar rival**: +20.0.
- **Morir sin haber dañado**: -5.0.
- **Navegación de menú**: incentiva atacar (ligero bonus al menú de ataque, penaliza defensa/huida y spam de menús >5 ticks).
- **Mapa inválido (map==2)**: penaliza y fuerza fin de episodio.

## Cómo se entrenó

- Algoritmo: PPO (stable-baselines3) sobre `RedGymEnv` v2.
- Pasos totales: **16 millones**.
- Duración aproximada: **2 semanas**.
- Configuración: acción cada 24 frames, 3 frames stackeados, recompensas anteriores.

## Apreciación final

El modelo enfocado en combate muestra capacidad de infligir daño inicial y conseguir algún faint, pero en la corrida reportada no logra ganar (pierde 3 veces al rival 1 vez, 27 de daño hecho vs 87 recibido en 386 pasos). No cumple todavía el objetivo de victoria consistente en Brock Gym; se sugieren más iteraciones de entrenamiento o ajustes en la recompensa para mayor estabilidad/agresividad.
