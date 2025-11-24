# Solución: Modelo Sub-Entrenado

## Problema Detectado

Tu modelo `pewter_brock_battle.zip` fue entrenado con **solo ~1023 pasos** (una prueba rápida), lo cual es insuficiente para aprender estrategias de combate efectivas.

**Síntomas:**
- ✅ Ventana de PyBoy se abre (pantalla gris/estática)
- ❌ Reward constante en 1843.04 (solo recompensa inicial, luego 0)
- ❌ Episodio termina en exactamente 2048 pasos (límite del entorno)
- ❌ El agente repite la misma acción sin progreso

## Causa Raíz

Con solo 1023 pasos de entrenamiento:
- El modelo apenas exploró el espacio de estados
- No aprendió secuencias de acciones efectivas
- PPO requiere miles de rollouts para convergencia

**Mínimo recomendado**: 40,000 pasos (lo que configuraste en `combat_plan_local`)

## Solución Paso a Paso

### 1. Entrenar el Modelo Completo

Abre `Local_Train.ipynb` y ejecuta la **Celda 18** (Sección 6):

```python
# Esta celda ejecuta el plan completo de combate
combat_runs_local = train_plan(
    agent_key='combat',
    plan=combat_plan_local,  # Ya configurado con 40,000 pasos
    default_timesteps=DEFAULT_TIMESTEPS_LOCAL,
    headless=DEFAULT_HEADLESS_LOCAL
)
```

**Tiempo estimado**: 
- GPU (RTX 3050): ~30-60 minutos
- CPU: ~2-4 horas

### 2. Verificar el Modelo Entrenado

Después del entrenamiento, verifica:

```python
from stable_baselines3 import PPO
model = PPO.load("models_local/combat/pewter_brock_battle.zip")
print(f"Pasos de entrenamiento: {model.num_timesteps:,}")
# Debería mostrar: ~40,000
```

### 3. Probar el Modelo Entrenado

```bash
python run_combat_agent_interactive.py --scenario pewter_brock --phase battle
```

**Resultado esperado:**
```
📦 Cargando modelo desde: ...\pewter_brock_battle.zip
   Pasos de entrenamiento del modelo: 40,960
   ✅ Modelo bien entrenado

🎮 Iniciando episodio (máx 10000 pasos)...

Paso 100/10000 | Reward: 2143.52 (+45.30) | Acción: A
Paso 200/10000 | Reward: 2398.76 (+32.18) | Acción: UP
🎯 Evento: battle_won
```

## Alternativa: Entrenar Baseline Ligero

Si solo quieres **comparar** sin esperar 40k pasos, usa la **Sección 11** de `Local_Train.ipynb`:

```python
# Entrena un baseline simple con los mismos 40k pasos
baseline_ligero_path = train_lightweight_baseline(
    scenario_id='pewter_brock',
    phase_name='battle',
    timesteps=40_000
)
```

Esto te dará un modelo PPO genérico para comparar contra tu Combat Agent especializado.

## Por Qué 40,000 Pasos?

| Pasos | Estado del Modelo | Uso Recomendado |
|-------|-------------------|-----------------|
| 1,000 | Sin aprendizaje | Solo pruebas de código |
| 10,000 | Aprendizaje básico | Debugging rápido |
| 40,000 | **Competente** | **Evaluación real** ✅ |
| 100,000 | Experto | Publicaciones/benchmarks |
| 200,000+ | Maestría | Competiciones |

Tu configuración actual (`40_000`) está en el punto óptimo para:
- ✅ Aprendizaje significativo
- ✅ Tiempo de entrenamiento razonable
- ✅ Comparaciones justas

## Verificación Post-Entrenamiento

Después de entrenar 40k pasos, deberías ver:

1. **Rewards variados**: No constantes, van cambiando
2. **Acciones diversas**: No solo una acción repetida
3. **Progreso visible**: Movimiento, combate, uso de items
4. **Episodios más cortos**: Termina antes de 2048 pasos si gana

## Troubleshooting

### "El entrenamiento tarda mucho"
- ✅ Verifica GPU activa: `torch.cuda.is_available()` debe ser `True`
- ✅ Usa `headless=True` en el plan de entrenamiento
- ✅ Reduce a 20,000 pasos si necesitas resultados rápidos

### "Quiero ver el progreso del entrenamiento"
La barra de progreso debería aparecer automáticamente (ya instalaste tqdm/rich).

### "El modelo entrenado sigue atascado"
- Verifica el escenario: `pewter_brock_battle.state` debe existir
- Revisa los logs de entrenamiento: ¿aumentó el reward promedio?
- Prueba con más pasos: 60,000 o 100,000

---

**🎯 Acción Inmediata**: Ejecuta la celda 18 de `Local_Train.ipynb` ahora para entrenar el modelo completo.
