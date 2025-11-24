# Scripts Interactivos para Agentes Entrenados

Este directorio contiene scripts Python independientes para ejecutar tus agentes entrenados en modo interactivo, similares a `run_pretrained_interactive.py` pero para tus modelos especializados.

## 📋 Scripts Disponibles

| Script | Descripción | Modelo Requerido |
|--------|-------------|------------------|
| `run_combat_agent_interactive.py` | Ejecuta CombatApexAgent | `models_local/combat/*.zip` |
| `run_puzzle_agent_interactive.py` | Ejecuta PuzzleSpeedAgent | `models_local/puzzle/*.zip` |
| `run_hybrid_agent_interactive.py` | Ejecuta HybridSageAgent | `models_local/hybrid/*.zip` |

## 🚀 Uso Básico

### Combat Agent
```bash
# Ejecutar en Pewter Brock (batalla)
python run_combat_agent_interactive.py --scenario pewter_brock --phase battle

# Ver el emulador (modo headless desactivado por defecto)
python run_combat_agent_interactive.py --scenario pewter_brock --phase battle

# Ejecutar sin ventana (más rápido)
python run_combat_agent_interactive.py --scenario pewter_brock --phase battle --headless

# Limitar pasos máximos
python run_combat_agent_interactive.py --scenario pewter_brock --phase battle --max-steps 5000
```

### Puzzle Agent
```bash
# Ejecutar en fase de puzzle
python run_puzzle_agent_interactive.py --scenario pewter_brock --phase puzzle

# Con ventana visible (para observar la navegación)
python run_puzzle_agent_interactive.py --scenario pewter_brock --phase puzzle

# Modo headless (para evaluación rápida)
python run_puzzle_agent_interactive.py --scenario pewter_brock --phase puzzle --headless
```

### Hybrid Agent
```bash
# Ejecutar agente híbrido
python run_hybrid_agent_interactive.py --scenario pewter_brock --phase battle

# Ver jugabilidad en tiempo real
python run_hybrid_agent_interactive.py --scenario vermillion_lt_surge --phase battle
```

## 📊 Parámetros Disponibles

| Parámetro | Descripción | Valores | Default |
|-----------|-------------|---------|---------|
| `--scenario` | ID del escenario | `pewter_brock`, `cerulean_misty`, etc. | `pewter_brock` |
| `--phase` | Fase del escenario | `battle`, `puzzle` | `battle` (combat/hybrid), `puzzle` (puzzle agent) |
| `--headless` | Sin ventana del emulador | flag (sin valor) | False |
| `--max-steps` | Límite de pasos | entero positivo | 10000 |

## 🎯 Escenarios Disponibles

Según `gym_scenarios/scenarios.json`:

- `pewter_brock` - Gimnasio de Pewter City (Brock)
- `cerulean_misty` - Gimnasio de Cerulean City (Misty)
- `vermillion_lt_surge` - Gimnasio de Vermillion City (Lt. Surge)
- `celadon_erika` - Gimnasio de Celadon City (Erika)
- `fuchsia_koga` - Gimnasio de Fuchsia City (Koga)
- `saffron_sabrina` - Gimnasio de Saffron City (Sabrina)
- `cinnabar_blaine` - Gimnasio de Cinnabar Island (Blaine)
- `viridian_giovanni` - Gimnasio de Viridian City (Giovanni)

## 📝 Ejemplos de Uso

### 1. Evaluar rendimiento del Combat Agent en Brock
```bash
python run_combat_agent_interactive.py --scenario pewter_brock --phase battle --max-steps 5000
```

**Salida esperada:**
```
============================================================
  EJECUTANDO COMBAT APEX AGENT - MODO INTERACTIVO
  Escenario: pewter_brock | Fase: battle
============================================================

📦 Cargando modelo desde: c:\...\models_local\combat\pewter_brock_battle.zip

🎮 Iniciando episodio (máx 5000 pasos)...

Paso 100/5000 | Reward acumulado: 12.34
Paso 200/5000 | Reward acumulado: 25.67
🎯 Evento: battle_won

============================================================
  RESUMEN DEL EPISODIO
============================================================
  Pasos ejecutados: 234
  Reward total: 45.89
  Reward promedio/paso: 0.1961
  Estado final: Completado
============================================================
```

### 2. Comparar visualmente Combat vs Baseline

**Baseline (PPO v2):**
```bash
python run_pretrained_interactive.py
```

**Tu Combat Agent:**
```bash
python run_combat_agent_interactive.py --scenario pewter_brock --phase battle
```

Observa las diferencias en:
- Velocidad de decisión
- Estrategia de combate
- Uso de items
- Selección de movimientos

### 3. Ver navegación del Puzzle Agent
```bash
# SIN headless para ver el movimiento del personaje
python run_puzzle_agent_interactive.py --scenario pewter_brock --phase puzzle
```

## 🔍 Comparación: Scripts vs Baseline

| Característica | `run_pretrained_interactive.py` | Tus scripts (`run_*_agent_interactive.py`) |
|----------------|--------------------------------|-------------------------------------------|
| **Modelo** | PPO genérico v2 (26M pasos) | Agentes especializados (40k-50k pasos) |
| **Tamaño** | ~10.5GB | ~100-500MB |
| **Entorno** | `RedGymEnv` estándar | `RedGymEnv` + wrappers especializados |
| **Observaciones** | RGB frames (3, 72, 80) | RGB + features adicionales (combate/puzzle) |
| **Rewards** | Recompensa genérica | Recompensas especializadas por tarea |
| **Carga de RAM** | Requiere >20GB total | Requiere ~6-8GB |

## ⚙️ Ventajas de los Scripts Interactivos

✅ **No necesitas el modelo gigante de 26M pasos**  
✅ **Consume menos RAM (6-8GB vs >20GB)**  
✅ **Carga más rápido (~5 segundos vs ~30 segundos)**  
✅ **Mismo formato que `run_pretrained_interactive.py`**  
✅ **Puedes modificarlos fácilmente**  
✅ **Compatibles con Windows, Linux, macOS**  

## 🛠️ Requisitos Previos

1. **Modelos entrenados** - Ejecuta primero `Local_Train.ipynb` sección 6-8
2. **Archivos .state** - Genera con `generate_gym_states.py`
3. **Dependencias instaladas**:
   ```bash
   pip install stable-baselines3 pyboy gymnasium numpy
   ```

## 🐛 Solución de Problemas

### Error: "Modelo no encontrado"
```bash
# Verifica que el modelo existe
ls models_local/combat/pewter_brock_battle.zip
```
Si no existe, entrénalo primero en `Local_Train.ipynb`.

### Error: "Archivo de estado no encontrado"
```bash
# Genera los archivos .state
python generate_gym_states.py
```

### Error: OpenMP conflict
Ya incluido en los scripts:
```python
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
```

### Kernel crash / ventana SDL no responde
Usa `--headless`:
```bash
python run_combat_agent_interactive.py --scenario pewter_brock --phase battle --headless
```

## 📈 Métricas de Evaluación

Los scripts muestran automáticamente:

- **Pasos ejecutados**: Eficiencia del agente
- **Reward total**: Desempeño acumulado
- **Reward promedio/paso**: Consistencia
- **Estado final**: Éxito (`Completado`) vs fallo (`Truncado`/`Máx pasos`)

## 🎓 Cómo Usar para Comparaciones

### Opción 1: Evaluación Manual (Visual)
```bash
# 1. Ejecutar baseline
python run_pretrained_interactive.py

# 2. Ejecutar tu agente
python run_combat_agent_interactive.py --scenario pewter_brock --phase battle

# 3. Comparar visualmente y anotar métricas
```

### Opción 2: Evaluación Automatizada (Recomendado)
Usa `Local_Train.ipynb` sección 11:
```python
# Entrena baseline ligero (40k pasos)
baseline_ligero_path = train_lightweight_baseline(
    scenario_id='pewter_brock',
    phase_name='battle',
    timesteps=40_000
)

# Compara automáticamente
df_comparison = run_comparison_lightweight(
    {'combat': combat_plan_local},
    baseline_path=baseline_ligero_path,
    skip_baseline=False
)
```

## 📚 Recursos Adicionales

- **Documentación de agentes**: `advanced_agents/README.md`
- **Guía de entrenamiento**: `README_LOCAL_TRAINING.md`
- **Escenarios disponibles**: `gym_scenarios/scenarios.json`
- **Configuración de entorno**: `advanced_agents/train_agents.py`

## 💡 Tips

1. **Headless para benchmarks** - Usa `--headless` para evaluaciones masivas
2. **Sin headless para depuración** - Observa el comportamiento visualmente
3. **Max-steps razonable** - 5000-10000 es suficiente para la mayoría de escenarios
4. **Escenarios progresivos** - Empieza con `pewter_brock`, luego avanza a gimnasios más difíciles

---

**¿Dudas?** Revisa los comentarios en el código de cada script o consulta `Local_Train.ipynb`.
