# Guía de Inicio Rápido - Agente de Combate Pokémon Red

## ⚡ Setup en 5 Minutos

### 1. Instalación

```bash
cd PokemonCombatAgent
pip install -r requirements.txt
```

### 2. Configuración de ROM

Copia tu ROM de Pokémon Red al directorio padre:

```bash
# La ROM debe estar en el directorio padre (junto con PokemonRedExperiments)
# Estructura esperada:
# repos_git/
#   ├── PokemonRed.gb              ← ROM aquí
#   ├── has_pokedex_nballs.state   ← Estado inicial aquí
#   ├── PokemonRedExperiments/
#   ├── TEL351-PokemonRed/
#   └── PokemonCombatAgent/        ← Este proyecto
```

**Verificar que la ROM existe:**
```powershell
Test-Path ..\PokemonRed.gb
# Debe devolver: True
```

### 3. Entrenamiento Rápido (Demo)

```bash
# Entrenamiento corto para probar (100K steps, ~10 minutos)
python train_combat_agent.py --timesteps 100000 --num-envs 4 --headless

# Output esperado:
# ============================================================
# Training Combat-Specialized Pokemon Red Agent
# ============================================================
# Session: combat_session_a1b2c3d4
# Timesteps: 100,000
# Parallel envs: 4
# ============================================================
```

### 4. Ver Progreso

El entrenamiento guardará archivos en `sessions/combat_session_XXXXX/`:
- `combat_agent_XXXX_steps.zip` - Checkpoints cada 50K steps
- `agent_stats_*.csv.gz` - Estadísticas detalladas
- `curframe_*.jpeg` - Captura de pantalla actual

## 🚀 Entrenamiento Completo

### Configuración Recomendada

```bash
python train_combat_agent.py \
    --timesteps 1000000 \
    --num-envs 16 \
    --max-steps 16384 \
    --checkpoint-freq 50000 \
    --session-name combat_v1 \
    --headless
```

**Parámetros Explicados:**
- `--timesteps 1000000`: Total de steps (1M steps ≈ 2-3 horas con 16 CPUs)
- `--num-envs 16`: Número de entornos paralelos (ajustar según tu CPU)
- `--max-steps 16384`: Steps por episodio (~1024 combates)
- `--checkpoint-freq 50000`: Guardar cada 50K steps
- `--session-name`: Nombre personalizado para la sesión
- `--headless`: Sin GUI (más rápido)

### Monitoreo con TensorBoard

```bash
# En otra terminal
cd sessions/combat_v1
tensorboard --logdir .

# Abrir navegador en: http://localhost:6006
```

## 📊 Comparación con Baseline

### 1. Entrenar Baseline PPO (usando repositorio original)

```bash
# Ir al repositorio original
cd ../PokemonRedExperiments/baselines

# Entrenar PPO básico
python run_baseline_parallel.py
# Esto generará: session_XXXXX/poke_YYYY_steps.zip
```

### 2. Ejecutar Comparación

```bash
cd ../../PokemonCombatAgent

python compare_agents.py \
    --combat-agent sessions/combat_v1/combat_agent_final \
    --baseline-agent ../PokemonRedExperiments/baselines/session_XXXXX/poke_1000000_steps \
    --episodes 100 \
    --output-dir comparison_results
```

### 3. Ver Resultados

Los resultados se guardarán en `comparison_results/`:

```
comparison_results/
├── combat_agent_metrics.csv     # Métricas detalladas del combat agent
├── baseline_agent_metrics.csv   # Métricas detalladas del baseline
├── comparison_results.csv       # Comparación estadística
└── summary.json                 # Resumen ejecutivo
```

**Ejemplo de Output:**

```
============================================================
Statistical Comparison
============================================================

Win Rate:
  Combat Agent:   0.8500 ± 0.0450
  Baseline Agent: 0.6500 ± 0.0620
  Difference:     +0.2000 (+30.8%)
  p-value:        0.0001 ✓ SIGNIFICANT
  Cohen's d:      3.521

HP Conservation:
  Combat Agent:   0.7200 ± 0.1100
  Baseline Agent: 0.4500 ± 0.1300
  Difference:     +0.2700 (+60.0%)
  p-value:        0.0000 ✓ SIGNIFICANT
  Cohen's d:      2.250
```

## 🎯 Casos de Uso Específicos

### Caso 1: Entrenar para Brock (Primer Gimnasio)

```bash
# Necesitas crear un estado inicial justo antes de Brock
# Opción A: Usa un estado pre-existente cerca de Pewter City
python train_combat_agent.py \
    --init-state ../pewter_gym_entrance.state \
    --timesteps 500000 \
    --session-name brock_specialist

# Opción B: Entrenar con estado general y evaluar solo en Brock
# (El agente aprenderá patrones generales que aplica a Brock)
```

### Caso 2: Entrenar Solo con Combates (Sin Exploración)

```bash
# Crear un estado .state justo antes de combate
# Luego entrenar episodios cortos que siempre terminan después del combate

python train_combat_agent.py \
    --init-state ../pre_battle.state \
    --max-steps 2048 \
    --timesteps 1000000 \
    --session-name pure_combat
```

### Caso 3: Continuar Entrenamiento Desde Checkpoint

```bash
python train_combat_agent.py \
    --load-checkpoint sessions/combat_v1/combat_agent_500000_steps \
    --timesteps 1000000 \
    --session-name combat_v1_continued
```

## 🔧 Ajuste de Hiperparámetros

### Agente Muy Conservador (Evita riesgos)

```bash
python train_combat_agent.py \
    --gamma 0.99 \
    --ent-coef 0.005 \
    --timesteps 1000000
```

### Agente Más Exploratorio (Toma más riesgos)

```bash
python train_combat_agent.py \
    --gamma 0.95 \
    --ent-coef 0.02 \
    --timesteps 1000000
```

### Más Enfoque en Recompensas a Largo Plazo

```bash
python train_combat_agent.py \
    --gamma 0.999 \
    --n-epochs 3 \
    --timesteps 1000000
```

## 🐛 Troubleshooting

### Error: "ROM not found"

```bash
# Verificar ruta
ls ..\PokemonRed.gb

# Si no existe, copiar desde donde la tengas:
Copy-Item "C:\ruta\a\tu\PokemonRed.gb" "..\PokemonRed.gb"
```

### Error: "Could not load state file"

```bash
# Verificar que el archivo .state existe
ls ..\has_pokedex_nballs.state

# Si no existe, puedes usar otro estado o crearlo tú mismo
# (jugar manualmente y guardar estado desde PyBoy)
```

### Entrenamiento Muy Lento

```bash
# Reducir número de entornos paralelos
python train_combat_agent.py --num-envs 4  # en lugar de 16

# Reducir frecuencia de acciones (cada acción dura menos)
python train_combat_agent.py --action-freq 12  # en lugar de 24

# Ambas opciones
python train_combat_agent.py --num-envs 4 --action-freq 12
```

### Out of Memory (RAM)

```bash
# Reducir batch size y número de entornos
python train_combat_agent.py \
    --num-envs 4 \
    --batch-size 256 \
    --max-steps 8192
```

### Agente No Aprende (Reward Plano)

```powershell
# Ver logs detallados
python train_combat_agent.py --timesteps 100000 --num-envs 2

# Si reward se queda en 0-10:
# 1. Verificar que el estado inicial está cerca de combates
# 2. Aumentar exploration
python train_combat_agent.py --ent-coef 0.02 --timesteps 100000

# 3. Reducir gamma para enfocarse en recompensas inmediatas
python train_combat_agent.py --gamma 0.95 --timesteps 100000
```

## 📈 Interpretando Resultados

### Métricas Clave

| Métrica | Bueno | Excelente | Interpretación |
|---------|-------|-----------|----------------|
| **Win Rate** | > 70% | > 85% | % de combates ganados |
| **HP Conserved** | > 60% | > 75% | % de HP al final vs inicio |
| **Deaths/Episode** | < 1.5 | < 0.5 | Pokemon derrotados por episodio |
| **Efficient Kills** | > 50% | > 80% | Victorias sin perder Pokemon |

### Signos de Buen Aprendizaje

```
step: 10000  victories: 5.00  hp_conserve: 35.00  W/L: 5/2
step: 20000  victories: 12.00 hp_conserve: 48.00  W/L: 12/3  ← Mejorando
step: 30000  victories: 20.00 hp_conserve: 65.00  W/L: 20/3  ← Bien!
```

### Signos de Problemas

```
step: 10000  victories: 1.00  hp_conserve: 10.00  W/L: 1/8   ← Perdiendo mucho
step: 20000  victories: 1.00  hp_conserve: 10.00  W/L: 1/15  ← NO aprende
```

**Solución:** Revisar estado inicial, aumentar exploración, o reducir complejidad.

## 🎓 Próximos Pasos

1. **Analizar Estadísticas Detalladas:**
   ```python
   import pandas as pd
   stats = pd.read_csv('sessions/combat_v1/agent_stats_*.csv.gz')
   print(stats.describe())
   ```

2. **Visualizar Progreso:**
   ```python
   import matplotlib.pyplot as plt
   plt.plot(stats['step'], stats['battles_won'])
   plt.xlabel('Steps')
   plt.ylabel('Battles Won')
   plt.show()
   ```

3. **Probar Interactivamente:**
   ```bash
   # Modificar train_combat_agent.py para no headless
   python train_combat_agent.py --no-headless --num-envs 1 --timesteps 10000
   ```

4. **Exportar para Publicación:**
   ```bash
   # Crear reporte completo
   python compare_agents.py \
       --combat-agent sessions/combat_v1/combat_agent_final \
       --baseline-agent ../PokemonRedExperiments/baselines/session_X/poke_Y \
       --episodes 200 \
       --output-dir final_comparison
   ```

## 📚 Recursos Adicionales

- **README Principal**: `README.md` - Documentación completa
- **Código del Entorno**: `combat_gym_env.py` - Ver recompensas y métricas
- **Configuración**: `train_combat_agent.py` - Ajustar hiperparámetros

¡Buena suerte con tu entrenamiento! 🎮🔥
