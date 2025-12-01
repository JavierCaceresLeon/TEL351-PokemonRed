# Plan de Acción Inmediato - Proyecto Combat Agent

## 🎯 Objetivo
Entrenar un agente PPO especializado en combates y compararlo científicamente con el PPO baseline del repositorio original.

---

## 📅 Cronograma Sugerido (3-5 días)

### Día 1: Setup y Verificación ✅

**Tareas:**
1. ✅ Leer `README.md` y `TECHNICAL_ANALYSIS.md` para entender el proyecto
2. ✅ Instalar dependencias: `pip install -r requirements.txt`
3. ✅ Verificar que tienes la ROM y archivos `.state`:
   ```powershell
   Test-Path ..\PokemonRed.gb
   Test-Path ..\has_pokedex_nballs.state
   ```
4. ✅ Ejecutar prueba rápida (100K steps, ~10 min):
   ```bash
   python train_combat_agent.py --timesteps 100000 --num-envs 4 --headless
   ```

**Criterio de Éxito:**
- ✅ Script corre sin errores
- ✅ Ves output como: `step: 1000  victories: 2.00  hp_conserve: 15.00  W/L: 2/1`
- ✅ Se crea directorio `sessions/combat_session_XXXXX/`

---

### Día 2: Entrenamiento Combat Agent 🚀

**Tareas:**
1. Lanzar entrenamiento completo (1M steps, ~2-3 horas con 16 CPUs):
   ```bash
   python train_combat_agent.py \
       --timesteps 1000000 \
       --num-envs 16 \
       --session-name combat_v1 \
       --checkpoint-freq 100000 \
       --headless
   ```

2. Mientras entrena, monitorear en otra terminal:
   ```bash
   cd sessions/combat_v1
   tensorboard --logdir .
   # Abrir: http://localhost:6006
   ```

3. Verificar checkpoints cada 100K steps:
   ```bash
   ls sessions/combat_v1/combat_agent_*.zip
   # Deberías ver: combat_agent_100000_steps.zip, ..., combat_agent_1000000_steps.zip
   ```

**Criterio de Éxito:**
- ✅ Entrenamiento completa 1M steps
- ✅ Recompensas incrementan con el tiempo (ver TensorBoard)
- ✅ Win Rate aumenta (de ~40% inicial a >70% final)
- ✅ Modelo final guardado: `combat_agent_final.zip`

---

### Día 3: Entrenar Baseline PPO (para comparación)

**Opción A: Usar modelo pre-existente del repositorio original**
```bash
# Si ya entrenaron en PokemonRedExperiments antes:
ls ../PokemonRedExperiments/baselines/session_*/poke_*.zip
# Usar el más reciente
```

**Opción B: Entrenar baseline desde cero**
```bash
cd ../PokemonRedExperiments/baselines

# Editar run_baseline_parallel.py:
# Cambiar: num_cpu = 16 (línea ~35)
# Cambiar: learn_steps = 10 (línea ~52, para 1M steps total)

python run_baseline_parallel.py
```

**Criterio de Éxito:**
- ✅ Tienes un modelo baseline PPO entrenado por ~1M steps
- ✅ Modelo guardado en: `../PokemonRedExperiments/baselines/session_XXXXX/poke_YYYYY_steps.zip`

---

### Día 4: Evaluación y Comparación 📊

**Tareas:**
1. Ejecutar comparación científica:
   ```bash
   cd PokemonCombatAgent
   
   python compare_agents.py \
       --combat-agent sessions/combat_v1/combat_agent_final \
       --baseline-agent ../PokemonRedExperiments/baselines/session_XXXXX/poke_1000000_steps \
       --episodes 100 \
       --output-dir comparison_results
   ```

2. Analizar resultados:
   ```bash
   # Ver resumen
   cat comparison_results/summary.json
   
   # Ver comparación detallada
   python -c "import pandas as pd; df = pd.read_csv('comparison_results/comparison_results.csv'); print(df)"
   ```

3. Visualizar métricas:
   ```python
   import pandas as pd
   import matplotlib.pyplot as plt
   
   combat = pd.read_csv('comparison_results/combat_agent_metrics.csv')
   baseline = pd.read_csv('comparison_results/baseline_agent_metrics.csv')
   
   # Gráfico Win Rate
   plt.figure(figsize=(10, 6))
   plt.hist([combat['win_rate'], baseline['win_rate']], label=['Combat Agent', 'Baseline'])
   plt.xlabel('Win Rate')
   plt.ylabel('Frequency')
   plt.legend()
   plt.title('Win Rate Distribution')
   plt.savefig('win_rate_comparison.png')
   ```

**Criterio de Éxito:**
- ✅ Combat Agent tiene **mayor Win Rate** que Baseline (esperado: +15-20%)
- ✅ Combat Agent **conserva más HP** (esperado: +20-30%)
- ✅ Diferencias son **estadísticamente significativas** (p < 0.05)

---

### Día 5: Análisis Cualitativo y Reporte 📝

**Tareas:**
1. Ver agente jugando interactivamente:
   ```bash
   python demo_interactive.py --model sessions/combat_v1/combat_agent_final --episodes 5
   ```

2. Analizar comportamientos específicos:
   - ¿Usa curación apropiadamente (solo cuando HP < 50%)?
   - ¿Cambia Pokémon cuando tiene desventaja de tipo?
   - ¿Evita combates innecesarios cuando está débil?

3. Crear reporte final con:
   - **Introducción**: Problema (agente generalista vs especialista)
   - **Metodología**: Arquitectura, recompensas, entrenamiento
   - **Resultados**: Tablas comparativas, gráficos
   - **Análisis**: Por qué funciona mejor (recompensas enfocadas)
   - **Conclusiones**: Combat-specialized PPO > Baseline PPO
   - **Trabajo Futuro**: Agentes para puzzles, exploración, etc.

**Criterio de Éxito:**
- ✅ Tienes evidencia visual de que el agente juega inteligentemente
- ✅ Reporte con datos cuantitativos (tablas, p-values)
- ✅ Reporte con datos cualitativos (videos, observaciones)

---

## 📋 Checklist de Entregables

### Código
- [ ] `PokemonCombatAgent/` - Proyecto completo
- [ ] `sessions/combat_v1/combat_agent_final.zip` - Modelo entrenado
- [ ] `comparison_results/` - Resultados de evaluación

### Datos
- [ ] `combat_agent_metrics.csv` - Métricas del combat agent (100 episodios)
- [ ] `baseline_agent_metrics.csv` - Métricas del baseline (100 episodios)
- [ ] `comparison_results.csv` - Comparación estadística

### Visualizaciones
- [ ] Gráfico: Win Rate (Combat vs Baseline)
- [ ] Gráfico: HP Conservation (Combat vs Baseline)
- [ ] Gráfico: Evolución de recompensas durante entrenamiento (TensorBoard)
- [ ] Video: Agente jugando (opcional, usar `demo_interactive.py`)

### Documentación
- [ ] README.md - Explicación del proyecto
- [ ] TECHNICAL_ANALYSIS.md - Por qué funciona vs TEL351
- [ ] Reporte Final (PDF) - Con toda la evidencia

---

## 🚨 Problemas Comunes y Soluciones

### Problema 1: "Training very slow"
**Síntoma:** 1000 steps toman >10 minutos

**Solución:**
```bash
# Reducir entornos paralelos y action_freq
python train_combat_agent.py --num-envs 4 --action-freq 12
```

### Problema 2: "Agent not learning (reward stuck)"
**Síntoma:** Reward se queda en 0-10 constantemente

**Diagnóstico:**
```bash
# Ver si está en combates
grep "W/L:" sessions/combat_v1/agent_stats_*.csv.gz
```

**Solución:**
```bash
# Aumentar exploración
python train_combat_agent.py --ent-coef 0.02

# O usar estado inicial diferente (más cerca de combates)
python train_combat_agent.py --init-state ../pewter_gym_entrance.state
```

### Problema 3: "Out of memory"
**Síntoma:** `CUDA out of memory` o `MemoryError`

**Solución:**
```bash
# Reducir batch size y número de entornos
python train_combat_agent.py --num-envs 4 --batch-size 256
```

### Problema 4: "ROM not found"
**Síntoma:** `FileNotFoundError: ../PokemonRed.gb`

**Solución:**
```powershell
# Verificar que ROM está en directorio correcto
Copy-Item "C:\ruta\a\PokemonRed.gb" "..\PokemonRed.gb"
```

---

## 🎯 Metas de Desempeño Esperadas

### Métricas Mínimas Aceptables
- **Win Rate**: >70% (vs ~60% baseline)
- **HP Conservation**: >60% (vs ~40% baseline)
- **Deaths per Episode**: <1.0 (vs ~1.5 baseline)

### Métricas Excelentes
- **Win Rate**: >85%
- **HP Conservation**: >75%
- **Deaths per Episode**: <0.5
- **p-value**: <0.01 (muy significativo)

---

## 📞 Próximos Pasos si Todo Funciona

1. **Publicar en GitHub** (si es proyecto público)
2. **Crear demo video** (YouTube, mostrar agente vs humano/baseline)
3. **Extender a otros tipos**:
   - Puzzle Agent (para resolver laberintos rápido)
   - Explorer Agent (para encontrar objetos raros)
   - Hybrid Agent (combina combat + puzzle + exploration)
4. **Paper académico** (si es para tesis o publicación)

---

## 🔬 Validación Científica

### Hipótesis
H0: Combat Agent tiene mismo desempeño que Baseline PPO
H1: Combat Agent tiene mejor desempeño que Baseline PPO en combates

### Test Estadístico
- **Método**: t-test pareado (100 episodios cada uno)
- **Significancia**: α = 0.05
- **Potencia**: >0.80 (con 100 episodios)

### Criterio de Rechazo
Si p-value < 0.05 en **al menos 3 de las 5 métricas principales** → Rechazamos H0

---

**¡Manos a la obra!** 🚀

Sigue este plan paso a paso y tendrás un proyecto completo, funcional y científicamente validado en menos de una semana.

**Dudas o problemas:** Revisar `TECHNICAL_ANALYSIS.md` para entender por qué ciertas cosas se hicieron de cierta manera.
