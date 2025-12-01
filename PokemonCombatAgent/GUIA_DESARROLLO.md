# Guía de Desarrollo del Combat Agent - Pokemon Red

## 📋 Resumen

Este documento describe los componentes exitosos del proyecto Combat Agent para Pokemon Red, desarrollado con PPO (Proximal Policy Optimization) y PyBoy. El objetivo es entrenar un agente especializado en combates Pokemon que supere al baseline general.

---

## 🎯 Componentes Principales

### 1. Generación de Estados de Batalla Válidos

#### `generate_clean_battle_states.py`
**Propósito:** Generar archivos `.state` con batallas activas para entrenamiento especializado.

**Problema resuelto:** Los estados guardados manualmente o pre-existentes no capturaban el momento exacto del inicio de batalla, causando que los agentes entrenados exploraran en vez de combatir.

**Funcionamiento:**
1. Carga el modelo baseline pre-entrenado
2. Navega automáticamente hasta un gimnasio específico
3. Detecta cuando inicia una batalla (monitor de `battle_type` en memoria `0xD057`)
4. Espera 5 frames para estabilizar el estado de batalla
5. Guarda el `.state` con la batalla activa

**Uso:**
```powershell
python generate_clean_battle_states.py --target-gym pewter --headless
```

**Gimnasios soportados:**
- `pewter` (Brock) - Map ID: 52
- `cerulean` (Misty) - Map ID: 65
- `vermilion` (Lt. Surge) - Map ID: 92

**Salida:**
- Directorio: `generated_battle_states/`
- Archivo: `clean_pewter_gym.state` (o similar según gimnasio)

---

#### `verify_battle_state.py`
**Propósito:** Verificar que un archivo `.state` contiene una batalla activa.

**Información que muestra:**
- ✅ Battle Type (0=none, 1=wild, 2=trainer)
- 📍 Map ID y posición (x, y)
- 💚 HP del jugador
- 💔 HP del enemigo
- 🏛️ Identificación de gimnasio

**Uso:**
```powershell
python verify_battle_state.py generated_battle_states\clean_pewter_gym.state
```

**Ejemplo de salida válida:**
```
⚔️  Estado de Batalla:
   Battle Type: 1 (Wild Pokemon)

📊 Diagnóstico:
   ✅ Estado VÁLIDO - Batalla en progreso
```

---

### 2. Entrenamiento Especializado en Combate

#### `train_battle_loop.py`
**Propósito:** Entrenar un agente PPO especializado en combate mediante loop de batallas repetidas.

**Sistema de recompensas optimizado:**
- ✅ Daño causado: **+3.0 por HP**
- ❌ Daño recibido: **-2.0 por HP**
- 🏆 Victoria: **+1000**
- 💎 Victoria perfecta (sin daño): **+1300**
- 🏃 Huir con >50% HP: **-500** (penalización fuerte)
- ☠️ Derrota: **-300**

**Características clave:**
- Loop infinito de la misma batalla (máximo aprendizaje por repetición)
- Reinicio automático tras cada batalla (victoria/derrota/huida)
- Checkpoints cada 50,000 timesteps
- Compatible con modelos pre-entrenados

**Uso:**
```powershell
python train_battle_loop.py \
  --model sessions\combat_agent_final\combat_agent_final.zip \
  --battle-state generated_battle_states\clean_pewter_gym.state \
  --timesteps 500000 \
  --learning-rate 0.0003
```

**Parámetros:**
- `--model`: Modelo base a continuar entrenando
- `--battle-state`: Estado de batalla válido (.state)
- `--timesteps`: Cantidad de pasos de entrenamiento
- `--learning-rate`: Tasa de aprendizaje (default: 0.0003)

**Salida:**
- Directorio: `sessions/{modelo}_battle_loop/`
- Modelo final: `{modelo}_battle_loop.zip`
- Checkpoints: `sessions/{modelo}_battle_loop/checkpoints/`

---

#### `red_gym_env_v2.py`
**Propósito:** Ambiente base de Gymnasium para Pokemon Red con PyBoy 2.6+.

**Configuración crítica para evitar crashes:**
```python
pyboy = PyBoy(
    rom_path,
    window='headless',  # NO usar 'null' (crashes)
    sound=False,
    cgb=False,          # Deshabilitar Game Boy Color
    sound_emulated=False
)
```

**Características:**
- Compatible con observaciones Dict (MultiInputPolicy)
- Carga automática de estados iniciales en `reset()`
- Lectura de memoria del juego para métricas
- Sin logging excesivo (optimizado para entrenamiento largo)

---

### 3. Estructura de Directorios

```
PokemonCombatAgent/
├── generated_battle_states/          # Estados de batalla válidos
│   ├── clean_pewter_gym.state        # ✅ Batalla activa (Pewter Gym)
│   └── manual_save_pewter.state      # Respaldo manual
│
├── sessions/                         # Modelos entrenados
│   ├── combat_agent_final/           # Entrenamiento base (1.5M steps)
│   │   ├── combat_agent_final.zip    # Modelo principal
│   │   ├── checkpoints/              # Checkpoints cada 50K
│   │   └── PPO_1/ ... PPO_6/         # Logs TensorBoard
│   │
│   └── combat_agent_final_battle_loop/  # Entrenamiento especializado
│       ├── combat_agent_final_battle_loop.zip
│       ├── checkpoints/
│       └── PPO_1/                    # Logs TensorBoard
│
└── comparison_results/               # Resultados de comparaciones
    └── analysis_YYYYMMDD_HHMMSS/
        ├── COMPARISON_REPORT.md
        └── *.png (gráficos)
```

---

### 4. Comparación y Análisis

#### `compare_models_interactive.py`
**Propósito:** Comparar rendimiento de dos modelos en el mismo escenario de batalla.

**Características:**
- Ejecución secuencial (evita conflictos SDL2)
- Métricas de combate: HP dealt/taken, victorias, recompensas
- Resultados guardados en JSON para análisis posterior

**Uso:**
```powershell
python compare_models_interactive.py \
  --combat-model sessions\combat_agent_final_battle_loop\combat_agent_final_battle_loop.zip \
  --baseline-model ..\PokemonRedExperiments\v2\runs\poke_26214400.zip \
  --battle-state generated_battle_states\clean_pewter_gym.state \
  --episodes 10 \
  --max-steps 2000
```

**Salida:**
- Directorio: `comparison_results/`
- JSON: `comparison_YYYYMMDD_HHMMSS.json`

---

#### `analyze_comparison.py`
**Propósito:** Generar visualizaciones y reporte de comparación entre modelos.

**Gráficos generados:**
1. Métricas de combate (HP dealt/taken, victorias)
2. Comparación por episodio
3. Fórmulas de recompensa
4. Engagement en batalla

**Uso:**
```powershell
python analyze_comparison.py
# Automáticamente busca el JSON más reciente
```

**Salida:**
- Directorio: `comparison_results/analysis_YYYYMMDD_HHMMSS/`
- 4 gráficos PNG
- `COMPARISON_REPORT.md` con conclusiones

---

#### `analyze_training_metrics.py`
**Propósito:** Extraer y visualizar métricas de entrenamiento desde logs TensorBoard.

**Métricas principales:**
- `rollout/ep_rew_mean`: Recompensa promedio por episodio
- `rollout/ep_len_mean`: Longitud promedio de episodio
- `train/explained_variance`: Qué tan bien predice el modelo
- `train/approx_kl`: Divergencia KL (estabilidad)
- `train/value_loss`: Pérdida de la función de valor
- `train/policy_gradient_loss`: Pérdida del gradiente de política

**Uso:**
```powershell
python analyze_training_metrics.py \
  --session-dir sessions\combat_agent_final_battle_loop \
  --output-dir training_analysis
```

**Salida:**
- Directorio: `training_analysis/`
- Gráficos PNG individuales por métrica
- `training_summary.png` (resumen de 4 métricas clave)
- `metrics.csv` (datos exportables)
- `summary.json` (resumen estadístico)

**Alternativa - TensorBoard en tiempo real:**
```powershell
tensorboard --logdir=sessions\combat_agent_final_battle_loop
# Abrir http://localhost:6006
```

---

## 🔄 Flujo de Trabajo Completo

### Paso 1: Generar Estado de Batalla Válido
```powershell
# Generar estado de batalla
python generate_clean_battle_states.py --target-gym pewter --headless

# Verificar que sea válido
python verify_battle_state.py generated_battle_states\clean_pewter_gym.state
```

**Resultado esperado:** `✅ Estado VÁLIDO - Batalla en progreso`

---

### Paso 2: Entrenar Agente Especializado
```powershell
# Entrenar desde modelo base
python train_battle_loop.py \
  --model sessions\combat_agent_final\combat_agent_final.zip \
  --battle-state generated_battle_states\clean_pewter_gym.state \
  --timesteps 500000
```

**Duración estimada:** ~1-2 horas con GPU (RTX 3050)  
**Velocidad:** ~100-104 it/s

---

### Paso 3: Comparar con Baseline
```powershell
# Comparar modelos
python compare_models_interactive.py \
  --combat-model sessions\combat_agent_final_battle_loop\combat_agent_final_battle_loop.zip \
  --baseline-model ..\PokemonRedExperiments\v2\runs\poke_26214400.zip \
  --battle-state generated_battle_states\clean_pewter_gym.state \
  --episodes 10 \
  --max-steps 2000

# Generar análisis
python analyze_comparison.py
```

**Salida:** Reporte markdown con gráficos comparativos

---

### Paso 4: Analizar Métricas de Entrenamiento
```powershell
# Extraer métricas
python analyze_training_metrics.py \
  --session-dir sessions\combat_agent_final_battle_loop

# O ver en tiempo real
tensorboard --logdir=sessions\combat_agent_final_battle_loop
```

---

## 📊 Métricas Clave para Evaluar Éxito

### Durante el Entrenamiento
- **`explained_variance`**: Debe estar > 0.9 (modelo predice bien)
- **`approx_kl`**: Debe estar < 0.05 (entrenamiento estable)
- **`ep_rew_mean`**: Debe aumentar progresivamente
- **`fps`**: ~100 it/s con GPU (buena velocidad)

### En Comparación
- **Win Rate**: % de batallas ganadas
- **Avg HP Dealt**: Daño promedio causado por episodio
- **Avg HP Taken**: Daño promedio recibido por episodio
- **Avg Reward**: Recompensa total promedio

**Objetivo:** Combat Agent > Baseline en todas las métricas

---

## 🛠️ Configuración GPU (NVIDIA)

```powershell
# Verificar PyTorch con CUDA
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
# Debe imprimir: CUDA: True

# Verificar dispositivo durante entrenamiento
# Los logs mostrarán: "Using cuda device"
```

**Hardware probado:**
- GPU: NVIDIA RTX 3050
- CUDA: 11.8
- PyTorch: 2.7.1+cu118

---

## 📝 Archivos de Configuración Importantes

### `requirements.txt`
Dependencias principales:
- `stable-baselines3[extra]`: Algoritmo PPO
- `pyboy`: Emulador Game Boy
- `gymnasium`: API de ambientes RL
- `torch`: Backend para redes neuronales
- `tensorboard`: Logging de métricas

### `PokemonRed.gb`
ROM de Pokemon Red (necesario en raíz del proyecto)

---

## 🚨 Problemas Comunes Resueltos

### 1. "Estado INVÁLIDO - NO hay batalla activa"
**Causa:** Estado guardado antes/después de la batalla  
**Solución:** Usar `generate_clean_battle_states.py` que detecta automáticamente el inicio exacto

### 2. "Sound buffer overrun! 1602 of 1602"
**Causa:** PyBoy con audio habilitado  
**Solución:** Configurar `window='headless'`, `sound=False`, `cgb=False`

### 3. Combat Agent no entra en batallas durante comparación
**Causa:** Usar estado de exploración (ej: `has_pokedex_nballs.state`)  
**Solución:** Usar estado de batalla válido de `generated_battle_states/`

### 4. Batalla se reinicia muy seguido
**Causa:** Agente termina batalla rápido (victoria/derrota/huida)  
**Explicación:** Esto es NORMAL y esperado. PPO aprende jugando muchas batallas cortas, no una batalla infinita.

---

## 🎯 Resultados Esperados

### Entrenamiento Exitoso
```
rollout/ep_rew_mean: +500 a +1000 (victorias frecuentes)
train/explained_variance: 0.90 - 0.99
train/approx_kl: 0.02 - 0.04
fps: 90-110 it/s (con GPU)
```

### Comparación Exitosa
```
Combat Agent Win Rate: 60-80%
Baseline Win Rate: 30-50%
Combat Agent Avg HP Dealt: Mayor que Baseline
Combat Agent Avg HP Taken: Menor que Baseline
```

---

## 📚 Referencias de Memoria del Juego

**Direcciones críticas (red_gym_env_v2.py):**
```python
0xD057  # Battle type (0=none, 1=wild, 2=trainer)
0xD16C  # Player HP (2 bytes, big endian)
0xCFE6  # Enemy HP (2 bytes, big endian)
0xD35E  # Current map ID
0xD362  # X position
0xD361  # Y position
0xD356  # Badges count
```

**Map IDs de gimnasios:**
```python
52  # Pewter Gym (Brock)
65  # Cerulean Gym (Misty)
92  # Vermilion Gym (Lt. Surge)
176 # Celadon Gym (Erika)
177 # Fuchsia Gym (Koga)
178 # Saffron Gym (Sabrina)
180 # Cinnabar Gym (Blaine)
181 # Viridian Gym (Giovanni)
```

---

## 🔬 Próximos Pasos Sugeridos

1. **Entrenar en múltiples gimnasios:**
   ```powershell
   python generate_clean_battle_states.py --target-gym cerulean
   python generate_clean_battle_states.py --target-gym vermilion
   ```

2. **Aumentar timesteps:**
   ```powershell
   python train_battle_loop.py --timesteps 1000000  # 1M steps
   ```

3. **Ajustar recompensas:**
   - Editar `BattleLoopEnv` en `train_battle_loop.py`
   - Experimentar con valores de daño/victoria

4. **Comparación multi-gimnasio:**
   - Generar estados de varios gimnasios
   - Comparar rendimiento en diferentes escenarios

---

## ✅ Checklist de Replicación

- [ ] Instalar dependencias: `pip install -r requirements.txt`
- [ ] Verificar GPU: `python -c "import torch; print(torch.cuda.is_available())"`
- [ ] Copiar `PokemonRed.gb` al directorio raíz
- [ ] Generar estado de batalla: `python generate_clean_battle_states.py --target-gym pewter --headless`
- [ ] Verificar estado: `python verify_battle_state.py generated_battle_states\clean_pewter_gym.state`
- [ ] Entrenar agente: `python train_battle_loop.py --model <modelo_base> --timesteps 500000`
- [ ] Comparar con baseline: `python compare_models_interactive.py ...`
- [ ] Analizar resultados: `python analyze_comparison.py`
- [ ] Revisar métricas: `python analyze_training_metrics.py --session-dir sessions\<nombre_sesion>`

---

## 📄 Archivos del Proyecto (Solo Exitosos)

### Scripts Principales
- ✅ `generate_clean_battle_states.py` - Generador de estados válidos
- ✅ `verify_battle_state.py` - Verificador de estados
- ✅ `train_battle_loop.py` - Entrenamiento especializado
- ✅ `compare_models_interactive.py` - Comparación de modelos
- ✅ `analyze_comparison.py` - Análisis con visualizaciones
- ✅ `analyze_training_metrics.py` - Extracción de métricas

### Archivos de Ambiente
- ✅ `red_gym_env_v2.py` - Ambiente base Gymnasium
- ✅ `requirements.txt` - Dependencias

### Archivos de Datos
- ✅ `generated_battle_states/clean_pewter_gym.state` - Estado de batalla válido
- ✅ `sessions/combat_agent_final/combat_agent_final.zip` - Modelo base entrenado
- ✅ `PokemonRed.gb` - ROM del juego

### Archivos Descartados (No usar)
- ❌ `battle_states/*.state` - Estados corruptos/bugueados
- ❌ `has_pokedex_nballs.state` - Estado de exploración (no combate)
- ❌ `train_battle_specialist_emergency.py` - Enfoque anterior no exitoso
- ❌ `combat_focused_env.py` - Reemplazado por train_battle_loop.py

---

**Última actualización:** 30 de Noviembre, 2025  
**Versión:** 1.0  
**Estado:** Funcional y probado con éxito
