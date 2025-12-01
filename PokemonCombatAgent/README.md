# Agente Especialista en Combate para Pokémon Red

## 🎯 Objetivo del Proyecto

Entrenar un agente PPO especializado en combates Pokémon que:
- **Maximice victorias** conservando recursos (HP, PP, items)
- **Sea más inteligente** que el PPO básico del repositorio original
- **Use heurísticas de combate** basadas en tipos, niveles y estado actual
- **Permita comparación científica** con métricas claras vs PPO baseline

## 📊 Diferencias Clave con Proyectos Anteriores

### ❌ Lo que NO funcionó en TEL351-PokemonRed:
- Wrappers complejos y anidados que dificultan el debugging
- Modelos auxiliares (GRU, dynamics) que no convergen
- Espacios de observación mal dimensionados
- Recompensas abstractas sin fundamento empírico
- Falta de estados iniciales adecuados para combates

### ✅ Lo que SÍ funciona (basado en PokemonRedExperiments):
- PPO simple y robusto con CnnPolicy
- Entorno directo heredando de `gymnasium.Env`
- Recompensas numéricas claras y medibles
- Estados iniciales (.state files) que realmente funcionan
- Configuración probada y documentada

### 🚀 Nuestra Innovación:
Tomamos la arquitectura **probada** y la modificamos mínimamente con:
- **Sistema de recompensas enfocado en combate**
- **Métricas de combate** (victorias, HP conservado, daño eficiente)
- **Heurísticas inteligentes** (ventajas de tipo, switching óptimo)
- **Comparación científica** con agente baseline

## 🏗️ Arquitectura del Sistema

```
PokemonCombatAgent/
├── memory_addresses.py          # Direcciones de memoria (copiado del original)
├── combat_gym_env.py            # Entorno GYM con recompensas de combate
├── combat_metrics.py            # Sistema de tracking de métricas
├── train_combat_agent.py        # Script de entrenamiento PPO
├── compare_agents.py            # Script de comparación vs baseline
├── requirements.txt             # Dependencias (idénticas al original)
└── README.md                    # Este archivo
```

## 🧠 Sistema de Recompensas de Combate

### Componentes de la Función de Recompensa

La función de recompensa está diseñada para optimizar comportamiento en combate:

```python
R_total = R_victory + R_hp_efficiency + R_damage_dealt + R_type_advantage - R_penalties
```

#### 1. **Recompensa por Victoria** (Victoria sin derrotas)
- Victoria sin pokemon derrotados: **+200**
- Victoria con 1 pokemon derrotado: **+150**
- Victoria con 2+ pokemon derrotados: **+100**
- Derrota (HP=0): **-100**

#### 2. **Eficiencia de HP** (Conservar recursos)
```python
R_hp_efficiency = 50 * (HP_after_battle / HP_before_battle)
```
- Ganar sin recibir daño: **+50**
- Ganar con 50% HP restante: **+25**
- Ganar con 10% HP: **+5**

#### 3. **Daño Efectivo** (Inteligencia de combate)
```python
R_damage = 10 * (Damage_dealt / Enemy_HP_max)
```
- Un golpe efectivo que quita 50% HP: **+5**
- Acumular damage incrementa recompensa progresivamente

#### 4. **Ventaja de Tipo** (Uso de estrategia)
```python
R_type = +20 si usamos movimiento super efectivo
R_type = -10 si usamos movimiento no muy efectivo
```

#### 5. **Penalizaciones** (Evitar comportamientos malos)
- Usar poción cuando HP > 80%: **-15**
- Quedarse en menú > 10 steps: **-5 por step**
- Atacar con pokemon debilitado cuando hay mejor opción: **-10**

### Comparación Numérica

| Escenario | PPO Baseline | Combat Agent | Ganancia |
|-----------|-------------|--------------|----------|
| Victoria rápida (3 turnos, sin daño) | +100 | +200 + 50 + 30 = **+280** | **+180%** |
| Victoria larga (10 turnos, 50% HP) | +100 | +150 + 25 + 50 = **+225** | **+125%** |
| Victoria con 1 pokemon derrotado | +100 | +150 - 100 = **+50** | **-50%** |
| Derrota | -50 | -100 | **-100%** |

## 📈 Métricas de Evaluación

Medimos el desempeño con las siguientes métricas:

### Métricas Primarias
1. **Win Rate**: % de combates ganados
2. **HP Conservation**: Promedio de HP% al final de combates ganados
3. **Turns per Battle**: Promedio de turnos para ganar
4. **Deaths per Episode**: Pokémon derrotados por episodio

### Métricas Secundarias  
5. **Type Advantage Usage**: % de veces que usa movimiento super efectivo
6. **Potion Efficiency**: Ratio de pociones usadas / HP curado
7. **Damage Efficiency**: Daño promedio por turno
8. **Switching Intelligence**: % de switches apropiados

## 🚀 Guía de Uso Rápida

### Paso 1: Instalación

```bash
cd PokemonCombatAgent
pip install -r requirements.txt
```

### Paso 2: Colocar ROM
Copiar `PokemonRed.gb` al directorio base (mismo que /PokemonRedExperiments)

### Paso 3: Crear Estado Inicial de Combate

```python
# Opción A: Usar estado existente cerca de combate
# has_pokedex_nballs.state es bueno para empezar

# Opción B: Crear tu propio estado
# 1. Juega manualmente hasta antes de un combate importante
# 2. Guarda el estado desde PyBoy
```

### Paso 4: Entrenar Agente de Combate

```bash
python train_combat_agent.py --timesteps 1000000 --num-envs 16
```

Parámetros clave:
- `--timesteps`: Total de timesteps de entrenamiento (default: 1M)
- `--num-envs`: Entornos paralelos (default: 16, ajustar según CPU)
- `--checkpoint-freq`: Guardar cada N steps (default: 50000)
- `--headless`: Modo sin GUI (default: True para entrenamiento)

### Paso 5: Evaluar y Comparar

```bash
# Evaluar solo el agente de combate
python compare_agents.py --combat-agent sessions/combat_agent_final.zip --episodes 100

# Comparar con PPO baseline
python compare_agents.py \
    --combat-agent sessions/combat_agent_final.zip \
    --baseline-agent sessions/baseline_ppo.zip \
    --episodes 100 \
    --output-csv results_comparison.csv
```

## 📊 Resultados Esperados

### Hipótesis
El agente especializado en combate debería superar al baseline en:

| Métrica | Baseline Esperado | Combat Agent Esperado | Mejora |
|---------|-------------------|----------------------|--------|
| Win Rate | 65% | **85%** | +20% |
| HP Conservation | 45% | **70%** | +25% |
| Turns/Battle | 8.5 | **6.0** | -29% |
| Deaths/Episode | 2.1 | **0.8** | -62% |

### Validación Estadística
Usamos **t-test pareado** con 100 episodios para determinar significancia estadística (p < 0.05).

## 🔬 Fundamentos Técnicos

### Por qué PPO y no otros algoritmos

**PPO (Proximal Policy Optimization)** es ideal para este problema porque:

1. **Estabilidad**: Evita cambios drásticos en la política (clipping)
2. **Eficiencia de muestra**: Reutiliza experiencia mediante múltiples epochs
3. **Paralelización**: Entrena en múltiples entornos simultáneamente
4. **Robustez**: Funciona bien con recompensas sparse y episódicas

### Configuración de Hiperparámetros

Basada en la configuración probada de PokemonRedExperiments:

```python
model = PPO(
    "CnnPolicy",           # Red CNN para procesar frames
    env,
    n_steps=2048,          # Rollout buffer (episodios típicos ~500 steps)
    batch_size=512,        # Balance entre velocidad y estabilidad
    n_epochs=1,            # Evita overfitting en datos de episodio
    gamma=0.99,            # Factor de descuento (valora futuro cercano)
    learning_rate=3e-4,    # LR estándar para PPO
    ent_coef=0.01,         # Coeficiente de entropía (exploración)
    clip_range=0.2,        # Rango de clipping PPO
    verbose=1
)
```

**¿Por qué estos valores?**

- `n_steps=2048`: Combates duran ~200-500 steps, queremos múltiples combates por update
- `n_epochs=1`: Dataset pequeño, evitamos overfitting
- `gamma=0.99`: Recompensa a 100 steps vale 0.99^100 ≈ 36% del valor actual (horizonte ~100)
- `ent_coef=0.01`: Suficiente exploración sin sacrificar convergencia

## 🛠️ Troubleshooting

### Error: "ROM not found"
```bash
# Verificar que PokemonRed.gb esté en el directorio correcto
ls ../PokemonRed.gb
# Debe devolver: ../PokemonRed.gb
```

### Error: "PyBoy rendering issues"
```bash
# Para entrenamiento, usar headless=True
python train_combat_agent.py --headless
```

### Entrenamiento muy lento
```bash
# Reducir número de entornos paralelos
python train_combat_agent.py --num-envs 8  # en lugar de 16

# O reducir action_freq
# Editar config: 'action_freq': 12  # en lugar de 24
```

### Agente no aprende (reward estancado)
1. Verificar que el estado inicial esté cerca de combates
2. Revisar que las recompensas sean escalas apropiadas (0-300 rango)
3. Aumentar `ent_coef` para más exploración
4. Verificar logs de TensorBoard

## 📚 Referencias

### Repositorios Base
- **PokemonRedExperiments**: https://github.com/PWhiddy/PokemonRedExperiments
- **Paper**: "Pokemon Red via Reinforcement Learning" (arXiv:2502.19920)

### Herramientas
- **PyBoy**: https://github.com/Baekalfen/PyBoy
- **Stable Baselines3**: https://github.com/DLR-RM/stable-baselines3
- **Gymnasium**: https://gymnasium.farama.org/

### Mecánicas de Pokémon Gen 1
- **Damage Formula**: https://bulbapedia.bulbagarden.net/wiki/Damage
- **Type Chart**: https://pokemondb.net/type
- **RAM Map**: https://datacrystal.romhacking.net/wiki/Pokémon_Red/Blue:RAM_map

## 🤝 Contribución

Este proyecto es para fines educativos (curso TEL351). 

**Autores**: Basado en el trabajo de PokemonRedExperiments, adaptado para agente especializado en combate.

**Licencia**: MIT (mismo que proyecto original)

---

**¡Buena suerte entrenando tu agente especialista en combate!** 🎮🔥
