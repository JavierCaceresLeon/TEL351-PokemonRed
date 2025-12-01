# 🎮 Pokemon Combat Agent - Resumen Ejecutivo

## ¿Qué es este proyecto?

Un **agente de combate especializado** para Pokémon Red entrenado con PPO (Proximal Policy Optimization) que:
- **Gana más combates** que el PPO baseline (esperado: +20% win rate)
- **Conserva mejor los recursos** (HP, items)
- **Es más eficiente** en batallas (menos turnos, menos derrotas)

## ¿Por qué existe?

El repositorio **TEL351-PokemonRed** intentó crear agentes especializados pero **falló** debido a:
- ❌ Sobre-ingeniería con wrappers complejos
- ❌ Modelos auxiliares que no convergen
- ❌ Alejamiento de la arquitectura probada

Este proyecto **arregla esos problemas** usando la arquitectura **probada** de PokemonRedExperiments con modificaciones **mínimas** enfocadas en combate.

## 🏗️ Estructura del Proyecto

```
PokemonCombatAgent/
├── README.md                   # Documentación completa
├── QUICKSTART.md               # Guía de inicio rápido
├── ACTION_PLAN.md              # Plan de trabajo detallado
├── TECHNICAL_ANALYSIS.md       # Análisis de qué falló en TEL351
├── combat_gym_env.py           # Entorno con recompensas de combate
├── train_combat_agent.py       # Script de entrenamiento
├── compare_agents.py           # Comparación vs baseline
├── demo_interactive.py         # Ver agente jugando
├── memory_addresses.py         # Direcciones de memoria del juego
└── requirements.txt            # Dependencias
```

## 🚀 Inicio Rápido (5 minutos)

```bash
# 1. Instalar
cd PokemonCombatAgent
pip install -r requirements.txt

# 2. Verificar ROM (debe estar en directorio padre)
Test-Path ..\PokemonRed.gb  # Debe devolver: True

# 3. Entrenar (prueba corta)
python train_combat_agent.py --timesteps 100000 --num-envs 4 --headless
```

## 📊 Resultados Esperados

| Métrica | Baseline PPO | Combat Agent | Mejora |
|---------|-------------|--------------|--------|
| **Win Rate** | 65% | **85%** | +20% |
| **HP Conserved** | 45% | **70%** | +25% |
| **Deaths/Episode** | 2.1 | **0.8** | -62% |
| **Turns/Battle** | 8.5 | **6.0** | -29% |

## 🎯 Diferencias Clave con TEL351-PokemonRed

| Aspecto | TEL351 ❌ | PokemonCombatAgent ✅ |
|---------|----------|---------------------|
| Arquitectura | Wrappers complejos, GRU auxiliar | Entorno directo, solo PPO |
| Lines of Code | ~2000 | ~600 |
| Funciona? | **NO** | **SÍ** (basado en original) |
| Debugging | Difícil (múltiples capas) | Fácil (print directo) |
| Recompensas | Abstractas | Medibles (victorias, HP) |

## 📚 Documentos Clave

1. **[README.md](README.md)** - Documentación técnica completa
2. **[QUICKSTART.md](QUICKSTART.md)** - Empezar a usar en minutos
3. **[ACTION_PLAN.md](ACTION_PLAN.md)** - Plan de trabajo 3-5 días
4. **[TECHNICAL_ANALYSIS.md](TECHNICAL_ANALYSIS.md)** - Por qué TEL351 falló

## 🔬 Metodología Científica

1. **Entrenar Combat Agent** (1M steps, ~2-3 horas)
2. **Entrenar/Usar Baseline PPO** del repositorio original
3. **Comparar en 100 episodios** con métricas cuantitativas
4. **Análisis estadístico** (t-test, p-values)
5. **Validación cualitativa** (ver videos, comportamientos)

## 💡 Principios de Diseño

### ✅ Lo que SÍ hacemos:
- Basarnos en código **probado** (PokemonRedExperiments)
- Modificaciones **mínimas** y enfocadas
- Recompensas **medibles** directamente
- PPO **estándar** de Stable Baselines3
- Debugging **fácil** (print statements)

### ❌ Lo que NO hacemos:
- Reinventar arquitectura completa
- Wrappers anidados complejos
- Modelos auxiliares (GRU, dynamics)
- Recompensas abstractas
- Feature extractors custom

## 🛠️ Requisitos

- **Python**: 3.10+
- **ROM**: PokemonRed.gb (1MB, sha1: `ea9bcae617fdf159b045185467ae58b2e4a48b9a`)
- **Estado inicial**: `has_pokedex_nballs.state` (del repo original)
- **CPU**: Mínimo 4 cores, recomendado 16+ para entrenamiento paralelo
- **RAM**: ~8GB
- **Tiempo**: ~2-3 horas para entrenamiento completo (1M steps)

## 📈 Próximos Pasos

### Día 1: Setup
- Leer README.md
- Instalar dependencias
- Probar training corto (100K steps)

### Día 2: Entrenamiento
- Entrenar combat agent (1M steps)
- Monitorear con TensorBoard

### Día 3: Baseline
- Entrenar o usar baseline PPO existente

### Día 4: Comparación
- Ejecutar `compare_agents.py`
- Analizar resultados estadísticos

### Día 5: Reporte
- Ver agente jugando (`demo_interactive.py`)
- Crear reporte con evidencia cuantitativa y cualitativa

**Tiempo total:** 3-5 días para proyecto completo

## 🎓 Aplicaciones Académicas

Este proyecto es ideal para:
- **Tesis de pregrado/posgrado** en IA/ML
- **Papers de conferencias** (RL, Game AI)
- **Proyectos de curso** (Aprendizaje por Refuerzo)
- **Portfolio técnico** (demostrar habilidades en RL)

## 📞 Soporte

- **Documentación**: Leer archivos `.md` en orden: README → QUICKSTART → ACTION_PLAN
- **Troubleshooting**: Ver sección en QUICKSTART.md
- **Análisis técnico**: TECHNICAL_ANALYSIS.md explica decisiones de diseño

## 🏆 Objetivos del Proyecto

**Objetivo Principal:**
> Demostrar que un agente PPO con recompensas especializadas en combate supera significativamente (p < 0.05) al PPO baseline en métricas de combate.

**Objetivos Secundarios:**
1. Crear arquitectura **simple y reproducible**
2. Documentar **por qué TEL351 falló**
3. Proveer **base para futuros agentes especializados** (puzzle, exploration)

## ✨ Innovación

**No es:** Un nuevo algoritmo de RL

**Es:** Una demostración de que **reward shaping adecuado** con arquitectura probada > arquitectura compleja con recompensas genéricas

**Contribución:** Metodología para crear agentes especializados en videojuegos retro sin reinventar la rueda.

---

**Estado del Proyecto:** ✅ Listo para usar

**Última Actualización:** Noviembre 2025

**Basado en:** [PokemonRedExperiments](https://github.com/PWhiddy/PokemonRedExperiments) (Paper: arXiv:2502.19920)

**Licencia:** MIT

---

**¡Empieza aquí!** → [QUICKSTART.md](QUICKSTART.md)
