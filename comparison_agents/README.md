# Pokemon Red Agent Comparison: PPO vs Epsilon Greedy

**Sistema de Comparación de Agentes para Pokemon Red con Algoritmo Epsilon Greedy Avanzado**

Este proyecto implementa un agente de búsqueda con algoritmo Epsilon Greedy y heurísticas avanzadas para el entorno Pokemon Red v2, proporcionando una comparación comprensiva con el agente PPO preentrenado.

## Novedades y Actualizaciones (Septiembre 2025)

### **Nuevas Implementaciones**
- **Agente Epsilon Greedy Completo** con 5 escenarios de detección automática
- **Sistema de Heurísticas Adaptativas** con 6 funciones especializadas
- **Comparación Automatizada** PPO vs Epsilon Greedy con métricas avanzadas
- **Análisis Estadístico Completo** con visualizaciones y reportes
- **Compatibilidad con Python 3.10** y ambiente conda especializado

### **Correcciones de Compatibilidad**
- **PyBoy API Fix**: Actualización de `botsupport_manager()` → acceso directo
- **Memory API Fix**: `get_memory_value()` → `pyboy.memory[]`
- **Screen API Fix**: `screen_ndarray()` → `screen.ndarray`
- **Dependencias Actualizadas**: PyBoy 2.4.0, scikit-image, websockets

## Requisitos del Sistema

### **Requisitos Críticos**
- **Python 3.10.x** (OBLIGATORIO - PyBoy 2.4.0 no es compatible con 3.11+)
- **PyBoy 2.4.0** (Versión específica para compatibilidad con v2)
- **Archivos del Proyecto v2** correctamente configurados
- **4GB+ RAM** para ejecución estable

### **Dependencias Principales**
```
PyBoy==2.4.0               # Emulador Game Boy (versión específica)
stable-baselines3>=2.7.0   # Algoritmos de RL
gymnasium>=1.2.0           # Environment interface
torch>=2.3.0               # PyTorch para PPO
scikit-image>=0.25.0       # Procesamiento de imágenes
websockets>=15.0.0         # Para stream_agent_wrapper
numpy>=2.1.0               # Cálculos numéricos
pandas>=2.2.0              # Análisis de datos
matplotlib>=3.9.0          # Visualizaciones
```

## Instalación Completa

### **Opción 1: Ambiente Conda (Recomendado)**

```bash
# 1. Crear ambiente conda específico
conda create -n pokemon-red-comparison python=3.10.18 -y
conda activate pokemon-red-comparison

# 2. Instalar dependencias científicas básicas
conda install numpy pandas matplotlib scipy scikit-learn seaborn plotly jupyter ipython tqdm -c conda-forge -y

# 3. Instalar PyTorch CPU
conda install pytorch cpuonly -c pytorch -y

# 4. Instalar librerías de RL y específicas
pip install gymnasium stable-baselines3
pip install pyboy==2.4.0
pip install scikit-image websockets
pip install mediapy einops

# 5. Navegar al directorio del proyecto
cd C:/ruta/a/tu/proyecto/TEL351-PokemonRed/comparison_agents

# 6. Copiar archivos necesarios del v2
copy ..\v2\events.json .
copy ..\v2\map_data.json .

# 7. Verificar instalación
python verify_environment.py
```

### **Opción 2: Ambiente Virtual Python**

```bash
# 1. Crear ambiente virtual
python -m venv pokemon_comparison_env
pokemon_comparison_env\Scripts\activate  # Windows
# source pokemon_comparison_env/bin/activate  # Linux/Mac

# 2. Instalar dependencias
pip install -r requirements_py310.txt

# 3. Verificar instalación
python verify_environment.py
```

### **Errores Comunes de Instalación**

#### **Error: PyBoy API Incompatibilidad**
```
AttributeError: 'pyboy.pyboy.PyBoy' object has no attribute 'botsupport_manager'
```
**Solución**: 
- Verificar PyBoy 2.4.0: `pip show pyboy`
- Si es diferente: `pip install pyboy==2.4.0`

#### **Error: CUDA Dependencies**
```
nvidia-nccl-cu12==2.21.5 requires Python >=3.11
```
**Solución**: 
- Usar CPU only: `conda install pytorch cpuonly -c pytorch`
- Crear ambiente conda con Python 3.10

#### **Error: Archivos v2 Faltantes**
```
FileNotFoundError: [Errno 2] No such file or directory: 'events.json'
```
**Solución**:
```bash
copy ..\v2\events.json .
copy ..\v2\map_data.json .
```

## Estructura del Proyecto

```
comparison_agents/
├── CORE COMPONENTS
│   ├── epsilon_greedy_agent.py      # Algoritmo Epsilon Greedy con heurísticas
│   ├── v2_agent.py                  # Wrapper para integración con ambiente v2
│   ├── agent_comparison.py          # Sistema de comparación automática
│   ├── metrics_analyzer.py          # Análisis estadístico avanzado
│   └── run_comparison.py            # Script principal de ejecución
│
├── CONFIGURATION
│   ├── config.py                    # Configuraciones centralizadas
│   ├── requirements_py310.txt       # Dependencias Python 3.10 compatible
│   ├── environment.yml              # Configuración conda
│   └── SETUP_INSTRUCTIONS.md        # Guía detallada de instalación
│
├── SETUP & TESTING
│   ├── verify_environment.py        # Verificación completa del ambiente
│   ├── test_pyboy_api.py            # Test de compatibilidad PyBoy
│   ├── example_usage.py             # Ejemplos de uso programático
│   └── install_dependencies.py      # Instalador automatizado
│
├── DATA & RESULTS
│   ├── events.json                  # Configuración de eventos del juego
│   ├── map_data.json               # Datos del mapa de Pokemon Red
│   ├── comparison_results/          # Resultados de comparaciones
│   ├── metrics_analysis/            # Análisis de métricas
│   ├── experiments/                 # Experimentos guardados
│   └── logs/                       # Logs de ejecución
│
└── DOCUMENTATION
    ├── README.md                    # Este archivo (documentación completa)
    └── __pycache__/                # Cache de Python
```

## Uso del Sistema

### **Ejecución Rápida (Recomendado para Inicio)**

```bash
# Activar ambiente
conda activate pokemon-red-comparison

# Prueba rápida (2 episodios)
python run_comparison.py --mode standalone --episodes 2

# Comparación básica
python run_comparison.py --mode comparison --episodes 5

# Análisis completo
python run_comparison.py --mode full --episodes 10
```

## Scripts Disponibles en /comparison_agents

### **1. Visualización Interactiva Individual**

#### `run_epsilon_greedy_interactive.py` - Agente Epsilon Greedy Visual
```bash
python run_epsilon_greedy_interactive.py
```

**¿Qué hace?**
- Ejecuta el agente Epsilon Greedy con ventana visual del Game Boy
- Se detiene automáticamente al obtener el primer Pokémon
- Guarda métricas detalladas en `results/epsilon_greedy_metrics_[timestamp].md`
- Ideal para ver el comportamiento heurístico paso a paso

**Características:**
- Visualización en tiempo real de decisiones del agente
- Detección automática de eventos (elegir Pokémon inicial)
- Evita presionar la tecla START automáticamente
- Métricas de rendimiento, tiempo y recursos

### **2. Comparación Simultánea Visual**

#### `run_dual_interactive.py` - Epsilon Greedy vs PPO (Nuevo)
```bash
python run_dual_interactive.py
```

**¿Qué hace?**
- Ejecuta **simultáneamente** dos ventanas del Game Boy:
  - Ventana 1: Agente Epsilon Greedy (Heurístico)
  - Ventana 2: Agente PPO (Deep Learning)
- Permite comparación visual directa entre ambos algoritmos
- Cada agente funciona independientemente

**Ideal para:**
- Comparar comportamientos en tiempo real
- Demostraciones educativas
- Análisis visual de estrategias diferentes

### **3. Comparación Automatizada**

#### `run_comparison.py` - Análisis Completo
```bash
# Modo básico - Solo Epsilon Greedy
python run_comparison.py --mode standalone --episodes 5

# Modo comparación - Epsilon Greedy vs PPO
python run_comparison.py --mode comparison --episodes 5

# Modo completo - Análisis estadístico detallado
python run_comparison.py --mode full --episodes 10
```

**¿Qué hace?**
- Ejecuta múltiples episodios sin interfaz visual (más rápido)
- Genera reportes estadísticos detallados
- Crea visualizaciones comparativas
- Guarda resultados en `comparison_results/`

**Modos disponibles:**
- `standalone`: Solo agente Epsilon Greedy
- `comparison`: Ambos agentes con comparación
- `full`: Análisis completo con métricas avanzadas

### **4. Scripts Especializados**

#### `agent_comparison.py` - Motor de Comparación
```bash
python -c "from agent_comparison import AgentComparator; print('Comparador disponible')"
```
- Clase principal para comparaciones programáticas
- Usado internamente por `run_comparison.py`

#### `metrics_analyzer.py` - Análisis de Métricas
```bash
python metrics_analyzer.py --input comparison_results/
```
- Análisis estadístico avanzado de resultados existentes
- Genera visualizaciones adicionales

## ¿Qué son los Episodios?

### **Definición de Episodio**
Un **episodio** en Pokemon Red es una sesión completa de juego desde el inicio hasta un punto de finalización específico. Cada episodio representa una "vida" o "intento" del agente.

### **Características de los Episodios:**

**Inicio del Episodio:**
- El agente comienza desde el estado inicial (`init.state`)
- Valores resetteados: salud, posición, inventario, etc.
- Contador de pasos en 0

**Finalización del Episodio:**
- Límite de pasos alcanzado (ej: 40,960 pasos)
- Objetivo completado (ej: obtener primer Pokémon)
- Condición de terminación específica
**Métricas por Episodio:**
- Recompensa total obtenida
- Número de pasos realizados
- Tiempo de ejecución
- Eventos completados (badges, captures, etc.)

### **Diferencias entre Algoritmos:**

**Epsilon Greedy:**
- Cada episodio es independiente
- No aprende entre episodios
- Usa las mismas heurísticas cada vez
- Consistencia en el comportamiento

**PPO (Deep Learning):**
- Aprende de episodios anteriores
- Mejora el rendimiento con la experiencia
- Cada episodio puede ser diferente
- Evolución del comportamiento

### **Ejemplos de Uso:**

```bash
# 1 episodio para prueba rápida
python run_comparison.py --episodes 1

# 5 episodios para análisis básico  
python run_comparison.py --episodes 5

# 20 episodios para análisis estadístico robusto
python run_comparison.py --episodes 20
```

**Recomendaciones:**
- **1-2 episodios**: Pruebas rápidas y debugging
- **5-10 episodios**: Comparaciones básicas
- **20+ episodios**: Análisis estadístico confiable

### **Opciones Avanzadas**

```bash
# Configuración personalizada completa
python run_comparison.py \
    --mode full \
    --episodes 10 \
    --max-steps 50000 \
    --epsilon-start 0.6 \
    --epsilon-decay 0.992 \
    --ppo-model "../v2/runs/poke_model.zip"

# Ejecución sin visualizaciones (más rápido)
python run_comparison.py --mode full --no-viz --episodes 5

# Solo análisis de métricas existentes
python metrics_analyzer.py --input comparison_results/
```

### **Verificación del Sistema**

```bash
# Verificación completa del ambiente
python verify_environment.py

# Test específico de PyBoy
python test_pyboy_api.py

# Ejemplo de uso programático
python example_usage.py
```

## Algoritmo Epsilon Greedy Avanzado

### **Detección Automática de Escenarios**

| Escenario | Descripción | Estrategia |
|-----------|-------------|------------|
| `EXPLORATION` | Exploración general del mapa | Maximiza áreas no visitadas |
| `BATTLE` | Combate con Pokemon/entrenadores | Prioriza acciones de batalla |
| `NAVIGATION` | Navegación hacia objetivos | Minimiza distancia a metas |
| `PROGRESSION` | Progresión en eventos clave | Enfoca completar misiones |
| `STUCK` | Comportamiento repetitivo | Fuerza exploración aleatoria |

### **Heurísticas Implementadas**

1. **Exploration Heuristic**: Favorece áreas no exploradas del mapa
2. **Objective Distance**: Calcula distancia óptima a objetivos conocidos  
3. **Health Consideration**: Adapta comportamiento según HP actual
4. **Level Progression**: Busca oportunidades de entrenamiento/experiencia
5. **Map Familiarity**: Evita áreas sobre-exploradas
6. **Event Completion**: Prioriza progresión en la historia del juego

### **Características Técnicas**

- **Decisiones en Tiempo Real**: Sin necesidad de entrenamiento previo
- **Adaptación Dinámica**: Pesos de heurísticas cambian según escenario
- **Epsilon Decay Inteligente**: Reduce exploración aleatoria gradualmente
- **Detección de Bloqueos**: Identifica y corrige comportamiento repetitivo

### **Comparación PPO vs Epsilon Greedy**

| Aspecto | PPO | Epsilon Greedy | Ventaja |
|---------|-----|----------------|---------|
| **Setup Time** | Requiere entrenamiento | Listo inmediatamente | Epsilon Greedy |
| **Interpretabilidad** | Caja negra | Lógica transparente | Epsilon Greedy |
| **Recursos** | GPU/Alto cómputo | CPU ligero | Epsilon Greedy |
| **Adaptabilidad** | Aprende patrones | Heurísticas fijas | PPO |
| **Rendimiento Máximo** | Potencial superior | Limitado por diseño | PPO |
| **Mantenimiento** | Reentrenamiento | Ajuste de parámetros | Epsilon Greedy |

## Métricas y Resultados

```bash
# Ejecutar comparación completa (recomendado)
python run_comparison.py --mode full --episodes 5

# Solo agente Epsilon Greedy
python run_comparison.py --mode standalone --episodes 3

# Solo comparación entre agentes
python run_comparison.py --mode comparison --episodes 5
```

### 2. Opciones Avanzadas

```bash
# Configuración personalizada
python run_comparison.py \
    --mode full \
    --episodes 10 \
    --max-steps 50000 \
    --epsilon-start 0.6 \
    --epsilon-decay 0.992 \
    --ppo-model "../v2/runs/poke_model.zip"

# Ejecución en paralelo
python run_comparison.py --mode comparison --parallel --episodes 5

### **Ejemplo de Uso Programático**

```python
from agent_comparison import AgentComparator
from epsilon_greedy_agent import EpsilonGreedyAgent

# Configurar entorno
env_config = {
    'headless': True,
    'max_steps': 40960,
    'gb_path': '../PokemonRed.gb',
    'init_state': '../init.state'
}

# Configurar agente Epsilon Greedy
epsilon_config = {
    'epsilon_start': 0.5,
    'epsilon_min': 0.05,
    'epsilon_decay': 0.995,
    'scenario_detection_enabled': True
}

# Ejecutar comparación
comparator = AgentComparator(env_config)
results = comparator.run_comparison(epsilon_config=epsilon_config)
```

### **Resultados de Pruebas Recientes**

**Prueba Exitosa**
```
Mode: standalone, Episodes: 2, Max Steps: 40960
- Total Steps: 81,920
- Mean Episode Reward: 19.08 ± 2.56
- Max Episode Reward: 21.64
- Mean Episode Length: 40,960
- Scenarios Detected: navigation, stuck
- Epsilon Decay: 0.5 → 0.05
- Execution Time: ~45 minutos
- Memory Usage: ~2GB
```

**Métricas Clave Observadas:**
- **Detección de Escenarios Funcional**: Transición navigation → stuck
- **Epsilon Decay Correcto**: Reducción gradual de exploración aleatoria
- **Estabilidad del Sistema**: Sin crashes ni errores de memoria
- **Integración v2 Exitosa**: Compatibilidad completa con ambiente
```

## Algoritmo Epsilon Greedy

### Características Técnicas

1. **Detección de Escenarios**
   - `EXPLORATION`: Exploración general del mapa
   - `BATTLE`: En combate con Pokemon salvajes/entrenadores
   - `NAVIGATION`: Navegación hacia objetivos específicos
   - `PROGRESSION`: Progresión en eventos clave (gimnasios, capturas)
   - `STUCK`: Detección de comportamiento repetitivo

2. **Heurísticas Implementadas**
   - **Exploración**: Favorece áreas no visitadas
   - **Distancia a Objetivos**: Navega hacia metas específicas
   - **Consideración de Salud**: Adapta comportamiento según HP
   - **Progresión de Niveles**: Busca oportunidades de entrenamiento
   - **Familiaridad del Mapa**: Evita áreas sobre-exploradas
   - **Completar Eventos**: Prioriza progresión en la historia

3. **Adaptación Dinámica**
   - Pesos de heurísticas cambian según el escenario
   - Epsilon decay adaptativo
   - Detección automática de bloqueos

### Comparación con PPO

| Métrica | PPO | Epsilon Greedy | Ventaja |
|---------|-----|----------------|---------|
| **Tiempo de Convergencia** | Requiere episodios de entrenamiento | Decisiones inmediatas | Epsilon Greedy |
| **Adaptabilidad** | Aprende de experiencia | Heurísticas predefinidas | PPO |
| **Interpretabilidad** | Caja negra | Lógica transparente | Epsilon Greedy |
| **Eficiencia Computacional** | Requiere GPU/entrenamiento | CPU ligero | Epsilon Greedy |
| **Rendimiento Óptimo** | Potencial superior tras entrenamiento | Limitado por heurísticas | PPO |

## Métricas de Evaluación

### Métricas Básicas
- **Recompensa por Episodio**: Media, mediana, desviación estándar
- **Longitud de Episodio**: Eficiencia temporal
- **Estabilidad**: Consistencia del rendimiento

### Métricas Avanzadas
- **Eficiencia de Exploración**: Ratio de áreas nuevas exploradas
- **Tasa de Convergencia**: Episodios hasta estabilización
- **Adaptabilidad de Escenarios**: Distribución de escenarios detectados
- **Ratio Sharpe**: Rendimiento ajustado por riesgo
- **Eficiencia de Pareto**: Balance recompensa vs longitud

### Métricas Específicas del Juego
- **Medallas Obtenidas**: Progreso en gimnasios
- **Eventos Completados**: Progresión en la historia
- **Pokemon Capturados**: Diversidad del equipo
- **Áreas Exploradas**: Cobertura del mapa

## Resultados Esperados

### Ventajas del Epsilon Greedy
1. **Inicio Inmediato**: No requiere entrenamiento previo
2. **Transparencia**: Decisiones explicables
3. **Eficiencia**: Menor costo computacional
4. **Adaptabilidad**: Respuesta inmediata a cambios

### Ventajas del PPO
1. **Aprendizaje**: Mejora con experiencia
2. **Optimización**: Puede encontrar estrategias no obvias
3. **Rendimiento**: Potencial superior a largo plazo

## Archivos de Salida

### Resultados de Comparación
- `comparison_results/comparison_report_[timestamp].json`: Reporte detallado
- `comparison_results/comparison_visualization_[timestamp].png`: Gráficos comparativos
- `comparison_results/detailed_metrics_[timestamp].csv`: Métricas por episodio

### Análisis de Métricas
- `detailed_metrics_analysis/comprehensive_comparison_[timestamp].png`: Análisis visual
- `detailed_metrics_analysis/comparison_report_[timestamp].json`: Análisis estadístico

## Configuración Avanzada

### Parámetros del Epsilon Greedy
```python
epsilon_config = {
    'epsilon_start': 0.5,        # Probabilidad inicial de exploración
    'epsilon_min': 0.05,         # Probabilidad mínima de exploración  
    'epsilon_decay': 0.995,      # Tasa de decaimiento de epsilon
    'scenario_detection_enabled': True  # Activar detección de escenarios
}
```

### Pesos de Heurísticas por Escenario
```python
# Ejemplo para escenario de exploración
HeuristicWeights(
    exploration=1.5,           # Mayor peso a exploración
    objective_distance=0.8,    # Menor peso a objetivos específicos
    health_consideration=0.6,  # Consideración moderada de salud
    level_progression=0.4,     # Menor prioridad a niveles
    map_familiarity=1.0,       # Evitar áreas familiares
    event_completion=0.8       # Moderada prioridad a eventos
)
```

## Troubleshooting Avanzado

### **Problemas Críticos Resueltos**

#### **1. Error PyBoy API Incompatibilidad**
```bash
AttributeError: 'pyboy.pyboy.PyBoy' object has no attribute 'botsupport_manager'
AttributeError: 'pyboy.pyboy.PyBoy' object has no attribute 'get_memory_value'  
AttributeError: 'pyboy.api.screen.Screen' object has no attribute 'screen_ndarray'
```
**Solución Implementada**:
- Downgrade a PyBoy 2.4.0: `pip install pyboy==2.4.0`
- API fixes aplicados en `red_gym_env_v2.py`:
  - `pyboy.botsupport_manager().screen()` → `pyboy.screen`
  - `pyboy.get_memory_value(addr)` → `pyboy.memory[addr]`
  - `screen.screen_ndarray()` → `screen.ndarray`

#### **2. Dependencias Faltantes**
```bash
ModuleNotFoundError: No module named 'skimage'
ModuleNotFoundError: No module named 'websockets'
```
**Solución**:
```bash
pip install scikit-image websockets mediapy einops
```

#### **3. Archivos de Configuración Faltantes**
```bash
FileNotFoundError: [Errno 2] No such file or directory: 'events.json'
```
**Solución**:
```bash
copy ..\v2\events.json .
copy ..\v2\map_data.json .
```

#### **4. Conflictos de Versión Python/CUDA**
```bash
nvidia-nccl-cu12==2.21.5 requires Python >=3.11
```
**Solución**:
- Usar ambiente conda con Python 3.10.18
- Instalar PyTorch CPU-only
- Usar requirements_py310.txt

### **Diagnóstico Rápido**

```bash
# Verificación completa del ambiente
python verify_environment.py

# Test específico de PyBoy
python test_pyboy_api.py

# Verificar versiones críticas
pip show pyboy pytorch stable-baselines3
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import pyboy; print(f'PyBoy: {pyboy.__version__}')"
```

### **Problemas de Rendimiento**

#### **Agente se Queda "Stuck"**
```
Step 1000: Reward=0.000, Scenario=stuck, Epsilon=0.050
```
**Posibles Causas**:
- Epsilon muy bajo (< 0.05)
- Heurísticas mal configuradas
- Mapa sin objetivos claros

**Soluciones**:
```bash
# Aumentar epsilon mínimo
python run_comparison.py --epsilon-min 0.1

# Reducir episodios para testing
python run_comparison.py --episodes 1 --max-steps 10000

# Verificar configuración de escenarios
python -c "from epsilon_greedy_agent import EpsilonGreedyAgent; print('OK')"
```

#### **Memoria Insuficiente**
```bash
# Ejecutar con menos pasos
python run_comparison.py --max-steps 20000

# Ejecutar headless (sin ventana)
python run_comparison.py --headless

# Limpiar cache
rm -rf __pycache__/ comparison_results/old_*
```

### **Optimización de Rendimiento**

#### **Para Testing Rápido**
```bash
python run_comparison.py --mode standalone --episodes 1 --max-steps 5000
```

#### **Para Análisis Completo**
```bash
python run_comparison.py --mode full --episodes 10 --max-steps 40960
```

#### **Para Producción**
```bash
python run_comparison.py --mode comparison --episodes 20 --headless --no-viz
```

## Desarrollo Futuro

### **Mejoras Planificadas**
1. **Heurísticas Más Sofisticadas**: Integración con análisis de imagen
2. **Aprendizaje Híbrido**: Combinación de heurísticas con aprendizaje por refuerzo
3. **Optimización de Parámetros**: Búsqueda automática de hiperparámetros
4. **Métricas Adicionales**: Análisis de eficiencia energética y memoria
5. **Soporte Multi-Juego**: Extensión a otros juegos de Pokemon

### **Contribuciones**
- Mejoras en heurísticas específicas del juego
- Nuevos escenarios de detección automática
- Optimizaciones de rendimiento
- Métricas adicionales de evaluación
- Documentación y tutoriales

## Resumen de Implementación

### **Logros Completados (Septiembre 2025)**

1. **Agente Epsilon Greedy Completo**
   - 5 escenarios de detección automática
   - 6 heurísticas especializadas 
   - Adaptación dinámica de parámetros
   - Detección de comportamiento repetitivo

2. **Compatibilidad y Estabilidad**
   - Resolución de conflictos PyBoy API
   - Ambiente conda especializado Python 3.10
   - Dependencias optimizadas para Windows
   - Testing automatizado completo

3. **Sistema de Análisis Avanzado**
   - Comparación automática PPO vs Epsilon Greedy
   - 20+ métricas de evaluación
   - Visualizaciones comprehensivas  
   - Reportes estadísticos detallados

4. **Documentación Completa**
   - Guías de instalación paso a paso
   - Troubleshooting basado en problemas reales
   - Ejemplos de uso programático
   - Configuraciones optimizadas

### **Resultados Verificados**
- **Sistema Completamente Funcional**: 2 episodios de 40,960 pasos c/u
- **Detección de Escenarios**: navigation → stuck transition
- **Métricas Completas**: Recompensa media 19.08, epsilon decay funcional  
- **Estabilidad**: Sin crashes, memoria estable, ejecución ~45 min

### **Impacto del Proyecto**
- **Investigación**: Comparación directa de enfoques clásicos vs deep RL
- **Educativo**: Implementación transparente y explicable
- **Práctico**: Sistema listo para producción y experimentación
- **Técnico**: Resolución de compatibilidad PyBoy/v2

---

## Licencia

Este proyecto utiliza la misma licencia que el proyecto Pokemon Red base.

## Referencias Técnicas

- [Proyecto Original Pokemon Red](../README.md)
- [Stable Baselines3 Documentation](https://stable-baselines3.readthedocs.io/)
- [PyBoy Emulator GitHub](https://github.com/Baekalfen/PyBoy)
- [Epsilon Greedy Algorithm Theory](https://en.wikipedia.org/wiki/Multi-armed_bandit#Epsilon-greedy_strategy)
- [Gymnasium Environment Interface](https://gymnasium.farama.org/)

---
