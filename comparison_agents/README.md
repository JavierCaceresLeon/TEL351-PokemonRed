# Pokemon Red Agent Comparison: PPO vs Epsilon Greedy

Este proyecto implementa un agente de búsqueda con algoritmo Epsilon Greedy y heurísticas avanzadas para el entorno Pokemon Red, proporcionando una comparación comprensiva con el agente PPO de la versión 2.

## Descripción del Proyecto

### Características Principales

1. **Agente Epsilon Greedy con Heurísticas Avanzadas**
   - Detección automática de escenarios de juego
   - Heurísticas adaptativas según el contexto
   - Toma de decisiones en tiempo real sin entrenamiento por episodios

2. **Sistema de Comparación Comprensivo**
   - Métricas detalladas de rendimiento
   - Análisis estadístico comparativo
   - Visualizaciones comprensivas

3. **Análisis de Múltiples Escenarios**
   - Exploración general del mapa
   - Navegación hacia objetivos específicos
   - Detección y manejo de estados de bloqueo
   - Progresión en eventos clave del juego

## Estructura del Proyecto

```
comparison_agents/
├── epsilon_greedy_agent.py    # Implementación del algoritmo Epsilon Greedy
├── v2_agent.py               # Wrapper para integración con v2
├── agent_comparison.py       # Sistema de comparación entre agentes
├── metrics_analyzer.py       # Análisis avanzado de métricas
├── run_comparison.py         # Script principal de ejecución
├── config.py                 # Configuraciones
├── requirements.txt          # Dependencias
└── README.md                # Este archivo
```

## Instalación

1. **Prerrequisitos**
   ```bash
   # Asegurar que el entorno v2 esté configurado
   cd ../v2
   pip install -r requirements.txt
   
   # Volver al directorio de comparación
   cd ../comparison_agents
   ```

2. **Instalar dependencias específicas**
   ```bash
   pip install -r requirements.txt
   ```

## Uso

### 1. Ejecución Básica

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

# Sin visualizaciones (más rápido)
python run_comparison.py --mode full --no-viz --episodes 3
```

### 3. Uso Programático

```python
from agent_comparison import AgentComparator
from epsilon_greedy_agent import EpsilonGreedyAgent

# Configurar entorno
env_config = {
    'headless': True,
    'max_steps': 40960,
    'gb_path': '../PokemonRed.gb',
    'init_state': '../init.state',
    # ... más configuraciones
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

## Troubleshooting

### Problemas Comunes

1. **Error de importación de v2**
   ```bash
   # Verificar que el directorio v2 existe y tiene los archivos necesarios
   ls ../v2/red_gym_env_v2.py
   ```

2. **Modelo PPO no encontrado**
   ```bash
   # Especificar ruta correcta o ejecutar sin modelo (usará PPO aleatorio)
   python run_comparison.py --ppo-model None
   ```

3. **Error de memoria**
   ```bash
   # Reducir episodios o pasos por episodio
   python run_comparison.py --episodes 3 --max-steps 20000
   ```

## Desarrollo Futuro

### Mejoras Planificadas
1. **Heurísticas Más Sofisticadas**: Integración con análisis de imagen
2. **Aprendizaje Híbrido**: Combinación de heurísticas con aprendizaje
3. **Optimización de Parámetros**: Búsqueda automática de hiperparámetros
4. **Métricas Adicionales**: Análisis de eficiencia energética y memoria

### Contribuciones
- Mejoras en heurísticas específicas del juego
- Nuevos escenarios de detección
- Optimizaciones de rendimiento
- Métricas adicionales de evaluación

## Licencia

Este proyecto utiliza la misma licencia que el proyecto Pokemon Red base.

## Referencias

- [Proyecto Original Pokemon Red](../README.md)
- [Documentación de Stable Baselines3](https://stable-baselines3.readthedocs.io/)
- [PyBoy Emulator](https://github.com/Baekalfen/PyBoy)
