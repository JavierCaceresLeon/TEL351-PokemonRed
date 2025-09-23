# Pokemon Red Search Algorithms - Organized Structure

Los algoritmos de búsqueda están ahora organizados en directorios separados para mejor claridad y mantenimiento.

## Estructura Organizada

### epsilon_greedy/
**RECOMENDADO** - Algoritmo que funciona de maravilla
- `epsilon_greedy_agent.py` - Implementación original que funciona excelente
- `epsilon_variable_agent.py` - **NUEVO**: Versión con epsilon configurable
- `test_epsilon_variants.py` - **NUEVO**: Prueba diferentes valores de epsilon
- `run_epsilon_greedy_interactive.py` - Script interactivo original
- `v2_agent.py` - Wrapper para entorno v2

### astar_search/
**ALTERNATIVA INTELIGENTE** - Pathfinding dirigido por objetivos
- `astar_agent.py` - Implementación completa de A*
- `v2_astar_agent.py` - Wrapper para entorno v2
- `run_astar_interactive_simple.py` - Ejecución continua como epsilon greedy
- `test_astar_agent.py` - Pruebas básicas

### tabu_search/
**NO RECOMENDADO** - Problemas conocidos con menu spam
- `tabu_agent.py` - Implementación con anti-spam (aún problemática)
- `demo_tabu_search.py` - Demo del algoritmo

## Uso Rápido

### Ejecutar Epsilon Greedy (Recomendado):
```bash
cd epsilon_greedy
python run_epsilon_greedy_interactive.py
```

### Probar Diferentes Valores de Epsilon:
```bash
cd epsilon_greedy
python test_epsilon_variants.py
```

### Ejecutar A* (Alternativa Inteligente):
```bash
cd astar_search
python run_astar_interactive_simple.py
```

## Estudio de Epsilon

El nuevo `epsilon_variable_agent.py` permite estudiar el impacto del parámetro epsilon:

### Configuraciones Predefinidas:
- `very_high_exploration` (ε=0.9) - 90% exploración
- `high_exploration` (ε=0.7) - 70% exploración  
- `balanced` (ε=0.5) - 50% exploración
- `moderate_exploitation` (ε=0.3) - 30% exploración
- `low_exploration` (ε=0.1) - 10% exploración
- `very_low_exploration` (ε=0.05) - 5% exploración
- `pure_exploitation` (ε=0.01) - 1% exploración

### Ejemplo de Uso:
```python
from epsilon_greedy.epsilon_variable_agent import create_agent_with_preset

# Crear agente balanceado
agent = create_agent_with_preset(env, 'balanced')

# O crear con epsilon específico
agent = VariableEpsilonGreedyAgent(env, epsilon=0.3)

# Cambiar epsilon durante ejecución
agent.set_epsilon(0.7)  # Aumentar exploración
```

## Comparación de Algoritmos

| Algoritmo | Estado | Rendimiento | Complejidad | Recomendación |
|-----------|--------|-------------|-------------|---------------|
| **Epsilon Greedy** | Excelente | Muy bueno | Baja | **USAR** |
| **A\*** | Funcional | Bueno | Media | Alternativa |
| **Tabu Search** | Problemático | Malo | Alta | **NO USAR** |

## Cómo Funciona Epsilon

- **Epsilon Alto (ε=0.7-0.9)**: 
  - Más exploración aleatoria
  - Descubre nuevas áreas
  - Puede ser errático al principio
  - Bueno para mapear el juego

- **Epsilon Medio (ε=0.3-0.5)**:
  - Balance entre exploración y explotación
  - Explora cuando es necesario
  - Aprovecha conocimiento previo
  - **Recomendado para uso general**

- **Epsilon Bajo (ε=0.05-0.1)**:
  - Principalmente explotación
  - Más eficiente en áreas conocidas
  - Puede quedarse atascado
  - Bueno cuando el agente ya conoce el entorno

## Migración desde comparison_agents/

Los archivos han sido movidos pero mantienen compatibilidad. Si tienes scripts que importan desde `comparison_agents/`, actualiza las rutas:

```python
# Antes:
from comparison_agents.epsilon_greedy_agent import EpsilonGreedyAgent

# Ahora:
from epsilon_greedy.epsilon_greedy_agent import EpsilonGreedyAgent
```

## Próximos Pasos

1. **Experimenta con diferentes epsilons** usando `test_epsilon_variants.py`
2. **Compara rendimiento** entre epsilon greedy y A*
3. **Ajusta epsilon dinámicamente** durante entrenamiento
4. **Documenta tus hallazgos** para optimizar parámetros