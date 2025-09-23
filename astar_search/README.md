# A* Search Algorithm

Esta carpeta contiene la implementación del algoritmo **A\*** para Pokemon Red.

## Archivos

- `astar_agent.py` - Implementación del algoritmo A* con pathfinding inteligente
- `v2_astar_agent.py` - Wrapper para integrar A* con el entorno v2
- `run_astar_interactive_simple.py` - Script interactivo simplificado para ejecutar A*
- `test_astar_agent.py` - Script básico para probar el agente A*

## Características del A*

- **Pathfinding inteligente**: Encuentra rutas óptimas hacia objetivos
- **Exploración dirigida por objetivos**: Prioriza áreas prometedoras
- **Función heurística**: Estima distancia a objetivos para guiar búsqueda
- **Evita ciclos**: Previene comportamientos repetitivos
- **Métricas detalladas**: Tracking completo de performance

## Uso

### Ejecutar A* interactivo:
```bash
python astar_search/run_astar_interactive_simple.py
```

### Probar A* básico:
```bash
python astar_search/test_astar_agent.py
```

### Usar A* en código:
```python
from astar_search.v2_astar_agent import V2AStarAgent

# Crear agente A*
agent = V2AStarAgent()

# Ejecutar episodio
stats = agent.run_episode(steps=1000)
```

## Algoritmo A*

A* utiliza una función de evaluación f(n) = g(n) + h(n):

- **g(n)**: Costo real desde el inicio hasta el nodo n
- **h(n)**: Heurística (estimación del costo desde n hasta el objetivo)
- **f(n)**: Estimación del costo total del mejor camino que pasa por n

## Ventajas

- **Óptimo**: Encuentra el camino más corto
- **Dirigido por objetivos**: No explora aleatoriamente
- **Inteligente**: Usa información del estado del juego
- **Eficiente**: Evita explorar áreas no prometedoras

## Desventajas

- **Complejidad**: Más complejo que epsilon greedy
- **Memoria**: Requiere mantener estructuras de pathfinding
- **Heurística dependiente**: Rendimiento depende de la calidad de la heurística