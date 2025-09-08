# Comparación de Agentes - Pokémon Red

Este proyecto compara diferentes enfoques para resolver el problema de salir de la habitación inicial en Pokémon Red.

## Agentes Implementados

### 1. Agente V2 (Entrenado)
- **Descripción**: Usa el modelo PPO entrenado en la carpeta `/v2`
- **Enfoque**: Aprendizaje por refuerzo
- **Ventajas**: Pre-entrenado, experiencia acumulada
- **Desventajas**: Caja negra, requiere entrenamiento previo

### 2. Agente A*
- **Descripción**: Algoritmo de búsqueda informada con heurística
- **Enfoque**: Búsqueda en espacio de estados
- **Ventajas**: Óptimo si la heurística es admisible, explícito
- **Desventajas**: Puede ser lento, requiere heurística bien diseñada

### 3. Agente Tabú Search
- **Descripción**: Algoritmo de búsqueda local con memoria
- **Enfoque**: Búsqueda local con diversificación
- **Ventajas**: Bueno para espacios grandes, evita óptimos locales
- **Desventajas**: No garantiza optimalidad, requiere ajuste de parámetros

## Estructura del Proyecto

```
comparison_agents/
├── search_env.py              # Entorno base para algoritmos de búsqueda
├── v2_agent.py               # Wrapper para el agente entrenado
├── run_comparison.py         # Script principal de comparación
├── search_algorithms/
│   ├── astar_agent.py       # Implementación A*
│   └── tabu_agent.py        # Implementación Tabú Search
└── results/                 # Resultados de las comparaciones
```

## Instalación y Configuración

### Dependencias
```bash
pip install numpy scikit-image matplotlib pyboy stable-baselines3
```

### Archivos Requeridos
- `PokemonRed.gb`: ROM de Pokémon Red
- `init.state`: Estado inicial del juego
- Modelo entrenado en `/v2/runs/` (archivo .zip)

## Uso

### Ejecutar Comparación Completa
```bash
cd comparison_agents
python run_comparison.py
```

### Configurar Parámetros
Edita el archivo `run_comparison.py` para modificar:
- Número de ejecuciones por agente
- Máximo de pasos permitidos
- Rutas de archivos
- Parámetros específicos de cada algoritmo

## Métricas de Comparación

### Métricas Principales
- **Tasa de Éxito**: Porcentaje de intentos exitosos
- **Tiempo Total**: Tiempo desde inicio hasta completar objetivo
- **Número de Pasos**: Acciones tomadas hasta el éxito
- **Eficiencia**: Pasos por segundo

### Métricas Específicas por Agente

#### A*
- Nodos explorados
- Tiempo de búsqueda vs. ejecución
- Calidad de la heurística

#### Tabú Search
- Iteraciones realizadas
- Fitness máximo alcanzado
- Convergencia

#### V2 Agent
- Tiempo de predicción del modelo
- Consistencia entre ejecuciones

## Resultados

Los resultados se guardan automáticamente en `results/comparison_results_YYYYMMDD_HHMMSS.json`

### Formato de Resultados
```json
{
  "v2_agent": [
    {
      "run_id": 0,
      "agent_type": "v2",
      "total_time": 15.2,
      "plan_length": 45,
      "success": true,
      "timestamp": "2024-01-01T12:00:00"
    }
  ],
  "astar_agent": [...],
  "tabu_agent": [...]
}
```

## Interpretación de Resultados

### Análisis Esperado
1. **V2 Agent**: Debería ser rápido y consistente debido al entrenamiento
2. **A***: Podría encontrar soluciones óptimas pero ser más lento
3. **Tabú Search**: Podría encontrar buenas soluciones con variabilidad

### Factores a Considerar
- **Determinismo**: V2 puede ser determinista, búsquedas pueden variar
- **Optimalidad**: A* debería encontrar caminos más cortos
- **Robustez**: Tabú Search podría manejar mejor situaciones complejas

## Limitaciones y Mejoras

### Limitaciones Actuales
- Heurística de A* simplificada
- Tabú Search con parámetros fijos
- Entorno simplificado para búsquedas
- Métricas limitadas

### Mejoras Potenciales
1. Heurísticas más sofisticadas para A*
2. Parámetros adaptativos para Tabú Search
3. Más algoritmos de búsqueda (Beam Search, Genetic Algorithm)
4. Análisis estadístico más profundo
5. Visualización de trayectorias

## Troubleshooting

### Problemas Comunes

#### "No se encontró modelo entrenado"
- Verifica que existe un archivo .zip en `/v2/runs/`
- Asegúrate de haber entrenado el modelo v2 previamente

#### "Error importing dependencies"
- Instala stable_baselines3: `pip install stable_baselines3`
- Verifica que PyBoy esté instalado correctamente

#### "ROM no encontrada"
- Verifica la ruta del archivo `PokemonRed.gb`
- Asegúrate de que el archivo `init.state` existe

#### "Agentes no encuentran solución"
- Aumenta `max_steps` en la configuración
- Ajusta parámetros de búsqueda (tabu_size, max_iterations)
- Verifica que el estado inicial es válido

## Contribuir

Para añadir nuevos agentes:
1. Crear nueva clase que implemente métodos `search()` y `execute_plan()`
2. Añadir al script de comparación
3. Documentar métricas específicas

## Referencias

- [Stable Baselines3](https://stable-baselines3.readthedocs.io/)
- [PyBoy](https://github.com/Baekalfen/PyBoy)
- [A* Algorithm](https://en.wikipedia.org/wiki/A*_search_algorithm)
- [Tabu Search](https://en.wikipedia.org/wiki/Tabu_search)
