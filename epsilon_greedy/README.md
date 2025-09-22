# Epsilon Greedy Search Algorithm

Esta carpeta contiene la implementación del algoritmo **Epsilon Greedy** para Pokemon Red.

## Archivos

- `epsilon_greedy_agent.py` - Implementación original del agente epsilon greedy que funciona muy bien
- `epsilon_variable_agent.py` - Versión avanzada con epsilon configurable para estudiar impacto en rendimiento
- `run_epsilon_greedy_interactive.py` - Script interactivo para ejecutar el agente original
- `test_epsilon_variants.py` - Script para probar diferentes valores de epsilon

## Variaciones de Epsilon

El archivo `epsilon_variable_agent.py` incluye configuraciones predefinidas:

- **very_high_exploration** (ε=0.9): 90% exploración - casi aleatorio
- **high_exploration** (ε=0.7): 70% exploración - mucha exploración  
- **balanced** (ε=0.5): 50% exploración - enfoque balanceado
- **moderate_exploitation** (ε=0.3): 30% exploración - más explotación
- **low_exploration** (ε=0.1): 10% exploración - principalmente explotación
- **very_low_exploration** (ε=0.05): 5% exploración - casi pura explotación
- **pure_exploitation** (ε=0.01): 1% exploración - casi greedy

## Uso

### Ejecutar agente original (que funciona de maravilla):
```bash
python epsilon_greedy/run_epsilon_greedy_interactive.py
```

### Probar diferentes valores de epsilon:
```bash
python epsilon_greedy/test_epsilon_variants.py
```

### Crear agente con epsilon específico:
```python
from epsilon_greedy.epsilon_variable_agent import VariableEpsilonGreedyAgent
agent = VariableEpsilonGreedyAgent(env, epsilon=0.3)  # 30% exploración
```

### Usar configuraciones predefinidas:
```python
from epsilon_greedy.epsilon_variable_agent import create_agent_with_preset
agent = create_agent_with_preset(env, 'balanced')  # ε=0.5
```

## Rendimiento

El epsilon greedy original ha demostrado **excelente rendimiento** en Pokemon Red. Los experimentos con diferentes valores de epsilon permiten estudiar:

- **Alto epsilon** → Más exploración, descubre nuevas áreas pero puede ser errático
- **Bajo epsilon** → Más explotación, más eficiente pero puede quedarse atascado
- **Epsilon balanceado** → Combina exploración y explotación efectivamente